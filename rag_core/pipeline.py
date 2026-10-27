"""门面：把"加载语料 -> 切块 -> 向量化 -> 入库 -> 建 BM25 -> 检索 -> 拼上下文 -> 生成"串起来。

对 Agent 层来说，这个类就是**唯一的检索原语**。P2 的工具层（``tools/``）只负责给它套上
name/description/JSON Schema，不在这里加任何"Agent 感知"的逻辑 —— 否则以后想换编排框架，
检索层要跟着改，消融实验也就不可比了。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .bm25 import BM25Index
from .config import AppConfig
from .context import ContextConfig
from .data_preparation import DatasetConfig, RecipeDataset
from .embeddings import build_embedder
from .generation import RecipeAnswerer
from .llm import LLMConfig, build_llm
from .retrieval import HybridRetriever, RetrievalConfig
from .schema import Recipe, SearchFilter
from .vector_store import LocalVectorStore, MilvusVectorStore, build_store

logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    recipes: int
    chunks: int
    dim: int
    backend: str
    tokenizer: str
    avg_chunk_chars: float

    def __str__(self) -> str:
        return (
            f"菜谱 {self.recipes} 篇 / 子块 {self.chunks} 个 / {self.dim} 维 / "
            f"后端 {self.backend} / 分词 {self.tokenizer} / 平均块长 {self.avg_chunk_chars} 字"
        )


class RecipeRAG:
    def __init__(
        self,
        cfg: AppConfig,
        dataset: RecipeDataset,
        retriever: HybridRetriever,
        answerer: Optional[RecipeAnswerer] = None,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.retriever = retriever
        self.answerer = answerer

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, cfg: AppConfig, *, skip_embedding: bool = False) -> "RecipeRAG":
        ds_cfg = DatasetConfig(
            root=cfg.dataset.root,
            exclude_dirs=list(cfg.dataset.exclude_dirs),
            max_level=cfg.dataset.max_level,
            drop_comments=cfg.dataset.drop_comments,
            min_chunk_chars=cfg.dataset.min_chunk_chars,
        )
        dataset = RecipeDataset(ds_cfg).load().build_chunks()
        if not dataset.chunks:
            raise RuntimeError("没有切出任何子块，请检查 dataset.root 是否指向 recipe 目录")
        logger.info("语料统计：%s", dataset.stats())

        # 落盘一份，评测脚本直接复用，避免重复切块导致 id 与 qrels 对不上
        dataset.save(Path(cfg.workspace) / cfg.name / "dataset")

        embedder = None
        store = None
        if not skip_embedding:
            embedder = build_embedder(
                cfg.embedding.provider,
                model=cfg.embedding.model,
                dim=cfg.embedding.dim,
                batch_size=cfg.embedding.batch_size,
                base_url=cfg.embedding.base_url,
                api_key=cfg.embedding.api_key,
            )
            cfg.store.dim = getattr(embedder, "dim", 0) or cfg.embedding.dim
            store = build_store(cfg.store)
            vectors = embedder.encode([c.text for c in dataset.chunks])
            store.upsert(dataset.chunks, vectors)
            logger.info("已写入向量库：%d 块（后端 %s）", store.count(), cfg.store.backend)
            if isinstance(store, LocalVectorStore):
                store.save(cfg.store.local_dir)

        bm25 = BM25Index(
            tokenizer_name=cfg.retrieval.tokenizer,
            k1=cfg.retrieval.bm25_k1,
            b=cfg.retrieval.bm25_b,
        ).fit(dataset.chunks)

        retriever = HybridRetriever(
            store=store,
            embedder=embedder,
            bm25=bm25,
            chunk_map=dataset.chunk_map(),
            recipe_map=dataset.recipe_map(),
            cfg=RetrievalConfig(
                top_k=cfg.retrieval.top_k,
                dense_k=cfg.retrieval.dense_k,
                sparse_k=cfg.retrieval.sparse_k,
                rrf_k=cfg.retrieval.rrf_k,
                filter_mode=cfg.retrieval.filter_mode,
                parent_strategy=cfg.retrieval.parent_strategy,
                parent_top_n=cfg.retrieval.parent_top_n,
                arms=tuple(cfg.retrieval.arms),
            ),
        )

        answerer = None
        if cfg.llm.model not in {"", "none"}:
            try:
                llm = build_llm(
                    LLMConfig(
                        model=cfg.llm.model,
                        base_url=cfg.llm.base_url,
                        api_key=cfg.llm.api_key,
                        temperature=cfg.llm.temperature,
                        max_tokens=cfg.llm.max_tokens,
                        timeout=cfg.llm.timeout,
                        max_retries=cfg.llm.max_retries,
                    )
                )
                answerer = RecipeAnswerer(
                    llm,
                    ContextConfig(
                        token_budget=cfg.context.token_budget,
                        per_doc_budget=cfg.context.per_doc_budget,
                        min_doc_tokens=cfg.context.min_doc_tokens,
                    ),
                )
            except Exception as exc:  # 没有 key 时不该阻断检索侧实验
                logger.warning("生成模块未启用（%s），仅可做检索实验", exc)

        return cls(cfg, dataset, retriever, answerer)

    # 便捷入口：本地向量库已落盘时直接加载，省掉重新向量化
    @classmethod
    def load_local(cls, cfg: AppConfig) -> "RecipeRAG":
        _, chunks = RecipeDataset.load_jsonl(Path(cfg.workspace) / cfg.name / "dataset")
        ds = RecipeDataset(DatasetConfig(root=cfg.dataset.root))
        ds.chunks = chunks
        ds.recipes, _ = RecipeDataset.load_jsonl(Path(cfg.workspace) / cfg.name / "dataset")
        store = LocalVectorStore.load(cfg.store.local_dir, cfg.store.dim)
        build = build_embedder(
            cfg.embedding.provider,
            model=cfg.embedding.model,
            dim=cfg.embedding.dim,
            batch_size=cfg.embedding.batch_size,
            base_url=cfg.embedding.base_url,
            api_key=cfg.embedding.api_key,
        )
        bm25 = BM25Index(cfg.retrieval.tokenizer, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b).fit(chunks)
        retriever = HybridRetriever(
            store, build, bm25, ds.chunk_map(), ds.recipe_map(), RetrievalConfig(**{
                "top_k": cfg.retrieval.top_k,
                "dense_k": cfg.retrieval.dense_k,
                "sparse_k": cfg.retrieval.sparse_k,
                "rrf_k": cfg.retrieval.rrf_k,
                "filter_mode": cfg.retrieval.filter_mode,
                "parent_strategy": cfg.retrieval.parent_strategy,
                "parent_top_n": cfg.retrieval.parent_top_n,
                "arms": tuple(cfg.retrieval.arms),
            })
        )
        return cls(cfg, ds, retriever, None)

    # ------------------------------------------------------------------ query
    def retrieve(
        self,
        query: str,
        flt: Optional[SearchFilter] = None,
        top_k: Optional[int] = None,
        arms: Optional[Sequence[str]] = None,
    ):
        return self.retriever.search(query, top_k=top_k, flt=flt, arms=arms)

    def ask(self, query: str, flt: Optional[SearchFilter] = None, top_k: Optional[int] = None):
        if self.answerer is None:
            raise RuntimeError("未配置 LLM，无法生成答案；请设置 llm.model 与 LLM_API_KEY")
        result = self.retrieve(query, flt=flt, top_k=top_k)
        return self.answerer.answer(query, result), result

    def info(self) -> Dict[str, object]:
        info = {
            "dataset": self.dataset.stats(),
            "retriever": self.retriever.info(),
            "config": self.cfg.describe(),
        }
        return info
