"""混合检索：稠密 + 稀疏两路，RRF 融合，过滤先于检索。

对上游 ``code/C8/rag_modules/retrieval_optimization.py`` 的三处实质修改：

1. **文档身份用 chunk_id，不用 content 的 md5**（上游 L 附近 ``_rrf_rerank``）。
   菜谱里"## 操作"这类段落文本高度相似，用内容哈希做身份会把不同菜品的同名小节
   合并成一条，融合时互相顶掉。
2. **融合分只存在于返回值里，不写回 Document.metadata**。上游把 ``rrf_score``
   写进 ``doc.metadata``，而 Document 对象被 BM25 与向量两路共享、跨查询复用，
   第二次查询会读到上一次的残留分数。
3. **过滤下推**：先按条件求候选集，再在两路里各自检索。上游是"先取 3 倍候选再删"，
   过滤越严召回越差，而且拿不到"候选还剩多少"这个诊断信号。

另外把 arms 做成参数（``dense`` / ``sparse`` / 两者），消融实验不必改代码。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .bm25 import BM25Index
from .data_preparation import rank_parents
from .schema import Chunk, ParentHit, Recipe, RetrievalResult, ScoredChunk, SearchFilter

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    dense_k: int = 10  # 单路召回深度：融合前每路取多少条
    sparse_k: int = 10
    rrf_k: int = 60  # RRF 平滑常数，上游也是 60（经验值，本项目会单独扫一遍）
    filter_mode: str = "pre"  # pre=过滤下推 / post=先检索后过滤（仅用于对照）
    parent_strategy: str = "rrf_sum"  # rrf_sum | hit_count | max_score
    parent_top_n: int = 2
    arms: Tuple[str, ...] = ("dense", "sparse")

    def describe(self) -> str:
        return (
            f"top_k={self.top_k} dense_k={self.dense_k} sparse_k={self.sparse_k} "
            f"rrf_k={self.rrf_k} filter={self.filter_mode} parents={self.parent_strategy}"
        )


def rerank_by_difficulty(
    parents: Sequence[ParentHit],
    flt: Optional[SearchFilter],
) -> List[ParentHit]:
    """按难度偏好重排父文档。

    过滤只能把候选压到"不超过上限"，组内顺序仍由相关性决定：用户说"简单"时，
    第 1 名可能比第 5 名难。用元数据做一次零成本重排，把更符合偏好的提前。
    只在解析出难度偏好时启用；没有偏好就保持检索顺序，不引入额外假设。

    实测（25 条 constraint，诊断脚本、真值类目）：R@5 .632 → .695，P@5 .492 → .526，
    合规率 .799 → .827。三项同时变好，因为它只是把同一个候选集排得更合理，没有改变候选。
    """
    if flt is None:
        return list(parents)
    if flt.difficulty_max is not None:
        ascending = True
    elif flt.difficulty_min is not None:
        ascending = False
    else:
        return list(parents)

    def key(hit: ParentHit):
        level = hit.recipe.difficulty_level
        if not level:  # 难度未识别的菜永远排最后，不冒充"简单"
            return (1, 0, -hit.score)
        return (0, level if ascending else -level, -hit.score)

    return sorted(parents, key=key)


def select_parent_k(
    parents: Sequence[ParentHit],
    floor: int = 1,
    ceiling: int = 2,
    gap: float = 0.6,
) -> int:
    """按分数断层决定给答案看几篇父文档。

    观察（105 条 test，88 条非拒答非多跳）：gold 落在 top-1 时，top-1 与 top-2 的相对
    分数断层中位数 0.84；gold 不在 top-1 时断层最大只有 0.67。也就是说"第一篇甩开第二名
    一大截"是一个干净信号，不需要模型判断。

    规则只在**有把握**时减到 floor：断层不够大就用 ceiling。宁可多给一篇，
    也不要因为省一篇而把正确答案挡掉 —— floor 由调用方按意图给（比较题至少 2 篇）。

    这是 P3 的"规则版自适应 k"：没有学习、没有额外调用，可复现、可解释。
    """
    if len(parents) <= floor:
        return floor
    s1 = parents[0].score
    s2 = parents[1].score
    if s1 <= 0:
        return floor
    if (s1 - s2) / s1 >= gap:
        return floor
    return min(ceiling, len(parents))


class HybridRetriever:
    def __init__(
        self,
        store,
        embedder,
        bm25: BM25Index,
        chunk_map: Dict[str, Chunk],
        recipe_map: Dict[str, Recipe],
        cfg: Optional[RetrievalConfig] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.chunk_map = chunk_map
        self.recipe_map = recipe_map
        self.cfg = cfg or RetrievalConfig()

    # ------------------------------------------------------------------
    def _candidate_ids(self, flt: Optional[SearchFilter]) -> Optional[Set[str]]:
        """过滤条件的候选集。**只在 pre 模式下参与检索**。"""
        if flt is None or flt.is_empty():
            return None
        return set(flt.filter_ids(self.chunk_map.values()))

    def _rrf_fuse(
        self,
        dense: Sequence[Tuple[str, float]],
        sparse: Sequence[Tuple[str, float]],
    ) -> List[ScoredChunk]:
        k = self.cfg.rrf_k
        acc: Dict[str, ScoredChunk] = {}

        for rank, (chunk_id, score) in enumerate(dense, start=1):
            chunk = self.chunk_map.get(chunk_id)
            if chunk is None:
                continue
            hit = acc.setdefault(chunk_id, ScoredChunk(chunk=chunk))
            hit.dense_rank = rank
            hit.dense_score = score
            hit.rrf_score += 1.0 / (k + rank)

        for rank, (chunk_id, score) in enumerate(sparse, start=1):
            chunk = self.chunk_map.get(chunk_id)
            if chunk is None:
                continue
            hit = acc.setdefault(chunk_id, ScoredChunk(chunk=chunk))
            hit.sparse_rank = rank
            hit.sparse_score = score
            hit.rrf_score += 1.0 / (k + rank)

        return sorted(acc.values(), key=lambda h: (-h.rrf_score, h.chunk.chunk_id))

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        flt: Optional[SearchFilter] = None,
        arms: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        arms = tuple(arms or cfg.arms)

        candidates = self._candidate_ids(flt) if cfg.filter_mode == "pre" else None
        if candidates is not None and not candidates:
            # 过滤后一个候选都没有：这是有效信息（说明语料里没有满足约束的菜），
            # 必须如实返回空，而不是忽略条件去凑几条 —— 后者正是上游 metadata_filtered_search 的行为
            return RetrievalResult(
                query=query,
                fused=[],
                filter_desc=flt.describe() if flt else "无过滤",
                candidates=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                extras={"empty_reason": "no_candidate_after_filter"},
            )

        dense_hits: List[Tuple[str, float]] = []
        if "dense" in arms:
            vec = self.embedder.encode([query])[0]
            dense_hits = self.store.search(
                vec, cfg.dense_k, flt if cfg.filter_mode == "pre" else None,
                mode=cfg.filter_mode,
            )

        sparse_hits: List[Tuple[str, float]] = []
        if "sparse" in arms:
            sparse_hits = self.bm25.search(query, cfg.sparse_k, allowed_ids=candidates)

        fused = self._rrf_fuse(dense_hits, sparse_hits)[:top_k]

        # 父块回填：把命中块映射回整篇菜谱，按融合分排序
        parents = rank_parents(
            fused,
            self.recipe_map,
            strategy=cfg.parent_strategy,
            top_n=cfg.parent_top_n,
        )

        return RetrievalResult(
            query=query,
            fused=fused,
            parents=parents,
            filter_desc=flt.describe() if flt and not flt.is_empty() else "无过滤",
            candidates=len(candidates) if candidates is not None else len(self.chunk_map),
            dense_hits=len(dense_hits),
            sparse_hits=len(sparse_hits),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    def info(self) -> Dict[str, object]:
        return {
            "chunks": len(self.chunk_map),
            "recipes": len(self.recipe_map),
            "bm25": self.bm25.stats(),
            "retrieval": self.cfg.describe(),
            "store": type(self.store).__name__,
            "embedder": getattr(self.embedder, "name", "unknown"),
        }
