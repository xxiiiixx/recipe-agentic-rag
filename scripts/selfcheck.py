"""自检：一次跑完"语料 -> 索引 -> 检索 -> 上下文"的可验证性证据。

存在的理由：P0 改的是检索底座，光"能跑通"不算证据。这里用**构造出来的 gold**
（菜名即标签）量化四件事，并把结果同时打到终端和 ``artifacts/selfcheck/``：

1. 索引确定性 —— 同一语料两次切块，chunk_id / parent_id 必须逐条相同；
2. 分词消融   —— char_bigram vs whitespace（上游 langchain 默认按空格切分）；
3. 过滤下推   —— pre 与 post 两种模式在强过滤下的合规率与耗时；
4. 上下文组装 —— 父子块回填后进 prompt 的 token 用量与截断次数。

用法：

    # 无需 Milvus / API key，纯本地跑通链路（向量为哈希特征，只验证机制）
    python scripts/selfcheck.py --root data/C8/cook --backend local --embedding hashing

    # 服务器上真实后端
    python scripts/selfcheck.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.bm25 import BM25Index
from rag_core.config import (
    AppConfig,
    ContextCfg,
    DatasetCfg,
    EmbeddingCfg,
    LLMCfg,
    RetrievalCfg,
    StoreCfg,
)
from rag_core.context import ContextConfig, build_context
from rag_core.data_preparation import CATEGORY_MAPPING, DatasetConfig, RecipeDataset
from rag_core.pipeline import RecipeRAG
from rag_core.retrieval import HybridRetriever, RetrievalConfig
from rag_core.schema import SearchFilter

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")

PROBE_TEMPLATES = ("{dish}怎么做", "{dish}需要哪些原料", "{dish}大概多久能做好")
FILTER_PROBES = (
    "简单的素菜，十分钟能上桌",
    "不用开火的凉菜",
    "早餐吃什么好",
    "适合新手的家常菜",
    "减脂期能吃的菜",
)


# --------------------------------------------------------------------------- 1
def check_dataset(root: str, seed: int) -> Tuple[RecipeDataset, List[str]]:
    cfg = DatasetConfig(root=root)
    first = RecipeDataset(cfg).load().build_chunks()
    second = RecipeDataset(cfg).load().build_chunks()

    ids_a = [c.chunk_id for c in first.chunks]
    ids_b = [c.chunk_id for c in second.chunks]
    pids_a = [r.parent_id for r in first.recipes]
    pids_b = [r.parent_id for r in second.recipes]
    stable = ids_a == ids_b and pids_a == pids_b
    unique = len(set(ids_a)) == len(ids_a)

    print("== 1. 索引确定性")
    print(f"   菜谱 {len(first.recipes)} 篇 / 子块 {len(first.chunks)} 个 "
          f"/ 平均块长 {first.stats()['avg_chunk_chars']} 字")
    print(f"   两次切块 id 完全一致：{'是' if stable else '否'}；chunk_id 唯一：{'是' if unique else '否'}")
    dirty = [r.source_rel for r in first.recipes if "template" in r.source_rel]
    print(f"   语料洁净度：模板/示例 {len(dirty)} 篇 {dirty[:3]}（应为 0，取数时已删除）")

    rng = random.Random(seed)
    names = sorted({r.dish_name for r in first.recipes if len(r.dish_name) >= 2})
    probes = rng.sample(names, min(12, len(names)))
    return first, probes


# --------------------------------------------------------------------------- 2
def check_tokenizer(dataset: RecipeDataset, probes: Sequence[str]) -> None:
    print("\n== 2. 稀疏路分词消融（chunk 级，gold=菜名）")
    print(f"   {'分词':<14}{'有命中':>8}{'Hit@1':>8}{'Hit@5':>8}")
    rows = []
    for tokenizer in ("char_bigram", "whitespace"):
        index = BM25Index(tokenizer_name=tokenizer).fit(dataset.chunks)
        hits = hit1 = hit5 = 0
        for dish in probes:
            found = index.search(PROBE_TEMPLATES[0].format(dish=dish), top_k=10)
            if found:
                hits += 1
            dishes = [dataset.chunk_map()[cid].dish_name for cid, _ in found[:5]]
            hit1 += int(bool(dishes) and dishes[0] == dish)
            hit5 += int(dish in dishes)
        n = len(probes)
        rows.append((tokenizer, hits / n, hit1 / n, hit5 / n))
        print(f"   {tokenizer:<14}{f'{hits / n:>8.0%}'}{f'{hit1 / n:>8.0%}'}{f'{hit5 / n:>8.0%}'}")
    if rows[0][1] == 0 and rows[1][1] == 0:
        print("   注意：两臂都零命中，可能是 --embedding hashing 之外的因素，检查语料路径")


# --------------------------------------------------------------------------- 3
def check_arms(rag: RecipeRAG, probes: Sequence[str]) -> None:
    print("\n== 3. 检索臂对比（父文档级，gold=菜名，top_n=2）")
    print(f"   {'臂':<16}{'Hit@1':>8}{'Hit@2':>8}{'MRR':>8}{'耗时ms':>10}")
    for arms in (("dense",), ("sparse",), ("dense", "sparse")):
        hit1 = hit2 = 0
        rr: List[float] = []
        cost: List[float] = []
        for dish in probes:
            result = rag.retrieve(PROBE_TEMPLATES[0].format(dish=dish), arms=arms, top_k=5)
            cost.append(result.latency_ms)
            names = [p.recipe.dish_name for p in result.parents]
            hit1 += int(bool(names) and names[0] == dish)
            hit2 += int(dish in names[:2])
            rank = names.index(dish) + 1 if dish in names else None
            rr.append(1.0 / rank if rank else 0.0)
        n = len(probes)
        label = "+".join(arms)
        print(f"   {label:<16}{f'{hit1 / n:>8.0%}'}{f'{hit2 / n:>8.0%}'}"
              f"{f'{sum(rr) / n:>8.3f}'}{f'{statistics.mean(cost):>10.1f}'}")


# --------------------------------------------------------------------------- 4
def check_filter(rag: RecipeRAG, filter_mode: str) -> List[Dict[str, object]]:
    flt = SearchFilter(category_keys=["vegetable_dish"], difficulty_max=2)
    base = rag.retriever.cfg
    retriever = HybridRetriever(
        rag.retriever.store, rag.retriever.embedder, rag.retriever.bm25,
        rag.retriever.chunk_map, rag.retriever.recipe_map,
        replace(base, filter_mode=filter_mode),
    )
    rows = []
    for query in FILTER_PROBES:
        result = retriever.search(query, flt=flt)
        names = [p.recipe for p in result.parents]
        compliant = sum(
            1 for r in names
            if r.category_key == "vegetable_dish" and 0 < r.difficulty_level <= 2
        )
        rows.append({
            "query": query,
            "candidates": result.candidates,
            "parents": len(names),
            "compliance": compliant / len(names) if names else 0.0,
            "latency_ms": result.latency_ms,
        })
    return rows


def check_context(rag: RecipeRAG, probes: Sequence[str]) -> None:
    print("\n== 5. 上下文组装（文档内截断）")
    cfg = ContextConfig(
        token_budget=rag.cfg.context.token_budget,
        per_doc_budget=rag.cfg.context.per_doc_budget,
        min_doc_tokens=rag.cfg.context.min_doc_tokens,
    )
    rows = []
    for dish in probes[:4]:
        result = rag.retrieve(PROBE_TEMPLATES[1].format(dish=dish))
        built = build_context(result.parents, cfg)
        rows.append((dish, built.tokens, len(built.used), built.truncated_docs,
                     built.dropped_docs))
        print(f"   {dish:<10} tokens={built.tokens:<5} 入选 {len(built.used)} 篇 "
              f"截断 {built.truncated_docs} 篇 丢弃 {built.dropped_docs} 篇")


# --------------------------------------------------------------------------- main
def build_config(args) -> AppConfig:
    if args.config:
        return AppConfig.from_yaml(args.config)
    return AppConfig(
        name="selfcheck",
        dataset=DatasetCfg(root=args.root),
        embedding=EmbeddingCfg(provider=args.embedding, model=args.model, dim=args.dim),
        store=StoreCfg(backend=args.backend, uri=args.uri, collection=args.collection,
                       recreate=True),
        retrieval=RetrievalCfg(top_k=5, dense_k=args.dense_k, sparse_k=args.sparse_k),
        context=ContextCfg(),
        llm=LLMCfg(model="none"),  # 自检不调生成，避免把 API 可用性混进来
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 检索底座自检")
    parser.add_argument("--config", default=None, help="用配置文件覆盖下面所有参数")
    parser.add_argument("--root", default="data/C8/cook")
    parser.add_argument("--backend", default="local", choices=["local", "milvus"])
    parser.add_argument("--embedding", default="hashing",
                        choices=["hashing", "sentence_transformers", "openai_compatible"])
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--uri", default="http://localhost:19530")
    parser.add_argument("--collection", default="recipe_selfcheck")
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--sparse-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", default="artifacts/selfcheck")
    args = parser.parse_args()

    if not Path(args.root).exists() and not args.config:
        print(f"语料目录不存在：{args.root}（先按下文第 3 节取数）", file=sys.stderr)
        return 2

    cfg = build_config(args)
    dataset, probes = check_dataset(cfg.dataset.root, args.seed)
    check_tokenizer(dataset, probes)

    started = time.perf_counter()
    rag = RecipeRAG.build(cfg)
    build_s = time.perf_counter() - started
    print(f"\n   建库完成：{build_s:.1f}s，后端 {cfg.store.backend}，向量维度 {cfg.embedding.dim}")
    check_arms(rag, probes)

    print("\n== 4. 过滤下推（category=素菜 且 difficulty<=2）")
    pre = check_filter(rag, "pre")
    post = check_filter(rag, "post")
    print(f"   {'模式':<8}{'平均候选':>10}{'平均父文档':>12}{'合规率':>10}{'耗时ms':>10}")
    for mode, rows in (("pre", pre), ("post", post)):
        cand = statistics.mean(r["candidates"] for r in rows)
        parents = statistics.mean(r["parents"] for r in rows)
        compliance = statistics.mean(r["compliance"] for r in rows)
        cost = statistics.mean(r["latency_ms"] for r in rows)
        print(f"   {mode:<8}{cand:>10.0f}{parents:>12.1f}{compliance:>9.0%}{cost:>10.1f}")

    check_context(rag, probes)

    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "timestamp": stamp,
        "dataset": dataset.stats(),
        "config": cfg.describe(),
        "filter_pre": pre,
        "filter_post": post,
        "probes": probes,
    }
    path = out / f"selfcheck-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {path}")
    print("指标口径：gold 由语料派生（查询里的菜名），因此它是**机制自检**而非问答质量评测；"
          "真正的人工标注评测集见 eval/。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
