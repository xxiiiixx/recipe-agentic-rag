"""命令行入口：建库 / 检索 / 问答 / 自检。

    python -m rag_core.cli build  --config configs/base.yaml
    python -m rag_core.cli search "简单的素菜，10 分钟内能做好" --max-difficulty 2 --arms dense sparse
    python -m rag_core.cli ask    "红烧肉要炖多久" --json
    python -m rag_core.cli stats  --config configs/base.yaml

``search`` 的 ``--arms`` 是消融开关：``dense`` / ``sparse`` / 两者都写，
同一份索引上跑三条臂，指标差异才有归因价值。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .config import AppConfig
from .data_preparation import CATEGORY_MAPPING
from .schema import SearchFilter

logger = logging.getLogger("rag_core.cli")


def build_filter(
    categories: Optional[List[str]] = None,
    max_difficulty: Optional[int] = None,
    min_difficulty: Optional[int] = None,
) -> SearchFilter:
    """把命令行参数翻译成结构化过滤器。

    既接受中文类目（``素菜``）也接受目录名（``vegetable_dish``），
    无法归类的词当菜名精确匹配，避免"传错了却静默忽略"。
    """
    flt = SearchFilter(difficulty_max=max_difficulty, difficulty_min=min_difficulty)
    for value in categories or []:
        if value in CATEGORY_MAPPING:
            flt.category_keys.append(value)
        elif value in CATEGORY_MAPPING.values():
            flt.categories.append(value)
        else:
            flt.dish_names.append(value)
    return flt


def _print_result(result, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_trace(), ensure_ascii=False, indent=2))
        return
    print(f"\n查询：{result.query}   过滤：{result.filter_desc}")
    print(f"候选块 {result.candidates} / 稠密命中 {result.dense_hits} / 稀疏命中 {result.sparse_hits}"
          f" / {result.latency_ms:.1f} ms\n")
    for rank, hit in enumerate(result.fused, start=1):
        c = hit.chunk
        arms = []
        if hit.dense_rank is not None:
            arms.append(f"dense#{hit.dense_rank}")
        if hit.sparse_rank is not None:
            arms.append(f"sparse#{hit.sparse_rank}")
        print(f"{rank}. [{hit.rrf_score:.5f}] {c.dish_name}｜{c.category}｜{c.section}"
              f"  ({', '.join(arms) or '—'})")
    if result.parents:
        print("\n父文档（按融合分）：")
        for i, p in enumerate(result.parents, start=1):
            print(f"  {i}. {p.recipe.dish_name}（{p.recipe.category}/{p.recipe.difficulty}）"
                  f" score={p.score:.5f} 命中块={p.hit_chunks}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_core", description="菜谱 Agentic RAG 检索底座")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="建索引并落盘数据集")
    p_build.add_argument("--skip-embedding", action="store_true", help="只切块与建 BM25，不调向量模型")

    for name, helper in (("search", "仅检索"), ("ask", "检索 + 生成")):
        p = sub.add_parser(name, help=helper)
        p.add_argument("query")
        p.add_argument("--top-k", type=int, default=None)
        p.add_argument("--category", nargs="*", default=None, help="中文类目或目录名，或菜名")
        p.add_argument("--max-difficulty", type=int, default=None)
        p.add_argument("--min-difficulty", type=int, default=None)
        p.add_argument("--arms", nargs="*", default=None, choices=["dense", "sparse"])
        p.add_argument("--json", action="store_true")

    sub.add_parser("stats", help="打印语料与索引统计")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cfg = AppConfig.from_yaml(args.config)

    if args.cmd == "build":
        from .pipeline import RecipeRAG

        rag = RecipeRAG.build(cfg, skip_embedding=args.skip_embedding)
        print(f"建库完成：{rag.dataset.stats()}")
        return 0

    if args.cmd == "stats":
        rag = RecipeRAG.load_local(cfg) if cfg.store.backend == "local" else RecipeRAG.build(cfg, skip_embedding=True)
        print(json.dumps(rag.info(), ensure_ascii=False, indent=2))
        return 0

    from .pipeline import RecipeRAG

    rag = RecipeRAG.build(cfg)
    flt = build_filter(args.category, args.max_difficulty, args.min_difficulty)
    if not flt.is_empty() and args.cmd == "search":
        logger.info("过滤条件：%s", flt.describe())

    if args.cmd == "search":
        result = rag.retrieve(args.query, flt=flt, top_k=args.top_k, arms=args.arms)
        _print_result(result, args.json)
        return 0

    answer, result = rag.ask(args.query, flt=flt, top_k=args.top_k)
    if args.json:
        print(json.dumps(
            {"answer": answer.to_dict(), "retrieval": result.to_trace()},
            ensure_ascii=False, indent=2,
        ))
    else:
        _print_result(result, as_json=False)
        print("\n回答：\n" + answer.text)
        print("\n来源：" + "；".join(answer.sources))
        if answer.usage:
            print(f"tokens：{answer.usage}")
        print(f"上下文统计：{answer.ctx_stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
