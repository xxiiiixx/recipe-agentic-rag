"""在评测集上计算检索指标（不调用 LLM）。

用法：
    python scripts/eval_retrieval.py \
      --config configs/local-hit.yaml \
      --queries eval/queries.jsonl \
      --split test \
      --arms dense sparse \
      --out /backup01/caiwang/recipe-rag/outputs/eval/retrieval-base.json

指标：Recall@5/10、MRR、nDCG@10（二值）、候选存活率、约束合规率、p50/p95 延迟。
refusal 不参与检索指标，单独计数量。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.config import AppConfig
from rag_core.pipeline import RecipeRAG
from rag_core.retrieval import HybridRetriever
from rag_core.schema import Chunk, SearchFilter


def recipe_view(recipe) -> Chunk:
    return Chunk(
        chunk_id=recipe.parent_id, parent_id=recipe.parent_id, source_rel=recipe.source_rel,
        dish_name=recipe.dish_name, category=recipe.category, category_key=recipe.category_key,
        difficulty=recipe.difficulty, difficulty_level=recipe.difficulty_level,
        section="", section_path="", chunk_index=0, text="",
    )


def ndcg(rels: Sequence[int], n_relevant: int, k: int = 10) -> float:
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索指标评测")
    parser.add_argument("--config", default="configs/local-hit.yaml")
    parser.add_argument("--queries", default="eval/queries.jsonl")
    parser.add_argument("--split", default="test", choices=["test", "dev", "all"])
    parser.add_argument("--arms", nargs="*", choices=["dense", "sparse"], default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    rag = RecipeRAG.load(cfg)
    base = rag.retriever.cfg
    eval_cfg = replace(
        base,
        top_k=max(args.top_k * 3, base.top_k),
        dense_k=max(args.top_k * 3, base.dense_k),
        sparse_k=max(args.top_k * 3, base.sparse_k),
        parent_top_n=args.top_k,
        arms=tuple(args.arms) if args.arms else base.arms,
    )
    retriever = HybridRetriever(
        rag.retriever.store, rag.retriever.embedder, rag.retriever.bm25,
        rag.retriever.chunk_map, rag.retriever.recipe_map, eval_cfg,
    )

    rows = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]

    total_chunks = len(retriever.chunk_map)
    details: List[Dict] = []
    by_type: Dict[str, List[Dict]] = {}
    for i, row in enumerate(rows, start=1):
        typ = row["type"]
        if typ == "refusal":
            details.append({"qid": row["qid"], "type": typ, "skipped": True})
            continue
        flt = SearchFilter(**row.get("filters", {}))
        result = retriever.search(row["query"], flt=flt, top_k=eval_cfg.top_k)
        parents = result.parents[: args.top_k]
        gold = set(row["gold_parent_ids"])
        rels = [1 if p.recipe.parent_id in gold else 0 for p in parents]
        hits5 = sum(1 for p in parents[:5] if p.recipe.parent_id in gold)
        hits10 = sum(1 for p in parents[:10] if p.recipe.parent_id in gold)
        first = next((i + 1 for i, v in enumerate(rels) if v), None)
        row_metrics = {
            "qid": row["qid"],
            "type": typ,
            "recall@5": hits5 / len(gold) if gold else 0.0,
            "recall@10": hits10 / len(gold) if gold else 0.0,
            "mrr": 1.0 / first if first else 0.0,
            "ndcg@10": ndcg(rels, len(gold)),
            "candidates": result.candidates,
            "survival": result.candidates / total_chunks if total_chunks else 0.0,
            "latency_ms": result.latency_ms,
            "top_dishes": [p.recipe.dish_name for p in parents],
        }
        if typ == "constraint":
            compliant = sum(1 for p in parents if flt.matches(recipe_view(p.recipe)))
            row_metrics["compliance"] = compliant / len(parents) if parents else 0.0
        if typ == "multi_hop":
            subs = row.get("subgoals", [])
            for k in (5, 10):
                covered = 0
                for sub in subs:
                    sub_flt = SearchFilter(**sub["filters"])
                    if any(sub_flt.matches(recipe_view(p.recipe)) for p in parents[:k]):
                        covered += 1
                row_metrics[f"subgoal@{k}"] = covered / len(subs) if subs else 0.0
        details.append(row_metrics)
        by_type.setdefault(typ, []).append(row_metrics)
        if i % 20 == 0:
            print(f"evaluated {i}/{len(rows)}")

    def agg(items: List[Dict]) -> Dict:
        items = [x for x in items if "recall@5" in x]
        if not items:
            return {"n": 0}
        lat = sorted(x["latency_ms"] for x in items)
        comp = [x["compliance"] for x in items if "compliance" in x]
        sub5 = [x["subgoal@5"] for x in items if "subgoal@5" in x]
        sub10 = [x["subgoal@10"] for x in items if "subgoal@10" in x]
        return {
            "n": len(items),
            "recall@5": statistics.mean(x["recall@5"] for x in items),
            "recall@10": statistics.mean(x["recall@10"] for x in items),
            "mrr": statistics.mean(x["mrr"] for x in items),
            "ndcg@10": statistics.mean(x["ndcg@10"] for x in items),
            "survival": statistics.mean(x["survival"] for x in items),
            "compliance": statistics.mean(comp) if comp else None,
            "subgoal@5": statistics.mean(sub5) if sub5 else None,
            "subgoal@10": statistics.mean(sub10) if sub10 else None,
            "p50_ms": statistics.median(lat),
            "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        }

    standard = [
        x for x in details
        if x.get("type") not in {"multi_hop"} and "recall@5" in x
    ]
    report = {
        "config": args.config,
        "split": args.split,
        "arms": list(eval_cfg.arms),
        "overall": agg(standard),
        "by_type": {k: agg(v) for k, v in by_type.items()},
        "refusal_count": sum(1 for x in details if x.get("skipped")),
        "details": details,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'type':<18}{'n':>4}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'nDCG':>8}{'surv':>8}{'p50ms':>8}")
    for typ, items in by_type.items():
        a = agg(items)
        print(f"{typ:<18}{a['n']:>4}{a['recall@5']:>8.3f}{a['recall@10']:>8.3f}"
              f"{a['mrr']:>8.3f}{a['ndcg@10']:>8.3f}{a['survival']:>8.3f}{a['p50_ms']:>8.1f}")
    a = report["overall"]
    print(f"{'overall':<18}{a['n']:>4}{a['recall@5']:>8.3f}{a['recall@10']:>8.3f}"
          f"{a['mrr']:>8.3f}{a['ndcg@10']:>8.3f}{a['survival']:>8.3f}{a['p50_ms']:>8.1f}")
    mh = report["by_type"].get("multi_hop")
    if mh and mh.get("n"):
        print(f"multi_hop subgoal@5={mh['subgoal@5']:.3f}  subgoal@10={mh['subgoal@10']:.3f}"
              f"  exact R@10={mh['recall@10']:.3f}")
    con = report["by_type"].get("constraint")
    if con and con.get("n"):
        print(f"constraint compliance={con['compliance']:.3f}")
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
