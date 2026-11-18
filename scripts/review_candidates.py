"""为评测候选生成人工审核辅助：每条 query 的 top-N 检索结果 + 可编辑 TSV。

用法：
    python scripts/review_candidates.py \
      --config configs/local-hit.yaml \
      --input /backup01/caiwang/recipe-rag/outputs/eval/queries.candidates.jsonl \
      --out-dir /backup01/caiwang/recipe-rag/outputs/eval/review \
      --top-k 8

输出：
  review.tsv          可编辑：qid/type/keep/split/query/filters/gold_current/top1..topN/notes
  <type>.md           按类型分组的可读版本

人工只需在 TSV 里填 keep（1/0）、split（test/dev）和 gold_current（分号分隔菜名），
后续再写 assemble 脚本把它转换成最终的 eval/queries.jsonl。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.config import AppConfig
from rag_core.pipeline import RecipeRAG
from rag_core.retrieval import HybridRetriever
from rag_core.schema import SearchFilter


def clean(text: str) -> str:
    return str(text).replace("\t", " ").replace("\n", " ").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成评测候选审核辅助")
    parser.add_argument("--config", default="configs/local-hit.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    rag = RecipeRAG.load(cfg)
    base = rag.retriever.cfg
    review_cfg = replace(
        base,
        top_k=max(base.top_k, args.top_k * 3),
        dense_k=max(base.dense_k, args.top_k * 3),
        sparse_k=max(base.sparse_k, args.top_k * 3),
        parent_top_n=args.top_k,
    )
    retriever = HybridRetriever(
        rag.retriever.store,
        rag.retriever.embedder,
        rag.retriever.bm25,
        rag.retriever.chunk_map,
        rag.retriever.recipe_map,
        review_cfg,
    )

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_type: dict[str, list[dict]] = {}
    for i, row in enumerate(rows, start=1):
        top = []
        if row["type"] != "refusal":
            try:
                flt = SearchFilter(**row.get("filters", {}))
            except TypeError:
                flt = SearchFilter()
            result = retriever.search(row["query"], flt=flt, top_k=review_cfg.top_k)
            for hit in result.parents[: args.top_k]:
                recipe = hit.recipe
                top.append(
                    f"{recipe.dish_name}（{recipe.category}/{recipe.difficulty}，{hit.score:.4f}）"
                )
        by_type.setdefault(row["type"], []).append({**row, "_top": top})
        if i % 20 == 0:
            print(f"reviewed {i}/{len(rows)}")

    header = [
        "qid", "type", "keep", "split", "query", "filters",
        "gold_current", "expected_reason",
    ] + [f"top{i}" for i in range(1, args.top_k + 1)] + ["notes"]

    with (out / "review.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for typ, items in by_type.items():
            for row in items:
                writer.writerow([
                    row["qid"], typ, "", "", clean(row["query"]),
                    json.dumps(row.get("filters", {}), ensure_ascii=False),
                    ";".join(row.get("gold_dishes", [])),
                    clean(row.get("expected_reason") or ""),
                    *[clean(t) for t in row["_top"]],
                    clean(row.get("notes", "")),
                ])

    for typ, items in by_type.items():
        lines = [f"# {typ}（{len(items)} 条）", ""]
        for row in items:
            lines.append(f"## {row['qid']}")
            lines.append("")
            lines.append(f"- query：{row['query']}")
            if row.get("filters"):
                lines.append(f"- filters：`{json.dumps(row['filters'], ensure_ascii=False)}`")
            if row.get("gold_dishes"):
                lines.append(f"- 当前 gold：{'、'.join(row['gold_dishes'])}")
            if row.get("expected_reason"):
                lines.append(f"- expected_reason：{row['expected_reason']}")
            if row["_top"]:
                lines.append("- 检索 top：")
                lines.extend(f"  {i}. {t}" for i, t in enumerate(row["_top"], start=1))
            lines.append("")
        (out / f"{typ}.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"写入 {out}/review.tsv 与 {len(by_type)} 个类型 Markdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
