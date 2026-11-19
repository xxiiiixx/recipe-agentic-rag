"""对比两份生成评测结果，按 query 配对做 bootstrap。

用法：
    python scripts/compare_arms.py --a outputs/agentic-static.jsonl \
      --b outputs/agentic-agent.jsonl --out outputs/agentic-compare.json

为什么按 query 配对：同一条 query 在不同臂上的难度差异远大于臂间的真实差异。
逐条相减再重采样，差值里已经消掉了"这条 query 本身难不难"。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.stats import format_ci, paired_bootstrap

# (报告里的键, 记录里的取值函数, 越大越好)
METRICS = [
    ("relevance", lambda r: r["judge"]["relevance"]),
    ("faithfulness", lambda r: r["judge"]["faithful"]),
    ("citation_precision", lambda r: r["citation_precision"]),
    ("tokens", lambda r: r["usage"].get("total_tokens", 0)),
    ("context_tokens", lambda r: r["context_tokens"]),
    ("latency_ms", lambda r: r["latency_ms"]),
]


def load(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # 判分失败的记录不参与对比，免得把一次 JSON 解析失败算成质量差异
        if not rec.get("judge"):
            continue
        rows[rec["qid"]] = rec
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="两份生成评测的配对对比")
    parser.add_argument("--a", required=True, help="基线臂")
    parser.add_argument("--b", required=True, help="实验臂")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a_rows = load(Path(args.a))
    b_rows = load(Path(args.b))
    qids = sorted(set(a_rows) & set(b_rows))
    if not qids:
        raise SystemExit("两份结果没有共同的 qid")

    report: Dict[str, Any] = {"a": args.a, "b": args.b, "n": len(qids), "overall": {}, "by_type": {}}

    print(f"配对 query：{len(qids)}")
    print(f"  A = {args.a}\n  B = {args.b}\n")
    # 拒答题单独看拒答准确率，不混进 relevance —— 否则均值被拉低，
    # 和 eval_generation 的汇总口径也对不上
    qids_nonref = [q for q in qids if a_rows[q]["type"] != "refusal"]
    print(f"{'metric':<20}{'A':>9}{'B':>9}{'差值 (B-A)':>28}")
    for name, getter in METRICS:
        scope = qids_nonref if name in {"relevance", "faithfulness", "citation_precision"} else qids
        va = [getter(a_rows[q]) for q in scope]
        vb = [getter(b_rows[q]) for q in scope]
        diff, lo, hi = paired_bootstrap(vb, va)
        report["overall"][name] = {
            "a": round(statistics.mean(va), 4), "b": round(statistics.mean(vb), 4),
            "diff": round(diff, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "significant": lo > 0 or hi < 0,
        }
        print(f"{name:<20}{statistics.mean(va):>9.3f}{statistics.mean(vb):>9.3f}"
              f"  {format_ci(diff, lo, hi):>26}")

    refuse = [q for q in qids if a_rows[q]["type"] == "refusal"]
    if refuse:
        va = [1 if a_rows[q]["judge"].get("refusal_correct") else 0 for q in refuse]
        vb = [1 if b_rows[q]["judge"].get("refusal_correct") else 0 for q in refuse]
        diff, lo, hi = paired_bootstrap(vb, va)
        report["overall"]["refusal_accuracy"] = {
            "a": round(statistics.mean(va), 4), "b": round(statistics.mean(vb), 4),
            "diff": round(diff, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "n": len(refuse), "significant": lo > 0 or hi < 0,
        }
        print(f"{'refusal_accuracy':<20}{statistics.mean(va):>9.3f}{statistics.mean(vb):>9.3f}"
              f"  {format_ci(diff, lo, hi):>26}  (n={len(refuse)})")

    # 分类型：小样本只报差值和区间，不下结论
    buckets: Dict[str, List[str]] = defaultdict(list)
    for q in qids:
        buckets[a_rows[q]["type"]].append(q)
    print(f"\n{'type':<18}{'n':>4}{'rel A':>8}{'rel B':>8}{'差值':>26}")
    for typ, group in sorted(buckets.items()):
        if typ == "refusal":
            continue
        va = [a_rows[q]["judge"]["relevance"] for q in group]
        vb = [b_rows[q]["judge"]["relevance"] for q in group]
        diff, lo, hi = paired_bootstrap(vb, va)
        report["by_type"][typ] = {
            "n": len(group), "relevance_a": round(statistics.mean(va), 3),
            "relevance_b": round(statistics.mean(vb), 3),
            "diff": round(diff, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "significant": lo > 0 or hi < 0,
        }
        print(f"{typ:<18}{len(group):>4}{statistics.mean(va):>8.2f}"
              f"{statistics.mean(vb):>8.2f}  {format_ci(diff, lo, hi):>24}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
