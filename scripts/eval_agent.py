"""Agent vs 静态 RAG 的对照评测。

四臂，逐级加码，差出来的每一档都对应一个可以写进报告的结论：

| 臂 | 约束来源 | 检索次数 | 反思 |
|---|---|---|---|
| ``static-oracle`` | 评测集给定（上界） | 1 | 无 |
| ``static-parsed`` | 解析器从 query 抽 | 1 | 无 |
| ``agent-plan`` | 解析器从 query 抽 | 按意图多路 | 无（max_rounds=0） |
| ``agent-full`` | 解析器从 query 抽 | 按意图多路 + 补检 | 有（≤2 轮） |

``static-oracle`` 与 ``static-parsed`` 的差是"约束解析的代价"；
``static-parsed`` 与 ``agent-plan`` 的差是"多路检索"的收益；
``agent-plan`` 与 ``agent-full`` 的差是"反思环"的收益。
三段拆开，才知道涨的分是哪来的 —— 一起报一个总数说明不了任何问题。

约束合规率**始终按评测集给定的 filters 判定**，不用各臂自己解析出来的条件。
否则解析器什么都不抽，合规率反而 1.0。

用法：
    python scripts/eval_agent.py --config configs/local-hit.yaml \
      --split test --understand rules \
      --out /backup01/caiwang/recipe-rag/outputs/agent/agent-rules.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.agent import CachedUnderstander, RecipeAgent, build_understander
from rag_core.config import AppConfig
from rag_core.context import ContextConfig, build_context
from rag_core.pipeline import RecipeRAG
from rag_core.retrieval import rerank_by_difficulty
from rag_core.schema import ParentHit, SearchFilter, recipe_as_chunk
from rag_core.stats import format_ci, paired_bootstrap

ARMS = ("static-oracle", "static-parsed", "agent-plan", "agent-full")


# --------------------------------------------------------------------- 指标
def ndcg(rels: Sequence[int], n_relevant: int, k: int = 10) -> float:
    import math

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal else 0.0


def score_row(parents: Sequence[ParentHit], row: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    gold = set(row["gold_parent_ids"])
    picked = list(parents)[:top_k]
    rels = [1 if p.recipe.parent_id in gold else 0 for p in picked]
    first = next((i + 1 for i, v in enumerate(rels) if v), None)
    out = {
        "recall@5": sum(rels[:5]) / len(gold) if gold else 0.0,
        "recall@10": sum(rels[:10]) / len(gold) if gold else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "ndcg@10": ndcg(rels, len(gold)),
        "top_dishes": [p.recipe.dish_name for p in picked],
    }
    # 合规率用评测集给定的约束判定，与各臂自己解析出什么无关
    truth = SearchFilter(**row.get("filters", {}))
    if row["type"] == "constraint" and not truth.is_empty():
        ok = sum(1 for p in picked if truth.matches(recipe_as_chunk(p.recipe)))
        out["compliance"] = ok / len(picked) if picked else 0.0
    subs = row.get("subgoals", [])
    if row["type"] == "multi_hop" and subs:
        for k in (5, 10):
            covered = 0
            for sub in subs:
                sub_flt = SearchFilter(**sub["filters"])
                if any(sub_flt.matches(recipe_as_chunk(p.recipe)) for p in picked[:k]):
                    covered += 1
            out[f"subgoal@{k}"] = covered / len(subs)
    return out


def agg(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in rows if "recall@5" in r]
    if not rows:
        return {"n": 0}
    lat = sorted(r["latency_ms"] for r in rows)
    def mean_of(key: str):
        vals = [r[key] for r in rows if key in r]
        return statistics.mean(vals) if vals else None
    return {
        "n": len(rows),
        "recall@5": statistics.mean(r["recall@5"] for r in rows),
        "recall@10": statistics.mean(r["recall@10"] for r in rows),
        "mrr": statistics.mean(r["mrr"] for r in rows),
        "ndcg@10": statistics.mean(r["ndcg@10"] for r in rows),
        "compliance": mean_of("compliance"),
        "subgoal@5": mean_of("subgoal@5"),
        "subgoal@10": mean_of("subgoal@10"),
        "p50_ms": statistics.median(lat),
        "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
    }


# --------------------------------------------------------------------- 各臂
def run_static(rag: RecipeRAG, query: str, filters: Dict[str, Any], top_k: int,
               answerer_on: bool, ctx_cfg: ContextConfig) -> Tuple[List[ParentHit], Dict[str, Any]]:
    t0 = time.perf_counter()
    flt = SearchFilter(**filters) if filters else SearchFilter()
    # 不覆盖 top_k：让 retriever 按 eval 深度（top_k*3）融合，再在外面切到 top_k。
    # 传 top_k=10 会把融合结果先截到 10 个块，父文档数量跟着变少，和基线口径对不上。
    res = rag.retrieve(query, flt=flt)
    # 四臂统一带难度重排，否则比较里混进了"某一臂少一个机制"
    parents = rerank_by_difficulty(res.parents, flt)[:top_k]
    latency = (time.perf_counter() - t0) * 1000
    usage = {}
    if answerer_on and rag.answerer is not None:
        ans = rag.answerer.answer(query, res)
        usage = ans.usage or {}
    return parents, {
        "latency_ms": latency, "tool_calls": 1, "rounds": 0,
        "stop_reason": "single_pass", "tokens": usage.get("total_tokens", 0),
        "ctx_tokens": build_context(parents, ctx_cfg).tokens,
        "filters": flt.describe(),
    }


def run_agent(agent: RecipeAgent, query: str, top_k: int,
              ctx_cfg: ContextConfig) -> Tuple[List[ParentHit], Dict[str, Any]]:
    run = agent.run(query)
    parents = run.parents[:top_k]
    trace = run.to_trace()
    usage = (run.answer.usage if run.answer else {}) or {}
    return parents, {
        "latency_ms": trace["latency_ms"],
        "tool_calls": len(trace["tool_calls"]),
        "rounds": trace["rounds"],
        "stop_reason": trace["stop_reason"],
        "reflection_reasons": [r["reason"] for r in trace["reflections"]],
        "tokens": usage.get("total_tokens", 0),
        # 不调模型也能算的上下文成本：Agent 多检索几路，多出来的证据最终都要进 prompt
        "ctx_tokens": build_context(parents, ctx_cfg).tokens,
        "filters": "; ".join(str(c["args"].get("filters")) for c in trace["tool_calls"][:1]),
        "parsed": trace["parsed"],
    }


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Agent vs 静态 RAG 对照评测")
    parser.add_argument("--config", default="configs/local-hit.yaml")
    parser.add_argument("--queries", default="eval/queries.jsonl")
    parser.add_argument("--split", default="test", choices=["test", "dev", "all"])
    parser.add_argument("--understand", default="rules", choices=["rules", "llm"])
    parser.add_argument("--arms", nargs="*", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--base-k", type=int, default=0,
                        help="Agent 每次检索要几篇父文档，默认与 --top-k 一致")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--answer", action="store_true", help="同时生成答案并统计 token")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于冒烟")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    rag = RecipeRAG.load(cfg)

    # 与 eval_retrieval.py 保持同一套召回深度：默认 parent_top_n=2 会把每个查询的
    # 父文档截到 2 条，R@10 直接失去意义。四臂共用这一份，比较才成立。
    base_rc = rag.retriever.cfg
    rag.retriever.cfg = replace(
        base_rc,
        top_k=max(args.top_k * 3, base_rc.top_k),
        dense_k=max(args.top_k * 3, base_rc.dense_k),
        sparse_k=max(args.top_k * 3, base_rc.sparse_k),
        parent_top_n=args.top_k,
    )

    understander = CachedUnderstander(build_understander(rag, args.understand))
    ctx_cfg = ContextConfig(
        token_budget=cfg.context.token_budget,
        per_doc_budget=cfg.context.per_doc_budget,
        min_doc_tokens=cfg.context.min_doc_tokens,
        on_overflow=cfg.context.on_overflow,
    )
    base_k = args.base_k or args.top_k
    agents = {
        "agent-plan": RecipeAgent(rag, understander=understander, max_rounds=0,
                                  base_k=base_k, answer_enabled=args.answer),
        "agent-full": RecipeAgent(rag, understander=understander, max_rounds=args.max_rounds,
                                  base_k=base_k, answer_enabled=args.answer),
    }

    rows = [json.loads(line) for line in
            Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]

    details: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        record: Dict[str, Any] = {"qid": row["qid"], "type": row["type"], "query": row["query"]}
        parsed = understander.parse(row["query"])
        record["parsed"] = parsed.to_dict()

        for arm in args.arms:
            if arm == "static-oracle":
                parents, info = run_static(rag, row["query"], row.get("filters", {}),
                                           args.top_k, args.answer, ctx_cfg)
            elif arm == "static-parsed":
                parents, info = run_static(rag, row["query"], parsed.filters,
                                           args.top_k, args.answer, ctx_cfg)
            else:
                parents, info = run_agent(agents[arm], row["query"], args.top_k, ctx_cfg)

            if row["type"] == "refusal":
                # 拒答题不参与检索指标：它检验的是"会不会硬答"，不是"召回得准不准"
                record[arm] = {"skipped": True, **info}
                continue
            record[arm] = {**score_row(parents, row, args.top_k), **info}

        details.append(record)
        if i % 20 == 0:
            print(f"evaluated {i}/{len(rows)}")

    report: Dict[str, Any] = {
        "config": args.config, "split": args.split, "understand": args.understand,
        "n": len(rows), "arms": {}, "by_type": {}, "paired": [],
    }
    for arm in args.arms:
        # 与 eval_retrieval.py 同一口径：拒答不参与检索指标，多跳单独看子目标覆盖，
        # 不混进 overall —— 混进去会把"固定 gold 对本来就对不上"算成检索变差
        arm_rows = [
            r[arm] for r in details
            if not r.get(arm, {}).get("skipped") and r["type"] != "multi_hop"
        ]
        report["arms"][arm] = agg(arm_rows)
        report["arms"][arm]["mean_tool_calls"] = (
            statistics.mean(r["tool_calls"] for r in arm_rows) if arm_rows else 0.0
        )
        report["arms"][arm]["mean_rounds"] = (
            statistics.mean(r["rounds"] for r in arm_rows) if arm_rows else 0.0
        )
        report["arms"][arm]["mean_latency_ms"] = (
            statistics.mean(r["latency_ms"] for r in arm_rows) if arm_rows else 0.0
        )
        report["arms"][arm]["mean_ctx_tokens"] = (
            statistics.mean(r["ctx_tokens"] for r in arm_rows) if arm_rows else 0.0
        )
        if args.answer:
            report["arms"][arm]["mean_tokens"] = (
                statistics.mean(r["tokens"] for r in arm_rows) if arm_rows else 0.0
            )
        by_type: Dict[str, List[Dict]] = {}
        for r in details:
            item = r.get(arm, {})
            if not item.get("skipped"):
                by_type.setdefault(r["type"], []).append(item)
        report["by_type"][arm] = {k: agg(v) for k, v in by_type.items()}
        # 反思到底有没有触发、因为什么触发 —— 没有这组分布，"+0.000" 分不清是
        # "反思没用"还是"反思根本没跑"
        report["arms"][arm]["stop_reasons"] = dict(Counter(
            r[arm].get("stop_reason", "") for r in details if not r.get(arm, {}).get("skipped")
        ))
        report["arms"][arm]["reflection_reasons"] = dict(Counter(
            reason for r in details for reason in r.get(arm, {}).get("reflection_reasons", [])
        ))

    # 相邻两臂的配对 bootstrap：每一档增益单独给 CI
    ladder = [a for a in args.arms if a != "static-oracle"]
    for hi_arm, lo_arm in zip(ladder[1:], ladder[:-1]):
        for metric in ("recall@5", "recall@10", "mrr", "ndcg@10", "compliance", "subgoal@10"):
            pairs = [
                (r[hi_arm][metric], r[lo_arm][metric]) for r in details
                if not r.get(hi_arm, {}).get("skipped") and metric in r.get(hi_arm, {})
                and metric in r.get(lo_arm, {})
            ]
            if not pairs:
                continue
            a_vals = [p[0] for p in pairs]
            b_vals = [p[1] for p in pairs]
            diff, lo, hi = paired_bootstrap(a_vals, b_vals)
            report["paired"].append({
                "a": hi_arm, "b": lo_arm, "metric": metric, "n": len(pairs),
                "diff": round(diff, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "significant": lo > 0 or hi < 0,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**report, "details": details},
                                         ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 打印
    header = (f"{'arm':<16}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG':>7}{'comp':>7}"
              f"{'sub10':>7}{'calls':>7}{'ctx':>7}{'p50ms':>8}")
    print("\n" + header)
    for arm in args.arms:
        a = report["arms"][arm]
        if not a.get("n"):
            continue
        comp = f"{a['compliance']:.3f}" if a["compliance"] is not None else "  -  "
        sub = f"{a['subgoal@10']:.3f}" if a["subgoal@10"] is not None else "  -  "
        print(f"{arm:<16}{a['recall@5']:>7.3f}{a['recall@10']:>7.3f}{a['mrr']:>7.3f}"
              f"{a['ndcg@10']:>7.3f}{comp:>7}{sub:>7}{a['mean_tool_calls']:>7.2f}"
              f"{a['mean_ctx_tokens']:>7.0f}{a['p50_ms']:>8.1f}")

    if report["paired"]:
        print("\n相邻臂配对 bootstrap（95% CI，ns=不显著）")
        for p in report["paired"]:
            print(f"  {p['a']:<12} - {p['b']:<14} {p['metric']:<11}"
                  f"{format_ci(p['diff'], p['ci_low'], p['ci_high'])}")

    for arm in args.arms:
        a = report["arms"][arm]
        if a.get("stop_reasons"):
            print(f"\n{arm} stop_reason: {a['stop_reasons']}")
        if a.get("reflection_reasons"):
            print(f"{arm} 反思触发: {a['reflection_reasons']}")

    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
