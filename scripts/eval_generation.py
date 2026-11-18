"""生成评测：truncate vs drop 上下文策略，含 LLM judge。

用法：
    python scripts/eval_generation.py \
      --config /backup01/caiwang/recipe-rag/configs/ablation/ab-context-truncate.yaml \
      --queries eval/queries.jsonl --split test \
      --out /backup01/caiwang/recipe-rag/outputs/generation/truncate.jsonl \
      --resume

每条 query：检索 -> 拼上下文 -> DeepSeek 生成 -> DeepSeek judge。
judge 只看问题、上下文、回答，不看 gold。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.config import AppConfig
from rag_core.generation import citation_precision
from rag_core.llm import ChatModel, LLMConfig
from rag_core.pipeline import RecipeRAG
from rag_core.schema import SearchFilter

JUDGE_SYSTEM = "你是严格的 RAG 评测员。只输出一个 JSON 对象，不要 markdown，不要解释。"


def parse_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"judge 未返回 JSON：{text[:120]!r}")
    return json.loads(cleaned[start : end + 1])


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def judge_answer(
    llm: ChatModel, query: str, context: str, answer: str, is_refusal: bool
) -> Dict[str, Any]:
    prompt = (
        f"问题：{query}\n\n"
        f"参考上下文：\n{context[:6000]}\n\n"
        f"模型回答：\n{answer[:3000]}\n\n"
        "请严格判断：\n"
        "1. faithful：1 表示回答中的每个事实都能在参考上下文中找到，0 表示有编造或引入上下文外信息。\n"
        "2. relevance：1–5，5 表示直接、完整回答了问题，1 表示完全无关。\n"
        f"3. refusal_correct：{'该问题应拒答，true 表示回答明确说没有相关信息且没有编造菜谱；false 表示给出了具体菜谱或凭空补全' if is_refusal else '该问题不需要拒答，输出 null'}。\n"
        '只输出 JSON：{"faithful": 0 或 1, "relevance": 1-5, "refusal_correct": true/false/null, "reason": "..."}'
    )
    result = llm.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    data = parse_json(result.text)
    faithful = 1 if as_bool(data.get("faithful")) else 0
    try:
        relevance = max(1, min(5, int(data.get("relevance", 1))))
    except Exception:
        relevance = 1
    refusal = as_bool(data.get("refusal_correct")) if is_refusal else None
    return {
        "faithful": faithful,
        "relevance": relevance,
        "refusal_correct": refusal,
        "reason": str(data.get("reason", ""))[:500],
        "judge_usage": result.usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成评测（truncate vs drop）")
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", default="eval/queries.jsonl")
    parser.add_argument("--split", default="test", choices=["test", "dev", "all"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    rag = RecipeRAG.load(cfg)
    if rag.answerer is None:
        raise SystemExit("未配置 LLM，无法生成")

    judge = ChatModel(
        LLMConfig(
            model=cfg.llm.model,
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
            temperature=0.0,
            max_tokens=300,
            timeout=cfg.llm.timeout,
            max_retries=cfg.llm.max_retries,
        )
    )

    rows = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.is_file():
        done = {
            json.loads(line)["qid"]
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    rows = [r for r in rows if r["qid"] not in done]
    if args.limit:
        rows = rows[: args.limit]

    mode = "a" if args.resume else "w"
    with out.open(mode, encoding="utf-8") as fh:
        for i, row in enumerate(rows, start=1):
            flt = SearchFilter(**row.get("filters", {}))
            t0 = time.perf_counter()
            try:
                answer, _ = rag.ask(row["query"], flt=flt)
                gen_ms = (time.perf_counter() - t0) * 1000
                is_refusal = row["type"] == "refusal"
                try:
                    judged = judge_answer(
                        judge, row["query"], answer.context, answer.text, is_refusal
                    )
                    judge_error = None
                except Exception as exc:
                    judged = {}
                    judge_error = f"{type(exc).__name__}: {exc}"
                record = {
                    "qid": row["qid"],
                    "type": row["type"],
                    "query": row["query"],
                    "answer": answer.text,
                    "sources": answer.sources,
                    "context_tokens": answer.ctx_stats.get("context_tokens", 0),
                    "used_docs": answer.ctx_stats.get("used_docs", 0),
                    "truncated_docs": answer.ctx_stats.get("truncated_docs", 0),
                    "dropped_docs": answer.ctx_stats.get("dropped_docs", 0),
                    "usage": answer.usage,
                    "latency_ms": round(gen_ms, 1),
                    "citation_precision": citation_precision(
                        answer.text, len(answer.sources)
                    ),
                    "judge": judged,
                    "judge_error": judge_error,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{i}/{len(rows)}] {row['qid']} {gen_ms:.0f}ms")
            except Exception as exc:
                print(f"[{i}/{len(rows)}] {row['qid']} FAIL {type(exc).__name__}: {exc}")

    # 汇总
    all_rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def agg(items: List[Dict]) -> Dict:
        if not items:
            return {"n": 0}
        judged = [x for x in items if x.get("judge")]
        lat = sorted(x["latency_ms"] for x in items)
        non_ref = [x for x in judged if x["type"] != "refusal"]
        ref = [x for x in judged if x["type"] == "refusal"]
        return {
            "n": len(items),
            "judged": len(judged),
            "judge_errors": sum(1 for x in items if x.get("judge_error")),
            "faithfulness": statistics.mean(x["judge"]["faithful"] for x in non_ref)
            if non_ref else None,
            "relevance": statistics.mean(x["judge"]["relevance"] for x in non_ref)
            if non_ref else None,
            "citation_precision": statistics.mean(x["citation_precision"] for x in non_ref)
            if non_ref else None,
            "refusal_accuracy": statistics.mean(
                1 if x["judge"].get("refusal_correct") else 0 for x in ref
            ) if ref else None,
            "tokens": statistics.mean(x["usage"].get("total_tokens", 0) for x in items),
            "context_tokens": statistics.mean(x["context_tokens"] for x in items),
            "used_docs": statistics.mean(x["used_docs"] for x in items),
            "truncated_docs": statistics.mean(x["truncated_docs"] for x in items),
            "dropped_docs": statistics.mean(x["dropped_docs"] for x in items),
            "p50_ms": statistics.median(lat),
            "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        }

    by_type: Dict[str, List[Dict]] = {}
    for x in all_rows:
        by_type.setdefault(x["type"], []).append(x)
    report = {
        "config": args.config,
        "split": args.split,
        "overall": agg(all_rows),
        "by_type": {k: agg(v) for k, v in by_type.items()},
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    a = report["overall"]
    print(f"\n{'type':<18}{'n':>4}{'faith':>8}{'rel':>7}{'cite':>7}{'refuse':>8}{'tokens':>8}{'ctx':>7}{'p50':>7}")
    for typ, items in by_type.items():
        b = agg(items)
        print(f"{typ:<18}{b['n']:>4}{b['faithfulness'] if b['faithfulness'] is not None else 0:>8.3f}"
              f"{b['relevance'] if b['relevance'] is not None else 0:>7.2f}"
              f"{b['citation_precision'] if b['citation_precision'] is not None else 0:>7.3f}"
              f"{b['refusal_accuracy'] if b['refusal_accuracy'] is not None else 0:>8.3f}"
              f"{b['tokens']:>8.0f}{b['context_tokens']:>7.0f}{b['p50_ms']:>7.0f}")
    print(f"{'overall':<18}{a['n']:>4}{a['faithfulness']:>8.3f}{a['relevance']:>7.2f}"
          f"{a['citation_precision']:>7.3f}"
          f"{a['refusal_accuracy'] if a['refusal_accuracy'] is not None else 0:>8.3f}"
          f"{a['tokens']:>8.0f}{a['context_tokens']:>7.0f}{a['p50_ms']:>7.0f}")
    print(f"\n写入 {out} 与 {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
