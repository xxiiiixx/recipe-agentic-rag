"""D1 模型探针：验证魔搭 OpenAI 兼容端点的 function calling 是否稳定。

用法：

    export MODELSCOPE_API_KEY=...
    python scripts/probe_models.py \
        --models Qwen/Qwen3.5-35B-A3B Qwen/Qwen3.5-122B-A10B \
                 deepseek-ai/DeepSeek-V4.1-Flash ZhipuAI/GLM-4.7-Flash

每个模型跑 N 次同一个检索工具调用，记录合法调用数、p50 延迟、错误。
判定标准：合法调用 >= 9/10 且没有持续 429，才有资格进入 Agent 跑批。
结果写入 artifacts/probe/models-<timestamp>.json，不写 key。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

BASE_URL = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
API_KEY = os.environ.get("MODELSCOPE_API_KEY") or os.environ.get("LLM_API_KEY", "")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": "在中文菜谱库中按语义检索菜谱，返回带来源的候选菜谱。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户的检索问题"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "description": "荤菜/素菜/汤品等"},
                            "max_difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                    },
                },
                "required": ["query"],
            },
        },
    }
]

PROMPT = (
    "用户问：红烧肉要炖多久？"
    "请调用 search_recipes 工具检索菜谱，不要直接回答。"
)


def one_call(client: OpenAI, model: str, timeout: float) -> tuple[float, dict]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
        max_tokens=256,
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    message = resp.choices[0].message
    calls = message.tool_calls or []
    if len(calls) != 1:
        raise ValueError(f"tool_calls={len(calls)}，期望 1")
    call = calls[0]
    if call.function.name != "search_recipes":
        raise ValueError(f"调用了 {call.function.name}，期望 search_recipes")
    args = json.loads(call.function.arguments)
    if not isinstance(args.get("query"), str) or not args["query"].strip():
        raise ValueError(f"query 非法：{args!r}")
    return latency_ms, args


def probe(model: str, runs: int, timeout: float) -> dict:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    ok = 0
    latencies: list[float] = []
    errors: list[str] = []
    for i in range(runs):
        try:
            latency, args = one_call(client, model, timeout)
            ok += 1
            latencies.append(latency)
            print(f"  [{i + 1}/{runs}] ok {latency:.0f} ms  args={args}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            print(f"  [{i + 1}/{runs}] fail {type(exc).__name__}: {exc}")
        time.sleep(0.3)
    passed = ok >= max(9, int(runs * 0.9))
    return {
        "model": model,
        "runs": runs,
        "valid": ok,
        "valid_rate": round(ok / runs, 3),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "errors": errors,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="魔搭 function calling 探针")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out-dir", default="artifacts/probe")
    args = parser.parse_args()

    if not API_KEY:
        print("缺少 MODELSCOPE_API_KEY（或 LLM_API_KEY）", file=sys.stderr)
        return 2

    print(f"base_url = {BASE_URL}")
    results = []
    for model in args.models:
        print(f"\n== {model} ==")
        results.append(probe(model, args.runs, args.timeout))

    print("\n== 汇总 ==")
    print(f"{'模型':<40}{'合法':>6}{'p50(ms)':>10}{'通过':>6}")
    for r in results:
        print(f"{r['model']:<40}{r['valid']:>4}/{r['runs']:<3}"
              f"{r['p50_latency_ms'] or 0:>10.0f}{'是' if r['passed'] else '否':>6}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out / f"models-{stamp}.json"
    path.write_text(
        json.dumps({"base_url": BASE_URL, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已写入 {path}")
    return 0 if any(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
