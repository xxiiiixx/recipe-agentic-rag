"""评测集校验：schema、gold 存在性、约束一致性、重复与配额。

用法：

    # 正式评测集
    python scripts/validate_eval.py --path eval/queries.jsonl

    # 候选文件（允许 split=candidate、允许 gold 为空、跳过配额）
    python scripts/validate_eval.py --path eval/queries.candidates.jsonl --candidates
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.data_preparation import DatasetConfig, RecipeDataset
from rag_core.schema import Chunk, SearchFilter

TYPES = ("single_hop", "constraint", "multi_ingredient", "compare", "multi_hop", "refusal")
TEST_QUOTA = {
    "single_hop": 35,
    "constraint": 25,
    "multi_ingredient": 20,
    "compare": 8,
    "multi_hop": 7,
    "refusal": 10,
}
DEV_QUOTA = {
    "single_hop": 5,
    "constraint": 4,
    "multi_ingredient": 3,
    "compare": 1,
    "multi_hop": 1,
    "refusal": 1,
}
FILTER_KEYS = {f.name for f in fields(SearchFilter)}
REQUIRED = {
    "qid", "split", "type", "query", "filters",
    "gold_parent_ids", "gold_dishes", "expected_reason", "notes",
}
_NORM_RE = re.compile(r"[\s，。？！、；：,.?!;:]+")


def normalize(text: str) -> str:
    return _NORM_RE.sub("", text).lower()


def recipe_view(recipe) -> Chunk:
    """把 Recipe 包装成 SearchFilter.matches 需要的字段集合。"""
    return Chunk(
        chunk_id=recipe.parent_id,
        parent_id=recipe.parent_id,
        source_rel=recipe.source_rel,
        dish_name=recipe.dish_name,
        category=recipe.category,
        category_key=recipe.category_key,
        difficulty=recipe.difficulty,
        difficulty_level=recipe.difficulty_level,
        section="",
        section_path="",
        chunk_index=0,
        text="",
    )


def validate_record(
    rec: Dict[str, Any],
    by_id: Dict[str, Any],
    seen_qids: set[str],
    seen_queries: Dict[str, str],
    candidates: bool,
) -> List[str]:
    errors: List[str] = []
    qid = str(rec.get("qid", "<no-qid>"))

    missing = REQUIRED - set(rec)
    if missing:
        return [f"{qid}: 缺字段 {sorted(missing)}"]

    if qid in seen_qids:
        errors.append(f"{qid}: qid 重复")
    seen_qids.add(qid)

    if rec["type"] not in TYPES:
        errors.append(f"{qid}: 未知 type {rec['type']!r}")
        return errors

    query = str(rec["query"]).strip()
    if not query:
        errors.append(f"{qid}: query 为空")
    norm = normalize(query)
    if norm in seen_queries:
        errors.append(f"{qid}: query 与 {seen_queries[norm]} 重复")
    seen_queries[norm] = qid

    split = rec["split"]
    if candidates:
        if split not in {"candidate"}:
            errors.append(f"{qid}: 候选文件 split 应为 candidate，收到 {split!r}")
    elif split not in {"test", "dev"}:
        errors.append(f"{qid}: split 应为 test/dev，收到 {split!r}")

    filters = rec["filters"]
    if not isinstance(filters, dict):
        errors.append(f"{qid}: filters 必须是对象")
        flt = SearchFilter()
    else:
        unknown = set(filters) - FILTER_KEYS
        if unknown:
            errors.append(f"{qid}: filters 有未知键 {sorted(unknown)}")
        try:
            flt = SearchFilter(**{k: v for k, v in filters.items() if k in FILTER_KEYS})
        except TypeError as exc:
            errors.append(f"{qid}: filters 非法：{exc}")
            flt = SearchFilter()

    gold_ids = rec["gold_parent_ids"]
    gold_names = rec["gold_dishes"]
    if not isinstance(gold_ids, list) or not isinstance(gold_names, list):
        errors.append(f"{qid}: gold_parent_ids / gold_dishes 必须是数组")
        return errors
    if len(gold_ids) != len(gold_names):
        errors.append(f"{qid}: gold_parent_ids 与 gold_dishes 长度不一致")
    if rec["type"] == "refusal":
        if gold_ids:
            errors.append(f"{qid}: refusal 的 gold 必须为空")
        if not str(rec["expected_reason"] or "").strip():
            errors.append(f"{qid}: refusal 必须写 expected_reason")
    else:
        if not gold_ids and not candidates:
            errors.append(f"{qid}: 非 refusal 的 gold 不能为空")
        if rec["type"] == "constraint" and not filters:
            errors.append(f"{qid}: constraint 的 filters 不能为空")

    for pid, dish in zip(gold_ids, gold_names):
        recipe = by_id.get(pid)
        if recipe is None:
            errors.append(f"{qid}: parent_id 不存在 {pid}")
            continue
        if recipe.dish_name != dish:
            errors.append(f"{qid}: {pid} 的菜名是 {recipe.dish_name!r}，不是 {dish!r}")
        if rec["type"] == "constraint" and not flt.matches(recipe_view(recipe)):
            errors.append(f"{qid}: gold {dish!r} 不满足 filters {filters}")

    if rec["type"] == "compare" and gold_names:
        if len(gold_names) < 2:
            errors.append(f"{qid}: compare 至少需要两个 gold")
        for dish in gold_names:
            if dish not in query:
                errors.append(f"{qid}: compare 的 query 未包含 {dish!r}")

    if rec["type"] in {"constraint", "multi_ingredient"} and gold_names:
        leaked = [d for d in gold_names if d in query]
        if leaked:
            errors.append(f"{qid}: query 泄露了 gold 菜名 {leaked}")

    if rec["type"] == "multi_hop":
        subs = rec.get("subgoals")
        if not isinstance(subs, list) or len(subs) != 2:
            errors.append(f"{qid}: multi_hop 必须有 2 个 subgoals")
        else:
            if gold_ids and len(gold_ids) != len(subs):
                errors.append(f"{qid}: gold 数量应与 subgoals 数量一致")
            for j, sub in enumerate(subs):
                filters = sub.get("filters") if isinstance(sub, dict) else None
                if not isinstance(filters, dict) or not filters:
                    errors.append(f"{qid}: subgoal[{j}] 缺少非空 filters")
                    continue
                try:
                    sub_flt = SearchFilter(**filters)
                except TypeError as exc:
                    errors.append(f"{qid}: subgoal[{j}] filters 非法：{exc}")
                    continue
                recipe = by_id.get(gold_ids[j]) if j < len(gold_ids) else None
                if recipe is not None and not sub_flt.matches(recipe_view(recipe)):
                    errors.append(f"{qid}: subgoal[{j}] 与 gold {recipe.dish_name!r} 不匹配")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="评测集校验")
    parser.add_argument("--path", default="eval/queries.jsonl")
    parser.add_argument("--root", default="data/C8/cook")
    parser.add_argument("--candidates", action="store_true", help="候选文件：跳过配额与空 gold 检查")
    parser.add_argument("--no-quota", action="store_true", help="只校验内容，不检查 105+15 配额")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"文件不存在：{path}", file=sys.stderr)
        return 2

    ds = RecipeDataset(DatasetConfig(root=args.root)).load()
    by_id = ds.recipe_map()

    records = []
    errors: List[str] = []
    seen_qids: set[str] = set()
    seen_queries: Dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"第 {lineno} 行 JSON 非法：{exc}")
            continue
        records.append(rec)
        errors.extend(validate_record(rec, by_id, seen_qids, seen_queries, args.candidates))

    counts: Dict[str, Dict[str, int]] = {}
    for rec in records:
        split = rec.get("split", "?")
        typ = rec.get("type", "?")
        counts.setdefault(split, {}).setdefault(typ, 0)
        counts[split][typ] += 1

    if not args.candidates and not args.no_quota:
        for split, quota in (("test", TEST_QUOTA), ("dev", DEV_QUOTA)):
            actual = counts.get(split, {})
            for typ, expected in quota.items():
                got = actual.get(typ, 0)
                if got != expected:
                    errors.append(f"{split}/{typ}: 应为 {expected} 条，实际 {got} 条")
            total = sum(actual.values())
            if total != sum(quota.values()):
                errors.append(f"{split}: 总数应为 {sum(quota.values())}，实际 {total}")

    print(f"文件：{path}（{len(records)} 条）")
    for split in sorted(counts):
        parts = "，".join(f"{t} {n}" for t, n in sorted(counts[split].items()))
        print(f"  {split}: {parts}")

    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\n校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
