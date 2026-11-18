"""生成评测集候选 query：只产候选，不产最终集。

用法：

    # 先看每类会发什么 prompt，不调用模型
    python scripts/gen_eval_candidates.py --dry-run

    # 真实生成（魔搭 OpenAI 兼容端点）
    export MODELSCOPE_API_KEY=...
    python scripts/gen_eval_candidates.py --counts single_hop=48 constraint=36 \
        multi_ingredient=28 compare=12 multi_hop=12 refusal=14

输出 eval/queries.candidates.jsonl。单跳与对比自动填 gold；
约束、多食材、多跳的 gold 留给人工审。生成后先跑 validate_eval.py --candidates。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.config import _load_dotenv
from rag_core.data_preparation import CATEGORY_MAPPING, DatasetConfig, RecipeDataset
from rag_core.llm import ChatModel, LLMConfig
from rag_core.markdown_splitter import split_markdown
from rag_core.schema import Recipe

TYPES = ("single_hop", "constraint", "multi_ingredient", "compare", "multi_hop", "refusal")
ABBR = {
    "single_hop": "s",
    "constraint": "c",
    "multi_ingredient": "m",
    "compare": "cmp",
    "multi_hop": "mh",
    "refusal": "r",
}
DEFAULT_COUNTS = {
    "single_hop": 48,
    "constraint": 36,
    "multi_ingredient": 28,
    "compare": 12,
    "multi_hop": 12,
    "refusal": 14,
}
SCENARIOS = ("快手晚餐", "一人食", "周末待客", "减脂期", "新手第一次做", "带饭")
QUERY_CATEGORIES = (
    "meat_dish",
    "vegetable_dish",
    "soup",
    "staple",
    "breakfast",
    "dessert",
    "aquatic",
)
REFUSAL_TOPICS = (
    "分子料理球化",
    "液氮冰淇淋",
    "专业法式甜点的温控",
    "菜品的精确热量与宏量营养素",
    "低温慢煮机的具体参数",
    "商用烤箱的蒸汽比例",
    "食品添加剂的法规限量与安全评估",
    "餐厅摆盘与装饰技巧",
    "食材的采购渠道、品牌与价格",
    "食物过敏原的实验室检测",
    "特定疾病（如糖尿病肾病）的饮食处方",
    "专业厨房设备的选购与维护参数",
    "面团发酵曲线的仪器测量",
    "食品微生物检测与保质期实验",
    "专业侍酒与餐酒搭配",
    "商用中央厨房的产能规划",
)
SYSTEM = "你是中文菜谱评测集的出题助手。只输出一个 JSON 对象，不要 markdown 代码块，不要解释。"


def excerpt(text: str, limit: int = 700) -> str:
    sections = split_markdown(text, max_level=3, drop_comments=True)
    body = "\n".join(s.body for s in sections[:3])
    return body[:limit]


def parse_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"模型未返回 JSON：{text[:120]!r}")
    return json.loads(cleaned[start : end + 1])


def normalize(text: str) -> str:
    return re.sub(r"[\s，。？！、；：,.?!;:]+", "", text).lower()


def build_single(
    rng: random.Random, by_name: Dict[str, List[Recipe]]
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    dish = rng.choice(sorted(by_name))
    group = by_name[dish]
    recipe = rng.choice(group)
    user = (
        f"菜谱：{dish}\n正文片段：\n{excerpt(recipe.text)}\n\n"
        f"请写一个用户会问的中文问题，必须包含菜名「{dish}」，"
        "答案能从这段正文中找到（时间、原料、步骤或技巧之一）。"
        '只输出 JSON：{"query": "..."}'
    )
    return (
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        {
            "gold_parent_ids": [r.parent_id for r in group],
            "gold_dishes": [r.dish_name for r in group],
            "filters": {},
            "source_recipe_ids": [recipe.parent_id],
            "notes": f"单跳，gold 收「{dish}」的全部变体",
        },
    )


def build_constraint(rng: random.Random) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    key = rng.choice(QUERY_CATEGORIES)
    level = rng.choice((1, 2, 3))
    scenario = rng.choice(SCENARIOS)
    user = (
        f"需求：{CATEGORY_MAPPING[key]}，难度不超过 {level}（1 最简单，5 最难），"
        f"场景：{scenario}。请写一个中文查询，不出现任何具体菜名，像用户随口说的。"
        '只输出 JSON：{"query": "..."}'
    )
    return (
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        {
            "gold_parent_ids": [],
            "gold_dishes": [],
            "filters": {"category_keys": [key], "difficulty_max": level},
            "source_recipe_ids": [],
            "notes": f"约束待人工补 gold；场景={scenario}",
        },
    )


def build_multi_ingredient(
    rng: random.Random, recipes: Sequence[Recipe]
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    recipe = rng.choice([r for r in recipes if r.category_key in QUERY_CATEGORIES])
    user = (
        f"菜谱：{recipe.dish_name}\n正文：\n{excerpt(recipe.text, 900)}\n\n"
        f"请从正文挑 2–3 种主要食材，写一个只提食材、不提菜名「{recipe.dish_name}」的中文查询，"
        "问这些食材能做什么。"
        '只输出 JSON：{"query": "...", "ingredients": ["...", "..."]}'
    )
    return (
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        {
            "gold_parent_ids": [],
            "gold_dishes": [],
            "filters": {},
            "source_recipe_ids": [recipe.parent_id],
            "source_dishes": [recipe.dish_name],
            "notes": "多食材，gold 待人工补",
        },
    )


def build_compare(
    rng: random.Random, by_category: Dict[str, List[Recipe]]
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    keys = [
        k for k in QUERY_CATEGORIES
        if len({r.dish_name for r in by_category.get(k, [])}) >= 2
    ]
    key = rng.choice(keys)
    names = sorted({r.dish_name for r in by_category[key]})
    name_a, name_b = rng.sample(names, 2)
    first = rng.choice([r for r in by_category[key] if r.dish_name == name_a])
    second = rng.choice([r for r in by_category[key] if r.dish_name == name_b])
    user = (
        f"两道菜：{first.dish_name}、{second.dish_name}\n"
        f"片段 A：\n{excerpt(first.text, 400)}\n片段 B：\n{excerpt(second.text, 400)}\n\n"
        "请写一个用户想比较两者做法、口味或难度的中文问题，必须同时包含两个菜名。"
        '只输出 JSON：{"query": "..."}'
    )
    return (
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        {
            "gold_parent_ids": [first.parent_id, second.parent_id],
            "gold_dishes": [first.dish_name, second.dish_name],
            "filters": {},
            "source_recipe_ids": [first.parent_id, second.parent_id],
            "notes": "对比，两道菜都必须进 top-k",
        },
    )


def build_multi_hop(
    rng: random.Random, by_category: Dict[str, List[Recipe]]
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    keys = [k for k in QUERY_CATEGORIES if by_category.get(k)]
    key_a, key_b = rng.sample(keys, 2)
    first = rng.choice(by_category[key_a])
    second = rng.choice(by_category[key_b])
    goal_a = f"一道{CATEGORY_MAPPING[key_a]}"
    goal_b = f"一道{CATEGORY_MAPPING[key_b]}"
    user = (
        f"子目标 1：{goal_a}；子目标 2：{goal_b}。"
        "请写一个需要分两步检索的中文问题，不要出现具体菜名。"
        '只输出 JSON：{"query": "..."}'
    )
    return (
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        {
            "gold_parent_ids": [],
            "gold_dishes": [],
            "filters": {},
            "source_recipe_ids": [first.parent_id, second.parent_id],
            "source_dishes": [first.dish_name, second.dish_name],
            "notes": "多跳，gold 待人工补；参考菜为源菜谱，不必是唯一答案",
        },
    )


def build_refusal(rng: random.Random) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    topic = rng.choice(REFUSAL_TOPICS)
    user = (
        "语料是 322 篇家常中餐菜谱。请写一个看起来合理的烹饪问题，但这个语料无法回答，"
        f"主题：{topic}。不要出现具体菜名。"
        '只输出 JSON：{"query": "...", "reason": "..."}'
    )
    return (
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        {
            "gold_parent_ids": [],
            "gold_dishes": [],
            "filters": {},
            "source_recipe_ids": [],
            "notes": "拒答，gold 为空",
            "expected_reason": "",
        },
    )


def parse_counts(items: Sequence[str]) -> Dict[str, int]:
    counts = dict(DEFAULT_COUNTS)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--counts 需要 key=value，收到 {item!r}")
        key, value = item.split("=", 1)
        if key not in TYPES:
            raise SystemExit(f"未知类型 {key!r}，可选 {TYPES}")
        counts[key] = int(value)
    return counts


class MockChat:
    """本地联调用：不联网，按 prompt 回一个结构合法的 query。"""

    def chat(self, messages: Sequence[Dict[str, str]], **overrides) -> SimpleNamespace:
        user = messages[-1]["content"]
        if "两道菜：" in user:
            pair = user.split("两道菜：", 1)[1].split("\n", 1)[0]
            first, second = pair.split("、", 1)
            text = json.dumps({"query": f"{first}和{second}做法上有什么区别？"}, ensure_ascii=False)
        elif "必须包含菜名" in user:
            dish = user.split("菜谱：", 1)[1].split("\n", 1)[0]
            text = json.dumps({"query": f"{dish}要炖多久？"}, ensure_ascii=False)
        elif "只提食材" in user:
            text = json.dumps({"query": "有鸡翅和生抽能做什么？"}, ensure_ascii=False)
        elif "分两步检索" in user:
            text = json.dumps({"query": "推荐一道荤菜和一道素菜"}, ensure_ascii=False)
        elif "无法回答" in user:
            text = json.dumps(
                {"query": "怎么做分子料理球化？", "reason": "语料没有分子料理技术"},
                ensure_ascii=False,
            )
        else:
            text = json.dumps({"query": "简单快手的菜"}, ensure_ascii=False)
        return SimpleNamespace(text=text)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成评测集候选 query")
    parser.add_argument("--root", default="data/C8/cook")
    parser.add_argument("--out", default="eval/queries.candidates.jsonl")
    parser.add_argument("--counts", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0, help="总调用上限，0 表示不限制")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不调用模型")
    parser.add_argument("--mock", action="store_true", help="用本地假模型跑通流程，不联网")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="MODELSCOPE_API_KEY")
    args = parser.parse_args()
    _load_dotenv()
    model_id = args.model or os.environ.get("LLM_MODEL", "deepseek-chat")
    base_url = args.base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")

    counts = parse_counts(args.counts)
    rng = random.Random(args.seed)
    ds = RecipeDataset(DatasetConfig(root=args.root)).load()
    by_name: Dict[str, List[Recipe]] = {}
    by_category: Dict[str, List[Recipe]] = {}
    for recipe in ds.recipes:
        by_name.setdefault(recipe.dish_name, []).append(recipe)
        by_category.setdefault(recipe.category_key, []).append(recipe)

    builders = {
        "single_hop": lambda: build_single(rng, by_name),
        "constraint": lambda: build_constraint(rng),
        "multi_ingredient": lambda: build_multi_ingredient(rng, ds.recipes),
        "compare": lambda: build_compare(rng, by_category),
        "multi_hop": lambda: build_multi_hop(rng, by_category),
        "refusal": lambda: build_refusal(rng),
    }

    if args.dry_run:
        for typ in TYPES:
            messages, partial = builders[typ]()
            print(f"\n=== {typ} ===")
            print(json.dumps({"messages": messages, "partial": partial}, ensure_ascii=False, indent=2))
        return 0

    api_key = os.environ.get(args.api_key_env) or os.environ.get("LLM_API_KEY", "")
    if args.mock:
        model = MockChat()
    else:
        if not api_key:
            print(f"缺少 {args.api_key_env}（或 LLM_API_KEY）", file=sys.stderr)
            return 2
        model = ChatModel(
            LLMConfig(
                model=model_id,
                base_url=base_url,
                api_key=api_key,
                temperature=0.8,
                max_tokens=512,
                timeout=60.0,
                max_retries=2,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    calls = 0
    rows: List[Dict[str, Any]] = []

    for typ in TYPES:
        made = 0
        attempts = 0
        while made < counts[typ] and attempts < counts[typ] * 4:
            if args.limit and calls >= args.limit:
                break
            attempts += 1
            messages, partial = builders[typ]()
            try:
                result = model.chat(messages)
                calls += 1
                data = parse_json(result.text)
                query = str(data.get("query", "")).strip()
                if not query:
                    raise ValueError("query 为空")

                if typ == "single_hop":
                    dish = partial["gold_dishes"][0]
                    if dish not in query:
                        raise ValueError(f"query 未包含菜名 {dish!r}")
                if typ == "compare":
                    missing = [d for d in partial["gold_dishes"] if d not in query]
                    if missing:
                        raise ValueError(f"query 未包含 {missing}")
                if typ == "refusal":
                    reason = str(data.get("reason", "")).strip()
                    if not reason:
                        raise ValueError("refusal 缺少 reason")
                    partial["expected_reason"] = reason
                if typ in {"constraint", "multi_ingredient", "multi_hop"}:
                    leaked = [d for d in partial.get("source_dishes", []) if d in query]
                    if leaked:
                        raise ValueError(f"query 泄露了源菜名 {leaked}")

                norm = normalize(query)
                if norm in seen:
                    continue
                seen.add(norm)

                row = {
                    "qid": f"cand-{ABBR[typ]}-{made + 1:03d}",
                    "split": "candidate",
                    "type": typ,
                    "query": query,
                    "filters": partial["filters"],
                    "gold_parent_ids": partial["gold_parent_ids"],
                    "gold_dishes": partial["gold_dishes"],
                    "expected_reason": partial.get("expected_reason"),
                    "notes": partial["notes"],
                    "source": "llm",
                    "source_recipe_ids": partial["source_recipe_ids"],
                    "reviewed": False,
                }
                rows.append(row)
                made += 1
                print(f"[{typ} {made}/{counts[typ]}] {query}")
            except Exception as exc:
                print(f"[{typ}] 跳过：{type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(0.2)
        if made < counts[typ]:
            print(f"[{typ}] 只生成 {made}/{counts[typ]} 条", file=sys.stderr)

    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    print(f"\n写入 {out}：{len(rows)} 条，调用模型 {calls} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
