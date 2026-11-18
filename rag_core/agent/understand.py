"""查询理解：把一句自然语言拆成"意图 + 结构化约束 + 实体"。

为什么规则优先、LLM 兜底：

- 约束（类目、难度上限）最终会变成 Milvus 的 ``expr`` 和 BM25 的候选集，一个字段解析错，
  召回直接归零。规则在这类**闭集**上准确率更高，而且完全可复现 —— 105 条 query 跑三遍
  结果一致，消融才站得住。
- 规则覆盖不到的句式（"我一个人吃饭""手残党也能做"）才交给 LLM。LLM 只负责把话翻成
  同一套字段，不负责决定"够不够"。
- 解析来源会写进 trace（``source: rules | llm | llm_fallback``），出问题能一眼定位是谁的锅。

词表不写死：类目键从数据集里的 ``category_key`` 来，菜名从 ``dish_name`` 来。
语料换了，解析器跟着换，不用改代码。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..llm import ChatModel

logger = logging.getLogger(__name__)

INTENTS = ("single_hop", "constraint", "compare", "multi_hop")

# 类目关键词。刻意用具体词而不是"肉/菜"这类单字：单字会在菜名里误命中
# （"肉末茄子"里的"肉"不代表它属于荤菜目录）。
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "soup": ["汤", "羹", "煲汤", "炖汤"],
    "vegetable_dish": ["素菜", "素食", "青菜", "蔬菜", "凉拌", "素炒"],
    "meat_dish": ["荤菜", "肉菜", "红烧肉", "排骨", "鸡翅", "牛腩"],
    "staple": ["主食", "面食", "米饭", "面条", "粥", "馒头", "饺子", "包子", "炒饭"],
    "dessert": ["甜品", "甜点", "蛋糕", "布丁", "糖水", "慕斯"],
    "breakfast": ["早餐", "早饭"],
    "drink": ["饮品", "饮料", "果汁", "奶茶", "豆浆"],
    "aquatic": ["水产", "海鲜", "鱼", "虾", "蟹", "贝"],
    "condiment": ["调料", "酱料", "蘸料"],
    "semi-finished": ["半成品"],
}

EASY_WORDS = ["新手", "零基础", "手残", "特别简单", "非常简单", "很简单", "简单",
              "容易", "快手", "省事", "不难", "小白"]
HARD_WORDS = ["复杂", "有挑战", "很难", "比较难", "费时", "耗时", "考验"]

COMPARE_MARKERS = ["哪个", "哪一个", "还是", "对比", "比较", "区别", "差别", "vs", "VS", "相比"]
MULTI_HOP_MARKERS = ["还要", "再配", "搭配", "配个", "配什么", "加一个", "并且", "同时", "另外"]

_LLM_PROMPT = """你是查询解析器。只输出 JSON，不要解释、不要 markdown 代码块。
字段：
- intent: single_hop | constraint | compare | multi_hop
- entities: 查询里提到的菜名数组，没有就空数组
- category_keys: 类目数组，可选 {categories}
- difficulty_max: 整数 1-5 或 null（1 最简单）
- difficulty_min: 整数 1-5 或 null

判定：问两道及以上菜的差别用 compare；要求同时满足两组条件（如"汤 + 还要主食"）用 multi_hop；
只给条件不给菜名用 constraint；其余 single_hop。

查询：{query}"""


@dataclass
class ParsedQuery:
    intent: str = "single_hop"
    entities: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    source: str = "rules"
    raw: Dict[str, Any] = field(default_factory=dict)

    def has_filters(self) -> bool:
        return bool(self.filters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "filters": self.filters,
            "source": self.source,
        }


class RuleUnderstander:
    """确定性解析器。构造时吃一次数据集，之后纯函数。"""

    def __init__(self, dish_names: Sequence[str], category_keys: Sequence[str]) -> None:
        # 长名优先：先匹配"红烧肉"再匹配"肉"，避免长菜名被短词切碎
        self.dish_names = sorted({d for d in dish_names if d}, key=len, reverse=True)
        self.category_keys = set(category_keys)

    def parse(self, query: str) -> ParsedQuery:
        q = query.strip()
        entities = self._match_dishes(q)
        filters: Dict[str, Any] = {}

        if entities:
            filters["dish_names"] = entities
        else:
            keys = self._match_categories(q)
            if keys:
                filters["category_keys"] = keys

        diff = self._match_difficulty(q)
        filters.update(diff)

        intent = self._intent(q, entities, filters)
        if intent == "compare":
            # "哪个更简单"里的"简单"是比较维度，不是过滤条件。
            # 当成过滤条件会把难度高的菜挡在候选外，恰恰是比较题要看的对象。
            filters.pop("difficulty_max", None)
            filters.pop("difficulty_min", None)

        return ParsedQuery(intent=intent, entities=entities, filters=filters, source="rules")

    # ------------------------------------------------------------------
    def _match_dishes(self, q: str) -> List[str]:
        found: List[str] = []
        consumed = [False] * len(q)
        for name in self.dish_names:
            start = q.find(name)
            while start != -1:
                end = start + len(name)
                if not any(consumed[start:end]):
                    for i in range(start, end):
                        consumed[i] = True
                    found.append(name)
                start = q.find(name, end)
        # 按在句中出现的顺序返回，便于 compare 时保持用户原本的表述顺序
        return sorted(set(found), key=q.find)

    def _match_categories(self, q: str) -> List[str]:
        keys = []
        for key, words in CATEGORY_KEYWORDS.items():
            if key not in self.category_keys:
                continue
            if any(w in q for w in words):
                keys.append(key)
        return keys

    @staticmethod
    def _match_difficulty(q: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if any(w in q for w in EASY_WORDS):
            out["difficulty_max"] = 2
        elif any(w in q for w in HARD_WORDS):
            out["difficulty_min"] = 3
        return out

    @staticmethod
    def _intent(q: str, entities: List[str], filters: Dict[str, Any]) -> str:
        if len(entities) >= 2 and any(m in q for m in COMPARE_MARKERS):
            return "compare"
        if any(m in q for m in MULTI_HOP_MARKERS):
            return "multi_hop"
        if entities:
            return "single_hop"
        if filters:
            return "constraint"
        return "single_hop"


class LLMUnderstander:
    """LLM 解析，失败一律回落到规则结果，绝不因为一次 JSON 解析失败让整条 query 挂掉。"""

    def __init__(self, llm: ChatModel, fallback: RuleUnderstander, categories: Sequence[str]) -> None:
        self.llm = llm
        self.fallback = fallback
        self.categories = list(categories)

    def parse(self, query: str) -> ParsedQuery:
        base = self.fallback.parse(query)
        try:
            prompt = _LLM_PROMPT.format(categories=", ".join(self.categories), query=query)
            chat = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            data = _parse_json(chat.text)
            return self._merge(data, base)
        except Exception as exc:
            logger.warning("LLM 解析失败，回落规则：%s", exc)
            base.source = "llm_fallback"
            return base

    def _merge(self, data: Dict[str, Any], base: ParsedQuery) -> ParsedQuery:
        intent = data.get("intent")
        if intent not in INTENTS:
            intent = base.intent

        entities = [e for e in data.get("entities") or [] if isinstance(e, str) and e.strip()]
        if not entities:
            entities = base.entities

        # 只有词表内的菜名能进 dish_names 过滤：过滤一个库里没有的名字必然返回空集。
        # 词表外的别名照样留在 entities 里，由状态机用纯文本检索去桥接（"西红柿炒蛋" -> "番茄炒蛋"）。
        vocab = set(self.fallback.dish_names)
        in_vocab = [e for e in entities if e in vocab]

        filters: Dict[str, Any] = {}
        if in_vocab:
            filters["dish_names"] = in_vocab
        else:
            keys = [k for k in data.get("category_keys") or [] if k in self.fallback.category_keys]
            keys = keys or base.filters.get("category_keys", [])
            if keys:
                filters["category_keys"] = keys

        for key in ("difficulty_max", "difficulty_min"):
            value = data.get(key)
            if isinstance(value, int) and 1 <= value <= 5:
                filters[key] = value
            elif key in base.filters:
                filters[key] = base.filters[key]

        return ParsedQuery(intent=intent, entities=entities, filters=filters,
                           source="llm", raw=data)


def _parse_json(text: str) -> Dict[str, Any]:
    """容忍 ```json 围栏和前后废话的 JSON 提取。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("响应里没有 JSON 对象")
    return json.loads(text[start:end + 1])


def build_understander(rag, mode: str = "rules"):
    """按模式构造解析器：``rules`` 纯规则，``llm`` 规则 + LLM 覆盖。"""
    rule = RuleUnderstander(
        dish_names=[r.dish_name for r in rag.dataset.recipes],
        category_keys=[r.category_key for r in rag.dataset.recipes],
    )
    if mode == "rules":
        return rule
    llm = getattr(getattr(rag, "answerer", None), "llm", None)
    if llm is None:
        logger.warning("未配置 LLM，查询理解退回规则模式")
        return rule
    categories = sorted({r.category_key for r in rag.dataset.recipes})
    return LLMUnderstander(llm, rule, categories)
