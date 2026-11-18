"""工具层：把 ``RecipeRAG`` 包成带 name / description / JSON Schema 的可调用单元。

三条约束，决定了这一层为什么长这样：

1. **工具不认识 Agent**。它不知道状态机、不知道反思、不知道轮次，只接受参数、返回结构化 dict。
   以后要换编排框架（LangGraph / 自研），这一层不动；消融实验里的静态臂也能直接调同一个工具。
2. **返回结构化字段，不返回自然语言**。反思节点读的是 ``data["parents"][i]["difficulty_level"]``
   这类字段，而不是让模型"读一下结果自己判断够不够"——后者不可复现，也没法做归因。
3. **参数用 JSON Schema 描述**。既是给人看的接口文档，也能直接喂给 function calling，
   不必为"真接模型"再写一份签名。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..schema import ParentHit, Recipe, ScoredChunk, SearchFilter, recipe_as_chunk

_TIME_RE = re.compile(r"预计(?:烹饪)?时间[：:]\s*([^\n]+)")
_INGREDIENT_SECTIONS = ("原料", "食材", "配料", "用料", "材料")
_STEP_SECTIONS = ("操作", "步骤", "做法", "制作")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


def to_filter(filters: Optional[Dict[str, Any]]) -> SearchFilter:
    """dict -> SearchFilter。只认 SearchFilter 自己的字段，多余键直接报错而不是静默忽略。"""
    if not filters:
        return SearchFilter()
    known = {f for f in ("categories", "category_keys", "difficulty_min",
                         "difficulty_max", "dish_names")}
    unknown = set(filters) - known
    if unknown:
        raise ValueError(f"未知过滤字段：{sorted(unknown)}")
    return SearchFilter(**{k: v for k, v in filters.items() if v not in (None, [], "")})


def parent_dict(hit: ParentHit) -> Dict[str, Any]:
    r = hit.recipe
    return {
        "parent_id": r.parent_id,
        "dish_name": r.dish_name,
        "category": r.category,
        "category_key": r.category_key,
        "difficulty": r.difficulty,
        "difficulty_level": r.difficulty_level,
        "score": round(hit.score, 6),
        "hit_chunks": hit.hit_chunks,
        "n_chunks": r.n_chunks,
    }


def chunk_dict(sc: ScoredChunk) -> Dict[str, Any]:
    return {
        "chunk_id": sc.chunk.chunk_id,
        "parent_id": sc.chunk.parent_id,
        "dish_name": sc.chunk.dish_name,
        "section": sc.chunk.section,
        "section_path": sc.chunk.section_path,
        "rrf_score": round(sc.rrf_score, 6),
        "dense_rank": sc.dense_rank,
        "sparse_rank": sc.sparse_rank,
    }


# --------------------------------------------------------------------------- 工具
class SearchRecipesTool:
    """混合检索。Agent 唯一的高频工具。"""

    spec = ToolSpec(
        name="search_recipes",
        description="按语义 + 关键词混合检索菜谱，可附带结构化约束（类目 / 难度范围 / 指定菜名）。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索语句，可以是菜名，也可以是一句需求"},
                "top_k": {"type": "integer", "default": 5, "description": "返回父文档数量"},
                "filters": {
                    "type": "object",
                    "description": "结构化约束，字段同 SearchFilter",
                    "properties": {
                        "category_keys": {"type": "array", "items": {"type": "string"}},
                        "difficulty_min": {"type": "integer"},
                        "difficulty_max": {"type": "integer"},
                        "dish_names": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "arms": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["dense", "sparse"]},
                    "description": "只跑某一路，默认两路",
                },
            },
            "required": ["query"],
        },
    )

    def __init__(self, rag) -> None:
        self.rag = rag

    def run(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        arms: Optional[Sequence[str]] = None,
    ) -> ToolResult:
        t0 = time.perf_counter()
        try:
            flt = to_filter(filters)
            res = self.rag.retrieve(query, flt=flt, top_k=top_k, arms=arms)
            return ToolResult(
                tool=self.spec.name,
                ok=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
                data={
                    "query": query,
                    "filters": flt.describe(),
                    "candidates": res.candidates,
                    "empty_reason": res.extras.get("empty_reason", ""),
                    "parents": [parent_dict(p) for p in res.parents],
                    "top_chunks": [chunk_dict(c) for c in res.fused[:top_k]],
                },
            )
        except Exception as exc:
            return ToolResult(tool=self.spec.name, ok=False, error=str(exc),
                              latency_ms=(time.perf_counter() - t0) * 1000)


class FilterByConstraintsTool:
    """按约束枚举整库，不做向量检索。

    为什么需要它：检索是"按相关性排序取前 k"，永远回答不了"一共有多少道简单的汤"。
    用户问"有哪些……"时，正确答案来自全量枚举；而枚举能力是静态 RAG 流程里没有的。
    """

    spec = ToolSpec(
        name="filter_by_constraints",
        description="按类目 / 难度 / 菜名枚举全库满足条件的菜谱，返回总数与列表（不做相关性排序）。",
        parameters={
            "type": "object",
            "properties": {
                "filters": {"type": "object", "description": "结构化约束，字段同 SearchFilter"},
                "limit": {"type": "integer", "default": 10, "description": "最多返回多少条"},
            },
            "required": ["filters"],
        },
    )

    def __init__(self, rag) -> None:
        self.rag = rag

    def run(self, filters: Dict[str, Any], limit: int = 10) -> ToolResult:
        t0 = time.perf_counter()
        try:
            flt = to_filter(filters)
            if flt.is_empty():
                return ToolResult(tool=self.spec.name, ok=False,
                                  error="空约束会枚举整个库，请至少给一个条件")
            matched = [r for r in self.rag.dataset.recipes if flt.matches(recipe_as_chunk(r))]
            # 简单菜排前面：用户问"有哪些简单的"时，同难度内按菜名稳定排序
            matched.sort(key=lambda r: (r.difficulty_level or 99, r.dish_name))
            return ToolResult(
                tool=self.spec.name,
                ok=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
                data={
                    "filters": flt.describe(),
                    "count": len(matched),
                    "dishes": [
                        {"parent_id": r.parent_id, "dish_name": r.dish_name,
                         "category": r.category, "difficulty_level": r.difficulty_level,
                         "difficulty": r.difficulty}
                        for r in matched[:limit]
                    ],
                },
            )
        except Exception as exc:
            return ToolResult(tool=self.spec.name, ok=False, error=str(exc),
                              latency_ms=(time.perf_counter() - t0) * 1000)


class GetRecipeDetailTool:
    """按菜名或 parent_id 取整篇 / 某一节。

    检索返回的是块，块可能只命中"原料"而没命中"操作"。要给用户分步做法时，
    需要能精确取到指定小节，而不是靠向量碰运气。
    """

    spec = ToolSpec(
        name="get_recipe_detail",
        description="按菜名或 parent_id 取菜谱原文，可只取某一节（如 原料 / 操作）。",
        parameters={
            "type": "object",
            "properties": {
                "dish_name": {"type": "string", "description": "菜名，可能对应多个版本"},
                "parent_id": {"type": "string", "description": "父文档 ID，与 dish_name 二选一"},
                "section": {"type": "string", "description": "只取标题包含该词的章节"},
            },
        },
    )

    def __init__(self, rag) -> None:
        self.rag = rag

    def _resolve(self, dish_name: Optional[str], parent_id: Optional[str]) -> List[Recipe]:
        recipes = self.rag.dataset.recipes
        if parent_id:
            return [r for r in recipes if r.parent_id == parent_id]
        if dish_name:
            exact = [r for r in recipes if r.dish_name == dish_name]
            if exact:
                return exact
            return [r for r in recipes if dish_name in r.dish_name]
        return []

    def run(
        self,
        dish_name: Optional[str] = None,
        parent_id: Optional[str] = None,
        section: Optional[str] = None,
    ) -> ToolResult:
        t0 = time.perf_counter()
        found = self._resolve(dish_name, parent_id)
        if not found:
            return ToolResult(tool=self.spec.name, ok=False, error="未找到该菜谱",
                              latency_ms=(time.perf_counter() - t0) * 1000)
        items = []
        for r in found:
            text = r.text
            if section:
                text = extract_section(text, section) or ""
            items.append({
                "parent_id": r.parent_id,
                "dish_name": r.dish_name,
                "category": r.category,
                "difficulty": r.difficulty,
                "difficulty_level": r.difficulty_level,
                "section_found": bool(text) if section else True,
                "text": text,
            })
        return ToolResult(
            tool=self.spec.name,
            ok=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            data={"count": len(items), "recipes": items},
        )


class CompareRecipesTool:
    """把多篇菜谱拉成同一张表，供"哪个更简单"这类问题直接比对。

    只靠文本检索，两道菜的做法会混在两段长文里，模型很容易只描述其中一道
    （生成评测里 compare 类的失败模式）。这里先把难度、耗时、原料列成结构化行，
    再附原文，让"对比"这件事在数据层就成立，而不是指望提示词约束住模型。
    """

    spec = ToolSpec(
        name="compare_recipes",
        description="把 2 篇及以上菜谱按难度 / 耗时 / 原料 / 步骤数整理成对照表，并附原文。",
        parameters={
            "type": "object",
            "properties": {
                "dish_names": {"type": "array", "items": {"type": "string"}},
                "parent_ids": {"type": "array", "items": {"type": "string"}},
                "aspects": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["difficulty", "time", "ingredients", "steps"]},
                    "description": "要对比的维度，默认全部",
                },
            },
        },
    )

    def __init__(self, rag) -> None:
        self.rag = rag

    def run(
        self,
        dish_names: Optional[Sequence[str]] = None,
        parent_ids: Optional[Sequence[str]] = None,
        aspects: Optional[Sequence[str]] = None,
    ) -> ToolResult:
        t0 = time.perf_counter()
        by_id = {r.parent_id: r for r in self.rag.dataset.recipes}
        picked: List[Recipe] = []
        for pid in parent_ids or []:
            if pid in by_id:
                picked.append(by_id[pid])
        for name in dish_names or []:
            picked.extend(r for r in self.rag.dataset.recipes if r.dish_name == name)
        # 同名多版本时只留第一篇，避免同一道菜重复占两列
        seen, unique = set(), []
        for r in picked:
            if r.dish_name not in seen:
                seen.add(r.dish_name)
                unique.append(r)
        if len(unique) < 2:
            return ToolResult(tool=self.spec.name, ok=False,
                              error=f"需要至少 2 道不同的菜，实际 {len(unique)} 道",
                              latency_ms=(time.perf_counter() - t0) * 1000)

        aspects = list(aspects or ("difficulty", "time", "ingredients", "steps"))
        rows = [recipe_profile(r) for r in unique]
        table = render_table(rows, aspects)
        return ToolResult(
            tool=self.spec.name,
            ok=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            data={
                "aspects": aspects,
                "rows": rows,
                "table": table,
                "parent_ids": [r.parent_id for r in unique],
            },
        )


# --------------------------------------------------------------------------- 解析辅助
def extract_section(text: str, keyword: str) -> str:
    """取标题含 keyword 的那一节（到下一个同级或更高级标题为止）。"""
    lines = text.splitlines()
    out: List[str] = []
    inside = False
    for line in lines:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            if inside and level <= 2:
                break
            inside = keyword in title
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def _section_text(text: str, keywords: Iterable[str]) -> str:
    for kw in keywords:
        found = extract_section(text, kw)
        if found:
            return found
    return ""


def recipe_profile(recipe: Recipe) -> Dict[str, Any]:
    text = recipe.text
    time_match = _TIME_RE.search(text)
    ingredients = _section_text(text, _INGREDIENT_SECTIONS)
    steps = _section_text(text, _STEP_SECTIONS)
    return {
        "parent_id": recipe.parent_id,
        "dish_name": recipe.dish_name,
        "category": recipe.category,
        "difficulty": recipe.difficulty,
        "difficulty_level": recipe.difficulty_level,
        "time": time_match.group(1).strip() if time_match else "",
        "ingredients": ingredients,
        "steps": steps,
        "n_ingredient_lines": sum(1 for ln in ingredients.splitlines() if ln.strip().startswith("-")),
        "n_step_lines": sum(1 for ln in steps.splitlines() if ln.strip().startswith("-")),
        "text": text,
    }


def render_table(rows: Sequence[Dict[str, Any]], aspects: Sequence[str]) -> str:
    header = ["菜名"] + [{"difficulty": "难度", "time": "耗时",
                          "ingredients": "原料", "steps": "步骤"}.get(a, a) for a in aspects]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [r["dish_name"]]
        for a in aspects:
            if a == "difficulty":
                cells.append(f"{r['difficulty']}（{r['difficulty_level']} 级）"
                             if r["difficulty_level"] else r["difficulty"])
            elif a == "time":
                cells.append(r["time"] or "未标注")
            elif a == "ingredients":
                cells.append(f"{r['n_ingredient_lines']} 项")
            elif a == "steps":
                cells.append(f"{r['n_step_lines']} 步")
            else:
                cells.append(str(r.get(a, "")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class ToolBox:
    """把四个工具收在一起，便于按名字调用与导出 schema。"""

    def __init__(self, rag) -> None:
        self.search = SearchRecipesTool(rag)
        self.filter = FilterByConstraintsTool(rag)
        self.detail = GetRecipeDetailTool(rag)
        self.compare = CompareRecipesTool(rag)
        self._by_name = {
            t.spec.name: t for t in (self.search, self.filter, self.detail, self.compare)
        }

    def specs(self) -> List[Dict[str, Any]]:
        return [t.spec.to_dict() for t in self._by_name.values()]

    def call(self, name: str, **kwargs) -> ToolResult:
        tool = self._by_name.get(name)
        if tool is None:
            return ToolResult(tool=name, ok=False, error=f"未知工具 {name!r}")
        return tool.run(**kwargs)
