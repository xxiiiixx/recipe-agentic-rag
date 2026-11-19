"""Agent 层的契约测试：工具边界、查询解析、反思环的终止条件。

全部用假检索器，不装 embedding、不联网 —— 这些测试要跑在每次改动的 CI 里，
不能因为"今天没 GPU"就跳过。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_core.agent import RecipeAgent, RuleUnderstander, ToolBox
from rag_core.agent.tools import extract_section, to_filter
from rag_core.retrieval import rerank_by_difficulty
from rag_core.schema import ParentHit, Recipe, SearchFilter, recipe_as_chunk


def make_recipe(pid: str, dish: str, category_key: str = "meat_dish",
                level: int = 2, text: str | None = None) -> Recipe:
    body = text or (
        f"# {dish}\n\n"
        f"预计烹饪时间：20 分钟\n\n"
        f"## 必备原料和工具\n\n- 主料\n- 辅料\n\n"
        f"## 操作\n\n- 第一步\n- 第二步\n"
    )
    return Recipe(
        parent_id=pid, source_rel=f"{dish}.md", dish_name=dish,
        category="素菜" if category_key == "vegetable_dish" else "荤菜",
        category_key=category_key, difficulty="简单" if level == 2 else "困难",
        difficulty_level=level, text=body, n_chunks=3,
    )


class FakeRAG:
    """最小检索器：按菜名命中，可指定"永远召不回"的菜来触发反思。"""

    def __init__(self, recipes, *, drop=(), answerer=None) -> None:
        self.dataset = SimpleNamespace(recipes=list(recipes))
        self.answerer = answerer
        self.drop = set(drop)
        self.calls: list[dict] = []

    def retrieve(self, query, flt=None, top_k=None, arms=None):
        f = flt or to_filter(None)
        self.calls.append({"query": query, "filter": f, "filters": f.describe()})
        pool = [r for r in self.dataset.recipes if f.matches(recipe_as_chunk(r))]
        if f.dish_names:
            hits = [r for r in pool if r.dish_name in f.dish_names]
        else:
            hits = [r for r in pool if r.dish_name in query]
        hits = [r for r in hits if r.dish_name not in self.drop]
        hits = hits[: (top_k or 5)]
        parents = [ParentHit(recipe=r, score=1.0 / (i + 1)) for i, r in enumerate(hits)]
        return SimpleNamespace(parents=parents, fused=[], candidates=len(pool),
                               extras={}, latency_ms=0.0)


@pytest.fixture
def rag() -> FakeRAG:
    return FakeRAG([
        make_recipe("p1", "红烧肉"),
        make_recipe("p2", "番茄炒蛋", "vegetable_dish", 1),
        make_recipe("p3", "冬瓜排骨汤", "soup", 2),
        make_recipe("p4", "紫菜蛋花汤", "soup", 1),
    ])


# --------------------------------------------------------------------- 工具层
def test_to_filter_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        to_filter({"categry_keys": ["soup"]})


def test_search_tool_returns_structured_parents(rag: FakeRAG) -> None:
    box = ToolBox(rag)
    res = box.call("search_recipes", query="红烧肉怎么做", top_k=3)
    assert res.ok
    assert res.data["parents"][0]["dish_name"] == "红烧肉"
    assert "difficulty_level" in res.data["parents"][0]


def test_filter_by_constraints_enumerates_whole_corpus(rag: FakeRAG) -> None:
    box = ToolBox(rag)
    res = box.call("filter_by_constraints", filters={"category_keys": ["soup"]})
    assert res.ok
    assert res.data["count"] == 2
    assert {d["dish_name"] for d in res.data["dishes"]} == {"冬瓜排骨汤", "紫菜蛋花汤"}


def test_filter_by_constraints_refuses_empty_filter(rag: FakeRAG) -> None:
    res = ToolBox(rag).call("filter_by_constraints", filters={})
    assert not res.ok


def test_get_recipe_detail_returns_one_section(rag: FakeRAG) -> None:
    res = ToolBox(rag).call("get_recipe_detail", dish_name="红烧肉", section="操作")
    assert res.ok
    text = res.data["recipes"][0]["text"]
    assert "第一步" in text
    assert "必备原料" not in text


def test_compare_requires_two_distinct_dishes(rag: FakeRAG) -> None:
    box = ToolBox(rag)
    assert not box.call("compare_recipes", dish_names=["红烧肉"]).ok
    res = box.call("compare_recipes", dish_names=["红烧肉", "番茄炒蛋"])
    assert res.ok
    assert "| 菜名 |" in res.data["table"]
    assert "难度" in res.data["table"]


def test_extract_section_stops_at_next_heading() -> None:
    text = "# 菜\n## 原料\n- a\n## 操作\n- b\n"
    assert "a" in extract_section(text, "原料")
    assert "b" not in extract_section(text, "原料")


# --------------------------------------------------------------------- 查询理解
def test_rule_understander_extracts_entity_and_difficulty(rag: FakeRAG) -> None:
    u = RuleUnderstander([r.dish_name for r in rag.dataset.recipes],
                         [r.category_key for r in rag.dataset.recipes])
    parsed = u.parse("番茄炒蛋难不难，我是新手")
    assert parsed.entities == ["番茄炒蛋"]
    assert parsed.filters["difficulty_max"] == 2
    assert parsed.intent == "single_hop"


def test_rule_understander_marks_compare(rag: FakeRAG) -> None:
    u = RuleUnderstander([r.dish_name for r in rag.dataset.recipes],
                         [r.category_key for r in rag.dataset.recipes])
    parsed = u.parse("红烧肉和番茄炒蛋哪个更简单")
    assert parsed.intent == "compare"
    assert set(parsed.entities) == {"红烧肉", "番茄炒蛋"}


def test_rule_understander_uses_category_when_no_dish(rag: FakeRAG) -> None:
    u = RuleUnderstander([r.dish_name for r in rag.dataset.recipes],
                         [r.category_key for r in rag.dataset.recipes])
    parsed = u.parse("有没有简单的汤")
    assert parsed.intent == "constraint"
    assert parsed.filters["category_keys"] == ["soup"]
    assert parsed.filters["difficulty_max"] == 2


# --------------------------------------------------------------------- 状态机
def test_compare_issues_one_search_per_dish(rag: FakeRAG) -> None:
    run = RecipeAgent(rag, max_rounds=0).run("红烧肉和番茄炒蛋哪个更简单")
    assert len(rag.calls) == 2
    assert all(c["filter"].dish_names for c in rag.calls)
    assert set(run.state.evidence.dishes()) == {"红烧肉", "番茄炒蛋"}
    assert run.state.stop_reason == "evidence_ok"


def test_reflection_retries_missing_entity_once_then_accepts(rag: FakeRAG) -> None:
    rag.drop = {"番茄炒蛋"}
    run = RecipeAgent(rag, max_rounds=2).run("红烧肉和番茄炒蛋哪个更简单")
    assert run.state.reflections[0].reason == "missing_entity"
    # 补检过一次就不再纠缠：词表外的别名或库里没有的菜，重试也不会收敛
    assert run.state.rounds == 1
    assert run.state.stop_reason == "compare_partial"


def test_reflection_relaxes_unmet_difficulty(rag: FakeRAG) -> None:
    # 只有 level=3 的菜：difficulty_max=2 会把唯一相关的菜挡在候选集外，
    # 反思应识别出"约束过严"并放宽重试，而不是直接返回空
    rag.dataset.recipes = [make_recipe("p9", "红烧肉", "meat_dish", 3)]
    run = RecipeAgent(rag, max_rounds=1).run("红烧肉简单吗")
    assert run.state.reflections[0].action == "relax"
    assert run.state.reflections[0].reason == "constraint_unmet"
    assert rag.calls[-1]["filter"].difficulty_max is None
    assert run.state.stop_reason == "evidence_ok"


def test_max_rounds_bounds_the_loop(rag: FakeRAG) -> None:
    rag.drop = {"红烧肉", "番茄炒蛋"}
    run = RecipeAgent(rag, max_rounds=2).run("红烧肉和番茄炒蛋哪个更简单")
    assert run.state.rounds <= 2
    assert len(run.state.reflections) == run.state.rounds + 1


def test_plan_only_arm_has_no_reflection_rounds(rag: FakeRAG) -> None:
    rag.drop = {"番茄炒蛋"}
    run = RecipeAgent(rag, max_rounds=0).run("红烧肉和番茄炒蛋哪个更简单")
    assert run.state.rounds == 0
    assert run.state.stop_reason in {"max_rounds", "evidence_ok"}


def test_enum_result_goes_to_context_not_evidence(rag: FakeRAG) -> None:
    run = RecipeAgent(rag, max_rounds=0, enum_context=True).run("有没有简单的汤")
    assert "全库枚举" in run.state.extra_context
    # 枚举结果不进证据池：它没有相关性信号
    assert all("enum:" not in q for acc in run.state.evidence.parents.values() for q in acc.queries)


def test_enum_context_can_be_disabled(rag: FakeRAG) -> None:
    run = RecipeAgent(rag, max_rounds=0, enum_context=False).run("有没有简单的汤")
    assert run.state.extra_context == ""


def test_trace_is_json_serialisable(rag: FakeRAG) -> None:
    import json

    run = RecipeAgent(rag, max_rounds=1).run("红烧肉和番茄炒蛋哪个更简单")
    payload = json.dumps(run.to_trace(), ensure_ascii=False)
    assert "tool_calls" in payload


# --------------------------------------------------------------------- 难度重排
def _hits(*levels: int) -> list[ParentHit]:
    return [ParentHit(recipe=make_recipe(f"p{i}", f"菜{i}", level=lvl), score=1.0 / (i + 1))
            for i, lvl in enumerate(levels)]


def test_rerank_puts_simplest_first() -> None:
    hits = _hits(3, 1, 2)
    out = rerank_by_difficulty(hits, SearchFilter(difficulty_max=2))
    assert [h.recipe.difficulty_level for h in out] == [1, 2, 3]


def test_rerank_keeps_unknown_difficulty_last() -> None:
    hits = _hits(0, 2, 1)
    out = rerank_by_difficulty(hits, SearchFilter(difficulty_max=3))
    assert [h.recipe.difficulty_level for h in out] == [1, 2, 0]


def test_rerank_reverses_for_hard_queries() -> None:
    hits = _hits(1, 3, 2)
    out = rerank_by_difficulty(hits, SearchFilter(difficulty_min=3))
    assert [h.recipe.difficulty_level for h in out] == [3, 2, 1]


def test_rerank_is_noop_without_difficulty_preference() -> None:
    hits = _hits(3, 1, 2)
    out = rerank_by_difficulty(hits, SearchFilter(category_keys=["soup"]))
    assert [h.recipe.difficulty_level for h in out] == [3, 1, 2]
