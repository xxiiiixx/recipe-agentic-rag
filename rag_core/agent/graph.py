"""Agent 状态机：understand → plan → act → reflect →（replan → act）→ answer。

**为什么手写而不是 LangGraph**：这里只有 5 个节点、一个带次数上限的环，没有多 Agent 协作、
没有需要持久化的长流程。LangGraph 的收益（checkpointer、子图、中断恢复）在这个规模换不回
它带来的抽象成本和一条新依赖。结构上仍然是显式图：``NODES`` 表 + ``_route`` 决定下一条边，
以后要换框架，把节点函数原样挂过去即可。

**为什么反思不用 LLM 判断"够了没"**：反思要回答的是一个可判定问题 ——
约束是否命中、compare 的两道菜是否都在、证据是否为空。这些都能从工具返回的结构化字段里
直接读出来。交给 LLM 判断会有三个代价：不可复现（同一 query 三遍结论不同）、贵（每轮一次调用）、
不可归因（答错了分不清是检索问题还是判断问题）。LLM 只用在该用的地方：理解查询、生成答案。

**循环必须有上限**：``max_rounds`` 默认 2。没有上限的反思环在坏查询上会一直重试，
既烧钱又掩盖"检索就是召不回来"这个真实结论。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..retrieval import select_parent_k
from ..schema import ParentHit, recipe_as_chunk
from .tools import ToolBox, ToolResult, to_filter
from .understand import ParsedQuery, build_understander

logger = logging.getLogger(__name__)

NODES = ("understand", "plan", "act", "reflect", "replan", "answer")
MAX_EVIDENCE = 20


@dataclass
class _Acc:
    """一条证据的累积分：跨多轮、多子查询的 RRF 和。"""

    recipe: Any
    score: float = 0.0
    best: float = 0.0
    hit_chunks: int = 0
    queries: List[str] = field(default_factory=list)


@dataclass
class Evidence:
    parents: Dict[str, _Acc] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, query: str, hits: Sequence[ParentHit], rrf_k: int = 60) -> None:
        """按名次累加 1/(k+rank)，等价于"跨子查询做一次 RRF"。

        不同子查询的原始分数不可比（一个查菜名、一个查条件），但名次可比。
        """
        for rank, hit in enumerate(hits, start=1):
            pid = hit.recipe.parent_id
            acc = self.parents.get(pid)
            contrib = 1.0 / (rrf_k + rank)
            if acc is None:
                self.parents[pid] = _Acc(
                    recipe=hit.recipe, score=contrib, best=hit.score,
                    hit_chunks=hit.hit_chunks, queries=[query],
                )
            else:
                acc.score += contrib
                acc.best = max(acc.best, hit.score)
                acc.hit_chunks = max(acc.hit_chunks, hit.hit_chunks)
                acc.queries.append(query)

    def ranked(self) -> List[ParentHit]:
        ordered = sorted(self.parents.values(), key=lambda a: (-a.score, a.recipe.parent_id))
        return [ParentHit(recipe=a.recipe, score=a.score, hit_chunks=a.hit_chunks)
                for a in ordered[:MAX_EVIDENCE]]

    def dishes(self) -> List[str]:
        return [p.recipe.dish_name for p in self.ranked()]

    def __len__(self) -> int:
        return len(self.parents)


@dataclass
class Reflection:
    done: bool
    reason: str
    missing_entities: List[str] = field(default_factory=list)
    unsatisfied: List[str] = field(default_factory=list)
    action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "done": self.done,
            "reason": self.reason,
            "missing_entities": self.missing_entities,
            "unsatisfied": self.unsatisfied,
            "action": self.action,
        }


@dataclass
class AgentState:
    query: str
    parsed: ParsedQuery
    plan: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)
    reflections: List[Reflection] = field(default_factory=list)
    rounds: int = 0
    stop_reason: str = ""
    answer: Any = None
    extra_context: str = ""
    latency_ms: float = 0.0
    k_used: int = 0
    # 已经补检过的实体。词表外的别名永远无法被"菜名相等"判定覆盖，
    # 不记下来就会一直重试到轮次上限，把不相关的菜塞进证据里。
    attempted: set = field(default_factory=set)


@dataclass
class AgentRun:
    state: AgentState

    @property
    def parents(self) -> List[ParentHit]:
        return self.state.evidence.ranked()

    @property
    def answer(self):
        return self.state.answer

    def to_trace(self) -> Dict[str, Any]:
        s = self.state
        return {
            "query": s.query,
            "parsed": s.parsed.to_dict(),
            "plan": s.plan,
            "rounds": s.rounds,
            "stop_reason": s.stop_reason,
            "reflections": [r.to_dict() for r in s.reflections],
            "tool_calls": s.evidence.tool_calls,
            "parents": [
                {"parent_id": p.recipe.parent_id, "dish": p.recipe.dish_name,
                 "category_key": p.recipe.category_key,
                 "difficulty_level": p.recipe.difficulty_level,
                 "score": round(p.score, 6), "hit_chunks": p.hit_chunks}
                for p in self.parents
            ],
            "answer": s.answer.to_dict() if s.answer else None,
            "latency_ms": round(s.latency_ms, 2),
        }


class RecipeAgent:
    def __init__(
        self,
        rag,
        *,
        understand: str = "rules",
        understander=None,
        max_rounds: int = 2,
        base_k: int = 5,
        per_entity_k: int = 3,
        use_compare_tool: bool = True,
        answer_enabled: bool = False,
        enum_context: bool = True,
        adaptive_k: bool = False,
        parent_ceiling: int = 2,
        k_gap: float = 0.6,
    ) -> None:
        self.rag = rag
        self.tools = ToolBox(rag)
        self.understander = understander or build_understander(rag, understand)
        self._vocab = {r.dish_name for r in rag.dataset.recipes}
        self.max_rounds = max_rounds
        self.base_k = base_k
        self.per_entity_k = per_entity_k
        self.use_compare_tool = use_compare_tool
        self.answer_enabled = answer_enabled
        self.enum_context = enum_context
        self.adaptive_k = adaptive_k
        self.parent_ceiling = parent_ceiling
        self.k_gap = k_gap

    # ------------------------------------------------------------------ 入口
    def run(self, query: str) -> AgentRun:
        t0 = time.perf_counter()
        state = AgentState(query=query, parsed=self.understander.parse(query))
        state.plan = self._plan(state)

        while True:
            self._act(state)
            reflection = self._reflect(state)
            state.reflections.append(reflection)
            if reflection.done:
                state.stop_reason = reflection.reason
                break
            if state.rounds >= self.max_rounds:
                state.stop_reason = "max_rounds"
                break
            state.rounds += 1
            state.plan = self._replan(state, reflection)

        state.answer = self._answer(state)
        state.latency_ms = (time.perf_counter() - t0) * 1000
        return AgentRun(state)

    def _entity_call(self, name: str) -> Dict[str, Any]:
        """单道菜的检索调用。

        在词表里的菜用 ``dish_names`` 过滤，把候选压到这一道菜，排序就不会被别的菜挤掉；
        不在词表里的（LLM 解析出的别名、口语说法）**不能加这个过滤** ——
        加了一定返回空集，只能靠稠密/稀疏检索去桥接同义词。
        """
        if name in self._vocab:
            return {"query": f"{name} 做法 难度 原料", "top_k": self.per_entity_k,
                    "filters": {"dish_names": [name]}}
        return {"query": name, "top_k": self.per_entity_k, "filters": {}}

    # ------------------------------------------------------------------ plan
    def _plan(self, state: AgentState) -> List[Dict[str, Any]]:
        parsed = state.parsed
        filters = parsed.filters
        calls: List[Dict[str, Any]] = []

        if parsed.intent == "compare" and len(parsed.entities) >= 2:
            # 每道菜单查一次：一次查询同时包含两个菜名时，排序由文档密度决定，
            # 弱的那道经常掉出 top-k —— 生成评测里 cmp-009/010 就是这么丢的。
            return [{"tool": "search_recipes", "args": self._entity_call(name)}
                    for name in parsed.entities]

        if parsed.intent == "multi_hop":
            # 先按完整查询带约束检索，再对每个实体做一次补充检索
            calls.append({
                "tool": "search_recipes",
                "args": {"query": state.query, "top_k": self.base_k, "filters": filters},
            })
            calls.extend({"tool": "search_recipes", "args": self._entity_call(name)}
                         for name in parsed.entities)
            return calls

        calls.append({
            "tool": "search_recipes",
            "args": {"query": state.query, "top_k": self.base_k, "filters": filters},
        })
        if parsed.intent == "constraint":
            # 纯条件型问题（"有哪些简单的汤"）用枚举补一手：检索给相关性，
            # 枚举给"一共有多少"，两者互补
            calls.append({
                "tool": "filter_by_constraints",
                "args": {"filters": filters, "limit": self.base_k},
            })
        return calls

    # ------------------------------------------------------------------ act
    def _act(self, state: AgentState) -> None:
        for call in state.plan:
            result = self.tools.call(call["tool"], **call["args"])
            record = {
                "round": state.rounds,
                "tool": result.tool,
                "args": call["args"],
                "ok": result.ok,
                "latency_ms": round(result.latency_ms, 2),
            }
            if not result.ok:
                record["error"] = result.error
                state.evidence.tool_calls.append(record)
                continue

            if result.tool == "search_recipes":
                record["n_parents"] = len(result.data["parents"])
                record["dishes"] = [p["dish_name"] for p in result.data["parents"]]
                state.evidence.add(call["args"]["query"], self._to_hits(result))
            elif result.tool == "filter_by_constraints":
                record["count"] = result.data["count"]
                record["dishes"] = [d["dish_name"] for d in result.data["dishes"]]
                # 枚举结果**不进证据池**：它没有相关性分数，混进按相关性排的池子只会
                # 把更相关的结果挤下去。是否把它作为答案层的补充上下文，由 enum_context 决定。
                if self.enum_context:
                    names = [d["dish_name"] for d in result.data["dishes"]]
                    state.extra_context = (
                        f"【全库枚举】满足条件的菜共 {result.data['count']} 道，"
                        f"其中前 {len(names)} 道：{'、'.join(names)}"
                    )
            elif result.tool == "compare_recipes":
                record["table"] = result.data["table"]
                state.extra_context = result.data["table"]
            state.evidence.tool_calls.append(record)

        # compare 的对照表在证据齐了之后再生成，此时才知道两篇原文都在
        if (self.use_compare_tool and state.parsed.intent == "compare"
                and not state.extra_context and len(state.evidence) >= 2):
            ids = [p.recipe.parent_id for p in state.evidence.ranked()[:3]]
            result = self.tools.call("compare_recipes", parent_ids=ids)
            state.evidence.tool_calls.append({
                "round": state.rounds, "tool": result.tool, "args": {"parent_ids": ids},
                "ok": result.ok, "latency_ms": round(result.latency_ms, 2),
            })
            if result.ok:
                state.extra_context = result.data["table"]

    def _to_hits(self, result: ToolResult) -> List[ParentHit]:
        by_id = {r.parent_id: r for r in self.rag.dataset.recipes}
        hits = []
        for p in result.data["parents"]:
            recipe = by_id.get(p["parent_id"])
            if recipe is not None:
                hits.append(ParentHit(recipe=recipe, score=p["score"],
                                      hit_chunks=p["hit_chunks"]))
        return hits

    def _covered(self, entity: str, ranked: Sequence[ParentHit]) -> bool:
        """实体是否已被证据覆盖。

        用包含关系而不是严格相等：语料里同一道菜常有"红烧肉（电饭煲版）"这类变体，
        严格相等会把已经召回的菜判成缺失，白白多跑一轮。
        """
        return any(
            entity == p.recipe.dish_name
            or entity in p.recipe.dish_name
            or p.recipe.dish_name in entity
            for p in ranked
        )

    # ------------------------------------------------------------------ reflect
    def _reflect(self, state: AgentState) -> Reflection:
        ranked = state.evidence.ranked()
        filters = state.parsed.filters

        missing = [
            e for e in state.parsed.entities
            if e not in state.attempted and not self._covered(e, ranked)
        ]
        unsatisfied: List[str] = []
        if filters.get("category_keys"):
            if not any(p.recipe.category_key in filters["category_keys"] for p in ranked):
                unsatisfied.append("category_keys")
        if filters.get("difficulty_max") is not None:
            cap = filters["difficulty_max"]
            if not any(0 < p.recipe.difficulty_level <= cap for p in ranked):
                unsatisfied.append("difficulty_max")
        if filters.get("difficulty_min") is not None:
            floor = filters["difficulty_min"]
            if not any(p.recipe.difficulty_level >= floor for p in ranked):
                unsatisfied.append("difficulty_min")

        if not ranked:
            # 一条都没召回时，要先分清是"这道菜召不回"还是"约束把菜挡在门外"。
            # 判断依据不是猜：菜在库里存在、却没有任何一个版本满足当前约束，
            # 那就是约束的锅，放宽它；菜本身不在库里，才是实体召回的锅。
            blocked = [e for e in missing if self._blocked_by_filter(e, filters)]
            if missing and len(blocked) == len(missing):
                return Reflection(done=False, reason="constraint_unmet",
                                  missing_entities=missing,
                                  unsatisfied=_relaxable(filters), action="relax")
            if missing:
                return Reflection(done=False, reason="no_evidence",
                                  missing_entities=missing, action="search_missing")
            return Reflection(done=False, reason="no_evidence",
                              unsatisfied=unsatisfied or _relaxable(filters),
                              action="relax")

        if missing:
            return Reflection(done=False, reason="missing_entity",
                              missing_entities=missing, action="search_missing")
        if unsatisfied:
            return Reflection(done=False, reason="constraint_unmet",
                              unsatisfied=unsatisfied, action="relax")
        if state.parsed.intent == "compare" and len(set(state.evidence.dishes())) < 2:
            # 缺的那道已经补检过还是没回来，再循环也只是重复同一次失败。
            # 收手，让答案层如实说"只找到一道"，而不是继续烧检索。
            return Reflection(done=True, reason="compare_partial")
        return Reflection(done=True, reason="evidence_ok")

    def _blocked_by_filter(self, dish_name: str, filters: Dict[str, Any]) -> bool:
        """菜在库里存在，但没有任何版本满足当前约束。"""
        variants = [r for r in self.rag.dataset.recipes if r.dish_name == dish_name]
        if not variants:
            return False
        flt = to_filter(filters)
        return not any(flt.matches(recipe_as_chunk(r)) for r in variants)

    # ------------------------------------------------------------------ replan
    def _replan(self, state: AgentState, reflection: Reflection) -> List[Dict[str, Any]]:
        if reflection.action == "search_missing" and reflection.missing_entities:
            # 补检时**去掉 dish_names 过滤**：上一轮已经证明"带着这个过滤查不到"，
            # 原样重试只会再空一次。改用菜名做纯文本检索，靠稠密/稀疏去桥接别名。
            state.attempted.update(reflection.missing_entities)
            return [
                {"tool": "search_recipes",
                 "args": {"query": name, "top_k": self.per_entity_k, "filters": {}}}
                for name in reflection.missing_entities
            ]

        relaxed = dict(state.parsed.filters)
        for key in reflection.unsatisfied:
            relaxed.pop(key, None)
        if reflection.action == "relax" and relaxed != state.parsed.filters:
            # 约束一个都没命中，最可能是解析过严（把"简单"当成了硬上限）。
            # 放宽后必须写回状态：否则下一轮反思仍拿旧约束判定，永远收敛不了。
            state.parsed.filters = relaxed
            return [{"tool": "search_recipes",
                     "args": {"query": state.query, "top_k": self.base_k, "filters": relaxed}}]

        # 兜底：扩召回深度。同一条 query 多要几倍候选，成本低且不会引入新错误
        widened = max(self.base_k * 2, self.base_k + 5)
        return [{"tool": "search_recipes",
                 "args": {"query": state.query, "top_k": widened,
                          "filters": state.parsed.filters}}]

    # ------------------------------------------------------------------ answer
    def _answer(self, state: AgentState):
        if not self.answer_enabled:
            return None
        answerer = getattr(self.rag, "answerer", None)
        if answerer is None:
            return None
        ranked = state.evidence.ranked()
        if not ranked:
            return None
        k = len(ranked) if not self.adaptive_k else select_parent_k(
            ranked, floor=self._k_floor(state.parsed.intent),
            ceiling=self.parent_ceiling, gap=self.k_gap,
        )
        state.k_used = k
        return answerer.answer(state.query, _RankedResult(ranked[:k]),
                               extra_context=state.extra_context)

    @staticmethod
    def _k_floor(intent: str) -> int:
        # 比较与多跳至少要两篇才有得比；条件型问题的 gold 通常有 4 道，别只给一篇
        return 2 if intent in {"compare", "multi_hop", "constraint"} else 1


def _relaxable(filters: Dict[str, Any]) -> List[str]:
    """可以安全放宽的约束键。dish_names 不放宽 —— 用户点名要的菜不能换掉。"""
    return [k for k in ("category_keys", "difficulty_max", "difficulty_min") if k in filters]


class _RankedResult:
    """把已排好序的证据伪装成 RetrievalResult，复用 ``RecipeAnswerer.answer``。

    只实现 ``parents`` 一个属性，是为了让答案层完全不知道证据来自几轮检索。
    """

    def __init__(self, parents: Sequence[ParentHit]) -> None:
        self.parents = list(parents)
        self.query = ""
        self.fused = []

