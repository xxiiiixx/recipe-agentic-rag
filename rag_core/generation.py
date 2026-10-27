"""答案生成：上下文 -> 提示词 -> 回答，并强制带引用。

核心约束来自 RAG 的老问题：模型会补全菜谱里没写的步骤和克数。
提示词里做了三件事而不是念咒：
1. **只许用给定菜谱**，没有依据就明说"菜谱里没写"，不许按常识补；
2. **每个结论标注来源编号** [1][2]，便于后面做引用准确率（citation precision）；
3. **约束不满足时要说清**（用户要"简单素菜"，命中的却是荤菜，就要承认没找到）。

这三条同时也是评估集的判分口径：faithfulness 看有没有编，answer relevance 看有没有答，
citation precision 看引的对不对。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .context import BuiltContext, ContextConfig, build_context
from .llm import ChatModel, ChatResult
from .schema import RetrievalResult

SYSTEM_PROMPT = """你是一位家常菜谱助手。规则：
1. 只能依据「参考菜谱」回答，不得引入菜谱之外的做法、配料或克数；
2. 每个关键结论后用方括号标注来源编号，如 [1]；跨多个菜谱时逐个标注；
3. 若参考菜谱里没有相关信息，直接说明"参考菜谱中未提及"，不要猜测；
4. 若检索结果与用户的约束（口味、难度、食材、菜系）不符，先指出不满足之处，再给出最接近的答案；
5. 输出简洁的分步说明，不要复述整个菜谱。"""

USER_TEMPLATE = """参考菜谱：
{context}

用户问题：{query}

请基于上述参考菜谱回答，并标注来源编号。"""


@dataclass
class Answer:
    text: str
    context: str
    sources: List[str] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    ctx_stats: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sources": self.sources,
            "usage": self.usage,
            "latency_ms": round(self.latency_ms, 2),
            "ctx": self.ctx_stats,
        }


class RecipeAnswerer:
    def __init__(
        self,
        llm: ChatModel,
        ctx_cfg: Optional[ContextConfig] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.ctx_cfg = ctx_cfg or ContextConfig()
        self.system_prompt = system_prompt

    def build_context(self, result: RetrievalResult) -> BuiltContext:
        return build_context(result.parents, self.ctx_cfg)

    def answer(self, query: str, result: RetrievalResult) -> Answer:
        ctx = self.build_context(result)
        if not ctx.text.strip():
            return Answer(
                text="没有检索到相关菜谱，无法回答。",
                context="",
                sources=[],
                ctx_stats=ctx.stats(),
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(context=ctx.text, query=query),
            },
        ]
        chat: ChatResult = self.llm.chat(messages)
        return Answer(
            text=chat.text,
            context=ctx.text,
            sources=ctx.citations(),
            usage=chat.usage,
            latency_ms=chat.latency_ms,
            ctx_stats=ctx.stats(),
        )


CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_precision(answer_text: str, n_sources: int) -> float:
    """回答里引用的编号有多少落在真实来源范围内。

    这是最便宜的"有没有瞎引"的代理指标：编号越界说明模型在凭空造引用，
    评测集里再配上人工打分就能标定阈值。上游没有任何引用层面的检查。
    """
    cited = {int(m) for m in CITATION_RE.findall(answer_text)}
    if not cited:
        return 0.0
    valid = {c for c in cited if 1 <= c <= n_sources}
    return len(valid) / len(cited)
