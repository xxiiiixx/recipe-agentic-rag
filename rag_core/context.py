"""上下文拼装：按 token 预算装父文档，超预算在**文档内**截断。

上游 ``generation_integration.py::_build_context(documents, max_length=2000)`` 的做法是：

    for doc in documents:
        if total + len(doc.page_content) > max_length:
            break          # <- 剩下几篇整体丢弃
        parts.append(doc.page_content)

两个问题：
1. **按字符数不是按 token 数**。中文 2000 字约 1300~2000 token，英文 2000 字符约 500 token，
   同一份预算在不同语料上差 3 倍，模型端要么爆要么浪费。
2. **break 而不是 truncate**。检索层排出 1、2、3 的顺序，第 3 篇刚好超预算就被整篇扔掉，
   第 4、5 篇也不进来了 —— 父子块"召回到小段"的精度收益在这里被抹平。

本实现：逐篇装，剩余预算不足就在该篇内按段落/行截断（``truncate_to_tokens``），
再换下一篇；只有剩余预算连一个最小片段都放不下时才停，并把丢弃数量记进 trace，
badcase 归因时"是检索没召回，还是被上下文预算挤掉了"一眼可分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .schema import ParentHit
from .text import estimate_tokens, truncate_to_tokens


@dataclass
class ContextConfig:
    token_budget: int = 1800
    per_doc_budget: int = 900
    min_doc_tokens: int = 80  # 剩余预算低于这个值就不再塞新文档
    separator: str = "\n\n---\n\n"
    with_header: bool = True


@dataclass
class BuiltContext:
    text: str
    used: List[ParentHit] = field(default_factory=list)
    truncated_docs: int = 0
    dropped_docs: int = 0
    tokens: int = 0

    def citations(self) -> List[str]:
        return [f"[{i + 1}] {hit.recipe.dish_name}" for i, hit in enumerate(self.used)]

    def stats(self) -> dict:
        return {
            "used_docs": len(self.used),
            "truncated_docs": self.truncated_docs,
            "dropped_docs": self.dropped_docs,
            "context_tokens": self.tokens,
        }


def _header(idx: int, hit: ParentHit) -> str:
    r = hit.recipe
    parts = [f"[{idx}] {r.dish_name}"]
    if r.category:
        parts.append(f"分类：{r.category}")
    if r.difficulty_level:
        parts.append(f"难度：{r.difficulty}")
    return "｜".join(parts)


def build_context(parents: Sequence[ParentHit], cfg: ContextConfig | None = None) -> BuiltContext:
    cfg = cfg or ContextConfig()
    blocks: List[str] = []
    used: List[ParentHit] = []
    truncated = 0
    remaining = cfg.token_budget

    for hit in parents:
        if remaining < cfg.min_doc_tokens:
            break
        budget = min(cfg.per_doc_budget, remaining)
        body = hit.recipe.text
        body_tokens = estimate_tokens(body)
        header = _header(len(used) + 1, hit) if cfg.with_header else ""
        header_tokens = estimate_tokens(header) + 1  # 换行

        if body_tokens > budget - header_tokens:
            body = truncate_to_tokens(body, max(budget - header_tokens, 1))
            truncated += 1
        block = f"{header}\n{body.strip()}"
        blocks.append(block)
        used.append(hit)
        remaining -= estimate_tokens(block)

    text = cfg.separator.join(blocks)
    return BuiltContext(
        text=text,
        used=used,
        truncated_docs=truncated,
        dropped_docs=max(len(parents) - len(used), 0),
        tokens=estimate_tokens(text),
    )
