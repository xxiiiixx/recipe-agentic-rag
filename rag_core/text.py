"""中文分词与 token 预算估算。

上游 C8 的混合检索用的是 langchain ``BM25Retriever.from_documents(chunks, k=5)``，
它的分词函数是默认的 ``default_preprocessing_func = lambda t: t.split()`` —— **按空格切**。
中文菜谱里几乎没有空格，于是"红烧肉怎么做"被当成一个整体 token，
BM25 只在这些超长 token 完全相等时才算命中，稀疏分支等于失效。
这是本项目第一个可量化、可复现的改进点：把分词换成字 bigram / jieba，BM25 单独就能打。

因此这里把分词做成**可切换策略**，其中 ``whitespace`` 保留为"上游行为"，
专门用来在评测里复现旧结果、证明改进来自分词而不是别的变量。
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List

_CJK = r"\u4e00-\u9fff"
# 去掉标点、空白、Markdown 符号，只留汉字与字母数字
_NOISE = re.compile(rf"[^{_CJK}A-Za-z0-9]+")
_CJK_RE = re.compile(rf"[{_CJK}]")

# 极小的停用词表：只保留高频虚词，避免"的/了/怎么"这类词稀释 BM25
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "也", "都", "就", "而", "被",
    "把", "对", "为", "以", "于", "之", "这", "那", "有", "个", "我", "你", "他",
    "怎么", "如何", "什么", "哪些", "多少", "可以", "需要", "一下", "请问",
    "a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "how", "what",
}


def _normalize(text: str) -> str:
    return _NOISE.sub(" ", text.lower())


def tokenize_char_bigram(text: str) -> List[str]:
    """字级 unigram + bigram。

    不引入词典依赖，对未登录菜名（"藜蒿炒腊肉"）也稳；
    bigram 承担"词组"语义，unigram 兜住单字查询的召回。
    这也是 Lucene CJKAnalyzer 的常见做法。
    """
    out: List[str] = []
    for seg in _normalize(text).split():
        if not seg:
            continue
        if _CJK_RE.search(seg):
            out.extend(seg)  # unigram
            out.extend(seg[i : i + 2] for i in range(len(seg) - 1))  # bigram
        else:
            out.append(seg)
    return [t for t in out if t not in _STOPWORDS]


def tokenize_jieba(text: str) -> List[str]:
    """词级分词。需要 ``pip install jieba``；未安装时抛 ImportError 由上层决定降级。"""
    import jieba  # 延迟导入：不装 jieba 的机器仍然能跑 char_bigram

    out: List[str] = []
    for seg in _normalize(text).split():
        if not seg:
            continue
        if _CJK_RE.search(seg):
            out.extend(w for w in jieba.cut(seg) if w.strip())
        else:
            out.append(seg)
    return [t for t in out if t not in _STOPWORDS]


def tokenize_whitespace(text: str) -> List[str]:
    """上游 C8 的实际行为：``text.split()``。**仅用于消融复现，不要在生产用。**"""
    return text.split()


TOKENIZERS: Dict[str, Callable[[str], List[str]]] = {
    "char_bigram": tokenize_char_bigram,
    "jieba": tokenize_jieba,
    "whitespace": tokenize_whitespace,
}


def get_tokenizer(name: str) -> Callable[[str], List[str]]:
    if name not in TOKENIZERS:
        raise ValueError(f"未知分词器 {name!r}，可选：{sorted(TOKENIZERS)}")
    return TOKENIZERS[name]


def estimate_tokens(text: str) -> int:
    """粗略的 token 数估算（避免为一个预算判断拖入 tiktoken 依赖）。

    中文 BPE 大致一字一 token，英文约 4 字符一 token。
    用于上下文预算控制这个量级的判断足够；如需精确值，把这里替换成 tiktoken。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(cjk + other / 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按 token 预算截断，尽量断在换行/句号处，避免把一句话切成两半。

    上游 C8 的 ``_build_context`` 是"整篇父文档累加，超预算就 break"，
    于是第 4、5 篇文档被整体丢弃，而不是把第 3 篇截一半 ——
    检索层辛苦排出的相关性顺序在生成层被抹平。这里改成"先在文档内截断，再换下一篇"。

    实现用二分找"最长的、估算不超过预算的前缀"，再回退到最近的语义边界。
    比"按段落逐块累加"好的地方：段落粒度往往很粗（一个 `## 操作` 就有 200 字），
    累加到某个段落装不下就整段丢，仍会浪费 30%~50% 的预算。
    """
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1

    cut = text[:lo]
    for boundary in ("\n\n", "\n", "。", "，", " "):
        idx = cut.rfind(boundary)
        # 只接受"回退损失不超过 35%"的边界，否则宁可硬截也不要丢掉大段内容
        if idx >= len(cut) * 0.65:
            return cut[: idx + len(boundary)]
    return cut

