"""BM25 稀疏检索（纯 Python 实现，带倒排表）。

为什么不直接用 ``rank_bm25``：
1. 需要**分词可切换**（char_bigram / jieba / 上游的 whitespace），现成库把分词写死成它的默认行为；
2. 需要**在候选集合内检索**（过滤条件先求交集，再在子集里打分），现成库不支持子集检索；
3. 300 篇菜谱量级下，50 行 BM25 的分数是可解释、可手算验证的，出问题能定位。

算法：Okapi BM25
    score(q, d) = Σ_{t∈q} idf(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 - b + b·|d|/avgdl) )
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
k1 控制词频饱和，b 控制长度归一。两个字都是**可调参的消融点**：中文短查询多，
b 偏大反而会惩罚"操作"这种本来就长的必需章节。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .schema import Chunk
from .text import get_tokenizer


class BM25Index:
    def __init__(
        self,
        tokenizer_name: str = "char_bigram",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.k1 = k1
        self.b = b
        self._tokenize = get_tokenizer(tokenizer_name)

        self.chunk_ids: List[str] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}
        # term -> [(doc_index, term_frequency)]
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    # ------------------------------------------------------------------
    def fit(self, chunks: Sequence[Chunk]) -> "BM25Index":
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._doc_len = []
        self._postings = defaultdict(list)
        df: Counter = Counter()

        for idx, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk.text)
            self._doc_len.append(max(len(tokens), 1))
            tf = Counter(tokens)
            for term, freq in tf.items():
                self._postings[term].append((idx, freq))
                df[term] += 1

        n_docs = len(chunks)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 0.0
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return self

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """返回 [(chunk_id, score)]，按分数降序。

        ``allowed_ids`` 不为空时只在这个子集内打分 —— 这就是"过滤下推"在稀疏分支的实现：
        先做集合求交，再算分数，而不是取 top-N 之后再把不合格的删掉。
        """
        if not self.chunk_ids or top_k <= 0:
            return []

        if allowed_ids is not None and not allowed_ids:
            return []  # 过滤后没有候选：直接空结果，不发散也不"凑数"

        terms = self._tokenize(query)
        if not terms:
            return []

        scores: Dict[int, float] = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_idx, freq in postings:
                if allowed_ids is not None and self.chunk_ids[doc_idx] not in allowed_ids:
                    continue
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * self._doc_len[doc_idx] / self._avgdl
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * freq * (self.k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.chunk_ids[idx], score) for idx, score in ranked]

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, object]:
        return {
            "docs": len(self.chunk_ids),
            "vocab": len(self._idf),
            "avgdl": round(self._avgdl, 2),
            "tokenizer": self.tokenizer_name,
            "k1": self.k1,
            "b": self.b,
        }
