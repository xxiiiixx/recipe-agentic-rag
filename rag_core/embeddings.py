"""向量化后端。

两个实现，用途不同，别混用：

* ``sentence_transformers``：本地跑 bge-small-zh-v1.5（512 维，CPU 上 300 篇语料几十秒）。
  这是**评测与消融的默认后端**，因为它确定、免费、可无限次重跑。
* ``openai_compatible``：走 ``/v1/embeddings`` 接口（SiliconFlow、Moonshot、通义、自建 TEI 都行）。
  服务器无 GPU 时用它，代价是每次消融都要花 token 钱，且供应商换了分数不可比。
* ``hashing``：**仅供冒烟**。不装任何模型也能把整条链路跑通，但不要用它报指标。
  有它才好在 CI 里验证"索引 -> 检索 -> 拼接 -> 报错路径"这些不该依赖模型的逻辑。
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """返回 (n, dim) 的 float32 矩阵，行已 L2 归一化（内积即余弦）。"""


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self._model = SentenceTransformer(model_name)
        self._batch = batch_size
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class OpenAICompatEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int) -> None:
        from openai import OpenAI  # 延迟导入

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self.name = f"openai-compatible:{model}"
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        mat = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        return _l2_normalize(mat)


class HashingEmbedder:
    """字符 n-gram 哈希向量。确定、零依赖，只用来打通链路。"""

    def __init__(self, dim: int = 512, ngram: int = 2) -> None:
        self.dim = dim
        self._ngram = ngram
        self.name = f"hashing-{dim}d-{ngram}gram"

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = text.replace("\n", " ")
        for i in range(len(text) - self._ngram + 1):
            gram = text[i : i + self._ngram]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            # 用哈希值同时决定维度与符号，降低碰撞带来的系统性偏置
            vec[h % self.dim] += 1.0 if (h >> 31) & 1 == 0 else -1.0
        return vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        return _l2_normalize(mat)


def build_embedder(provider: str, **kwargs) -> Embedder:
    provider = provider.lower()
    if provider in {"sentence_transformers", "st", "local"}:
        return SentenceTransformerEmbedder(
            model_name=kwargs.get("model", "BAAI/bge-small-zh-v1.5"),
            batch_size=kwargs.get("batch_size", 32),
        )
    if provider in {"openai_compatible", "openai", "api"}:
        return OpenAICompatEmbedder(
            base_url=kwargs["base_url"],
            api_key=kwargs["api_key"],
            model=kwargs["model"],
            dim=int(kwargs["dim"]),
        )
    if provider in {"hashing", "debug"}:
        return HashingEmbedder(dim=int(kwargs.get("dim", 512)))
    raise ValueError(f"未知 embedding provider：{provider}")
