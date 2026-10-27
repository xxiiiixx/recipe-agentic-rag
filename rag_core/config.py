"""配置：一份 YAML 描述"跑哪一臂"，代码里不出现魔数。

上游把 ``top_k=3``、``temperature=0.1``、``data_path="../../data/C8/cook"`` 散落在
``config.py`` 与各模块的默认参数里，做消融时要改代码。这里把每次实验的全部变量收进一个
YAML，配置文件本身就可以作为实验记录（``configs/ablation/*.yaml`` 与报告一一对应）。

支持 ``${ENV_VAR}`` 展开，密钥不进仓库。
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            return os.environ.get(m.group(1), m.group(2) or "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _fill(obj: Any, data: Dict[str, Any]) -> Any:
    """把 dict 填进 dataclass，未知键直接报错（拼错的配置项必须立刻暴露）。"""
    known = {f.name: f for f in fields(obj)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{type(obj).__name__} 不认识的配置项：{sorted(unknown)}")
    for key, value in data.items():
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _fill(current, value)
        elif isinstance(value, list) and key == "arms":
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
    return obj


@dataclass
class DatasetCfg:
    root: str = "data/C8/cook"
    max_level: int = 3
    drop_comments: bool = True
    min_chunk_chars: int = 8


@dataclass
class EmbeddingCfg:
    provider: str = "sentence_transformers"  # sentence_transformers | openai_compatible | hashing
    model: str = "BAAI/bge-small-zh-v1.5"
    dim: int = 512
    batch_size: int = 32
    base_url: str = ""
    api_key: str = "${EMBEDDING_API_KEY}"


@dataclass
class StoreCfg:
    backend: str = "milvus"  # milvus | local
    uri: str = "http://localhost:19530"
    token: str = "${MILVUS_TOKEN}"
    collection: str = "recipe_chunks"
    metric: str = "COSINE"
    dim: int = 0  # 0 表示跟随 embedding.dim；与已有 collection 维度不一致会自动重建
    recreate: bool = False
    local_dir: str = "artifacts/local_index"


@dataclass
class RetrievalCfg:
    top_k: int = 5
    dense_k: int = 10
    sparse_k: int = 10
    rrf_k: int = 60
    filter_mode: str = "pre"
    parent_strategy: str = "rrf_sum"
    parent_top_n: int = 2
    arms: tuple = ("dense", "sparse")
    tokenizer: str = "char_bigram"  # char_bigram | jieba | whitespace(上游行为)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    embed_query_prefix: str = ""  # bge-zh 官方建议检索侧加查询指令，可留空做对照


@dataclass
class ContextCfg:
    token_budget: int = 1800
    per_doc_budget: int = 900
    min_doc_tokens: int = 80


@dataclass
class LLMCfg:
    model: str = "moonshot-v1-8k"
    base_url: str = "https://api.moonshot.cn/v1"
    api_key: str = "${LLM_API_KEY}"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 2


@dataclass
class AppConfig:
    name: str = "base"
    workspace: str = "artifacts"
    seed: int = 42
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    store: StoreCfg = field(default_factory=StoreCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    context: ContextCfg = field(default_factory=ContextCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)

    # ---------------- io ----------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        import yaml  # 只有读配置需要它

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw = _expand_env(raw)
        cfg = cls(name=raw.pop("name", Path(path).stem), workspace=raw.pop("workspace", "artifacts"),
                  seed=raw.pop("seed", 42))
        return _fill(cfg, raw)

    def artifacts_dir(self) -> Path:
        if is_dataclass(self.store) and self.store.backend == "local":
            return Path(self.store.local_dir)
        return Path(self.workspace) / self.name

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset.root,
            "embedding": f"{self.embedding.provider}:{self.embedding.model}",
            "store": f"{self.store.backend}:{self.store.collection}",
            "retrieval": {
                "arms": list(self.retrieval.arms),
                "top_k": self.retrieval.top_k,
                "dense_k": self.retrieval.dense_k,
                "sparse_k": self.retrieval.sparse_k,
                "rrf_k": self.retrieval.rrf_k,
                "tokenizer": self.retrieval.tokenizer,
                "filter_mode": self.retrieval.filter_mode,
                "parent_strategy": self.retrieval.parent_strategy,
            },
            "context": {
                "token_budget": self.context.token_budget,
                "per_doc_budget": self.context.per_doc_budget,
            },
        }
