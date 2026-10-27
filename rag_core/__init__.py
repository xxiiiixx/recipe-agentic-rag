"""recipe-agentic-rag 的检索底座。

对外只暴露"建库 + 检索 + 生成"三件事，Agent 编排层（P2 引入 LangGraph）在 ``agent/``，
评估层在 ``eval/``，二者都只依赖这里的公开函数，不直接碰 Milvus 或 BM25 内部结构。
"""

from .bm25 import BM25Index
from .config import AppConfig
from .context import ContextConfig, build_context
from .data_preparation import RecipeDataset, rank_parents
from .generation import RecipeAnswerer
from .retrieval import HybridRetriever, RetrievalConfig
from .schema import (
    DIFFICULTY_LABELS,
    Chunk,
    ParentHit,
    Recipe,
    RetrievalResult,
    ScoredChunk,
    SearchFilter,
)
from .vector_store import LocalVectorStore, MilvusVectorStore

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "BM25Index",
    "Chunk",
    "ContextConfig",
    "DIFFICULTY_LABELS",
    "HybridRetriever",
    "LocalVectorStore",
    "MilvusVectorStore",
    "ParentHit",
    "Recipe",
    "RecipeAnswerer",
    "RecipeDataset",
    "RetrievalConfig",
    "RetrievalResult",
    "ScoredChunk",
    "SearchFilter",
    "build_context",
    "rank_parents",
]
