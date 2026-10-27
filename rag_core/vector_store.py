"""向量库后端：Milvus（主）+ 本地 numpy（对照/离线）。

## 为什么主后端是 Milvus

上游 C8 用 ``FAISS.from_documents`` + 后置过滤：先取 ``top_k*3`` 条，再在 Python 里
逐条比对 metadata，凑够 ``top_k`` 就 break。过滤条件越严，候选越少，召回直接掉下去
（"找 2 道简单的素菜"这种查询几乎必然漏）。Milvus 的价值不在"存向量"，而在两件事：

1. **标量过滤与向量检索在同一次调用里完成**（``filter="category_key in [...] and difficulty_level<=2"``），
   先求候选集再算距离，没有"取多了浪费、取少了漏"的两难；
2. **标量字段建 INVERTED 索引**，过滤本身也走索引而不是全表扫。

`mode="post"` 保留给**消融实验**：同一份数据、同一个索引，只把过滤挪到检索之后，
就能量化"后置过滤损失了多少召回"。没有这个开关，"我用 Milvus 所以更快更好"只是口号。

## 为什么还留一个本地 numpy 后端

不是技术洁癖：消融实验要跑几十轮，每轮都起容器、清 collection 会让单次迭代从 3 秒变 30 秒，
实验就做不密。并且评测脚本必须在没有 Docker 的机器上也能复现结果。
本地实现走**同一套接口、同一份过滤对象**，两条后端的差异只剩"谁执行过滤"。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

from .schema import Chunk, SearchFilter

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    dim: int

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int: ...

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        flt: Optional[SearchFilter] = None,
        mode: str = "pre",
    ) -> List[Tuple[str, float]]: ...

    def count(self) -> int: ...


# ---------------------------------------------------------------------------
# 本地实现
# ---------------------------------------------------------------------------
class LocalVectorStore:
    """numpy 精确内积检索。语料 10^3~10^4 块时比 FAISS 还快（省掉索引构建与一次拷贝）。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: List[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._index: Dict[str, int] = {}

    # ---- 写入 ----
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 数量不一致：{len(chunks)} vs {len(vectors)}")
        vectors = np.asarray(vectors, dtype=np.float32)
        # 用 upsert 语义覆盖同 id，保证重复建库不会翻倍
        new_rows: List[np.ndarray] = []
        new_chunks: List[Chunk] = []
        for chunk, vec in zip(chunks, vectors):
            if chunk.chunk_id in self._index:
                self._matrix[self._index[chunk.chunk_id]] = vec
                self._chunks[self._index[chunk.chunk_id]] = chunk
                continue
            self._index[chunk.chunk_id] = len(self._chunks)
            new_chunks.append(chunk)
            new_rows.append(vec)
        if new_rows:
            self._matrix = np.vstack([self._matrix, np.vstack(new_rows)])
            self._chunks.extend(new_chunks)
        return len(chunks)

    # ---- 检索 ----
    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        flt: Optional[SearchFilter] = None,
        mode: str = "pre",
    ) -> List[Tuple[str, float]]:
        if not self._chunks or top_k <= 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        scores = self._matrix @ q  # 行已归一化，内积即余弦

        if flt is None or flt.is_empty():
            order = np.argsort(-scores)[:top_k]
            return [(self._chunks[i].chunk_id, float(scores[i])) for i in order]

        if mode == "pre":
            keep = np.array([flt.matches(c) for c in self._chunks], dtype=bool)
            if not keep.any():
                return []
            masked = np.where(keep, scores, -np.inf)
            order = np.argsort(-masked)[:top_k]
            return [
                (self._chunks[i].chunk_id, float(masked[i]))
                for i in order
                if np.isfinite(masked[i])
            ]

        # post：先取全量 top-N，再过滤 —— 复现上游行为，用于量化召回损失
        n = min(len(scores), max(top_k * 3, top_k))
        order = np.argsort(-scores)[:n]
        out: List[Tuple[str, float]] = []
        for i in order:
            chunk = self._chunks[i]
            if flt.matches(chunk):
                out.append((chunk.chunk_id, float(scores[i])))
                if len(out) >= top_k:
                    break
        return out

    def count(self) -> int:
        return len(self._chunks)

    # ---- 持久化 ----
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self._matrix)
        (out / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self._chunks),
            encoding="utf-8",
        )
        logger.info("本地向量库已保存：%s（%d 块，%d 维）", out, len(self._chunks), self.dim)

    @classmethod
    def load(cls, in_dir: str | Path, dim: int) -> "LocalVectorStore":
        src = Path(in_dir)
        store = cls(dim)
        store._matrix = np.load(src / "vectors.npy")
        store._chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (src / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        store._index = {c.chunk_id: i for i, c in enumerate(store._chunks)}
        return store


# ---------------------------------------------------------------------------
# Milvus 实现
# ---------------------------------------------------------------------------
class MilvusVectorStore:
    """Milvus 后端。

    schema 里的标量字段不是"随手加的元数据"，每一个都对应一个必须下推的过滤条件：
    ``category_key`` / ``difficulty_level`` 建 INVERTED 索引，用于"简单素菜"这类组合过滤；
    ``dish_name`` 支持"同菜名不同做法"的精确定位；
    ``parent_id`` 用于把命中块一次性反查回父文档（避免上游 O(n²) 的线性扫描）。
    """

    TEXT_MAX_LEN = 16384

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection: str = "recipe_chunks",
        dim: int = 512,
        metric_type: str = "COSINE",
    ) -> None:
        from pymilvus import MilvusClient  # 延迟导入

        self.dim = dim
        self.collection = collection
        self.metric_type = metric_type
        self.client = MilvusClient(uri=uri, token=token or None)

    # ---- schema ----
    def create_collection(self, drop_existing: bool = False) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.collection):
            if not drop_existing:
                logger.info("collection %s 已存在，跳过创建", self.collection)
                return
            self.client.drop_collection(self.collection)
            logger.info("已删除旧 collection %s", self.collection)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("dish_name", DataType.VARCHAR, max_length=256)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("category_key", DataType.VARCHAR, max_length=64)
        schema.add_field("difficulty_level", DataType.INT8)
        schema.add_field("section", DataType.VARCHAR, max_length=256)
        schema.add_field("source_rel", DataType.VARCHAR, max_length=512)
        schema.add_field("text", DataType.VARCHAR, max_length=self.TEXT_MAX_LEN)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type=self.metric_type
        )
        for scalar in ("category_key", "difficulty_level", "parent_id"):
            try:
                index_params.add_index(field_name=scalar, index_type="INVERTED")
            except Exception as exc:  # 老版本可能不支持 INVERTED，退化为暴力过滤
                logger.warning("标量索引 %s 创建失败（改用暴力过滤）：%s", scalar, exc)

        self.client.create_collection(
            collection_name=self.collection, schema=schema, index_params=index_params
        )
        logger.info("已创建 collection %s（%d 维，%s）", self.collection, self.dim, self.metric_type)

    # ---- 写入 ----
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 数量不一致：{len(chunks)} vs {len(vectors)}")
        rows = []
        for chunk, vec in zip(chunks, vectors):
            if len(chunk.text) > self.TEXT_MAX_LEN:
                logger.warning("块 %s 超过 VARCHAR(%d)，将被截断", chunk.chunk_id, self.TEXT_MAX_LEN)
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                    "dish_name": chunk.dish_name[:256],
                    "category": chunk.category[:64],
                    "category_key": chunk.category_key[:64],
                    "difficulty_level": int(chunk.difficulty_level),
                    "section": (chunk.section or "")[:256],
                    "source_rel": chunk.source_rel[:512],
                    "text": chunk.text[: self.TEXT_MAX_LEN],
                    "vector": vec.tolist(),
                }
            )
        # 分批写入：Milvus 单次请求有大小上限，300 篇规模下 200/批比较稳
        batch = 200
        for i in range(0, len(rows), batch):
            self.client.upsert(collection_name=self.collection, data=rows[i : i + batch])
        return len(rows)

    # ---- 检索 ----
    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        flt: Optional[SearchFilter] = None,
        mode: str = "pre",
    ) -> List[Tuple[str, float]]:
        expr = None
        limit = top_k
        if flt is not None and not flt.is_empty():
            expr = flt.to_expr()
            if mode == "post":
                # 复现"先检索后过滤"：过滤表达式不生效，靠放大候选数硬凑
                expr = None
                limit = max(top_k * 3, top_k)

        res = self.client.search(
            collection_name=self.collection,
            data=[np.asarray(vector, dtype=np.float32).reshape(-1).tolist()],
            limit=limit,
            filter=expr,
            output_fields=["category", "category_key", "difficulty_level", "dish_name"],
            search_params={"metric_type": self.metric_type},
        )
        hits = res[0] if res else []
        rows = [
            (str(hit.get("id", hit.get("chunk_id"))), float(hit.get("distance", 0.0)))
            for hit in hits
        ]

        if flt is None or flt.is_empty() or mode != "post":
            return rows[:top_k]

        # post 模式：结果里的标量字段就是判断依据，不需要再回表
        kept: List[Tuple[str, float]] = []
        for hit in hits:
            ent = hit.get("entity", {}) or {}
            if flt.category_keys and ent.get("category_key") not in flt.category_keys:
                continue
            if flt.categories and ent.get("category") not in flt.categories:
                continue
            if flt.dish_names and ent.get("dish_name") not in flt.dish_names:
                continue
            level = int(ent.get("difficulty_level", 0) or 0)
            if flt.difficulty_min is not None and level < flt.difficulty_min:
                continue
            if flt.difficulty_max is not None and (level == 0 or level > flt.difficulty_max):
                continue
            kept.append((str(hit.get("id", hit.get("chunk_id"))), float(hit.get("distance", 0.0))))
            if len(kept) >= top_k:
                break
        return kept

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)


def build_store(cfg) -> VectorStore:
    """按配置造后端。cfg 为 :class:`rag_core.config.StoreConfig`。"""
    backend = cfg.backend.lower()
    if backend == "milvus":
        store = MilvusVectorStore(
            uri=cfg.uri,
            token=cfg.token,
            collection=cfg.collection,
            dim=cfg.dim,
            metric_type=cfg.metric,
        )
        store.create_collection(drop_existing=cfg.recreate)
        return store
    if backend == "local":
        return LocalVectorStore(dim=cfg.dim)
    raise ValueError(f"未知向量库后端：{cfg.backend}（可选 milvus / local）")
