"""结构化数据模型。

设计要点（面试可讲的三件事）：

1. **ID 全部是确定性哈希**，不用 uuid4。上游 C8 用 ``uuid.uuid4()`` 生成 chunk_id，
   同一份语料每次建库 ID 都不同：评测集里的 qrels 无法指向一个稳定的块，
   trace 也无法跨运行对比。这里 chunk_id = md5(父文档相对路径 + 章节路径 + 序号)。
2. **元数据字段集合是三方唯一契约**：检索层、向量库 schema、评测集都读这一份定义，
   避免"库里有字段但过滤器用不到"这类漂移。
3. **同时保留 difficulty(str) 与 difficulty_level(int)**：上游只存 ``'★★★'`` 这类
   字符串标签，没法做 ``difficulty_level <= 2`` 的范围过滤；
   而"简单的素菜"这种查询天生需要范围条件 + 类目条件同时下推。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# 难度星数 -> 中文标签（与上游 C8 的映射保持一致，便于对照）
DIFFICULTY_LABELS: Dict[int, str] = {
    5: "非常困难",
    4: "困难",
    3: "中等",
    2: "简单",
    1: "非常简单",
}
UNKNOWN_DIFFICULTY = "未知"


def stable_id(*parts: Any) -> str:
    """由若干部分拼出稳定的 md5 十六进制串（32 字符，可直接做 VARCHAR 主键）。"""
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@dataclass
class Recipe:
    """父文档：一整篇菜谱。用于父块回填（parent expansion）与上下文拼装。"""

    parent_id: str
    source_rel: str  # 相对语料根的路径，用 posix 分隔符
    dish_name: str
    category: str  # 中文标签，如 "荤菜"
    category_key: str  # 目录名，如 "meat_dish"（ASCII，过滤更安全）
    difficulty: str  # 中文标签，用于展示与提示词
    difficulty_level: int  # 1-5，0 表示未识别
    text: str
    n_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Recipe":
        return cls(**d)


@dataclass
class Chunk:
    """子块：Markdown 标题切分后的片段，向量与 BM25 索引的基本单位。"""

    chunk_id: str
    parent_id: str
    source_rel: str
    dish_name: str
    category: str
    category_key: str
    difficulty: str
    difficulty_level: int
    section: str  # 末级标题，如 "操作"
    section_path: str  # 完整标题路径，如 "红烧肉/操作"（用 / 连接，便于字符串匹配）
    chunk_index: int  # 在父文档内的序号
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(**d)

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass
class SearchFilter:
    """检索过滤条件。

    故意做成结构化对象而不是一句 SQL/表达式字符串：同一份条件要同时
    - 交给 Milvus 变成 ``expr``（真正的下推，先过滤再算距离）；
    - 交给本地实现变成集合求交（同样是先过滤再算距离）；
    - 写进评测集，说明"这条 query 带了什么约束"。
    三种用法同源，才不会出现"线上过滤、离线不过滤"的失真。
    """

    categories: List[str] = field(default_factory=list)  # 中文类目名
    category_keys: List[str] = field(default_factory=list)  # 目录名
    difficulty_min: Optional[int] = None
    difficulty_max: Optional[int] = None
    dish_names: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.categories
            or self.category_keys
            or self.dish_names
            or self.difficulty_min is not None
            or self.difficulty_max is not None
        )

    # --- 本地实现：内存里做谓词判断 -------------------------------------
    def matches(self, chunk: Chunk) -> bool:
        if self.categories and chunk.category not in self.categories:
            return False
        if self.category_keys and chunk.category_key not in self.category_keys:
            return False
        if self.dish_names and chunk.dish_name not in self.dish_names:
            return False
        if self.difficulty_min is not None and chunk.difficulty_level < self.difficulty_min:
            return False
        if self.difficulty_max is not None:
            lvl = chunk.difficulty_level
            # 0 表示未识别难度：范围过滤时视为不满足，避免"未知"混进"简单"里
            if lvl == 0 or lvl > self.difficulty_max:
                return False
        return True

    def filter_ids(self, chunks: Iterable[Chunk]) -> List[str]:
        return [c.chunk_id for c in chunks if self.matches(c)]

    # --- Milvus：布尔表达式下推 ----------------------------------------
    @staticmethod
    def _quote(values: Iterable[str]) -> str:
        # 双引号字符串字面量；转义反斜杠与双引号，防止表达式注入
        escaped = [str(v).replace("\\", "\\\\").replace('"', '\\"') for v in values]
        return ", ".join(f'"{v}"' for v in escaped)

    def to_expr(self) -> str:
        """转成 Milvus 的过滤表达式；空过滤器返回空串（``search`` 时传 None）。"""
        clauses: List[str] = []
        if self.categories:
            clauses.append(f"category in [{self._quote(self.categories)}]")
        if self.category_keys:
            clauses.append(f"category_key in [{self._quote(self.category_keys)}]")
        if self.dish_names:
            clauses.append(f"dish_name in [{self._quote(self.dish_names)}]")
        if self.difficulty_min is not None:
            clauses.append(f"difficulty_level >= {int(self.difficulty_min)}")
        if self.difficulty_max is not None:
            clauses.append("difficulty_level > 0")
            clauses.append(f"difficulty_level <= {int(self.difficulty_max)}")
        return " and ".join(clauses)

    def describe(self) -> str:
        parts = []
        if self.category_keys:
            parts.append("类目=" + "/".join(self.category_keys))
        if self.categories:
            parts.append("标签=" + "/".join(self.categories))
        if self.difficulty_max is not None:
            parts.append(f"难度<={self.difficulty_max}")
        if self.difficulty_min is not None:
            parts.append(f"难度>={self.difficulty_min}")
        if self.dish_names:
            parts.append("菜名=" + "/".join(self.dish_names))
        return ", ".join(parts) if parts else "无过滤"


@dataclass
class ScoredChunk:
    """一个召回块的完整打分记录：两路各自的排名/分数 + 融合分。

    上游 C8 把 ``rrf_score`` 直接写回 ``Document.metadata``，而 Document 对象在
    BM25 与向量两路之间是共享的：第二次查询会读到上一次写进去的分数。
    这里改成不可变的返回对象，块本身不被加分数字段。
    """

    chunk: Chunk
    dense_rank: Optional[int] = None
    dense_score: float = 0.0
    sparse_rank: Optional[int] = None
    sparse_score: float = 0.0
    rrf_score: float = 0.0

    @property
    def from_dense(self) -> bool:
        return self.dense_rank is not None

    @property
    def from_sparse(self) -> bool:
        return self.sparse_rank is not None


@dataclass
class ParentHit:
    """父文档 + 其融合分（用于父块回填与上下文排序）。"""

    recipe: Recipe
    score: float
    hit_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = self.recipe.to_dict()
        d.update({"score": self.score, "hit_chunks": self.hit_chunks})
        return d


@dataclass
class RetrievalResult:
    """一次检索的完整输出，可直接序列化成 trace 落盘。"""

    query: str
    fused: List[ScoredChunk]
    parents: List[ParentHit] = field(default_factory=list)
    filter_desc: str = "无过滤"
    candidates: int = 0  # 过滤后进入检索的块数
    dense_hits: int = 0
    sparse_hits: int = 0
    latency_ms: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def top_parents(self, n: int) -> List[ParentHit]:
        return self.parents[:n]

    def to_trace(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "filter": self.filter_desc,
            "candidates": self.candidates,
            "dense_hits": self.dense_hits,
            "sparse_hits": self.sparse_hits,
            "latency_ms": round(self.latency_ms, 2),
            "fused": [
                {
                    "chunk_id": s.chunk.chunk_id,
                    "dish": s.chunk.dish_name,
                    "section": s.chunk.section,
                    "dense_rank": s.dense_rank,
                    "sparse_rank": s.sparse_rank,
                    "rrf": round(s.rrf_score, 6),
                }
                for s in self.fused
            ],
            "parents": [
                {"parent_id": p.recipe.parent_id, "dish": p.recipe.dish_name,
                 "score": round(p.score, 6), "hit_chunks": p.hit_chunks}
                for p in self.parents
            ],
            "extras": self.extras,
        }
