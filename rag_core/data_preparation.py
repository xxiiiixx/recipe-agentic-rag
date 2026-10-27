"""语料加载与分块：C8 的父子块设计，重写为确定性、可过滤、可缓存的实现。

与上游 ``code/C8/rag_modules/data_preparation.py`` 的差异（逐条对应 docs/01-vs-c8.md 的缺陷表）：

| 上游问题 | 上游位置 | 本实现 |
| --- | --- | --- |
| ``rglob("*.md")`` 把 ``template/`` 投稿模板当菜谱入库 | L59 | 取数时删掉 ``dishes/template``，语料只有真菜谱 |
| ``chunk_id = uuid4()``，两次建库 ID 不同，qrels 无法锚定 | L200 | ``chunk_id = md5(parent_id, 章节路径, 序号)`` |
| 只存 ``difficulty='★★★'`` 字符串，无法范围过滤 | L117-123 | 额外存 ``difficulty_level``（1-5，0=未识别） |
| ``CATEGORY_LABELS = list(set(...))``，顺序不稳定 | L31 | 用 ``OrderedDict`` 保序，标签顺序固定 |
| 贡献者 HTML 注释被切进块、送进 embedding | — | 切分前剥离 ``<!-- -->`` |
| 父文档排序只看命中块个数，丢弃融合分 | L302-350 | ``rank_parents`` 支持 ``rrf_sum`` / ``hit_count`` 两种策略并可切换 |
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .markdown_splitter import split_markdown
from .schema import (
    DIFFICULTY_LABELS,
    UNKNOWN_DIFFICULTY,
    Chunk,
    ParentHit,
    Recipe,
    ScoredChunk,
    stable_id,
)

logger = logging.getLogger(__name__)

# 目录名 -> 中文类目。用 OrderedDict 保证顺序固定（上游用 list(set(...)) 会随哈希顺序变）。
CATEGORY_MAPPING: "OrderedDict[str, str]" = OrderedDict(
    [
        ("meat_dish", "荤菜"),
        ("vegetable_dish", "素菜"),
        ("soup", "汤品"),
        ("dessert", "甜品"),
        ("breakfast", "早餐"),
        ("staple", "主食"),
        ("aquatic", "水产"),
        ("condiment", "调料"),
        ("drink", "饮品"),
        ("semi-finished", "半成品"),
    ]
)
CATEGORY_LABELS: List[str] = list(CATEGORY_MAPPING.values())

_DIFFICULTY_RE = re.compile(r"烹饪难度[：:\s]*([★☆]+)")
_STAR_ONLY_RE = re.compile(r"[★☆]+")


def parse_difficulty(text: str) -> Tuple[int, str]:
    """从"预估烹饪难度：★★★★★"里解析等级。

    只数 ★，不数 ☆ —— ``★★★☆☆`` 与 ``★★★`` 都是 3 级。收不到就返回 (0, '未知')，
    交给 ``SearchFilter`` 决定"未知难度要不要参与范围过滤"。
    """
    match = _DIFFICULTY_RE.search(text) or _STAR_ONLY_RE.search(text)
    if not match:
        return 0, UNKNOWN_DIFFICULTY
    stars = len(re.findall("★", match.group(0)))
    return stars, DIFFICULTY_LABELS.get(stars, UNKNOWN_DIFFICULTY)


@dataclass
class DatasetConfig:
    root: str
    max_level: int = 3
    drop_comments: bool = True
    min_chunk_chars: int = 8


class RecipeDataset:
    """一次加载，产出父文档与子块；可序列化到 jsonl 供评测与建库复用。"""

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.recipes: List[Recipe] = []
        self.chunks: List[Chunk] = []

    # ------------------------------------------------------------------ load
    def _iter_files(self) -> Iterable[Path]:
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            yield path

    def load(self) -> "RecipeDataset":
        if not self.root.exists():
            raise FileNotFoundError(f"语料根目录不存在：{self.root}")

        recipes: List[Recipe] = []
        for path in self._iter_files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("跳过非 UTF-8 文件：%s", path)
                continue
            if not text.strip():
                continue
            rel = path.relative_to(self.root).as_posix()
            category_key = next(
                (p for p in Path(rel).parts[:-1] if p in CATEGORY_MAPPING), "other"
            )
            level, label = parse_difficulty(text)
            recipes.append(
                Recipe(
                    parent_id=stable_id(rel),
                    source_rel=rel,
                    dish_name=path.stem,
                    category=CATEGORY_MAPPING.get(category_key, "其他"),
                    category_key=category_key,
                    difficulty=label,
                    difficulty_level=level,
                    text=text,
                )
            )

        self.recipes = recipes
        logger.info("加载 %d 篇菜谱（根目录 %s）", len(recipes), self.root)
        return self

    # ----------------------------------------------------------------- chunk
    def build_chunks(self) -> "RecipeDataset":
        if not self.recipes:
            raise RuntimeError("请先调用 load()")

        chunks: List[Chunk] = []
        for recipe in self.recipes:
            sections = split_markdown(
                recipe.text,
                max_level=self.cfg.max_level,
                drop_comments=self.cfg.drop_comments,
            )
            count = 0
            for idx, section in enumerate(sections):
                body = section.body.strip()
                if len(body) < self.cfg.min_chunk_chars:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=stable_id(recipe.parent_id, section.path_str, idx),
                        parent_id=recipe.parent_id,
                        source_rel=recipe.source_rel,
                        dish_name=recipe.dish_name,
                        category=recipe.category,
                        category_key=recipe.category_key,
                        difficulty=recipe.difficulty,
                        difficulty_level=recipe.difficulty_level,
                        section=section.title or recipe.dish_name,
                        section_path=section.path_str,
                        chunk_index=idx,
                        text=body,
                    )
                )
                count += 1
            recipe.n_chunks = count

        self.chunks = chunks
        logger.info("切出 %d 个子块，平均每篇 %.1f 块", len(chunks), len(chunks) / max(len(self.recipes), 1))
        return self

    # ------------------------------------------------------------------ util
    def chunk_map(self) -> Dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}

    def recipe_map(self) -> Dict[str, Recipe]:
        return {r.parent_id: r for r in self.recipes}

    def stats(self) -> Dict[str, object]:
        cats = Counter(r.category for r in self.recipes)
        levels = Counter(r.difficulty for r in self.recipes)
        sizes = [len(c.text) for c in self.chunks] or [0]
        return {
            "recipes": len(self.recipes),
            "chunks": len(self.chunks),
            "categories": dict(cats),
            "difficulties": dict(levels),
            "avg_chunk_chars": round(sum(sizes) / len(sizes), 1),
            "max_chunk_chars": max(sizes),
        }

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "recipes.jsonl").open("w", encoding="utf-8") as f:
            for r in self.recipes:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (out / "stats.json").write_text(
            json.dumps(self.stats(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("已写入 %s（recipes.jsonl / chunks.jsonl / stats.json）", out)

    @staticmethod
    def load_jsonl(path: str | Path) -> Tuple[List[Recipe], List[Chunk]]:
        recipes: List[Recipe] = []
        chunks: List[Chunk] = []
        with Path(path, "recipes.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recipes.append(Recipe.from_dict(json.loads(line)))
        with Path(path, "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk.from_dict(json.loads(line)))
        return recipes, chunks


def rank_parents(
    fused: Sequence[ScoredChunk],
    recipes: Dict[str, Recipe],
    strategy: str = "rrf_sum",
    top_n: Optional[int] = None,
) -> List[ParentHit]:
    """子块 -> 父文档的聚合排序。

    ``rrf_sum``（默认）：按融合分累加。第 1 名与第 9 名贡献不同，
        这修掉了上游"只数命中块个数"导致的相关性丢失。
    ``hit_count``：复现上游行为，仅用于对照实验。
    ``max_score``：取单块最高分，适合"只要最相关那一段"的问答。

    注意：上游在 ``get_parent_documents`` 里对每个父 ID 做了一次
    ``for doc in self.documents`` 的线性扫描（O(n²)），这里直接用 id->Recipe 字典。
    """
    scores: Dict[str, float] = {}
    counts: Counter = Counter()
    best: Dict[str, float] = {}

    for hit in fused:
        pid = hit.chunk.parent_id
        counts[pid] += 1
        best[pid] = max(best.get(pid, 0.0), hit.rrf_score)
        if strategy == "rrf_sum":
            scores[pid] = scores.get(pid, 0.0) + hit.rrf_score
        elif strategy == "hit_count":
            scores[pid] = scores.get(pid, 0.0) + 1.0
        elif strategy == "max_score":
            scores[pid] = best[pid]
        else:
            raise ValueError(f"未知父文档排序策略：{strategy}")

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return [
        ParentHit(recipe=recipes[pid], score=score, hit_chunks=counts[pid])
        for pid, score in ordered
        if pid in recipes
    ]
