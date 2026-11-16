"""契约测试：只覆盖最容易静默出错、且被实验直接依赖的行为。"""

from __future__ import annotations

import pytest

import rag_core.cli as cli_module
from rag_core.bm25 import BM25Index
from rag_core.cli import build_filter, main
from rag_core.config import AppConfig, DatasetCfg, EmbeddingCfg, LLMCfg, StoreCfg
from rag_core.context import ContextConfig, build_context
from rag_core.data_preparation import rank_parents
from rag_core.embeddings import HashingEmbedder
from rag_core.markdown_splitter import split_markdown
from rag_core.pipeline import RecipeRAG
from rag_core.schema import Chunk, ParentHit, Recipe, ScoredChunk, SearchFilter, stable_id
from rag_core.text import estimate_tokens, truncate_to_tokens
from rag_core.vector_store import LocalVectorStore


def make_chunk(
    cid: str,
    pid: str = "p",
    text: str = "红烧肉 五花肉 炖煮",
    category_key: str = "meat_dish",
    level: int = 2,
    section: str = "操作",
) -> Chunk:
    category = "素菜" if category_key == "vegetable_dish" else "荤菜"
    return Chunk(
        chunk_id=cid,
        parent_id=pid,
        source_rel=f"{cid}.md",
        dish_name=cid,
        category=category,
        category_key=category_key,
        difficulty="简单" if level == 2 else "未知",
        difficulty_level=level,
        section=section,
        section_path=f"{cid}/{section}",
        chunk_index=0,
        text=text,
    )


def make_recipe(pid: str, dish: str = "菜", text: str = "正文") -> Recipe:
    return Recipe(
        parent_id=pid,
        source_rel=f"{dish}.md",
        dish_name=dish,
        category="荤菜",
        category_key="meat_dish",
        difficulty="简单",
        difficulty_level=2,
        text=text,
    )


def test_stable_id_is_deterministic_and_unambiguous() -> None:
    assert stable_id("a", "b") == stable_id("a", "b")
    assert len(stable_id("a")) == 32
    assert stable_id("a", "b") != stable_id("ab")


def test_search_filter_expr_and_unknown_difficulty() -> None:
    flt = SearchFilter(
        category_keys=["meat_dish"],
        difficulty_max=2,
        dish_names=['a"b'],
    )
    expr = flt.to_expr()
    assert 'category_key in ["meat_dish"]' in expr
    assert "difficulty_level > 0" in expr
    assert "difficulty_level <= 2" in expr
    assert '\\"b' in expr
    assert SearchFilter().to_expr() == ""

    unknown = make_chunk("c", level=0)
    assert not SearchFilter(difficulty_max=2).matches(unknown)
    assert not SearchFilter(difficulty_min=1).matches(unknown)
    assert SearchFilter().matches(unknown)


def test_markdown_splitter_ignores_fences_and_comments() -> None:
    text = (
        "# 菜\n\n正文\n\n"
        "```\n## 不是标题\n```\n\n"
        "<!-- 请填写 -->\n"
        "## 操作\n\n步骤"
    )
    sections = split_markdown(text)
    assert [s.title for s in sections] == ["菜", "操作"]
    assert all("请填写" not in s.body for s in sections)


def test_truncate_respects_budget_and_boundary() -> None:
    text = "第一段。" * 200
    out = truncate_to_tokens(text, 50)
    assert out
    assert estimate_tokens(out) <= 50
    assert text.startswith(out)
    assert truncate_to_tokens("abc", 10) == "abc"


def test_rank_parents_three_strategies() -> None:
    r1, r2 = make_recipe("p1"), make_recipe("p2")
    fused = [
        ScoredChunk(chunk=make_chunk("c1", "p1"), rrf_score=0.5),
        ScoredChunk(chunk=make_chunk("c2", "p2"), rrf_score=0.9),
        ScoredChunk(chunk=make_chunk("c3", "p1"), rrf_score=0.1),
    ]
    recipes = {"p1": r1, "p2": r2}

    rrf = rank_parents(fused, recipes, strategy="rrf_sum")
    assert [p.recipe.parent_id for p in rrf] == ["p2", "p1"]

    hits = rank_parents(fused, recipes, strategy="hit_count")
    assert [p.recipe.parent_id for p in hits] == ["p1", "p2"]

    best = rank_parents(fused, recipes, strategy="max_score")
    assert [p.recipe.parent_id for p in best] == ["p2", "p1"]


def test_bm25_allowed_ids_is_subset_search() -> None:
    chunks = [
        make_chunk("c1", "p1", "红烧肉 五花肉"),
        make_chunk("c2", "p2", "清炒青菜"),
    ]
    index = BM25Index(tokenizer_name="char_bigram").fit(chunks)

    found = index.search("红烧肉", top_k=5)
    assert found and found[0][0] == "c1"
    assert index.search("红烧肉", top_k=5, allowed_ids={"c2"}) == []
    assert index.search("红烧肉", top_k=5, allowed_ids=set()) == []


def test_local_store_pre_filters_before_ranking() -> None:
    chunks = [
        make_chunk("c1", "p1", "青菜 青菜 青菜"),
        make_chunk("c2", "p2", "青菜 青菜"),
        make_chunk("c3", "p3", "青菜"),
        make_chunk("c4", "p4", "红烧肉", category_key="vegetable_dish"),
    ]
    embedder = HashingEmbedder(dim=64)
    store = LocalVectorStore(dim=64)
    store.upsert(chunks, embedder.encode([c.text for c in chunks]))
    query = embedder.encode(["青菜"])[0]
    flt = SearchFilter(category_keys=["vegetable_dish"])

    pre = store.search(query, top_k=1, flt=flt, mode="pre")
    assert [cid for cid, _ in pre] == ["c4"]

    # post 只取全量 top-3，再筛：前三名都是不满足条件的荤菜，结果必为空
    post = store.search(query, top_k=1, flt=flt, mode="post")
    assert post == []


def test_context_truncate_vs_drop() -> None:
    parents = [
        ParentHit(recipe=make_recipe("p1", "长菜", "步骤。" * 1000), score=1.0),
        ParentHit(recipe=make_recipe("p2", "短菜", "原料。" * 500), score=0.9),
    ]
    truncated = build_context(
        parents,
        ContextConfig(token_budget=300, per_doc_budget=200, min_doc_tokens=10, on_overflow="truncate"),
    )
    assert truncated.used
    assert truncated.truncated_docs >= 1
    assert truncated.tokens <= 320

    dropped = build_context(
        parents,
        ContextConfig(token_budget=300, per_doc_budget=900, min_doc_tokens=10, on_overflow="drop"),
    )
    assert dropped.used == []


def test_config_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nunknown_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        AppConfig.from_yaml(path)


def test_env_expansion_uses_default_for_missing_or_empty(monkeypatch) -> None:
    from rag_core.config import _expand_env

    monkeypatch.setenv("RAG_TEST_EMPTY", "")
    monkeypatch.setenv("RAG_TEST_SET", "value")
    monkeypatch.delenv("RAG_TEST_MISSING", raising=False)

    assert _expand_env("${RAG_TEST_EMPTY:-fallback}") == "fallback"
    assert _expand_env("${RAG_TEST_SET:-fallback}") == "value"
    assert _expand_env("${RAG_TEST_MISSING:-fallback}") == "fallback"


def test_build_filter_maps_category_and_dish() -> None:
    flt = build_filter(["素菜", "meat_dish", "不存在的菜"], max_difficulty=2)
    assert flt.categories == ["素菜"]
    assert flt.category_keys == ["meat_dish"]
    assert flt.dish_names == ["不存在的菜"]
    assert flt.difficulty_max == 2


def test_cli_stats_accepts_config_before_and_after_subcommand(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"name: t\nworkspace: {tmp_path}\n", encoding="utf-8")

    class FakeRAG:
        @classmethod
        def load(cls, cfg, skip_embedding: bool = False):
            return cls()

        def info(self):
            return {"ok": True}

    monkeypatch.setattr(cli_module, "RecipeRAG", FakeRAG)
    assert main(["--config", str(cfg_path), "stats"]) == 0
    assert main(["stats", "--config", str(cfg_path)]) == 0


def test_build_then_load_reuses_local_index(tmp_path) -> None:
    root = tmp_path / "cook"
    dish_dir = root / "dishes" / "vegetable_dish"
    dish_dir.mkdir(parents=True)
    (dish_dir / "炒青菜.md").write_text(
        "# 炒青菜\n\n预估烹饪难度：★\n\n"
        "## 必备原料\n\n青菜 300 克，蒜 2 瓣。\n\n"
        "## 操作\n\n热锅下油，蒜爆香，下青菜翻炒两分钟。\n",
        encoding="utf-8",
    )

    cfg = AppConfig(
        name="exp",
        workspace=str(tmp_path / "artifacts"),
        dataset=DatasetCfg(root=str(root)),
        embedding=EmbeddingCfg(provider="hashing", dim=16),
        store=StoreCfg(backend="local"),
        llm=LLMCfg(model="none"),
    )
    built = RecipeRAG.build(cfg)
    assert built.dataset.chunks
    index_dir = cfg.effective_local_dir()
    assert (index_dir / "vectors.npy").is_file()

    loaded = RecipeRAG.load(cfg)
    assert len(loaded.dataset.chunks) == len(built.dataset.chunks)
    assert loaded.retriever.store.dim == 16
    assert loaded.retrieve("炒青菜").fused

    other = AppConfig(
        name="exp2",
        workspace=str(tmp_path / "artifacts"),
        dataset=DatasetCfg(root=str(root)),
        embedding=EmbeddingCfg(provider="hashing", dim=16),
        store=StoreCfg(backend="local"),
        llm=LLMCfg(model="none"),
    )
    assert other.effective_local_dir() != index_dir
