"""Tests for keyword (Postgres full-text) search.

Two layers, mirroring test_vector_search.py:
* Always-on: the service's wiring/formatting contract against a fake repository
  -- no DB, no network.
* Opt-in: real Postgres full-text behaviour (generated tsvector column,
  websearch_to_tsquery, ts_rank_cd weighting), enabled only when
  TEST_DATABASE_URL is set. The ranking claims -- exact identifiers rank top,
  natural-language queries return sensible hits -- live here, because they are
  claims about *Postgres*, and asserting them against a hand-rolled fake would
  only test the fake.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.search.keyword_search import KeywordSearchService
from app.storage.repository import KeywordHit


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def make_hit(cid: str, content: str, rank: float, name: str | None = None) -> KeywordHit:
    return KeywordHit(
        id=cid,
        repo_url="https://example.com/r",
        commit_sha="abc",
        file_path=f"{cid}.py",
        language="python",
        chunk_type="function",
        name=name or cid,
        qualified_name=name or cid,
        parent_name=None,
        start_line=1,
        end_line=5,
        content=content,
        docstring=None,
        rank=rank,
    )


def fake_repo_keyword_search(hits):
    def _search(query, top_k=10, repo_url=None):
        return list(hits)[:top_k]

    return _search


# --------------------------------------------------------------------------- #
# Service contract
# --------------------------------------------------------------------------- #


def test_service_maps_rank_onto_score_and_keyword_score() -> None:
    hits = [make_hit("a", "def a(): ...", 0.9), make_hit("b", "def b(): ...", 0.3)]
    service = KeywordSearchService(search_fn=fake_repo_keyword_search(hits))
    results = service.search("a")

    assert [r.id for r in results] == ["a", "b"]
    assert [r.score for r in results] == [0.9, 0.3]
    assert [r.keyword_score for r in results] == [0.9, 0.3]
    # Keyword search knows nothing about vectors or fusion.
    assert all(r.similarity is None for r in results)
    assert all(r.fused_score is None for r in results)


def test_service_preserves_repository_ordering() -> None:
    hits = [make_hit(f"c{i}", f"content {i}", 1.0 - i / 10) for i in range(5)]
    service = KeywordSearchService(search_fn=fake_repo_keyword_search(hits))
    scores = [r.score for r in service.search("q")]
    assert scores == sorted(scores, reverse=True)


def test_service_passes_query_top_k_and_repo_filter_through() -> None:
    captured = {}

    def search_fn(query, top_k=10, repo_url=None):
        captured.update(query=query, top_k=top_k, repo_url=repo_url)
        return []

    service = KeywordSearchService(search_fn=search_fn)
    service.search("validate_token", top_k=7, repo_url="https://example.com/only")
    assert captured == {
        "query": "validate_token",
        "top_k": 7,
        "repo_url": "https://example.com/only",
    }


def test_service_snippet_truncation() -> None:
    service = KeywordSearchService(
        search_fn=fake_repo_keyword_search([make_hit("big", "line\n" * 500, 1.0)]),
        snippet_chars=50,
    )
    (result,) = service.search("q")
    assert len(result.snippet) <= 60  # snippet_chars + ellipsis slack
    assert result.snippet.endswith("…")


def test_service_rejects_empty_query() -> None:
    service = KeywordSearchService(search_fn=fake_repo_keyword_search([]))
    with pytest.raises(ValueError):
        service.search("   ")


def test_repository_short_circuits_blank_queries_without_touching_the_db() -> None:
    """A blank query has no tsquery to run, so it must not open a connection."""
    from app.storage.repository import search_keyword

    # Not a real pool: if the guard fails to short-circuit, .connection() on
    # this raises AttributeError and the test fails loudly.
    not_a_pool = object()

    assert search_keyword("", top_k=10, pool=not_a_pool) == []
    assert search_keyword("   ", top_k=10, pool=not_a_pool) == []
    assert search_keyword("real query", top_k=0, pool=not_a_pool) == []


# --------------------------------------------------------------------------- #
# Optional real Postgres full-text search
# --------------------------------------------------------------------------- #

TEST_DB = os.environ.get("TEST_DATABASE_URL")
requires_db = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set; skipping real-DB keyword tests"
)

# A tiny corpus with one deliberately rare identifier and several chunks that
# are topically similar but lexically different.
CORPUS = [
    (
        "validate_token",
        "AuthService.validate_token",
        "Verify a bearer token's signature and expiry.",
        "def validate_token(self, token):\n"
        "    payload = jwt.decode(token, self.secret)\n"
        "    return payload\n",
    ),
    (
        "refresh_session",
        "AuthService.refresh_session",
        "Issue a new session for an authenticated user.",
        "def refresh_session(self, user):\n    return Session(user)\n",
    ),
    (
        "hash_password",
        "AuthService.hash_password",
        "Hash a plaintext password with bcrypt before storing it.",
        "def hash_password(self, password):\n    return bcrypt.hashpw(password)\n",
    ),
    (
        "read_config",
        "config.read_config",
        "Load settings from disk.",
        "def read_config(path):\n    return json.loads(path.read_text())\n",
    ),
    (
        "render_widget",
        "ui.render_widget",
        "Draw a widget onto the canvas.",
        "def render_widget(widget):\n    canvas.draw(widget)\n",
    ),
]


def _seed(pool, repo: str) -> None:
    """Insert CORPUS as real rows (with dummy embeddings) and return."""
    from app.embeddings.embedder import EmbeddedChunk
    from app.ingestion.models import Chunk
    from app.storage.repository import upsert_chunks

    items = []
    for i, (name, qualified, doc, content) in enumerate(CORPUS):
        items.append(
            EmbeddedChunk(
                chunk=Chunk(
                    repo_url=repo,
                    commit_sha="c0ffee",
                    file_path=f"{name}.py",
                    language="python",
                    chunk_type="function",
                    name=name,
                    qualified_name=qualified,
                    parent_name=None,
                    start_line=i * 10 + 1,
                    end_line=i * 10 + 5,
                    content=content,
                    docstring=doc,
                ),
                # Keyword search ignores embeddings, but the column is NOT NULL.
                embedding=[0.0] * 768,
            )
        )
    upsert_chunks(items, pool=pool)


@pytest.fixture()
def seeded_db():
    """A real DB with the schema applied and CORPUS loaded into a fresh repo."""
    os.environ["DATABASE_URL"] = TEST_DB
    from app.config import Settings
    from app.storage import db as db_module

    db_module.close_pool()
    settings = Settings(database_url=TEST_DB)
    pool = db_module.get_pool(settings)
    db_module.apply_schema(settings)

    repo = f"https://example.com/kw-{uuid.uuid4().hex[:8]}"
    _seed(pool, repo)
    try:
        yield pool, repo
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE repo_url = %s", (repo,))
            conn.commit()
        db_module.close_pool()


@requires_db
def test_generated_tsvector_backfills_existing_rows(seeded_db) -> None:
    """The migration is additive: rows inserted normally get a search_vector."""
    pool, repo = seeded_db
    with pool.connection() as conn:
        missing = conn.execute(
            "SELECT count(*) FROM chunks "
            "WHERE repo_url = %s AND search_vector IS NULL",
            (repo,),
        ).fetchone()
    assert missing[0] == 0


@requires_db
def test_exact_identifier_query_ranks_that_chunk_first(seeded_db) -> None:
    pool, repo = seeded_db
    from app.storage.repository import search_keyword

    hits = search_keyword("validate_token", top_k=5, repo_url=repo, pool=pool)
    assert hits, "expected the rare identifier to match something"
    assert hits[0].name == "validate_token"
    # And it beats the rest by a clear margin, not a coin flip.
    if len(hits) > 1:
        assert hits[0].rank > hits[1].rank


@requires_db
def test_qualified_name_weighting_beats_a_body_only_mention(seeded_db) -> None:
    """Weight A (qualified_name) must outrank weight D (content).

    `bcrypt` appears only in hash_password's body; `hash_password` appears in
    its qualified_name. The latter should score higher.
    """
    pool, repo = seeded_db
    from app.storage.repository import search_keyword

    by_name = search_keyword("hash_password", top_k=5, repo_url=repo, pool=pool)
    by_body = search_keyword("bcrypt", top_k=5, repo_url=repo, pool=pool)

    assert by_name[0].name == "hash_password"
    assert by_body[0].name == "hash_password"
    assert by_name[0].rank > by_body[0].rank


@requires_db
def test_natural_language_query_returns_sensible_results(seeded_db) -> None:
    """Users type questions; websearch_to_tsquery must cope and rank sanely."""
    pool, repo = seeded_db
    from app.storage.repository import search_keyword

    hits = search_keyword(
        "how is a password hashed before storing?", top_k=5, repo_url=repo, pool=pool
    )
    names = [h.name for h in hits]
    assert names, "natural-language query returned nothing"
    assert names[0] == "hash_password"
    # Nothing about widgets or config should surface for that question.
    assert "render_widget" not in names


@requires_db
def test_websearch_syntax_is_honoured(seeded_db) -> None:
    pool, repo = seeded_db
    from app.storage.repository import search_keyword

    # Quoted phrase, and negation, both of which plainto_tsquery would ignore.
    quoted = search_keyword('"bearer token"', top_k=5, repo_url=repo, pool=pool)
    assert [h.name for h in quoted] == ["validate_token"]

    negated = search_keyword("session -refresh", top_k=5, repo_url=repo, pool=pool)
    assert "refresh_session" not in [h.name for h in negated]


@requires_db
def test_punctuation_heavy_query_does_not_raise(seeded_db) -> None:
    """to_tsquery would throw on this; websearch_to_tsquery must not."""
    pool, repo = seeded_db
    from app.storage.repository import search_keyword

    assert search_keyword("what & how | ?? ()", top_k=5, repo_url=repo, pool=pool) == []


@requires_db
def test_repo_filter_and_top_k_are_respected(seeded_db) -> None:
    pool, repo = seeded_db
    from app.storage.repository import search_keyword

    assert len(search_keyword("def", top_k=2, repo_url=repo, pool=pool)) <= 2
    other = search_keyword(
        "validate_token", top_k=5, repo_url=repo + "-nonexistent", pool=pool
    )
    assert other == []
