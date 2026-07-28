"""Tests for flat vector search.

Two layers:
* Always-on: the search *service* wiring + ordering contract, using a fake
  embedder and a fake repository (numpy cosine oracle) -- no DB, no network.
* Opt-in: a real pgvector round-trip (upsert + `<=>` ordering + idempotent
  re-upsert), enabled only when TEST_DATABASE_URL is set.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.search.vector_search import VectorSearchService, cosine_rank
from app.storage.repository import SearchHit


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeEmbedder:
    """Returns a preset query vector; records the query it was asked to embed."""

    def __init__(self, query_vector):
        self.query_vector = query_vector
        self.seen: list[str] = []

    def embed_query(self, query: str):
        self.seen.append(query)
        return self.query_vector


def make_hit(cid: str, content: str, similarity: float) -> SearchHit:
    return SearchHit(
        id=cid,
        repo_url="https://example.com/r",
        commit_sha="abc",
        file_path=f"{cid}.py",
        language="python",
        chunk_type="function",
        name=cid,
        qualified_name=cid,
        parent_name=None,
        start_line=1,
        end_line=5,
        content=content,
        docstring=None,
        similarity=similarity,
    )


def fake_repo_search(corpus):
    """Build a search_fn that cosine-ranks an in-memory corpus.

    corpus: list of (id, vector, content).
    """

    def _search(query_embedding, top_k=10, repo_url=None):
        ranked = cosine_rank(query_embedding, [(cid, vec) for cid, vec, _ in corpus])
        content_by_id = {cid: content for cid, _, content in corpus}
        hits = [make_hit(cid, content_by_id[cid], sim) for cid, sim in ranked]
        return hits[:top_k]

    return _search


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #


def test_cosine_rank_orders_by_similarity() -> None:
    query = [1.0, 0.0]
    ranked = cosine_rank(
        query,
        [("orthogonal", [0.0, 1.0]), ("same", [1.0, 0.0]), ("close", [0.9, 0.1])],
    )
    assert [cid for cid, _ in ranked] == ["same", "close", "orthogonal"]
    assert ranked[0][1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Service wiring / ordering contract
# --------------------------------------------------------------------------- #


def test_service_returns_results_ordered_by_similarity() -> None:
    corpus = [
        ("auth", [1.0, 0.0, 0.0], "def validate_token(): ..."),
        ("io", [0.0, 1.0, 0.0], "def read_file(): ..."),
        ("authish", [0.8, 0.2, 0.0], "def check_login(): ..."),
    ]
    service = VectorSearchService(
        FakeEmbedder([1.0, 0.0, 0.0]), search_fn=fake_repo_search(corpus)
    )
    results = service.search("how are tokens validated?", top_k=3)

    assert [r.name for r in results] == ["auth", "authish", "io"]
    # Similarities are monotonically non-increasing.
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_service_respects_top_k() -> None:
    corpus = [(f"c{i}", [float(i), 1.0], f"content {i}") for i in range(10)]
    service = VectorSearchService(
        FakeEmbedder([5.0, 1.0]), search_fn=fake_repo_search(corpus)
    )
    assert len(service.search("q", top_k=3)) == 3


def test_service_passes_repo_filter_through() -> None:
    captured = {}

    def search_fn(query_embedding, top_k=10, repo_url=None):
        captured["repo_url"] = repo_url
        captured["top_k"] = top_k
        return []

    service = VectorSearchService(FakeEmbedder([1.0]), search_fn=search_fn)
    service.search("q", top_k=7, repo_url="https://example.com/only")
    assert captured == {"repo_url": "https://example.com/only", "top_k": 7}


def test_service_snippet_truncation() -> None:
    long_content = "line\n" * 500
    corpus = [("big", [1.0], long_content)]
    service = VectorSearchService(
        FakeEmbedder([1.0]), search_fn=fake_repo_search(corpus), snippet_chars=50
    )
    (result,) = service.search("q")
    assert len(result.snippet) <= 60  # snippet_chars + ellipsis slack
    assert result.snippet.endswith("…")


def test_service_rejects_empty_query() -> None:
    service = VectorSearchService(FakeEmbedder([1.0]), search_fn=fake_repo_search([]))
    with pytest.raises(ValueError):
        service.search("   ")


# --------------------------------------------------------------------------- #
# Optional real pgvector round-trip
# --------------------------------------------------------------------------- #

TEST_DB = os.environ.get("TEST_DATABASE_URL")
pytestmark_db = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set; skipping real-DB vector tests"
)


@pytestmark_db
def test_real_pgvector_upsert_search_and_idempotency() -> None:
    # Import here so the module imports fine without a DB configured.
    os.environ["DATABASE_URL"] = TEST_DB
    from app.config import Settings
    from app.embeddings.embedder import EmbeddedChunk
    from app.ingestion.models import Chunk
    from app.storage import db as db_module
    from app.storage.repository import count_chunks, search_similar, upsert_chunks

    db_module.close_pool()
    settings = Settings(database_url=TEST_DB, embedding_dim=4)
    pool = db_module.get_pool(settings)
    db_module.apply_schema(settings)

    repo = f"https://example.com/test-{uuid.uuid4().hex[:8]}"

    def ec(name, line, vector) -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk=Chunk(
                repo_url=repo,
                commit_sha="c0ffee",
                file_path=f"{name}.py",
                language="python",
                chunk_type="function",
                name=name,
                qualified_name=name,
                parent_name=None,
                start_line=line,
                end_line=line + 1,
                content=f"def {name}(): ...",
                docstring=None,
            ),
            embedding=vector,
        )

    items = [
        ec("east", 1, [1.0, 0.0, 0.0, 0.0]),
        ec("north", 2, [0.0, 1.0, 0.0, 0.0]),
        ec("eastish", 3, [0.9, 0.1, 0.0, 0.0]),
    ]

    first = upsert_chunks(items, pool=pool)
    assert first.inserted == 3
    assert count_chunks(repo, pool=pool) == 3

    # Query pointing "east" should rank east first, then eastish, then north.
    hits = search_similar([1.0, 0.0, 0.0, 0.0], top_k=3, repo_url=repo, pool=pool)
    assert [h.name for h in hits] == ["east", "eastish", "north"]
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)

    # Re-upsert the same commit: no new rows (idempotent), counts stable.
    second = upsert_chunks(items, pool=pool)
    assert second.inserted == 0
    assert second.updated == 3
    assert count_chunks(repo, pool=pool) == 3

    # Cleanup.
    with pool.connection() as conn:
        conn.execute("DELETE FROM chunks WHERE repo_url = %s", (repo,))
        conn.commit()
    db_module.close_pool()
