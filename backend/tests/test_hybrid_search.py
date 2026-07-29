"""Tests for hybrid retrieval (vector + keyword, RRF-fused).

Layers:
* Always-on: orchestration against fake retrievers -- fusion ordering, the
  three modes, overfetch, concurrency, and the headline case where hybrid
  rescues an exact-identifier match that vector-only buries.
* Opt-in: a real round-trip through Postgres (pgvector `<=>` + full-text) with
  only the *embedder* faked, enabled when TEST_DATABASE_URL is set.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from app.search.hybrid_search import DEFAULT_OVERFETCH_FACTOR, HybridSearchService
from app.search.results import SearchResult


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def result(cid: str, score: float, *, keyword: bool = False) -> SearchResult:
    """A SearchResult as one of the two retrievers would emit it."""
    return SearchResult(
        id=cid,
        repo_url="https://example.com/r",
        file_path=f"{cid}.py",
        language="python",
        chunk_type="function",
        name=cid,
        qualified_name=cid,
        start_line=1,
        end_line=5,
        snippet=f"def {cid}(): ...",
        score=score,
        similarity=None if keyword else score,
        keyword_score=score if keyword else None,
    )


class FakeRetriever:
    """Returns a fixed ranked list; records the calls it received."""

    def __init__(self, ids, *, keyword: bool = False, delay: float = 0.0):
        self._ids = list(ids)
        self._keyword = keyword
        self._delay = delay
        self.calls: list[dict] = []
        self.intervals: list[tuple[float, float]] = []

    def search(self, query: str, top_k: int = 10, repo_url=None):
        started = time.monotonic()
        self.calls.append({"query": query, "top_k": top_k, "repo_url": repo_url})
        if self._delay:
            time.sleep(self._delay)
        self.intervals.append((started, time.monotonic()))
        # Descending synthetic scores so ordering is unambiguous.
        out = [
            result(cid, 1.0 - i / 100, keyword=self._keyword)
            for i, cid in enumerate(self._ids)
        ]
        return out[:top_k]


def build(vector_ids, keyword_ids, **kwargs) -> HybridSearchService:
    return HybridSearchService(
        vector=FakeRetriever(vector_ids),
        keyword=FakeRetriever(keyword_ids, keyword=True),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# The headline: hybrid rescues an exact-identifier match
# --------------------------------------------------------------------------- #


def test_hybrid_beats_vector_only_on_an_exact_identifier_query() -> None:
    """A rare identifier that embeddings under-rank must be recovered by fusion.

    Setup mirrors the real failure mode: the query is a rare function name,
    `parse_frontmatter`. Vector search drifts semantically -- it surfaces eight
    other parsing/markdown-ish chunks first and puts the actual function 9th.
    Keyword search matches the identifier exactly and puts it 1st.
    """
    target = "parse_frontmatter"
    decoys = [f"decoy_{i}" for i in range(8)]

    vector_ranking = decoys + [target]  # target is 9th
    # Keyword nails the identifier; the runner-up is a chunk that merely
    # mentions it, which vector search never surfaced at all.
    keyword_ranking = [target, "mentions_target"]

    service = build(vector_ranking, keyword_ranking)
    top_k = 5

    # Vector-only: the target is ranked 9th, so it doesn't even make top 5.
    vector_only = service.search_sync(target, top_k=top_k, mode="vector")
    assert [r.id for r in vector_only] == decoys[:5]
    assert target not in [r.id for r in vector_only]

    # Hybrid: fusion pulls it to the very top.
    hybrid = service.search_sync(target, top_k=top_k, mode="hybrid")
    ids = [r.id for r in hybrid]
    assert ids[0] == target, f"expected {target} first, got {ids}"

    # And the diagnostics explain *why* it won: 9th by vector, 1st by keyword.
    top = hybrid[0]
    assert top.vector_rank == 9
    assert top.keyword_rank == 1
    assert top.fused_score == pytest.approx(1 / (60 + 9) + 1 / (60 + 1))

    # Guard the mechanism, not just the outcome: without overfetching, the
    # vector list would have been truncated at 5 and the target lost.
    assert 9 <= top_k * DEFAULT_OVERFETCH_FACTOR


def test_overfetch_is_what_makes_the_rescue_possible() -> None:
    """Without overfetch, the vector list is cut off before the target.

    Same inputs as the test above, changing only `overfetch_factor`. At the
    default (3) the vector side sees 15 candidates and contributes the target's
    9th-place rank. At 1 it sees only 5, the target falls off the end, and the
    fused result loses that corroborating signal entirely -- which is the whole
    argument for overfetching before fusion.
    """
    target = "parse_frontmatter"
    decoys = [f"decoy_{i}" for i in range(8)]
    lists = (decoys + [target], [target, "mentions_target"])

    def target_hit(factor: int):
        service = build(*lists, overfetch_factor=factor)
        results = service.search_sync(target, top_k=5, mode="hybrid")
        return next(r for r in results if r.id == target)

    with_overfetch = target_hit(DEFAULT_OVERFETCH_FACTOR)
    assert with_overfetch.vector_rank == 9
    assert with_overfetch.keyword_rank == 1

    without_overfetch = target_hit(1)
    assert without_overfetch.vector_rank is None, "vector list should be truncated"
    assert without_overfetch.keyword_rank == 1
    assert without_overfetch.fused_score < with_overfetch.fused_score


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def test_hybrid_mode_fuses_both_retrievers() -> None:
    service = build(["A", "B", "C"], ["B", "D", "A"])
    results = service.search_sync("q", top_k=10, mode="hybrid")

    # Same expected ordering as the hand-computed case in test_fusion.py.
    assert [r.id for r in results] == ["B", "A", "D", "C"]


def test_vector_mode_returns_the_vector_ranking_unfused() -> None:
    service = build(["A", "B", "C"], ["Z", "Y"])
    results = service.search_sync("q", top_k=10, mode="vector")

    assert [r.id for r in results] == ["A", "B", "C"]
    assert all(r.fused_score is None for r in results)
    assert all(r.similarity is not None for r in results)


def test_keyword_mode_returns_the_keyword_ranking_unfused() -> None:
    service = build(["A", "B", "C"], ["Z", "Y"])
    results = service.search_sync("q", top_k=10, mode="keyword")

    assert [r.id for r in results] == ["Z", "Y"]
    assert all(r.fused_score is None for r in results)
    assert all(r.keyword_score is not None for r in results)


def test_single_mode_calls_only_its_own_retriever() -> None:
    vector = FakeRetriever(["A"])
    keyword = FakeRetriever(["Z"], keyword=True)
    service = HybridSearchService(vector=vector, keyword=keyword)

    service.search_sync("q", mode="vector")
    assert len(vector.calls) == 1 and keyword.calls == []

    service.search_sync("q", mode="keyword")
    assert len(vector.calls) == 1 and len(keyword.calls) == 1


def test_unknown_mode_is_rejected() -> None:
    service = build(["A"], ["A"])
    with pytest.raises(ValueError, match="unknown mode"):
        service.search_sync("q", mode="bm25")


def test_empty_query_is_rejected() -> None:
    service = build(["A"], ["A"])
    with pytest.raises(ValueError):
        service.search_sync("   ")


# --------------------------------------------------------------------------- #
# Overfetch + top_k
# --------------------------------------------------------------------------- #


def test_each_retriever_is_asked_for_top_k_times_the_overfetch_factor() -> None:
    vector = FakeRetriever([f"v{i}" for i in range(50)])
    keyword = FakeRetriever([f"k{i}" for i in range(50)], keyword=True)
    service = HybridSearchService(vector=vector, keyword=keyword, overfetch_factor=3)

    service.search_sync("q", top_k=10, mode="hybrid")
    assert vector.calls[0]["top_k"] == 30
    assert keyword.calls[0]["top_k"] == 30


def test_single_modes_do_not_overfetch() -> None:
    vector = FakeRetriever([f"v{i}" for i in range(50)])
    keyword = FakeRetriever([f"k{i}" for i in range(50)], keyword=True)
    service = HybridSearchService(vector=vector, keyword=keyword, overfetch_factor=3)

    service.search_sync("q", top_k=10, mode="vector")
    assert vector.calls[0]["top_k"] == 10


def test_fused_output_is_truncated_to_top_k() -> None:
    service = build([f"v{i}" for i in range(30)], [f"k{i}" for i in range(30)])
    assert len(service.search_sync("q", top_k=4, mode="hybrid")) == 4


def test_repo_filter_is_forwarded_to_both_retrievers() -> None:
    vector = FakeRetriever(["A"])
    keyword = FakeRetriever(["A"], keyword=True)
    service = HybridSearchService(vector=vector, keyword=keyword)

    service.search_sync("q", repo_url="https://example.com/only", mode="hybrid")
    assert vector.calls[0]["repo_url"] == "https://example.com/only"
    assert keyword.calls[0]["repo_url"] == "https://example.com/only"


def test_overfetch_factor_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        HybridSearchService(
            vector=FakeRetriever([]), keyword=FakeRetriever([]), overfetch_factor=0
        )


def test_rrf_k_is_configurable() -> None:
    service = build(["A", "B"], ["B", "A"], rrf_k=1)
    (first, second) = service.search_sync("q", top_k=2, mode="hybrid")
    # Both hold one 1st and one 2nd, so with any k they tie...
    assert first.fused_score == pytest.approx(second.fused_score)
    # ...but at k=1 the scores are 1/2 + 1/3, not 1/61 + 1/62.
    assert first.fused_score == pytest.approx(1 / 2 + 1 / 3)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_retrievers_run_concurrently_not_sequentially() -> None:
    """Assert the two searches actually overlap in time, not just that it's fast."""
    delay = 0.25
    vector = FakeRetriever(["A"], delay=delay)
    keyword = FakeRetriever(["B"], keyword=True, delay=delay)
    service = HybridSearchService(vector=vector, keyword=keyword)

    started = time.monotonic()
    service.search_sync("q", mode="hybrid")
    elapsed = time.monotonic() - started

    # Wall clock: concurrent means ~1 delay, sequential would be ~2.
    assert elapsed < delay * 1.8, f"took {elapsed:.3f}s; looks sequential"

    # Stronger and timing-noise-proof: the two intervals genuinely overlap.
    (v_start, v_end), (k_start, k_end) = vector.intervals[0], keyword.intervals[0]
    overlap = min(v_end, k_end) - max(v_start, k_start)
    assert overlap > 0, "vector and keyword searches did not overlap"


def test_diagnostics_are_populated_for_every_fused_result() -> None:
    service = build(["A", "B", "C"], ["C", "D"])
    results = service.search_sync("q", top_k=10, mode="hybrid")

    for r in results:
        assert r.fused_score is not None
        assert r.score == r.fused_score
        assert r.vector_rank is not None or r.keyword_rank is not None
        # A result found by only one side reports None for the other's score.
        if r.vector_rank is None:
            assert r.similarity is None
        if r.keyword_rank is None:
            assert r.keyword_score is None

    by_id = {r.id: r for r in results}
    assert by_id["C"].vector_rank == 3 and by_id["C"].keyword_rank == 1
    assert by_id["D"].vector_rank is None and by_id["D"].keyword_rank == 2
    assert by_id["A"].keyword_rank is None and by_id["A"].vector_rank == 1


def test_results_from_only_one_retriever_are_not_dropped() -> None:
    service = build(["A", "B"], ["C", "D"])
    ids = {r.id for r in service.search_sync("q", top_k=10, mode="hybrid")}
    assert ids == {"A", "B", "C", "D"}


# --------------------------------------------------------------------------- #
# The /search endpoint
#
# Storage and both retrievers are stubbed, so these test the API contract --
# default mode, mode switching, diagnostics in the payload -- not retrieval.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    service = build(["A", "B", "C"], ["C", "D"])
    monkeypatch.setattr(main, "_get_hybrid_service", lambda: service)
    monkeypatch.setattr(main, "_ensure_storage_ready", lambda: None)
    return TestClient(main.app), service


def test_search_endpoint_defaults_to_hybrid(client) -> None:
    api, _ = client
    response = api.post("/search", json={"query": "parse config", "top_k": 10})
    assert response.status_code == 200

    body = response.json()
    assert body["mode"] == "hybrid"
    assert body["query"] == "parse config"
    assert body["count"] == len(body["results"]) == 4
    # Fused ordering, not either retriever's own.
    assert [r["id"] for r in body["results"]] == ["C", "A", "B", "D"]


def test_search_endpoint_exposes_fusion_diagnostics(client) -> None:
    api, _ = client
    body = api.post("/search", json={"query": "q", "top_k": 10}).json()
    top = body["results"][0]

    assert top["id"] == "C"
    assert top["vector_rank"] == 3
    assert top["keyword_rank"] == 1
    assert top["fused_score"] == pytest.approx(1 / 63 + 1 / 61)
    assert top["score"] == pytest.approx(top["fused_score"])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("vector", ["A", "B", "C"]), ("keyword", ["C", "D"])],
)
def test_search_endpoint_single_modes_still_work(client, mode, expected) -> None:
    api, _ = client
    body = api.post("/search", json={"query": "q", "mode": mode}).json()

    assert body["mode"] == mode
    assert [r["id"] for r in body["results"]] == expected
    assert all(r["fused_score"] is None for r in body["results"])


def test_search_endpoint_rejects_an_unknown_mode(client) -> None:
    api, _ = client
    response = api.post("/search", json={"query": "q", "mode": "bm25"})
    assert response.status_code == 422  # rejected by the Literal, before the service


def test_search_endpoint_rejects_a_blank_query(client) -> None:
    api, _ = client
    assert api.post("/search", json={"query": "   "}).status_code == 400


def test_search_endpoint_reports_missing_credentials_as_503(monkeypatch) -> None:
    """A missing Gemini key must be a clean 503, not a traceback."""
    from fastapi.testclient import TestClient

    from app import main
    from app.config import ConfigError

    class Exploding:
        def search(self, query, top_k=10, repo_url=None):
            raise ConfigError("GEMINI_API_KEY is not set.")

    service = HybridSearchService(vector=Exploding(), keyword=FakeRetriever([]))
    monkeypatch.setattr(main, "_get_hybrid_service", lambda: service)
    monkeypatch.setattr(main, "_ensure_storage_ready", lambda: None)

    response = TestClient(main.app).post("/search", json={"query": "q"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Optional real-DB integration
# --------------------------------------------------------------------------- #

TEST_DB = os.environ.get("TEST_DATABASE_URL")
requires_db = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set; skipping real-DB hybrid tests"
)


class FakeEmbedder:
    """Maps queries/chunks to preset vectors so no Gemini call is made."""

    def __init__(self, query_vector):
        self.query_vector = query_vector

    def embed_query(self, query: str):
        return self.query_vector


def _vec(*head: float) -> list[float]:
    """Pad a synthetic vector to the production schema's 768 dims."""
    return list(head) + [0.0] * (768 - len(head))


@pytest.fixture()
def live_service():
    """Hybrid service wired to a real DB, with only the embedder faked."""
    os.environ["DATABASE_URL"] = TEST_DB
    from app.config import Settings
    from app.embeddings.embedder import EmbeddedChunk
    from app.ingestion.models import Chunk
    from app.search.keyword_search import KeywordSearchService
    from app.search.vector_search import VectorSearchService
    from app.storage import db as db_module
    from app.storage.repository import search_keyword, search_similar, upsert_chunks

    db_module.close_pool()
    settings = Settings(database_url=TEST_DB)
    pool = db_module.get_pool(settings)
    db_module.apply_schema(settings)

    repo = f"https://example.com/hybrid-{uuid.uuid4().hex[:8]}"

    # `parse_frontmatter` is the rare identifier. Its embedding is deliberately
    # *far* from the query vector, while several decoys sit close to it -- the
    # semantic-drift failure that keyword search is meant to fix.
    rows = [
        ("parse_frontmatter", "md.parse_frontmatter", "Split YAML front matter.",
         "def parse_frontmatter(text):\n    return yaml.safe_load(text)\n",
         _vec(0.0, 1.0)),
        ("read_markdown", "md.read_markdown", "Read a markdown document.",
         "def read_markdown(path):\n    return path.read_text()\n", _vec(1.0, 0.0)),
        ("render_markdown", "md.render_markdown", "Render markdown to HTML.",
         "def render_markdown(doc):\n    return html(doc)\n", _vec(0.98, 0.02)),
        ("parse_headers", "md.parse_headers", "Extract headers from a document.",
         "def parse_headers(doc):\n    return doc.headers\n", _vec(0.95, 0.05)),
        ("load_document", "md.load_document", "Load and parse a document.",
         "def load_document(path):\n    return Document(path)\n", _vec(0.9, 0.1)),
    ]
    upsert_chunks(
        [
            EmbeddedChunk(
                chunk=Chunk(
                    repo_url=repo, commit_sha="c0ffee", file_path=f"{name}.py",
                    language="python", chunk_type="function", name=name,
                    qualified_name=qualified, parent_name=None,
                    start_line=i * 10 + 1, end_line=i * 10 + 5,
                    content=content, docstring=doc,
                ),
                embedding=vector,
            )
            for i, (name, qualified, doc, content, vector) in enumerate(rows)
        ],
        pool=pool,
    )

    # Query vector points at the "markdown document" cluster, away from the
    # target -- so vector search ranks parse_frontmatter last.
    service = HybridSearchService(
        vector=VectorSearchService(
            FakeEmbedder(_vec(1.0, 0.0)),
            search_fn=lambda emb, top_k=10, repo_url=None: search_similar(
                emb, top_k=top_k, repo_url=repo_url, pool=pool
            ),
        ),
        keyword=KeywordSearchService(
            search_fn=lambda q, top_k=10, repo_url=None: search_keyword(
                q, top_k=top_k, repo_url=repo_url, pool=pool
            )
        ),
    )
    try:
        yield service, repo
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE repo_url = %s", (repo,))
            conn.commit()
        db_module.close_pool()


@requires_db
def test_live_all_three_modes_return_results(live_service) -> None:
    service, repo = live_service
    for mode in ("hybrid", "vector", "keyword"):
        results = service.search_sync(
            "parse_frontmatter", top_k=5, repo_url=repo, mode=mode
        )
        assert results, f"mode={mode} returned nothing"
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), f"mode={mode} unsorted"


@requires_db
def test_live_hybrid_recovers_what_vector_only_buries(live_service) -> None:
    """The Stage 3 headline claim, end-to-end against real Postgres."""
    service, repo = live_service
    target = "parse_frontmatter"

    vector_only = service.search_sync(target, top_k=5, repo_url=repo, mode="vector")
    vector_rank = [r.name for r in vector_only].index(target) + 1

    keyword_only = service.search_sync(target, top_k=5, repo_url=repo, mode="keyword")
    assert keyword_only[0].name == target

    hybrid = service.search_sync(target, top_k=5, repo_url=repo, mode="hybrid")
    hybrid_rank = [r.name for r in hybrid].index(target) + 1

    assert vector_rank > 1, "test setup is wrong: vector should under-rank the target"
    assert hybrid_rank == 1
    assert hybrid_rank < vector_rank

    top = hybrid[0]
    assert top.keyword_rank == 1
    assert top.vector_rank == vector_rank
    assert top.fused_score is not None


@requires_db
def test_live_hybrid_diagnostics_round_trip(live_service) -> None:
    service, repo = live_service
    results = service.search_sync(
        "how is markdown rendered?", top_k=5, repo_url=repo, mode="hybrid"
    )
    assert results
    for r in results:
        assert r.fused_score is not None
        if r.vector_rank is not None:
            assert r.similarity is not None
        if r.keyword_rank is not None:
            assert r.keyword_score is not None
