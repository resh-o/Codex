"""Orchestration tests for Stage 4: retrieve -> generate -> validate -> retry.

The LLM is always a scripted fake -- these tests assert on control flow (when a
retry happens, what feedback it gets, what gets stripped), never on model
quality, and they never touch the network. The generator's own parsing is
exercised through :class:`AnswerGenerator` with an injected transport, so the
error paths for malformed responses are covered without an API key either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import pytest

from app.answering.answer_service import AnswerService, NoContextError
from app.answering.generator import AnswerGenerator, GenerationError
from app.answering.models import Answer, Citation, Claim
from app.answering.validator import InvalidReason
from app.config import Settings


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass
class FakeChunk:
    """Stands in for a Stage 3 SearchResult in the retrieved set."""

    id: str
    file_path: str
    start_line: int
    end_line: int
    qualified_name: str = "mod.thing"
    language: str = "python"
    chunk_type: str = "function"
    snippet: str = "def thing():\n    return 1\n"
    score: float = 0.5


AUTH = FakeChunk("c1", "app/auth/session.py", 10, 40, qualified_name="session.refresh")
ROUTES = FakeChunk("c2", "app/api/routes.py", 100, 130, qualified_name="routes.login")
CHUNKS = [AUTH, ROUTES]


class FakeSearch:
    """Async retriever returning a fixed set; records how it was called."""

    def __init__(self, chunks=CHUNKS):
        self._chunks = list(chunks)
        self.calls: list[dict] = []

    async def search(self, query, top_k=10, repo_url=None, mode="hybrid"):
        self.calls.append(
            {"query": query, "top_k": top_k, "repo_url": repo_url, "mode": mode}
        )
        return list(self._chunks)


class ScriptedGenerator:
    """Returns pre-built answers in order; records the prompts' feedback."""

    def __init__(self, *answers, error_on_attempt: Optional[int] = None):
        self._answers = list(answers)
        self._error_on_attempt = error_on_attempt
        self.calls: list[dict] = []

    def generate(self, query, chunks, feedback=None):
        self.calls.append({"query": query, "chunks": chunks, "feedback": feedback})
        attempt = len(self.calls)
        if self._error_on_attempt == attempt:
            raise GenerationError("simulated API failure")
        index = min(attempt - 1, len(self._answers) - 1)
        answer = self._answers[index]
        return Answer(query=query, claims=list(answer.claims), summary=answer.summary)

    @property
    def feedbacks(self) -> list[Optional[str]]:
        return [c["feedback"] for c in self.calls]


def claim(text: str, *citations: tuple[str, int, int]) -> Claim:
    return Claim(
        text=text,
        citations=[
            Citation(file_path=p, start_line=s, end_line=e) for p, s, e in citations
        ],
    )


def answer(*claims: Claim, summary: str | None = None) -> Answer:
    return Answer(query="q", claims=list(claims), summary=summary)


GOOD = answer(
    claim("Sessions refresh on access.", ("app/auth/session.py", 12, 30)),
    claim("The login route calls it.", ("app/api/routes.py", 100, 120)),
    summary="Auth works in two layers:",
)

#: A hallucinated file that was never retrieved -- the adversarial case.
HALLUCINATED = answer(
    claim("Sessions refresh on access.", ("app/auth/session.py", 12, 30)),
    claim("Tokens are signed in the JWT helper.", ("app/auth/jwt_helper.py", 1, 25)),
)


def build(*answers, search=None, error_on_attempt=None):
    generator = ScriptedGenerator(*answers, error_on_attempt=error_on_attempt)
    service = AnswerService(search=search or FakeSearch(), generator=generator)
    return service, generator


# --------------------------------------------------------------------------- #
# Happy path: valid on the first attempt
# --------------------------------------------------------------------------- #


def test_a_fully_valid_answer_is_returned_without_a_retry() -> None:
    service, generator = build(GOOD)
    result = service.answer_question_sync("how does auth work?")

    assert result.attempts == 1
    assert result.retried is False
    assert len(generator.calls) == 1
    assert generator.calls[0]["feedback"] is None
    assert result.validation.is_valid
    assert result.stripped_claims == []
    assert len(result.answer.claims) == 2
    assert result.answer.summary == "Auth works in two layers:"
    assert result.answer.query == "how does auth work?"


def test_the_retrieved_chunks_are_reported_as_sources() -> None:
    service, _ = build(GOOD)
    result = service.answer_question_sync("q")

    assert [s.file_path for s in result.sources] == [
        "app/auth/session.py",
        "app/api/routes.py",
    ]
    assert result.sources[0].chunk_id == "c1"
    assert result.sources[0].qualified_name == "session.refresh"


def test_retrieval_is_hybrid_and_forwards_top_k_and_repo_url() -> None:
    search = FakeSearch()
    service, _ = build(GOOD, search=search)
    service.answer_question_sync("q", repo_url="https://example.com/r", top_k=7)

    assert search.calls == [
        {
            "query": "q",
            "top_k": 7,
            "repo_url": "https://example.com/r",
            "mode": "hybrid",
        }
    ]


def test_the_generator_sees_the_retrieved_chunks_as_context() -> None:
    service, generator = build(GOOD)
    service.answer_question_sync("q")
    assert generator.calls[0]["chunks"] == CHUNKS


# --------------------------------------------------------------------------- #
# Retry: invalid first, valid second
# --------------------------------------------------------------------------- #


def test_an_invalid_citation_triggers_exactly_one_retry_that_can_succeed() -> None:
    service, generator = build(HALLUCINATED, GOOD)
    result = service.answer_question_sync("how does auth work?")

    assert result.attempts == 2
    assert result.retried is True
    assert len(generator.calls) == 2
    assert result.validation.is_valid
    assert result.stripped_claims == []
    assert len(result.answer.claims) == 2


def test_the_retry_is_told_exactly_which_citation_failed_and_why() -> None:
    """Targeted feedback is the point -- a blind re-roll would be a coin flip."""
    service, generator = build(HALLUCINATED, GOOD)
    service.answer_question_sync("q")

    feedback = generator.feedbacks[1]
    assert feedback is not None
    assert "app/auth/jwt_helper.py:1-25" in feedback
    assert InvalidReason.FILE_NOT_IN_RETRIEVED_SET.value in feedback
    # It also restates what *was* citable, so the model has somewhere to go.
    assert "app/auth/session.py" in feedback
    assert "app/api/routes.py" in feedback
    # And it names the claim that has to change.
    assert "Tokens are signed in the JWT helper." in feedback


def test_the_retry_feedback_does_not_mention_the_claims_that_passed() -> None:
    service, generator = build(HALLUCINATED, GOOD)
    service.answer_question_sync("q")
    assert "Sessions refresh on access." not in generator.feedbacks[1]


def test_both_attempts_are_recorded_for_stage_5_scoring() -> None:
    service, _ = build(HALLUCINATED, GOOD)
    result = service.answer_question_sync("q")

    assert len(result.validation_history) == 2
    assert result.validation_history[0].citation_precision == pytest.approx(0.5)
    assert result.validation_history[1].citation_precision == 1.0


# --------------------------------------------------------------------------- #
# Retry, then strip
# --------------------------------------------------------------------------- #


def test_a_claim_still_invalid_after_the_retry_is_stripped_not_shipped() -> None:
    """The acceptance case: the model keeps citing a file nobody retrieved.

    A question with thin context tempts the model into inventing a plausible
    path (`app/auth/jwt_helper.py`). It does so twice, so the answer that ships
    contains only the claim that is actually supported -- and says what it lost.
    """
    service, generator = build(HALLUCINATED, HALLUCINATED)
    result = service.answer_question_sync("where are tokens signed?")

    assert result.attempts == 2
    assert len(generator.calls) == 2, "exactly one retry, never more"

    # The bad claim is gone; the good one survives.
    assert [c.text for c in result.answer.claims] == ["Sessions refresh on access."]
    assert result.claims_stripped == 1
    assert result.all_claims_stripped is False

    stripped = result.stripped_claims[0]
    assert stripped.text == "Tokens are signed in the JWT helper."
    assert stripped.reasons == [InvalidReason.FILE_NOT_IN_RETRIEVED_SET]
    assert "not in the retrieved context" in stripped.details[0]

    # Every citation that survived is one the validator verified.
    assert all(
        c.valid for c in result.validation.claims if c.claim_index == 0
    )


def test_a_bad_line_range_survives_to_the_strip_path_too() -> None:
    out_of_range = answer(
        claim("Refresh happens at the end of the file.", ("app/auth/session.py", 300, 320))
    )
    service, _ = build(out_of_range, out_of_range)
    result = service.answer_question_sync("q")

    assert result.answer.claims == []
    assert result.stripped_claims[0].reasons == [InvalidReason.LINE_RANGE_NOT_IN_CHUNK]
    assert result.all_claims_stripped is True


def test_an_answer_whose_every_claim_fails_comes_back_empty_and_flagged() -> None:
    all_bad = answer(
        claim("A.", ("app/nope.py", 1, 5)),
        claim("B.", ("app/also_nope.py", 1, 5)),
    )
    service, _ = build(all_bad, all_bad)
    result = service.answer_question_sync("q")

    assert result.answer.claims == []
    assert result.claims_stripped == 2
    assert result.all_claims_stripped is True
    assert result.validation.is_valid is False


def test_a_partially_repaired_retry_keeps_what_it_fixed() -> None:
    """The retry fixes one of two bad citations; only the other is stripped."""
    both_bad = answer(
        claim("A.", ("app/nope.py", 1, 5)),
        claim("B.", ("app/other_nope.py", 1, 5)),
    )
    one_fixed = answer(
        claim("A.", ("app/auth/session.py", 12, 20)),
        claim("B.", ("app/other_nope.py", 1, 5)),
    )
    service, _ = build(both_bad, one_fixed)
    result = service.answer_question_sync("q")

    assert [c.text for c in result.answer.claims] == ["A."]
    assert [s.text for s in result.stripped_claims] == ["B."]
    assert result.validation.citation_precision == pytest.approx(0.5)


def test_the_reported_validation_describes_what_the_model_produced() -> None:
    """Scoring data must reflect the raw answer, not the sanitised one.

    If `validation` described only the surviving claims it would always read as
    perfect, and Stage 5's citation-accuracy metric would be meaningless.
    """
    service, _ = build(HALLUCINATED, HALLUCINATED)
    result = service.answer_question_sync("q")

    assert result.validation.total_claims == 2
    assert result.validation.valid_claims == 1
    assert len(result.answer.claims) == 1


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_no_retrieved_chunks_is_an_error_not_an_uncited_answer() -> None:
    service, generator = build(GOOD, search=FakeSearch(chunks=[]))
    with pytest.raises(NoContextError):
        service.answer_question_sync("q")
    assert generator.calls == [], "the model must not be asked to answer from nothing"


def test_a_blank_query_is_rejected() -> None:
    service, _ = build(GOOD)
    with pytest.raises(ValueError):
        service.answer_question_sync("   ")


def test_a_first_attempt_generation_failure_propagates() -> None:
    service, _ = build(GOOD, error_on_attempt=1)
    with pytest.raises(GenerationError):
        service.answer_question_sync("q")


def test_a_failed_retry_falls_back_to_stripping_the_first_attempt() -> None:
    """A transport blip on the retry must not throw away a usable first answer."""
    service, generator = build(HALLUCINATED, GOOD, error_on_attempt=2)
    result = service.answer_question_sync("q")

    assert len(generator.calls) == 2
    assert result.attempts == 1
    assert result.retry_error is not None and "simulated" in result.retry_error
    assert [c.text for c in result.answer.claims] == ["Sessions refresh on access."]
    assert result.claims_stripped == 1


# --------------------------------------------------------------------------- #
# The generator's response handling (no network: transport is injected)
# --------------------------------------------------------------------------- #


def generator_returning(raw: str) -> AnswerGenerator:
    return AnswerGenerator(Settings(), generate_fn=lambda *, system, user: raw)


VALID_PAYLOAD = {
    "summary": "Here is how auth works:",
    "claims": [
        {
            "text": "Sessions refresh on access.",
            "citations": [
                {"file_path": "app/auth/session.py", "start_line": 12, "end_line": 30}
            ],
        }
    ],
}


def test_a_schema_valid_response_becomes_an_answer_bound_to_the_query() -> None:
    gen = generator_returning(json.dumps(VALID_PAYLOAD))
    result = gen.generate("how does auth work?", CHUNKS)

    assert result.query == "how does auth work?"
    assert result.summary == "Here is how auth works:"
    assert result.claims[0].citations[0].file_path == "app/auth/session.py"


def test_a_fenced_json_response_is_still_parsed() -> None:
    gen = generator_returning("```json\n" + json.dumps(VALID_PAYLOAD) + "\n```")
    assert gen.generate("q", CHUNKS).claims


def test_a_claim_with_no_citations_is_rejected_at_parse_time() -> None:
    """Belt and braces: the schema forbids it, and so does parsing.

    A provider that ignored `minItems` would otherwise slip an uncited claim
    past the validator, which only checks citations that exist.
    """
    payload = {"claims": [{"text": "Trust me.", "citations": []}]}
    gen = generator_returning(json.dumps(payload))

    with pytest.raises(GenerationError, match="did not match the Answer schema"):
        gen.generate("q", CHUNKS)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("", "empty response"),
        ("   ", "empty response"),
        ("not json at all", "malformed JSON"),
        ('["a", "b"]', "not an answer object"),
        ('{"claims": [{"text": "x"}]}', "did not match the Answer schema"),
        ('{"claims": [{"citations": []}]}', "did not match the Answer schema"),
    ],
)
def test_bad_responses_raise_a_clear_error_rather_than_a_broken_answer(raw, match) -> None:
    with pytest.raises(GenerationError, match=match):
        generator_returning(raw).generate("q", CHUNKS)


def test_a_transport_exception_is_wrapped_as_a_generation_error() -> None:
    def explode(*, system, user):
        raise RuntimeError("connection reset")

    gen = AnswerGenerator(Settings(), generate_fn=explode)
    with pytest.raises(GenerationError, match="connection reset"):
        gen.generate("q", CHUNKS)


def test_the_schema_sent_to_the_api_makes_an_uncited_claim_impossible() -> None:
    """The guarantee is structural, so check the wire schema, not the prompt.

    `citations` must arrive at Gemini as a *required* array with `min_items: 1`.
    If a refactor ever drops the `min_length=1`, the model becomes free to emit
    uncited claims and the enforcement quietly downgrades to a polite request in
    the system prompt -- which is exactly what this stage is not.
    """
    genai_transformers = pytest.importorskip("google.genai._transformers")
    from app.answering.models import GeneratedAnswer

    schema = genai_transformers.t_schema(None, GeneratedAnswer).model_dump(
        exclude_none=True
    )
    claim_schema = schema["properties"]["claims"]["items"]

    assert "citations" in claim_schema["required"]
    assert claim_schema["properties"]["citations"]["min_items"] == 1
    citation_schema = claim_schema["properties"]["citations"]["items"]
    assert set(citation_schema["required"]) == {"file_path", "start_line", "end_line"}


def test_the_prompt_hands_the_model_exact_values_to_cite() -> None:
    captured: dict[str, str] = {}

    def capture(*, system, user):
        captured.update(system=system, user=user)
        return json.dumps(VALID_PAYLOAD)

    AnswerGenerator(Settings(), generate_fn=capture).generate("how does auth work?", CHUNKS)

    assert "app/auth/session.py" in captured["user"]
    assert "10-40" in captured["user"]
    assert "how does auth work?" in captured["user"]
    assert "ONLY the code excerpts" in captured["system"]


# --------------------------------------------------------------------------- #
# The /ask endpoint
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    def _install(service):
        monkeypatch.setattr(main, "_get_answer_service", lambda: service)
        monkeypatch.setattr(main, "_ensure_storage_ready", lambda: None)
        return TestClient(main.app)

    return _install


def test_ask_returns_the_answer_with_its_citations(client) -> None:
    service, _ = build(GOOD)
    body = client(service).post("/ask", json={"query": "how does auth work?"}).json()

    assert body["attempts"] == 1
    assert body["retried"] is False
    assert body["answer"]["summary"] == "Auth works in two layers:"
    claims = body["answer"]["claims"]
    assert len(claims) == 2
    assert all(c["citations"] for c in claims), "every claim carries a citation"
    assert claims[0]["citations"][0] == {
        "file_path": "app/auth/session.py",
        "start_line": 12,
        "end_line": 30,
    }


def test_ask_reports_validation_metadata_for_scoring(client) -> None:
    service, _ = build(GOOD)
    body = client(service).post("/ask", json={"query": "q"}).json()

    validation = body["validation"]
    assert validation["is_valid"] is True
    assert validation["citation_precision"] == 1.0
    assert validation["claim_support_rate"] == 1.0
    assert validation["retrieved_files"] == [
        "app/api/routes.py",
        "app/auth/session.py",
    ]
    assert len(body["validation_history"]) == 1
    assert len(body["sources"]) == 2


def test_ask_reports_stripped_claims_and_the_retry(client) -> None:
    service, _ = build(HALLUCINATED, HALLUCINATED)
    body = client(service).post("/ask", json={"query": "where are tokens signed?"}).json()

    assert body["attempts"] == 2 and body["retried"] is True
    assert body["claims_stripped"] == 1
    assert body["all_claims_stripped"] is False
    stripped = body["stripped_claims"][0]
    assert stripped["text"] == "Tokens are signed in the JWT helper."
    assert stripped["reasons"] == ["file_not_in_retrieved_set"]
    # The surviving answer contains no trace of the bad citation.
    cited_files = {
        c["file_path"] for cl in body["answer"]["claims"] for c in cl["citations"]
    }
    assert "app/auth/jwt_helper.py" not in cited_files


def test_ask_forwards_top_k_and_repo_url(client) -> None:
    search = FakeSearch()
    service, _ = build(GOOD, search=search)
    client(service).post(
        "/ask", json={"query": "q", "repo_url": "https://example.com/r", "top_k": 3}
    )
    assert search.calls[0]["top_k"] == 3
    assert search.calls[0]["repo_url"] == "https://example.com/r"


def test_ask_returns_404_when_nothing_relevant_was_retrieved(client) -> None:
    service, _ = build(GOOD, search=FakeSearch(chunks=[]))
    response = client(service).post("/ask", json={"query": "q"})

    assert response.status_code == 404
    assert "nothing to cite" in response.json()["detail"]


def test_ask_returns_502_when_the_model_call_fails(client) -> None:
    service, _ = build(GOOD, error_on_attempt=1)
    response = client(service).post("/ask", json={"query": "q"})

    assert response.status_code == 502
    assert "Answer generation failed" in response.json()["detail"]


def test_ask_rejects_a_blank_query(client) -> None:
    service, _ = build(GOOD)
    assert client(service).post("/ask", json={"query": "   "}).status_code == 400


def test_ask_reports_missing_credentials_as_503(client) -> None:
    from app.config import ConfigError

    class Exploding:
        async def search(self, query, top_k=10, repo_url=None, mode="hybrid"):
            raise ConfigError("GEMINI_API_KEY is not set.")

    service = AnswerService(search=Exploding(), generator=ScriptedGenerator(GOOD))
    response = client(service).post("/ask", json={"query": "q"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
