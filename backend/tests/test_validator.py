"""Unit tests for citation validation -- Stage 4's core guarantee.

Pure and offline: fake retrieved chunks, fake claims, no LLM, no database. Each
invalid category from the spec gets its own test, plus the aggregate rates Stage
5 will score on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from app.answering.models import Answer, Citation, Claim
from app.answering.validator import (
    InvalidReason,
    normalize_path,
    validate_answer,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@dataclass
class FakeChunk:
    """The three fields validation reads, plus an id it echoes back."""

    file_path: str
    start_line: int
    end_line: int
    id: Optional[str] = None


AUTH = FakeChunk("app/auth/session.py", 10, 40, id="chunk-auth")
ROUTES = FakeChunk("app/api/routes.py", 100, 130, id="chunk-routes")
# A second chunk from the same file as AUTH, deliberately non-adjacent: the gap
# at 41-79 is what "the union is not a chunk" tests exercise.
AUTH_TAIL = FakeChunk("app/auth/session.py", 80, 95, id="chunk-auth-tail")

CHUNKS = [AUTH, ROUTES, AUTH_TAIL]


def claim(text: str, *citations: tuple[str, int, int]) -> Claim:
    return Claim(
        text=text,
        citations=[
            Citation(file_path=path, start_line=start, end_line=end)
            for path, start, end in citations
        ],
    )


def answer(*claims: Claim) -> Answer:
    return Answer(query="how does auth work?", claims=list(claims))


# --------------------------------------------------------------------------- #
# Valid citations
# --------------------------------------------------------------------------- #


def test_a_citation_matching_a_chunk_exactly_is_valid() -> None:
    result = validate_answer(
        answer(claim("Sessions are refreshed here.", ("app/auth/session.py", 10, 40))),
        CHUNKS,
    )

    assert result.is_valid
    assert result.valid_claims == 1
    citation = result.claims[0].citations[0]
    assert citation.valid
    assert citation.reason is None
    # The verdict says *which* chunk vouched for it, not merely that one did.
    assert citation.matched_chunk_id == "chunk-auth"


def test_a_narrower_range_inside_a_chunk_is_valid() -> None:
    """Citing a few lines of a chunk is the common, correct case."""
    result = validate_answer(
        answer(claim("The token is decoded here.", ("app/auth/session.py", 18, 22))),
        CHUNKS,
    )
    assert result.is_valid


def test_a_single_line_citation_is_valid() -> None:
    result = validate_answer(
        answer(claim("One line.", ("app/api/routes.py", 100, 100))), CHUNKS
    )
    assert result.is_valid


def test_a_claim_with_several_valid_citations_is_valid() -> None:
    result = validate_answer(
        answer(
            claim(
                "The route calls the session manager.",
                ("app/api/routes.py", 100, 110),
                ("app/auth/session.py", 12, 20),
            )
        ),
        CHUNKS,
    )
    assert result.is_valid
    assert result.total_citations == 2 and result.valid_citations == 2


def test_a_citation_matching_the_second_chunk_of_the_same_file_is_valid() -> None:
    """File lookup considers every chunk from that file, not just the first."""
    result = validate_answer(
        answer(claim("Cleanup happens later in the file.", ("app/auth/session.py", 82, 90))),
        CHUNKS,
    )
    assert result.is_valid
    assert result.claims[0].citations[0].matched_chunk_id == "chunk-auth-tail"


# --------------------------------------------------------------------------- #
# Invalid category 1: the file was never retrieved
# --------------------------------------------------------------------------- #


def test_a_file_not_in_the_retrieved_set_is_rejected() -> None:
    """The classic hallucination: a plausible path nobody retrieved."""
    result = validate_answer(
        answer(claim("Auth uses middleware.", ("app/auth/middleware.py", 1, 20))),
        CHUNKS,
    )

    assert not result.is_valid
    citation = result.claims[0].citations[0]
    assert citation.reason is InvalidReason.FILE_NOT_IN_RETRIEVED_SET
    assert citation.matched_chunk_id is None
    assert result.invalid_claim_indices == {0}


def test_a_file_that_exists_in_the_repo_but_was_not_retrieved_is_still_rejected() -> None:
    """Validation is against *this question's* evidence, not the repo at large.

    This is the distinction the whole stage turns on: `app/config.py` is a real
    file, and a citation to it would pass any "does this path exist?" check --
    but nothing was retrieved from it, so the model cannot have read it.
    """
    retrieved = [AUTH]
    result = validate_answer(
        answer(claim("Settings are read from env.", ("app/config.py", 50, 60))),
        retrieved,
    )
    assert result.claims[0].citations[0].reason is InvalidReason.FILE_NOT_IN_RETRIEVED_SET


def test_the_error_detail_names_what_was_actually_available() -> None:
    result = validate_answer(
        answer(claim("Nope.", ("app/auth/middleware.py", 1, 20))), CHUNKS
    )
    assert result.retrieved_files == ["app/api/routes.py", "app/auth/session.py"]
    assert "not in the retrieved context" in result.claims[0].citations[0].detail


# --------------------------------------------------------------------------- #
# Invalid category 2: right file, wrong lines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("start", "end", "why"),
    [
        (200, 220, "entirely past the chunk"),
        (1, 5, "entirely before the chunk"),
        (5, 20, "overhangs the start"),
        (35, 60, "overhangs the end"),
        (10, 95, "spans a gap between two chunks of the same file"),
        (41, 50, "lands in the gap between two chunks"),
    ],
)
def test_line_ranges_not_contained_in_a_single_chunk_are_rejected(start, end, why) -> None:
    """Containment in one chunk is the rule; overlap and unions do not count.

    The overhang cases are the ones that matter most: those lines were never in
    the model's context, so a citation covering them points at something the
    reader was not shown -- "nearly right" is still fabricated.
    """
    result = validate_answer(
        answer(claim("Something.", ("app/auth/session.py", start, end))), CHUNKS
    )

    assert not result.is_valid, why
    assert result.claims[0].citations[0].reason is InvalidReason.LINE_RANGE_NOT_IN_CHUNK


def test_the_line_range_error_lists_the_ranges_that_were_available() -> None:
    result = validate_answer(
        answer(claim("Something.", ("app/auth/session.py", 200, 220))), CHUNKS
    )
    detail = result.claims[0].citations[0].detail
    assert "10-40" in detail and "80-95" in detail


# --------------------------------------------------------------------------- #
# Invalid category 3: the range is malformed
# --------------------------------------------------------------------------- #


def test_a_reversed_line_range_is_rejected_as_malformed() -> None:
    """start > end is caught as malformed, not as a containment failure."""
    result = validate_answer(
        answer(claim("Backwards.", ("app/auth/session.py", 40, 10))), CHUNKS
    )

    assert not result.is_valid
    assert result.claims[0].citations[0].reason is InvalidReason.LINE_RANGE_INVALID


def test_a_malformed_range_is_rejected_before_the_file_is_even_looked_up() -> None:
    """Ordering matters: a nonsense range is nonsense regardless of the file."""
    result = validate_answer(
        answer(claim("Backwards and absent.", ("no/such/file.py", 40, 10))), CHUNKS
    )
    assert result.claims[0].citations[0].reason is InvalidReason.LINE_RANGE_INVALID


def test_a_zero_or_negative_start_line_cannot_even_be_constructed() -> None:
    """Line numbers are 1-based, enforced by the model itself."""
    with pytest.raises(Exception):
        Citation(file_path="app/auth/session.py", start_line=0, end_line=10)


# --------------------------------------------------------------------------- #
# Claim-level semantics
# --------------------------------------------------------------------------- #


def test_one_bad_citation_invalidates_the_whole_claim() -> None:
    """A claim is only as trustworthy as its worst citation."""
    result = validate_answer(
        answer(
            claim(
                "Auth is layered.",
                ("app/auth/session.py", 12, 20),  # good
                ("app/auth/middleware.py", 1, 9),  # fabricated
            )
        ),
        CHUNKS,
    )

    assert not result.is_valid
    assert not result.claims[0].valid
    assert result.valid_citations == 1 and result.total_citations == 2
    assert result.claims[0].reasons == [InvalidReason.FILE_NOT_IN_RETRIEVED_SET]


def test_valid_and_invalid_claims_are_reported_independently() -> None:
    result = validate_answer(
        answer(
            claim("Good one.", ("app/auth/session.py", 12, 20)),
            claim("Bad one.", ("app/nope.py", 1, 5)),
            claim("Another good one.", ("app/api/routes.py", 105, 115)),
        ),
        CHUNKS,
    )

    assert [c.valid for c in result.claims] == [True, False, True]
    assert result.invalid_claim_indices == {1}
    assert result.valid_claims == 2 and result.total_claims == 3


def test_a_claim_cannot_be_constructed_without_a_citation() -> None:
    """The citation requirement is structural, so the validator never sees this."""
    with pytest.raises(Exception):
        Claim(text="Unsupported assertion.", citations=[])


def test_an_answer_with_no_claims_validates_vacuously() -> None:
    result = validate_answer(Answer(query="q", claims=[]), CHUNKS)
    assert result.is_valid
    assert result.total_claims == 0 and result.total_citations == 0


# --------------------------------------------------------------------------- #
# Scoring output for Stage 5
# --------------------------------------------------------------------------- #


def test_aggregate_rates_are_computed_for_scoring() -> None:
    result = validate_answer(
        answer(
            claim("A.", ("app/auth/session.py", 12, 20)),
            claim("B.", ("app/nope.py", 1, 5), ("app/api/routes.py", 100, 101)),
            claim("C.", ("app/api/routes.py", 105, 115)),
        ),
        CHUNKS,
    )

    assert result.total_citations == 4
    assert result.valid_citations == 3
    assert result.citation_precision == pytest.approx(3 / 4)
    assert result.claim_support_rate == pytest.approx(2 / 3)


def test_an_empty_answer_scores_one_but_is_distinguishable_from_a_perfect_one() -> None:
    """Precision of 1.0 with zero citations must not read as a great answer."""
    empty = validate_answer(Answer(query="q", claims=[]), CHUNKS)
    good = validate_answer(
        answer(claim("A.", ("app/auth/session.py", 12, 20))), CHUNKS
    )

    assert empty.citation_precision == good.citation_precision == 1.0
    assert empty.total_citations == 0 and good.total_citations == 1


def test_the_result_serializes_with_its_metrics_for_downstream_scoring() -> None:
    """Stage 5 reads this as data, so the rollups must survive dumping."""
    result = validate_answer(
        answer(claim("A.", ("app/nope.py", 1, 5))), CHUNKS
    )
    payload = result.model_dump()

    assert payload["is_valid"] is False
    assert payload["citation_precision"] == 0.0
    assert payload["total_claims"] == 1
    claim_payload = payload["claims"][0]
    assert claim_payload["valid"] is False
    assert claim_payload["reasons"] == [InvalidReason.FILE_NOT_IN_RETRIEVED_SET]
    assert claim_payload["citations"][0]["citation"]["file_path"] == "app/nope.py"


# --------------------------------------------------------------------------- #
# Path normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cited",
    ["app/auth/session.py", "./app/auth/session.py", "app\\auth\\session.py", "  app/auth/session.py  "],
)
def test_paths_differing_only_in_notation_still_match(cited) -> None:
    result = validate_answer(answer(claim("A.", (cited, 12, 20))), CHUNKS)
    assert result.is_valid, f"{cited!r} should match the retrieved chunk"


def test_case_differences_are_not_normalized_away() -> None:
    """Case-folding paths would validate citations to files that do not exist."""
    result = validate_answer(
        answer(claim("A.", ("app/auth/Session.py", 12, 20))), CHUNKS
    )
    assert result.claims[0].citations[0].reason is InvalidReason.FILE_NOT_IN_RETRIEVED_SET


def test_normalize_path_leaves_parent_traversal_alone() -> None:
    assert normalize_path("../src/a.py") == "../src/a.py"
    assert normalize_path("././a.py") == "a.py"
