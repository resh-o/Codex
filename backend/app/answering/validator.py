"""Mechanical citation validation -- the point of Stage 4.

Everything else in this package is plumbing around this module. The generator
produces citations; this decides whether they are *real*, by checking each one
against the chunks that were actually retrieved for this specific question. Not
against the repo, and not against a plausibility judgement -- against the exact
evidence the model was shown.

The rule
--------
A citation is valid when a single retrieved chunk **fully contains** it:

    chunk.file_path == citation.file_path
    chunk.start_line <= citation.start_line and citation.end_line <= chunk.end_line

Full containment within *one* chunk, not overlap, and not containment within the
union of several chunks. Two consequences worth being explicit about:

* A citation that half-overlaps a chunk is rejected. Those overhanging lines were
  never in the model's context, so it cannot have read them; "close to something
  real" is exactly the failure mode this stage exists to catch.
* A citation spanning two adjacent retrieved chunks of the same file is also
  rejected, even though every line was in fact shown. That is a deliberate false
  negative: the prompt hands the model exact ranges and tells it to copy one, so
  a span it had to synthesise is off-spec even when it happens to be covered.
  The cost is a stripped claim, which is a much cheaper error than a citation
  that points somewhere the reader was never given.

A claim is valid only when *every* one of its citations is valid. One fabricated
citation on an otherwise well-sourced claim still ships a fabricated citation.

Stage 5
-------
:class:`ValidationResult` is built to be scoring data, not a one-off HTTP
response. Per-citation verdicts with reasons, per-claim rollups, and the
aggregate rates (``citation_precision``, ``claim_support_rate``) are all present
on every result, so an eval harness can average them across a question set
without re-deriving anything or re-running the validator.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, computed_field

from .models import Answer, Citation, Claim


class InvalidReason(str, Enum):
    """Why a citation failed. Stable strings -- Stage 5 groups metrics by these."""

    #: No retrieved chunk came from that file at all (the classic hallucination).
    FILE_NOT_IN_RETRIEVED_SET = "file_not_in_retrieved_set"
    #: Right file, but no single retrieved chunk covers the cited lines.
    LINE_RANGE_NOT_IN_CHUNK = "line_range_not_in_chunk"
    #: The range is not a range: start > end, or a non-positive line number.
    LINE_RANGE_INVALID = "line_range_invalid"


class RetrievedChunk(Protocol):
    """The three fields validation needs from a retrieved chunk.

    Structural, so this module never imports the search layer: a Stage 3
    ``SearchResult`` satisfies it, and so does a two-line test stub.
    """

    file_path: str
    start_line: int
    end_line: int


def normalize_path(path: str) -> str:
    """Normalize a path for comparison without making matching *loose*.

    Only the differences that are pure notation are collapsed: surrounding
    whitespace, Windows separators, and a leading ``./``. Case is preserved --
    ``Auth.ts`` and ``auth.ts`` are different files on the platforms this indexes
    code from, and treating them as one would validate a citation to a file that
    does not exist.
    """
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


class CitationValidation(BaseModel):
    """The verdict on one citation, with enough context to explain itself."""

    citation: Citation
    valid: bool
    reason: Optional[InvalidReason] = None
    #: Which retrieved chunk vouched for it (``SearchResult.id``), when valid.
    matched_chunk_id: Optional[str] = None
    #: Human-readable explanation; fed verbatim into the retry prompt.
    detail: Optional[str] = None


class ClaimValidation(BaseModel):
    """The verdict on one claim: valid only if all of its citations are."""

    claim_index: int
    claim_text: str
    citations: list[CitationValidation]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valid(self) -> bool:
        return all(c.valid for c in self.citations)

    @property
    def invalid_citations(self) -> list[CitationValidation]:
        return [c for c in self.citations if not c.valid]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reasons(self) -> list[InvalidReason]:
        """Distinct failure reasons, order-preserved -- Stage 5 tallies these."""
        seen: list[InvalidReason] = []
        for c in self.invalid_citations:
            if c.reason is not None and c.reason not in seen:
                seen.append(c.reason)
        return seen


class ValidationResult(BaseModel):
    """Per-claim verdicts plus the aggregate rates Stage 5 will score on."""

    claims: list[ClaimValidation] = Field(default_factory=list)
    #: The files that were actually available to cite, for error messages and
    #: for after-the-fact debugging of a run.
    retrieved_files: list[str] = Field(default_factory=list)

    # -- rollups ------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        return all(c.valid for c in self.claims)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valid_claims(self) -> int:
        return sum(1 for c in self.claims if c.valid)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_citations(self) -> int:
        return sum(len(c.citations) for c in self.claims)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valid_citations(self) -> int:
        return sum(1 for c in self.claims for cit in c.citations if cit.valid)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation_precision(self) -> float:
        """Fraction of emitted citations that check out. 1.0 when none emitted.

        An answer with no citations at all has nothing false in it, so it scores
        1.0 here; ``total_citations`` is reported alongside precisely so an empty
        answer is never mistaken for a perfect one.
        """
        if self.total_citations == 0:
            return 1.0
        return self.valid_citations / self.total_citations

    @computed_field  # type: ignore[prop-decorator]
    @property
    def claim_support_rate(self) -> float:
        """Fraction of claims where every citation checks out."""
        if self.total_claims == 0:
            return 1.0
        return self.valid_claims / self.total_claims

    # -- helpers ------------------------------------------------------------

    @property
    def invalid_claims(self) -> list[ClaimValidation]:
        return [c for c in self.claims if not c.valid]

    @property
    def invalid_claim_indices(self) -> set[int]:
        return {c.claim_index for c in self.invalid_claims}


def validate_citation(
    citation: Citation, chunks_by_file: dict[str, list[RetrievedChunk]]
) -> CitationValidation:
    """Check one citation against the retrieved set."""
    if citation.start_line < 1 or citation.end_line < citation.start_line:
        return CitationValidation(
            citation=citation,
            valid=False,
            reason=InvalidReason.LINE_RANGE_INVALID,
            detail=(
                f"{citation} is not a valid line range "
                "(lines start at 1 and start_line must be <= end_line)."
            ),
        )

    candidates = chunks_by_file.get(normalize_path(citation.file_path))
    if not candidates:
        return CitationValidation(
            citation=citation,
            valid=False,
            reason=InvalidReason.FILE_NOT_IN_RETRIEVED_SET,
            detail=(
                f"{citation} refers to a file that is not in the retrieved "
                "context, so nothing was read from it."
            ),
        )

    for chunk in candidates:
        if chunk.start_line <= citation.start_line and citation.end_line <= chunk.end_line:
            return CitationValidation(
                citation=citation,
                valid=True,
                matched_chunk_id=str(getattr(chunk, "id", "") or "") or None,
            )

    available = ", ".join(
        f"{c.start_line}-{c.end_line}" for c in candidates
    )
    return CitationValidation(
        citation=citation,
        valid=False,
        reason=InvalidReason.LINE_RANGE_NOT_IN_CHUNK,
        detail=(
            f"{citation} is not contained in any retrieved chunk of that file. "
            f"The ranges available for {normalize_path(citation.file_path)} are: "
            f"{available}."
        ),
    )


def validate_claim(
    claim: Claim, index: int, chunks_by_file: dict[str, list[RetrievedChunk]]
) -> ClaimValidation:
    """Check every citation on one claim."""
    return ClaimValidation(
        claim_index=index,
        claim_text=claim.text,
        citations=[validate_citation(c, chunks_by_file) for c in claim.citations],
    )


def validate_answer(
    answer: Answer, chunks: Sequence[RetrievedChunk]
) -> ValidationResult:
    """Validate every claim in ``answer`` against the chunks retrieved for it.

    This is the whole contract of the module: in go an answer and the evidence
    it was generated from, out comes a per-claim, per-citation verdict that both
    the retry loop and Stage 5's scoring read.
    """
    chunks_by_file = _index_by_file(chunks)
    return ValidationResult(
        claims=[
            validate_claim(claim, index, chunks_by_file)
            for index, claim in enumerate(answer.claims)
        ],
        retrieved_files=sorted(chunks_by_file),
    )


def _index_by_file(
    chunks: Iterable[RetrievedChunk],
) -> dict[str, list[RetrievedChunk]]:
    """Group chunks by normalized path so lookup is one dict hit per citation."""
    index: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        index.setdefault(normalize_path(chunk.file_path), []).append(chunk)
    return index
