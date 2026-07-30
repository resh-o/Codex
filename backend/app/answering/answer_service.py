"""Orchestration: retrieve -> generate -> validate -> retry once -> strip.

The pipeline, and why it ends the way it does:

1. **Retrieve** with Stage 3's hybrid search. The retrieved set is both the
   model's context *and* the ground truth the validator checks against, which is
   the property that makes citation validation meaningful rather than
   decorative -- a citation is verified against exactly the evidence that
   produced it.
2. **Generate** a schema-constrained answer.
3. **Validate** every citation mechanically.
4. **Retry once, with the failures named.** The second attempt is told which
   citations failed and why, so it is a correction and not a re-roll.
5. **Strip, don't ship.** Claims still carrying a bad citation are removed from
   the response and reported in the metadata. A stripped claim is a visible gap;
   a fabricated citation is an invisible lie, and the whole stage exists to
   prefer the former.

Exactly one retry, by design (Stage 4 spec): the interesting question is whether
targeted feedback fixes a bad citation, and a fixed budget makes that measurable
in Stage 5 instead of hiding it behind a loop that eventually gets lucky.

The retry's answer replaces the first outright rather than the two being merged
or compared. That is the simple reading of "retry, then strip", and it keeps the
attempt count honest -- picking whichever attempt scored better is an
optimisation worth making once Stage 5 can say whether it helps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Protocol, Sequence

from pydantic import BaseModel, Field, computed_field

from .generator import AnswerGenerator, GenerationError
from .models import Answer, Claim
from .prompt import build_retry_feedback
from .validator import (
    ClaimValidation,
    InvalidReason,
    RetrievedChunk,
    ValidationResult,
    validate_answer,
)

logger = logging.getLogger(__name__)

#: One generation, then at most one corrective retry. Fixed, not configurable.
MAX_ATTEMPTS = 2


class NoContextError(RuntimeError):
    """Retrieval returned nothing, so there is nothing to answer from."""


class SearchService(Protocol):
    """The slice of Stage 3's hybrid service this needs."""

    async def search(
        self,
        query: str,
        top_k: int = 10,
        repo_url: Optional[str] = None,
        mode: str = "hybrid",
    ) -> list: ...


class StrippedClaim(BaseModel):
    """A claim removed from the answer because its citations did not check out."""

    claim_index: int
    text: str
    reasons: list[InvalidReason]
    details: list[str]

    @classmethod
    def from_validation(cls, claim: ClaimValidation) -> "StrippedClaim":
        return cls(
            claim_index=claim.claim_index,
            text=claim.claim_text,
            reasons=claim.reasons,
            details=[
                c.detail or str(c.citation) for c in claim.invalid_citations
            ],
        )


class SourceRef(BaseModel):
    """A chunk that was retrieved and therefore citable, for the response."""

    chunk_id: Optional[str] = None
    file_path: str
    start_line: int
    end_line: int
    qualified_name: Optional[str] = None
    score: Optional[float] = None


class AnswerResult(BaseModel):
    """The final answer plus everything needed to score or debug the run.

    ``validation`` is the *unstripped* verdict on the last attempt: it reports
    what the model actually produced, not what survived. Stage 5 wants the
    former (that is the citation-accuracy signal); the user gets the latter in
    ``answer``.
    """

    answer: Answer
    validation: ValidationResult
    attempts: int
    stripped_claims: list[StrippedClaim] = Field(default_factory=list)
    #: One ValidationResult per attempt, oldest first -- lets Stage 5 measure
    #: whether the corrective retry actually repairs citations.
    validation_history: list[ValidationResult] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    #: Set when the corrective retry itself failed to run (API error) and the
    #: first attempt was stripped instead.
    retry_error: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def retried(self) -> bool:
        return self.attempts > 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def claims_stripped(self) -> int:
        return len(self.stripped_claims)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_claims_stripped(self) -> bool:
        """Every claim failed validation -- an answer that is entirely empty."""
        return bool(self.stripped_claims) and not self.answer.claims


class AnswerService:
    """Turns a question into an answer whose every citation has been checked."""

    def __init__(
        self,
        search: SearchService,
        generator: AnswerGenerator,
        default_top_k: int = 10,
    ) -> None:
        self._search = search
        self._generator = generator
        self._default_top_k = default_top_k

    async def answer_question(
        self,
        query: str,
        repo_url: Optional[str] = None,
        top_k: int = 10,
    ) -> AnswerResult:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        chunks = await self._search.search(
            query, top_k=top_k or self._default_top_k, repo_url=repo_url, mode="hybrid"
        )
        if not chunks:
            raise NoContextError(
                "No relevant code was retrieved for this question, so there is "
                "nothing to cite. Has this repository been embedded via /embed?"
            )

        history: list[ValidationResult] = []

        # --- attempt 1 ------------------------------------------------------
        answer = await self._generate(query, chunks)
        validation = validate_answer(answer, chunks)
        history.append(validation)
        attempts = 1
        retry_error: Optional[str] = None

        # --- attempt 2: corrective retry, failures named --------------------
        if not validation.is_valid:
            logger.info(
                "Citation validation failed on attempt 1 (%d/%d claims valid); "
                "retrying with feedback",
                validation.valid_claims,
                validation.total_claims,
            )
            feedback = build_retry_feedback(validation)
            try:
                retry_answer = await self._generate(query, chunks, feedback=feedback)
            except GenerationError as exc:
                # Losing the whole request because the *retry* call failed would
                # throw away a partially-valid first answer for no reason. Keep
                # attempt 1 and strip it instead.
                retry_error = str(exc)
                logger.warning("Corrective retry failed to generate: %s", exc)
            else:
                attempts = 2
                answer = retry_answer
                validation = validate_answer(answer, chunks)
                history.append(validation)

        # --- strip whatever is still unsupported ----------------------------
        stripped: list[StrippedClaim] = []
        if not validation.is_valid:
            invalid = validation.invalid_claim_indices
            kept: list[Claim] = [
                claim
                for index, claim in enumerate(answer.claims)
                if index not in invalid
            ]
            stripped = [
                StrippedClaim.from_validation(c) for c in validation.invalid_claims
            ]
            answer = answer.with_claims(kept)
            logger.warning(
                "Stripped %d claim(s) with unverifiable citations after %d attempt(s)",
                len(stripped),
                attempts,
            )

        return AnswerResult(
            answer=answer,
            validation=validation,
            attempts=attempts,
            stripped_claims=stripped,
            validation_history=history,
            sources=[_source_ref(c) for c in chunks],
            retry_error=retry_error,
        )

    def answer_question_sync(
        self,
        query: str,
        repo_url: Optional[str] = None,
        top_k: int = 10,
    ) -> AnswerResult:
        """Blocking wrapper, for callers outside an event loop (scripts, tests)."""
        return asyncio.run(
            self.answer_question(query, repo_url=repo_url, top_k=top_k)
        )

    async def _generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        feedback: Optional[str] = None,
    ) -> Answer:
        """Run the (blocking) model call off the event loop."""
        return await asyncio.to_thread(
            self._generator.generate, query, chunks, feedback
        )


def _source_ref(chunk) -> SourceRef:
    return SourceRef(
        chunk_id=str(getattr(chunk, "id", "") or "") or None,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        qualified_name=getattr(chunk, "qualified_name", None),
        score=getattr(chunk, "score", None),
    )
