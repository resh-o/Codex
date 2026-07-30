"""The shapes an answer is allowed to take.

Two related models live here, and the split matters:

* :class:`GeneratedAnswer` is the *LLM-facing* schema. It is handed to Gemini as
  the response schema, so it contains only what the model is allowed to invent:
  an optional summary line and a list of claims. Nothing else.
* :class:`Answer` is the *caller-facing* result: the same content plus the query
  it answers, stitched on by the generator afterwards. The query is ours, not the
  model's, so it deliberately never appears in the schema the model fills in --
  a field the model can rewrite is a field the model can get wrong.

The citation requirement is structural, not advisory: ``Claim.citations`` is
``min_length=1``, which becomes ``minItems: 1`` in the JSON schema sent to the
API *and* is re-checked by pydantic when the response is parsed. A claim with no
citation cannot be constructed, so it cannot reach the validator, let alone a
user.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A pointer to an exact line range in an exact file.

    Line numbers are 1-based and inclusive on both ends, matching the
    ``start_line``/``end_line`` convention Stage 1's chunker emits and Stage 3's
    search results carry -- so a citation and a chunk are directly comparable
    without any off-by-one translation.
    """

    file_path: str = Field(
        ...,
        description=(
            "Repo-relative path, copied verbatim from the provided context. "
            "Do not guess, shorten, or reformat it."
        ),
    )
    start_line: int = Field(
        ..., ge=1, description="First line of the cited range (1-based, inclusive)."
    )
    end_line: int = Field(
        ..., ge=1, description="Last line of the cited range (1-based, inclusive)."
    )

    def __str__(self) -> str:  # used in retry feedback and log lines
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class Claim(BaseModel):
    """One factual statement plus the evidence for it.

    A claim may cite several locations (a function and its caller, say), but
    never zero -- that is what makes "every factual claim carries a citation" a
    property of the type rather than a hope about the prompt.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="A single, self-contained factual statement about the code.",
    )
    citations: list[Citation] = Field(
        ...,
        min_length=1,
        description=(
            "At least one location from the provided context that supports this "
            "exact statement."
        ),
    )


class GeneratedAnswer(BaseModel):
    """Exactly what the model is asked to produce -- the JSON response schema."""

    summary: str | None = Field(
        None,
        description=(
            "Optional one-line framing sentence, e.g. 'Authentication is handled "
            "in two layers:'. It carries no citations, so it must contain no "
            "factual claims -- put every fact in a claim instead."
        ),
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="The answer, broken into individually-cited statements.",
    )


class Answer(BaseModel):
    """A generated answer bound to the question it answers."""

    query: str
    claims: list[Claim] = Field(default_factory=list)
    summary: str | None = None

    @classmethod
    def from_generated(cls, query: str, generated: GeneratedAnswer) -> "Answer":
        return cls(query=query, claims=generated.claims, summary=generated.summary)

    def with_claims(self, claims: list[Claim]) -> "Answer":
        """Return a copy carrying a different claim list (used when stripping)."""
        return Answer(query=self.query, claims=claims, summary=self.summary)
