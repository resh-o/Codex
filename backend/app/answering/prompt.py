"""Prompt construction, kept away from everything that calls a model.

Wording is the part of an LLM system that gets rewritten most often and reasoned
about least reliably, so it lives in its own module with no API client, no
orchestration, and no I/O. Iterating on the instructions means editing strings
here; nothing downstream has to change.

Two ideas drive the format below:

1. **Give exact values to copy, not values to infer.** Every chunk is labelled
   with the precise ``file_path`` and ``start_line``/``end_line`` it must be
   cited as. The model is never asked to count lines or estimate where a
   function begins -- that is arithmetic it is bad at and the validator is
   unforgiving about.
2. **Make the retry specific.** A failed attempt comes back with the exact
   citations that failed and why, not "please try again", so the second attempt
   is corrective rather than a re-roll of the same dice.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from .validator import ValidationResult


class ContextChunk(Protocol):
    """What prompt rendering needs from a retrieved chunk (a ``SearchResult``)."""

    file_path: str
    language: str
    chunk_type: str
    qualified_name: str
    start_line: int
    end_line: int
    snippet: str


SYSTEM_PROMPT = """\
You are a codebase question-answering assistant. You answer questions about a \
specific repository using ONLY the code excerpts supplied in the user message.

Rules, in order of importance:

1. Use only the supplied excerpts. You have no other knowledge of this \
repository. If the excerpts do not answer the question, say so in a claim that \
cites the closest relevant excerpt, or return no claims at all. Never fill a \
gap with what a codebase like this usually does.
2. Break the answer into discrete claims. Each claim is one self-contained \
factual statement about the code. Do not bundle three facts into one sentence; \
split them, so each carries its own evidence.
3. Every claim must carry at least one citation, and every citation must be \
copied from the excerpt headers exactly as given: the same file_path string, \
and a start_line/end_line range that lies inside a single excerpt's stated \
range. Copy the excerpt's own range unless you are citing a narrower part of \
that same excerpt.
4. Never cite a file that is not in the excerpts below. Never cite line numbers \
outside a single excerpt's stated range, and never merge two excerpts' ranges \
into one citation. Citations are checked mechanically against the excerpts; an \
invented file or range is discarded and the claim is thrown away with it.
5. The summary is optional, carries no citation, and therefore must carry no \
facts -- it is a framing line such as "Authentication happens in two layers:". \
Put every fact in a claim.
6. Be concrete and technical. Name the actual functions, classes, and modules \
from the excerpts.\
"""


def format_chunk(chunk: ContextChunk, index: int) -> str:
    """Render one retrieved chunk with the exact values it must be cited as."""
    header = (
        f"[{index}] file_path: {chunk.file_path}\n"
        f"    lines: {chunk.start_line}-{chunk.end_line}   "
        f"(cite this file_path with a range inside {chunk.start_line}-"
        f"{chunk.end_line})\n"
        f"    symbol: {chunk.qualified_name} "
        f"({chunk.chunk_type}, {chunk.language})"
    )
    body = chunk.snippet.rstrip()
    if body.endswith("…"):
        # The retrieval layer truncates long chunks. Say so, rather than letting
        # the model conclude the function simply ends there.
        body += "\n[excerpt truncated -- the chunk's full range is still as stated above]"
    return f"{header}\n```{chunk.language}\n{body}\n```"


def format_context(chunks: Sequence[ContextChunk]) -> str:
    """Render the whole retrieved set as the citable context block."""
    if not chunks:
        return "(no code excerpts were retrieved)"
    return "\n\n".join(format_chunk(c, i + 1) for i, c in enumerate(chunks))


def build_retry_feedback(validation: ValidationResult) -> str:
    """Turn a failed validation into instructions specific enough to act on.

    Names the offending claim, the offending citation, and the reason -- and
    restates what *was* available -- because "your previous answer was invalid"
    gives the model nothing to correct.
    """
    lines = [
        "Your previous answer was rejected: some citations did not exist in the "
        "excerpts you were given. Citations are checked mechanically, so this is "
        "not a matter of opinion.",
        "",
        "Problems found:",
    ]
    for claim in validation.invalid_claims:
        lines.append(f'- Claim: "{claim.claim_text}"')
        for citation in claim.invalid_citations:
            reason = citation.reason.value if citation.reason else "invalid"
            detail = citation.detail or str(citation.citation)
            lines.append(f"    * {citation.citation} [{reason}] {detail}")

    if validation.retrieved_files:
        available = ", ".join(validation.retrieved_files)
        lines += ["", f"The only citable files are: {available}."]

    lines += [
        "",
        "Answer the question again. Do not repeat any of the citations above. "
        "Copy every file_path and line range from the excerpt headers exactly. "
        "If a claim cannot be supported by an excerpt, drop the claim rather "
        "than citing something that is not there.",
    ]
    return "\n".join(lines)


def build_user_prompt(
    query: str,
    chunks: Sequence[ContextChunk],
    feedback: Optional[str] = None,
) -> str:
    """Assemble the user message: context, question, and any retry feedback.

    Feedback goes last so the correction is the final thing read before the
    model answers.
    """
    parts = [
        "Code excerpts retrieved for this question:",
        "",
        format_context(chunks),
        "",
        f"Question: {query}",
    ]
    if feedback:
        parts += ["", feedback]
    return "\n".join(parts)
