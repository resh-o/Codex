"""Data models for the ingestion pipeline.

The :class:`Chunk` is the single unit of currency that later stages
(embedding, vector search, hybrid retrieval, citation enforcement) will
consume.  Keeping it small and explicit here means the interface stays stable
as the rest of the system is built out.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Language = Literal["python", "typescript", "javascript"]
ChunkType = Literal["function", "method", "class", "module"]


class Chunk(BaseModel):
    """A single retrievable slice of source code.

    Chunks are produced at AST boundaries (functions, methods, classes) plus a
    per-file ``module`` chunk for top-level code that is not inside any
    definition.  A whole-file ``module`` chunk is also emitted as a fallback
    when a file cannot be parsed.
    """

    repo_url: str
    commit_sha: str
    file_path: str = Field(..., description="Path relative to the repo root, POSIX separators.")
    language: Language
    chunk_type: ChunkType

    name: str = Field(..., description="The bare definition name, e.g. 'validate_token'.")
    qualified_name: str = Field(
        ...,
        description="Dot-path including parents, e.g. 'UserAuth.validate_token'. "
        "Equals `name` for a top-level definition.",
    )
    parent_name: Optional[str] = Field(
        None, description="Immediate enclosing definition name, or None if top-level."
    )

    start_line: int = Field(..., ge=1, description="1-indexed, inclusive.")
    end_line: int = Field(..., ge=1, description="1-indexed, inclusive.")

    content: str = Field(..., description="Raw source text of the chunk.")
    docstring: Optional[str] = Field(
        None,
        description="Leading docstring / JSDoc / comment block if present, extracted "
        "separately even though it is also contained in `content`.",
    )

    def slice_from(self, source: str) -> str:
        """Return the lines of ``source`` this chunk claims to span.

        Useful for tests and debugging: for AST chunks this reproduces
        ``content`` exactly.  (For ``module`` chunks the content is a
        concatenation of non-contiguous top-level statements, so this may
        differ -- see ``chunker`` for details.)
        """
        lines = source.splitlines()
        return "\n".join(lines[self.start_line - 1 : self.end_line])
