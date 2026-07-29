"""The result shape every search mode returns.

One dataclass is shared by vector-only, keyword-only, and hybrid search so the
API response schema does not change shape depending on ``mode``. Fields that a
given mode cannot fill stay ``None`` (a keyword-only hit has no cosine
similarity; a vector-only hit has no fused score).

The diagnostic fields (``vector_rank`` / ``keyword_rank`` / ``fused_score``)
exist for Stage 5: comparing hybrid against vector-only quantitatively is much
easier when every hybrid response already says where each result came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_SNIPPET_CHARS = 400


@dataclass
class SearchResult:
    """A single result formatted for API responses."""

    id: str
    repo_url: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    snippet: str

    # The score this result was ranked by, in whichever mode produced it:
    # cosine similarity, ts_rank_cd rank, or the fused RRF score.
    score: float

    # --- Per-mode diagnostics (None when the mode didn't produce them) ----
    similarity: Optional[float] = None
    keyword_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    fused_score: Optional[float] = None


def snippet(content: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    """Truncate content to a readable snippet on a line/word boundary."""
    content = content.strip()
    if len(content) <= max_chars:
        return content
    cut = content[:max_chars]
    # Prefer to end at the last newline, else the last space, for readability.
    boundary = max(cut.rfind("\n"), cut.rfind(" "))
    if boundary > max_chars // 2:
        cut = cut[:boundary]
    return cut.rstrip() + " …"
