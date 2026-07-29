"""Reciprocal Rank Fusion (RRF).

Pure and dependency-free on purpose: no DB access, no I/O, no knowledge of what
a "chunk" is. It takes ranked lists of ids and returns one fused ranked list,
which makes the ranking maths trivially unit-testable against hand-computed
expected scores.

RRF combines rankings by *position*, never by raw score::

    score(d) = sum over lists L containing d of  1 / (k + rank_L(d))

with ``rank`` 1-indexed. That matters here because the two retrievers produce
scores on incomparable scales -- pgvector cosine similarity lives in [-1, 1]
while ``ts_rank_cd`` is an unbounded density number -- so any attempt to blend
the scores directly would need per-corpus normalisation that quietly breaks as
the corpus changes. Ranks need no normalisation.

The ``k`` constant damps the contribution of top ranks: with k = 60, rank 1
scores 1/61 and rank 2 scores 1/62, a ~1.6% gap, so a document must place well
in *both* lists to beat one that placed first in only one. Small k makes the
fusion trust each retriever's top hit much more strongly.

Reference: Cormack, Clarke & Buettcher (SIGIR 2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods" -- the paper that
introduced RRF and used k = 60.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

#: The RRF damping constant from the original paper. Named, not inlined, and
#: overridable per call so Stage 5 can sweep it once eval numbers exist.
DEFAULT_RRF_K = 60


@runtime_checkable
class HasId(Protocol):
    id: str


@dataclass
class FusedItem:
    """One fused result: an id, its combined score, and where it came from."""

    id: str
    score: float
    #: 1-indexed rank per input list, keyed by list label. A label is absent
    #: when that list did not contain this id at all.
    ranks: dict[str, int] = field(default_factory=dict)


def _id_of(item: Any) -> str:
    """Accept either a bare id or any object exposing ``.id``."""
    if isinstance(item, str):
        return item
    ident = getattr(item, "id", None)
    if ident is None:
        raise TypeError(
            f"ranked lists must contain ids or objects with an `id` attribute, "
            f"got {type(item).__name__}"
        )
    return str(ident)


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Any]],
    k: int = DEFAULT_RRF_K,
) -> list[FusedItem]:
    """Fuse labelled ranked lists into one list, best first.

    Parameters
    ----------
    ranked_lists:
        Label -> ranked sequence, best first. Elements may be ids (``str``) or
        objects with an ``id`` attribute. Labels are arbitrary (we use
        ``"vector"`` and ``"keyword"``) and show up in :attr:`FusedItem.ranks`.
    k:
        The RRF damping constant. Must be positive.

    Returns
    -------
    Every id appearing in *any* input list, scored and sorted by descending
    fused score. Nothing is dropped: an id found by only one retriever still
    scores, it just scores once instead of twice. Truncation to ``top_k`` is
    the caller's job -- fusing is not the place to lose candidates.

    Ties are broken by first appearance (scanning lists in mapping order, then
    by rank), so the output is deterministic for identical inputs.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    items: dict[str, FusedItem] = {}
    for label, ranked in ranked_lists.items():
        for position, entry in enumerate(ranked, start=1):
            ident = _id_of(entry)
            item = items.get(ident)
            if item is None:
                item = items[ident] = FusedItem(id=ident, score=0.0)
            if label in item.ranks:
                # A duplicate within one list: keep the better (earlier) rank
                # and don't let it collect the contribution twice.
                continue
            item.ranks[label] = position
            item.score += 1.0 / (k + position)

    # dict preserves insertion order and sorted() is stable, so equal scores
    # keep first-seen order rather than shuffling between runs.
    return sorted(items.values(), key=lambda i: i.score, reverse=True)
