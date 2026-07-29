"""Search: vector similarity (Stage 2) + keyword & RRF-fused hybrid (Stage 3)."""

from __future__ import annotations

from .fusion import DEFAULT_RRF_K, FusedItem, reciprocal_rank_fusion
from .hybrid_search import (
    DEFAULT_OVERFETCH_FACTOR,
    VALID_MODES,
    HybridSearchService,
)
from .keyword_search import KeywordSearchService
from .results import SearchResult
from .vector_search import VectorSearchService, cosine_rank

__all__ = [
    "SearchResult",
    "VectorSearchService",
    "KeywordSearchService",
    "HybridSearchService",
    "reciprocal_rank_fusion",
    "FusedItem",
    "DEFAULT_RRF_K",
    "DEFAULT_OVERFETCH_FACTOR",
    "VALID_MODES",
    "cosine_rank",
]
