"""Tests for Reciprocal Rank Fusion.

Pure unit tests -- no DB, no network, no fakes beyond a trivial id holder.
The headline test hand-computes every expected score so the RRF arithmetic is
pinned to the formula, not to whatever the implementation happens to return.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.search.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


def rrf(k: int, *ranks: int) -> float:
    """Independently compute sum(1 / (k + rank)) for a doc's 1-indexed ranks."""
    return sum(1.0 / (k + r) for r in ranks)


# --------------------------------------------------------------------------- #
# The formula itself
# --------------------------------------------------------------------------- #


def test_k_defaults_to_60_from_the_rrf_paper() -> None:
    assert DEFAULT_RRF_K == 60


def test_scores_match_hand_computed_values() -> None:
    # vector: A(1) B(2) C(3)      keyword: B(1) D(2) A(3)
    fused = reciprocal_rank_fusion(
        {"vector": ["A", "B", "C"], "keyword": ["B", "D", "A"]}
    )
    scores = {item.id: item.score for item in fused}

    # Hand-computed with k = 60:
    #   A = 1/61 + 1/63 = 0.0163934426 + 0.0158730159 = 0.0322664585
    #   B = 1/62 + 1/61 = 0.0161290323 + 0.0163934426 = 0.0325224749
    #   C = 1/63                                      = 0.0158730159
    #   D = 1/62                                      = 0.0161290323
    assert scores["A"] == pytest.approx(0.0322664585, abs=1e-9)
    assert scores["B"] == pytest.approx(0.0325224749, abs=1e-9)
    assert scores["C"] == pytest.approx(0.0158730159, abs=1e-9)
    assert scores["D"] == pytest.approx(0.0161290323, abs=1e-9)

    # ...and the resulting order. Note B beats A despite A leading the vector
    # list: B's 2nd + 1st beats A's 1st + 3rd.
    assert [item.id for item in fused] == ["B", "A", "D", "C"]


def test_scores_agree_with_an_independent_oracle() -> None:
    vector = ["v1", "v2", "v3", "shared", "v5"]
    keyword = ["shared", "k2", "v2"]
    fused = reciprocal_rank_fusion({"vector": vector, "keyword": keyword})
    scores = {item.id: item.score for item in fused}

    assert scores["shared"] == pytest.approx(rrf(60, 4, 1))
    assert scores["v2"] == pytest.approx(rrf(60, 2, 3))
    assert scores["v1"] == pytest.approx(rrf(60, 1))
    assert scores["k2"] == pytest.approx(rrf(60, 2))


# --------------------------------------------------------------------------- #
# The properties that make fusion worth doing
# --------------------------------------------------------------------------- #


def test_appearing_in_both_lists_outranks_appearing_in_one() -> None:
    # All at rank 1 or 2, so the only thing separating them is list coverage.
    fused = reciprocal_rank_fusion(
        {"vector": ["both", "vector_only"], "keyword": ["both", "keyword_only"]}
    )
    assert fused[0].id == "both"

    scores = {item.id: item.score for item in fused}
    assert scores["both"] > scores["vector_only"]
    assert scores["both"] > scores["keyword_only"]
    # Sanity: "both" scores exactly the sum of its two contributions.
    assert scores["both"] == pytest.approx(rrf(60, 1, 1))


def test_single_list_chunks_are_included_not_dropped() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": ["a", "b"], "keyword": ["c", "d"]}
    )
    assert {item.id for item in fused} == {"a", "b", "c", "d"}
    assert all(item.score > 0 for item in fused)


def test_lower_ranked_in_one_list_can_beat_top_of_the_other() -> None:
    """The exact-identifier rescue, in miniature.

    A chunk the vector retriever buried at rank 9 but keyword put first should
    still beat a chunk that only one retriever saw at rank 1.
    """
    vector = [f"v{i}" for i in range(1, 9)] + ["target"]
    keyword = ["target"]
    fused = reciprocal_rank_fusion({"vector": vector, "keyword": keyword})

    # target = 1/69 + 1/61 = 0.0144927536 + 0.0163934426 = 0.0308861962
    # v1     = 1/61                                      = 0.0163934426
    assert fused[0].id == "target"
    assert fused[0].score == pytest.approx(rrf(60, 9, 1))
    assert fused[1].id == "v1"


def test_output_is_sorted_by_descending_score() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": list("abcdefgh"), "keyword": list("hgfedcba")}
    )
    scores = [item.score for item in fused]
    assert scores == sorted(scores, reverse=True)


def test_ranks_record_the_source_positions() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": ["a", "b"], "keyword": ["b", "c"]}
    )
    ranks = {item.id: item.ranks for item in fused}
    assert ranks["a"] == {"vector": 1}
    assert ranks["b"] == {"vector": 2, "keyword": 1}
    assert ranks["c"] == {"keyword": 2}


# --------------------------------------------------------------------------- #
# Inputs, edges, and the k knob
# --------------------------------------------------------------------------- #


@dataclass
class Doc:
    id: str


def test_accepts_objects_with_an_id_attribute() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": [Doc("x"), Doc("y")], "keyword": [Doc("y")]}
    )
    assert [item.id for item in fused] == ["y", "x"]


def test_rejects_elements_without_an_id() -> None:
    with pytest.raises(TypeError):
        reciprocal_rank_fusion({"vector": [object()]})


def test_empty_and_single_list_inputs() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []

    # One retriever returning nothing must not disturb the other's order.
    fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"], "keyword": []})
    assert [item.id for item in fused] == ["a", "b", "c"]


def test_duplicate_ids_within_one_list_score_once_at_the_better_rank() -> None:
    fused = reciprocal_rank_fusion({"vector": ["a", "b", "a"]})
    scores = {item.id: item.score for item in fused}
    assert scores["a"] == pytest.approx(rrf(60, 1))  # not rrf(60, 1, 3)
    assert fused[0].id == "a"


def test_smaller_k_sharpens_the_advantage_of_a_top_rank() -> None:
    lists = {"vector": ["top", "second"], "keyword": ["second", "top"]}

    # With a tiny k, rank 1 dominates rank 2 far more strongly...
    sharp = reciprocal_rank_fusion(lists, k=1)
    sharp_scores = {i.id: i.score for i in sharp}
    # ...though here both docs hold one 1st and one 2nd, so they stay tied.
    assert sharp_scores["top"] == pytest.approx(sharp_scores["second"])

    # The real effect of k: the gap between rank 1 and rank 2 contributions.
    gap_small_k = (1 / (1 + 1)) - (1 / (1 + 2))
    gap_big_k = (1 / (60 + 1)) - (1 / (60 + 2))
    assert gap_small_k > gap_big_k


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"vector": ["a"]}, k=0)


def test_fusion_is_deterministic_for_equal_scores() -> None:
    lists = {"vector": ["a", "b", "c"], "keyword": ["a", "b", "c"]}
    first = [i.id for i in reciprocal_rank_fusion(lists)]
    second = [i.id for i in reciprocal_rank_fusion(lists)]
    assert first == second == ["a", "b", "c"]
