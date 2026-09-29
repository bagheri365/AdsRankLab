from ads_rank_lab.evaluation.tie_aware import (
    normalized_rank_shift_summary,
    tie_aware_cutoff,
    tie_aware_top_k_comparison,
)


def test_tie_aware_cutoff_exposes_ambiguous_boundary() -> None:
    result = tie_aware_cutoff(
        [10.0, 8.0, 8.0, 8.0, 1.0],
        fraction=0.4,
    )

    assert result["k"] == 2
    assert result["strictly_above_cutoff"] == 1
    assert result["tied_at_cutoff"] == 3
    assert result["selected_from_cutoff_tie"] == 1


def test_tie_aware_top_k_is_identical_for_identical_ties() -> None:
    result = tie_aware_top_k_comparison(
        [10.0, 8.0, 8.0, 1.0],
        [10.0, 8.0, 8.0, 1.0],
        fraction=0.5,
    )

    assert result["guaranteed_jaccard"] == 1.0
    assert result["possible_jaccard"] == 1.0


def test_normalized_rank_shift_reports_population_fraction() -> None:
    result = normalized_rank_shift_summary(
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 1.0, 3.0, 2.0],
    )

    assert 0.0 <= result["median_absolute_rank_shift_fraction"] <= 1.0
    assert 0.0 <= result["max_absolute_rank_shift_fraction"] <= 1.0
