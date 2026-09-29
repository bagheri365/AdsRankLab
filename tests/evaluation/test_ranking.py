import numpy as np
import pandas as pd

from ads_rank_lab.evaluation.ranking import (
    advertiser_top_share,
    rank_shift_summary,
    score_margin_summary,
    top_k_overlap,
)


def test_top_k_overlap_detects_replacement() -> None:
    result = top_k_overlap(
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 1.0, 3.0, 2.0],
        fraction=0.5,
    )

    assert result["k"] == 2
    assert result["overlap_count"] == 1
    assert result["overlap_rate"] == 0.5
    assert result["replacement_rate"] == 0.5


def test_rank_shift_is_zero_for_positive_rescaling() -> None:
    result = rank_shift_summary(
        [4.0, 3.0, 2.0, 1.0],
        [40.0, 30.0, 20.0, 10.0],
    )

    assert result["rank_changed_fraction"] == 0.0
    assert result["median_absolute_rank_shift"] == 0.0
    assert result["adjacent_order_flip_rate"] == 0.0


def test_rank_shift_detects_reordering() -> None:
    result = rank_shift_summary(
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 1.0, 3.0, 2.0],
    )

    assert result["rank_changed_fraction"] > 0.0
    assert result["max_absolute_rank_shift"] > 0


def test_score_margin_reports_top_boundary() -> None:
    result = score_margin_summary(
        [10.0, 8.0, 6.0, 4.0, 2.0],
        top_fraction=0.4,
    )

    assert result["k"] == 2
    assert result["boundary_score"] == 8.0
    assert result["next_score"] == 6.0
    assert result["absolute_boundary_gap"] == 2.0


def test_advertiser_top_share_reports_representation() -> None:
    frame = pd.DataFrame(
        {"advertiser_id": ["a", "a", "b", "b"]}
    )
    scores = np.array([4.0, 3.0, 2.0, 1.0])

    result = advertiser_top_share(
        frame,
        scores,
        policy="example",
        top_fraction=0.5,
    )

    a = result.loc[result["advertiser_id"] == "a"].iloc[0]
    assert a["selected_rows"] == 2
    assert a["population_share"] == 0.5
    assert a["selected_share"] == 1.0
    assert a["representation_ratio"] == 2.0
