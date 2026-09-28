import pandas as pd
import pytest

from ads_rank_lab.data.events import build_observed_outcomes


def _bids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_id": ["b1", "b2", "b3"],
            "bid_price": [100, 200, 300],
            "advertiser_id": ["a", "a", "b"],
        }
    )


def test_join_is_anchored_on_all_bid_requests() -> None:
    bids = _bids()
    impressions = pd.DataFrame(
        {"bid_id": ["b1", "b3"], "paying_price": [40, 90]}
    )
    clicks = pd.DataFrame({"bid_id": ["b3"]})
    conversions = pd.DataFrame({"bid_id": ["b3"]})

    result, diagnostics = build_observed_outcomes(
        bids, impressions, clicks, conversions
    )

    assert result["bid_id"].tolist() == ["b1", "b2", "b3"]
    assert result["won"].tolist() == [True, False, True]
    assert result["clicked"].tolist() == [False, False, True]
    assert result["converted"].tolist() == [False, False, True]
    assert diagnostics.unique_bids == 3
    assert diagnostics.unique_impressions == 2


def test_multiple_clicks_are_counted_without_row_duplication() -> None:
    result, _ = build_observed_outcomes(
        _bids(),
        pd.DataFrame({"bid_id": ["b1"], "paying_price": [40]}),
        pd.DataFrame({"bid_id": ["b1", "b1"]}),
        pd.DataFrame({"bid_id": []}),
    )

    assert len(result) == 3
    assert result.loc[result["bid_id"] == "b1", "click_count"].item() == 2
    assert result.loc[result["bid_id"] == "b1", "clicked"].item()


def test_lost_bid_has_no_observed_paying_price() -> None:
    result, _ = build_observed_outcomes(
        _bids(),
        pd.DataFrame({"bid_id": ["b1"], "paying_price": [40]}),
        pd.DataFrame({"bid_id": []}),
        pd.DataFrame({"bid_id": []}),
    )

    lost = result.loc[result["bid_id"] == "b2"].iloc[0]
    assert lost["won"] is False or lost["won"] == False
    assert pd.isna(lost["observed_paying_price"])


def test_diagnostics_report_orphan_events_without_silently_dropping_them() -> None:
    _, diagnostics = build_observed_outcomes(
        _bids(),
        pd.DataFrame({"bid_id": ["b1", "outside"], "paying_price": [40, 50]}),
        pd.DataFrame({"bid_id": ["b1", "click-only"]}),
        pd.DataFrame({"bid_id": ["conv-only"]}),
    )

    assert diagnostics.impressions_without_bid == 1
    assert diagnostics.clicks_without_impression == 1
    assert diagnostics.conversions_without_impression == 1


def test_strict_mode_rejects_inconsistent_event_evidence() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        build_observed_outcomes(
            _bids(),
            pd.DataFrame({"bid_id": ["b1"], "paying_price": [40]}),
            pd.DataFrame({"bid_id": ["orphan-click"]}),
            pd.DataFrame({"bid_id": []}),
            strict=True,
        )


def test_duplicate_impressions_with_same_price_are_aggregated() -> None:
    result, diagnostics = build_observed_outcomes(
        _bids(),
        pd.DataFrame({"bid_id": ["b1", "b1"], "paying_price": [40, 40]}),
        pd.DataFrame({"bid_id": []}),
        pd.DataFrame({"bid_id": []}),
    )

    row = result.loc[result["bid_id"] == "b1"].iloc[0]
    assert row["won"]
    assert row["impression_count"] == 2
    assert row["observed_paying_price"] == 40
    assert not row["paying_price_conflict"]
    assert diagnostics.duplicate_impression_bid_ids == 1
    assert diagnostics.conflicting_paying_price_bid_ids == 0


def test_conflicting_duplicate_paying_prices_are_not_silently_selected() -> None:
    result, diagnostics = build_observed_outcomes(
        _bids(),
        pd.DataFrame({"bid_id": ["b1", "b1"], "paying_price": [40, 41]}),
        pd.DataFrame({"bid_id": []}),
        pd.DataFrame({"bid_id": []}),
    )

    row = result.loc[result["bid_id"] == "b1"].iloc[0]
    assert row["won"]
    assert row["impression_count"] == 2
    assert pd.isna(row["observed_paying_price"])
    assert row["paying_price_conflict"]
    assert diagnostics.conflicting_paying_price_bid_ids == 1


def test_strict_mode_rejects_conflicting_paying_prices() -> None:
    with pytest.raises(ValueError, match="conflicting paying prices"):
        build_observed_outcomes(
            _bids(),
            pd.DataFrame({"bid_id": ["b1", "b1"], "paying_price": [40, 41]}),
            pd.DataFrame({"bid_id": []}),
            pd.DataFrame({"bid_id": []}),
            strict=True,
        )
