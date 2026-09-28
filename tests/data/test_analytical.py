import pandas as pd

from ads_rank_lab.data.analytical import build_analytical_table


def test_orphan_click_is_not_used_as_positive_label() -> None:
    bids = pd.DataFrame(
        {"bid_id": ["b1", "b2"], "bid_price": [100, 100]}
    )
    impressions = pd.DataFrame(
        {"bid_id": ["b1"], "paying_price": [40]}
    )
    clicks = pd.DataFrame(
        {"bid_id": ["b1", "b2"], "paying_price": [40, 0]}
    )
    conversions = pd.DataFrame({"bid_id": []})

    table, diagnostics = build_analytical_table(
        bids,
        impressions,
        clicks,
        conversions,
        season=2,
        date="20130606",
    )

    won = table.loc[table["bid_id"] == "b1"].iloc[0]
    lost = table.loc[table["bid_id"] == "b2"].iloc[0]

    assert won["clicked"]
    assert not lost["clicked"]
    assert not lost["response_observed"]
    assert diagnostics.orphan_click_rows == 1
    assert diagnostics.orphan_click_bid_ids == 1


def test_click_labels_are_observed_only_on_wins() -> None:
    bids = pd.DataFrame(
        {"bid_id": ["b1", "b2", "b3"], "bid_price": [100, 100, 100]}
    )
    impressions = pd.DataFrame(
        {"bid_id": ["b1", "b3"], "paying_price": [40, 50]}
    )
    clicks = pd.DataFrame({"bid_id": ["b3"]})
    conversions = pd.DataFrame({"bid_id": []})

    table, _ = build_analytical_table(
        bids,
        impressions,
        clicks,
        conversions,
        season=2,
        date="20130606",
    )

    assert table["response_observed"].tolist() == [True, False, True]
    assert table["click_label_observed"].tolist() == [True, False, True]
    assert table["clicked"].tolist() == [False, False, True]


def test_orphan_conversion_is_not_used_as_positive_label() -> None:
    bids = pd.DataFrame({"bid_id": ["b1", "b2"]})
    impressions = pd.DataFrame(
        {"bid_id": ["b1"], "paying_price": [40]}
    )
    clicks = pd.DataFrame({"bid_id": []})
    conversions = pd.DataFrame({"bid_id": ["b2"]})

    table, diagnostics = build_analytical_table(
        bids,
        impressions,
        clicks,
        conversions,
        season=2,
        date="20130606",
    )

    assert not table.loc[table["bid_id"] == "b2", "converted"].item()
    assert diagnostics.orphan_conversion_rows == 1
    assert diagnostics.orphan_conversion_bid_ids == 1


def test_duplicate_impression_diagnostics_flow_through() -> None:
    bids = pd.DataFrame({"bid_id": ["b1"]})
    impressions = pd.DataFrame(
        {"bid_id": ["b1", "b1"], "paying_price": [40, 40]}
    )
    clicks = pd.DataFrame({"bid_id": []})
    conversions = pd.DataFrame({"bid_id": []})

    table, diagnostics = build_analytical_table(
        bids,
        impressions,
        clicks,
        conversions,
        season=2,
        date="20130606",
    )

    assert table.loc[0, "impression_count"] == 2
    assert diagnostics.duplicate_impression_bid_ids == 1
    assert diagnostics.conflicting_paying_price_bid_ids == 0


def test_conflicting_price_remains_flagged_and_missing() -> None:
    bids = pd.DataFrame({"bid_id": ["b1"]})
    impressions = pd.DataFrame(
        {"bid_id": ["b1", "b1"], "paying_price": [40, 41]}
    )
    clicks = pd.DataFrame({"bid_id": []})
    conversions = pd.DataFrame({"bid_id": []})

    table, diagnostics = build_analytical_table(
        bids,
        impressions,
        clicks,
        conversions,
        season=2,
        date="20130606",
    )

    assert table.loc[0, "paying_price_conflict"]
    assert pd.isna(table.loc[0, "observed_paying_price"])
    assert diagnostics.conflicting_paying_price_bid_ids == 1
