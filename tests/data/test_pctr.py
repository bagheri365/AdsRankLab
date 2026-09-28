import pandas as pd

from ads_rank_lab.data.pctr import build_won_impression_population


def test_pctr_population_keeps_one_row_per_impression_bid() -> None:
    impressions = pd.DataFrame({
        "bid_id": ["b1", "b1", "b2"],
        "timestamp": ["20130606090000000", "20130606090000001", "20130606100000000"],
        "region": ["1", "1", "2"], "city": ["10", "10", "20"],
        "ad_exchange": ["1", "1", "2"], "slot_width": ["300", "300", "728"],
        "slot_height": ["250", "250", "90"], "slot_visibility": ["1", "1", "0"],
        "slot_format": ["1", "1", "0"], "slot_price": ["0", "0", "5"],
        "advertiser_id": ["a", "a", "b"], "paying_price": ["20", "20", "30"],
        "log_type": ["1", "1", "1"], "key_page": ["x", "x", "y"],
    })
    clicks = pd.DataFrame({"bid_id": ["b2"]})
    result = build_won_impression_population(impressions, clicks)
    assert result["bid_id"].tolist() == ["b1", "b2"]
    assert result["clicked"].tolist() == [0, 1]
    assert "paying_price" not in result.columns
    assert "log_type" not in result.columns
    assert "key_page" not in result.columns


def test_pctr_population_derives_hour_and_weekday() -> None:
    impressions = pd.DataFrame({"bid_id": ["b1"], "timestamp": ["20130606090000000"]})
    clicks = pd.DataFrame({"bid_id": []})
    result = build_won_impression_population(impressions, clicks)
    assert result.loc[0, "hour"] == 9
    assert result.loc[0, "weekday"] == "3"
