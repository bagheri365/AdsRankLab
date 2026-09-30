import pandas as pd
from ads_rank_lab.evaluation.observed_cost import advertiser_observed_cost, summarize_observed_cost, valid_observed_cost_population

def test_valid_observed_cost_population_removes_missing_and_conflicts():
    frame = pd.DataFrame({
        "observed_paying_price":[10,None,30,40],
        "bid_price":[20,20,50,60],
        "clicked":[0,1,1,0],
        "advertiser_id":["a","a","b","b"],
        "paying_price_conflict":[False,False,True,False],
    })
    result = valid_observed_cost_population(frame)
    assert result["observed_paying_price"].tolist() == [10.0,40.0]

def test_summarize_observed_cost_uses_selected_rows():
    frame = pd.DataFrame({
        "observed_paying_price":[10.0,20.0,30.0,40.0],
        "bid_price":[20.0,40.0,60.0,80.0],
        "clicked":[1.0,0.0,1.0,0.0],
        "advertiser_id":["a","a","b","b"],
    })
    result = summarize_observed_cost(frame, scores=[4,3,2,1], fraction=0.5)
    assert result["rows"] == 2
    assert result["observed_clicks"] == 1
    assert result["observed_total_paying_price"] == 30.0
    assert result["observed_cost_per_click"] == 30.0

def test_advertiser_observed_cost_reports_cost_share():
    frame = pd.DataFrame({
        "observed_paying_price":[10.0,30.0,20.0,40.0],
        "bid_price":[20.0,50.0,30.0,60.0],
        "clicked":[1.0,0.0,0.0,1.0],
        "advertiser_id":["a","a","b","b"],
    })
    result = advertiser_observed_cost(frame, scores=[4,3,2,1], policy="example", fraction=0.5)
    a = result.loc[result["advertiser_id"]=="a"].iloc[0]
    assert a["selected_rows"] == 2
    assert a["observed_cost_share"] == 1.0
    assert a["observed_ctr"] == 0.5
