import pandas as pd

from ads_rank_lab.policies.value import build_policy_scores


def test_value_scores_multiply_bid_by_probability() -> None:
    frame = pd.DataFrame({"bid_price": ["100", "200"]})

    scores = build_policy_scores(
        frame,
        raw_pctr=[0.01, 0.02],
        intercept_pctr=[0.005, 0.01],
        platt_pctr=[0.008, 0.015],
    )

    assert scores["bid_only"].tolist() == [100.0, 200.0]
    assert scores["pctr_raw"].tolist() == [0.01, 0.02]
    assert scores["bid_x_raw_pctr"].tolist() == [1.0, 4.0]
    assert scores["bid_x_intercept_pctr"].tolist() == [0.5, 2.0]
    assert scores["bid_x_platt_pctr"].tolist() == [0.8, 3.0]
