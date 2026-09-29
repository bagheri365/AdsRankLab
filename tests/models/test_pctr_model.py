import pandas as pd

from ads_rank_lab.models.pctr import PctrBaselineConfig, build_pctr_pipeline, feature_columns, validate_feature_frame


def test_feature_contract_excludes_post_auction_fields() -> None:
    features = set(feature_columns())
    assert "paying_price" not in features
    assert "observed_paying_price" not in features
    assert "won" not in features
    assert "clicked" not in features


def test_pipeline_fits_small_sparse_problem() -> None:
    frame = pd.DataFrame({
        "hour": [9, 9, 10, 10, 11, 11],
        "slot_width": [300, 300, 728, 728, 300, 300],
        "slot_height": [250, 250, 90, 90, 250, 250],
        "slot_price": [0, 0, 5, 5, 0, 0],
        "weekday": ["3"] * 6,
        "region": ["1", "1", "2", "2", "1", "1"],
        "city": ["10", "10", "20", "20", "10", "10"],
        "ad_exchange": ["1", "1", "2", "2", "1", "1"],
        "slot_visibility": ["1", "1", "0", "0", "1", "1"],
        "slot_format": ["1", "1", "0", "0", "1", "1"],
        "advertiser_id": ["a", "a", "b", "b", "a", "a"],
        "clicked": [0, 1, 0, 1, 0, 1],
    })
    validate_feature_frame(frame)
    pipeline = build_pctr_pipeline(PctrBaselineConfig(max_iter=1000))
    pipeline.fit(frame[list(feature_columns())], frame["clicked"])
    probabilities = pipeline.predict_proba(frame[list(feature_columns())])[:, 1]
    assert len(probabilities) == len(frame)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_default_config_uses_tight_stability_settings() -> None:
    config = PctrBaselineConfig()
    assert config.max_iter == 3000
    assert config.tol == 1e-6
