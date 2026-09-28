import pandas as pd
import pytest

from ads_rank_lab.data.audit import build_dataset_audit


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "click": pd.Series([0, 1, 0, 1], dtype="Int64"),
            "conversion": pd.Series([0, 0, 1, pd.NA], dtype="Int64"),
            "bid_price": [10.0, 20.0, 30.0, 40.0],
            "paying_price": [5.0, 11.0, 13.0, 17.0],
            "advertiser_id": pd.Series(["a", "a", "b", None], dtype="string"),
        }
    )


def test_dataset_audit_reports_shape_and_valid_schema() -> None:
    audit = build_dataset_audit(sample_frame())

    assert audit["rows"] == 4
    assert audit["column_count"] == 5
    assert audit["schema"]["is_valid"] is True
    assert audit["schema"]["missing_required"] == []


def test_dataset_audit_reports_label_prevalence() -> None:
    audit = build_dataset_audit(sample_frame())

    assert audit["labels"]["click"]["positive"] == 2
    assert audit["labels"]["click"]["prevalence"] == 0.5
    assert audit["labels"]["conversion"]["observed"] == 3
    assert audit["labels"]["conversion"]["positive"] == 1


def test_dataset_audit_reports_numeric_summary() -> None:
    audit = build_dataset_audit(sample_frame())

    bid = audit["numeric"]["bid_price"]
    assert bid["min"] == 10.0
    assert bid["median"] == 25.0
    assert bid["mean"] == 25.0
    assert bid["max"] == 40.0


def test_dataset_audit_reports_missingness_and_cardinality() -> None:
    audit = build_dataset_audit(sample_frame(), top_n=1)

    assert audit["missingness"]["advertiser_id"]["missing"] == 1
    advertiser = audit["categorical"]["advertiser_id"]
    assert advertiser["cardinality"] == 2
    assert advertiser["top_values"] == [{"value": "a", "count": 2}]


def test_dataset_audit_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_dataset_audit(pd.DataFrame())
