import pandas as pd
import pytest

from ads_rank_lab.data.schema import (
    canonicalize_column_name,
    canonicalize_columns,
    coerce_schema_types,
    validate_schema,
)


def test_canonicalize_known_ipinyou_names() -> None:
    assert canonicalize_column_name("bidprice") == "bid_price"
    assert canonicalize_column_name("payprice") == "paying_price"
    assert canonicalize_column_name("slotprice") == "floor_price"
    assert canonicalize_column_name("advertiser") == "advertiser_id"


def test_canonicalize_columns_preserves_unknown_fields() -> None:
    frame = pd.DataFrame(columns=["click", "bidprice", "custom_feature"])

    result = canonicalize_columns(frame)

    assert list(result.columns) == ["click", "bid_price", "custom_feature"]


def test_canonicalize_columns_rejects_collisions() -> None:
    frame = pd.DataFrame(columns=["bidprice", "bid_price"])

    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_columns(frame)


def test_validate_schema_reports_missing_required_and_unknown() -> None:
    frame = pd.DataFrame(
        {
            "click": [0],
            "bid_price": [10],
            "custom_feature": ["x"],
        }
    )

    result = validate_schema(frame)

    assert not result.is_valid
    assert result.missing_required == ("paying_price",)
    assert result.unknown_columns == ("custom_feature",)


def test_validate_schema_accepts_minimum_required_fields() -> None:
    frame = pd.DataFrame(
        {
            "click": [0, 1],
            "bid_price": [10, 20],
            "paying_price": [5, 11],
        }
    )

    result = validate_schema(frame)

    assert result.is_valid
    assert result.missing_required == ()


def test_coerce_schema_types() -> None:
    frame = pd.DataFrame(
        {
            "click": ["0", "1"],
            "conversion": [0, 1],
            "bid_price": ["10", "20"],
            "paying_price": ["5", "bad"],
            "advertiser_id": [123, 456],
        }
    )

    result = coerce_schema_types(frame)

    assert str(result["click"].dtype) == "Int64"
    assert result["bid_price"].tolist() == [10, 20]
    assert pd.isna(result["paying_price"].iloc[1])
    assert str(result["advertiser_id"].dtype) == "string"


def test_coerce_schema_types_rejects_nonbinary_label() -> None:
    frame = pd.DataFrame(
        {
            "click": [0, 2],
            "bid_price": [10, 20],
            "paying_price": [5, 11],
        }
    )

    with pytest.raises(ValueError, match="binary"):
        coerce_schema_types(frame)
