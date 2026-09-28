import pandas as pd

from ads_rank_lab.data.preprocess import coerce_numeric, fill_missing_categories


def test_fill_missing_categories_uses_explicit_token() -> None:
    frame = pd.DataFrame({"slot": ["top", None]})

    result = fill_missing_categories(frame, ["slot"])

    assert result["slot"].tolist() == ["top", "__MISSING__"]
    assert frame["slot"].isna().sum() == 1


def test_coerce_numeric_marks_invalid_values_missing() -> None:
    frame = pd.DataFrame({"bid": ["10", "bad", "30"]})

    result = coerce_numeric(frame, ["bid"])

    assert result["bid"].iloc[0] == 10
    assert pd.isna(result["bid"].iloc[1])
    assert result["bid"].iloc[2] == 30
