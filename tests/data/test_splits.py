import pandas as pd

from ads_rank_lab.data.splits import deterministic_split


def test_deterministic_split_is_reproducible() -> None:
    frame = pd.DataFrame({"row_id": range(100)})

    first = deterministic_split(frame, random_state=7)
    second = deterministic_split(frame, random_state=7)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_deterministic_split_partitions_all_rows() -> None:
    frame = pd.DataFrame({"row_id": range(20)})

    splits = deterministic_split(frame, train_fraction=0.6, validation_fraction=0.2)

    assert len(splits.train) == 12
    assert len(splits.validation) == 4
    assert len(splits.test) == 4

    observed = set(splits.train["row_id"]) | set(splits.validation["row_id"]) | set(splits.test["row_id"])
    assert observed == set(frame["row_id"])
