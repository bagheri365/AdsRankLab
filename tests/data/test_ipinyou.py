import bz2
from pathlib import Path

import pandas as pd
import pytest

from ads_rank_lab.data.ipinyou import (
    BID_COLUMNS_S23,
    EVENT_COLUMNS_S23,
    coerce_event_types,
    discover_training_logs,
    read_event_log,
    read_training_day,
)


def _write_bz2(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _bid_row() -> list[str]:
    return [
        "bid-1", "20130606000104008", "user-1", "Mozilla/5.0",
        "180.127.189.*", "80", "87", "1", "example.com", "url-hash", "",
        "slot-1", "300", "250", "1", "1", "0", "creative-1", "227",
        "3427", "10006,10063",
    ]


def _event_row(log_type: str) -> list[str]:
    return [
        "bid-1", "20130606000104192", log_type, "user-1", "Mozilla/5.0",
        "114.100.37.*", "106", "117", "1", "example.com", "url-hash",
        "null", "slot-1", "950", "90", "0", "1", "0", "creative-1",
        "227", "207", "key-page", "3427", "10063,10684",
    ]


def test_supported_schema_widths_match_raw_samples() -> None:
    assert len(BID_COLUMNS_S23) == 21
    assert len(EVENT_COLUMNS_S23) == 24


def test_read_bid_log_preserves_raw_fields(tmp_path: Path) -> None:
    source = tmp_path / "bid.20130606.txt.bz2"
    _write_bz2(source, [_bid_row()])

    frame = read_event_log(source, season=2, event_type="bid")

    assert list(frame.columns) == list(BID_COLUMNS_S23)
    assert frame.loc[0, "bid_id"] == "bid-1"
    assert frame.loc[0, "bid_price"] == "227"
    assert frame.loc[0, "advertiser_id"] == "3427"
    assert pd.isna(frame.loc[0, "url_id"])


@pytest.mark.parametrize(
    ("event_type", "log_type"),
    [("imp", "1"), ("clk", "2"), ("conv", "3")],
)
def test_read_outcome_event_log_validates_log_type(
    tmp_path: Path,
    event_type: str,
    log_type: str,
) -> None:
    source = tmp_path / f"{event_type}.20130606.txt.bz2"
    _write_bz2(source, [_event_row(log_type)])

    frame = read_event_log(source, season=2, event_type=event_type)

    assert list(frame.columns) == list(EVENT_COLUMNS_S23)
    assert frame.loc[0, "paying_price"] == "207"
    assert frame.loc[0, "log_type"] == log_type


def test_read_event_log_rejects_wrong_log_type(tmp_path: Path) -> None:
    source = tmp_path / "clk.20130606.txt.bz2"
    _write_bz2(source, [_event_row("1")])

    with pytest.raises(ValueError, match="unexpected log_type"):
        read_event_log(source, season=2, event_type="clk")


def test_discover_training_logs_is_date_sorted(tmp_path: Path) -> None:
    directory = tmp_path / "training2nd"
    _write_bz2(directory / "bid.20130607.txt.bz2", [_bid_row()])
    _write_bz2(directory / "bid.20130606.txt.bz2", [_bid_row()])

    logs = discover_training_logs(tmp_path, season=2, event_type="bid")

    assert [log.date for log in logs] == ["20130606", "20130607"]


def test_read_training_day_builds_original_release_path(tmp_path: Path) -> None:
    source = tmp_path / "training3rd" / "imp.20131019.txt.bz2"
    _write_bz2(source, [_event_row("1")])

    frame = read_training_day(
        tmp_path,
        season=3,
        event_type="imp",
        date="20131019",
    )

    assert len(frame) == 1
    assert frame.loc[0, "advertiser_id"] == "3427"


def test_coerce_event_types_converts_prices_without_touching_ids() -> None:
    frame = pd.DataFrame(
        {
            "bid_id": ["001"],
            "timestamp": ["20130606000104008"],
            "bid_price": ["227"],
            "paying_price": ["207"],
            "advertiser_id": ["3427"],
        }
    )

    result = coerce_event_types(frame)

    assert result.loc[0, "bid_price"] == 227
    assert result.loc[0, "paying_price"] == 207
    assert result.loc[0, "bid_id"] == "001"
    assert result.loc[0, "timestamp"] == "20130606000104008"


def test_season_one_is_not_silently_parsed_with_season_two_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Seasons 2 and 3"):
        discover_training_logs(tmp_path, season=1, event_type="bid")  # type: ignore[arg-type]
