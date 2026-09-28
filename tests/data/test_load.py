from pathlib import Path

import pandas as pd
import pytest

from ads_rank_lab.data.load import load_table


def test_load_table_reads_tab_delimited_data(tmp_path: Path) -> None:
    source = tmp_path / "sample.tsv"
    source.write_text("click\tbid\n0\t10\n1\t20\n", encoding="utf-8")

    frame = load_table(source)

    assert list(frame.columns) == ["click", "bid"]
    assert frame.shape == (2, 2)


def test_load_table_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_table(tmp_path / "missing.tsv")


def test_load_table_rejects_empty_table(tmp_path: Path) -> None:
    source = tmp_path / "empty.tsv"
    source.write_text("click\\tbid\\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_table(source)
