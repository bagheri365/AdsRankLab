"""CLI for building one frozen daily analytical table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ads_rank_lab.data.analytical import (
    build_training_day_table,
    diagnostics_to_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one AdsRankLab bid-level analytical table."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/ipinyou.contest.dataset"),
    )
    parser.add_argument("--season", type=int, choices=[2, 3], required=True)
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional parquet output path.",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional JSON diagnostics path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    table, diagnostics = build_training_day_table(
        args.dataset_root,
        season=args.season,
        date=args.date,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            table.to_parquet(args.output, index=False)
        except ImportError as exc:
            raise SystemExit(
                "Writing parquet requires an optional parquet engine such as "
                "pyarrow. Install it before using --output."
            ) from exc

    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(
            json.dumps(diagnostics_to_dict(diagnostics), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    print(diagnostics)


if __name__ == "__main__":
    main()
