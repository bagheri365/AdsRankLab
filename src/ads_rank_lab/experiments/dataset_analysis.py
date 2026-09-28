"""Command-line dataset audit for Milestone 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ads_rank_lab.data.audit import build_dataset_audit
from ads_rank_lab.data.load import load_table
from ads_rank_lab.data.schema import canonicalize_columns, coerce_schema_types


def audit_dataset(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    sep: str = "\t",
    top_n: int = 10,
) -> dict:
    """Load, normalize, audit, and optionally persist one RTB table."""
    frame = load_table(input_path, sep=sep)
    frame = canonicalize_columns(frame)
    frame = coerce_schema_types(frame)
    audit = build_dataset_audit(frame, top_n=top_n)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a descriptive AdsRankLab audit from an RTB log file."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/dataset_audit.json"),
        help="JSON output path (default: artifacts/results/dataset_audit.json)",
    )
    parser.add_argument(
        "--sep",
        default="\\t",
        help=r"Input delimiter. Use '\\t' for tab-delimited files.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    separator = "\t" if args.sep == r"\t" else args.sep
    audit = audit_dataset(
        args.input_path,
        output_path=args.output,
        sep=separator,
        top_n=args.top_n,
    )
    print(
        f"Wrote audit for {audit['rows']} rows and "
        f"{audit['column_count']} columns to {args.output}"
    )


if __name__ == "__main__":
    main()
