import json
from pathlib import Path

from ads_rank_lab.experiments.dataset_analysis import audit_dataset


def test_audit_dataset_writes_json_artifact(tmp_path: Path) -> None:
    source = tmp_path / "sample.tsv"
    destination = tmp_path / "audit.json"
    source.write_text(
        "click\tbidprice\tpayprice\tadvertiser\n"
        "0\t10\t5\t100\n"
        "1\t20\t11\t100\n",
        encoding="utf-8",
    )

    audit = audit_dataset(source, output_path=destination)

    assert audit["schema"]["is_valid"] is True
    assert audit["rows"] == 2
    assert destination.exists()

    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted["labels"]["click"]["prevalence"] == 0.5
    assert persisted["categorical"]["advertiser_id"]["cardinality"] == 1
