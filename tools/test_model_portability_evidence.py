"""Evidence must reject missing vendors and unequal model inputs or outputs."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import check_model_portability_evidence as evidence

BUNDLE = Path(__file__).resolve().parents[1] / "bench/results/model_portability/2026-10-20"


@pytest.fixture
def record(tmp_path):
    return Path(shutil.copytree(BUNDLE, tmp_path / "record"))


def rewrite_record(root, name, change):
    path = root / name
    row = json.loads(path.read_text())
    change(row)
    path.write_text(json.dumps(row))
    data = path.read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"][name].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_retained_record_matches_review():
    assert evidence.review(BUNDLE) == json.loads((BUNDLE / "review.json").read_text())


def test_missing_vendor_model_is_not_admitted(record):
    path = record / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["models"]["metal"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model coverage"):
        evidence.review(record)


def test_different_model_input_is_not_admitted(record):
    rewrite_record(record, "raw/models/metal/resnet18.json",
                   lambda row: row.update(input_digest="0" * 16))
    with pytest.raises(ValueError, match="input_digest"):
        evidence.review(record)


def test_different_model_output_is_not_admitted(record):
    rewrite_record(record, "raw/models/hip/resnet18.json",
                   lambda row: row["ours"].update(digest="0" * 16))
    with pytest.raises(ValueError, match="output mismatch"):
        evidence.review(record)


def test_different_saved_model_is_not_admitted(record):
    rewrite_record(record, "raw/reload/metal.json",
                   lambda row: row["model"].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="saved model: sha256"):
        evidence.review(record)


def test_changed_retained_bytes_are_not_admitted(record):
    with (record / "raw/training/metal.log").open("a") as log:
        log.write("\nchanged\n")
    with pytest.raises(ValueError, match="byte count"):
        evidence.review(record)
