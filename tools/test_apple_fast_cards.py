"""Round trip: the Apple fast card tables reproduce every former card file, value for value.

The former files are read from git at the commit the tables were cut from (``CUT_FROM``); the only
values that may differ are the caller paths the table's ``path_moves`` records.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apple_fast_cards as cards  # noqa: E402

ROOT = cards.ROOT
# The integration head the per-card files were last on (lane apple-consolidate, 2026-11-22).
CUT_FROM = "79f545588a0672aef9838739f6f5cf8e58973763"


def moved(value, moves):
    if isinstance(value, str):
        return moves.get(value, value)
    if isinstance(value, list):
        return [moved(v, moves) for v in value]
    if isinstance(value, dict):
        return {k: moved(v, moves) for k, v in value.items()}
    return value


def former(path):
    result = subprocess.run(["git", "show", f"{CUT_FROM}:{path}"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"cannot read {path} at {CUT_FROM} (fetch full history for this test): {result.stderr.strip()}")
    return json.loads(result.stdout)


@pytest.mark.parametrize("family", sorted(cards.TABLES))
def test_table_reproduces_every_former_card_file(family):
    table = cards.load(family)
    moves = cards.load("perf")["path_moves"]
    paths = cards.legacy_paths(family)
    assert paths, family
    for path in paths:
        assert not (ROOT / path).exists(), f"{path}: the per-card file is back beside its table"
        assert cards.document(path) == moved(former(path), moves), path
    ids = [row["id"] for row in table["rows"]]
    assert len(ids) == len(set(ids)), f"{family}: duplicate card ids"


def test_every_former_card_file_is_in_a_table():
    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", CUT_FROM, "experiments/"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    held = {p for family in cards.TABLES for p in cards.legacy_paths(family)}
    expected = {p for p in listing if p.endswith(".json") and (
        p.startswith(("experiments/apple_fast_classical_20261006/", "experiments/apple_fast_trees/",
                      "experiments/apple_fast_neural_20261006/", "experiments/apple_fast_bf16_20261015/"))
        or (p.startswith("experiments/performance_ideas/") and p.split("/")[2][:1] == "F" and p.split("/")[2][1:3].isdigit())
        or p.startswith("experiments/performance_ideas/AFCL-"))}
    assert expected == held


def test_moved_scripts_exist():
    for old, new in cards.load("perf")["path_moves"].items():
        assert (ROOT / new).is_file() and not (ROOT / old).exists(), (old, new)


def test_catalog_fast_rows_are_generated_from_the_tables():
    # catalog.json's AF.* rows are generated (tools/six_lane_catalog.py --write-fast), never hand-kept.
    import six_lane_catalog
    assert six_lane_catalog.check_fast() == []


@pytest.mark.parametrize("family", sorted(cards.TABLES))
def test_loader_rows_are_the_cards(family):
    for path in cards.legacy_paths(family):
        doc = cards.document(path)
        assert cards.read_json(path) == doc and cards.exists(path)
        assert cards.source_file(path) == cards.table_path(family)
        assert len(cards.digest(path)) == 64
