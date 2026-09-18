"""Recorded host-only routes must be reachable and retain honest evidence scope."""
import json
from pathlib import Path

import pytest

from mojolearn import _verify_all as va
from mojolearn import _verify_reference as ref
from mojolearn import host_surface


ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "bench/results/identity_break/2026-09-18_tokenized-corpus"
LANES = ("bpe-vocabulary", "tokenized-corpus")


@pytest.mark.parametrize("lane", LANES)
def test_host_only_routes_are_selected_without_pending_opt_in(lane):
    harness = va.load_harness()
    table = ref.load_table()
    selected, fixtures = va.select_lanes(harness, table, "cpu", "full", [])
    assert lane in selected
    assert fixtures == list(harness.FIXTURES)
    assert va.select_lanes(harness, table, "cpu", "full", [lane])[0] == [lane]
    family = host_surface.PUBLIC_HOST_ONLY_LANES[lane]
    assert family == "tokenizer"
    assert any(f["family"] == family and f["ships_in_wheel"]
               for f in host_surface.FAMILIES)


@pytest.mark.skipif(not RECORDS.is_dir(), reason="needs committed capture evidence")
@pytest.mark.parametrize("lane", LANES)
def test_host_only_replay_matches_and_both_fault_controls_diverge(lane):
    table = ref.load_table()
    records = {name: json.loads((RECORDS / name).read_text()) for name in (
        "m4.json", "m4.replay.json", "m4.trainer.sabotage.json",
        "m4.host.sabotage.json")}
    for fixture in va.load_harness().FIXTURES:
        key = f"{lane}/{fixture}"
        entry = ref.entry(table, lane, fixture, "train")
        # These operations have only a host implementation. A CPU reference
        # must never be presented as an Apple/NVIDIA/AMD GPU witness.
        assert set(entry["cols"]) == {"cpu"}
        for name, record in records.items():
            cell = record["cells"][key]
            assert len(cell["hashes"]) >= 2
            assert len(set(cell["hashes"])) == 1
            expected = ref.DIVERGENT if "sabotage" in name else ref.IDENTICAL
            assert ref.judge(cell["hashes"][0], entry)[0] == expected
            if "sabotage" in name:
                assert ref.admit(record, str(RECORDS / name)) is not None
        for part, part_entry in table["cells"][key].items():
            assert set(part_entry["cols"]) == {"cpu"}
            if part != "train":
                assert part_entry["ref"].startswith("n/a:")
    assert table["lane_admission"][lane]["policy"] == dict(
        min_repeats=2, input_witness_required=True, property_protocol_required=True)
