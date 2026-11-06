# SPDX-License-Identifier: Apache-2.0
"""tools/logical_shard_identity.py without a device: the fixed list, the fail-closed verdicts, the summary the
release judges, and the unconditional wiring in the box job. (The shard boundaries, per-shard seeds and fold order
live in compiled Mojo dispatchers that no binding exposes; they are checked on the box by the tool itself.)"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import logical_shard_identity as LS  # noqa: E402


def part(name, sha):
    return dict(name=name, shape=[2], dtype="<f4", sha256=sha * 64)


def doc(name, arms):
    return dict(lane=name, error=None, arms={str(c): dict(parts=p, error=None) for c, p in arms.items()})


def test_the_list_is_short_fixed_and_well_formed():
    assert LS.lane_errors() == []
    assert 6 <= len(LS.LANES) <= 12
    assert LS.COUNTS[0] == 1 and len(LS.COUNTS) >= 2


def test_a_malformed_list_is_refused():
    row = LS.LANES[0]
    assert any("duplicate" in e for e in LS.lane_errors(LS.LANES[:6] + (row,)))
    assert any("not 6..12" in e for e in LS.lane_errors(LS.LANES[:3]))
    bad = (row[0] + "-x", "other.mojo", "no-such-recipe") + row[3:]
    assert any("unknown recipe" in e for e in LS.lane_errors(LS.LANES[:6] + (bad,)))


def test_devices_repeat_the_one_card():
    assert LS.devices_for(1) == (0,)
    assert LS.devices_for(3) == (0, 0, 0)
    assert LS.devices_for(2, 1) == (1, 1)


def test_equal_arms_match_and_a_changed_arm_differs():
    same = {c: [part("a", "1")] for c in LS.COUNTS}
    assert LS.lane_verdict("x", doc("x", same))[0] == "MATCH"
    changed = dict(same)
    changed[LS.COUNTS[-1]] = [part("a", "2")]
    verdict, reason, digests = LS.lane_verdict("x", doc("x", changed))
    assert verdict == "DIFFER" and "n%d" % LS.COUNTS[-1] in reason
    line = LS.shardcheck_line("x", verdict, reason, digests)
    assert line.startswith("SHARDCHECK x single=") and " n2=" in line and " DIFFER " in line


def test_everything_missing_fails_closed():
    assert LS.lane_verdict("x", None)[0] == "FAIL"
    assert LS.lane_verdict("x", dict(lane="x", error="ImportError: no wheel"))[:2] == ("FAIL", "ImportError: no wheel")
    no_arm = doc("x", {1: [part("a", "1")]})
    assert LS.lane_verdict("x", no_arm)[0] == "FAIL"
    empty = doc("x", {c: [] for c in LS.COUNTS})
    verdict, reason, _ = LS.lane_verdict("x", empty)
    assert verdict == "FAIL" and "no digest" in reason
    raised = doc("x", {c: [part("a", "1")] for c in LS.COUNTS})
    raised["arms"]["2"]["error"] = "ValueError: devices must be distinct\nnonnegative"
    verdict, reason, _ = LS.lane_verdict("x", raised)
    assert verdict == "FAIL" and "\n" not in reason


def test_the_release_summary_requires_every_lane_on_the_list_to_match():
    lanes = {n: dict(verdict="MATCH", reason="") for n in LS.lane_names()}
    good = dict(schema=LS.SCHEMA, ok=True, counts=list(LS.COUNTS), lanes=lanes)
    assert LS.summary_errors(good) == []
    assert LS.summary_errors(None) == ["no logical-shard summary"]
    short = dict(good, lanes={n: r for n, r in list(lanes.items())[1:]})
    assert LS.summary_errors(short)
    failed = dict(good, lanes=dict(lanes, **{LS.lane_names()[0]: dict(verdict="FAIL", reason="timed out")}))
    assert any("FAIL timed out" in e for e in LS.summary_errors(failed))
    assert LS.summary_errors(json.loads(json.dumps(dict(good, counts=[1, 2]))))


def test_the_box_job_runs_it_unconditionally_and_it_decides_the_verdict():
    job = (ROOT / "tools" / "ops" / "release_gate_job.sh").read_text()
    lines = job.splitlines()
    at = lines.index("PHASE=shardcheck")
    # top level: not inside the identity phase's `if [ -n "${GATE_IDENTITY_COLUMNS:-}" ]` block (or any other)
    depth = 0
    for line in lines[:at]:
        stripped = line.strip()
        if stripped.startswith("if ") and stripped.endswith("then"):
            depth += 1
        elif stripped == "fi":
            depth -= 1
    assert depth == 0
    assert "tools/logical_shard_identity.py" in job and '[ "$grc" != 0 ] || grc=$src' in job
    # its record comes home with the job's reports
    assert '"shardcheck")' in (ROOT / "tools" / "ops" / "release_gate.py").read_text()
