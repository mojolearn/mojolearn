# SPDX-License-Identifier: Apache-2.0
"""The release pass's anchor (the last FINISHED pass on the backend) and the
opt-in sharded Metal fan-out. No GPU, no fit: records are written by hand and
the fan-out children are plain Python processes. Run from tools/:

    python3 -m pytest -q test_release_pass_anchor.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metal_fanout                                             # noqa: E402
import verify_lanes                                             # noqa: E402

ROOT = verify_lanes.ROOT
FIX = verify_lanes.APPLE_PASS_FIXTURES


def _rev(ref):
    return subprocess.run(["git", "-C", ROOT, "rev-parse", ref], capture_output=True, text=True,
                          check=True).stdout.strip()


try:
    HEAD, PARENT = _rev("HEAD"), _rev("HEAD~1")
except (subprocess.CalledProcessError, FileNotFoundError) as exc:
    # A shipped tarball or an exported tree has no .git; the anchor tests
    # need two real commits to name, so they skip rather than abort the
    # collection of every file after this one.
    pytest.skip(f"not a git checkout with two commits: {exc}", allow_module_level=True)


def record(base, commit, backend="metal", covers=None, dirty_lanes=(), complete=True, column=True,
           fixtures=FIX, shards=1, age=0.0, **extra):
    d = Path(base) / commit[:12] / backend
    d.mkdir(parents=True, exist_ok=True)
    manifest = dict(commit=commit, backend=backend, fixtures=fixtures, repeats=1, probe_group="core",
                    covers=covers if covers is not None else dict(mode="all"),
                    dirty_lanes=list(dirty_lanes) if dirty_lanes is not None else None,
                    metal_shards=shards, **extra)
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "run-summary.json").write_text(json.dumps(dict(complete=complete, validation_failures=[])))
    if column:
        (d / "column.json").write_text("{}")
    t = time.time() - age
    os.utime(d / "run-summary.json", (t, t))
    return d


def test_no_records_means_no_anchor(tmp_path):
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) is None
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path / "missing")) is None


def test_a_complete_clean_full_pass_anchors(tmp_path):
    record(tmp_path, PARENT)
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) == PARENT
    assert verify_lanes.verified_anchor("cpu", FIX, base=str(tmp_path)) is None, "another backend's pass"


@pytest.mark.parametrize("change", [
    dict(complete=False), dict(column=False), dict(dirty_lanes=["ridge"]), dict(dirty_lanes=None),
    dict(fixtures="base"), dict(shards=2), dict(covers=dict(mode="since-record")),
    dict(covers=dict(mode="whatever")),
])
def test_anything_short_of_a_clean_finished_pass_does_not_anchor(tmp_path, change):
    record(tmp_path, PARENT, **change)
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) is None, change


def test_a_record_without_the_new_fields_never_anchors(tmp_path):
    d = record(tmp_path, PARENT)
    manifest = json.loads((d / "manifest.json").read_text())
    del manifest["covers"], manifest["dirty_lanes"]
    (d / "manifest.json").write_text(json.dumps(manifest))
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) is None


def test_a_commit_the_repository_does_not_have_never_anchors(tmp_path):
    record(tmp_path, "f" * 40)
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) is None


def test_a_narrow_pass_anchors_only_on_a_chain_down_to_a_full_one(tmp_path):
    record(tmp_path, HEAD, covers=dict(mode="since-record", since=PARENT))
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) is None, \
        "a narrow pass with nothing verified underneath vouched for itself"
    record(tmp_path, PARENT, age=100)
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) == HEAD
    tag = subprocess.run(["git", "-C", ROOT, "rev-list", "-n", "1", "v0.8.8"], capture_output=True,
                         text=True).stdout.strip()
    if tag:
        other = tmp_path / "tagged"
        record(other, HEAD, covers=dict(mode="since-tag", since=tag, tag="v0.8.8"))
        assert verify_lanes.verified_anchor("metal", FIX, base=str(other)) == HEAD


def test_the_newest_qualifying_record_wins(tmp_path):
    record(tmp_path, PARENT, age=100)
    record(tmp_path, HEAD, age=10)
    assert verify_lanes.verified_anchor("metal", FIX, base=str(tmp_path)) == HEAD


def test_a_pass_diffs_against_the_anchor_before_the_tag(monkeypatch, tmp_path, capsys):
    record(tmp_path, PARENT)
    monkeypatch.setenv("MOJOLEARN_RELEASE_CHECK_DIR", str(tmp_path))
    monkeypatch.setattr(verify_lanes, "last_release_tag", lambda: pytest.fail("the tag was consulted"))
    seen = {}

    def selection(args):
        seen.update(changed_since=args.changed_since, all=args.all)
        return ["ridge"], dict(mode="derived", fallback=False), None
    monkeypatch.setattr(verify_lanes, "_selection", selection)
    monkeypatch.setattr(verify_lanes, "dirty_paths", lambda: [])
    assert verify_lanes.main(["--apple-pass", "--plan"]) == 0
    assert seen == dict(changed_since=PARENT, all=False)
    out = capsys.readouterr().out
    assert "last completed metal pass" in out and "anchors the next one" in out


def test_a_dirty_tree_is_said_to_be_unable_to_anchor(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MOJOLEARN_RELEASE_CHECK_DIR", str(tmp_path))
    monkeypatch.setattr(verify_lanes, "last_release_tag", lambda: "")
    monkeypatch.setattr(verify_lanes, "_selection",
                        lambda args: (["ridge"], dict(mode="all", fallback=False), None))
    # An untracked new binding source cannot be attributed, so it reads as
    # every lane. (A tracked file that is not really dirty would read as
    # "docstrings only" against HEAD, correctly.)
    monkeypatch.setattr(verify_lanes, "dirty_paths", lambda: ["bindings/_mojolearn_armprobe_new.mojo"])
    assert verify_lanes.main(["--apple-pass", "--plan"]) == 0
    assert "cannot anchor the next one" in capsys.readouterr().out


# ---------------------------------------------------------------- sharded Metal

@pytest.mark.parametrize("argv", [
    ["--lane", "ridge", "--metal-shards", "2", "--plan"],                        # CPU backend
    ["--lane", "ridge", "--apple-pass", "--metal-shards", "4", "--plan"],        # over the cap
    ["--lane", "ridge", "--apple-pass", "--metal-shards", "0", "--plan"],
    ["--lanes", "ridge,ols", "--apple-pass", "--metal-shards", "2", "--shards", "3", "--plan"],
])
def test_metal_shards_refuse_outside_their_scope(argv):
    with pytest.raises(SystemExit):
        verify_lanes.main(argv)


def test_metal_shards_set_the_split(monkeypatch, capsys):
    monkeypatch.setattr(verify_lanes.lane_select, "shard",
                        lambda lanes, shards: ([[n] for n in lanes][:shards], [1] * shards))
    assert verify_lanes.main(["--lanes", "ridge,ols", "--apple-pass", "--metal-shards", "2", "--plan"]) == 0
    out = capsys.readouterr().out
    assert out.count("# shard ") == 2


def test_fanout_codes_fail_closed(tmp_path):
    assert verify_lanes._fanout_codes(str(tmp_path), 2, 0) == {0: 1, 1: 1}, "no codes file is a failure"
    assert verify_lanes._fanout_codes(str(tmp_path), 2, 124) == {0: 124, 1: 124}
    (tmp_path / "fanout.codes.json").write_text(json.dumps({"0": 0}))
    assert verify_lanes._fanout_codes(str(tmp_path), 2, 0) == {0: 1, 1: 1}, "a missing shard is a failure"
    (tmp_path / "fanout.codes.json").write_text(json.dumps({"0": 0, "1": 0}))
    assert verify_lanes._fanout_codes(str(tmp_path), 2, 0) == {0: 0, 1: 0}
    assert verify_lanes._fanout_codes(str(tmp_path), 2, 124) == {0: 124, 1: 124}, \
        "a slot that timed out cannot report its shards as passed"


def test_the_fanout_runs_children_at_once_and_reports_each(tmp_path):
    code = ("import pathlib,sys,time\n"
            "root=pathlib.Path(sys.argv[1]); (root/sys.argv[2]).touch(); end=time.monotonic()+5\n"
            "while not (root/sys.argv[3]).exists():\n"
            "    if time.monotonic()>end: raise SystemExit(8)\n"
            "    time.sleep(.01)\n"
            "raise SystemExit(int(sys.argv[4]))\n")
    spec = dict(codes=str(tmp_path / "codes.json"), children=[
        dict(cmd=[sys.executable, "-c", code, str(tmp_path), "a", "b", "0"], log=str(tmp_path / "a.log")),
        dict(cmd=[sys.executable, "-c", code, str(tmp_path), "b", "a", "3"], log=str(tmp_path / "b.log"))])
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    assert metal_fanout.main([str(tmp_path / "spec.json")]) == 1
    assert json.loads((tmp_path / "codes.json").read_text()) == {"0": 0, "1": 3}


def test_run_local_holds_one_metal_slot_for_every_shard(monkeypatch, tmp_path):
    """The fan-out path end to end, with fake shards that each wait for the
    other: both must be running at once, under ONE mac_slot lease."""
    monkeypatch.setenv("MOJOLEARN_MAC_SLOT_BASE", str(tmp_path / "slots"))
    monkeypatch.setenv("MOJOLEARN_METAL_LOCK", str(tmp_path / "gpu"))
    monkeypatch.setenv("MOJOLEARN_METAL_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("MAC_SLOTS", "2")
    code = ("import os,pathlib,sys,time\n"
            "root=pathlib.Path(sys.argv[1]); (root/sys.argv[2]).write_text(os.environ.get('MOJOLEARN_SLOT_TOKEN',''))\n"
            "end=time.monotonic()+5\n"
            "while not (root/sys.argv[3]).exists():\n"
            "    if time.monotonic()>end: raise SystemExit(8)\n"
            "    time.sleep(.01)\n")
    monkeypatch.setattr(verify_lanes, "_identity_break_cmd",
                        lambda group, *a: [sys.executable, "-c", code, str(tmp_path), group[0],
                                           "b" if group[0] == "a" else "a"])
    started = time.monotonic()
    args = SimpleNamespace(backend="metal", host_dir=None, jobs=1, started=started, deadline=started + 20,
                           budget=20, timeout=15, wait_timeout=5, metal_shards=2)
    out = tmp_path / "out"
    out.mkdir()
    _, codes, _ = verify_lanes._run_local([["a"], ["b"]], [1, 1], args, str(out))
    assert codes == {0: 0, 1: 0}
    tokens = {(tmp_path / n).read_text() for n in ("a", "b")}
    assert len(tokens) == 1 and "" not in tokens, "both shards must run under the same held slot"
    assert not (tmp_path / "gpu").exists(), "the Metal lock was not released"


def test_a_cpu_pass_beside_the_apple_pass_leaves_it_one_slot(monkeypatch, capsys):
    """tools/release_check.py runs both passes at once; the CPU pass takes
    MAC_SLOTS - 1 shards so the Metal job's slot is never contended."""
    monkeypatch.setenv("MAC_SLOTS", "5")
    monkeypatch.setenv("MOJOLEARN_CPU_PASS_SLOTS", "4")
    assert verify_lanes.main(["--lanes", "ridge,ols,kmeans,knn,lasso", "--cpu-pass", "--plan"]) == 0
    assert "backend=cpu jobs=4" in capsys.readouterr().out
    monkeypatch.setenv("MOJOLEARN_CPU_PASS_SLOTS", "6")
    with pytest.raises(SystemExit):
        verify_lanes.main(["--lanes", "ridge", "--cpu-pass", "--plan"])
