# SPDX-License-Identifier: Apache-2.0
"""tools/identity_wheel_record.py: the one identity run on the built wheels. A fake recorder stands in for
record_identity_column.sh (no venv, no device, no lane runs)."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import identity_wheel_record as R  # noqa: E402
import release_identity_wheels as RIW  # noqa: E402

FAKE = r'''#!/bin/bash
# a fake record_identity_column.sh: a chunk "records" (the lane choice is identity_wheel_record.py's own:
# --record-families, tools/release_families.py)
label=$1; out=$2
[ "$RECORD_FROM_INSTALLED" = 1 ] && case "$LANE_SELECTION" in family-chunk|full-chunk) true;; *) false;; esac \
  || { echo "bad env"; exit 9; }
echo "{}" > "$out/$label.lanes-x.identical.json"
echo "DEVICE ${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-none}} LANES $LANES"
echo "CPU_THREADS ${CPU_THREADS:-unset}"
echo "RECORDED $label.lanes-x rc=0 admissible=yes json=$out/$label.lanes-x.identical.json"
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_recorder.sh"
    script.write_text(FAKE)
    monkeypatch.setattr(R, "RECORDER", script)
    monkeypatch.setattr(R.Run, "install", lambda self: (pathlib.Path(sys.executable),
                                                        self.out / "installed-wheels.txt"))
    monkeypatch.setattr(R, "gpu_free_gib", lambda vendor: 40.0)
    # the extended reference captures need an installed wheel; their own test is below
    monkeypatch.setattr(R.Run, "capture_extended", lambda self, py, columns, devices: [])
    return monkeypatch


def test_extended_references_are_captured_once_per_evidence_column_all_at_once(tmp_path, monkeypatch):
    launched = []

    class Proc:
        def __init__(self, argv, **kw):
            launched.append((argv, kw["env"].get("CUDA_VISIBLE_DEVICES")))
            out = argv[argv.index("--out" if "--out" in argv else "--output") + 1]
            if "verify-partitioned-program" not in argv:
                pathlib.Path(out).write_text("{}")

        def wait(self):
            return 0
    run = R.Run(args(tmp_path, "--columns", "nvidia,ptx", "--label-device", "l40s", "--gpus", "2"), popen=Proc)
    rows = run.capture_extended(pathlib.Path(sys.executable), ["nvidia", "ptx"], [0, 1])
    assert len(launched) == 3 and all(dev == "0" for _, dev in launched)
    assert ["--vendor", "cuda"] == launched[0][0][launched[0][0].index("--vendor"):][:2]
    by = {r["profile"]: r["rc"] for r in rows}
    assert by == {"verify-graph": 0, "verify-distributed-program": 0, "verify-partitioned-program": 1}
    assert (tmp_path / "out" / "extended-references" / "nvidia" / "verify-graph.json").is_file()


def args(tmp_path, *extra):
    w = tmp_path / "w.whl"
    w.write_bytes(b"w")
    (tmp_path / "out").mkdir(exist_ok=True)
    (tmp_path / "out" / "installed-wheels.txt").write_text("w.whl " + "a" * 64 + "\n")
    return R.parse(["--wheel", str(w), "--sha256", "w.whl=" + "a" * 64, "--out", str(tmp_path / "out"), *extra])


def test_slots_put_each_column_on_its_own_gpu_when_there_are_enough():
    assert R.plan_slots(["nvidia", "ptx"], [0, 1], 2) == {"nvidia": [(0, 0), (0, 1)], "ptx": [(1, 0), (1, 1)]}
    assert R.plan_slots(["nvidia", "ptx"], [0], 1) == {"nvidia": [(0, 0)], "ptx": [(0, 0)]}
    assert R.plan_slots(["nvidia"], [0, 1, 2], 1) == {"nvidia": [(0, 0), (1, 0), (2, 0)]}
    assert R.plan_slots(["cpu"], [], 3) == {"cpu": [(None, 0), (None, 1), (None, 2)]}


def test_workers_use_every_core_bounded_by_memory():
    assert R.auto_workers(0, 4, None) == 1 and R.auto_workers(0, 16, None) == 5
    assert R.auto_workers(1, 32, 80.0) == 26         # cores: 32 // 1 = 32, GPU memory: 80 // 3 = 26
    assert R.auto_workers(1, 13, 44.0) == 13         # the L40S of nvc-ab-20261126T214026Z-7116a8 (was 3 workers)
    assert R.auto_workers(1, 13, 44.0, host_gib=20.0) == 4   # host memory: (20 - 4 kept free) // 4
    assert R.auto_workers(2, 16, None) == 1
    assert R.auto_workers(0, 16, None, host_gib=8.0) == 1    # (8 - 4) // 4
    assert R.auto_workers(0, 16, None, host_gib=3.0) == 1    # less than the reserve still plans one worker


def test_the_apple_plan_on_a_ten_core_mac_is_bounded_by_memory_not_by_a_3_thread_pool(monkeypatch):
    """lane/verify-speed-1 (2026-11-28): the 0.9.0 Apple record planned 10 // 3 = 3 workers. A GPU column's worker runs
    one host thread, so the cores allow 10; the Metal working set of a 16 GiB Mac (2/3, 10.7 GiB) at 1 GiB a worker
    allows 10; host memory decides. lane/verify-speed-2 (2026-11-28): at the measured 10.06 GiB worker of the 0.9.0
    record and 4 GiB kept free, 10 GiB available plans ONE worker; at the steady workers' 2.2 GiB it plans two."""
    metal = 16.0 * R.METAL_WORKING_SET_SHARE
    assert R.auto_workers(0, 10, None, host_gib=10.0, column="apple", threads=1, metal_gib=metal) == 1
    assert R.auto_workers(0, 10, None, column="apple", threads=1, metal_gib=metal) == 1   # unreadable memory: one
    monkeypatch.setitem(R.PEAKS, "apple", 2.2)
    assert R.auto_workers(0, 10, None, host_gib=10.0, column="apple", threads=1, metal_gib=metal) == 2  # 6 // 2.75
    assert R.auto_workers(0, 10, None, host_gib=40.0, column="apple", threads=1, metal_gib=metal) == 10
    assert R.auto_workers(0, 10, None, host_gib=40.0, column="apple", threads=1, metal_gib=4.0) == 4
    assert R.auto_workers(0, 10, None, host_gib=40.0, column="apple", threads=3) == 3   # the 0.9.0 plan, pinned threads
    assert R.auto_workers(0, 4, None, column="cpu") == 1                    # the CPU column keeps its 3-thread pool


def test_a_measured_peak_replaces_the_estimate(monkeypatch):
    assert R.worker_gib("apple") == R.PEAKS["apple"] * R.PEAK_HEADROOM and R.worker_gib("cpu") == R.HOST_GIB_PER_WORKER
    monkeypatch.setitem(R.PEAKS, "apple", 0.8)
    assert R.worker_gib("apple") == 0.8 * R.PEAK_HEADROOM
    assert R.auto_workers(0, 10, None, host_gib=10.0, column="apple", threads=1) == 6   # (10 - 4) // 1.0
    monkeypatch.delitem(R.PEAKS, "apple")
    monkeypatch.setitem(R.ESTIMATES, "apple", 1.5)
    assert R.worker_gib("apple") == 1.5


def test_threads_per_worker_one_on_a_small_cell_gpu_column(monkeypatch):
    monkeypatch.delenv("RECORD_CPU_THREADS", raising=False)
    assert R.record_threads("cpu", {"a": 0.1}) == R.CPU_THREADS == 3
    assert R.record_threads("apple", {"a": 4.0, "b": 2.0}) == R.GPU_COLUMN_THREADS == 1
    assert R.record_threads("nvidia", {}) == 1
    assert R.record_threads("apple", {"a": 30.0, "b": 20.0}) == R.CPU_THREADS   # cells of minutes keep the pool
    monkeypatch.setenv("RECORD_CPU_THREADS", "2")
    assert R.record_threads("apple", {"a": 4.0}) == 2 and R.record_threads("cpu", {}) == 2


def test_a_column_packs_by_its_own_seconds_and_scales_the_rest():
    shipped = {"robust": 1.2, "trees": 50.0, "knn": 2.0, "new": 4.0}
    own = {"robust": 49.0, "trees": 60.0, "knn": 6.0}                     # ratios 40.8, 1.2, 3.0 -> median 3.0
    got = R.column_hints("apple", shipped=shipped, own=own)
    assert got["robust"] == 49.0 and got["trees"] == 60.0 and got["new"] == 12.0
    assert R.column_hints("nvidia", shipped=shipped, own={}) == shipped
    cost = R.lane_costs(list(got), hints=got)
    assert R.chunks(list(got), 2, cost)[0][0] == "trees" and R.chunks(list(got), 2, cost)[1][0] == "robust"
    committed = R.read_hints(R.COLUMN_HINTS / "apple.json")
    assert committed and all(v >= 0 for v in committed.values())


def test_macos_available_memory_reads_vm_stat():
    text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      100.\n"
            "Pages active:   999.\nPages inactive:  200.\nPages speculative:  50.\nPages purgeable:  10.\n")

    class P:
        stdout = text
    assert R.darwin_available_bytes(lambda *a, **k: P()) == 16384 * 360
    assert R.darwin_available_bytes(lambda *a, **k: (_ for _ in ()).throw(OSError("no vm_stat"))) is None


def test_cell_means_read_the_cell_lines(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("# CELL knn/base STABLE 3.0s\n# CELL knn/odd STABLE 1.0s\n# CELL ols/base STABLE 0.5s\n"
                   "# CELL lm/base N/A n/a:one-sided: no GPU implementation\n")
    assert R.cell_means([log]) == {"knn": 2.0, "ols": 0.5}


def test_an_apple_record_plans_from_memory_and_runs_one_thread_a_worker(tmp_path, fake, capsys):
    fake.delenv("RECORD_CPU_THREADS", raising=False)
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["a", "b", "c", "d", "e", "f", "g", "h"], 0))
    fake.setattr(R, "host_available_gib", lambda *a, **k: 10.0)
    fake.setattr(R, "darwin_memsize_gib", lambda *a, **k: 16.0)
    fake.setattr(R, "alloc_cores", lambda: 10)
    fake.setitem(R.PEAKS, "apple", 2.2)   # the steady workers' figure: (10 - 4) // (2.2 x 1.25) = 2 workers
    a = args(tmp_path, "--columns", "apple", "--label-device", "m4")
    assert R.Run(a).main() == 0
    out = capsys.readouterr().out
    assert "busy plan: 2 worker(s) of 2" in out and "1 CPU thread(s) per worker" in out
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["threads_per_worker"] == {"apple": 1} and len(doc["columns"]["apple"]["chunks"]) == 2
    log = (tmp_path / "out" / "apple-m4" / "w0" / "record.log").read_text()
    assert "CPU_THREADS 1" in log


def test_chunks_pack_equal_cost_longest_lane_first():
    # equal costs deal round robin, as before
    assert R.chunks(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]] and R.chunks([], 3) == []
    cost = {"trees": 180.0, "big": 100.0, "x": 40.0, "y": 40.0, "z": 30.0, "w": 10.0}
    got = R.chunks(list(cost), 2, cost)
    assert got[0][0] == "trees" and got[1][0] == "big"
    loads = sorted(sum(cost[x] for x in c) for c in got)
    assert loads == [190.0, 210.0]          # round robin would give 250 / 150
    assert sorted(x for c in got for x in c) == sorted(cost)


def test_lane_costs_read_the_shipped_hints_and_default_to_the_family(tmp_path):
    """rel-fix-3: a lane with no hint packs at its family's slowest measured lane (longest shared name prefix), else
    the 90th percentile of every lane; never the median (42 shipped-table lanes packed at 0.41 s)."""
    hints = tmp_path / "cell_seconds.json"
    hints.write_text(json.dumps(dict(format="x", seconds={"a": 10.0, "b": 2.0, "c": 4.0, "gbdt-x": 5.0,
                                                          "gbdt-y": 3.0, "@models": 60.0})))
    got = R.lane_costs(["a", "b", "c", "new", "gbdt-new"], hints)
    assert got == {"a": 10.0 + R.LANE_OVERHEAD_S, "b": 2.0 + R.LANE_OVERHEAD_S, "c": 4.0 + R.LANE_OVERHEAD_S,
                   "new": 10.0 + R.LANE_OVERHEAD_S, "gbdt-new": 5.0 + R.LANE_OVERHEAD_S}
    assert R.lane_costs(["a"], tmp_path / "missing.json") == {"a": 1.0 + R.LANE_OVERHEAD_S}
    shipped = R.lane_costs(["generic-pickle-trees", "pca"])
    assert shipped["generic-pickle-trees"] > shipped["pca"]


def test_the_record_prints_its_wall_busy_cores_and_slowest_lanes(tmp_path, fake, capsys):
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["knn", "ols", "ridge"], 0))
    a = args(tmp_path, "--columns", "amd", "--label-device", "mi325x", "--gpus", "1", "--workers-per-gpu", "2")
    assert R.Run(a).main() == 0
    w0 = tmp_path / "out" / "amd-mi325x" / "w0"
    (w0 / "amd-mi325x.lanes-x.log").write_text("# DONE knn/base repeat=1/train 3.5s\n# DONE knn/odd repeat=1/train 1.5s\n"
                                              "# DONE ols/base repeat=1/train 0.25s\n")
    assert R.lane_seconds([w0 / "amd-mi325x.lanes-x.log"]) == {"knn": 5.0, "ols": 0.25}
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert set(doc["timing"]) >= {"wall_s", "busy_cores_mean", "cores", "chunk_seconds", "slowest_lanes"}
    assert "identity-wheels: timing: record " in capsys.readouterr().out


def families_map(fake, lanes):
    """tools/release_families.py's map with these explicit lanes ({lane: family}) and no catch-all rule."""
    import release_families as RF
    real = RF.load()
    doc = dict(real.doc, lanes=dict(lanes), lane_rules=[])
    fake.setattr(RF, "load", lambda *a, **k: RF.Families(doc))


def test_two_columns_on_two_gpus_record_once_in_parallel(tmp_path, fake, capsys):
    # --record-families (tools/release.py's family plan): only the recording families' lanes, on every column
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["knn", "ols", "ridge", "gbdt-rmse"], 0))
    families_map(fake, {"knn": "cluster", "ols": "linear", "ridge": "linear", "gbdt-rmse": "trees"})
    a = args(tmp_path, "--columns", "nvidia,ptx", "--label-device", "l40s", "--gpus", "2", "--workers-per-gpu", "1",
             "--record-families", "linear,cluster")
    assert R.Run(a).main() == 0
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["ok"] and doc["columns"]["nvidia"]["owed"] == 3 and doc["columns"]["ptx"]["owed"] == 3
    assert {c["device"] for c in doc["columns"]["nvidia"]["chunks"]} == {0}
    assert {c["device"] for c in doc["columns"]["ptx"]["chunks"]} == {1}
    out = capsys.readouterr().out
    assert "busy plan: 2 worker(s) of 2" in out and "nvidia-l40s: REUSED 1 / RUN 3 (families: linear,cluster)" in out
    assert RIW.summary_errors(doc, ["nvidia", "ptx"]) == []


def test_no_family_recording_records_nothing_and_is_ok(tmp_path, fake, capsys):
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["knn", "ols"], 0))
    a = args(tmp_path, "--columns", "amd", "--label-device", "mi325x", "--gpus", "1", "--record-families", "none")
    assert R.Run(a).main() == 0
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["columns"]["amd"]["owed"] == 0 and doc["columns"]["amd"]["chunks"] == []
    assert "amd-mi325x: REUSED 2 / RUN 0" in capsys.readouterr().out
    assert RIW.summary_errors(doc, ["amd"]) == []


def test_an_unknown_family_is_refused(tmp_path, fake):
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["knn"], 0))
    a = args(tmp_path, "--columns", "amd", "--label-device", "mi325x", "--gpus", "1", "--record-families", "nope")
    with pytest.raises(SystemExit, match="unknown famil"):
        R.Run(a).main()


def test_a_cpu_shard_records_its_share(tmp_path, fake):
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["a", "b", "c", "d", "e", "f"], 0))
    families_map(fake, {"a": "linear", "b": "linear", "c": "linear", "d": "linear", "e": "linear", "f": "trees"})
    a = args(tmp_path, "--columns", "cpu", "--shard", "1/2", "--workers-per-gpu", "2", "--record-families", "linear")
    assert R.Run(a).main() == 0
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["columns"]["cpu"]["owed"] == 2 and sum(c["lanes"] for c in doc["columns"]["cpu"]["chunks"]) == 2


def test_a_wheel_that_is_not_the_release_bytes_is_refused(tmp_path):
    w = tmp_path / "w.whl"
    w.write_bytes(b"other")
    a = R.parse(["--columns", "cpu", "--wheel", str(w), "--sha256", "w.whl=" + "a" * 64, "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="not the release's"):
        R.Run(a).install()


def test_summary_errors_name_what_is_missing():
    assert RIW.summary_errors(None, ["amd"]) == ["no identity-wheels summary"]
    doc = dict(schema="mojolearn.identity-wheels.v1", ok=False,
               columns={"nvidia": dict(rc=1, owed=2, chunks=[dict(worker=0, lanes=1, records=[dict(admissible=False)])])})
    errs = RIW.summary_errors(doc, ["nvidia", "ptx"])
    assert any("columns" in e for e in errs) and any("exited 1" in e for e in errs)
    assert any("recorded 1 of 2" in e for e in errs) and any("inadmissible" in e for e in errs)


def test_by_default_every_lane_runs_and_no_selector_is_asked(tmp_path, fake, capsys):
    # --record-families defaults to all (tools/release.py passes its plan's); no per-lane selector exists any more
    assert not hasattr(R.Run, "select")
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["knn", "ols", "ridge"], 0))
    a = args(tmp_path, "--columns", "amd", "--label-device", "mi325x", "--gpus", "1", "--workers-per-gpu", "3")
    assert R.Run(a).main() == 0
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["columns"]["amd"]["owed"] == 3 and len(doc["columns"]["amd"]["chunks"]) == 3
    out = capsys.readouterr().out
    assert "amd-mi325x: REUSED 0 / RUN all (3)" in out and "busy plan: 3 worker(s) of 3" in out


def test_a_failing_chunk_does_not_stop_the_others(tmp_path, fake, capsys):
    fake.setattr(R.Run, "all_lanes", lambda self, py, column, lab: (["knn", "bad", "ols"], 0))
    real = R.Run.env

    def env(self, py, listed, device, column, extra=None):
        e = real(self, py, listed, device, column, extra)
        if "bad" in (extra or {}).get("LANES", ""):
            e["RECORD_FROM_INSTALLED"] = "0"     # the fake recorder exits 9 for this chunk
        return e
    fake.setattr(R.Run, "env", env)
    a = args(tmp_path, "--columns", "amd", "--label-device", "mi325x", "--gpus", "1", "--workers-per-gpu", "3")
    assert R.Run(a).main() == 1
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert sum(1 for c in doc["columns"]["amd"]["chunks"] if c["rc"] == 0) == 2 and len(doc["failures"]) == 1
    assert "FAILURES (1 chunk(s); every other chunk ran)" in capsys.readouterr().out


def test_the_full_lane_list_is_every_lane_on_every_column(tmp_path):
    """A one-sided lane is recorded on its missing side too: its cell reads n/a by name (2026-11-24)."""
    class P:
        def __init__(self, out):
            self.returncode, self.stdout, self.stderr = 0, out, ""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return P("a,b,gpu1\n")
    a = args(tmp_path, "--columns", "cpu")
    r = R.Run(a, run=run)
    assert r.all_lanes(pathlib.Path(sys.executable), "cpu", "cpu") == (["a", "b", "gpu1"], 0)
    assert r.all_lanes(pathlib.Path(sys.executable), "amd", "amd-x") == (["a", "b", "gpu1"], 0)
    assert len(calls) == 2


def test_the_venv_gets_the_harness_numpy_after_the_release_wheels(tmp_path):
    # identity_break.py imports numpy at module level and the wheels carry none: the venv must install it, after the
    # exact wheels, in its own pip call (so the wheel install line stays the release's bytes only).
    w = tmp_path / "w.whl"
    w.write_bytes(b"w")
    calls = []
    a = R.parse(["--columns", "cpu", "--wheel", str(w), "--sha256", "w.whl=" + R.sha256(str(w)),
                 "--out", str(tmp_path / "o"), "--venv", str(tmp_path / "venv")])
    (tmp_path / "o").mkdir()
    R.Run(a, run=lambda cmd, **kw: calls.append(list(cmd))).install()
    pips = [c for c in calls if c[1:3] == ["-m", "pip"]]
    assert len(pips) == 2
    assert pips[0][-1] == str(w) and not any(x.startswith("numpy") for x in pips[0])
    assert pips[1][-len(R.HARNESS_REQUIREMENTS):] == list(R.HARNESS_REQUIREMENTS)
    assert any(x.startswith("numpy>=") for x in pips[1])


def test_the_recorder_reads_the_one_allocation_function(monkeypatch):
    """rel-fix-3: the recorder's cores are python/mojolearn/_allocation.py's; the job's measured count wins."""
    monkeypatch.setenv("MOJOLEARN_ALLOC_CORES", "14")
    assert R.alloc_cores() == 14
    monkeypatch.delenv("MOJOLEARN_ALLOC_CORES")
    assert R.alloc_cores() == R.ALLOCATION.cores()[0] >= 1
