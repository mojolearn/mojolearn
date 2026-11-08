# SPDX-License-Identifier: Apache-2.0
"""tools/release_measurement_smoke.py without a device: the plan is the whole measurement plan at its smoke shape, the
verdicts fail closed, the summary the release judges, and the unconditional wiring in the box job."""
import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release_measurement_smoke as MS  # noqa: E402

VENDORS = ("nvidia", "amd")


# ------------------------------------------------------------------ the plan
@pytest.mark.parametrize("vendor", VENDORS)
def test_the_smoke_plan_is_the_whole_measurement_plan_at_its_smoke_shape(vendor):
    import bench_board as B
    import gpu_gap_campaign as G
    smoke = MS.plan(vendor)
    full = B.plan_races(vendor, ["identical"])
    campaign = G.selected_plan(vendor, None, smoke=True, cpu_fallback=True)
    keys = sorted(B.smoke_key(r) for r in smoke)
    assert len(keys) == len(set(keys)) and len(keys) >= 600
    assert keys == sorted(B.smoke_key(r) for r in full) == sorted(B.smoke_key(r) for r in campaign)
    for r in smoke:
        assert r["our_arms"], r["id"]
        assert set(r["our_arms"].values()) == {"identical"}
        assert r["id"].endswith("/shape=small" if r["family"] == "neural" else "/rows=%d" % B.SMOKE_ROWS)


def test_every_recipe_is_in_exactly_one_batch_and_the_neural_recipes_share_one():
    ids = [r["id"] for r in MS.plan("nvidia")]
    for workers in (1, 5, 12, 20, 200):
        groups = MS.split(ids, workers)
        flat = [i for g in groups for i in g]
        assert sorted(flat) == sorted(ids) and len(flat) == len(set(flat))
        neural = [g for g in groups if any(i.startswith("neural/") for i in g)]
        assert len(neural) == 1 and all(i.startswith("neural/") for i in neural[0])
    assert MS.split([], 4) == []


def test_the_worker_count_comes_from_cores_and_memory_and_names_its_bound():
    assert MS.worker_count(128, 416, 48, 653) == (12, MS.worker_count(128, 416, 48, 653)[1])
    assert "device memory" in MS.worker_count(128, 416, 48, 653)[1]
    assert MS.worker_count(20, 230, 256, 653)[0] == 20 and "cores" in MS.worker_count(20, 230, 256, 653)[1]
    assert MS.worker_count(64, 16, 256, 653)[0] == 4                       # host memory bounds it
    assert MS.worker_count(128, 416, None, 653)[0] == MS.UNKNOWN_DEVICE_WORKERS  # never the host's cores blindly
    assert MS.worker_count(128, 416, 48, 3)[0] == 3


# ------------------------------------------------------------------ the verdicts
RACE = dict(id="algos/x/taxi/rows=2000", family="algos", lane="x", dataset="taxi", rows=2000,
            our_arms={"ours": "identical"}, opponents=["sklearn-cpu"], arms=["ours", "sklearn-cpu"])


def cell(**over):
    c = dict(arm="ours", library="mojolearn", status="ok", median_ms=1.0, quality={"r2": 0.91},
             hash="aa99a3dc8c61a3b4", mode="identical", library_version="0.8.67", device="NVIDIA L40S")
    c.update(over)
    return c


def rec(*cells, **over):
    r = dict(status="done", params_check="MATCHED", cells=list(cells) or [cell()], skipped_opponents=["sklearn-cpu"])
    r.update(over)
    return r


def verdict(r, fails=(), unsupported=None, exempt=False, why=None):
    return MS.recipe_verdict(RACE, r, list(fails), unsupported or {}, exempt, why)


def test_a_complete_own_cell_passes_and_board_smoke_agrees():
    import bench_board as B
    assert verdict(rec()) == (MS.PASS, "")
    # bench_board's own smoke verdict on the same record: the unrun opponent is skipped, not a failure
    assert B.smoke_verdict(RACE, rec(), "nvidia") == []


def test_no_record_is_a_failure_with_its_batch_reason():
    assert verdict(None) == (MS.FAIL, "not run: no record from its batch")
    assert verdict(None, why="its batch exited 1 without a record for it: Traceback") == (
        MS.FAIL, "its batch exited 1 without a record for it: Traceback")


@pytest.mark.parametrize("bad, expect", [
    (rec(status="failed", failure="ValueError: boom"), "status failed"),               # raised
    (rec(cell(status="REFUSED(quality scorer unavailable)")), "REFUSED"),              # refused
    (rec(cell(status="SKIPPED: not built yet")), "SKIPPED"),                           # skipped
    (rec(status="timeout"), "status timeout"),                                         # timed out
])
def test_raise_refuse_skip_and_timeout_fail_through_the_board_verdict(bad, expect):
    import bench_board as B
    fails = B.smoke_verdict(RACE, bad, "nvidia")
    assert fails
    v, why = verdict(bad, fails)
    assert v == MS.FAIL and expect in why


@pytest.mark.parametrize("over, expect", [
    (dict(hash=None), "no hash"),
    (dict(mode=None), "no mode"),
    (dict(mode="fast"), "mode fast"),
    (dict(library_version=""), "no library_version"),
    (dict(device=None), "no device"),
    (dict(quality={}), "no finite quality value"),
    (dict(quality=None), "no quality record"),
    (dict(quality={"r2": float("nan")}), "quality not finite: r2"),
    (dict(quality={"r2": float("inf"), "auc": 0.5}), "quality not finite: r2"),
    (dict(quality={"label": "x", "ok": True}), "no finite quality value"),
    (dict(quality={"r2": 0.9, "error": "scorer failed"}), "quality error"),
    (dict(status="REFUSED(x)"), "status REFUSED"),
])
def test_a_missing_or_unusable_field_fails(over, expect):
    v, why = verdict(rec(cell(**over)))
    assert v == MS.FAIL and expect in why


def test_a_missing_own_cell_fails_and_the_exempt_lane_needs_no_quality():
    assert verdict(rec(cell(arm="other"))) == (MS.FAIL, "ours: no cell")
    assert verdict(rec(cell(quality={})), exempt=True) == (MS.PASS, "")


def test_a_declared_unsupported_own_arm_is_listed_not_passed_silently():
    v, why = verdict(rec(cell(status="UNSUPPORTED: int8")), unsupported={"ours": "UNSUPPORTED: int8"})
    assert v == MS.UNSUPPORTED and "ours" in why


def test_judging_that_raises_is_a_failure(monkeypatch, tmp_path):
    class FakeB:
        QUALITY_EXEMPT = ()

        @staticmethod
        def load_result(path):
            return {"races": {RACE["id"]: rec()}}

        @staticmethod
        def smoke_key(race):
            return "algos/x/taxi"

        @staticmethod
        def smoke_verdict(race, r, vendor):
            raise KeyError("cells")

        @staticmethod
        def unsupported_our_arms(race):
            return {}

    monkeypatch.setattr(MS, "_bb", lambda: FakeB)
    args = type("A", (), dict(vendor="nvidia"))
    batches = [dict(batch=0, ids=[RACE["id"]], dir=str(tmp_path), exit_code=0, recipes=1)]
    got = MS.judge(args, [RACE], batches)
    assert got[RACE["id"]]["verdict"] == MS.FAIL and "judging raised" in got[RACE["id"]]["reason"]


def test_a_recipe_the_preflight_held_back_fails_with_the_preflight_reason(tmp_path):
    import bench_board as B
    held = {"races": {}, "skipped_recipes": {RACE["id"]: dict(status="REFUSED", problems=["parameter probe failed"])}}
    (tmp_path / "board.json").write_text(json.dumps(held))
    args = type("A", (), dict(vendor="nvidia"))
    got = MS.judge(args, [RACE], [dict(batch=0, ids=[RACE["id"]], dir=str(tmp_path), exit_code=0, recipes=1)])
    assert got[RACE["id"]] == dict(verdict=MS.FAIL, reason="preflight REFUSED: parameter probe failed", batch=0)
    assert B.load_result(str(tmp_path / "board.json")) is not None


def test_why_a_batch_left_no_record():
    assert "before its batch started" in MS.missing_reason(dict(started_late=True))
    assert "while its batch ran" in MS.missing_reason(dict(ceiling_hit=True, timed_out=True))
    assert "per-recipe timeout" in MS.missing_reason(dict(timed_out=True, recipes=18))
    assert "exited 139" in MS.missing_reason(dict(exit_code=139, log="/nonexistent"))


# ------------------------------------------------------------------ the summary the release judges
IDS = ["algos/a/taxi/rows=2000", "neural/b/gaussian/shape=small", "trees/c/istella/rows=2000"]


def summary(verdicts=None, ids=IDS):
    verdicts = verdicts if verdicts is not None else {i: dict(verdict=MS.PASS, reason="", batch=0) for i in ids}
    return MS.summarize("nvidia", ids, verdicts, dict(workers=12))


def test_a_full_pass_is_accepted_and_carries_no_timing():
    doc = summary()
    assert doc["ok"] and MS.summary_errors(doc) == []
    assert "median_ms" not in json.dumps(doc) and "times_ms" not in json.dumps(doc)
    assert doc["ours_only"] is True and doc["opponents_run"] is False and doc["mode"] == "identical"
    unsupported = summary({IDS[0]: dict(verdict=MS.UNSUPPORTED, reason="declared", batch=0),
                           IDS[1]: dict(verdict=MS.PASS, reason="", batch=0),
                           IDS[2]: dict(verdict=MS.PASS, reason="", batch=0)})
    assert unsupported["ok"] and unsupported["unsupported"] == [IDS[0]] and MS.summary_errors(unsupported) == []


def test_any_failure_or_gap_fails_the_release():
    assert MS.summary_errors(None) == [MS.NO_SUMMARY]
    # the refusal says plainly what to do: it comes only from a box job, never from --reuse-unchanged on the Mac
    assert "--job measuresmoke" in MS.NO_SUMMARY and "--reuse-unchanged" in MS.NO_SUMMARY
    assert "rerun the gate-<vendor> step" in MS.NO_SUMMARY
    one = summary({IDS[0]: dict(verdict=MS.FAIL, reason="status failed", batch=0),
                   IDS[1]: dict(verdict=MS.PASS, reason="", batch=0), IDS[2]: dict(verdict=MS.PASS, reason="", batch=0)})
    assert not one["ok"] and one["failures"] == [dict(id=IDS[0], reason="status failed")]
    assert any("1 of 3 recipes failed" in e and IDS[0] in e for e in MS.summary_errors(one))
    missing = summary({IDS[0]: dict(verdict=MS.PASS, reason="", batch=0)})       # two recipes never judged
    assert not missing["ok"] and len(missing["failures"]) == 2 and MS.summary_errors(missing)
    forged = dict(summary(), ok=True)
    forged["recipes"] = dict(forged["recipes"], **{IDS[0]: dict(verdict=MS.FAIL, reason="x")})
    assert any("failed" in e for e in MS.summary_errors(forged))
    short = summary()
    short["recipes"].pop(IDS[0])
    assert any("recipe verdicts for a plan of 3" in e for e in MS.summary_errors(short))
    other = summary()
    other["plan_sha256"] = "0" * 64
    assert MS.summary_errors(other) == ["recipe verdicts are not the recorded plan"]
    with_opponents = dict(summary(), opponents_run=True)
    assert MS.summary_errors(with_opponents) == ["not an identical, ours-only run"]
    assert MS.summary_errors(dict(summary(), schema="other"))[0].startswith("measurement-smoke summary schema")
    empty = MS.summarize("nvidia", [], {}, {})
    assert not empty["ok"] and MS.summary_errors(empty)


# ------------------------------------------------------------------ the runner command: ours only, the board's smoke
def test_each_batch_is_the_boards_own_smoke_with_our_arm_only(tmp_path):
    args = type("A", (), dict(python="/v/bin/python", vendor="amd", work=str(tmp_path), data_root="/root/datasets/gbm-bench"))
    cmd = MS.board_command(args, tmp_path / "b000", ["--race-ids", "@ids.json"])
    assert cmd[1].endswith("tools/bench_board.py") and "--smoke" in cmd and "--skip-install" in cmd
    assert cmd[cmd.index("--modes") + 1] == "identical" and "--evidence-local-only" in cmd
    for flag in ("--with-opponents", "--retime-opponents", "--opponent-store", "--opponents-only", "--rows",
                 "--neural-shape", "--retime-cpu-opponents"):
        assert flag not in cmd
    env = MS.batch_env(args)
    assert env["MOJOLEARN_NUMERIC_MODE"] == "identical" and "no-run-summary" in env["MOJOLEARN_SUMMARY_REPO"]
    assert env["MOJOLEARN_NEURAL_FIXTURES"].startswith(str(tmp_path))


def test_box_knobs_never_reach_the_measured_process(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_VENDOR", "cpu")
    monkeypatch.setenv("MOJOLEARN_IDN_ALL_OFF", "1")
    args = type("A", (), dict(work=str(tmp_path), data_root="/d"))
    env = MS.batch_env(args)
    assert "MOJOLEARN_VENDOR" not in env and "MOJOLEARN_IDN_ALL_OFF" not in env


# ------------------------------------------------------------------ data
def test_every_file_the_plan_reads_is_pinned_and_staged_where_the_board_looks(tmp_path):
    import bench_board as B
    pins = MS.pins()
    for vendor in VENDORS:
        keys = MS.data_keys(vendor)
        assert set(keys) >= {B.R2_KEYS[d] for d in ("taxi", "istella", "istella-rank", "higgs")}
        assert all(k in pins for k in keys)
        for k, rel in keys.items():
            assert rel == ("r2-stage/" + k if k.startswith("corpus/") else "datasets/" + k), k
    entries = MS.write_data_plan("nvidia", tmp_path / "d.json", presign=lambda k, s: "https://r2/%s" % k)
    doc = json.loads((tmp_path / "d.json").read_text())
    assert doc["files"] == entries and all(e["url"].startswith("https://") for e in entries)
    assert (tmp_path / "d.json").stat().st_mode & 0o077 == 0
    with pytest.raises(SystemExit):
        MS.write_data_plan("nvidia", tmp_path / "e.json", presign=lambda k, s: "")


def test_staging_keeps_a_file_at_its_pin_and_refuses_one_without(tmp_path):
    data = b"pinned bytes"
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "f.npz").write_bytes(data)
    entry = dict(key="k", path="datasets/f.npz", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), url=None)
    assert MS.stage_one(entry, tmp_path) == "present"
    assert MS.stage_one(dict(entry, sha256="0" * 64), tmp_path) == "missing and no URL"


# ------------------------------------------------------------------ the wiring: unconditional, in order, judged
JOB = (ROOT / "tools" / "ops" / "release_gate_job.sh").read_text()


def test_only_the_smokes_own_job_runs_it_and_that_job_runs_nothing_else():
    lines = JOB.splitlines()
    overlay, measure, shard, gate = (lines.index("PHASE=" + p) for p in ("overlay", "measuresmoke", "shardcheck", "gate"))
    assert overlay < measure < shard < gate          # after the overlay: the box runs the Mac's copy of the tool
    block = JOB[JOB.index("\nPHASE=measuresmoke"):JOB.index("\nPHASE=shardcheck")]
    code = block[block.index('if [ "$GATE_JOB" = measuresmoke ]; then'):]
    for word in ("GATE_REUSE", "GATE_UNITS", "changed", "selector", "GATE_IDENTITY_COLUMNS"):
        assert word not in code, word                # no change detection decides whether it runs
    assert "tools/release_measurement_smoke.py run" in code and '--work "$GATE_WORK/measuresmoke-work"' in code
    assert code.rstrip().endswith('exit "$mrc"\nfi')   # the smoke's job ends there: no shardcheck, gate or qualifier
    assert "grc=$mrc" not in JOB                     # the gate job's exit no longer carries it
    # the smoke's job builds nothing: pixi and the autograd build run in the gate job only
    pixi = JOB[JOB.index("\nPHASE=pixi"):JOB.index("\nPHASE=overlay")]
    assert pixi.split("\n")[3].startswith('if [ "$GATE_JOB" = gate ]; then')
    assert '[[ "$GATE_JOB" =~ ^(gate|measuresmoke)$ ]]' in JOB
    # its setup steps run at once, and its busy-cores lines reach the job log as they happen
    assert code.count("> \"$J/") >= 3 and 'wait "$vpid"' in code and 'wait "$fpid"' in code
    assert "MEASURESMOKE (plan|busy|vendor=)" in code and "${PIPESTATUS[0]}" in code


def test_the_smoke_is_its_own_persisted_job_with_its_own_cap(tmp_path):
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    import release_gate as RG
    out = tmp_path / "abc-release-light-v1"
    gate = RG.Plan("nvidia", "0.8.67", "a" * 40, tmp_path, out, [tmp_path / "w.whl"], "t1")
    smoke = RG.Plan("nvidia", "0.8.67", "a" * 40, tmp_path, out / "measuresmoke-job", [tmp_path / "w.whl"], "t2",
                    job="measuresmoke", identity_columns=["nvidia"], label_device="l40s")
    assert gate.launch()[gate.launch().index("--cap") + 1] == str(RG.NV_CAP_MIN) == "240"
    assert smoke.launch()[smoke.launch().index("--cap") + 1] == str(RG.MEASURE_SMOKE_CAP_MIN)
    assert RG.MEASURE_SMOKE_CAP_MIN >= RG.MEASURE_SMOKE_MINUTES + 30 and RG.MEASURE_SMOKE_CAP_MIN <= 240
    assert smoke.launch()[:3] == ["bash", "tools/nvidia_central.sh", "submit"]   # the persisted-job entry point
    assert smoke.indir != gate.indir and smoke.work != gate.work and "abc-release-light-v1-measuresmoke" in smoke.indir
    env = smoke.env_text()
    assert "GATE_JOB=measuresmoke" in env and "GATE_IDENTITY_COLUMNS=\n" in env and "GATE_BUILD_AUTOGRAD=0" in env
    assert "GATE_JOB=gate" in gate.env_text() and "GATE_RUNTIME_CPU=0" in env
    assert "MEASURE_SMOKE_PIP=(" in env and "MEASURE_SMOKE_FIXTURE_PIP=(" in env
    assert "GATE_MEASURE_SMOKE_MINUTES=%d" % RG.MEASURE_SMOKE_MINUTES in env
    assert any(r.endswith("/in/" + RG.MEASURE_SMOKE_DATA) for _, r, _ in smoke.uploads())
    assert not any(r.endswith("/in/" + RG.MEASURE_SMOKE_DATA) for _, r, _ in gate.uploads())
    amd = RG.Plan("amd", "0.8.67", "a" * 40, tmp_path, out / "measuresmoke-job", [tmp_path / "w.whl"], "t3",
                  job="measuresmoke")
    assert amd.launch()[:3] == ["bash", "tools/do_amd_steward.sh", "run"]
    with pytest.raises(SystemExit):
        RG.Plan("amd", "0.8.67", "a" * 40, tmp_path, out, [], "t4", job="other")
    assert RG.parse(["amd", "0.8.67", "--source-commit", "a" * 40, "--source-tree", ".", "--out", "o",
                     "--candidate-wheel", "w", "--job", "measuresmoke"]).job == "measuresmoke"


def test_release_runs_the_smoke_job_after_the_gate_job_and_only_what_is_missing(tmp_path, monkeypatch):
    import release as R
    calls = []
    r = R.Release.__new__(R.Release)
    r.dry = False
    out = tmp_path / "gate"
    state = {"gate": False, "smoke": False}
    monkeypatch.setattr(r, "readiness_status", lambda write: None, raising=False)
    monkeypatch.setattr(r, "readiness", lambda: type("X", (), dict(begin=lambda s, k: "id", record=lambda s, *a, **k: None))(),
                        raising=False)
    monkeypatch.setattr(r, "record_candidate_requirement", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(R.Release, "rel", tmp_path, raising=False)
    monkeypatch.setattr(r, "gate_out", lambda vendor: out, raising=False)

    def gate_ok(vendor, measure=True):
        if not state["gate"]:
            return False, "no gate report"
        if measure and not state["smoke"]:
            return False, "measuresmoke: no summary"
        return True, "PASS"

    def gate_job(vendor, run_id, work, output):
        calls.append("gate")
        state["gate"] = True

    def smoke_job(vendor, work, output):
        assert state["gate"], "the smoke job starts only after the gate job has ended"
        calls.append("smoke")
        state["smoke"] = True
    monkeypatch.setattr(r, "gate_ok", gate_ok, raising=False)
    monkeypatch.setattr(r, "launch_gate_job", gate_job, raising=False)
    monkeypatch.setattr(r, "launch_measure_smoke_job", smoke_job, raising=False)
    assert r.run_gate("nvidia") == "PASS" and calls == ["gate", "smoke"]
    calls.clear()
    state["smoke"] = False                       # a gate job that already passed is not launched again
    assert r.run_gate("nvidia") == "PASS" and calls == ["smoke"]
    calls.clear()
    assert r.run_gate("nvidia") == "PASS" and calls == []
    state.update(gate=True, smoke=False)
    monkeypatch.setattr(r, "launch_measure_smoke_job", lambda *a: calls.append("smoke"), raising=False)
    with pytest.raises(R.StepFailed, match="measuresmoke"):
        r.run_gate("nvidia")                     # a smoke that did not pass fails the step (and so the release)


def test_the_smoke_job_command(tmp_path, monkeypatch):
    import release as R
    r = R.Release.__new__(R.Release)
    r.dry = True
    monkeypatch.setattr(R.Release, "commit", "a" * 40, raising=False)
    monkeypatch.setattr(R.Release, "version", "0.8.67", raising=False)
    monkeypatch.setattr(r, "source_checkout", lambda create=True: tmp_path, raising=False)
    monkeypatch.setattr(r, "candidate_gate_wheels", lambda vendor: [tmp_path / "w.whl"], raising=False)
    cmd = r.measure_smoke_cmd("amd", tmp_path / "gates", tmp_path / "x-release-light-v1")
    assert cmd[cmd.index("--job") + 1] == "measuresmoke"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "x-release-light-v1" / "measuresmoke-job")
    assert "--identity-columns" not in cmd and "--units-plan" not in cmd and "--autograd-prior" not in cmd


def test_boxes_down_waits_for_the_gate_steps_that_hold_both_jobs():
    import release as R
    assert {"gate-nvidia", "gate-amd"} <= set(R.AFTER["boxes-down"])


def test_the_busy_cores_figure():
    assert MS.busy_cores((100, 1000), (600, 2000), 20) == 10.0
    assert MS.busy_cores(None, (1, 2), 4) is None and MS.busy_cores((5, 10), (5, 10), 4) is None


# ------------------------------------------------------------------ the two scorer gaps, closed in the harness
# macOS Accelerate's matmul raises spurious floating-point flags under NumPy 2.x (the values are exact; asserted below)
@pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
def test_mlp_predict_scores_ours_alone_against_the_fp64_forward():
    import numpy as np
    import bench_board_neural as N
    rng = np.random.default_rng(7)
    data = {"w:" + n: rng.uniform(-0.3, 0.3, d).astype(np.float32) for n, d in zip(N.MLP_NAMES, N.MLP_DIMS)}
    data["X"] = rng.standard_normal((2, 64, 8)).astype(np.float32)
    w1, b1, w2, b2 = (data["w:" + n] for n in N.MLP_NAMES)
    x = data["X"][0].astype(np.float64)
    y = (np.maximum(x @ w1.T.astype(np.float64) + b1, 0) @ w2.T.astype(np.float64) + b2).astype(np.float32)  # ours
    q = N.quality("mlp-predict", data, {"ours": {"y": y.astype(np.float32)}})
    err = q["ours"]["max_rel_err_vs_fp64"]
    assert isinstance(err, float) and 0 <= err < 1e-5
    bad = N.quality("mlp-predict", data, {"ours": {"y": (y + 1).astype(np.float32)}})["ours"]["max_rel_err_vs_fp64"]
    assert bad > 0.01                             # a wrong output is visible, not a constant
    assert MS.own_cell_errors(dict(cell(), quality=q["ours"]), False) == []


def test_jl_min_dim_gets_its_untimed_sklearn_reference_when_ours_races_alone():
    import bench_board_algos as A
    assert A.quality_kind("jl-min-dim") == "x-jl"
    assert A.independent_reference_arm("jl-min-dim") == "sklearn-cpu"
    # the outlier lanes keep their rule (an "sk" lane), and a lane naming no sklearn arm gets none
    assert all(A.independent_reference_arm(l) in (None, "sklearn-cpu") for l in A.LANES
               if A.quality_kind(l) == "outlier")


def test_the_release_verdict_requires_it_in_both_gate_paths():
    riw = (ROOT / "tools" / "release_identity_wheels.py").read_text()
    single = riw[riw.index("def single_gate_ok"):riw.index("def gate_identity_errors")]
    assert "measure_smoke_errors(output)" in single
    rel = (ROOT / "tools" / "release.py").read_text()
    light = rel[rel.index("    def gate_ok(self, vendor, measure=True):"):rel.index("RUNTIME_GATE_FAILED = ")]
    assert "measure_smoke_errors(output)" in light
    import release_identity_wheels as RIW
    assert RIW.measure_smoke_errors(pathlib.Path("/nonexistent")) == ["measuresmoke: " + MS.NO_SUMMARY]
    assert str(RIW.measure_smoke_summary(pathlib.Path("/g"))) == "/g/measuresmoke-job/measuresmoke/" + MS.SUMMARY


def test_the_tool_is_overlaid_and_its_summary_comes_home():
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    import release_gate as RG
    assert "tools/release_measurement_smoke.py" in RG.GATE_OVERLAY
    assert "measuresmoke/" + MS.SUMMARY in RG.GATE_FETCH
    assert not any(f.startswith("measuresmoke") and ("work" in f or "batches" in f) for f in RG.GATE_FETCH)
    # harness packages only: no opponent library is installed for the smoke
    for pkg in RG.MEASURE_SMOKE_PIP + RG.MEASURE_SMOKE_FIXTURE_PIP:
        name = pkg.split("=")[0].split(">")[0].lower()
        assert name not in ("cuml", "cupy", "xgboost", "lightgbm", "catboost", "transformers", "mamba-ssm",
                            "faiss-cpu", "implicit", "cuvs", "cugraph", "gpytorch"), pkg
    assert "torch" not in [p.split("=")[0] for p in RG.MEASURE_SMOKE_PIP]   # torch only in the fixture venv
