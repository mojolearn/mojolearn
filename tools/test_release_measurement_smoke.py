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


def test_every_recipe_is_in_exactly_one_group_and_batch_and_the_neural_recipes_share_one():
    import bench_board as B
    races = MS.plan("nvidia")
    ids = [r["id"] for r in races]
    groups = MS.staging_groups(races, B)
    flat = [i for g in groups for i in g["ids"]]
    assert sorted(flat) == sorted(ids) and len(flat) == len(set(flat))
    neural = [g for g in groups if g["name"] == "neural"]
    assert len(neural) == 1 and sorted(neural[0]["ids"]) == sorted(i for i in ids if i.startswith("neural/"))
    assert all(len(g["ids"]) <= MS.GROUP_MAX_RECIPES for g in groups if g["name"] != "neural")
    by_id = {r["id"]: r for r in races}
    for g in groups:      # a group stages exactly the files its recipes read, never the whole plan's
        assert set(g["keys"]) == set().union(*[set(MS.race_keys(by_id[i], B)) for i in g["ids"]])
        assert len({by_id[i]["dataset"] for i in g["ids"]}) == 1 or g["name"] == "neural"
    assert sum(len(g["keys"]) for g in groups) > len(MS.data_keys("nvidia"))   # shared files: staged per group
    for size in (1, 7, 42, 1000):
        for g in groups:
            batches = MS.group_batches(g, size)
            got = [i for b in batches for i in b]
            assert sorted(got) == sorted(g["ids"]) and len(got) == len(set(got))
            if g["name"] == "neural":
                assert len(batches) == 1
            else:
                assert max(len(b) for b in batches) <= size


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
    assert "phase measuresmoke" in MS.NO_SUMMARY and "--reuse-unchanged" in MS.NO_SUMMARY
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
    assert "--inputs" not in cmd                                       # no input map: the datasets
    args.inputs_map = tmp_path / "smoke_inputs.json"
    cmd = MS.board_command(args, tmp_path / "b000", ["--race-ids", "@ids.json"])
    assert cmd[cmd.index("--inputs") + 1] == "synthetic" and cmd[cmd.index("--inputs-map") + 1] == str(
        tmp_path / "smoke_inputs.json")
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
        keys = MS.data_keys(vendor, inputs=MS.DATASETS_INPUTS)
        assert set(keys) >= {B.R2_KEYS[d] for d in ("taxi", "istella", "istella-rank", "higgs")}
        assert all(k in pins for k in keys)
        for k, rel in keys.items():
            assert rel == ("r2-stage/" + k if k.startswith("corpus/") else "datasets/" + k), k
    entries = MS.write_data_plan("nvidia", tmp_path / "d.json", presign=lambda k, s: "https://r2/%s" % k,
                                 inputs=MS.DATASETS_INPUTS)
    doc = json.loads((tmp_path / "d.json").read_text())
    assert doc["files"] == entries and all(e["url"].startswith("https://") for e in entries)
    assert doc["inputs"] == MS.DATASETS_INPUTS
    assert (tmp_path / "d.json").stat().st_mode & 0o077 == 0
    with pytest.raises(SystemExit):
        MS.write_data_plan("nvidia", tmp_path / "e.json", presign=lambda k, s: "", inputs=MS.DATASETS_INPUTS)


def test_the_smoke_stages_no_data_and_its_data_plan_is_empty(tmp_path):
    import bench_board as B
    assert MS.board_takes_inputs()
    for vendor in VENDORS:
        races = MS.plan(vendor, B)
        imap = MS.smoke_inputs(races, B)
        assert MS.data_keys(vendor, B, MS.SYNTHETIC_INPUTS, imap) == {}
        assert all(MS.race_keys(r, B, imap[r["id"]]) == {} for r in races)
        assert all(g["keys"] == [] for g in MS.staging_groups(races, B, imap=imap))
    entries = MS.write_data_plan("nvidia", tmp_path / "d.json", presign=lambda k, s: pytest.fail("no GET"))
    doc = json.loads((tmp_path / "d.json").read_text())
    assert entries == [] and doc["files"] == [] and doc["inputs"] == MS.SYNTHETIC_INPUTS


@pytest.mark.parametrize("vendor", VENDORS)
def test_the_work_dir_is_within_budget_from_the_plan(vendor):
    import bench_board as B
    races = MS.plan(vendor, B)
    bud = MS.budget(races, MS.smoke_inputs(races, B), B, MS.MAX_BOX_WORKERS)
    assert sum(bud["counts"].values()) == len(races) and bud["staged_bytes"] == 0
    assert 0 < bud["work_bound_bytes"] < MS.WORK_MAX_BYTES
    assert bud["parts"]["workers"] == MS.MAX_BOX_WORKERS and bud["parts"]["synthetic"] > 0
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_gate_disk_plan", str(ROOT / "tools/ops/gate_disk_plan.py"))
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    assert round(D.SMOKE_CAP_GB * 1e9) == MS.WORK_MAX_BYTES          # the gate's cap is the budget


# ------------------------------------------------------------------ the wiring: unconditional, in order, judged
JOB = (ROOT / "tools" / "ops" / "release_gate_job.sh").read_text()


def test_the_smoke_is_the_last_phase_of_the_one_gate_job():
    lines = JOB.splitlines()
    order = ["measuresmoke-setup", "identity", "pixi", "build", "overlay", "shardcheck", "gate", "runtime-qualify",
             "hard-done", "measuresmoke", "complete"]
    at = [lines.index("PHASE=" + p) for p in order]
    assert at == sorted(at)                        # after the verification, last; its setup in the background first
    assert "GATE_JOB=" not in JOB and "$GATE_JOB\"" not in JOB   # no second job: one submit, one sync, one copy
    setup = JOB[JOB.index("\nPHASE=measuresmoke-setup"):JOB.index("\nPHASE=identity")]
    assert setup.count("> \"$J/") >= 2 and "MS_VPID=$!" in setup and "MS_FPID=$!" in setup   # backgrounded
    assert "release_measurement_smoke.py stage" not in JOB          # the smoke stages its data per group itself
    code = JOB[JOB.index("\nPHASE=measuresmoke\n"):JOB.index("\nPHASE=complete")]
    run = code[code.index("if [ \"$mrc\" = 0 ]; then\n    # run by"):]
    for word in ("GATE_REUSE", "GATE_UNITS", "selector", "GATE_IDENTITY_COLUMNS"):
        assert word not in run, word               # no change detection decides whether it runs
    assert "tools/release_measurement_smoke.py run" in code and '--work "$GATE_WORK/measuresmoke-work"' in code
    assert 'wait "$MS_VPID"' in code and 'wait "$MS_FPID"' in code and "MS_DPID" not in JOB
    assert '--data-plan "$IN/measuresmoke-data.json"' in code and "--max-work-gb" in code
    assert "MEASURESMOKE (plan|busy|disk|vendor=)" in code and "${PIPESTATUS[0]}" in code
    # SOFT: the smoke never changes the job's exit, which hard-done settled before it
    assert JOB.rstrip().endswith('PHASE=complete\nexit "$grc"') and "grc=$mrc" not in JOB
    hard = JOB[JOB.index("\nPHASE=hard-done"):JOB.index("\nPHASE=measuresmoke\n")]
    assert "[ \"$grc\" != 0 ] || grc=$irc" in hard and "[ \"$grc\" != 0 ] || grc=$src" in hard
    assert "gate-hard.json" in hard and 'copy_reports "$GATE_WORK/out" "$RES/out"' in hard
    assert '--snapshot "$json" "$RES" hard' in hard  # uploaded at once, ahead of the smoke
    # the capped qualifier leaves the smoke its ceiling
    assert "- 10 - MS_RESERVE" in JOB and "MS_RESERVE=${GATE_MEASURE_SMOKE_MINUTES:-90}" in JOB


def test_the_one_gate_job_carries_the_smoke_data_and_no_second_job(tmp_path):
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    import release_gate as RG
    out = tmp_path / "abc-release-light-v1"
    gate = RG.Plan("nvidia", "0.8.67", "a" * 40, tmp_path, out, [tmp_path / "w.whl"], "t1")
    assert gate.launch()[gate.launch().index("--cap") + 1] == str(RG.NV_CAP_MIN) == "240"
    env = gate.env_text()
    assert "GATE_JOB=" not in env and "MEASURE_SMOKE_PIP=(" in env and "MEASURE_SMOKE_FIXTURE_PIP=(" in env
    assert "GATE_MEASURE_SMOKE_MINUTES=%d" % RG.MEASURE_SMOKE_MINUTES in env
    assert any(r.endswith("/in/" + RG.MEASURE_SMOKE_DATA) for _, r, _ in gate.uploads())
    assert not hasattr(RG, "JOBS") and not hasattr(RG, "MEASURE_SMOKE_CAP_MIN")
    with pytest.raises(TypeError):
        RG.Plan("amd", "0.8.67", "a" * 40, tmp_path, out, [], "t4", job="measuresmoke")
    with pytest.raises(SystemExit):
        RG.parse(["amd", "0.8.67", "--source-commit", "a" * 40, "--source-tree", ".", "--out", "o",
                  "--candidate-wheel", "w", "--job", "measuresmoke"])
    assert RG.parse(["amd", "0.8.67", "--source-commit", "a" * 40, "--source-tree", ".", "--out", "o",
                     "--candidate-wheel", "w"]).until == "hard"
    assert RG.HARD_MARKER in RG.GATE_FETCH


def fake_release(tmp_path, monkeypatch, strict=False):
    import release as R
    r = R.Release.__new__(R.Release)
    r.dry, r.lines = False, []
    r.say = r.lines.append
    r.args = type("A", (), dict(strict=strict, identity_on_wheels=True))()
    out = tmp_path / "gate"
    monkeypatch.setattr(r, "readiness_status", lambda write: None, raising=False)
    monkeypatch.setattr(r, "readiness", lambda: type("X", (), dict(begin=lambda s, k: "id",
                                                                   record=lambda s, *a, **k: None))(), raising=False)
    monkeypatch.setattr(r, "record_candidate_requirement", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(R.Release, "rel", tmp_path, raising=False)
    monkeypatch.setattr(r, "gate_out", lambda vendor: out, raising=False)
    monkeypatch.setattr(r, "soft_fail", lambda vendor, name, reason, log=None: r.lines.append(
        f"SOFT FAIL {vendor} {name}: {reason[:60]}"), raising=False)
    return r, out


def write_smoke(out, ok):
    ids = ["algos/a/taxi/rows=2000"]
    (out / "measuresmoke").mkdir(parents=True, exist_ok=True)
    (out / "measuresmoke" / MS.SUMMARY).write_text(json.dumps(MS.summarize("nvidia", ids, {ids[0]: dict(
        verdict=MS.PASS if ok else MS.FAIL, reason="x", batch=0)}, {})))


def test_the_gate_step_completes_on_its_hard_parts_and_the_smoke_lands_later(tmp_path, monkeypatch):
    r, out = fake_release(tmp_path, monkeypatch)
    state = {"hard": False}
    seen = []

    def gate_ok(vendor, measure=None):
        seen.append(measure)
        return (True, "PASS job") if state["hard"] else (False, "no gate report")

    def gate_job(vendor, run_id, work, output):
        # release_gate.py --until hard returned: the HARD reports and the job's state are home, the smoke is not
        state["hard"] = True
        work.mkdir(parents=True, exist_ok=True)
        (work / f"nvidia-{output.name}.job.json").write_text(json.dumps(dict(tag="t", vendor="nvidia")))
        output.mkdir(parents=True, exist_ok=True)
        (output / "gate-hard.json").write_text("{}")
    spawned = []
    monkeypatch.setattr(r, "gate_ok", gate_ok, raising=False)
    monkeypatch.setattr(r, "launch_gate_job", gate_job, raising=False)
    monkeypatch.setattr(r, "spawn", lambda cmd, env: spawned.append((cmd, env)) or 4242, raising=False)
    why = r.run_gate("nvidia")
    # the verdict is final without the smoke: identity-admit (AFTER the gate steps) proceeds now
    assert why.startswith("PASS job; measurement smoke running on the box") and seen == [None, False, False]
    assert len(spawned) == 1 and spawned[0][0][-5:-3] == ["await-final", "--state"]
    assert spawned[0][1]["MOJOLEARN_SMOKE_FETCH_DONE"].endswith("measuresmoke-fetch-nvidia.done")
    assert (tmp_path / "gates" / "measuresmoke-fetch-nvidia.pid").read_text().strip() == "4242"
    import release as R
    assert R.AFTER["boxes-down"] and {"gate-nvidia", "gate-amd"} <= set(R.AFTER["boxes-down"])
    # the fetch brought the summary home (its done file written): settle records the failure SOFT, once per call
    write_smoke(out, ok=False)
    (tmp_path / "gates" / "measuresmoke-fetch-nvidia.done").write_text("0\n")
    monkeypatch.setattr(R, "GATE_VENDORS", ("nvidia",))
    lines = r.settle_measure_smoke("record")
    assert lines and "SOFT FAIL" in lines[0] and any(l.startswith("SOFT FAIL nvidia measuresmoke") for l in r.lines)
    write_smoke(out, ok=True)
    assert r.settle_measure_smoke("boxes-down")[0].endswith("PASS")


def test_a_running_smoke_holds_its_box_and_settle_waits_for_it(tmp_path, monkeypatch):
    import os
    import release as R
    r, out = fake_release(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "GATE_VENDORS", ("nvidia",))
    gates = tmp_path / "gates"
    gates.mkdir()
    (gates / "measuresmoke-fetch-nvidia.pid").write_text(str(os.getpid()))   # a live process
    (gates / "nvidia-gate.job.json").write_text("{}")
    assert r.smoke_pending_vendors() == {"nvidia"}       # the keeper holds it; the watchdog counts it busy
    monkeypatch.setattr(r, "selected_steps", lambda: set(), raising=False)
    assert r.boxes_pending() == {"nvidia"}
    ticks = []

    def sleep(seconds):
        ticks.append(seconds)
        if len(ticks) == 3:                               # the fetch ends: summary home, done file written
            write_smoke(out, ok=True)
            (gates / "measuresmoke-fetch-nvidia.done").write_text("0\n")
    monkeypatch.setattr(R.time, "sleep", sleep)
    lines = r.settle_measure_smoke("boxes-down")
    assert len(ticks) == 3 and lines[0].endswith("PASS") and r.smoke_pending_vendors() == set()


def test_strict_waits_for_the_smoke_and_stops_on_it(tmp_path, monkeypatch):
    import release as R
    r, out = fake_release(tmp_path, monkeypatch, strict=True)
    state = {"hard": False}

    def gate_ok(vendor, measure=None):
        if not state["hard"]:
            return False, "no gate report"
        if measure is False:
            return True, "PASS job"
        errors = __import__("release_identity_wheels").measure_smoke_errors(out)
        return (False, "; ".join(errors)) if errors else (True, "PASS")
    cmds = []

    def gate_job(vendor, run_id, work, output):
        state["hard"] = True                              # --until final: the final upload is home
        write_smoke(out, ok=False)
    monkeypatch.setattr(r, "gate_ok", gate_ok, raising=False)
    monkeypatch.setattr(r, "launch_gate_job", gate_job, raising=False)
    monkeypatch.setattr(r, "spawn", lambda cmd, env: cmds.append(cmd), raising=False)
    with pytest.raises(R.StepFailed, match="measuresmoke"):
        r.run_gate("nvidia")
    assert cmds == []                                     # nothing in the background under --strict
    write_smoke(out, ok=True)
    assert r.run_gate("nvidia") == "PASS"


def test_identity_admit_never_waits_on_the_smoke():
    import release as R
    # identity-admit settles after the gate steps (whose verdict no longer includes the smoke); no smoke step exists
    for table in (R.AFTER, R.NEEDS):
        assert not any("smoke" in step and step.startswith(("measure", "gate-")) for step in table)
    assert not any(name.startswith("measure") for name, *_ in R.STEP_TABLE)
    assert "record" in R.AFTER and set(R.AFTER["record"]) >= {"gate-nvidia", "gate-amd"}


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


def test_the_release_verdict_judges_it_in_both_gate_paths(tmp_path):
    riw = (ROOT / "tools" / "release_identity_wheels.py").read_text()
    single = riw[riw.index("def single_gate_ok"):riw.index("def gate_identity_errors")]
    assert "measure_smoke_errors(output)" in single and "measure = self.strict()" in single
    rel = (ROOT / "tools" / "release.py").read_text()
    light = rel[rel.index("    def gate_ok(self, vendor, measure=None):"):rel.index("RUNTIME_GATE_FAILED = ")]
    assert "measure_smoke_errors(output)" in light and "measure = self.strict()" in light
    import release_identity_wheels as RIW
    assert RIW.measure_smoke_errors(pathlib.Path("/nonexistent")) == ["measuresmoke: " + MS.NO_SUMMARY]
    assert str(RIW.measure_smoke_summary(pathlib.Path("/g"))) == "/g/measuresmoke/" + MS.SUMMARY
    # a gate of an earlier freeze (the smoke's own job) is still read
    legacy = tmp_path / RIW.MEASURE_JOB_DIR / "measuresmoke"
    legacy.mkdir(parents=True)
    (legacy / MS.SUMMARY).write_text("{}")
    assert RIW.measure_smoke_summary(tmp_path) == legacy / MS.SUMMARY


def test_the_tool_is_overlaid_and_its_summary_comes_home():
    sys.path.insert(0, str(ROOT / "tools" / "ops"))
    import release_gate as RG
    assert "tools/release_measurement_smoke.py" in RG.GATE_OVERLAY
    assert "measuresmoke/" + MS.SUMMARY in RG.GATE_FETCH
    assert not any(f.startswith("measuresmoke") and ("work" in f or "batches" in f) for f in RG.GATE_FETCH)
    # harness packages only: no opponent library is installed for the smoke. CatBoost is the one exception, the
    # untimed border oracle of our pre-binned trees arms (tools/speed_gbdt_arm.py prebinned_data), in the measured
    # venv only, pinned as the trees board pins it.
    for pkg in RG.MEASURE_SMOKE_PIP + RG.MEASURE_SMOKE_FIXTURE_PIP:
        name = pkg.split("=")[0].split(">")[0].lower()
        if name == "catboost" and pkg in RG.MEASURE_SMOKE_PIP:
            assert pkg == "catboost==1.2.10", pkg
            continue
        assert name not in ("cuml", "cupy", "xgboost", "lightgbm", "catboost", "transformers", "mamba-ssm",
                            "faiss-cpu", "implicit", "cuvs", "cugraph", "gpytorch"), pkg
    assert "torch" not in [p.split("=")[0] for p in RG.MEASURE_SMOKE_PIP]   # torch only in the fixture venv


# ------------------------------------------------------------------ the disk: stage per group, clean after it, cap
class FakeBoard:
    """The bench_board surface the smoke reads, over three tiny datasets (no device, no data, no board run)."""
    SMOKE_ROWS = 2000
    SMOKE_STEP_S = 5
    QUALITY_EXEMPT = ()
    DATASETS = ("d1", "d2", "d3")
    R2_KEYS = {"d1": "gbm-bench/d1/d1.npz", "d2": "gbm-bench/d2/d2.npz", "d3": "gbm-bench/d3/d3.npz"}
    DATA_FILES = {"d1": "d1/d1.npz", "d2": "d2/d2.npz", "d3": "d3/d3.npz"}
    ALGOS = None

    @staticmethod
    def data_needs(races, datasets):
        return sorted({r["dataset"] for r in races if r["dataset"] in datasets}), []

    @staticmethod
    def load_result(path):
        return json.loads(pathlib.Path(path).read_text())

    @staticmethod
    def smoke_key(race):
        return race["id"]

    @staticmethod
    def smoke_verdict(race, rec, vendor):
        return []

    @staticmethod
    def unsupported_our_arms(race):
        return {}

    @staticmethod
    def repo_commit():
        return "c" * 40


def fake_races():
    out = []
    for ds, lanes in (("d1", ("a", "b")), ("d2", ("a", "b", "c")), ("d3", ("a",))):
        for lane in lanes:
            out.append(dict(id="classical/%s/%s/rows=2000" % (lane, ds), family="classical", lane=lane, dataset=ds,
                            rows=2000, our_arms={"ours": "identical"}, opponents=[], arms=["ours"]))
    # the neural recipe reads d1 too: d1 stays until the neural group closes
    out.append(dict(id="neural/mlp-fit/d1/shape=small", family="neural", lane="mlp-fit", dataset="d1", rows=None,
                    our_arms={"ours": "identical"}, opponents=[], arms=["ours"]))
    return out


def fake_box(tmp_path, monkeypatch, bytes_per_file=100, inputs=False):
    """The smoke's run over the fake plan: staging writes the file, prep writes a block, a batch writes its board.
    Returns the observation log ([(event, group, sorted data files present)]) and the run's argv."""
    import release_gate_profiles as P
    races = fake_races()
    home = tmp_path / "home"
    monkeypatch.setattr(MS, "_bb", lambda: FakeBoard)
    monkeypatch.setattr(MS, "plan", lambda vendor, B=None: races)
    monkeypatch.setattr(MS, "device_memory_mb", lambda vendor, field: None)
    monkeypatch.setattr(MS, "board_drops_scratch", lambda: True)
    monkeypatch.setattr(MS, "board_takes_inputs", lambda here=None: inputs)
    if inputs:
        # d2's recipes on the verify fixture; d1, d3 and the neural recipe on synthetic inputs (nothing staged)
        monkeypatch.setattr(MS, "smoke_inputs", lambda rs, B=None, index=None: {
            r["id"]: dict(input=MS.INPUT_VERIFY if r["dataset"] == "d2" else MS.INPUT_NONE, verify_lanes=[],
                          files=[FakeBoard.R2_KEYS[r["dataset"]]]) for r in rs})
        monkeypatch.setattr(MS, "budget", lambda *a, **k: dict(
            counts={MS.INPUT_VERIFY: 3, MS.INPUT_FILE: 0, MS.INPUT_NONE: 4}, staged_bytes=0,
            work_bound_bytes=10 ** 6, parts={}))
    seen = []

    def present():
        return sorted(str(p.relative_to(home)) for p in home.rglob("*.npz")) if home.exists() else []

    def stage_one(entry, root):
        dest = pathlib.Path(root) / entry["path"]
        if dest.is_file() and dest.stat().st_size == entry["bytes"]:
            return "present"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * entry["bytes"])
        return "fetched"

    def bounded_run(cmd, log, timeout, env=None, cwd=None):
        pathlib.Path(log).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(log).write_text("fake\n")
        ids = json.loads(pathlib.Path(cmd[cmd.index("--race-ids") + 1][1:]).read_text())
        cache = pathlib.Path(cmd[cmd.index("--cache") + 1])
        group = cache.parent.name
        if "--prep-only" in cmd:
            (cache / "blocks").mkdir(parents=True, exist_ok=True)
            (cache / "blocks" / "blk.npz.bin").write_bytes(b"b" * 50)
            seen.append(("prep", group, present()))
        else:
            assert "--drop-race-scratch" in cmd
            assert (cache / "blocks" / "blk.npz.bin").exists()          # its group's prep ran first
            out = pathlib.Path(cmd[cmd.index("--out") + 1] + "-smoke")
            (out / "raw").mkdir(parents=True, exist_ok=True)
            (out / "raw" / "scratch.json").write_text("{}")
            cell = dict(arm="ours", status="ok", quality={"auc": 0.9}, hash="h", mode="identical",
                        library_version="0.8.67", device="gpu0")
            (out / "board.json").write_text(json.dumps(dict(races={i: dict(cells=[cell]) for i in ids})))
            seen.append(("batch", group, present()))
        return dict(exit_code=0, timed_out=False, elapsed_seconds=0.0)

    monkeypatch.setattr(MS, "stage_one", stage_one)
    monkeypatch.setattr(P, "bounded_run", bounded_run)
    entries = [dict(key=FakeBoard.R2_KEYS[d], path="datasets/gbm-bench/" + FakeBoard.DATA_FILES[d],
                    bytes=bytes_per_file, sha256="0" * 64, url="https://r2/" + d) for d in FakeBoard.DATASETS]
    plan = tmp_path / "data.json"
    plan.write_text(json.dumps(dict(files=entries)))
    argv = ["run", "--vendor", "nvidia", "--python", "/v/bin/python", "--out", str(tmp_path / "out"),
            "--work", str(tmp_path / "work"), "--data-plan", str(plan), "--home", str(home), "--jobs", "1",
            "--data-root", str(home / "datasets" / "gbm-bench")]
    return seen, argv, home


def test_a_fake_plan_stages_and_cleans_per_group(tmp_path, monkeypatch, capsys):
    seen, argv, home = fake_box(tmp_path, monkeypatch)
    # d3 is already on the box at its pin (another job's file): used, never counted, never deleted
    (home / "datasets/gbm-bench/d3").mkdir(parents=True)
    (home / "datasets/gbm-bench/d3/d3.npz").write_bytes(b"x" * 100)
    assert MS.main(argv) == 0
    d1, d2, d3 = ("datasets/gbm-bench/%s/%s.npz" % (d, d) for d in ("d1", "d2", "d3"))
    by_group = {}
    for event, group, files in seen:
        by_group.setdefault(group, []).append((event, files))
    # each group's data is there before its prep and its batches; d2 is not staged before its own group
    assert all(d1 in f for _, f in by_group["ds-d1"]) and all(d2 not in f for _, f in by_group["ds-d1"])
    assert all(d2 in f for _, f in by_group["ds-d2"])
    # d2 is deleted as soon as its only group closed: the later groups never see it
    assert all(d2 not in f for _, f in by_group["ds-d3"] + by_group["neural"])
    # d1 stays for the neural group, which reads it too
    assert all(d1 in f for _, f in by_group["neural"])
    # after the run: every file this run fetched is gone, the pre-existing one is kept
    assert not (home / d1).exists() and not (home / d2).exists() and (home / d3).exists()
    # every group's cache and every passing batch's scratch is gone
    work = tmp_path / "work"
    assert not list(work.rglob("blk.npz.bin")) and not list(work.rglob("scratch.json"))
    assert not list(work.rglob("board.json"))
    doc = json.loads((tmp_path / "out" / MS.SUMMARY).read_text())
    assert doc["ok"] and doc["counts"]["PASS"] == len(fake_races())
    assert [g["name"] for g in doc["disk"]["groups"]] == ["ds-d1", "ds-d2", "ds-d3", "neural"]
    assert doc["disk"]["cap_bytes"] == int(MS.MAX_WORK_GB * 1e9) and doc["disk"]["peak_footprint_bytes"] > 0
    printed = capsys.readouterr().out
    assert "MEASURESMOKE disk free=" in printed and "after group ds-d2" in printed and "after staging ds-d1" in printed


def test_on_the_synthetic_inputs_nothing_is_staged_and_every_batch_reads_the_map(tmp_path, monkeypatch, capsys):
    seen, argv, home = fake_box(tmp_path, monkeypatch, inputs=True)
    monkeypatch.setattr(MS, "stage_one", lambda e, h: pytest.fail("the smoke staged %s" % e["key"]))
    assert MS.main(argv) == 0                                  # a data plan given is not read
    assert seen and all(files == [] for _, _, files in seen)   # no data file on the box at any point
    imap = json.loads((tmp_path / "work" / "smoke_inputs.json").read_text())
    assert imap["classical/a/d2/rows=2000"]["input"] == MS.INPUT_VERIFY
    doc = json.loads((tmp_path / "out" / MS.SUMMARY).read_text())
    assert doc["ok"] and doc["inputs"]["mode"] == MS.SYNTHETIC_INPUTS and doc["inputs"]["staged_bytes"] == 0
    assert doc["disk"]["cap_bytes"] == MS.WORK_MAX_BYTES                     # the budget is the cap
    out = capsys.readouterr().out
    assert "\nSMOKE inputs: generated on the box, 0 bytes staged\n" in out
    assert "MEASURESMOKE inputs verify-fixture=3 synthetic-file=0 NO-VERIFY-LANE=4 staged=0 work_bound=" in out


def test_the_cap_refuses_a_group_by_name_and_never_stages_it(tmp_path, monkeypatch, capsys):
    seen, argv, home = fake_box(tmp_path, monkeypatch, bytes_per_file=2 * 10 ** 6)
    assert MS.main(argv + ["--max-work-gb", "0.001"]) == 1           # 1 MB: every 2 MB file is past it
    assert not list(home.rglob("*.npz"))                              # nothing was staged
    assert all(event != "prep" for event, _, _ in seen)
    doc = json.loads((tmp_path / "out" / MS.SUMMARY).read_text())
    assert not doc["ok"] and doc["counts"]["FAIL"] == len(fake_races())
    reasons = {f["reason"] for f in doc["failures"]}
    assert all("--max-work-gb" in r and "refused, never staged" in r for r in reasons)
    assert any("staging group ds-d2 (3 recipes) needs 0.00 GB" in r for r in reasons)
    assert "MEASURESMOKE measuresmoke: staging group" in capsys.readouterr().out


def test_the_stager_counts_and_deletes_only_what_this_run_fetched(tmp_path):
    groups = [dict(name="g1", ids=["r1"], keys=["k1"]), dict(name="g2", ids=["r2"], keys=["k2"])]
    entries = [dict(key="k1", path="a/k1", bytes=600), dict(key="k2", path="a/k2", bytes=600)]
    st = MS.DataStager(entries, tmp_path, groups)
    assert st.need_bytes(groups[0]) == 600 and st.owned_bytes() == 0
    assert st.stage(groups[0], stage_fn=lambda e, h: "fetched") == [] and st.owned_bytes() == 600
    assert st.need_bytes(groups[0]) == 0                       # held by this run: nothing more to stage
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "k1").write_bytes(b"x" * 600)
    assert st.release(groups[0]) == 600 and not (tmp_path / "a" / "k1").exists() and st.owned_bytes() == 0
    assert "--max-work-gb" in MS.cap_refusal(groups[1], 600, 600, 1000)


def test_a_race_scratch_is_gone_once_its_record_is_written(tmp_path):
    import bench_board as B
    out = tmp_path / "b000-smoke"
    (out / "work").mkdir(parents=True)
    (out / "work" / "r1-ours.npz").write_bytes(b"x" * 10)
    (out / "evidence" / "r1").mkdir(parents=True)
    (out / "evidence" / "r1" / "raw.tar.gz").write_bytes(b"y" * 20)
    (out / "raw").mkdir()
    (out / "raw" / "r1.json").write_text("{}")
    (tmp_path / "outside").mkdir()
    assert B.drop_race_scratch(str(out), dict(evidence_directory="evidence/r1")) == 30
    assert not (out / "work").exists() and not (out / "evidence" / "r1").exists() and (out / "raw" / "r1.json").exists()
    assert B.drop_race_scratch(str(out), dict(evidence_directory="../outside")) == 0 and (tmp_path / "outside").exists()
    src = (ROOT / "tools" / "bench_board.py").read_text()
    loop = src[src.index("        rec = run_race(ctx, r)"):src.index('    print("bench_board: board %s"')]
    assert loop.index("save_result(rpath, result)") < loop.index("drop_race_scratch(out, rec)")
    assert MS.board_drops_scratch()


# ------------------------------------------------------------------ the inputs: one fixture set for verify and smoke
def test_the_smoke_input_map_names_verify_lanes_from_names_only_and_is_committed():
    import bench_board as B
    import identity_break as I
    races = MS.plan("nvidia", B)
    got = MS.smoke_inputs(races, B)
    assert set(got) == {r["id"] for r in races}
    for r in races:
        rec = got[r["id"]]
        assert all(l in I.LANES for l in rec["verify_lanes"])          # never an invented lane
        if rec["input"] == MS.INPUT_FILE:
            assert rec["files"] and (r["family"] in MS.FILE_LOADER_FAMILIES
                                     or any(f.startswith("corpus/") for f in rec["files"]))
        elif rec["input"] == MS.INPUT_VERIFY:
            assert rec["verify_lanes"]
        else:
            assert rec["input"] == MS.INPUT_NONE and not rec["verify_lanes"]
        if r["lane"] in I.LANES:
            assert rec["verify_lanes"] == [r["lane"]]
    text, counts = MS.inputs_table()
    assert (ROOT / "docs/measurements/SMOKE_INPUTS.md").read_text() == text   # regenerate with `inputs`
    assert sum(counts.values()) == len({r["id"] for v in VENDORS for r in MS.plan(v, B)})
