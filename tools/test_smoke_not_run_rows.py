# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ours-only smoke never reports a planned row as a bare "not run" (candidate 4c2340386abc, 2026-11-13).

Ten paper rows smoked as "arm=- not run" on NVIDIA and AMD: conv2d, gcn, graphsage, moe (no neural-layer fixture
set on the box: the ours probe refused) and gbdt-depthwise / gbdt-lossguide (`ours-prebinned` / `ours-gain0`
need catboost, the border oracle, which the ours-only venv lacked). The preflight dropped each race and the smoke
verdict lost the cause. No mojolearn, no GPU, no torch, no dataset:

    .pixi/envs/test/bin/python -m pytest -n auto tools/test_smoke_not_run_rows.py
"""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bench_board", os.path.join(HERE, "bench_board.py"))
bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bb)

NF_MISSING = ('stage=ready detail={"class": "infrastructure", "error": "neural fixtures missing: run '
              'tools/neural_fixtures.py generate (no /root/pc-cache/c7/algos-data/neural-fixtures/manifest.json)", '
              '"event": "error", "stage": "ready", "status": "not_ready"}')


def _algos_race(lane="conv2d", dataset="synthetic", rows=2000):
    return {"id": "algos/%s/%s/rows=%d" % (lane, dataset, rows), "family": "algos", "lane": lane,
            "dataset": dataset, "rows": rows, "arms": ["ours"], "our_arms": {"ours": "identical"},
            "opponents": []}


def _dropped(problems, status="FAILED"):
    return {"status": status, "problems": list(problems), "recipe": {}}


# --- the smoke verdict carries the preflight's cause ---------------------------------------------------------------

def test_smoke_verdict_reports_the_preflight_cause_not_not_run():
    race = _algos_race()
    pre = _dropped(["no parameter records", "missing parameter/version probe: ours",
                    "missing worker GPU/driver/OS evidence: ours", "probe refused ours: " + NF_MISSING])
    fails = bb.smoke_verdict(race, None, "nvidia", pre)
    assert len(fails) == 1 and fails[0][0] == "-"
    why = fails[0][1]
    assert why != "not run" and why.startswith("preflight FAILED: probe refused ours: ")
    assert "neural fixtures missing" in why


def test_smoke_verdict_without_a_preflight_record_is_still_not_run():
    assert bb.smoke_verdict(_algos_race(), None, "nvidia") == [("-", "not run")]
    assert bb.smoke_verdict(_algos_race(), None, "nvidia", None) == [("-", "not run")]


def test_preflight_dropped_reads_failed_and_skipped():
    result = {"failed_preflight_recipes": {"a": _dropped(["x"])},
              "skipped_recipes": {"b": {"problems": ["y"]}}}
    assert bb.preflight_dropped(result, "a")["status"] == "FAILED"
    assert bb.preflight_dropped(result, "b")["status"] == "REFUSED"
    assert bb.preflight_dropped(result, "c") is None
    assert bb.preflight_dropped({}, "a") is None


def test_write_smoke_names_the_cause_in_smoke_json(tmp_path, capsys):
    races = [_algos_race(), _algos_race("gcn", "istella")]
    result = {"races": {}, "config": {},
              "failed_preflight_recipes": {races[0]["id"]: _dropped(
                  ["missing parameter/version probe: ours", "probe refused ours: " + NF_MISSING])}}
    assert bb.write_smoke(str(tmp_path), races, result, "nvidia", "7/8") == 2
    doc = json.loads((tmp_path / "smoke.json").read_text())
    conv = doc["races"]["algos/conv2d/synthetic"]["failures"]
    assert conv[0]["reason"].startswith("preflight FAILED: probe refused ours:")
    # a race nothing reached at all keeps the plain verdict
    assert doc["races"]["algos/gcn/istella"]["failures"] == [{"arm": "-", "reason": "not run"}]
    out = capsys.readouterr().out
    assert "SMOKE-FAIL algos/conv2d/synthetic/rows=2000 arm=- preflight FAILED" in out


# --- the probe log's refusal line reaches the preflight ------------------------------------------------------------

def test_probe_refusals_parses_driver_lines_last_run_only(tmp_path):
    log = tmp_path / "probe.log"
    log.write_text("\n".join([
        "=== 2026-11-12T01:00:00Z python old run",
        "ALGOS-REFUSED lane=conv2d dataset=synthetic arm=ours stage=ready detail=OLD",
        "",
        "=== 2026-11-13T05:06:35Z python tools/bench_board_algos.py race --lane conv2d --params-only",
        "=== bench_board: host-memory watchdog not started: OSError()",
        "ALGOS-REFUSED lane=conv2d dataset=synthetic arm=ours " + NF_MISSING,
        "PARAMS-ONLY /x/conv2d-synthetic.params.json",
    ]) + "\n")
    got = bb.probe_refusals(str(log), ["ours"])
    assert list(got) == ["ours"] and got["ours"].startswith("stage=ready") and "OLD" not in got["ours"]
    assert "neural fixtures missing" in got["ours"]


def test_probe_refusals_keeps_the_specific_line_over_not_built_here(tmp_path):
    log = tmp_path / "probe.log"
    log.write_text("\n".join([
        "=== 2026-11-13T05:06:44Z python bench/speed/forest_speed_arm.py --lane gbdt-depthwise --params-only",
        "FSPEED-REFUSED lane=gbdt-depthwise arm=ours-prebinned reason=ModuleNotFoundError: No module named 'catboost'",
        "FSPEED-REFUSED lane=gbdt-depthwise arm=ours reason=asked for by --arms and not built here; built:",
        "FSPEED-REFUSED lane=gbdt-depthwise arm=ours-prebinned reason=asked for by --arms and not built here; built:",
    ]) + "\n")
    got = bb.probe_refusals(str(log), ["ours-prebinned"])
    assert got == {"ours-prebinned": "reason=ModuleNotFoundError: No module named 'catboost'"}
    assert bb.probe_refusals(str(tmp_path / "absent.log"), ["ours"]) == {}


# --- the ours-only venv holds the pre-binned arms' border oracle ---------------------------------------------------

def test_ours_only_venv_installs_the_trees_catboost_pin():
    pins = bb.opponent_pins()
    trees = next(v for k, v in pins.items() if k.startswith("trees-"))
    want = [r for r in trees[1] if r.lower().startswith("catboost==")]
    assert want, trees
    reqs = bb.ours_only_requirements(pins)
    assert reqs[0].startswith("scikit-learn==")
    assert want[0] in reqs
    # no other opponent library comes with it
    assert not any(r.lower().startswith(("xgboost", "lightgbm", "torch")) for r in reqs), reqs


# --- the neural-layer fixtures are made by the board, not assumed staged ---------------------------------------------

def test_fixtures_requirements_are_cpu_torch_at_the_board_pin():
    for vendor in ("nvidia", "amd", "apple"):
        reqs = bb.fixtures_requirements(vendor)
        assert reqs[0] == "numpy"
        assert reqs[1].startswith("torch==") and "+" not in reqs[1]
        assert reqs[1] == bb.DEFAULT_TORCH_SPEC[vendor].split("+")[0]
        assert reqs[2].startswith("torch-geometric==")


def test_layer_races_are_the_fixture_readers():
    races = [_algos_race("conv2d"), _algos_race("gcn", "istella"), _algos_race("graphsage", "taxi"),
             _algos_race("moe"), _algos_race("cross-val-predict", "istella"),
             dict(_algos_race("gbdt-depthwise", "istella"), family="trees")]
    assert [r["lane"] for r in bb.layer_races(races)] == ["conv2d", "gcn", "graphsage", "moe"]


def _ctx(tmp_path, fixtures_venv=None):
    return {"out": str(tmp_path / "out"), "algos_data": str(tmp_path / "algos-data"), "vendor": "nvidia",
            "nice": 0, "fixtures_venv": fixtures_venv}


def test_ensure_neural_fixtures_reuses_a_covering_set(tmp_path, monkeypatch):
    monkeypatch.delenv("MOJOLEARN_NEURAL_FIXTURES", raising=False)
    ctx = _ctx(tmp_path, fixtures_venv=str(tmp_path / "venv-nf"))
    race = _algos_race()
    data = os.path.abspath(bb.algos_data_dir(ctx, race["rows"], race))
    fixdir = tmp_path / "algos-data" / "neural-fixtures"
    fixdir.mkdir(parents=True)
    (fixdir / "manifest.json").write_text(json.dumps({"data_dirs": [data], "entries": {
        "conv2d/synthetic/nb4": {"lane": "conv2d"}}}))
    monkeypatch.setattr(bb, "ensure_fixtures_venv", lambda *a: (_ for _ in ()).throw(AssertionError("no venv")))
    assert bb.ensure_neural_fixtures(ctx, [race]) == str(fixdir)
    assert not (tmp_path / "venv-nf").exists()


def test_ensure_neural_fixtures_without_a_venv_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MOJOLEARN_NEURAL_FIXTURES", raising=False)
    assert bb.ensure_neural_fixtures(_ctx(tmp_path), [_algos_race()]) is None
    assert "neural fixtures missing for 1 layer race(s)" in capsys.readouterr().out
    assert bb.ensure_neural_fixtures(_ctx(tmp_path), [dict(_algos_race("lof", "taxi"))]) is None


def test_ensure_neural_fixtures_generates_over_every_mirror(tmp_path, monkeypatch):
    monkeypatch.delenv("MOJOLEARN_NEURAL_FIXTURES", raising=False)
    ctx = _ctx(tmp_path, fixtures_venv=str(tmp_path / "venv-nf"))
    (tmp_path / "algos-data" / "rows-full").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(bb, "ensure_fixtures_venv", lambda vdir, vendor, log: "/nf/bin/python")
    monkeypatch.setattr(bb, "child_env", lambda ctx, extra=None: {"MOJOLEARN_NEURAL_FIXTURES": "/elsewhere"})

    def fake_run(cmd, env, log, timeout, cwd=None, nice=0):
        calls.append((cmd, env))
        if "generate" in cmd:
            out = cmd[cmd.index("--out") + 1]
            os.makedirs(out)
            with open(os.path.join(out, "manifest.json"), "w") as fh:
                json.dump({"data_dirs": [], "entries": {}}, fh)
        return 0
    monkeypatch.setattr(bb, "run_logged", fake_run)
    races = [_algos_race(), _algos_race("gcn", "istella")]
    fixdir = bb.ensure_neural_fixtures(ctx, races)
    assert fixdir == str(tmp_path / "algos-data" / "neural-fixtures")
    assert os.path.isfile(os.path.join(fixdir, "manifest.json")) and not os.path.exists(fixdir + ".new")
    gen = calls[0][0]
    assert gen[0] == "/nf/bin/python" and gen[gen.index("--lanes") + 1] == "conv2d,gcn"
    datas = [gen[i + 1] for i, a in enumerate(gen) if a == "--data"]
    assert sorted(os.path.basename(d) for d in datas) == ["rows-2000", "rows-full"]
    assert "MOJOLEARN_NEURAL_FIXTURES" not in calls[0][1]
    assert "verify" in calls[1][0]


def test_the_measurement_job_passes_the_fixtures_venv():
    with open(os.path.join(HERE, "ops", "paper_compare_job.sh")) as fh:
        text = fh.read()
    common = [line for line in text.splitlines() if line.startswith("COMMON=(")]
    assert len(common) == 1 and '--fixtures-venv "$CACHE_ROOT/venv-nf"' in common[0]
