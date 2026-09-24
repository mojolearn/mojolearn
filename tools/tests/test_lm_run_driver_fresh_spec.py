"""The driver re-reads the spec's rental parameters when a segment starts,
and refuses a spec whose plan changed under the run."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lm_run_driver as drv  # noqa: E402


def _spec(tmp_path, wheel="0.8.15", steps=10):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"run": "runs/x", "recipe": "r.json", "recipe_key": "runs/x/recipe.json",
                             "tokens_stage": "corpus/t", "wheel": wheel, "amd_size": "gpu-mi325x1-256gb",
                             "routes": {"A": [{"segment": "1", "vendor": "nvidia", "steps": steps}],
                                        "B": [{"segment": "1", "vendor": "amd", "steps": steps}]}}))
    return p


def test_a_changed_wheel_and_size_are_read_at_segment_start(tmp_path):
    p = _spec(tmp_path)
    plan = drv.segment_plan(drv.load_spec(p))
    p.write_text(p.read_text().replace('"0.8.15"', '"0.8.17"').replace("gpu-mi325x1-256gb", "gpu-mi325x8-2048gb"))
    now = drv.fresh_spec(p, plan)
    assert now["wheel"] == "0.8.17" and now["amd_size"] == "gpu-mi325x8-2048gb"


def test_a_changed_plan_is_refused_by_name(tmp_path):
    p = _spec(tmp_path)
    plan = drv.segment_plan(drv.load_spec(p))
    p.write_text(_spec(tmp_path, steps=12).read_text())
    with pytest.raises(RuntimeError) as err:
        drv.fresh_spec(p, plan)
    assert str(p) in str(err.value) and "step counts changed" in str(err.value)


def test_an_unchanged_spec_reads_back_equal(tmp_path):
    p = _spec(tmp_path)
    spec = drv.load_spec(p)
    assert drv.fresh_spec(p, drv.segment_plan(spec)) == spec


def test_after_adds_a_dependency_and_a_changed_after_is_a_plan_change(tmp_path):
    p = _spec(tmp_path)
    spec = drv.load_spec(p)
    plan = drv.segment_plan(spec)
    b1 = [e for e in plan if e["route"] == "B"][0]
    assert b1["depends"] == [("A", "1")]  # the seed and the chain it is held to are one dependency
    d = json.loads(p.read_text())
    d["routes"]["B"][0]["after"] = ["A/1", "A/9"]
    p.write_text(json.dumps(d))
    b1 = [e for e in drv.segment_plan(drv.load_spec(p)) if e["route"] == "B"][0]
    assert b1["depends"] == [("A", "1"), ("A", "9")]
    with pytest.raises(RuntimeError):
        drv.fresh_spec(p, plan)


def test_route_a_starts_first_and_a_ready_live_a_segment_holds_route_b(tmp_path, monkeypatch):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"run": "runs/x", "recipe": "r.json", "recipe_key": "k", "tokens_stage": "t",
                             "routes": {"A": [{"segment": "1", "vendor": "nvidia", "steps": 10},
                                              {"segment": "2", "vendor": "live", "steps": 4, "first": "nvidia", "shards": [44, 20]}],
                                        "B": [{"segment": "1", "vendor": "amd", "steps": 10, "after": ["A/2"]},
                                              {"segment": "2", "vendor": "live", "steps": 4, "first": "amd", "shards": [20, 44]}]}}))
    started = []

    def fake_start(spec, e, out, ledger):
        started.append("%s/%s" % (e["route"], e["segment"]))
        raise RuntimeError("stop after the first start")

    monkeypatch.setattr(drv, "_start", fake_start)
    out = tmp_path / "out"
    out.mkdir()
    # A/1 landed: A/2 (live) and B/1 (after A/2) and B/2 (needs A/2) are the candidates; only A/2 may start
    drv.Ledger(out).land(dict(route="A", segment="1"), dict(verdict="PASS", checkpoints={}))
    monkeypatch.setattr(drv.time, "sleep", lambda s: None)
    rc = drv.cmd_run(type("A", (), dict(spec=str(p), out=str(out), parallel=2))())
    assert rc == 1 and started == ["A/2"]


def test_a_live_segment_rents_with_its_own_lease_and_cap(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(drv.subprocess, "run", lambda argv, **kw: (calls.append(argv), type("R", (), {"returncode": 0})())[1])
    spec = {"run": "runs/x", "lease_minutes": 1440, "dollar_cap": 120, "routes": {}}
    e = dict(route="B", segment="2", vendor="live", lease_minutes=2000, dollar_cap=260)
    drv.rent_live(spec, e, tmp_path / "nv.sh", tmp_path / "amd.sh", tmp_path)
    argv = calls[-1]
    assert argv[argv.index("--minutes") + 1] == "2000" and argv[argv.index("--dollar-cap") + 1] == "260"
    e2 = dict(route="A", segment="3", vendor="live")
    drv.rent_live(spec, e2, tmp_path / "nv.sh", tmp_path / "amd.sh", tmp_path)
    argv = calls[-1]
    assert argv[argv.index("--minutes") + 1] == "1440" and argv[argv.index("--dollar-cap") + 1] == "120"
