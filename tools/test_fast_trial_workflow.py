"""Tooling test for .github/workflows/fast-trial.yml and tools/ops/fast_trial.py (the per-estimator `fast` trial)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = ROOT / "python/mojolearn/tests"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ft = _load("fast_trial", ROOT / "tools/ops/fast_trial.py")


def _steps_text(job) -> str:
    return "\n".join(s.get("run", "") for s in job["steps"])


def test_workflow_shape():
    wf = yaml.safe_load((ROOT / ".github/workflows/fast-trial.yml").read_text())
    on = wf.get("on", wf.get(True))
    assert set(on) == {"workflow_dispatch"}  # manual only (light-checks.yml refuses push/schedule)
    inputs = on["workflow_dispatch"]["inputs"]
    assert {"ref", "wheel_url", "wheel_name", "wheel_sha256"} <= set(inputs)  # verify-smoke.yml's wheel inputs
    assert inputs["families"]["default"] == "0xfffffff"
    assert inputs["exact_families"]["default"] == "" and inputs["shards"]["default"] == "1"
    assert set(wf["jobs"]) == {"plan", "trial", "summary"}
    trial = wf["jobs"]["trial"]
    assert trial["strategy"]["fail-fast"] is False and "needs.plan.outputs.shards" in trial["strategy"]["matrix"]["shard"]
    run = _steps_text(trial)
    # the one fetch path and the smoke's install step, never a second copy
    assert "tooling/tools/ops/fetch_built_wheel.sh" in run and "curl" not in run
    install = next(s for s in trial["steps"] if "wheel_verify_smoke.sh" in s.get("run", ""))
    assert install["env"]["STEPS"] == "install"
    # every core of the runner; the units split across shards
    assert 'fast_trial.py run' in run and '--workers "$(nproc)"' in run
    assert '--shard "$SHARD" --shards "$SHARDS"' in run and '--exact "$EXACT"' in run
    upload = trial["steps"][-1]
    assert upload["if"] == "always()" and "trial-out" in upload["with"]["path"]
    summary = wf["jobs"]["summary"]
    assert summary["if"] == "always()" and summary["needs"] == "trial"
    srun = _steps_text(summary)
    assert "fast_trial.py summary" in srun and "fast_ledger.json" in srun and "SUMMARY.md" in srun
    assert summary["steps"][-1]["if"] == "always()" and summary["steps"][-1]["with"]["path"] == "report/"


def test_verify_smoke_uses_the_shared_fetch():
    wf = yaml.safe_load((ROOT / ".github/workflows/verify-smoke.yml").read_text())
    run = _steps_text(wf["jobs"]["smoke"])
    assert "tooling/tools/ops/fetch_built_wheel.sh" in run and "curl" not in run
    script = (ROOT / "tools/ops/fetch_built_wheel.sh").read_text()
    assert "sha256sum -c" in script and "^[0-9a-f]{40}$" in script


def test_units_cover_every_estimator_once_across_shards():
    module = ft.load_test(TESTS_ROOT)
    # The trial runs FAMILIES and WALK (tools/ops/fast_trial.py); lane fx-fast-per-estimator (2026-11-19) added
    # the 67 WALK estimators and their family labels, so the label set is FAMILIES + WALK, each name once.
    names = [row[0] for row in module.FAMILIES + module.WALK]
    assert len(names) == len(set(names))
    assert set(names) == set(module.ESTIMATOR_FAMILY)  # every estimator has a family label
    trial_list = ft.trials("0xfffffff", "0x88")
    every = [u for k in range(3) for u in ft.units(names, trial_list, k, 3)]
    assert sorted(every) == sorted((t, m, n) for t, m in trial_list for n in names)
    assert ft.trials("0xfffffff", "") == [("families", "0xfffffff")]


def test_mask_range():
    assert ft.mask("0xfffffff") == "0xfffffff" and ft.mask("136") == "0x88"
    try:
        ft.mask("0x10000000")
    except Exception as error:  # argparse.ArgumentTypeError
        assert "outside" in str(error)
    else:
        raise AssertionError("a mask above MODE_FAMILIES_ALL must be refused")


def test_selected_families_switch(monkeypatch):
    module = ft.load_test(TESTS_ROOT)
    monkeypatch.setenv("MOJOLEARN_FAST_ESTIMATORS", "PCA,Ridge")
    assert [row[0] for row in module.selected_families()] == ["PCA", "Ridge"]
    monkeypatch.setenv("MOJOLEARN_FAST_ESTIMATORS", "NoSuchEstimator")
    try:
        module.selected_families()
    except ValueError as error:
        assert "NoSuchEstimator" in str(error)
    else:
        raise AssertionError("an unknown estimator must be refused")


def test_summary_rolls_rows_up_by_family(tmp_path):
    out = tmp_path / "shard-0"
    (out / "units").mkdir(parents=True)
    rows = [("families", "0xfffffff", "PCA", "0"), ("families", "0xfffffff", "Ridge", "1"),
            ("families", "0xfffffff", "Lasso", "-11")]
    (out / "units.tsv").write_text("trial\tmask\testimator\trc\ttimed_out\tseconds\tmaxrss_kb\tledger\n"
                                   + "".join(f"{t}\t{m}\t{e}\t{rc}\tFalse\t3.0\t1000\tTrue\n" for t, m, e, rc in rows))
    (out / "units" / "families-PCA.json").write_text(json.dumps(dict(answer="identical", vendor="cpu", rows={
        "PCA": dict(status="answered", family="PCA", ops={"transform": "same"}, attribute="fast", param="fast")})))
    (out / "units" / "families-Ridge.json").write_text(json.dumps(dict(answer="identical", vendor="cpu", rows={
        "Ridge": dict(status="mixed", family="LINEAR", attribute="fast", param="fast",
                      ops={"predict": "same", "score": "refused refused: ValueError: linear_score_native_pack"})})))
    (out / "units" / "families-Lasso.log").write_text("collected 1 item\nFatal Python error: Segmentation fault\n")
    ledger, md = tmp_path / "fast_ledger.json", tmp_path / "SUMMARY.md"
    assert ft.main(["summary", "--out", str(out), "--ledger", str(ledger), "--md", str(md)]) == 0
    text = md.read_text()
    assert "- PCA: **answered**" in text and "- LINEAR: **mixed**" in text
    assert "score: refused refused: ValueError: linear_score_native_pack" in text  # the door to edit
    assert "Segmentation fault" in text and "crashed" in text
    merged = json.loads(ledger.read_text())
    assert merged["families"]["rows"]["Lasso"]["status"] == "crashed"


def test_family_status():
    assert ft.family_status([{"status": "answered"}, {"status": "unavailable"}]) == "answered"
    assert ft.family_status([{"status": "answered"}, {"status": "refused"}]) == "mixed"
    assert ft.family_status([{"status": "unavailable"}]) == "unavailable"
    assert ft.family_status([{"status": "answered"}, {"status": "no_parameter"}]) == "answered"
    assert ft.family_status([{"status": "no_parameter"}, {"status": "unavailable"}]) == "no_parameter"
    assert ft.family_status([{"status": "refused"}, {"status": "crashed"}]) == "crashed"


# ---- the CPU coverage mode (lane fx-coverage-smokes, 2026-11-20) ---------------------------------------------------

def test_workflow_coverage_mode():
    wf = yaml.safe_load((ROOT / ".github/workflows/fast-trial.yml").read_text())
    on = wf.get("on", wf.get(True))
    assert on["workflow_dispatch"]["inputs"]["mode"]["default"] == "trial"
    assert "mode must be trial or coverage" in _steps_text(wf["jobs"]["plan"])
    run = _steps_text(wf["jobs"]["trial"])
    # the same install, the same runner, every core, the same shards
    assert 'fast_trial.py coverage --python "$VENV/bin/python"' in run
    cov = run[run.index("fast_trial.py coverage"):run.index("fast_trial.py run")]
    assert '--workers "$(nproc)"' in cov and '--shard "$SHARD" --shards "$SHARDS"' in cov
    # GitHub's bash is `-eo pipefail`: every rc is captured with ||, so the log tail always prints
    # no bare `rc=$?` on a line of its own (under `bash -e` it never runs after a failure); `cmd || rc=$?` is the form.
    # The old substring test `"rc=$?\n" not in run` also rejected `|| rc=$?` at a line end (cpc2-0007, 2026-11-20).
    assert all(line.strip() != "rc=$?" for line in run.splitlines()) and "|| rc=$?" in run
    srun = _steps_text(wf["jobs"]["summary"])
    assert "fast_trial.py coverage-summary" in srun and "coverage_ledger.json" in srun
    assert "cpu-coverage" in wf["jobs"]["summary"]["steps"][-1]["with"]["name"]


def test_coverage_split_and_dispatch_argv():
    probe, other = ft.coverage_split(["PCA", "cross_val_score", "Ridge", "PCA", "linalg"])
    assert probe == ["PCA", "Ridge"] and other == ["cross_val_score", "linalg"]
    d = _load("verify_smoke_dispatch", ROOT / "tools/ops/verify_smoke_dispatch.py")
    trial = d.fast_trial_argv("b", "c" * 40, "w.whl", "u", "s", "t", "0xfffffff")
    assert not any(x.startswith("mode=") for x in trial)  # a --ref whose workflow predates `mode` still takes it
    cov = d.fast_trial_argv("b", "c" * 40, "w.whl", "u", "s", "t", "0xfffffff", shards=4, mode="coverage")
    assert "mode=coverage" in cov and "shards=4" in cov


FAKE_PACKAGE = """
import os, time
__all__ = ["Good", "Refuser", "Crasher", "NeedsThing", "Sleeper", "CONST", "helper"]
CONST = 3
def helper():
    return 1
def vendor():
    return "cpu"
class Good:
    def __init__(self, numeric_mode=None, n_estimators=100):
        assert numeric_mode == "identical" and n_estimators == 4
    def fit(self, X, y):
        assert len(X) == len(y)
        return self
    def predict(self, X):
        return X[:, 0]
    def transform(self, X):
        return X[:, :2]
    def inverse_transform(self, X):
        assert X.shape[1] == 2
        return X
    def save(self, path):
        open(path, "w").write("m")
    @classmethod
    def load(cls, path):
        assert open(path).read() == "m"
        return cls(numeric_mode="identical", n_estimators=4)
class Refuser:
    def fit(self, X):
        raise RuntimeError("Refuser has no CPU route")
class Crasher:
    def fit(self, X, y=None):
        return self
    def boom(self, X):
        os.abort()
    def later(self):
        return 1
    def predict(self, X):
        raise ValueError("predict has no host binding")
class NeedsThing:
    def __init__(self, thing):
        pass
class Sleeper:
    def fit(self, X):
        return self
    def wait(self):
        time.sleep(120)
"""


def test_coverage_child_end_to_end(tmp_path):
    """The real child against a fake installed `mojolearn`: ran, refused, not covered, a crash the parent restarts
    past (the methods after it still run), a timeout, and a capitalised name that is not a class."""
    import pytest
    pytest.importorskip("numpy")
    pkg = tmp_path / "site" / "mojolearn"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_PACKAGE)
    wrapper = tmp_path / "python"
    # the child runs `python -I`, which ignores PYTHONPATH: the wrapper drops -I so the fake package is the install
    wrapper.write_text(f'#!/bin/sh\n[ "$1" = -I ] && shift\nPYTHONPATH={tmp_path / "site"} exec {sys.executable} "$@"\n')
    wrapper.chmod(0o755)
    out = tmp_path / "out"
    assert ft.main(["coverage", "--python", str(wrapper), "--out", str(out), "--workers", "3",
                    "--unit-timeout", "15"]) == 0
    ledger, md = tmp_path / "coverage_ledger.json", tmp_path / "SUMMARY.md"
    assert ft.main(["coverage-summary", "--out", str(out), "--ledger", str(ledger), "--md", str(md)]) == 0
    rows = json.loads(ledger.read_text())["rows"]
    assert rows["Good"]["word"] == "ran", rows["Good"]
    assert {k: v["status"] for k, v in rows["Good"]["steps"].items()} == {
        "construct": "ran", "fit": "ran", "predict": "ran", "transform": "ran", "inverse_transform": "ran",
        "save": "ran", "load": "ran"}
    assert rows["Refuser"]["word"] == "refused" and "no CPU route" in rows["Refuser"]["steps"]["fit"]["why"]
    crash = rows["Crasher"]
    assert crash["word"] == "crashed" and crash["runs"] == 2
    assert crash["steps"]["boom"]["status"] == "crashed"
    assert crash["steps"]["later"]["status"] == "ran"  # the method after the crash still ran
    assert crash["steps"]["predict"]["status"] == "refused"
    assert rows["NeedsThing"]["word"] == "not_covered" and "thing" in rows["NeedsThing"]["steps"]["construct"]["why"]
    assert rows["Sleeper"]["steps"]["wait"] == dict(status="crashed", why="timed out", seconds=None)
    assert rows["CONST"]["kind"] == "int"
    text = md.read_text()
    assert "Crasher.boom: **crashed**" in text and "Refuser.fit: **refused**: RuntimeError" in text
    assert "Public names listed but not probed: 1 functions/submodules" in text and "CONST" in text
    assert "| **total** | **5** |" in text


def test_coverage_class_word():
    w = ft.coverage_class_word
    assert w(dict(steps={})) == "crashed"
    assert w(dict(steps={"construct": dict(status="ran"), "fit": dict(status="ran"), "predict": dict(status="refused")})) == "partial"
    assert w(dict(steps={"construct": dict(status="ran"), "fit": dict(status="refused")})) == "refused"
    assert w(dict(steps={"construct": dict(status="not_covered")})) == "not_covered"
    assert w(dict(steps={"construct": dict(status="ran"), "p": dict(status="not_covered")})) == "ran"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
