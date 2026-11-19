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
    assert inputs["families"]["default"] == "0x7ffffff"
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
    names = [row[0] for row in module.FAMILIES]
    assert set(names) == set(module.ESTIMATOR_FAMILY)  # every estimator has a family label
    trial_list = ft.trials("0x7ffffff", "0x88")
    every = [u for k in range(3) for u in ft.units(names, trial_list, k, 3)]
    assert sorted(every) == sorted((t, m, n) for t, m in trial_list for n in names)
    assert ft.trials("0x7ffffff", "") == [("families", "0x7ffffff")]


def test_mask_range():
    assert ft.mask("0x7ffffff") == "0x7ffffff" and ft.mask("136") == "0x88"
    try:
        ft.mask("0x8000000")
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
    rows = [("families", "0x7ffffff", "PCA", "0"), ("families", "0x7ffffff", "Ridge", "1"),
            ("families", "0x7ffffff", "Lasso", "-11")]
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


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
