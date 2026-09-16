# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Exercise interruption/resume and reporting with fake fits, no native bindings."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import identity_break as ib
from identity_iterate import command
from identity_timing import summarize


@pytest.fixture
def run_fixture(monkeypatch, tmp_path):
    package = tmp_path / "mojolearn"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture package\n")
    ml = ModuleType("mojolearn")
    ml.__file__ = str(package / "__init__.py")
    ml.__version__ = "test"
    ml.numeric_mode = lambda: "identical"
    ml.vendor = lambda: "test-device"
    monkeypatch.setitem(sys.modules, "mojolearn", ml)
    verify = ModuleType("mojolearn._verify")
    verify.binding_artifacts = lambda: []
    monkeypatch.setitem(sys.modules, "mojolearn._verify", verify)
    monkeypatch.setenv("MOJOLEARN_NUMERIC_MODE", "identical")
    monkeypatch.setattr(ib, "commit_witness", lambda: ("a" * 40, "fixture"))
    monkeypatch.setattr(ib, "refuse_routine_apple_column", lambda *a: "")
    monkeypatch.setattr(ib, "_par_devices", lambda: [0])
    monkeypatch.setattr(ib, "FIXTURES", ["base", "ties"])
    monkeypatch.setattr(ib, "fixture", lambda f: (np.ones((2, 2), dtype=np.float32), np.zeros(2), np.ones(2)))
    monkeypatch.setattr(ib, "heldout", lambda f: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(ib, "BATCH", {"tiny": None})
    monkeypatch.setattr(ib, "RLPAIR", {})
    state = dict(calls=0, crash_at=None, refuse=False)
    def fit(*args):
        state["calls"] += 1
        if state["calls"] == state["crash_at"]:
            raise KeyboardInterrupt
        if state["refuse"]:
            raise ValueError("deliberate refusal")
        return ib._fit(dict(value="a" * 16))
    monkeypatch.setattr(ib, "LANES", {"tiny": fit})
    args = argparse.Namespace(json=str(tmp_path / "record.json"), resume=False, lanes="tiny", skip="",
                              fixtures="base,ties", repeats=2, vendor="fixture-cpu", allow_fast=False,
                              batch_alone=1, no_batch=True, no_rlpair=True, verbose=False,
                              fail_on_refused=True)
    for _, _, flag, _ in ib.EXTRA_PARTS.values():
        setattr(args, flag, False)
    return args, state, package


def test_interruption_keeps_completed_fixture_and_resume_skips_it(run_fixture, capsys):
    args, state, package = run_fixture
    state["crash_at"] = 3
    with pytest.raises(KeyboardInterrupt):
        ib._run_reference(args)
    prior = json.loads(Path(args.json).read_text())
    assert list(prior["cells"]) == ["tiny/base"]
    assert prior["complete"] is False
    assert "repeat=1/train" in prior["cells"]["tiny/base"]["timing"]["stages"]
    args.resume = True
    state["crash_at"] = None
    assert ib._run_reference(args) == 0
    final = json.loads(Path(args.json).read_text())
    assert state["calls"] == 5  # first cell's two fits were not repeated
    assert final["complete"] is True and len(final["cells"]) == 2
    assert prior["cells"]["tiny/base"] == final["cells"]["tiny/base"]
    output = capsys.readouterr().out
    assert "# START tiny/base repeat=1/train" in output
    assert "# REUSED tiny/base" in output


@pytest.mark.parametrize("change", ["binary", "python", "protocol", "env"])
def test_resume_rejects_changed_execution(run_fixture, monkeypatch, change):
    args, state, package = run_fixture
    assert ib._run_reference(args) == 0
    args.resume = True
    if change == "binary":
        (package / "binding.so").write_bytes(b"new native bytes")
    elif change == "python":
        (package / "__init__.py").write_text("changed = True\n")
    elif change == "protocol":
        args.repeats = 3
    else:
        monkeypatch.setenv("MOJOLEARN_CPU_THREADS", "99")
    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        ib._run_reference(args)
    assert state["calls"] == 4


def test_refused_cell_is_not_success(run_fixture):
    args, state, _ = run_fixture
    state["refuse"] = True
    assert ib._run_reference(args) == 1
    assert all(c["verdict"] == "REFUSED" for c in json.loads(Path(args.json).read_text())["cells"].values())


def test_atomic_checkpoint_preserves_old_record_on_write_error(tmp_path, monkeypatch):
    p = tmp_path / "out.json"
    ib.atomic_json(p, {"old": 1})
    monkeypatch.setattr(ib.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        ib.atomic_json(p, {"new": 2})
    assert json.loads(p.read_text()) == {"old": 1}
    assert list(tmp_path.iterdir()) == [p]


def test_iteration_keeps_two_fits_and_all_default_probes(tmp_path):
    cmd = command("python", "mamba1", "base", tmp_path / "record.json", 30, "metal", False)
    assert cmd[cmd.index("--repeats") + 1] == "2"
    assert "--no-batch" not in cmd and "--no-rlpair" not in cmd
    assert "--fail-on-refused" in cmd
    assert cmd[cmd.index("--timeout") + 1] == "30"


def test_timing_report_does_not_count_resume_twice(run_fixture):
    args, _, _ = run_fixture
    assert ib._run_reference(args) == 0
    report = summarize([args.json, args.json])
    assert report["timed_cells"] == 2
    assert report["stage_seconds"]["train"] >= 0


def test_merge_ignores_timing_but_rejects_changed_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "LANES", {"tiny": None})
    record = dict(vendor="fixture-cpu", commit="a" * 40, mode="identical", complete=True,
                  fixtures={"base": {}}, package={"bindings": [{"module": "native", "sha256": "same"}]},
                  cells={"tiny/base": dict(verdict="STABLE", hashes=["abc", "abc"], timing={"total_seconds": 1})})
    a, b, out = [tmp_path / name for name in ("a.json", "b.json", "merged.json")]
    a.write_text(json.dumps(record))
    record["cells"]["tiny/base"]["timing"]["total_seconds"] = 2
    b.write_text(json.dumps(record))
    assert ib.merge([str(a), str(b)], str(out)) == 0
    record["cells"]["tiny/base"]["hashes"] = ["wrong", "wrong"]
    b.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="different contents"):
        ib.merge([str(a), str(b)], str(out))


def test_resume_rejects_changed_external_binding(run_fixture, monkeypatch, tmp_path):
    args, state, _ = run_fixture
    host = tmp_path / "host"
    host.mkdir()
    binary = host / "oracle.so"
    binary.write_bytes(b"production")
    monkeypatch.setenv("MOJOLEARN_HOST_DIR", str(host))
    assert ib._run_reference(args) == 0
    args.resume = True
    binary.write_bytes(b"sabotage")
    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        ib._run_reference(args)


def test_packaged_harness_import_has_no_tools_dependency(tmp_path):
    import subprocess
    copy = tmp_path / "_identity_break.py"
    copy.write_text(Path(ib.__file__).read_text())
    p = subprocess.run([sys.executable, "-c", "import _identity_break"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_chunking_does_not_bypass_release_guard(monkeypatch, tmp_path):
    import identity_iterate as runner
    import lane_applicability
    lanes = [n for n, scope in lane_applicability.scopes().items()
             if scope.applicable("apple-metal")[0]][:30]
    assert len(lanes) == 30
    selection = dict(lanes=lanes, fallback=False, reasons={})
    monkeypatch.setattr(runner, "plan", lambda *args: selection)
    seen = []
    def guard(lanes, host):
        seen.append(lanes)
        return "release record required"
    monkeypatch.setattr(ib, "refuse_routine_apple_column", guard)
    monkeypatch.setattr(runner, "run_job", lambda *args: pytest.fail("must not start a job"))
    with pytest.raises(SystemExit):
        runner.main(["--mode", "metal", "--out", str(tmp_path)])
    assert seen == [selection["lanes"]]


def test_resume_signature_does_not_publish_environment_secrets(run_fixture, monkeypatch):
    args, _, _ = run_fixture
    monkeypatch.setenv("MOJOLEARN_PRIVATE_TOKEN", "secret-that-must-not-be-recorded")
    assert ib._run_reference(args) == 0
    assert "secret-that-must-not-be-recorded" not in Path(args.json).read_text()


def test_cpu_requirement_refuses_gpu_before_any_fit(run_fixture):
    args, state, _ = run_fixture
    args.require_cpu = True
    with pytest.raises(SystemExit, match="--require-cpu loaded a GPU"):
        ib._run_reference(args)
    assert state["calls"] == 0
