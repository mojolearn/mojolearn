# SPDX-License-Identifier: Apache-2.0
"""The MAX/mojolearn two-process split of tools/check_max_training_adapters.py (subprocess.run and the
fixtures faked; no torch, MAX, GPU or wheel).

The device-comparison helpers of tools/check_native_adapters.py and the per-op MAX checker
tools/check_max_native_adapters.py were removed by 5034860934 (the torch phase now checks the whole-model
torch.compile backend); their tests went with them. tools/test_adapter_wheel_smoke.py and
tools/test_adapter_surfaces.py assert the per-op checker stays gone."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAXT = load("check_max_training_adapters")


def fake_run(calls, write, exit_code=0):
    """subprocess.run stand-in: records argv, lets WRITE(side_dir) play the MAX child, returns EXIT_CODE."""
    def run(argv, stdout=None, stderr=None, **kw):
        calls.append(list(argv))
        side = Path(argv[argv.index("--out") + 1])
        side.mkdir(parents=True, exist_ok=True)
        write(side)
        if stdout is not None:
            stdout.write("child log line\n")
        return types.SimpleNamespace(returncode=exit_code)
    return run


# ------------------------------------------------------------------ MAX training: the split
API_EXPECTED = [np.arange(4, dtype=np.float32), np.zeros(4, np.float32), np.ones(4, np.float32),
                np.arange(24, dtype=np.int32), np.array([.5, .25], np.float32)]


@pytest.fixture
def fake_fixtures(monkeypatch):
    mod = types.ModuleType("native_training_fixtures")
    arrays = [np.ones((7, 3), np.float32), np.ones((7, 2), np.float32), np.zeros(4, np.float32)]
    mod.api_case = lambda: (arrays, [1, 3, 4], [0, 0, 2, 0, 1, 2, 3, 2], [np.float32(.03125), .9], API_EXPECTED)
    monkeypatch.setitem(sys.modules, "native_training_fixtures", mod)


def training_child(corrupt_slot=None):
    def write(side):
        spec = json.loads((side / "python_api-input.json").read_text())
        assert spec["arrays"] == 3 and spec["program"] == ["1", "3", "4"] and spec["scalars"] == ["0.03125", "0.9"]
        assert (side / "python_api-input.npz").exists()
        outs = [v.copy() for v in API_EXPECTED]
        if corrupt_slot is not None:
            outs[corrupt_slot].view(np.uint32).flat[1] ^= 1
        np.savez(side / "python_api.npz", **{f"output_{i}": v for i, v in enumerate(outs)})
        (side / "max-side.json").write_text(json.dumps(dict(complete=True, checks=["analytic_sgd: ok"], failures=[])))
    return write


def training_args(tmp_path):
    return ["--library", str(tmp_path / "ops.mojoc"), "--native-library", str(tmp_path / "lib.dylib"),
            "--torch-results", str(tmp_path / "tt"), "--out", str(tmp_path / "out"), "--smoke"]


def test_max_training_hands_the_api_inputs_to_a_max_child(tmp_path, monkeypatch, fake_fixtures):
    calls = []
    monkeypatch.setattr(MAXT.subprocess, "run", fake_run(calls, training_child()))
    assert MAXT.main(training_args(tmp_path)) == 0
    (argv,) = calls
    assert argv[0] == sys.executable and argv[2] == "--max-side" and argv[-1] == "--smoke"
    report = json.loads((tmp_path / "out" / "max-training-results.json").read_text())
    assert report["complete"] and any(c.startswith("python_api: Sequential.fit") for c in report["checks"])


def test_max_training_reports_the_slot_and_words(tmp_path, monkeypatch, fake_fixtures):
    monkeypatch.setattr(MAXT.subprocess, "run", fake_run([], training_child(corrupt_slot=3)))
    assert MAXT.main(training_args(tmp_path)) == 1
    report = json.loads((tmp_path / "out" / "max-training-results.json").read_text())
    (failure,) = report["failures"]
    assert "output 3: 1 of 24 words differ: [1]" in failure


def test_max_training_child_abort_fails(tmp_path, monkeypatch, fake_fixtures):
    monkeypatch.setattr(MAXT.subprocess, "run", fake_run([], lambda side: None, exit_code=-6))
    assert MAXT.main(training_args(tmp_path)) == 1
    report = json.loads((tmp_path / "out" / "max-training-results.json").read_text())
    assert any("MAX side exit -6" in f for f in report["failures"])
