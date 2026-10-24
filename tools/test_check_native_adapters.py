# SPDX-License-Identifier: Apache-2.0
"""The adapter checkers' comparison plumbing, pure Python: the device-type comparison of
tools/check_native_adapters.py (fake torch devices), and the MAX/mojolearn two-process split of
tools/check_max_native_adapters.py and tools/check_max_training_adapters.py (subprocess.run and mojolearn faked;
no torch, MAX, GPU or wheel)."""
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


CNA = load("check_native_adapters")
MAXN = load("check_max_native_adapters")
MAXT = load("check_max_training_adapters")


class FakeDevice:
    """torch.device's shape: .type and .index (torch reports an MPS tensor on index 0)."""
    def __init__(self, type, index=None):
        self.type, self.index = type, index

    def __eq__(self, other):  # torch.device semantics: mps:0 != mps
        return isinstance(other, FakeDevice) and (self.type, self.index) == (other.type, other.index)

    __hash__ = None


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.device = lambda spec, index=None: (FakeDevice(*CNA.device_parts(spec)) if index is None
                                            else FakeDevice(spec, index))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# ------------------------------------------------------------------ device comparison
def test_mps_tensor_matches_the_requested_mps_string(fake_torch):
    actual = FakeDevice("mps", 0)
    assert actual != fake_torch.device("mps")  # the 0.8.56 Apple gate's failure, plain ==
    assert CNA.same_device(actual, "mps")
    assert CNA.same_device(actual, fake_torch.device("mps"))
    assert CNA.same_device(actual, "mps:0")


def test_device_type_and_named_index_must_agree(fake_torch):
    assert CNA.same_device(FakeDevice("cuda", 0), "cuda:0")
    assert CNA.same_device(FakeDevice("cuda", 0), fake_torch.device("cuda"))
    assert not CNA.same_device(FakeDevice("cuda", 1), "cuda:0")
    assert not CNA.same_device(FakeDevice("cpu"), "mps")
    assert not CNA.same_device(FakeDevice("mps", 0), "cuda:0")


def test_device_parts():
    assert CNA.device_parts("mps") == ("mps", None)
    assert CNA.device_parts("cuda:3") == ("cuda", 3)
    assert CNA.device_parts(FakeDevice("mps", 0)) == ("mps", 0)


def test_no_plain_device_equality_left():
    source = (HERE / "check_native_adapters.py").read_text()
    assert ".device == " not in source and "== device" not in source


def test_word_diff_reports_the_bytes():
    a = np.array([1, 2, 3, 4], np.int32)
    assert CNA.word_diff(a, a.copy()) is None
    b = a.copy()
    b[2] ^= 1
    msg = CNA.word_diff(a, b)
    assert "1 of 4 words differ" in msg and "[2] 00000003 != 00000002" in msg
    assert CNA.word_diff(a, a[:3]) == "4 words != 3 words"


# ------------------------------------------------------------------ fake mojolearn
class FakeSession:
    """mojolearn.ops.Session's surface used by the MAX checker; handles are the numpy arrays (test only)."""
    def __init__(self, numeric_mode):
        assert numeric_mode == "identical"

    tensor = staticmethod(lambda a: np.asarray(a))
    numpy = staticmethod(lambda h: h)
    relu = staticmethod(lambda a: np.maximum(a, np.float32(0)))
    matmul = staticmethod(lambda a, b: a @ b)
    full = staticmethod(lambda shape, v: np.full(shape, v, np.float32))
    scale = staticmethod(lambda a, s: a * np.float32(s))
    transpose = staticmethod(lambda a: a.T.copy())
    embedding = staticmethod(lambda a, ids: a[ids])
    take = staticmethod(lambda a, ids, axis: np.take(a, ids, axis=axis))
    argmax = staticmethod(lambda a, axis, int32: a.argmax(axis=axis).astype(np.int32))
    linear = staticmethod(lambda a, w, b: a @ w + b)
    add = staticmethod(lambda a, b: a + b)


@pytest.fixture
def fake_mojolearn(monkeypatch):
    ops = types.ModuleType("mojolearn.ops")
    ops.Session = FakeSession
    pkg = types.ModuleType("mojolearn")
    pkg.ops = ops
    monkeypatch.setitem(sys.modules, "mojolearn", pkg)
    monkeypatch.setitem(sys.modules, "mojolearn.ops", ops)


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


def native_child(corrupt=None, complete=True):
    """The MAX-native child's files: every output the Python API would give (one word flipped in CORRUPT)."""
    def write(side):
        refs = MAXN.api_references(FakeSession("identical"), *MAXN.fixtures())
        outputs = {}
        for name, ref in refs.items():
            value = np.ascontiguousarray(ref())
            if name == corrupt:
                value = value.copy()
                value.view(np.int32).flat[0] ^= 1
            np.save(side / (name + ".npy"), value)
            outputs[name] = name + ".npy"
        (side / "max-side.json").write_text(json.dumps(dict(
            complete=complete, checks=list(refs), failures=[] if complete else ["Traceback\nAssertionError: x"],
            outputs=outputs, max="26.5.0")))
    return write


def native_args(tmp_path):
    return ["--library", str(tmp_path / "ops.mojoc"), "--native-library", str(tmp_path / "lib.dylib"),
            "--out", str(tmp_path / "out")]


# ------------------------------------------------------------------ MAX native: the split
def test_max_native_runs_max_in_a_child_and_compares_in_the_parent(tmp_path, monkeypatch, fake_mojolearn):
    calls = []
    monkeypatch.setattr(MAXN.subprocess, "run", fake_run(calls, native_child()))
    had_max = "max" in sys.modules
    assert MAXN.main(native_args(tmp_path)) == 0
    assert ("max" in sys.modules) == had_max  # the parent never imports MAX
    (argv,) = calls
    assert argv[0] == sys.executable and argv[1].endswith("check_max_native_adapters.py") and argv[2] == "--max-side"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "out" / "max-side")
    report = json.loads((tmp_path / "out" / "max-results.json").read_text())
    assert report["complete"] and not report["failures"] and report["max_side"]["exit"] == 0
    assert {"python_api:" + n for n in MAXN.api_references(FakeSession("identical"), *MAXN.fixtures())} \
        <= set(report["checks"])


def test_max_native_reports_the_word_diff(tmp_path, monkeypatch, fake_mojolearn):
    monkeypatch.setattr(MAXN.subprocess, "run", fake_run([], native_child(corrupt="transpose_view")))
    assert MAXN.main(native_args(tmp_path)) == 1
    report = json.loads((tmp_path / "out" / "max-results.json").read_text())
    (failure,) = report["failures"]
    assert "transpose_view: MAX and Python API words differ: 1 of 6 words differ: [0]" in failure
    assert "AssertionError" in failure  # check_max_adapter_fault looks for it


def test_max_native_child_abort_fails(tmp_path, monkeypatch, fake_mojolearn):
    monkeypatch.setattr(MAXN.subprocess, "run", fake_run([], lambda side: None, exit_code=-6))
    assert MAXN.main(native_args(tmp_path)) == 1
    report = json.loads((tmp_path / "out" / "max-results.json").read_text())
    assert any("MAX side exit -6" in f and "child log line" in f for f in report["failures"])
    assert any("kept no output" in f for f in report["failures"])


def test_max_native_child_failure_is_carried(tmp_path, monkeypatch, fake_mojolearn):
    monkeypatch.setattr(MAXN.subprocess, "run", fake_run([], native_child(complete=False), exit_code=1))
    assert MAXN.main(native_args(tmp_path)) == 1
    report = json.loads((tmp_path / "out" / "max-results.json").read_text())
    assert any("AssertionError: x" in f for f in report["failures"])


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
