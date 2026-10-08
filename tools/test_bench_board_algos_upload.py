# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/bench_board_algos.py OUR separate upload clock (`_ours_upload_probe`).

The probe uploads OUR arm's declared host inputs once more after every clock of a
round has closed, records `upload_separate`, and the conductor puts
`upload_ms_separate` + the derived `fit_minus_upload_ms` in OUR span only. These
tests use a fake resident-array binding (no GPU, no Mojo build). They run under
pytest, or as `python3 tools/test_bench_board_algos_upload.py` (needs NumPy) where
pytest is not installed.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location("t_bba_upload", os.path.join(HERE, "bench_board_algos.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load()


class _FakeBinding:
    def __init__(self):
        self.calls = []

    def x_cnn_res_alloc(self, words):
        self.calls.append(("alloc", int(words)))
        return 4096

    def x_cnn_res_upload(self, h, addr, words):
        self.calls.append(("upload", int(h), int(words)))

    def x_cnn_res_free(self, h):
        self.calls.append(("free", int(h)))


def _fake_ml(binding):
    ml = types.ModuleType("mojolearn")
    ml._backend = types.SimpleNamespace(binding=lambda name, mode: binding if name == "_mojolearn_x_cnn" else None,
                                        default_mode=lambda: "identical")
    return ml


def _with_fake_ml(binding, fn):
    old = sys.modules.get("mojolearn")
    sys.modules["mojolearn"] = _fake_ml(binding)
    try:
        return fn()
    finally:
        if old is None:
            sys.modules.pop("mojolearn", None)
        else:
            sys.modules["mojolearn"] = old


def _runner(info=None, inputs=None):
    r = A.Runner(info if info is not None else {"library": "mojolearn", "input_home": "host"},
                 lambda: None, lambda: {})
    if inputs is not None:
        A._with_upload(r, inputs)
    return r


def test_probe_uploads_the_declared_inputs_as_float32_words():
    b = _FakeBinding()
    X = np.ones((10, 3), dtype=np.float64)              # float64 goes up as float32 words
    y = np.arange(10, dtype=np.int64)                   # integers go up as their own bytes
    rec = _with_fake_ml(b, lambda: A._ours_upload_probe(_runner(inputs=[("X", X), ("y", y), ("none", None)])))
    assert rec["scope"] == "host_to_device_upload_separate" and rec["outside_scored_clock"] is True
    assert rec["inputs"] == ["X", "y"]
    assert rec["bytes"] == 10 * 3 * 4 + 10 * 8
    assert rec["ms"] >= 0.0
    # one buffer (the largest input), one upload per array, freed after
    assert b.calls == [("alloc", 30), ("upload", 4096, 30), ("upload", 4096, 20), ("free", 4096)]


def test_probe_stands_down_without_inputs_or_with_device_inputs():
    b = _FakeBinding()
    dev = _with_fake_ml(b, lambda: A._ours_upload_probe(
        _runner(info={"library": "mojolearn", "input_home": "device"}, inputs=[("X", np.ones(4))])))
    assert "input_home=device" in dev["unavailable"] and "ms" not in dev
    none = _with_fake_ml(b, lambda: A._ours_upload_probe(_runner()))
    assert "declared no host inputs" in none["unavailable"]
    assert b.calls == []


def test_ours_span_carries_the_separate_clock_and_opponents_do_not():
    ms = [10.0, 12.0, 11.0]
    ours = {"info": {"library": "mojolearn", "input_home": "host"}, "ms": list(ms), "median_ms": 11.0,
            "upload_separate": [dict(ms=9.0, bytes=400, inputs=["X"], round=0, warmup=True),
                                dict(ms=2.0, bytes=400, inputs=["X"], round=1, warmup=False),
                                dict(ms=4.0, bytes=400, inputs=["X"], round=2, warmup=False),
                                dict(ms=3.0, bytes=400, inputs=["X"], round=3, warmup=False)]}
    span = A.arm_span("eigh", "ours", ours, True)
    assert span["upload_ms_separate"] == 3.0                     # the scored rounds only
    assert span["fit_minus_upload_ms"] == 8.0
    assert span["upload_bytes_separate"] == 400 and span["upload_inputs_separate"] == ["X"]
    assert span["upload_scope_separate"] == "host_to_device_upload_separate"
    assert span["input_home"] == "host"
    # the stored clock is untouched
    assert ours["ms"] == ms and ours["median_ms"] == 11.0
    torch = {"info": {"library": "torch", "input_home": "device"}, "ms": [5.0], "median_ms": 5.0}
    tspan = A.arm_span("eigh", "torch-eager-fp32", torch, True)
    assert not any("upload_ms_separate" in k or "fit_minus" in k or k.endswith("_separate") for k in tspan)
    # a probe that could not run says why and derives nothing
    miss = {"info": {"library": "mojolearn"}, "ms": [1.0], "median_ms": 1.0,
            "upload_separate": [{"unavailable": "no resident-array binding", "round": 1, "warmup": False}]}
    mspan = A.arm_span("eigh", "ours-fast", miss, True)
    assert mspan["upload_ms_separate"] is None and mspan["fit_minus_upload_ms"] is None
    assert mspan["upload_separate_unavailable"] == "no resident-array binding"


def test_board_audit_derives_our_kernel_clock_from_the_span():
    spec = importlib.util.spec_from_file_location("t_bca", os.path.join(HERE, "board_clock_audit.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    ours = {"info": {"library": "mojolearn", "input_home": "host"}, "ms": [11.0], "median_ms": 11.0,
            "upload_separate": [dict(ms=3.0, bytes=400, inputs=["X"], round=1, warmup=False)]}
    cell = {"library": "mojolearn", "family": "algos", "phase": "fit", "status": "ok", "median_ms": 11.0,
            "device": "gpu", "comparability": {"span": A.arm_span("eigh", "ours", ours, True)}}
    c = audit.cell_clock(cell)
    assert c["whole_ms"] == 11.0 and c["kernel_ms"] == 8.0 and c["copy_source"] == "upload_ms_separate"


_WORKER_DRIVER = textwrap.dedent(r'''
    import importlib.util, io, json, os, sys, time, types
    import numpy as np
    here, arm = sys.argv[1], sys.argv[2]
    sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location("bba", os.path.join(here, "bench_board_algos.py"))
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

    class B:
        def x_cnn_res_alloc(self, w): return 1
        def x_cnn_res_upload(self, h, a, w): time.sleep(0.2)
        def x_cnn_res_free(self, h): pass
    ml = types.ModuleType("mojolearn")
    ml._backend = types.SimpleNamespace(binding=lambda n, m: B(), default_mode=lambda: "identical")
    sys.modules["mojolearn"] = ml

    class Mem:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): return {}
    A._MODS["bench_board_probe"] = types.SimpleNamespace(MemProbe=Mem, library_identity=lambda info: {})
    A._load_block = lambda lane, ds, data: ({}, {})
    A.lane_arrays = lambda lane, B_: {"X": np.ones((64, 4), dtype=np.float32)}
    lib = "mojolearn" if arm in A.OURS_ARMS else "torch"

    def build(lane, a, D):
        r = A.Runner({"library": lib, "input_home": "host" if lib == "mojolearn" else "device"},
                     lambda: None, lambda: {"y": np.zeros(2)})
        return A._with_upload(r, [("X", D["X"])]) if a in A.OURS_ARMS else r
    A.build = build
    args = types.SimpleNamespace(arm=arm, lane="eigh", dataset="synthetic", data="-", neural_ab_config=None)
    sys.exit(A.worker(args))
''')


def _worker_rounds(arm, tmpdir):
    drv = os.path.join(tmpdir, "drv.py")
    with open(drv, "w") as fh:
        fh.write(_WORKER_DRIVER)
    env = dict(os.environ)
    env.pop("MOJOLEARN_BENCH_WHOLE_OPERATION", None)
    p = subprocess.run([sys.executable, drv, HERE, arm], input="round 0\nround 1\nquit\n",
                       capture_output=True, text=True, timeout=120, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    msgs = [json.loads(ln) for ln in p.stdout.splitlines() if ln.strip()]
    return [m for m in msgs if m.get("event") == "round"]


def test_worker_probe_runs_after_the_clocks_for_ours_only(tmp_path=None):
    import tempfile
    d = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp(prefix="bba-upload-")
    rounds = _worker_rounds("ours", d)
    assert len(rounds) == 2
    for m in rounds:
        up = m["upload_separate"]
        assert up["ms"] >= 150.0 and up["bytes"] == 64 * 4 * 4 and up["inputs"] == ["X"]
        # the probe's 200 ms upload is in none of the round's clocks
        assert m["ms"] < 100.0 and m["operation"]["ms"] < 100.0
    for m in _worker_rounds("torch-eager-fp32", d):
        assert m["upload_separate"] is None


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(exc)[:500])
    sys.exit(1 if fails else 0)
