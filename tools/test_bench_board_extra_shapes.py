# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/bench_board_extra.py: every arm of the synthetic extra lanes records the input arrays it
times, so bench_board_evidence.measured_shape() finds a measured input shape (the 0.8.49 AMD board
stopped at algos/clip-grad-norm on "incomplete worker evidence"). CPU only: no GPU, no mojolearn,
no torch (a stand-in module names the torch calls the arm builder makes)."""
import importlib.util
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location("t_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("bench_board_algos")
E = _load("bench_board_evidence")
X = _load("bench_board_extra")
BP = A._BP()


class _T:
    """A host tensor stand-in: the few methods the clip arm builder calls."""

    def __init__(self, a):
        self.a = np.array(a)

    def to(self, dev):
        return self

    def detach(self):
        return self

    def clone(self):
        return _T(self.a.copy())


def _fake_torch():
    def clip_grad_norm_(params, max_norm, norm_type=2.0, error_if_nonfinite=False):
        return 0.0
    clip_grad_norm_.__module__ = "torch.nn.utils.clip_grad"
    nn = types.SimpleNamespace(Parameter=lambda t: types.SimpleNamespace(data=t, grad=None),
                               utils=types.SimpleNamespace(clip_grad_norm_=clip_grad_norm_))
    return types.SimpleNamespace(
        __version__="0.0-test", Tensor=_T, nn=nn, manual_seed=lambda s: None, compile=lambda f: f,
        zeros=lambda n, device=None: _T(np.zeros(n, np.float32)), from_numpy=_T,
        backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace())),
        no_grad=lambda: None)


def _globals(monkeypatch):
    G = dict(vars(A))
    G["_ours_class"] = lambda lane: ("clip_grad_norm_", lambda *a, **k: 0.0)
    G["_ours_info"] = lambda lane: {}
    G["_torch_device"] = lambda torch: "cpu"
    G["_torch_sync"] = lambda torch, dev: None
    G["_staging"] = lambda: None
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    return G


def _cell(arm, runner):
    _lib, _source, raw = BP.read_params(runner.record)
    return dict(arm=arm, params={"raw_params": raw, "implementation": runner.record.get("implementation")})


def test_clip_grad_norm_every_arm_records_its_input_arrays(monkeypatch):
    G = _globals(monkeypatch)
    lane = "clip-grad-norm"
    arms = ("ours",) + X.opponents("amd", lane, A.LANES[lane], True)
    assert arms == ("ours", "torch-eager-fp32", "torch-compile-fp32")
    want = A._input_array_records([("grad%d" % i, g) for i, g in enumerate(X._grads(np, A.SEED))])
    assert [a["name"] for a in want] == ["grad%d" % i for i in range(X.GRAD_TENSORS)]
    for arm in arms:
        runner = X.build(lane, arm, {}, G)
        cell = _cell(arm, runner)
        # no cell shape and no race inputs: the record alone carries the measured shape
        assert E.measured_shape(cell, {}) == {"input_arrays": want}, arm
        assert all(a["shape"] == [X.GRAD_SIZE] and a["dtype"] == "float32" for a in want)


def test_tensor_lanes_without_staged_arrays_record_inputs_on_every_arm():
    G = dict(vars(A))
    G["_staging"] = lambda: None
    for lane in ("lr-step", "jl-min-dim"):
        recs = X.input_arrays(lane, {}, G)
        cell = dict(params={"raw_params": {"input_arrays": recs}})
        assert E.measured_shape(cell, {}) == {"input_arrays": recs}, lane
    docs = X._text_arrays(np, ["ab", "cé"])
    assert [n for n, _ in docs] == ["docs_utf8", "doc_bytes"]
    assert docs[1][1].tolist() == [2, 3]
