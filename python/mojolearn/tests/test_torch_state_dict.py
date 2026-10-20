# SPDX-License-Identifier: Apache-2.0
"""Weights from a real PyTorch ``state_dict``: Sequential.load_state_dict and
Session.load_weights given ``module.state_dict()`` of a torch.nn.Sequential
(Linear, LayerNorm, Linear) in float32, bfloat16 and float16.

tests/test_weight_import.py covers the same entries without torch (NumPy
mappings and safetensors files of known words); this file is the torch twin
of that surface. Checks: the imported parameters hold torch's words exactly
(the (out, in) linear weight transposed to (in, out); bfloat16 and float16
widen to float32 exactly). Torch's own forward on the same rows is printed
as a reference distance, never asserted: torch's bits are a reference, never
a gate. Prints TORCH_STATE_DICT_DIGEST lines for the orchestrator to diff
across CPU, Metal, NVIDIA and AMD.

Skipped loudly without torch (``no torch in this environment (<error>)``);
``MOJOLEARN_REQUIRE_TORCH=1`` (the box runs set it) makes that a failure.
"""
import hashlib
import os

import numpy as np
import pytest

try:
    import torch
    _TORCH_ERROR = None
except Exception as _exc:  # torch absent
    torch = None
    _TORCH_ERROR = "%s: %s" % (type(_exc).__name__, str(_exc)[:160])

#: MOJOLEARN_REQUIRE_TORCH=1 (set by the box runs): a missing torch fails these tests instead of skipping them
REQUIRE_TORCH = os.environ.get("MOJOLEARN_REQUIRE_TORCH") == "1"

D, H, O = 4, 8, 3
DTYPES = ["float32", "bfloat16", "float16"]


@pytest.fixture(autouse=True)
def _torch_and_binding():
    if torch is None:
        reason = "no torch in this environment (%s)" % _TORCH_ERROR
        if REQUIRE_TORCH:
            pytest.fail("MOJOLEARN_REQUIRE_TORCH=1: " + reason)
        pytest.skip(reason)
    from mojolearn import _backend
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        ok = hasattr(b, "model_import_params") and hasattr(b, "ops_load_weights")
    except Exception:
        ok = False
    if not ok:
        pytest.skip("the sequence binding has no weight import entries (rebuild it)")


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _module(dtype):
    torch.manual_seed(7)
    m = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.LayerNorm(H, eps=1e-5), torch.nn.Linear(H, O))
    with torch.no_grad():
        torch.nn.init.uniform_(m[1].weight, 0.5, 1.5)
        torch.nn.init.uniform_(m[1].bias, -0.2, 0.2)
    return m.to(getattr(torch, dtype)).eval()


def _f32(t):
    """A state_dict tensor as the float32 words it widens to (exact for the half types)."""
    return t.detach().float().numpy()


def _expected_params(state):
    # params_' layout: linear weight (in, out), its bias; layer norm weight, bias
    return np.concatenate([_f32(state["0.weight"]).T.ravel(), _f32(state["0.bias"]),
                           _f32(state["1.weight"]), _f32(state["1.bias"]),
                           _f32(state["2.weight"]).T.ravel(), _f32(state["2.bias"])]).astype(np.float32)


def _x():
    return np.random.default_rng(3).standard_normal((32, D)).astype(np.float32)


def _reference_line(name, ours, m):
    # torch's forward of the float32-widened module: a printed distance, never asserted
    with torch.no_grad():
        ref = m.float()(torch.from_numpy(_x())).numpy()
    d = np.abs(np.asarray(ours, dtype=np.float64) - ref.astype(np.float64))
    print(f"TORCH_STATE_DICT_REFERENCE {name} max_abs={float(d.max()):.3e} "
          f"max_rel={float((d / np.maximum(np.abs(ref), 1e-30)).max()):.3e}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_sequential_load_torch_state_dict(dtype):
    import mojolearn as ml
    m = _module(dtype)
    state = m.state_dict()
    model = ml.Sequential().linear(H).layer_norm().linear(O).load_state_dict(state)
    expected = _expected_params(state)
    assert model.params_.shape == expected.shape
    assert np.array_equal(_bits(model.params_), _bits(expected))
    again = ml.Sequential().linear(H).layer_norm().linear(O).load_state_dict(state, source="torch")
    assert np.array_equal(_bits(again.params_), _bits(model.params_))
    pred = np.asarray(model.predict(_x()))
    assert pred.shape == (32, O) and np.isfinite(pred).all()
    print(f"TORCH_STATE_DICT_DIGEST sequential_{dtype} {_digest(model.params_, pred)}")
    _reference_line(f"sequential_{dtype}", pred, m)


@pytest.mark.parametrize("dtype", DTYPES)
def test_session_load_torch_state_dict(dtype):
    from mojolearn import ops
    m = _module(dtype)
    state = m.state_dict()
    with ops.Session() as sess:
        w0 = sess.tensor(np.zeros((D, H), dtype=np.float32))
        b0 = sess.tensor(np.zeros((1, H), dtype=np.float32))
        g = sess.tensor(np.zeros((1, H), dtype=np.float32))
        beta = sess.tensor(np.zeros((1, H), dtype=np.float32))
        w2 = sess.tensor(np.zeros((H, O), dtype=np.float32))
        b2 = sess.tensor(np.zeros((1, O), dtype=np.float32))
        sess.load_weights(state, [(w0, "0.weight", "transpose"), (b0, "0.bias"), (g, "1.weight"),
                                  (beta, "1.bias"), (w2, "2.weight", "transpose"), (b2, "2.bias")])
        assert np.array_equal(_bits(sess.numpy(w0)), _bits(_f32(state["0.weight"]).T))
        assert np.array_equal(_bits(sess.numpy(g)), _bits(_f32(state["1.weight"])[None, :]))
        assert np.array_equal(_bits(sess.numpy(b2)), _bits(_f32(state["2.bias"])[None, :]))
        h = sess.layer_norm(sess.linear(sess.tensor(_x()), w0, b0), g, beta)
        out = sess.numpy(sess.linear(h, w2, b2))
    assert out.shape == (32, O) and np.isfinite(out).all()
    print(f"TORCH_STATE_DICT_DIGEST session_{dtype} {_digest(out)}")
    _reference_line(f"session_{dtype}", out, m)
