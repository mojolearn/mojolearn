# SPDX-License-Identifier: Apache-2.0
"""The torch.export reader (docs/plans/MODEL_IMPORT.md lane 4).

Skipped loudly unless torch imports: each torch test skips with
``no torch in this environment (<the import error>)``, and with
``MOJOLEARN_REQUIRE_TORCH=1`` (the box runs set it) the skip is a failure
instead. The package itself never imports torch; the first test checks that
in a fresh interpreter. Torch's bits are a reference here, never a gate. The shared fixtures of
tests/_onnx_fixture.py are rebuilt as ``nn.Module``s holding the same
weights and exported with ``torch.export.export``:

* reader checks (torch only, no binding): every fixture reads into an
  import graph whose shapes resolve to the fixture's output shapes and whose
  weights are the fixture's bytes; bfloat16 and float16 weights arrive as
  their bit patterns; a dynamic batch stays an open axis that ``shapes``
  fixes, through the folded ``sym_size`` subgraph; unsupported operations
  and training-mode exports are refused by name, all in one error; the
  source dispatch sends a path to the ONNX reader and a program to this one.
* import checks (need the lowering and the sequence binding): each fixture
  imported from torch.export runs equal word for word to the same fixture
  imported from its ONNX file (which the lowering's own tests hold equal to
  the eager session composition), a second run gives the same words, and an
  ``IMPORT_DIGEST torch_<fixture> <digest>`` line is printed for the
  orchestrator to diff across CPU, Metal, NVIDIA and AMD. The distance to
  torch's own output is printed as a reference, never asserted.
"""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from mojolearn import importers
from mojolearn.importers import UnsupportedModel
from mojolearn.importers._graph import GraphIR, resolve
from mojolearn.tests import _onnx_fixture as fx

try:
    import torch
    import torch.export  # noqa: F401
    _TORCH_ERROR = None
except Exception as _exc:  # torch absent or too old for torch.export
    torch = None
    _TORCH_ERROR = "%s: %s" % (type(_exc).__name__, str(_exc)[:160])

#: MOJOLEARN_REQUIRE_TORCH=1 (set by the box runs): a missing torch fails the torch tests instead of skipping them
REQUIRE_TORCH = os.environ.get("MOJOLEARN_REQUIRE_TORCH") == "1"


@pytest.fixture
def _torch_required():
    if torch is None:
        reason = "no torch in this environment (%s)" % _TORCH_ERROR
        if REQUIRE_TORCH:
            pytest.fail("MOJOLEARN_REQUIRE_TORCH=1: " + reason)
        pytest.skip(reason)


needs_torch = pytest.mark.usefixtures("_torch_required")

FIXTURES = ["linear_layernorm_linear", "conv_relu_maxpool_flatten_gemm", "attention_two_head_causal",
            "batchnorm_relu"]
BATCH = 3


def test_package_never_imports_torch():
    # The child imports the same package this test imported (its parent directory goes first on PYTHONPATH),
    # and exits 3 when torch was loaded, so a child that cannot import the package (exit 1) is not read as torch.
    import mojolearn
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(mojolearn.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = ("import sys, mojolearn.importers, mojolearn.importers.torch_export; "
            "sys.exit(3 if 'torch' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode != 3, "importing mojolearn.importers imported torch"
    assert r.returncode == 0, "the child could not import the package: " + r.stderr[-400:]


# ------------------------------------------------------------ the modules --

def _t(init):
    """A fixture initializer as a torch tensor (the same bytes)."""
    dt = {"f32": torch.float32, "i64": torch.int64}[init.dtype]
    return torch.frombuffer(bytearray(init.data), dtype=dt).reshape(init.shape).clone()


def _linear(w, b):
    m = torch.nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        m.weight.copy_(w)
        m.bias.copy_(b)
    return m


def _module(name, f):
    I = {k: _t(v) for k, v in f.initializers.items()}
    if name == "linear_layernorm_linear":
        ln = torch.nn.LayerNorm(5, eps=1e-5)
        with torch.no_grad():
            ln.weight.copy_(I["ln_scale"])
            ln.bias.copy_(I["ln_bias"])
        return torch.nn.Sequential(_linear(I["w1"], I["b1"]), ln, _linear(I["w2"], I["b2"]))
    if name == "conv_relu_maxpool_flatten_gemm":
        conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        with torch.no_grad():
            conv.weight.copy_(I["conv_w"])
            conv.bias.copy_(I["conv_b"])
        return torch.nn.Sequential(conv, torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2), torch.nn.Flatten(),
                                   _linear(I["fc_w"], I["fc_b"]))
    if name == "attention_two_head_causal":
        return _Attention(I)
    if name == "batchnorm_relu":
        bn = torch.nn.BatchNorm2d(4, eps=1e-5)
        with torch.no_grad():
            bn.weight.copy_(I["bn_scale"])
            bn.bias.copy_(I["bn_bias"])
            bn.running_mean.copy_(I["bn_mean"])
            bn.running_var.copy_(I["bn_var"])
        return torch.nn.Sequential(bn, torch.nn.ReLU())
    raise KeyError(name)


if torch is not None:
    class _Attention(torch.nn.Module):
        """The fixture's two-head causal attention block: [in, out] projection
        weights used as x @ w, a [1, 1, t, t] additive mask buffer."""

        def __init__(self, I, heads=2):
            super().__init__()
            self.heads = heads
            for k in ("wq", "wk", "wv", "wo"):
                setattr(self, k, torch.nn.Parameter(I[k]))
            self.register_buffer("mask", I["mask"])
            self.scale = float(np.frombuffer(bytes(_scale_bytes(I)), dtype=np.float32)[0])

        def forward(self, x):
            b, t, d = x.shape
            dh = d // self.heads
            q = (x @ self.wq).reshape(b, t, self.heads, dh).permute(0, 2, 1, 3)
            k = (x @ self.wk).reshape(b, t, self.heads, dh).permute(0, 2, 3, 1)
            v = (x @ self.wv).reshape(b, t, self.heads, dh).permute(0, 2, 1, 3)
            p = torch.softmax(q @ k * self.scale + self.mask, dim=-1)
            o = (p @ v).permute(0, 2, 1, 3).reshape(b, t, d)
            return o @ self.wo


def _scale_bytes(I):
    return I["scale"].numpy().tobytes()


def _input_shape(f, batch):
    (_, _, shp), = f.inputs
    return tuple(batch if isinstance(d, str) else d for d in shp)


def _x(f, batch, seed=7):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=_input_shape(f, batch)).astype(np.float32)


def _export(name, f, batch=BATCH):
    m = _module(name, f).eval()
    return torch.export.export(m, (torch.from_numpy(_x(f, batch)),)), m


# ------------------------------------------------------- reader checks ----

@needs_torch
@pytest.mark.parametrize("name", FIXTURES)
def test_read_fixture(tmp_path, name):
    from mojolearn.importers import torch_export
    f = fx.write_fixture(name, tmp_path)
    ep, _ = _export(name, f)
    g = torch_export.read(ep)
    assert isinstance(g, GraphIR) and g.source == "torch.export"
    assert all(n.domain == "" and n.name for n in g.nodes)
    (iname, idt, ishape), = g.inputs
    assert idt == "f32" and ishape == _input_shape(f, BATCH)
    r = resolve(g, {iname: ishape})
    assert not r.problems and not r.no_rule, (r.problems, r.no_rule)
    (oname, _, _), = g.outputs
    assert r.shapes[oname] == tuple(f.output_shape(BATCH).values())[0]
    # the fixture's float weights are among the initializers, byte for byte
    # (num_batches_tracked and other unused buffers are dropped)
    got = {bytes(t.data) for t in g.initializers.values() if t.dtype == "f32"}
    for k, t in f.initializers.items():
        if t.dtype == "f32" and k != "scale":
            assert t.data in got, k


@needs_torch
@pytest.mark.parametrize("dtype,code", [("bfloat16", "bf16"), ("float16", "f16")])
def test_half_weights_are_bit_patterns(dtype, code):
    from mojolearn.importers import torch_export
    tdt = getattr(torch, dtype)
    m = torch.nn.Linear(4, 3).to(tdt).eval()
    ep = torch.export.export(m, (torch.zeros(2, 4, dtype=tdt),))
    g = torch_export.read(ep)
    want = {m.weight.detach().view(torch.int16).numpy().tobytes(), m.bias.detach().view(torch.int16).numpy().tobytes()}
    got = {bytes(t.data) for t in g.initializers.values() if t.dtype == code}
    assert want <= got
    assert g.inputs[0][1] == code


@needs_torch
def test_dynamic_batch_resolves_through_sym_size():
    from mojolearn.importers import torch_export

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(6, 4)

        def forward(self, x):
            return torch.relu(self.fc(x.reshape(x.shape[0], -1)))

    batch = torch.export.Dim("batch", min=2, max=64)
    ep = torch.export.export(M().eval(), (torch.zeros(5, 2, 3),), dynamic_shapes={"x": {0: batch}})
    g = torch_export.read(ep)
    (iname, _, dims), = g.inputs
    assert isinstance(dims[0], str) and dims[1:] == (2, 3)
    r = resolve(g)
    assert any(p[0] == iname for p in r.problems)  # the open axis is named
    r = resolve(g, {iname: (7, 2, 3)})
    assert not r.problems and not r.no_rule, (r.problems, r.no_rule)
    assert r.shapes[g.outputs[0][0]] == (7, 4)


@needs_torch
def test_unsupported_ops_listed_together():
    from mojolearn.importers import torch_export

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.ConvTranspose2d(2, 2, 2, stride=2)

        def forward(self, x):
            return torch.cumprod(self.up(x), dim=1)

    ep = torch.export.export(M().eval(), (torch.zeros(1, 2, 3, 3),))
    with pytest.raises(UnsupportedModel) as e:
        torch_export.read(ep)
    reasons = [(op, why) for _, op, why in e.value.problems]
    assert any(op == "aten.cumprod" for op, _ in reasons), reasons
    assert any("transposed convolution" in why for _, why in reasons), reasons
    assert "cannot be imported" in str(e.value) and isinstance(e.value, ValueError)


@needs_torch
def test_training_mode_export_refused():
    from mojolearn.importers import torch_export
    m = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.ReLU()).train()
    ep = torch.export.export(m, (torch.ones(2, 3, 4, 4),))
    with pytest.raises(UnsupportedModel) as e:
        torch_export.read(ep)
    assert "eval()" in str(e.value)


@needs_torch
def test_source_dispatch(tmp_path):
    f = fx.write_fixture("linear_layernorm_linear", tmp_path)
    assert importers._read_source(f.path).source == f.path
    ep, m = _export("linear_layernorm_linear", f)
    assert importers._read_source(ep).source == "torch.export"
    with pytest.raises(TypeError, match="torch.export.export"):
        importers._read_source(m)
    with pytest.raises(TypeError):
        importers._read_source(42)


# ------------------------------------------------------- import checks ----

def _can_import():
    try:
        from mojolearn import _backend
        from mojolearn.importers._program import import_model  # noqa: F401
        b = _backend.binding("_mojolearn_x_sequence")
        return hasattr(b, "ops_capture") and hasattr(b, "ops_program_run")
    except Exception:
        return False


def _outs(r):
    return list(r) if isinstance(r, (list, tuple)) else [r]


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


@needs_torch
@pytest.mark.parametrize("name", FIXTURES)
def test_import_equals_onnx_twin(tmp_path, name):
    # the binding skip sits in the body, after the torch fixture, so MOJOLEARN_REQUIRE_TORCH=1 fails first
    if not _can_import():
        pytest.skip("the import lowering or the sequence binding's Program entries are not available")
    import mojolearn
    f = fx.write_fixture(name, tmp_path)
    ep, m = _export(name, f)
    x = _x(f, BATCH, seed=11)
    (iname, _, _), = f.inputs
    from_torch = mojolearn.importers.import_model(ep)
    from_onnx = mojolearn.importers.import_model(f.path, shapes={iname: x.shape})
    a = _outs(from_torch.run(x))
    a2 = _outs(from_torch.run(x))
    b = _outs(from_onnx.run(x))
    assert len(a) == len(b) == 1
    for u, v in zip(a, a2):
        assert np.array_equal(_words(u), _words(v)), "a second run differs"
    for u, v in zip(a, b):
        assert u.shape == v.shape
        assert np.array_equal(_words(u), _words(v)), f"{name}: the torch.export import differs from the ONNX import"
    print(f"IMPORT_DIGEST torch_{name} {_digest(a)}")
    # torch's own output: a reference distance, printed, never asserted
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    d = np.abs(a[0].astype(np.float64) - ref.astype(np.float64))
    print(f"IMPORT_TORCH_REFERENCE torch_{name} max_abs={float(d.max()):.3e} "
          f"max_rel={float((d / np.maximum(np.abs(ref), 1e-30)).max()):.3e}")
