# SPDX-License-Identifier: Apache-2.0
"""The torch.compile backend "mojolearn" (mojolearn/adapters/_torch_compile.py).

The glue tests run without torch: a stand-in ``torch`` in ``sys.modules``
provides tensors, modules, the ONNX export and TorchDynamo's registry, and
the file import is replaced by a recorder returning a stand-in program. They
check the registration, the export of the captured graph, that each call's
outputs carry the program's result bytes, the refusals and the cache. The
last test needs PyTorch and a binding with model import.
"""
from __future__ import annotations

import ast
import contextlib
import pathlib
import sys
import types

import numpy as np
import pytest

from mojolearn.adapters import _torch_compile as tc
from mojolearn.importers import _program
from mojolearn.importers._graph import UnsupportedModel

REPO = pathlib.Path(__file__).resolve().parents[3]


def _fake_torch():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    onnx = types.ModuleType("torch.onnx")
    dynamo = types.ModuleType("torch._dynamo")
    backends = types.ModuleType("torch._dynamo.backends")
    registry = types.ModuleType("torch._dynamo.backends.registry")

    class SymInt:
        def __int__(self):
            return 4

    class dtype:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return "torch." + self.name

    float32, bfloat16 = dtype("float32"), dtype("bfloat16")

    class device:
        def __init__(self, kind):
            self.type = kind

        def __eq__(self, other):
            return isinstance(other, device) and other.type == self.type

        def __repr__(self):
            return f"device({self.type!r})"

    class Tensor:
        def __init__(self, array=None, shape=None, dtype=float32, device=None):
            self.array = None if array is None else np.asarray(array)
            self.shape = tuple(self.array.shape) if shape is None else tuple(shape)
            self.dtype = dtype
            self.device = device or torch.device("cpu")
            self._version = 0
            self.moved = []

        def detach(self):
            self.moved.append("detach")
            return self

        def cpu(self):
            self.moved.append("cpu")
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self.array

    Tensor.__module__ = "torch"

    class Parameter(Tensor):
        pass

    class Module:
        def __init__(self):
            self.training = True

        def register_buffer(self, name, value):
            setattr(self, name, value)

        def eval(self):
            self.training = False
            return self

        def __call__(self, *args):
            return self.forward(*args)

    Module.__module__ = "torch.nn.modules.module"

    exports = []

    def export(model, args, f, *, opset_version=None, input_names=None, do_constant_folding=True, dynamo=True):
        exports.append({"model": model, "args": args, "input_names": input_names, "dynamo": dynamo,
                        "traced": model(*args)})
        with open(f, "wb") as fh:
            fh.write(b"onnx")

    compiler_fns = {}

    def register_backend(compiler_fn=None, name=None):
        assert name not in compiler_fns, f"duplicate name: {name}"
        compiler_fns[name] = compiler_fn
        return compiler_fn

    made = []

    def tensor(y, device=None):
        t = Tensor(np.array(y, copy=True), device=device)
        made.append((y, t))
        return t

    nn.Module, nn.Parameter = Module, Parameter
    onnx.export = export
    registry._COMPILER_FNS = compiler_fns
    backends.registry = registry
    dynamo.backends = backends
    dynamo.register_backend = register_backend
    torch.nn, torch.onnx, torch._dynamo = nn, onnx, dynamo
    torch.Tensor, torch.SymInt, torch.device = Tensor, SymInt, device
    torch.float32, torch.bfloat16 = float32, bfloat16
    torch.zeros = lambda shape, dtype=float32, device=None: Tensor(np.zeros(shape, np.float32), dtype=dtype,
                                                                   device=device)
    torch.as_tensor = lambda a: a if isinstance(a, Tensor) else Tensor(a)
    torch.tensor = tensor
    torch.no_grad = contextlib.nullcontext
    torch.exports, torch.made = exports, made
    modules = {"torch": torch, "torch.nn": nn, "torch.onnx": onnx, "torch._dynamo": dynamo,
               "torch._dynamo.backends": backends, "torch._dynamo.backends.registry": registry}
    return torch, modules


class _Node:
    def __init__(self, op, args=()):
        self.op, self.args = op, args


class _GraphModule:
    """A stand-in captured graph: returns its inputs' shapes; called by the export trace only."""

    def __init__(self, outputs=("y",)):
        self.graph = types.SimpleNamespace(nodes=[_Node("placeholder"), _Node("output", (outputs,))])
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return [a.shape for a in args]

    def eval(self):
        return self


class _Program:
    def __init__(self, outputs):
        self.outputs = [(f"out_{i}", tuple(y.shape), str(y.dtype)) for i, y in enumerate(outputs)]
        self._results = outputs
        self.runs = []

    def run(self, *arrays):
        self.runs.append(arrays)
        return self._results[0] if len(self._results) == 1 else tuple(self._results)


@pytest.fixture
def fake(monkeypatch):
    torch, modules = _fake_torch()
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(tc, "_PROGRAMS", tc.weakref.WeakKeyDictionary())
    imports = []
    results = [np.arange(10, dtype=np.float32).reshape(5, 2) * np.float32(0.1)]

    def record(path, label, shapes, numeric_mode):
        assert open(path, "rb").read() == b"onnx"
        imports.append({"label": label, "shapes": shapes, "numeric_mode": numeric_mode})
        return _Program(results)

    monkeypatch.setattr(_program, "_import", record)
    torch.imports, torch.results = imports, results
    return torch


def test_register_names_the_backend_once(fake):
    tc.register()
    tc.register()
    assert fake._dynamo.backends.registry._COMPILER_FNS == {"mojolearn": tc.backend}


def test_register_refuses_a_different_backend_under_the_name(fake):
    fake._dynamo.backends.registry._COMPILER_FNS["mojolearn"] = lambda gm, xs: gm
    with pytest.raises(RuntimeError, match="another torch.compile backend"):
        tc.register()


def test_the_entry_point_names_the_backend():
    text = (REPO / "python/pyproject.toml").read_text()
    assert "[project.entry-points.torch_dynamo_backends]" in text
    assert 'mojolearn = "mojolearn.adapters._torch_compile:backend"' in text


def test_the_backend_exports_the_graph_with_bound_weights_and_imports_it(fake):
    gm = _GraphModule()
    weight = fake.nn.Parameter(shape=(4, 3))
    x = fake.Tensor(np.ones((5, 3), np.float32))
    compiled = tc.backend(gm, [weight, x], numeric_mode="identical")
    assert compiled is not gm and callable(compiled)
    (export,) = fake.exports
    assert type(export["model"]).__name__ == "CompiledGraph" and isinstance(export["model"], fake.nn.Module)
    assert export["dynamo"] is False and export["input_names"] == ["input"]
    # the weight is held by the exported module; the program takes the activation only
    assert [a.shape for a in export["args"]] == [(5, 3)]
    assert export["model"].weight_0 is weight
    assert export["traced"] == ((4, 3), (5, 3))
    assert fake.imports == [{"label": "CompiledGraph", "shapes": None, "numeric_mode": "identical"}]


def test_a_call_returns_the_program_output_bytes_on_the_input_device(fake):
    gm = _GraphModule()
    dev = fake.device("cuda")
    x = fake.Tensor(np.full((5, 3), 2.0, np.float32), device=dev)
    compiled = tc.backend(gm, [x])
    traced = len(gm.calls)  # the export trace
    (y,) = compiled(x)  # the captured graph returns a tuple
    assert len(gm.calls) == traced and x.moved[:2] == ["detach", "cpu"]
    ((src, made),) = fake.made
    assert made is y
    assert src is fake.results[0]
    assert y.device == dev and y.array.tobytes() == fake.results[0].tobytes()


def test_a_single_tensor_graph_returns_a_tensor(fake):
    gm = _GraphModule(outputs=_Node("call_function"))
    x = fake.Tensor(np.ones((5, 3), np.float32))
    y = tc.backend(gm, [x])(x)
    assert isinstance(y, fake.Tensor)


def test_a_tuple_graph_returns_a_tuple(fake):
    x = fake.Tensor(np.ones((5, 3), np.float32))
    y = tc.backend(_GraphModule(), [x])(x)
    assert isinstance(y, tuple) and len(y) == 1


def test_an_unsupported_model_propagates_and_nothing_runs_eagerly(fake, monkeypatch):
    def refuse(path, label, shapes, numeric_mode):
        raise UnsupportedModel(label, [("node_3", "Erf", "not supported"), ("node_9", "Gather", "axis 2")])

    monkeypatch.setattr(_program, "_import", refuse)
    gm = _GraphModule()
    x = fake.Tensor(np.ones((5, 3), np.float32))
    with pytest.raises(UnsupportedModel) as err:
        tc.backend(gm, [x])
    assert "node_3" in str(err.value) and "node_9" in str(err.value)
    assert len(gm.calls) == 1  # the export trace only
    assert not tc._PROGRAMS.get(gm)


def test_an_export_failure_is_an_unsupported_model(fake, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("aten::bad has no ONNX symbolic")

    monkeypatch.setattr(fake.onnx, "export", broken)
    with pytest.raises(UnsupportedModel, match="aten::bad"):
        tc.backend(_GraphModule(), [fake.Tensor(np.ones((5, 3), np.float32))])


def test_a_symbolic_size_is_refused_with_dynamic_false(fake):
    with pytest.raises(ValueError, match="static shapes only.*dynamic=False"):
        tc.backend(_GraphModule(), [fake.SymInt(), fake.Tensor(shape=(5, 3))])
    with pytest.raises(ValueError, match="symbolic shape.*dynamic=False"):
        tc.backend(_GraphModule(), [fake.Tensor(shape=(fake.SymInt(), 3))])
    assert fake.exports == [] and fake.imports == []


def test_a_non_tensor_input_and_an_unsupported_dtype_are_refused(fake):
    with pytest.raises(ValueError, match="tensor inputs only"):
        tc.backend(_GraphModule(), [3.0])
    with pytest.raises(ValueError, match="dtype bfloat16"):
        tc.backend(_GraphModule(), [fake.Tensor(shape=(5, 3), dtype=fake.bfloat16)])


def test_unknown_options_and_modes_are_refused(fake):
    x = fake.Tensor(np.ones((5, 3), np.float32))
    with pytest.raises(ValueError, match="numeric_mode only, got bogus"):
        tc.backend(_GraphModule(), [x], bogus=1)
    with pytest.raises(ValueError, match="no mode 'max-autotune'"):
        tc.backend(_GraphModule(), [x], mode="max-autotune")
    tc.backend(_GraphModule(), [x], mode="default")


def test_the_program_is_cached_per_graph_module(fake):
    gm, other = _GraphModule(), _GraphModule()
    x = fake.Tensor(np.ones((5, 3), np.float32))
    a = tc.backend(gm, [x])
    b = tc.backend(gm, [x])
    assert len(fake.imports) == 1 and a._program is b._program
    tc.backend(other, [x])
    assert len(fake.imports) == 2
    tc.backend(gm, [fake.Tensor(np.ones((6, 3), np.float32))])
    assert len(fake.imports) == 3


def test_an_in_place_weight_change_reimports(fake):
    gm = _GraphModule()
    weight = fake.nn.Parameter(shape=(4, 3))
    x = fake.Tensor(np.ones((5, 3), np.float32))
    compiled = tc.backend(gm, [weight, x])
    compiled(weight, x)
    assert len(fake.imports) == 1
    weight._version += 1  # an optimizer step or load_state_dict
    compiled(weight, x)
    assert len(fake.imports) == 2
    compiled(weight, x)
    assert len(fake.imports) == 2


def test_the_backend_module_does_not_import_torch_at_load():
    tree = ast.parse(pathlib.Path(tc.__file__).read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(name.split(".")[0] == "torch" for name in names)


def test_the_per_operation_surfaces_are_excluded_from_the_docs():
    rows = {line.split("\t", 1)[0]: line.split("\t", 1)[1] for line in
            (REPO / "docs-site/api_exclude.txt").read_text().splitlines() if "\t" in line and not line.startswith("#")}
    names = ["torch.ops.mojolearn.operation", "max.mojolearn_matmul", "max.mojolearn_relu"] + \
        [f"max.mojolearn_operation{i}" for i in range(4)]
    for name in names:
        assert rows.get(name) == "building block; a whole model runs as one program", name
    guide = (REPO / "docs-site/user_guide/framework_adapters.md").read_text()
    for name in ("mojolearn.operation", "mojolearn_matmul", "mojolearn_relu", "mojolearn_operation"):
        assert name not in guide, name
    for name in ('backend="mojolearn"', "torch.ops.mojolearn.train", "mojolearn_train", "import_model"):
        assert name in guide, name
    assert "In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs." in guide


def _real_torch_and_binding():
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in this environment")
    from mojolearn import _backend
    try:
        b = _backend.binding("_mojolearn_x_sequence")
    except Exception as e:  # no built bindings in this tree
        pytest.skip(f"no x_sequence binding: {e}")
    if not hasattr(b, "onnx_import_file"):
        pytest.skip("this x_sequence binding has no model import (rebuild it)")
    return torch


def test_a_real_torch_compile_matches_eager_and_import_model_bit_for_bit():
    torch = _real_torch_and_binding()
    tc.register()
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((5, 3), dtype=np.float32))
    compiled = torch.compile(net, backend="mojolearn", fullgraph=True, dynamic=False)
    with torch.no_grad():
        got = compiled(x)
        ref = net(x)
    assert got.shape == (5, 2) and got.dtype == torch.float32
    # torch sums in another order: a close match, not the same bits
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)
    direct = _program.import_model(net, example_inputs=x).run(x.numpy())
    assert np.array_equal(got.numpy().view(np.uint32), direct.view(np.uint32))


def test_the_installed_torch_abi_probe(fake, monkeypatch):
    # The registration's C++ ABI check: torch's own report wins; a torch >= 2.9
    # that reports nothing is ABI 1 (every Linux wheel since 2.9 is C++11 ABI).
    import importlib
    import types
    monkeypatch.delitem(sys.modules, "mojolearn.adapters.torch", raising=False)
    adapter = importlib.import_module("mojolearn.adapters.torch")
    monkeypatch.setattr(adapter, "_torch", types.SimpleNamespace(compiled_with_cxx11_abi=lambda: False))
    assert adapter._torch_cxx11_abi("2.13") == 0
    monkeypatch.setattr(adapter, "_torch", types.SimpleNamespace(_C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)))
    assert adapter._torch_cxx11_abi("2.13") == 1
    monkeypatch.setattr(adapter, "_torch", types.SimpleNamespace(_C=types.SimpleNamespace()))
    assert adapter._torch_cxx11_abi("2.13") == 1
    assert adapter._torch_cxx11_abi("2.8") is None
    monkeypatch.setattr(adapter, "_torch", types.SimpleNamespace(
        __version__="2.13.0", version=types.SimpleNamespace(cuda="12.8", hip=None), _C=types.SimpleNamespace()))
    record = {"torch": {"series": "2.13", "backend": "cuda", "cxx11_abi": 1}}
    assert adapter._refusal(record) is None
    assert "C++ ABI" in adapter._refusal({"torch": record["torch"] | {"cxx11_abi": 0}})
