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

    float32, bfloat16, int32 = dtype("float32"), dtype("bfloat16"), dtype("int32")

    class device:
        def __init__(self, kind, index=None):
            self.type = kind
            self.index = index

        def __eq__(self, other):
            return isinstance(other, device) and (other.type, other.index) == (self.type, self.index)

        def __repr__(self):
            return f"device({self.type!r}, {self.index!r})"

    addresses = iter(range(0x7000_0000, 0x8000_0000, 0x1000))
    events = []

    class Tensor:
        def __init__(self, array=None, shape=None, dtype=float32, device=None):
            self.array = None if array is None else np.asarray(array)
            self.shape = tuple(self.array.shape) if shape is None else tuple(shape)
            self.dtype = dtype
            self.device = device or torch.device("cpu")
            self._version = 0
            self.moved = []
            self.address = next(addresses)

        def data_ptr(self):
            return self.address

        def detach(self):
            self.moved.append("detach")
            return self

        def cpu(self):
            self.moved.append("cpu")
            return self

        def contiguous(self):
            self.moved.append("contiguous")
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
    torch.float32, torch.bfloat16, torch.int32 = float32, bfloat16, int32
    empties = []

    def empty(shape, dtype=float32, device=None):
        t = Tensor(shape=shape, dtype=dtype, device=device)
        empties.append(t)
        return t

    class _Stream:
        def __init__(self, dev):
            self.dev = dev

        def synchronize(self):
            events.append(("sync", self.dev))

    torch.empty = empty
    torch.cuda = types.SimpleNamespace(current_stream=lambda dev=None: _Stream(dev))
    torch.empties, torch.events = empties, events
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


class _DeviceProgram(_Program):
    """A stand-in program whose binding takes device buffers (ImportedProgram._run_device)."""

    def __init__(self, outputs, events):
        super().__init__(outputs)
        self.device_runs = []
        self._events = events

    def _run_device(self, program, inputs, outputs):
        assert program is self
        self._events.append(("native", None))
        self.device_runs.append((inputs, outputs))


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


@pytest.fixture
def device_program(fake, monkeypatch):
    """The recorder returns a program whose binding takes device buffers."""
    made = []

    def record(path, label, shapes, numeric_mode):
        fake.imports.append({"label": label, "shapes": shapes, "numeric_mode": numeric_mode})
        made.append(_DeviceProgram(fake.results, fake.events))
        return made[-1]

    monkeypatch.setattr(_program, "_import", record)
    return made


def test_a_cuda_tensor_takes_the_device_route(fake, device_program):
    dev = fake.device("cuda", 0)
    x = fake.Tensor(np.full((5, 3), 2.0, np.float32), device=dev)
    compiled = tc.backend(_GraphModule(), [x])
    assert compiled._route([x]) == ("device", "")
    (y,) = compiled(x)
    (program,) = device_program
    # no host copy, no NumPy result: the program got the input's device address and shape
    assert program.runs == [] and "cpu" not in x.moved and fake.made == []
    ((inputs, outputs),) = program.device_runs
    assert inputs == [(x.address, (5, 3), "float32")]
    # the outputs are torch.empty tensors on the input's device, of the declared shape and dtype,
    # and the program wrote into them by address
    (out,) = fake.empties
    assert out is y and outputs == [y.address]
    assert y.device == dev and y.shape == (5, 2) and y.dtype is fake.float32


def test_the_device_route_drains_the_torch_stream_around_the_native_call(fake, device_program):
    dev = fake.device("cuda", 0)
    x = fake.Tensor(np.ones((5, 3), np.float32), device=dev)
    compiled = tc.backend(_GraphModule(), [x])
    fake.events.clear()
    compiled(x)
    assert fake.events == [("sync", dev), ("native", None), ("sync", dev)]
    # the output storage was allocated before the first drain
    assert len(fake.empties) == 1


def test_the_device_route_allocates_int32_outputs_as_int32(fake, device_program):
    fake.results[:] = [np.zeros((5,), np.int32), np.zeros((5, 2), np.float32)]
    dev = fake.device("cuda", 0)
    x = fake.Tensor(np.ones((5, 3), np.float32), device=dev)
    a, b = tc.backend(_GraphModule(outputs=("i", "y")), [x])(x)
    assert (a.dtype, a.shape) == (fake.int32, (5,)) and (b.dtype, b.shape) == (fake.float32, (5, 2))
    ((_, outputs),) = device_program[0].device_runs
    assert outputs == [a.address, b.address]


def test_the_device_route_passes_each_input_dtype_by_name(fake, device_program):
    dev = fake.device("cuda", 0)
    ids = fake.Tensor(np.ones((5, 3), np.int32), dtype=fake.int32, device=dev)
    tc.backend(_GraphModule(), [ids])(ids)
    ((inputs, _),) = device_program[0].device_runs
    assert inputs == [(ids.address, (5, 3), "int32")]
    assert "contiguous" in ids.moved


def test_a_cpu_tensor_takes_the_host_route(fake, device_program):
    x = fake.Tensor(np.ones((5, 3), np.float32))
    compiled = tc.backend(_GraphModule(), [x])
    route, reason = compiled._route([x])
    assert route == "host" and "cpu" in reason
    (y,) = compiled(x)
    (program,) = device_program
    assert program.device_runs == [] and len(program.runs) == 1
    assert y.array.tobytes() == fake.results[0].tobytes() and fake.events == []


def test_an_mps_tensor_takes_the_host_route_with_the_reason(fake, device_program):
    dev = fake.device("mps", 0)
    x = fake.Tensor(np.ones((5, 3), np.float32), device=dev)
    compiled = tc.backend(_GraphModule(), [x])
    route, reason = compiled._route([x])
    assert route == "host" and "MPS" in reason and "raw pointer" in reason
    (y,) = compiled(x)
    (program,) = device_program
    assert program.device_runs == [] and x.moved[:2] == ["detach", "cpu"]
    assert y.device == dev and y.array.tobytes() == fake.results[0].tobytes()


def test_another_gpu_or_a_host_only_binding_takes_the_host_route(fake, device_program, monkeypatch):
    x1 = fake.Tensor(np.ones((5, 3), np.float32), device=fake.device("cuda", 1))
    compiled = tc.backend(_GraphModule(), [x1])
    route, reason = compiled._route([x1])
    assert route == "host" and "GPU 0" in reason
    compiled(x1)
    assert device_program[0].device_runs == [] and len(device_program[0].runs) == 1
    # a binding without the device entry (the CPU-only install): the host route on a cuda tensor too
    x0 = fake.Tensor(np.ones((5, 3), np.float32), device=fake.device("cuda", 0))
    compiled = tc.backend(_GraphModule(), [x0])
    compiled._program._run_device = None
    route, reason = compiled._route([x0])
    assert route == "host" and "no device buffers" in reason
    compiled(x0)
    assert compiled._program.device_runs == []


def test_an_unsupported_model_propagates_on_the_device_route_too(fake, monkeypatch):
    def refuse(path, label, shapes, numeric_mode):
        raise UnsupportedModel(label, [("node_1", "Erf", "not supported")])

    monkeypatch.setattr(_program, "_import", refuse)
    gm = _GraphModule()
    x = fake.Tensor(np.ones((5, 3), np.float32), device=fake.device("cuda", 0))
    with pytest.raises(UnsupportedModel, match="node_1"):
        tc.backend(gm, [x])
    assert fake.events == [] and fake.empties == []


def test_the_backend_module_does_not_import_torch_at_load():
    tree = ast.parse(pathlib.Path(tc.__file__).read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(name.split(".")[0] == "torch" for name in names)


# The docs and whole-model-only surface checks live in tools/test_adapter_surfaces.py (no bindings needed).


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


def test_a_real_device_route_matches_the_host_route_and_import_model_bit_for_bit():
    # RUN OWED on nv and amd (the device route); on the Mac and a CPU torch both calls take the host route
    # and the test checks host == import_model only.
    torch = _real_torch_and_binding()
    if not torch.cuda.is_available():
        pytest.skip("no CUDA or ROCm device in this torch")
    tc.register()
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)).eval()
    x_host = torch.from_numpy(np.random.default_rng(1).standard_normal((64, 16), dtype=np.float32))
    dev = torch.device("cuda", 0)
    net_dev = net.to(dev)
    x_dev = x_host.to(dev)
    compiled = torch.compile(net_dev, backend="mojolearn", fullgraph=True, dynamic=False)
    with torch.no_grad():
        got_dev = compiled(x_dev)
    entries = [e for per_gm in tc._PROGRAMS.values() for e in per_gm.values()]
    program = entries[-1][0]
    if program._run_device is None:
        pytest.skip("this binding has no ops_imported_run_device (rebuild the x_sequence bindings)")
    # the host route's arithmetic is exactly program.run on the host copy (_Compiled._run_host)
    got_host = program.run(x_dev.detach().cpu().contiguous().numpy())
    a = got_dev.cpu().numpy().view(np.uint32)
    assert got_dev.device == dev and got_dev.dtype == torch.float32
    assert np.array_equal(a, np.asarray(got_host).view(np.uint32))
    direct_import = _program.import_model(net.cpu(), example_inputs=x_host).run(x_host.numpy())
    assert np.array_equal(a, direct_import.view(np.uint32))


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
