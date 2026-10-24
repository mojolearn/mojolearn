# SPDX-License-Identifier: Apache-2.0
"""import_model on a torch.nn.Module or an onnx.ModelProto.

The glue tests run without torch: a stand-in ``torch`` module in
``sys.modules`` records the export call, and the file import is replaced by a
recorder, so they check the detection, the arguments torch.onnx.export gets,
the temporary file's cleanup and each error. The last test needs PyTorch and
a binding with model import, and imports and runs a small real module.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest

from mojolearn.importers import _program
from mojolearn.importers._graph import UnsupportedModel


def _fake_torch(export=None, signature_has_dynamo=True):
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    onnx = types.ModuleType("torch.onnx")

    class Module:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False
            return self

    Module.__module__ = "torch.nn.modules.module"

    class Tensor:
        def __init__(self, shape):
            self.shape = tuple(shape)

    Tensor.__module__ = "torch"

    calls = []

    def default_export(model, args, f, **kwargs):
        calls.append((model, args, f, kwargs))
        with open(f, "wb") as fh:
            fh.write(b"onnx")

    if signature_has_dynamo:
        def export_fn(model, args, f, *, opset_version=None, input_names=None, do_constant_folding=True,
                      dynamo=True):
            return (export or default_export)(model, args, f, opset_version=opset_version,
                                              input_names=input_names, do_constant_folding=do_constant_folding,
                                              dynamo=dynamo)
    else:
        def export_fn(model, args, f, *, opset_version=None, input_names=None, do_constant_folding=True):
            return (export or default_export)(model, args, f, opset_version=opset_version,
                                              input_names=input_names, do_constant_folding=do_constant_folding)

    nn.Module = Module
    onnx.export = export_fn
    torch.nn = nn
    torch.onnx = onnx
    torch.Tensor = Tensor
    torch.zeros = lambda shape: Tensor(shape)
    torch.as_tensor = lambda a: Tensor(getattr(a, "shape", ()))
    torch.no_grad = contextlib.nullcontext
    torch.calls = calls
    return torch


@pytest.fixture
def fake(monkeypatch):
    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    seen = []

    def record(path, label, shapes, numeric_mode):
        seen.append({"path": path, "exists": os.path.exists(path), "label": label, "shapes": shapes,
                     "numeric_mode": numeric_mode})
        return "program"

    monkeypatch.setattr(_program, "_import", record)
    return torch, seen


def _net(torch):
    class Net(torch.nn.Module):
        pass

    Net.__module__ = "__main__"
    return Net()


def test_a_module_with_example_inputs_is_exported_and_imported(fake):
    torch, seen = fake
    net = _net(torch)
    x = torch.zeros((8, 3, 32, 32))
    assert _program.import_model(net, example_inputs=x, numeric_mode="identical") == "program"
    (model, args, path, kwargs), = torch.calls
    assert model is net and not net.training
    assert args == (x,)
    assert kwargs["opset_version"] >= 13
    assert kwargs["dynamo"] is False
    assert kwargs["input_names"] == ["input"]
    (got,) = seen
    assert got["path"] == path and got["exists"] and got["label"] == "Net"
    assert got["numeric_mode"] == "identical"
    # the temporary file and its directory are gone
    assert not os.path.exists(path) and not os.path.exists(os.path.dirname(path))


def test_several_example_inputs_are_named_in_order(fake):
    torch, seen = fake
    a, b = torch.zeros((2, 3)), torch.zeros((2, 5))
    _program.import_model(_net(torch), example_inputs=(a, b))
    (_, args, _, kwargs), = torch.calls
    assert args == (a, b) and kwargs["input_names"] == ["input_0", "input_1"]


def test_shapes_alone_build_float32_zeros_named_by_key(fake):
    torch, seen = fake
    _program.import_model(_net(torch), shapes={"x": (4, 3), "mask": (4,)})
    (_, args, _, kwargs), = torch.calls
    assert [t.shape for t in args] == [(4, 3), (4,)]
    assert kwargs["input_names"] == ["x", "mask"]
    assert seen[0]["shapes"] == {"x": (4, 3), "mask": (4,)}


def test_shapes_name_the_example_inputs(fake):
    torch, _ = fake
    _program.import_model(_net(torch), shapes={"img": (1, 2)}, example_inputs=torch.zeros((1, 2)))
    assert torch.calls[0][3]["input_names"] == ["img"]
    with pytest.raises(ValueError, match="shapes names 2 inputs but example_inputs holds 1"):
        _program.import_model(_net(torch), shapes={"a": (1,), "b": (1,)}, example_inputs=torch.zeros((1,)))


def test_an_older_torch_without_the_dynamo_flag(monkeypatch):
    torch = _fake_torch(signature_has_dynamo=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_program, "_import", lambda *a: "program")
    _program.import_model(_net(torch), example_inputs=torch.zeros((1, 3)))
    assert "dynamo" not in torch.calls[0][3]


def test_missing_example_inputs_says_what_to_pass(fake):
    torch, seen = fake
    with pytest.raises(ValueError, match=r"example_inputs=torch\.zeros\(8, 3, 32, 32\).*shapes="):
        _program.import_model(_net(torch))
    assert not torch.calls and not seen


def test_an_export_failure_is_an_unsupported_model_and_cleans_up(monkeypatch):
    paths = []

    def broken(model, args, f, **kwargs):
        paths.append(f)
        with open(f, "wb") as fh:
            fh.write(b"partial")
        raise RuntimeError("Exporting the operator 'aten::fancy' to ONNX opset version 17 is not supported")

    torch = _fake_torch(export=broken)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_program, "_import", lambda *a: pytest.fail("no import after a failed export"))
    with pytest.raises(UnsupportedModel, match="aten::fancy") as e:
        _program.import_model(_net(torch), example_inputs=torch.zeros((1, 3)))
    assert e.value.problems[0][0] == "torch.onnx.export"
    assert not os.path.exists(os.path.dirname(paths[0]))


def test_no_torch_is_an_import_error_naming_pip(monkeypatch):
    class Module:
        pass

    Module.__module__ = "torch.nn.modules.module"
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch raises ImportError
    monkeypatch.setattr(_program, "_import", lambda *a: pytest.fail("no import without torch"))
    with pytest.raises(ImportError, match="pip install torch"):
        _program.import_model(Module(), example_inputs=object())


def test_a_torch_object_that_is_not_a_module_is_refused(fake):
    torch, _ = fake

    class ExportedProgram:
        pass

    ExportedProgram.__module__ = "torch.export.exported_program"
    with pytest.raises(ValueError, match=r'torch\.onnx\.export\(model, args, "model\.onnx"\)'):
        _program.import_model(ExportedProgram(), example_inputs=torch.zeros((1,)))
    assert not torch.calls


def test_an_onnx_model_proto_is_written_and_imported(monkeypatch):
    class ModelProto:
        def SerializeToString(self):
            return b"\x08\x07"

    ModelProto.__module__ = "onnx.onnx_ml_pb2"
    seen = []

    def record(path, label, shapes, numeric_mode):
        with open(path, "rb") as fh:
            seen.append((path, fh.read(), label, shapes))
        return "program"

    monkeypatch.setattr(_program, "_import", record)
    assert _program.import_model(ModelProto(), shapes={"x": (2, 2)}) == "program"
    (path, data, label, shapes), = seen
    assert data == b"\x08\x07" and label == "ModelProto" and shapes == {"x": (2, 2)}
    assert not os.path.exists(os.path.dirname(path))


def test_example_inputs_with_an_onnx_file_is_refused(monkeypatch):
    monkeypatch.setattr(_program, "_import", lambda *a: pytest.fail("refused before the import"))
    with pytest.raises(ValueError, match="example_inputs is for a torch.nn.Module"):
        _program.import_model("model.onnx", example_inputs=object())


def test_the_import_module_does_not_import_torch_at_load():
    # torch is imported inside import_model, only when the caller passes a torch object
    import ast
    tree = ast.parse(open(_program.__file__, encoding="utf-8").read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(name.split(".")[0] == "torch" for name in names)


def test_a_real_torch_sequential_imports_and_runs():
    import numpy as np  # before torch: some installs abort on a second OpenMP runtime otherwise
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in this environment")
    from mojolearn import _backend
    try:
        b = _backend.binding("_mojolearn_x_sequence")
    except Exception as e:  # no built bindings in this tree
        pytest.skip(f"no x_sequence binding: {e}")
    if not hasattr(b, "onnx_import_file"):
        pytest.skip("this x_sequence binding has no model import (rebuild it)")
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    prog = _program.import_model(net, example_inputs=torch.zeros(5, 3))
    assert [shape for _, shape, _ in prog.inputs] == [(5, 3)]
    x = np.random.default_rng(0).standard_normal((5, 3), dtype=np.float32)
    y = prog.run(x)
    assert y.shape == (5, 2) and np.isfinite(y).all()
    assert np.array_equal(prog.run(x).view(np.uint32), y.view(np.uint32))
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    # torch sums in another order: a close match, not the same bits
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-5)
