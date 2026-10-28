# SPDX-License-Identifier: Apache-2.0
"""Artifact-only Torch front-end: real native execution and callback refusals."""
from pathlib import Path
import numpy as np
import pytest

from mojolearn import import_model
from mojolearn._verify_native_runtime import audit_native_call


def _artifact(path, scale=1.0):
    # Independent test preparation, never called by the product.
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    graph = helper.make_graph([
        helper.make_node("Gemm", ["x", "w", "b"], ["linear"]),
        helper.make_node("Relu", ["linear"], ["y"]),
    ], "artifact-boundary", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
       [numpy_helper.from_array(np.asarray([[1, -2], [3, 4], [-1, 2]], dtype="f4") * scale, "w"),
        numpy_helper.from_array(np.asarray([0.5, -0.5], dtype="f4"), "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)


def test_artifact_torch_execution_matches_array_route_and_refreshes_inputs(tmp_path):
    import torch
    path = tmp_path / "model.onnx"
    _artifact(path)
    admitted = audit_native_call(import_model, str(path), numeric_mode="identical")
    admitted.assert_native()
    program = admitted.result
    x = torch.tensor([[1., 2., 3.], [-2., 1., 4.]])
    for values in (x, x + 2, x[:, [2, 1, 0]]):
        audit = audit_native_call(program.run_torch, values)
        audit.assert_native()
        expected = np.asarray(program.run(values.numpy()))
        assert audit.result.numpy().tobytes() == expected.tobytes()
    with pytest.raises(ValueError, match="shape"):
        program.run_torch(torch.ones(1, 3))


def test_weights_are_artifact_snapshots_and_change_only_on_explicit_import(tmp_path):
    import torch
    path = tmp_path / "model.onnx"
    _artifact(path)
    first = import_model(str(path), numeric_mode="identical")
    x = torch.ones(2, 3)
    before = first.run_torch(x).numpy().copy()
    _artifact(path, scale=2.0)
    assert first.run_torch(x).numpy().tobytes() == before.tobytes()
    second = import_model(str(path), numeric_mode="identical")
    actual = second.run_torch(x).numpy()
    assert actual.tobytes() == np.asarray(second.run(x.numpy())).tobytes()
    assert actual.tobytes() != before.tobytes()


def test_retired_backend_refuses_without_inspecting_or_executing_model():
    from mojolearn.adapters._torch_compile import backend
    calls = []
    class Model:
        def forward(self, *args):
            calls.append("forward")
            raise AssertionError("capture executed")
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("model inspected")
    audit = audit_native_call(backend, Model(), ())
    assert isinstance(audit.exception, TypeError)
    assert "export the model first" in str(audit.exception)
    assert not audit.frames and not audit.forbidden_native, audit.report()
    assert not audit.unresolved_native and not audit.wrapper_work, audit.report()
    assert not audit.unobserved_threads, audit.report()
    assert calls == []


def test_torchdynamo_is_not_registered_by_packaging():
    root = Path(__file__).resolve().parents[3]
    assert "torch_dynamo_backends" not in (root / "python/pyproject.toml").read_text()
