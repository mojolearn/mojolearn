"""Actual exported-program Torch boundary: import order and hook-free admission."""
import os
import subprocess
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    path = tmp_path_factory.mktemp("torch-program-module") / "add.onnx"
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "offset"], ["y"])], "add",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(np.full((2, 3), 2, dtype=np.float32), "offset")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)
    return path


def run_case(artifact, imports, body):
    code = imports + """
import pathlib, sys, types
assert pathlib.Path(mojolearn.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)), mojolearn.__file__
program = mojolearn.import_model(sys.argv[1], numeric_mode='identical')
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
expected = x + 2
""" + body
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code, str(artifact)],
                            cwd=artifact.parent, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("imports", ["import torch\nimport mojolearn\n", "import mojolearn\nimport torch\n"])
def test_optional_framework_import_order(artifact, imports):
    run_case(artifact, imports, "torch.testing.assert_close(program.run_torch(x), expected, rtol=0, atol=0)\n")


def test_ordinary_module_hook_is_not_executed(artifact):
    run_case(artifact, "import torch\nimport mojolearn\n", """
calls = []
def hostile(name):
    calls.append(name)
    raise AssertionError('module hook executed')
previous = torch.__dict__.get('__getattr__')
torch.__dict__['__getattr__'] = hostile
try:
    result = program.run_torch(x)
finally:
    torch.__dict__['__getattr__'] = previous
assert calls == [], calls
torch.testing.assert_close(result, expected, rtol=0, atol=0)
""")


@pytest.mark.parametrize("replacement", ["module", "allocator", "dtype"])
def test_replaced_namespace_is_refused_without_hooks(artifact, replacement):
    body = """
calls = []
class HostileModule(types.ModuleType):
    def __getattribute__(self, name):
        calls.append(name)
        raise AssertionError('module subclass hook executed')
"""
    if replacement == "module":
        body += "owner, key, value = sys.modules, 'torch', HostileModule('torch')\n"
    elif replacement == "allocator":
        body += "owner, key, value = torch._C.__dict__, '_VariableFunctions', HostileModule('allocator')\n"
    else:
        body += "owner, key, value = torch.__dict__, 'float32', torch.float64\n"
    body += """
previous = owner[key]
owner[key] = value
try:
    try:
        program.run_torch(x)
    except TypeError:
        pass
    else:
        raise AssertionError('replaced native namespace accepted')
finally:
    owner[key] = previous
assert calls == [], calls
torch.testing.assert_close(program.run_torch(x), expected, rtol=0, atol=0)
"""
    run_case(artifact, "import torch\nimport mojolearn\n", body)
