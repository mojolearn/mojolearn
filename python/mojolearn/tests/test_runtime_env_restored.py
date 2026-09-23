"""Loading a binding leaves the process environment as it found it.

The bundled runtime writes PYTHONEXECUTABLE, PYTHONPATH and MOJO_PYTHON_LIBRARY
into os.environ when the first Mojo binding is executed (measured 2026-09-23,
0.8.15 wheel, a uv venv on a RunPod CPU pod). A child started afterwards with
`subprocess.run([sys.executable, ...])` inherits them, and on CPython 3.11 the
child took its executable and prefix from PYTHONEXECUTABLE, lost the venv's
site-packages and could not import numpy. `_backend._exec_binding` restores
the three variables after every load; these tests hold it to that from a
fresh interpreter, so an earlier import in the pytest process cannot mask it.
"""
import json
import os
import subprocess
import sys

import pytest

VARS = ("PYTHONEXECUTABLE", "PYTHONPATH", "MOJO_PYTHON_LIBRARY")

LOAD_ONE_BINDING = (
    "import json, os, sys\n"
    "before = {k: os.environ.get(k) for k in %r}\n"
    "import mojolearn\n"
    "from mojolearn import _buffer\n"
    "_buffer._native('cast_f64_to_f32')\n"
    "after = {k: os.environ.get(k) for k in %r}\n"
    "child = subprocess_prefix = None\n"
    "import subprocess\n"
    "r = subprocess.run([sys.executable, '-c', 'import sys, json; print(json.dumps([sys.executable, sys.prefix]))'],\n"
    "                   capture_output=True, text=True)\n"
    "print(json.dumps(dict(before=before, after=after, parent=[sys.executable, sys.prefix],\n"
    "                      child_rc=r.returncode, child=r.stdout.strip(), child_err=r.stderr[-800:])))\n"
) % (VARS, VARS)


def _fresh(extra_env):
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    env.update(extra_env)
    r = subprocess.run([sys.executable, "-c", LOAD_ONE_BINDING], capture_output=True, text=True,
                       env=env, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if r.returncode != 0:
        pytest.skip(f"the package does not load a binding here:\n{r.stderr[-1500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_absent_variables_stay_absent_after_a_binding_loads():
    out = _fresh({})
    assert out["before"] == {k: None for k in VARS}
    assert out["after"] == {k: None for k in VARS}, out["after"]


def test_a_variable_the_caller_set_keeps_its_value():
    out = _fresh({"PYTHONPATH": os.environ.get("PYTHONPATH", "") or "."})
    assert out["after"]["PYTHONPATH"] == out["before"]["PYTHONPATH"], out
    assert out["after"]["PYTHONEXECUTABLE"] is None and out["after"]["MOJO_PYTHON_LIBRARY"] is None, out


def test_a_child_of_the_parent_keeps_its_interpreter_and_prefix():
    out = _fresh({})
    assert out["child_rc"] == 0, out["child_err"]
    assert json.loads(out["child"]) == out["parent"], out
