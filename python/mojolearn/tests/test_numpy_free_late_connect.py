# SPDX-License-Identifier: Apache-2.0
"""`import mojolearn` never loads NumPy; the two owners that used to load it
at import connect it natively at their first call instead.

PolynomialCountSketch (random_state=None or a RandomState) reads NumPy's RNG
metadata at its first such fit (providers/python/pcs_rng_pyapi.mojo
_pcs_connect). The CNN layers import an installed NumPy at their first
operation (bindings/xcnn_common.mojo xcnn_connect_numpy). A child process
imports mojolearn first and must return the bits of this NumPy-first process.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import mojolearn as ml

_X = [[0.25 * ((3 * i + 5 * j) % 11) - 1.0 for j in range(4)] for i in range(12)]

_LATE_CHILD = r'''
import json, sys
import mojolearn as ml
assert "numpy" not in sys.modules, "import mojolearn loaded NumPy"
X = json.loads(sys.stdin.read())
out = {}
# CNN first, before the process ever imports NumPy itself.
conv = ml.Conv2d(1, 2, 3, random_state=0)
out["conv_weight"] = memoryview(conv.weight_).tobytes().hex()
import numpy as np
x = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) / np.float32(8.0)
out["conv_forward"] = np.asarray(conv.forward(x)).tobytes().hex()
pcs = ml.PolynomialCountSketch(n_components=8, random_state=np.random.RandomState(7)).fit(X)
out["pcs_rs"] = np.asarray(pcs.transform(X)).tobytes().hex()
np.random.seed(3)
pcs = ml.PolynomialCountSketch(n_components=8, random_state=None).fit(X)
out["pcs_none"] = np.asarray(pcs.transform(X)).tobytes().hex()
print("RESULT " + json.dumps(out))
'''

_BLOCKED_CHILD = r'''
import json, sys
class BlockNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
sys.meta_path.insert(0, BlockNumpy())
import mojolearn as ml
X = json.loads(sys.stdin.read())
ml.PolynomialCountSketch(n_components=8, random_state=0).fit(X)
try:
    ml.PolynomialCountSketch(n_components=8, random_state=None).fit(X)
except ImportError as error:
    assert "requires optional NumPy support" in str(error), str(error)
else:
    raise AssertionError("random_state=None ran without NumPy")
assert "numpy" not in sys.modules
print("RESULT ok")
'''


def _child(code):
    env = dict(os.environ, PYTHONPATH=str(Path(ml.__file__).resolve().parents[1]))
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          input=json.dumps(_X), timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, done.stdout[-2000:]
    return line[-1][len("RESULT "):]


def test_late_numpy_cnn_and_pcs_match_a_numpy_first_process():
    got = json.loads(_child(_LATE_CHILD))
    conv = ml.Conv2d(1, 2, 3, random_state=0)
    assert got["conv_weight"] == np.asarray(conv.weight_).tobytes().hex()
    x = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) / np.float32(8.0)
    assert got["conv_forward"] == np.asarray(conv.forward(x)).tobytes().hex()
    pcs = ml.PolynomialCountSketch(n_components=8, random_state=np.random.RandomState(7)).fit(_X)
    assert got["pcs_rs"] == np.asarray(pcs.transform(_X)).tobytes().hex()
    state = np.random.get_state()
    try:
        np.random.seed(3)
        pcs = ml.PolynomialCountSketch(n_components=8, random_state=None).fit(_X)
    finally:
        np.random.set_state(state)
    assert got["pcs_none"] == np.asarray(pcs.transform(_X)).tobytes().hex()


def test_numpy_free_install_refuses_only_the_numpy_seed():
    assert _child(_BLOCKED_CHILD) == "ok"
