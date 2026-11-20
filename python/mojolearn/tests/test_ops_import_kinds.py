# SPDX-License-Identifier: Apache-2.0
"""The elementwise operations of imported models in mojolearn.ops.Session
(docs/plans/MODEL_IMPORT.md section 3.3), the Python side: the refusals. The
values and gradients (erf, gelu, softplus, silu, leaky relu, elu, hard
sigmoid and swish, clip, abs, floor, ceil, round, sin, cos, sign,
reciprocal, pow, minimum, maximum, where, the comparisons and logical
operations, expand, argmax, pad, any, all, cumsum and index) run natively
against float64 oracles in checks/native_tests/core_ops_import_kinds_*.mojo."""
import numpy as np
import pytest

from mojolearn import _backend


def _has_import_kinds():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        if not hasattr(b, "ops_session_call"):
            return False
        from mojolearn import ops
        with ops.Session() as s:
            return s.shape(s.erf(s.tensor(np.zeros((2, 3), dtype=np.float32)))) == (2, 3)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_import_kinds(),
                                reason="the sequence binding has no import kinds (rebuild it)")


def _ops():
    from mojolearn import ops
    return ops


def _x(shape, seed, low=-2.0, high=2.0):
    return np.random.default_rng(seed).uniform(low, high, size=shape).astype(np.float32)


def test_refusals():
    with _ops().Session() as s:
        x = s.tensor(np.zeros((2, 3), dtype=np.float32))
        y = s.tensor(np.zeros((4, 3), dtype=np.float32))
        with pytest.raises(Exception):
            s.maximum(x, y)
        with pytest.raises(ValueError):
            s.gelu(x, approximate="sigmoid")
        with pytest.raises(Exception):
            s.pad(x, (1, 1))
        with pytest.raises(Exception):
            s.clip(x, 1.0, -1.0)


# ------------------------------------------------- the attention-mask ops (lane ag-import-lm-ops) --

def test_index_refusals():
    with _ops().Session() as s:
        x = s.tensor(_x((4, 5), 26))
        far = s.tensor(np.array([[0, 4]], dtype=np.float32))
        with pytest.raises(Exception):
            s.numpy(s.index(x, (far, far)))  # position 4 is outside axis 0 (4 rows)
        with pytest.raises(Exception):
            s.index(x, (far, far), axes=(1, 0))  # the axes must ascend
        with pytest.raises(ValueError):
            s.index(x, (far,), axes=(0, 1))
