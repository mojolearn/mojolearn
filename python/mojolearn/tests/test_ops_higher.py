# SPDX-License-Identifier: Apache-2.0
"""Higher-order gradients through mojolearn.ops.Session: the refusals.

The numerics (backward with create_graph=True, grad_tensor, hvp, vjp and jvp
against float64 nested central differences, the adjoint identity and the
closed forms, each repeated for the same words) run natively in
checks/native_tests/ops_higher_{cpu,gpu}.mojo."""
import numpy as np
import pytest

from mojolearn import _backend


def _has_higher_entries():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        return hasattr(b, "ops_session_call")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_higher_entries(),
                                reason="the sequence binding has no higher-order entries (rebuild it)")


def _ops():
    from mojolearn import ops
    return ops


# ---------------------------------------------------------------- refusals

def test_refusals():
    ops = _ops()
    with ops.Session(record=True) as s:
        x = s.tensor(np.array([[0.3, -0.8, 1.2], [0.7, 0.2, -0.4]], dtype=np.float32), requires_grad=True)
        loss = s.sum(s.mul(x, x))
        with pytest.raises(ValueError):
            s.backward(loss, accumulate=True, create_graph=True)
        s.backward(loss)
        with pytest.raises(Exception):
            s.grad_tensor(x)  # the last backward did not record
        wt = s.tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
        bs = s.tensor(np.zeros((1, 3), dtype=np.float32), requires_grad=True)
        normed = s.sum(s.mul(s.layer_norm(x, wt, bs), x))
        s.backward(normed, create_graph=True)
        assert np.isfinite(s.numpy(s.grad_tensor(x))).all()
        s.backward(normed)
        assert np.isfinite(s.grad(x)).all()
    with ops.Session() as s:
        with pytest.raises(Exception):
            s.backward(s.tensor(np.ones((1, 1), dtype=np.float32)), create_graph=True)


