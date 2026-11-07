# SPDX-License-Identifier: Apache-2.0
"""Drivers return the native result to the package and the public one to a user.

THE par-kernel-ridge DTYPE FLIP (AMD identity record at e548c5b01,
2026-10-24): the old worker ran as `python -m mojolearn._parallel_worker`, so its
module is `__main__` and `_mode._caller_outside` took it for a USER. Once the
worker loaded NumPy whenever the parent had it, `KernelRidge.predict` inside
the worker was widened to float64 (`_mode._public_result`) and pickled back,
while the identity harness's plain `predict` (an internal caller) is the
native float32 Array: "apply_kernel_method predict and plain predict differ:
3963 bytes of 4096" on all nine fixtures. The drivers are native now (no
worker): the native owner returns the native result and the driver applies
the public conversion for its own caller (`tests/_parallel_pool.public_method_result`).
"""
import sys

import numpy as np
import pytest

from mojolearn import _mode
from mojolearn import parallel_classical
from mojolearn._array import Array


def _frame_in(globs):
    """A live frame whose globals are `globs` (a function defined there)."""
    exec("import sys\ndef _here():\n    return sys._getframe()\n", globs)
    return globs["_here"]()


def _call_in(globs, function, *args):
    """`function(*args)` called from code whose module globals are `globs`."""
    globs["_target"] = function
    exec("def _call(*a):\n    return _target(*a)\n", globs)
    return globs["_call"](*args)


def _user_globals():
    return {"__name__": "user_script"}


class _Regressor:
    """A regressor whose `predict` carries the same `_public_output` wrapper
    `install_public_outputs` puts on every estimator's result methods."""
    _estimator_type = "regressor"

    def _raw_predict(self, X):
        return _raw(4)

    predict = _mode._public_output(_raw_predict, "predict")


def _raw(n):
    out = Array((n,), "<f4")
    return out


def _fake_native(model, operation, args, context):
    """The native `family_km_par_apply` door: the raw float32 Array, exactly
    what the plain native predict returns to an internal caller."""
    assert operation == "family_km_par_apply"
    return _raw(4)


# The driver is one native call (family_km_par_apply); the native owner
# refuses a patched estimator class ("descriptor changed after connection"),
# so the driver runs on a real fitted KernelRidge. The internal caller's raw
# result is tested by test_parallel_pool_numpy_parity.py::
# test_native_kernel_ridge_driver_predict_is_the_plain_native_result.
def test_apply_kernel_method_matches_plain_predict_for_a_user():
    from mojolearn import _backend
    if _backend.vendor() in (None, "cpu"):
        pytest.skip("the multi-GPU kernel-methods driver needs a GPU device")
    import mojolearn as ml
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 3)).astype(np.float32)
    est = ml.KernelRidge(alpha=1.0, kernel="rbf").fit(X, rng.standard_normal(64).astype(np.float32))
    got = _call_in(_user_globals(), parallel_classical.apply_kernel_method, est, X)
    want = _call_in(_user_globals(), est.predict, X)
    assert type(got) is type(want) is np.ndarray
    assert got.dtype == want.dtype == np.float64
    assert got.tobytes() == want.tobytes()


def test_public_method_result_converts_only_public_output_names():
    from mojolearn.tests._parallel_pool import public_method_result
    frame = _frame_in(_user_globals())
    assert type(public_method_result(_Regressor, "solve", _raw(2), frame)) is Array
    out = public_method_result(_Regressor, "predict", _raw(2), frame)
    assert type(out) is np.ndarray and out.dtype == np.float64
