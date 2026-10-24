# SPDX-License-Identifier: Apache-2.0
"""Pool workers return raw native results; drivers convert for their caller.

THE par-kernel-ridge DTYPE FLIP (AMD identity record at e548c5b01,
2026-10-24): the worker runs as `python -m mojolearn._parallel_worker`, so its
module is `__main__` and `_mode._caller_outside` took it for a USER. Once the
worker loaded NumPy whenever the parent had it, `KernelRidge.predict` inside
the worker was widened to float64 (`_mode._public_result`) and pickled back,
while the identity harness's plain `predict` (an internal caller) is the
native float32 Array: "apply_kernel_method predict and plain predict differ:
3963 bytes of 4096" on all nine fixtures. The worker now marks itself
internal; the driver applies the public conversion for its own caller.
"""
import subprocess
import sys

import numpy as np

from mojolearn import _mode
from mojolearn import _parallel_worker
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


def test_worker_module_carries_the_internal_marker():
    assert _parallel_worker.__mojolearn_internal__ is True
    assert _mode._caller_outside(_frame_in(vars(_parallel_worker))) is False


def test_a_frame_with_the_worker_marker_is_internal_under_main():
    # `python -m` runs the module with `__name__ == '__main__'`; the marker is
    # in those same globals.
    globs = {"__name__": "__main__", "__mojolearn_internal__": True}
    assert _mode._caller_outside(_frame_in(globs)) is False
    assert _mode._caller_outside(_frame_in({"__name__": "__main__"})) is True


def test_worker_run_as_main_is_internal():
    # The real `-m` execution: run_module with run_name='__main__' executes the
    # worker's `main()`, which returns at EOF on an empty stdin; the returned
    # globals are the worker's `__main__` globals.
    code = ("import runpy, sys\n"
            "from mojolearn import _mode\n"
            "g = runpy.run_module('mojolearn._parallel_worker', run_name='__main__')\n"
            "assert g['__name__'] == '__main__'\n"
            "assert g.get('__mojolearn_internal__') is True\n"
            "exec('import sys\\ndef _here():\\n    return sys._getframe()\\n', g)\n"
            "sys.exit(3 if _mode._caller_outside(g['_here']()) else 0)\n")
    done = subprocess.run([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]


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


def test_public_method_seen_from_the_worker_is_raw():
    est = _Regressor()
    raw = _call_in(vars(_parallel_worker), est.predict, None)
    assert type(raw) is Array and raw.dtype == "<f4"
    public = _call_in(_user_globals(), est.predict, None)
    assert type(public) is np.ndarray and public.dtype == np.float64


class _FakePool:
    """The DevicePool a driver opens, answering `km_apply` with the worker's
    raw float32 Array (what the internal worker now always sends)."""
    def __init__(self, devices, cooperative=False):
        pass

    def map(self, requests):
        assert requests[0][0] == "km_apply"
        return [_raw(4)]

    def close(self):
        pass


def _patch_kernel_driver(monkeypatch):
    from mojolearn.kernel_methods import KernelRidge
    monkeypatch.setattr(parallel_classical, "DevicePool", _FakePool)
    monkeypatch.setattr(parallel_classical, "_admit_kernel_method", lambda est: None)
    monkeypatch.setattr(KernelRidge, "predict",
                        _mode._public_output(lambda self, X: _raw(4), "predict"))
    return KernelRidge()


def test_apply_kernel_method_matches_plain_predict_for_a_user(monkeypatch):
    est = _patch_kernel_driver(monkeypatch)
    X = np.zeros((4, 2), np.float32)
    got = _call_in(_user_globals(), parallel_classical.apply_kernel_method, est, X)
    want = _call_in(_user_globals(), est.predict, X)
    assert type(got) is type(want) is np.ndarray
    assert got.dtype == want.dtype == np.float64


def test_apply_kernel_method_is_raw_for_an_internal_caller(monkeypatch):
    # The identity harness marks itself `__mojolearn_internal__`.
    est = _patch_kernel_driver(monkeypatch)
    X = np.zeros((4, 2), np.float32)
    harness = {"__name__": "__main__", "__mojolearn_internal__": True}
    got = _call_in(harness, parallel_classical.apply_kernel_method, est, X)
    want = _call_in(harness, est.predict, X)
    assert type(got) is type(want) is Array
    assert got.dtype == want.dtype == "<f4"


def test_public_method_result_converts_only_public_output_names():
    from mojolearn._parallel_pool import public_method_result
    frame = _frame_in(_user_globals())
    assert type(public_method_result(_Regressor, "solve", _raw(2), frame)) is Array
    out = public_method_result(_Regressor, "predict", _raw(2), frame)
    assert type(out) is np.ndarray and out.dtype == np.float64
