"""The default scorer never kills the interpreter, on every supported Python (CPU-only install included).

A downstream CI ran one test that used the default scorer and lost the process under Python 3.10: "illegal
instruction" on Linux, "trace trap" on macOS. Reproduced on a GitHub Linux runner with the published 0.8.66 wheel
(fx-user-crash-scorer, 2026-11-19): `ABORT: std/ffi: symbol not found: PyObject_CallOneArg`. Before CPython 3.11
that function is a header inline, not an exported symbol, so the binding's lookup missed and Mojo's ffi aborted
(ud2 = SIGILL on x86, brk = SIGTRAP on arm64). Each case runs in a fresh interpreter under `-X faulthandler` and
must exit normally: a negative return code (a signal) or an `ABORT:` line is the regression. Python exceptions are
fine here; what is checked is that the process survives `import mojolearn`, `fit`, `score` and
`cross_val_score` with the estimator's own scorer, and that the resident device records (whose capsule destructor
called the missing function) are dropped without killing the process.
"""
import builtins
import io

import pytest

_CASES = {
    "logistic": "from mojolearn.linear_model import LogisticRegression as E; y = yc",
    "ridge": "from mojolearn.linear_model import Ridge as E; y = yr",
    "linear": "from mojolearn.linear_model import LinearRegression as E; y = yr",
    "forest": "from mojolearn.randomforest import RandomForestClassifier as E; y = yc",
    "boosting": "from mojolearn.ensemble import GradientBoostingRegressor as E; y = yr",
    "neighbors": "from mojolearn.neighbors import KNeighborsClassifier as E; y = yc",
    "mixture": "from mojolearn.mixture import GaussianMixture as E; y = None",
}

_BODY = r"""
import gc, sys
import numpy as np
rng = np.random.default_rng(0)
X = rng.standard_normal((200, 5)).astype(np.float32)
yc = (X[:, 0] > 0).astype(np.int64)
yr = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
{case}
est = E()
if y is None:
    est.fit(X); value = est.score(X)
else:
    est.fit(X, y); value = est.score(X, y)
    from mojolearn.model_selection import cross_val_score
    scores = cross_val_score(E(), X, y, cv=3)
    assert len(scores) == 3, scores
del est
gc.collect()
print("SCORED", float(value))
"""


@pytest.mark.parametrize("name", sorted(_CASES))
def test_default_scorer_process_survives(name):
    from mojolearn.tests._fresh_process import child_env
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-X", "faulthandler", "-c", _BODY.format(case=_CASES[name])],
                          env=child_env(), capture_output=True, text=True, timeout=600)
    tail = (done.stdout + done.stderr)[-3000:]
    assert done.returncode >= 0, f"{name}: the interpreter died by signal {-done.returncode}:\n{tail}"
    assert "ABORT:" not in done.stderr, f"{name}: the native runtime aborted:\n{tail}"
    assert done.returncode == 0, f"{name}: rc={done.returncode}\n{tail}"
    assert "SCORED" in done.stdout, tail


def _guard_with(monkeypatch, flags_line, machine="x86_64", platform="linux"):
    from mojolearn import _host_guard
    import os
    import sys

    class _Uname:
        pass

    uname = _Uname()
    uname.machine = machine
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            return io.StringIO("processor\t: 0\n" + flags_line + "\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "uname", lambda: uname)
    monkeypatch.setattr(builtins, "open", fake_open)
    return _host_guard


def test_host_guard_refuses_a_cpu_below_x86_64_v3(monkeypatch):
    guard = _guard_with(monkeypatch, "flags\t\t: fpu sse sse2 pni ssse3 cx16 sse4_1 sse4_2 popcnt xsave avx")
    with pytest.raises(ImportError, match=r"x86-64-v3.*avx2"):
        guard.check()


def test_host_guard_admits_an_x86_64_v3_cpu(monkeypatch):
    guard = _guard_with(monkeypatch, "flags\t\t: fpu sse sse2 pni ssse3 cx16 sse4_1 sse4_2 popcnt xsave avx avx2 "
                                     "bmi1 bmi2 f16c fma abm movbe")
    guard.check()


def test_host_guard_ignores_other_platforms(monkeypatch):
    guard = _guard_with(monkeypatch, "flags\t\t: fpu", machine="arm64", platform="darwin")
    guard.check()
