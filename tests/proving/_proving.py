# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the proving suite (tests/proving).

The proving suite exercises an INSTALLED Mojolearn wheel at its Python boundary: lifetimes, threads, sessions,
digests, resident scopes and the model-selection member routes the 2026-11-21 plan wave changed. It never imports
the source tree's `python/` directory. Everything here is test tooling, not product execution.
"""
import importlib.util
import inspect
import os
import sys
import threading

import numpy as np
import pytest

mojolearn = pytest.importorskip("mojolearn")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def has_gpu():
    """True when the installed wheel has a GPU device (a CPU-only install reports vendor 'cpu')."""
    try:
        return mojolearn.vendor() != "cpu"
    except Exception:
        return False


def identical():
    return mojolearn.numeric_mode() == "identical"


#: The device word of each run: "cpu" always works; "gpu" (the default device) needs a GPU install.
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]


def cls(name):
    """The public class or function `name`, looked up on `mojolearn` and then on `mojolearn.model_selection`."""
    for owner in (mojolearn, getattr(mojolearn, "model_selection", None)):
        if owner is None:
            continue
        try:
            return getattr(owner, name)
        except AttributeError:
            continue
    pytest.fail(f"mojolearn has no public name {name!r}")


def _takes(klass, word):
    try:
        parameters = inspect.signature(klass.__init__).parameters
    except (TypeError, ValueError):
        return True     # a native constructor without a Python signature: pass the word and let it decide
    return word in parameters or any(p.kind is p.VAR_KEYWORD for p in parameters.values())


def make(name, device="cpu", **params):
    """`name(**params)` on `device`: "cpu" passes device="cpu", "gpu" leaves the default device. Parameters the
    class does not take (random_state on Ridge, ...) are dropped by name, never silently on a typo: only the
    words in OPTIONAL are dropped."""
    klass = cls(name)
    kept = {k: v for k, v in params.items() if k not in OPTIONAL or _takes(klass, k)}
    if device == "cpu" and _takes(klass, "device"):
        kept["device"] = "cpu"
    return klass(**kept)


#: Parameters that some estimators do not take; `make` drops them where absent.
OPTIONAL = {"random_state", "n_estimators", "max_iter", "max_depth", "max_epochs"}


def data(rows=160, features=6, seed=0):
    """Small float32 regression X, y and a three-class int32 label vector."""
    rng = np.random.default_rng(seed)
    X = rng.random((rows, features), dtype=np.float32)
    y = (X @ np.arange(1, features + 1, dtype=np.float32) + np.float32(0.1) * rng.random(rows, dtype=np.float32))
    edges = np.quantile(y, [1 / 3, 2 / 3])
    c = np.searchsorted(edges, y).astype(np.int32)
    return X, y.astype(np.float32), c


def binary(c):
    return (c > 0).astype(np.int32)


def plain(value):
    """`value` as a NumPy array (a Mojolearn Array or list of labels included), for comparisons."""
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return np.asarray(value, dtype=str)
    try:
        return np.asarray(value)
    except Exception:
        return np.asarray(memoryview(value))


def assert_same(a, b, what=""):
    """Bit equality of two results (NaN equal to NaN), dtype included."""
    a, b = plain(a), plain(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"
    if a.dtype.kind in "fc":
        assert np.array_equal(a, b, equal_nan=True), f"{what}: values differ"
    else:
        assert np.array_equal(a, b), f"{what}: values differ"


def in_threads(call, threads=2):
    """`call(i)` for i in range(threads), started together on Python threads (a barrier releases them at once).
    Returns the results in thread order; the first exception of any thread is raised here."""
    barrier = threading.Barrier(threads)
    results = [None] * threads
    errors = []

    def worker(i):
        try:
            barrier.wait()
            results[i] = call(i)
        except BaseException as error:  # re-raised in the caller below
            errors.append(error)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=600)
        assert not t.is_alive(), "a thread did not finish within 600 s (deadlock?)"
    if errors:
        raise errors[0]
    return results


def churn(megabytes=64):
    """Allocate and free memory so that a freed buffer still read through a dangling pointer gets new contents."""
    import gc

    gc.collect()
    junk = [np.full(1 << 18, np.float32(7.5)) for _ in range(megabytes)]
    del junk
    gc.collect()


def reexport(namespace, path, module_name):
    """Load the existing test module at `path` (repo-relative) and copy its tests, fixtures and `pytestmark` into
    `namespace`, so the proving suite runs them without a second copy of their code."""
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        pytest.fail(f"proving suite: reused test module {path} is missing")
    spec = importlib.util.spec_from_file_location(module_name, full)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _copy_tests(namespace, module)
    return module


def _copy_tests(namespace, module):
    """Copy `module`'s names into `namespace`; its module-level `pytestmark` is applied to its own test functions
    only (never to the tests of the proving file that reuses them)."""
    marks = getattr(module, "pytestmark", [])
    marks = marks if isinstance(marks, list) else [marks]
    for key, value in vars(module).items():
        if key.startswith("__") or key == "pytestmark":
            continue
        if key.startswith("test_") and callable(value):
            for mark in marks:
                value = mark(value)
        namespace.setdefault(key, value)


def _mount_source_tests():
    """Mount the source tree's python/mojolearn/tests as `mojolearn.tests` over the INSTALLED runtime, as
    tools/run_python_tests.py INSTALLED_BOOTSTRAP does (the wheel does not ship the tests package)."""
    if "mojolearn.tests" in sys.modules:
        return
    root = os.path.join(REPO, "python", "mojolearn", "tests")
    runtime = os.path.realpath(mojolearn.__file__)
    if runtime.startswith(os.path.realpath(os.path.join(REPO, "python")) + os.sep):
        pytest.fail("proving suite: mojolearn is imported from the source tree, not from the installed wheel")
    spec = importlib.util.spec_from_file_location("mojolearn.tests", os.path.join(root, "__init__.py"),
                                                  submodule_search_locations=[root])
    tests = importlib.util.module_from_spec(spec)
    sys.modules["mojolearn.tests"] = tests
    mojolearn.tests = tests
    spec.loader.exec_module(tests)


def reexport_installed(namespace, module_name):
    """Same as `reexport`, for a module of the source tree's `mojolearn.tests` package, run against the installed
    runtime (`mojolearn.tests.<name>`)."""
    _mount_source_tests()
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(f"proving suite: {module_name} cannot be imported ({error})")
    _copy_tests(namespace, module)
    return module


def workers_for(units, memory_per_worker_gb=1.5):
    """Worker processes for `units` independent units: every core, limited by the memory a worker needs
    (one interpreter with Mojolearn imported) and by MOJOLEARN_PROVING_WORKERS when set."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    limit = cores
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        limit = min(limit, max(1, int(available / (memory_per_worker_gb * (1 << 30)))))
    except (ValueError, OSError, AttributeError):
        pass
    if os.environ.get("MOJOLEARN_PROVING_WORKERS"):
        limit = int(os.environ["MOJOLEARN_PROVING_WORKERS"])
    return max(1, min(limit, units))
