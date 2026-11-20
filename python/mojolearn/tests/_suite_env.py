"""Environment probes shared by conftest.py and test modules (see conftest.py).

Each probe returns None when the environment can run the test, else the exact
reason it cannot. Standard library only at import time.
"""
import functools
import os
import subprocess
import sys
import tempfile

import pytest

@functools.lru_cache(maxsize=None)
def _fast_tier():
    try:
        from mojolearn import _backend, _merged
    except Exception as error:  # the import failure is the reason
        return f"mojolearn did not import: {type(error).__name__}: {error}"
    try:
        directory = _backend.tier_dir("fast")
    except ImportError as error:  # a CPU-only install has no GPU tier layout at all: the refusal is the reason
        return f"no numeric_mode='fast' tier on this install: {type(error).__name__}: {error}"
    names = os.listdir(directory) if os.path.isdir(directory) else []
    if any(name.startswith("_mojolearn") and name.endswith((".so", ".dylib", ".pyd")) for name in names):
        return None
    if os.path.exists(os.path.join(directory, _merged.INDEX_NAME)):
        return None
    return f"no numeric_mode='fast' binary set under {directory} (a development build has none)"


@functools.lru_cache(maxsize=None)
def _identity_harness():
    # The IMPORTED package, not the directory above this file: a suite run
    # from a source checkout against the installed wheel must see the
    # wheel's copy (the old form read the checkout and always skipped).
    try:
        import mojolearn
    except Exception as error:  # the import failure is the reason
        return f"mojolearn did not import: {type(error).__name__}: {error}"
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(mojolearn.__file__)), "_identity_break.py")):
        return None
    return ("mojolearn._identity_break is absent: the wheel build copies tools/identity_break.py "
            "into the package; a checkout does not carry it")


@functools.lru_cache(maxsize=None)
def _numpy_free_python():
    if os.environ.get("MOJOLEARN_NUMPY_FREE_PYTHON", "").strip():
        return None
    return "MOJOLEARN_NUMPY_FREE_PYTHON is unset (it names an installed interpreter without NumPy)"


@functools.lru_cache(maxsize=None)
def _installed_package():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    probe = "import importlib.util, sys; sys.exit(importlib.util.find_spec('mojolearn') is None)"
    try:
        done = subprocess.run([sys.executable, "-c", probe], cwd=tempfile.gettempdir(),
                              env=env, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return f"could not probe a child interpreter: {error}"
    if done.returncode == 0:
        return None
    return ("mojolearn is not importable in a child without PYTHONPATH: install it into this environment "
            "(pip install -e python) to run this test")


@functools.lru_cache(maxsize=None)
def _gpu():
    try:
        import mojolearn
        vendor = mojolearn.vendor()
    except Exception as error:  # the import or read-back failure is the reason
        return f"could not read the vendor: {type(error).__name__}: {error}"
    if vendor in ("metal", "cuda", "hip"):
        return None
    return f"no GPU binary set on this install (vendor() is {vendor!r}); the test reads a GPU vendor"


ENVIRONMENT = {
    "needs_fast_tier": _fast_tier,
    "needs_identity_harness": _identity_harness,
    "needs_numpy_free_python": _numpy_free_python,
    "needs_installed_package": _installed_package,
    "needs_gpu": _gpu,
}
