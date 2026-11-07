"""Environment probes shared by conftest.py and test modules (see conftest.py).

Each probe returns None when the environment can run the test, else the exact
reason it cannot. Standard library only at import time.
"""
import functools
import importlib
import os
import subprocess
import sys
import tempfile

import pytest

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=None)
def _fast_tier():
    try:
        from mojolearn import _backend, _merged
    except Exception as error:  # the import failure is the reason
        return f"mojolearn did not import: {type(error).__name__}: {error}"
    directory = _backend.tier_dir("fast")
    names = os.listdir(directory) if os.path.isdir(directory) else []
    if any(name.startswith("_mojolearn") and name.endswith((".so", ".dylib", ".pyd")) for name in names):
        return None
    if os.path.exists(os.path.join(directory, _merged.INDEX_NAME)):
        return None
    return f"no numeric_mode='fast' binary set under {directory} (a development build has none)"


@functools.lru_cache(maxsize=None)
def _identity_harness():
    if os.path.exists(os.path.join(_PACKAGE_DIR, "_identity_break.py")):
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


ENVIRONMENT = {
    "needs_fast_tier": _fast_tier,
    "needs_identity_harness": _identity_harness,
    "needs_numpy_free_python": _numpy_free_python,
    "needs_installed_package": _installed_package,
}



def identity_harness_or_skip():
    """Import `mojolearn._identity_break` for a module that needs it at import
    time, or skip the whole module with the reason (a module-level import
    error would be a collection error, not a skip)."""
    reason = _identity_harness()
    if reason is not None:
        pytest.skip(f"needs_identity_harness: {reason}", allow_module_level=True)
    return importlib.import_module("mojolearn._identity_break")
