"""Which binaries this process runs: the numeric-mode selector.

Two builds of every extension module can sit in the package:

    python/mojolearn/_mojolearn*.so              NUMERIC_FAST (the default)
    python/mojolearn/identical/_mojolearn*.so    NUMERIC_IDENTICAL

`MOJOLEARN_NUMERIC_MODE=identical` in the environment AT IMPORT TIME makes
`mojolearn` load the identical set under the same module names, so every
caller's `from . import _mojolearn_gbdt` sees the bits-the-same-everywhere
arithmetic (IDENTITY_PATHS.md; E1/E2_RESULTS.md are the measurements).
Unset or `fast` loads the default set. Anything else raises: a mode that is
accepted and ignored is worse than one refused.

The mode is a BUILD DEFINE (`-D MOJOLEARN_NUMERIC_IDENTICAL=1`, read by
`mojo_only/numerics.mojo` through `is_defined`), and the identical binaries
come from `MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_*.sh`. It
used to be a line in numerics.mojo flipped by sed and rebuilt in place, which
is fine for one lab session and wrong for a product (and for two sessions
sharing one checkout: an edit made during a flip window was lost on
2026-08-23). `numeric_mode()` reports what was actually loaded, read back
from the binary where it can be (`gbdt_numeric_mode`), so a wrong-arm
measurement is impossible to label correctly by accident.
"""

import importlib.machinery
import importlib.util
import os
import sys

# DEVIATION 869, 2026-08-24. THIS TUPLE AND `_build_script` BELOW MUST LIST
# EVERY EXTENSION, AND THE COST OF FORGETTING ONE IS A MISLABELLED
# MEASUREMENT RATHER THAN A FAILURE.
#
# `select()` only installs the identical binary for names IT KNOWS. An
# extension absent from this tuple is never re-pointed, so under
# MOJOLEARN_NUMERIC_MODE=identical a plain `from . import _mojolearn_x`
# resolves to the FAST binary sitting beside it and returns the fast
# arithmetic under the identical label. That is the exact failure this
# module's docstring says is impossible to make by accident, and it was
# possible for five extensions at once until this edit.
#
# Five bindings landed on 2026-08-24 (svm/isolation-forest, solver/hierarchy,
# metrics/spectral, holtwinters/tsa, and the linalg GEMM surface). FOUR of
# their authors independently found this tuple stale and each wrote a private
# mode-aware loader to work around it. Those workarounds are now dead code
# and their authors marked them for deletion; delete them when convenient.
#
# When you add a binding, add it in BOTH places or the build will work and
# the numbers will be quietly wrong.
_MODULES = (
    "_mojolearn",
    "_mojolearn_estimators",
    "_mojolearn_gbdt",
    "_mojolearn_rf",
    "_mojolearn_trees",
    "_mojolearn_svm",
    "_mojolearn_solver",
    "_mojolearn_metrics",
    "_mojolearn_tsa",
    "_mojolearn_linalg",
)
_SELECTED = None


def requested_mode():
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    if mode not in ("fast", "identical"):
        raise ImportError(
            f"mojolearn: MOJOLEARN_NUMERIC_MODE={mode!r}; it must be 'fast' "
            "(the default) or 'identical'"
        )
    return mode


def select():
    """Install the requested binary set under the canonical module names.
    Called once from `mojolearn/__init__.py` before any submodule imports a
    binding. Idempotent."""
    global _SELECTED
    if _SELECTED is not None:
        return _SELECTED
    mode = requested_mode()
    if mode == "fast":
        _SELECTED = "fast"
        return _SELECTED
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    ident_dir = os.path.join(pkg_dir, "identical")
    pkg = sys.modules[__name__.rsplit(".", 1)[0]]
    missing = []
    for name in _MODULES:
        full = f"{pkg.__name__}.{name}"
        path = os.path.join(ident_dir, name + ".so")
        if not os.path.exists(path):
            # NEVER fall back to the FAST binary under the identical name:
            # a wrong-mode module that imports is a mislabelled
            # measurement. Install a stub that raises BY NAME on use, so
            # the estimators that need this binding fail loudly and the
            # rest of the package (the tree families on an AMD box whose
            # linalg binding did not build, E2 round 2) keeps working.
            missing.append(name)
            module = _MissingIdentical(full, path, _build_script(name))
        else:
            loader = importlib.machinery.ExtensionFileLoader(full, path)
            spec = importlib.util.spec_from_loader(full, loader, origin=path)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
        sys.modules[full] = module
        setattr(pkg, name, module)
    if len(missing) == len(_MODULES):
        raise ImportError(
            "mojolearn: MOJOLEARN_NUMERIC_MODE=identical but no identical "
            f"binary exists under {ident_dir}. Build them with\n    "
            "MOJOLEARN_NUMERIC_MODE=identical bash bindings/build*.sh"
        )
    _SELECTED = "identical"
    _MISSING.extend(missing)
    return _SELECTED


_MISSING = []


class _MissingIdentical(type(sys)):
    """Stands in for an identical binary that is not built. Importing it
    succeeds (the package imports every binding at load); touching any
    attribute raises with the build command."""

    def __init__(self, full, path, script):
        super().__init__(full)
        self.__missing_path = path
        self.__script = script

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        raise ImportError(
            f"mojolearn: MOJOLEARN_NUMERIC_MODE=identical but "
            f"{self.__missing_path} is not built; build it with\n    "
            f"MOJOLEARN_NUMERIC_MODE=identical bash bindings/{self.__script}"
        )


def _build_script(name):
    # `.get` with a derived fallback, not `[name]`. A KeyError here would
    # fire from inside the MISSING-binary path, replacing a clear "build it
    # with this command" message with a traceback about a dict, at exactly
    # the moment the caller most needs to be told what to run.
    return {
        "_mojolearn": "build.sh",
        "_mojolearn_estimators": "build_estimators.sh",
        "_mojolearn_gbdt": "build_gbdt.sh",
        "_mojolearn_rf": "build_rf.sh",
        "_mojolearn_trees": "build_trees.sh",
        "_mojolearn_svm": "build_svm.sh",
        "_mojolearn_solver": "build_solver.sh",
        "_mojolearn_metrics": "build_metrics.sh",
        "_mojolearn_tsa": "build_tsa.sh",
        "_mojolearn_linalg": "build_linalg.sh",
    }.get(name, "build" + name[len("_mojolearn"):] + ".sh")


def numeric_mode():
    """'fast' or 'identical' -- what this process LOADED, cross-checked
    against the gbdt binary's own compile-time answer when it exposes one."""
    loaded = _SELECTED or "fast"
    pkg = sys.modules[__name__.rsplit(".", 1)[0]]
    gb = getattr(pkg, "_mojolearn_gbdt", None)
    if gb is not None and hasattr(gb, "gbdt_numeric_mode"):
        compiled = "identical" if gb.gbdt_numeric_mode() == 1 else "fast"
        if compiled != loaded:
            raise RuntimeError(
                f"mojolearn: the loaded gbdt binary was compiled {compiled} "
                f"but the selector loaded the {loaded} set -- a binary is in "
                "the wrong directory; rebuild both sets"
            )
    return loaded
