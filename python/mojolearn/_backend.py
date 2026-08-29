"""Which binaries this process runs: the numeric-mode selector.

THREE builds of every extension module can sit in the package, one per tier
of a LADDER in which each rung keeps the rung below it:

    python/mojolearn/_mojolearn*.so                   NUMERIC_FAST
    python/mojolearn/deterministic/_mojolearn*.so     NUMERIC_DETERMINISTIC
    python/mojolearn/identical/_mojolearn*.so         NUMERIC_IDENTICAL

    fast           no promise; speed only. The same fit on the same box may
                   return different bits on two runs, and on the histogram
                   lanes it measurably does.
    deterministic  same box, same build, same input -> the same bits, every
                   run. Says NOTHING about a second box.
    identical      all of the above, AND the same bits on Metal, CUDA and
                   HIP. A strict superset, which is why `PIN_DETERMINISM` in
                   `mojo_only/numerics.mojo` is true under both upper tiers.

`MOJOLEARN_NUMERIC_MODE=<tier>` in the environment AT IMPORT TIME makes
`mojolearn` load that set under the canonical module names, so every caller's
`from . import _mojolearn_gbdt` sees the right arithmetic (IDENTITY_PATHS.md;
E1/E2_RESULTS.md are the measurements). Unset or `fast` loads the default
set. Anything else raises: a mode that is accepted and ignored is worse than
one refused -- which is exactly what this selector did to `deterministic`
until 2026-08-29, when the tier existed in the compiler and was unreachable
from Python because this function's allow-list had two entries in it.

The mode is a BUILD DEFINE (`-D MOJOLEARN_NUMERIC_IDENTICAL=1`, read by
`mojo_only/numerics.mojo` through `is_defined`), and the identical binaries
come from `MOJOLEARN_NUMERIC_MODE=<tier> bash bindings/build_*.sh`. It
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


#: Tier name -> the code `<ext>_numeric_mode()` reports, which is the
#: `NUMERIC_*` constant in `mojo_only/numerics.mojo`. Keep the two in step:
#: this dict is how a binary in the wrong directory is caught.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}
_CODE_MODE = {v: k for k, v in _MODE_CODE.items()}


def requested_mode():
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    if mode not in _MODE_CODE:
        raise ImportError(
            f"mojolearn: MOJOLEARN_NUMERIC_MODE={mode!r}; it must be 'fast' "
            "(the default), 'deterministic' or 'identical'"
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
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == "fast":
        # FAST USED TO RETURN HERE, INSTALLING NOTHING, and that made it the
        # ONLY tier that cannot survive a partial build. An upper tier gets a
        # `_MissingUpperTier` stub for each binding that did not build, so the
        # package imports and the estimators that need that binding raise BY
        # NAME on use. Under fast there were no stubs, so `from . import
        # _mojolearn_trees` in extratrees.py raised at PACKAGE IMPORT and took
        # the whole library down.
        #
        # Measured on a rented RTX 4090, 2026-08-29: a leg that deliberately
        # built four of the ten bindings got tables from the deterministic and
        # identical arms -- three lanes REFUSED by name, the rest measured --
        # and from the fast arm got a traceback ending "cannot import name
        # '_mojolearn_trees' ... (most likely due to a circular import)",
        # which names the wrong cause and loses every lane that would have
        # worked. Same partial build, two entirely different outcomes,
        # decided by which tier was asked for.
        #
        # Present bindings are left to normal import: this installs a stub for
        # a MISSING one and touches nothing else.
        for name in _MODULES:
            if os.path.exists(os.path.join(pkg_dir, name + ".so")):
                continue
            full = f"{sys.modules[__name__.rsplit('.', 1)[0]].__name__}.{name}"
            if full in sys.modules:
                continue
            module = _MissingUpperTier(
                full, os.path.join(pkg_dir, name + ".so"),
                _build_script(name), "fast",
            )
            sys.modules[full] = module
            setattr(sys.modules[__name__.rsplit(".", 1)[0]], name, module)
            _MISSING.append(name)
        _SELECTED = "fast"
        return _SELECTED
    # The directory IS the mode name, for every tier above fast. Derived
    # rather than branched, so adding a fourth tier is one dict entry.
    ident_dir = os.path.join(pkg_dir, mode)
    pkg = sys.modules[__name__.rsplit(".", 1)[0]]
    missing = []
    for name in _MODULES:
        full = f"{pkg.__name__}.{name}"
        path = os.path.join(ident_dir, name + ".so")
        if not os.path.exists(path):
            # NEVER fall back to the FAST binary under an upper-tier name:
            # a wrong-mode module that imports is a mislabelled
            # measurement. Install a stub that raises BY NAME on use, so
            # the estimators that need this binding fail loudly and the
            # rest of the package (the tree families on an AMD box whose
            # linalg binding did not build, E2 round 2) keeps working.
            missing.append(name)
            module = _MissingUpperTier(full, path, _build_script(name), mode)
        else:
            loader = importlib.machinery.ExtensionFileLoader(full, path)
            spec = importlib.util.spec_from_loader(full, loader, origin=path)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
        sys.modules[full] = module
        setattr(pkg, name, module)
    if len(missing) == len(_MODULES):
        raise ImportError(
            f"mojolearn: MOJOLEARN_NUMERIC_MODE={mode} but no {mode} "
            f"binary exists under {ident_dir}. Build them with\n    "
            f"MOJOLEARN_NUMERIC_MODE={mode} bash bindings/build*.sh"
        )
    _SELECTED = mode
    _MISSING.extend(missing)
    return _SELECTED



# ===================================================================
# THE MODE AS A PARAMETER, NOT AN ENVIRONMENT VARIABLE
# ===================================================================
# `select()` above is the ORIGINAL mechanism and it is process-wide: it reads
# an environment variable ONCE, before the first estimator is imported, and
# rebinds `sys.modules` so every caller in the process gets one tier. That is
# a global, set outside the program, that cannot be changed afterwards and
# cannot differ between two estimators in one script.
#
# `load_set` is the mechanism underneath the parameter form. It loads a WHOLE
# TIER side by side with the others, under private dotted names, and hands
# back a namespace. It does not touch `sys.modules` under the canonical names
# and does not disturb whatever `select()` installed.
#
# **THAT THREE SETS CAN COEXIST IS MEASURED, NOT ASSUMED** (2026-08-29). Each
# `.so` carries its own Mojo runtime and opens its own device context, so
# "they will conflict" was the live risk and the reason the parameter form was
# not attempted earlier. All three were loaded into one process, then called
# INTERLEAVED -- fast, deterministic, identical, fast -- twice over, on one
# 256x4096 @ 4096x128 product on an Apple M4. Each returned its own
# arithmetic every time (fast and deterministic bit-identical to each other,
# which is correct because no determinism pin exists in the GEMM path;
# identical differing, which is the pinned profile), and a call after the
# identical set did not inherit its answer.
#
# THE PyInit SYMBOL IS WHY THE NAMES ARE DOTTED. CPython derives the init
# symbol it looks for from the LAST dotted component of the module name, so a
# flat name like `probe_fast__mojolearn_linalg` makes the loader hunt for
# `PyInit_probe_fast__mojolearn_linalg` and fail. The tail must stay the real
# module name; the prefix does the disambiguating.

_SETS = {}


class _ModeSet:
    """One tier's binaries, loaded together and addressed by attribute.

    `getattr` raises BY NAME for a binding this tier has not built, rather
    than falling back to another tier's -- the same rule `select()` follows,
    and for the same reason: a wrong-mode module that imports cleanly is a
    mislabelled measurement.
    """

    def __init__(self, mode, modules, missing):
        self.mode = mode
        self._modules = modules
        self.missing = missing

    def __getattr__(self, name):
        try:
            return self._modules[name]
        except KeyError:
            pass
        if name in self.missing:
            raise ImportError(
                f"mojolearn: numeric_mode={self.mode!r} needs "
                f"python/mojolearn/{'' if self.mode == 'fast' else self.mode + '/'}"
                f"{name}.so, which is not built. Build it with\n    "
                f"{'' if self.mode == 'fast' else 'MOJOLEARN_NUMERIC_MODE=' + self.mode + ' '}"
                f"bash bindings/{_build_script(name)}"
            )
        raise AttributeError(name)

    def __repr__(self):
        return f"<mojolearn binaries: {self.mode}>"


def load_set(mode):
    """Load (and cache) every binding for one tier, side by side with the
    others. The mechanism behind a per-call `numeric_mode=`."""
    mode = (mode or "fast").strip().lower()
    if mode not in _MODE_CODE:
        raise ValueError(
            f"mojolearn: numeric_mode={mode!r}; it must be 'fast' (the "
            "default), 'deterministic' or 'identical'"
        )
    if mode in _SETS:
        return _SETS[mode]
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    tier_dir = pkg_dir if mode == "fast" else os.path.join(pkg_dir, mode)
    modules, missing = {}, []
    for name in _MODULES:
        path = os.path.join(tier_dir, name + ".so")
        if not os.path.exists(path):
            missing.append(name)
            continue
        full = f"mojolearn._sets.{mode}.{name}"
        existing = sys.modules.get(full)
        if existing is not None:
            modules[name] = existing
            continue
        loader = importlib.machinery.ExtensionFileLoader(full, path)
        spec = importlib.util.spec_from_loader(full, loader, origin=path)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        sys.modules[full] = module
        modules[name] = module
    if not modules:
        raise ImportError(
            f"mojolearn: numeric_mode={mode!r} but no binary for that tier "
            f"exists under {tier_dir}. Build them with\n    "
            f"{'' if mode == 'fast' else 'MOJOLEARN_NUMERIC_MODE=' + mode + ' '}"
            "bash bindings/build*.sh"
        )
    # READ THE TIER BACK OUT OF THE BINARY, never trust the directory. A .so
    # in the wrong folder is the one failure this whole file exists to catch,
    # and it is cheaper to catch here than in a results table.
    gb = modules.get("_mojolearn_gbdt")
    if gb is not None and hasattr(gb, "gbdt_numeric_mode"):
        compiled = _CODE_MODE.get(gb.gbdt_numeric_mode(), "unknown")
        if compiled != mode:
            raise RuntimeError(
                f"mojolearn: {tier_dir}/_mojolearn_gbdt.so was compiled "
                f"{compiled} but sits in the {mode} directory; rebuild it"
            )
    _SETS[mode] = _ModeSet(mode, modules, missing)
    return _SETS[mode]


#: The tier used when a call names none. Starts at whatever the environment
#: selected, so existing scripts are unaffected, and is settable IN CODE.
_DEFAULT_MODE = None


def default_mode():
    global _DEFAULT_MODE
    if _DEFAULT_MODE is None:
        _DEFAULT_MODE = _SELECTED or requested_mode()
    return _DEFAULT_MODE


def set_default_mode(mode):
    """Choose the tier IN CODE, at runtime. Returns the previous value.

    Loading is eager and deliberate: a name that cannot be honoured must fail
    HERE, at the line that asked for it, not at some later fit that would
    otherwise silently run on the tier it was already holding.
    """
    global _DEFAULT_MODE
    mode = (mode or "fast").strip().lower()
    load_set(mode)
    prev = default_mode()
    _DEFAULT_MODE = mode
    return prev


def binding(name, mode=None):
    """The one accessor an estimator needs: give me `name`, in `mode`."""
    return getattr(load_set(mode or default_mode()), name)


_MISSING = []


class _MissingUpperTier(type(sys)):
    """Stands in for a deterministic or identical binary that is not built.
    Importing it succeeds (the package imports every binding at load);
    touching any attribute raises with the build command FOR THE TIER THAT
    WAS ASKED FOR -- it used to say "identical" whatever you asked for, which
    hands the operator a command that builds the wrong binary."""

    def __init__(self, full, path, script, mode):
        super().__init__(full)
        self.__missing_path = path
        self.__script = script
        self.__mode = mode

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        raise ImportError(
            f"mojolearn: MOJOLEARN_NUMERIC_MODE={self.__mode} but "
            f"{self.__missing_path} is not built; build it with\n    "
            f"MOJOLEARN_NUMERIC_MODE={self.__mode} bash "
            f"bindings/{self.__script}"
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
    """'fast', 'deterministic' or 'identical' -- what this process LOADED,
    cross-checked against the gbdt binary's own compile-time answer when it
    exposes one.

    The cross-check read `== 1 else "fast"` until 2026-08-29. A deterministic
    binary reports 2, so that spelling called it "fast" and AGREED with a
    selector that had loaded fast, reporting no conflict while the caller
    held the wrong arm."""
    loaded = _SELECTED or "fast"
    pkg = sys.modules[__name__.rsplit(".", 1)[0]]
    gb = getattr(pkg, "_mojolearn_gbdt", None)
    if gb is not None and hasattr(gb, "gbdt_numeric_mode"):
        compiled = _CODE_MODE.get(gb.gbdt_numeric_mode(), "unknown")
        if compiled != loaded:
            raise RuntimeError(
                f"mojolearn: the loaded gbdt binary was compiled {compiled} "
                f"but the selector loaded the {loaded} set -- a binary is in "
                "the wrong directory; rebuild both sets"
            )
    return loaded
