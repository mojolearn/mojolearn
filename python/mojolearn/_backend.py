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

THE VENDOR AXIS (2026-08-29, docs/LINUX_WHEEL.md)
-------------------------------------------------
The tier ladder above is one axis. The Linux wheel adds a second: ONE PyPI
name carries a CUDA set and a HIP set, each in all three tiers, and the
vendor is picked AT IMPORT. The layout is one directory per accelerator API,
and the tier layout above repeats INSIDE it unchanged:

    macOS (unchanged)          python/mojolearn/{,deterministic,identical}/*.so
    Linux                      python/mojolearn/cuda/{,deterministic,identical}/*.so
                               python/mojolearn/hip/{,deterministic,identical}/*.so
    a source checkout on a     python/mojolearn/{,deterministic,identical}/*.so
    Linux box (every E1 leg)   ("flat": whatever bindings/build_*.sh wrote)

`_layout()` decides which of the three this install is by LOOKING AT THE
DISK, not at the platform: a vendor directory with binaries in it means the
wheel layout, otherwise the flat one. The flat case is what every rented leg
has ever built and it keeps working exactly as before.

THE ORDER OF TRUST, most to least:

  1. WHAT THE BINARY SAYS. Every binding exports `<prefix>_vendor()`, a
     compile-time constant (`mojo_only/vendor.mojo`): 'metal', 'cuda',
     'hip' or 'none'. After a set is loaded, EVERY module in it is asked,
     and one that disagrees with the directory it was loaded from is
     refused at import, the same refusal as a tier mismatch. A CUDA `.so`
     filed under `hip/` imports cleanly on an NVIDIA box and does not
     touch the device until the first fit; this is the only place that
     catches it.
  2. `MOJOLEARN_VENDOR` in the environment, which picks the DIRECTORY and
     nothing else. It cannot make a `hip/` binary say 'cuda'; it can only
     make the selector open `hip/` on a box that has no AMD device, and
     that fails at the first device call with the runtime's own error.
  3. WHAT THE BOX APPEARS TO HAVE (`_probe_box`): the device nodes and the
     driver libraries each API needs, checked with `os.path.exists` and
     `ctypes.CDLL`. This picks the directory when the environment did not.
     It is evidence about the box, and it is deliberately the LAST word,
     not the first: a probe can be fooled by a container that mounts a
     driver it cannot use, and the binary cannot be.

THERE IS NO CPU PATH, so when the wheel layout is present and no vendor can
be picked the import RAISES, naming every path and library it looked for
and what it found, rather than importing a package whose every fit would
fail. `vendor()` reports what was picked, cross-checked against the loaded
binaries; `NumericModeMixin.vendor_used()` reports it per estimator.
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


# ===================================================================
# THE VENDOR AXIS. See the module docstring, "THE VENDOR AXIS".
# ===================================================================

#: The accelerator APIs a set can be compiled for, in the order the box
#: probe consults them. These are the DIRECTORY names under the package on
#: Linux and the strings `<prefix>_vendor()` returns.
_VENDORS = ("cuda", "hip", "metal")
#: The two that can share one Linux wheel. `metal` is never a directory: the
#: macOS wheel keeps the flat layout.
_LINUX_VENDORS = ("cuda", "hip")

#: What `_probe_box` looks for, per API. Every entry is checked and every
#: result is reported, so the no-GPU refusal can say exactly what was looked
#: for and what was found. Paths are the device nodes the driver creates;
#: libraries are the ones the MAX runtime dlopens to reach the device
#: (the CUDA driver API and the HIP runtime). Versioned sonames first, then
#: the bare name, because a driver install ships the former and a dev
#: install adds the latter.
_PROBE = {
    "cuda": {
        "paths": ("/dev/nvidiactl", "/dev/nvidia0"),
        "libs": ("libcuda.so.1", "libcuda.so"),
    },
    "hip": {
        "paths": ("/dev/kfd", "/dev/dri/renderD128"),
        "libs": ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"),
    },
}


def _vendor_fn(name):
    """The read-back function each binding exports: `mojolearn_vendor` on
    `_mojolearn`, `<suffix>_vendor` on every `_mojolearn_<suffix>`."""
    if name == "_mojolearn":
        return "mojolearn_vendor"
    return name[len("_mojolearn_"):] + "_vendor"


def read_vendor(module):
    """What `module` says it was compiled for, or None when it predates the
    read-back (a binary built before 2026-08-29). A stub raises by name on
    any attribute, and `hasattr` does not swallow ImportError, so the probe
    is guarded the way `numeric_mode()` guards its own."""
    fn = _vendor_fn(module.__name__.rsplit(".", 1)[-1])
    try:
        f = getattr(module, fn, None)
    except ImportError:
        return None
    if f is None:
        return None
    return str(f())


_VENDOR_SELECTED = None
_VENDOR_HOW = None
_LAYOUT = None


def _pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _has_binaries(d):
    try:
        return any(n.endswith(".so") for n in os.listdir(d))
    except OSError:
        return False


def _probe_box():
    """Evidence, per Linux vendor, that this box can reach its device.

    Returns {vendor: {"paths": {path: bool}, "libs": {lib: bool},
    "found": bool}}. `found` is True when ANY device node or ANY library
    resolved. Every lookup is recorded so the refusal below can print the
    whole table rather than a verdict."""
    import ctypes
    out = {}
    for v, spec in _PROBE.items():
        paths = {p: os.path.exists(p) for p in spec["paths"]}
        libs = {}
        for lib in spec["libs"]:
            try:
                ctypes.CDLL(lib)
                libs[lib] = True
            except OSError:
                libs[lib] = False
        out[v] = {
            "paths": paths, "libs": libs,
            "found": any(paths.values()) or any(libs.values()),
        }
    return out


def _probe_lines(probe):
    lines = []
    for v in _LINUX_VENDORS:
        r = probe[v]
        lines.append(f"  {v}:")
        for p, ok in r["paths"].items():
            lines.append(f"    {p:<28} {'FOUND' if ok else 'absent'}")
        for lib, ok in r["libs"].items():
            lines.append(f"    {lib:<28} {'loads' if ok else 'not loadable'}")
    return lines


def _layout():
    """('flat', <pkg dir>) or ('vendor', <pkg dir>/<vendor>), decided ONCE.

    The wheel layout is recognised by a vendor directory WITH BINARIES IN
    IT. A bare directory does not count: the macOS wheel never has one, a
    source checkout never has one, and an empty one left by a failed build
    must not turn a working flat install into a vendor lookup."""
    global _LAYOUT, _VENDOR_SELECTED, _VENDOR_HOW
    if _LAYOUT is not None:
        return _LAYOUT
    pkg = _pkg_dir()
    present = [v for v in _LINUX_VENDORS
               if _has_binaries(os.path.join(pkg, v))]
    if not present:
        # macOS, or a Linux source checkout. The vendor is whatever the
        # binaries say; `vendor()` reads it after `select()` has loaded them.
        _LAYOUT = ("flat", pkg)
        _VENDOR_HOW = "flat layout; read from the loaded binaries"
        return _LAYOUT
    forced = os.environ.get("MOJOLEARN_VENDOR", "").strip().lower()
    if forced:
        if forced not in _LINUX_VENDORS:
            raise ImportError(
                f"mojolearn: MOJOLEARN_VENDOR={forced!r}; it must be "
                f"'cuda' or 'hip' (this install carries {present})"
            )
        if forced not in present:
            raise ImportError(
                f"mojolearn: MOJOLEARN_VENDOR={forced} but this install "
                f"carries no {forced} set under {os.path.join(pkg, forced)}; "
                f"it carries {present}"
            )
        _VENDOR_SELECTED = forced
        _VENDOR_HOW = "MOJOLEARN_VENDOR in the environment"
        _LAYOUT = ("vendor", os.path.join(pkg, forced))
        return _LAYOUT
    probe = _probe_box()
    hits = [v for v in present if probe[v]["found"]]
    if len(hits) == 1:
        _VENDOR_SELECTED = hits[0]
        _VENDOR_HOW = "the box probe (device nodes and driver libraries)"
        _LAYOUT = ("vendor", os.path.join(pkg, hits[0]))
        return _LAYOUT
    if len(hits) > 1:
        raise ImportError(
            "mojolearn: this box shows evidence of MORE THAN ONE supported "
            f"GPU API ({hits}) and this install carries a set for each. "
            "Choose one with MOJOLEARN_VENDOR=cuda or MOJOLEARN_VENDOR=hip "
            "before import. What was looked for and found:\n"
            + "\n".join(_probe_lines(probe))
        )
    raise ImportError(
        "mojolearn: NO SUPPORTED GPU FOUND ON THIS BOX, and there is no CPU "
        "path in this package. This install carries binary sets for "
        f"{present} under {pkg}. What was looked for and what was found:\n"
        + "\n".join(_probe_lines(probe))
        + "\n  MOJOLEARN_VENDOR is not set."
        "\n\nA device node or a driver library for one of the sets above must "
        "be visible to this process. In a container that means the GPU is "
        "passed through (`--gpus all` for NVIDIA, `--device /dev/kfd "
        "--device /dev/dri` for AMD). If the device is present and this "
        "probe is wrong, MOJOLEARN_VENDOR=cuda or MOJOLEARN_VENDOR=hip picks "
        "the directory directly and the first fit reports the runtime's own "
        "error."
    )


def tier_dir(mode):
    """The directory one tier's binaries live in, on this install and this
    vendor. `fast` is the vendor directory itself (the package directory on
    the flat layout); every other tier is one directory down under its own
    name. THE ONE PLACE THIS IS COMPUTED: the four bindings with private
    loaders (`_linalg_impl`, `_metrics_impl`, `_tsa_impl`, `_svm_impl`) call
    this rather than joining paths themselves."""
    _, base = _layout()
    if mode == "fast":
        return base
    return os.path.join(base, mode)


def _check_vendor(module, name, path):
    """Refuse a binary whose compiled vendor disagrees with the directory it
    was loaded from. Binaries that predate the read-back are let through
    with None, which `vendor()` reports as such rather than inventing an
    answer."""
    global _VENDOR_SELECTED
    kind, base = _layout()
    said = read_vendor(module)
    if said is None:
        return None
    if said == "none":
        raise ImportError(
            f"mojolearn: {path} was compiled with NO accelerator target "
            "(its vendor read-back says 'none'); it cannot run a kernel "
            "anywhere. Rebuild it on a box with the GPU present."
        )
    if kind == "vendor":
        expected = os.path.basename(base)
        if said != expected:
            raise ImportError(
                f"mojolearn: {path} was compiled for {said} but sits in the "
                f"{expected} set ({base}); a binary is in the wrong vendor "
                "directory. The set is refused rather than loaded under a "
                "label it does not answer to. Rebuild the sets with "
                "packaging/linux/build_sets.sh on the right box and repack."
            )
    else:
        # Flat layout: the first binary to answer decides, and every later
        # one must agree with it. Two vendors' binaries in one flat
        # directory is a build that went wrong, not a choice.
        if _VENDOR_SELECTED is None:
            _VENDOR_SELECTED = said
        elif said != _VENDOR_SELECTED:
            raise ImportError(
                f"mojolearn: {path} was compiled for {said} but the other "
                f"binaries in {base} were compiled for {_VENDOR_SELECTED}; "
                "a flat layout holds one vendor. Rebuild."
            )
    return said


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
    pkg_dir = _pkg_dir()
    pkg = sys.modules[__name__.rsplit(".", 1)[0]]
    # THE DIRECTORY COMES FROM tier_dir(), which folds in the vendor axis:
    # the package directory on macOS and on a flat Linux checkout, and
    # python/mojolearn/<vendor>/ on the Linux wheel. `_layout()` raises here,
    # at import, when the wheel layout is present and no vendor can be
    # picked; that is the no-GPU refusal and it is deliberate.
    ident_dir = tier_dir(mode)
    if mode == "fast" and ident_dir == pkg_dir:
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
        # a MISSING one and touches nothing else. The vendor of the present
        # ones is read back lazily by `vendor()`, because on this layout the
        # binaries are imported by the estimator modules, not here.
        for name in _MODULES:
            if os.path.exists(os.path.join(pkg_dir, name + ".so")):
                continue
            full = f"{pkg.__name__}.{name}"
            if full in sys.modules:
                continue
            module = _MissingUpperTier(
                full, os.path.join(pkg_dir, name + ".so"),
                _build_script(name), "fast",
            )
            sys.modules[full] = module
            setattr(pkg, name, module)
            _MISSING.append(name)
        _SELECTED = "fast"
        return _SELECTED
    # Every tier above fast, AND fast on the Linux wheel layout, where the
    # binaries sit under python/mojolearn/<vendor>/ and a plain
    # `from . import _mojolearn_x` would not find them. Explicit load,
    # installed under the canonical names.
    missing = []
    for name in _MODULES:
        full = f"{pkg.__name__}.{name}"
        path = os.path.join(ident_dir, name + ".so")
        if not os.path.exists(path):
            # NEVER fall back to the FAST binary under an upper-tier name,
            # and NEVER fall back to the OTHER VENDOR'S binary under this
            # one's: a wrong-mode or wrong-vendor module that imports is a
            # mislabelled measurement. Install a stub that raises BY NAME on
            # use, so the estimators that need this binding fail loudly and
            # the rest of the package (the tree families on an AMD box whose
            # linalg binding did not build, E2 round 2) keeps working.
            missing.append(name)
            module = _MissingUpperTier(full, path, _build_script(name), mode)
        else:
            loader = importlib.machinery.ExtensionFileLoader(full, path)
            spec = importlib.util.spec_from_loader(full, loader, origin=path)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            # WHAT THE BINARY SAYS BEATS THE DIRECTORY IT SAT IN. Raises on
            # a vendor mismatch; see the module docstring.
            _check_vendor(module, name, path)
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
    # The vendor axis is folded in by tier_dir(): the same `<vendor>/` root
    # `select()` used, so a per-call `numeric_mode=` can never reach across
    # to the other vendor's set.
    tier_dir_ = tier_dir(mode)
    modules, missing = {}, []
    for name in _MODULES:
        path = os.path.join(tier_dir_, name + ".so")
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
        # WHAT THE BINARY SAYS BEATS THE DIRECTORY IT SAT IN.
        _check_vendor(module, name, path)
        sys.modules[full] = module
        modules[name] = module
    if not modules:
        raise ImportError(
            f"mojolearn: numeric_mode={mode!r} but no binary for that tier "
            f"exists under {tier_dir_}. Build them with\n    "
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
                f"mojolearn: {tier_dir_}/_mojolearn_gbdt.so was compiled "
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
    held the wrong arm.

    IT REPORTS THE CURRENT DEFAULT, NOT THE IMPORT-TIME ONE. It returned
    `_SELECTED` -- what `select()` loaded before the first estimator existed
    -- until 2026-08-29, which meant that after

        mojolearn.set_numeric_mode("deterministic")

    this function still answered "fast" while every estimator built after
    that line ran deterministic. A function whose whole purpose is that "a
    run cannot be mislabeled by accident" was the one thing mislabeling it.
    `numeric_mode_used()` on an estimator instance was already right, so the
    two disagreed. Found by running the pair, not by reading them."""
    loaded = default_mode()
    pkg = sys.modules[__name__.rsplit(".", 1)[0]]
    # The cross-check has to read the binary of the tier being REPORTED. The
    # package attributes hold whatever `select()` bound at import, so once the
    # default has moved they belong to a different tier and comparing against
    # them would raise the "binary is in the wrong directory" error below on a
    # perfectly healthy install.
    if loaded == (_SELECTED or "fast"):
        gb = getattr(pkg, "_mojolearn_gbdt", None)
    else:
        try:
            gb = getattr(load_set(loaded), "_mojolearn_gbdt", None)
        except Exception:
            gb = None
    # `hasattr` ON A STUB RAISES, IT DOES NOT RETURN FALSE. A missing binding
    # is represented by `_MissingUpperTier`, whose `__getattr__` raises
    # ImportError BY NAME -- and `hasattr` only swallows AttributeError, so
    # probing a stub propagates. Before this guard, asking a package with any
    # unbuilt binding for its own mode raised, which took down every caller
    # including `repeat_run_stability.py` on a leg that had deliberately built
    # a subset (2026-08-29, AMD MI325X: "MOJOLEARN_NUMERIC_MODE=identical but
    # ..._mojolearn_gbdt.so is not built" out of a function whose entire job is
    # to REPORT the mode).
    #
    # The stub is right to raise on use; the read-back is wrong to treat a
    # probe as use. An absent gbdt binary means the cross-check cannot run,
    # which is not the same as the cross-check failing.
    try:
        readable = gb is not None and hasattr(gb, "gbdt_numeric_mode")
    except ImportError:
        readable = False
    if readable:
        compiled = _CODE_MODE.get(gb.gbdt_numeric_mode(), "unknown")
        if compiled != loaded:
            raise RuntimeError(
                f"mojolearn: the loaded gbdt binary was compiled {compiled} "
                f"but the selector loaded the {loaded} set -- a binary is in "
                "the wrong directory; rebuild both sets"
            )
    return loaded


def vendor():
    """'metal', 'cuda' or 'hip': the accelerator API of the binaries this
    process runs, READ BACK FROM THE BINARIES of the current default tier.

    Same shape as `numeric_mode()`: the selector's choice is only reported
    after the loaded set has been asked and agrees. On the flat layout
    (macOS, or a Linux source checkout) the selector made no choice and the
    binaries are the only source; on the Linux wheel layout the directory
    was chosen by `_layout()` and every binary in it was checked against
    that directory when it loaded, so this cannot disagree with the
    directory without having already raised.

    Returns None only for binaries built before the read-back existed
    (before 2026-08-29), and says so in `vendor_how()` rather than
    inventing a name from the platform."""
    global _VENDOR_SELECTED
    kind, base = _layout()
    loaded = default_mode()
    try:
        s = load_set(loaded)
    except Exception:
        s = None
    said = None
    if s is not None:
        for name in _MODULES:
            if name in s.missing:
                continue
            try:
                said = read_vendor(getattr(s, name))
            except ImportError:
                continue
            if said is not None:
                break
    if kind == "vendor":
        expected = os.path.basename(base)
        if said is not None and said != expected:
            raise RuntimeError(
                f"mojolearn: the loaded {loaded} binaries were compiled for "
                f"{said} but the selector opened the {expected} set -- a "
                "binary is in the wrong vendor directory; rebuild both sets"
            )
        return expected
    if said is not None:
        _VENDOR_SELECTED = said
    return _VENDOR_SELECTED


def vendor_how():
    """How the vendor was decided, for `python -m mojolearn verify` and
    `mojolearn doctor`: 'flat layout; read from the loaded binaries',
    'MOJOLEARN_VENDOR in the environment', or 'the box probe (device nodes
    and driver libraries)'."""
    _layout()
    return _VENDOR_HOW
