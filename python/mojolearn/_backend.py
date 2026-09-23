# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
                   `checks/numerics.mojo` is true under both upper tiers.

`MOJOLEARN_NUMERIC_MODE=<tier>` in the environment AT IMPORT TIME makes
`mojolearn` load that set under the canonical module names, so every caller's
`from . import _mojolearn_gbdt` sees the right arithmetic (IDENTITY_PATHS.md;
E1/archive/evidence/E2_RESULTS.md are the measurements). Unset selects `identical`. Explicit `fast` loads the baseline
set. Anything else raises: a mode that is accepted and ignored is worse than
one refused -- which is exactly what this selector did to `deterministic`
until 2026-08-29, when the tier existed in the compiler and was unreachable
from Python because this function's allow-list had two entries in it.

The mode is a BUILD DEFINE (`-D MOJOLEARN_NUMERIC_IDENTICAL=1`, read by
`checks/numerics.mojo` through `is_defined`), and the identical binaries
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
     compile-time constant (`checks/vendor.mojo`): 'metal', 'cuda',
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

THE ARCHITECTURE AXIS (2026-08-30, LEGS_2026-08-30.md)
------------------------------------------------------
One `mojo build` emits device code for EXACTLY ONE GPU architecture and no
PTX, so there is no JIT fallback: a binary runs on the architecture family
it was built for and nothing else. Measured 2026-08-30: a set built on an
H100 carried `sm_90a` only, installed cleanly on an A40, and failed 27 of
29 lanes with CUDA_ERROR_NO_BINARY_FOR_GPU. A portable wheel therefore
carries one set PER ARCHITECTURE, one directory level under the vendor:

    python/mojolearn/cuda/sm_80/{,deterministic,identical}/*.so
    python/mojolearn/cuda/sm_90a/...
    python/mojolearn/hip/gfx942/...

and the architecture is picked at import, right after the vendor, in this
order:

  1. `MOJOLEARN_GPU_ARCH` in the environment picks the DIRECTORY. It must
     name a set the install carries or the import raises.
  2. The device's own architecture, read WITHOUT loading any extension:
     for cuda, `cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR/MINOR)`
     through ctypes on the driver library the probe already loads; for
     hip, `gfx_target_version` out of /sys/class/kfd/kfd/topology (the
     KFD topology needs no ROCm library at all). An exact match wins; on
     cuda a device `sm_XY` also accepts a carried `sm_XYa` (same chip,
     architecture-specific build), and failing both, the HIGHEST carried
     non-`a` architecture of the same major family that does not exceed
     the device (NVIDIA documents cubin forward compatibility within a
     family; sm_80 code runs on sm_86/sm_89). `a`-suffixed builds are
     architecture-specific and never chosen by the family rule. On hip
     there is NO family rule: gfx code objects are ISA-exact, so anything
     but an exact match refuses.
  3. When the device architecture cannot be determined and the install
     carries exactly ONE architecture for the chosen vendor, that one is
     used (there is nothing better to do and it is what an arch-less
     install always did). More than one and no answer refuses, naming
     `MOJOLEARN_GPU_ARCH`.

A vendor directory whose binaries sit directly in it (no architecture
subdirectory) keeps working as before: that is every set built before the
axis existed, and the flat/legacy behaviour is a supported layout, not a
deprecation. `gpu_arch()` and `gpu_arch_how()` report what was picked and
why.

THERE IS NO CPU PATH, so when the wheel layout is present and no vendor can
be picked the import RAISES, naming every path and library it looked for
and what it found, rather than importing a package whose every fit would
fail. `vendor()` reports what was picked, cross-checked against the loaded
binaries; `NumericModeMixin.vendor_used()` reports it per estimator.
"""

import importlib.machinery
import importlib.util
import os
import re
import sys

from . import host_surface


# THE RUNTIME REWRITES THE PROCESS ENVIRONMENT WHEN A BINDING LOADS.
# Measured 2026-09-23 on a RunPod CPU pod (0.8.15 wheel, a uv venv): after the
# first Mojo binding is executed the parent process carries
# PYTHONEXECUTABLE=<the first python3 on PATH>, PYTHONPATH=":" and
# MOJO_PYTHON_LIBRARY, none of which were set before. The bundled runtime
# writes them for its embedded interpreter. A child started afterwards with
# `subprocess.run([sys.executable, ...])` inherits them, and on CPython 3.11
# that child took its executable and prefix from PYTHONEXECUTABLE: it came up
# as the pixi env's python3 with the venv's site-packages gone and `import
# numpy` failing (tests/test_crossvendor_coverage.py's CLI test, red on 3.11
# only; 3.10 and 3.14 ignore the variable on Linux). The runtime has read them
# by the time exec_module returns, so every binding load restores the
# caller's environment: a variable that was absent is removed again and one
# that was set keeps its value.
#
# THE RUNTIME WRITES THE C ENVIRON, NOT os.environ. os.environ is a mapping
# copied at interpreter start; a C setenv() from a loaded library changes what
# a child inherits without changing the mapping, so `os.environ.get` read
# None before AND after the load while the child still saw the variables
# (pod vzzgf45v5abi16, 2026-09-23). The restore therefore goes through
# os.unsetenv and os.putenv, which act on the C environ directly, and keeps
# the mapping in step.
_RUNTIME_ENV = ("PYTHONEXECUTABLE", "PYTHONPATH", "MOJO_PYTHON_LIBRARY")


def _exec_binding(loader, module):
    """The loader's exec_module, with the process environment restored."""
    before = {k: os.environ.get(k) for k in _RUNTIME_ENV}
    try:
        loader.exec_module(module)
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
                os.unsetenv(k)
            else:
                os.environ[k] = v
                os.putenv(k, v)

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
    "_mojolearn_preprocessing",
    "_mojolearn_tsa",
    "_mojolearn_linalg",
    "_mojolearn_arima",
    "_mojolearn_training",
    # Runtime-shaped decoder LM trainer; source integration is not qualification.
    "_mojolearn_byte_lm",
    # Added 2026-09-01 with the GaussianProcessRegressor exposure. The
    # binding itself (bindings/_mojolearn_gp.mojo + build_gp.sh) is OWED at
    # the time of this edit; listing the name FIRST is deliberate, because
    # this tuple is the difference between an unbuilt extension raising BY
    # NAME and a wrong-tier binary answering under the right label
    # (DEVIATION 869, the header above).
    "_mojolearn_gp",
    # Added 2026-09-01 with the Mamba block surface (Mamba1Block /
    # Mamba2Block, `_mamba_impl.py`; Mamba3Block joined later the same
    # day). The binding (bindings/_mojolearn_mamba.mojo + build_mamba.sh)
    # landed in the same commit; the Mamba-1/2 entries built and gated
    # green in all three tiers that evening, and the Mamba-3 entries
    # followed: gated green at 08a38a13 in all three tiers, with the
    # corpus arm added 2026-09-03. The "RUN OWED per tier until rebuilt"
    # note here was stale (60a90de9 fixed the twin sentence in
    # __init__.py and missed this one). ONE APPLE M4 throughout, and the
    # binding FAULTS on AMD in every tier. Listing the name here and in
    # `_build_script`, both, is what makes an unbuilt extension raise BY
    # NAME with the build command instead of a wrong-tier binary
    # answering under the right label (DEVIATION 869, the header above).
    "_mojolearn_mamba",
    # Added 2026-09-02 with the transformer block surface
    # (TransformerBlock, `_transformer_impl.py`). The binding
    # (bindings/_mojolearn_transformer.mojo + build_transformer.sh, the
    # FIFTEENTH) landed in the same commit and COMPILED FOR THE FIRST TIME
    # 2026-09-02, on APPLE ONLY; `tests/test_transformer_surface.py` is
    # green in all three tiers on that one box, and no NVIDIA or AMD box
    # has built or run it. Listing the name here and in
    # `_build_script`, both, is what makes an unbuilt extension raise BY
    # NAME with the build command instead of a wrong-tier binary
    # answering under the right label (DEVIATION 869, the header above).
    "_mojolearn_transformer",
    # Workstream D, 2026-09-14: four families that were built and identity-gated with no
    # public door. Each is its own binding and build script, IDENTICAL only,
    # listed here and in `_build_script` both so an unbuilt one raises BY
    # NAME with the build command (DEVIATION 869). Compile-checked on one
    # Apple M4 the day they were written; no box has run them through the
    # Python door yet, and the identity_break lanes and three columns are
    # owed. The Cholesky door is inside
    # `_mojolearn_gp` (bindings/build_gp.sh already links cholesky/).
    "_mojolearn_kernel_methods",
    "_mojolearn_mixture",
    "_mojolearn_hdbscan",
    "_mojolearn_resample",
    # 2026-09-14, lane/expose-ivf-embedding: the two `_NOT_YET` entries given
    # their doors. IVFIndex (bindings/_mojolearn_ivf.mojo, prepared earlier
    # the same day and listed now that check-ivf reads ALL OK with one card
    # on Apple, NVIDIA and AMD) and Embedding (bindings/_mojolearn_embedding.mojo,
    # profile mojolearn.identical.embedding.fp32.v1 with padding_idx and the
    # microbatch carry). IDENTICAL only, like every non-tree binding.
    "_mojolearn_ivf",
    "_mojolearn_embedding",
)

#: ONE RULE FOR TIERS (DEVIATION 2490, 2026-09-10): THE TREE LANES SHIP
#: THREE TIERS, EVERYTHING ELSE SHIPS IDENTICAL ONLY.
#:
#: The product is cross-vendor bitwise identity. That is the thing no other
#: library sells, on any platform, and it is the default tier. A FAST tier
#: only earns its place where we have measured a win against the opponent's
#: own CPU, and that is trees on Apple silicon: tree fitting calls no BLAS
#: (histogram building and split finding are scatter-gather over integers),
#: so the opponent gets nothing from Accelerate's AMX coprocessor, and
#: ExtraTrees measured 1.25-1.61x scikit-learn on ALL TEN cores at covtype
#: 581k. Nothing else has that argument:
#:
#:   * The classical families (k-means, kNN, PCA, SVD, the linear models,
#:     UMAP, GP, ARIMA, preprocessing) have a BLAS call as their inner loop.
#:     On an M4 (2026-09-10) Accelerate does 1438 GFLOP/s fp32 GEMM on four
#:     P-cores against ~4000 for the ten-core GPU, and ONE CPU thread already
#:     draws 88 of the 120 GB/s the two share. A FAST kernel there wins ~2.5x
#:     at best over a CPU that scikit-learn gets for free, for the price of
#:     the reproducibility guarantee. Not a product.
#:   * SVC and SVR could beat libsvm's single thread on a Mac, but two
#:     families with a fast tier that is not "trees" is a rule a user has to
#:     look up. One rule beats two wins.
#:   * The neural lanes (transformer, mamba, training, byte LM) gate every
#:     fused kernel on the identical contract, so their lower tiers were
#:     SLOWER than the default (DEVIATION 2300 is the cost: a `k_last`
#:     failure that lived only in a tier nobody ran).
#:
#: This is an ALLOWLIST on purpose. A binding added tomorrow is identical
#: only until someone measures a win and adds it here, which is the rule in
#: CONTRIBUTING.md (Numeric modes): a tier we will not benchmark is a
#: tier we do not ship. The build scripts of every binding outside this set
#: exit 2 on any other MOJOLEARN_NUMERIC_MODE, and this set is why the Python
#: side raises a sentence a caller can act on instead of an ImportError about
#: a missing `.so`.
_TIERED = frozenset({
    "_mojolearn_gbdt",   # GradientBoosting
    "_mojolearn_rf",     # RandomForest
    "_mojolearn_trees",  # ExtraTrees
})

_IDENTICAL_ONLY = frozenset(_MODULES) - _TIERED
_SELECTED = None

_IDENTICAL_ONLY_REASON = (
    "Only the tree lanes (GradientBoosting, RandomForest, ExtraTrees) ship "
    "fast and deterministic tiers; every other family ships IDENTICAL only "
    "(DEVIATION 2490)."
)


def _identical_only_reason(name):
    """The sentence that explains why this binding has one tier."""
    return _IDENTICAL_ONLY_REASON


#: Tier name -> the code `<ext>_numeric_mode()` reports, which is the
#: `NUMERIC_*` constant in `checks/numerics.mojo`. Keep the two in step:
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
        # `/dev/kfd` ONLY. `/dev/dri/renderD128` was here until 2026-08-31 and
        # it is NOT AMD-specific: it is the generic DRM render node and every
        # GPU creates one, NVIDIA included. On an A40 that has /dev/dri the
        # probe found "hip evidence" beside real cuda evidence, and because a
        # shipped wheel carries BOTH vendor sets the selector saw two
        # candidates and REFUSED TO IMPORT, telling the user to set
        # MOJOLEARN_VENDOR on a machine with one NVIDIA card in it. It
        # surfaced as three sabotage cases failing on an A40 and passing on an
        # H100, which is the same code on two boxes and therefore the box.
        # `/dev/kfd` is ROCm's kernel fusion driver node, created by the
        # amdgpu KFD path alone, so it is the honest device-node test.
        "paths": ("/dev/kfd",),
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


def _force_requested():
    """MOJOLEARN_VENDOR_FORCE, read the way a shell user would expect. Anything
    other than these spellings is NOT a request to override, because a
    misspelled override that silently engages is the same defect as no
    override at all."""
    return os.environ.get("MOJOLEARN_VENDOR_FORCE", "").strip().lower() in (
        "1", "true", "yes", "on")


def _probe_lines(probe, only=None):
    """The probe table. `only` narrows it to one vendor, for a refusal that is
    about that vendor alone."""
    lines = []
    for v in (_LINUX_VENDORS if only is None else (only,)):
        r = probe[v]
        lines.append(f"  {v}:")
        for p, ok in r["paths"].items():
            lines.append(f"    {p:<28} {'FOUND' if ok else 'absent'}")
        for lib, ok in r["libs"].items():
            lines.append(f"    {lib:<28} {'loads' if ok else 'not loadable'}")
    return lines


# ===================================================================
# THE ARCHITECTURE AXIS. See the module docstring.
# ===================================================================

#: Directory names that count as an architecture level under a vendor
#: directory: sm_80, sm_90a, gfx90a, gfx942, ... Anything else under the
#: vendor directory (deterministic/, identical/, .libs/) is not one.
_ARCH_RE = re.compile(r"^(sm_[0-9]+a?|gfx[0-9a-f]+)$")

_ARCH_SELECTED = None
_ARCH_HOW = None


def _arch_dirs(vdir):
    """The architecture subdirectories of one vendor directory that hold
    binaries, sorted. Empty list means the arch-less (legacy) layout."""
    try:
        names = os.listdir(vdir)
    except OSError:
        return []
    return sorted(n for n in names
                  if _ARCH_RE.match(n) and _has_binaries(os.path.join(vdir, n)))


def _vendor_has_set(vdir):
    """Does this vendor directory carry ANY loadable set: binaries directly
    in it (legacy) or under an architecture subdirectory."""
    return _has_binaries(vdir) or bool(_arch_dirs(vdir))


def _device_arch(vendor):
    """(arch, how) for the first visible device of `vendor`, or
    (None, why-not). NEVER loads an extension and never opens a MAX device
    context: cuda is asked through ctypes on the driver library the probe
    already loads, hip through the KFD topology files, which need no ROCm
    library at all. Every failure returns a reason a refusal can print."""
    if vendor == "cuda":
        import ctypes
        lib = None
        for name in _PROBE["cuda"]["libs"]:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            return None, "no libcuda could be loaded"
        try:
            rc = lib.cuInit(0)
            if rc != 0:
                return None, f"cuInit returned {rc}"
            n = ctypes.c_int(0)
            rc = lib.cuDeviceGetCount(ctypes.byref(n))
            if rc != 0 or n.value < 1:
                return None, f"cuDeviceGetCount rc={rc} count={n.value}"
            dev = ctypes.c_int(0)
            rc = lib.cuDeviceGet(ctypes.byref(dev), 0)
            if rc != 0:
                return None, f"cuDeviceGet returned {rc}"
            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            # 75/76: CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR.
            rc1 = lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
            rc2 = lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)
            if rc1 != 0 or rc2 != 0:
                return None, f"cuDeviceGetAttribute rc={rc1}/{rc2}"
        except (OSError, AttributeError) as exc:
            return None, f"driver call failed: {exc}"
        return (f"sm_{major.value}{minor.value}",
                "compute capability from the CUDA driver")
    if vendor == "hip":
        # gfx_target_version encodes major*10000 + minor*100 + step; the
        # gfx name spells minor and step in hex (gfx90a is 9.0.10). CPU
        # nodes in the topology carry 0 and are skipped.
        base = "/sys/class/kfd/kfd/topology/nodes"
        try:
            nodes = sorted(os.listdir(base))
        except OSError:
            return None, f"{base} not readable"
        for node in nodes:
            props = os.path.join(base, node, "properties")
            try:
                with open(props) as f:
                    text = f.read()
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == "gfx_target_version":
                    v = int(parts[1])
                    if v > 0:
                        return (f"gfx{v // 10000:d}{(v // 100) % 100:x}{v % 100:x}",
                                f"gfx_target_version in {props}")
        return None, f"no GPU node with gfx_target_version under {base}"
    return None, f"no device-architecture probe for vendor {vendor!r}"


def _sm_parts(arch):
    """('sm_86' -> (8, 6, False)); ('sm_90a' -> (9, 0, True)). The minor
    digit is the LAST digit: sm_121a is 12.1."""
    body = arch[len("sm_"):]
    specific = body.endswith("a")
    if specific:
        body = body[:-1]
    return int(body[:-1]), int(body[-1]), specific


def _pick_arch(vendor, vdir, archs):
    """Which of `archs` (the carried architecture directories) this box
    should load, and how that was decided. Raises with the whole table when
    nothing carried can run here."""
    forced = os.environ.get("MOJOLEARN_GPU_ARCH", "").strip().lower()
    if forced:
        if forced not in archs:
            raise ImportError(
                f"mojolearn: MOJOLEARN_GPU_ARCH={forced} but this install "
                f"carries no such set under {vdir}; it carries {archs}"
            )
        return forced, "MOJOLEARN_GPU_ARCH in the environment"
    dev, how = _device_arch(vendor)
    if dev is None:
        if len(archs) == 1:
            return archs[0], (f"the only architecture carried (the device's "
                              f"own could not be read: {how})")
        raise ImportError(
            f"mojolearn: this install carries {vendor} sets for {archs} and "
            f"the device's architecture could not be determined ({how}), so "
            "there is nothing to choose by. Set MOJOLEARN_GPU_ARCH to one "
            "of the names above."
        )
    if dev in archs:
        return dev, f"exact match for the device ({how}: {dev})"
    if vendor == "cuda":
        # The device reports sm_XY; an sm_XYa build targets exactly that
        # chip and runs on it (the `a` restricts WHICH DEVICES, not this
        # one). Prefer it before any family fallback.
        if dev + "a" in archs:
            return dev + "a", (f"architecture-specific build for this exact "
                               f"device ({dev} -> {dev}a)")
        # Cubin forward compatibility WITHIN a family: NVIDIA guarantees
        # sm_X0-class code runs on later minor revisions of major X, and
        # `a`-suffixed builds are excluded from that guarantee, so they are
        # excluded here. The highest carried candidate that does not exceed
        # the device wins.
        dmaj, dmin, _ = _sm_parts(dev)
        cands = []
        for a in archs:
            if not a.startswith("sm_"):
                continue
            maj, minr, specific = _sm_parts(a)
            if specific or maj != dmaj or minr > dmin:
                continue
            cands.append((minr, a))
        if cands:
            _, best = max(cands)
            return best, (f"same-family lower architecture ({best} code "
                          f"runs forward on this {dev} device)")
    lines = "\n".join(f"    {a}" for a in archs)
    hint = ("gfx code objects are ISA-exact; there is no cross-architecture "
            "compatibility on hip" if vendor == "hip" else
            "cubins run forward only within one family, and `a`-suffixed "
            "builds only on their exact chip")
    raise ImportError(
        f"mojolearn: this device is {dev} ({how}) and no {vendor} set this "
        f"install carries can run on it ({hint}). Carried:\n{lines}\n"
        "A release carrying this device's architecture is needed; "
        "MOJOLEARN_GPU_ARCH can force one of the directories above, and the "
        "first kernel launch will then report the runtime's own error."
    )


def _vendor_base(pkg, vendor):
    """The directory `tier_dir` roots at for one chosen vendor: the vendor
    directory itself on the arch-less layout, or the chosen architecture
    subdirectory. Sets _ARCH_SELECTED/_ARCH_HOW either way."""
    global _ARCH_SELECTED, _ARCH_HOW
    vdir = os.path.join(pkg, vendor)
    archs = _arch_dirs(vdir)
    if not archs:
        _ARCH_SELECTED = None
        _ARCH_HOW = "no architecture level (arch-less set)"
        return vdir
    arch, how = _pick_arch(vendor, vdir, archs)
    _ARCH_SELECTED = arch
    _ARCH_HOW = how
    return os.path.join(vdir, arch)


def gpu_arch():
    """The architecture directory this process loads from ('sm_80',
    'gfx942', ...), or None on the flat and arch-less layouts (where no
    choice was made)."""
    if _CPU_ONLY is not None:
        return None
    _layout()
    return _ARCH_SELECTED


def gpu_arch_how():
    """How the architecture was decided, for `mojolearn doctor` and the
    smoke: 'exact match for the device (...)', 'MOJOLEARN_GPU_ARCH in the
    environment', 'same-family lower architecture (...)', ..."""
    if _CPU_ONLY is not None:
        return "no GPU binary set loaded (CPU-only install, DEVIATION 2615)"
    _layout()
    return _ARCH_HOW


def _layout():
    """('flat', <pkg dir>) or ('vendor', <pkg dir>/<vendor>[/<arch>]),
    decided ONCE.

    The wheel layout is recognised by a vendor directory WITH BINARIES IN
    IT. A bare directory does not count: the macOS wheel never has one, a
    source checkout never has one, and an empty one left by a failed build
    must not turn a working flat install into a vendor lookup."""
    global _LAYOUT, _VENDOR_SELECTED, _VENDOR_HOW
    if _LAYOUT is not None:
        return _LAYOUT
    pkg = _pkg_dir()
    present = [v for v in _LINUX_VENDORS
               if _vendor_has_set(os.path.join(pkg, v))]
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
        # A FORCED VENDOR WHOSE RUNTIME IS NOT ON THIS BOX AT ALL KILLS THE
        # PROCESS. Measured 2026-08-30 against the finished 0.3.0 wheel, in a
        # container with no device of either kind:
        #
        #   MOJOLEARN_VENDOR=cuda   imports, and the FIRST FIT raises a clean
        #                           Python exception naming libnvidia-ml.so.1
        #   MOJOLEARN_VENDOR=hip    SIGILL during import, no traceback at all
        #
        # The two vendors do not degrade alike. The hip binary's module init
        # takes a trap when the HIP runtime library is absent, so a user who
        # mistypes this variable, or sets it on the wrong box, gets a dead
        # process and nothing to read. Refusing here, in Python, is the only
        # place a message survives.
        #
        # THIS DOES NOT CLOSE THE ESCAPE HATCH the no-GPU refusal offers. That
        # hatch is for "the device IS present and this probe is wrong", and a
        # box with the device present has that vendor's driver library
        # loadable, which is exactly what `found` means. Only the case with no
        # device node AND no loadable driver library is refused, which is the
        # case that cannot work whatever the binary does.
        # MOJOLEARN_VENDOR_FORCE=1 proceeds anyway, for a box where the probe
        # is wrong about a device that really is there. It says plainly that
        # it may abort, because on the hip branch it may.
        probe = _probe_box()
        if not probe[forced]["found"] and not _force_requested():
            raise ImportError(
                f"mojolearn: MOJOLEARN_VENDOR={forced}, but this box shows NO "
                f"evidence of a {forced} device. Neither a device node nor a "
                f"driver library was found:\n"
                + "\n".join(_probe_lines(probe, only=forced))
                + f"\n\nLoading the {forced} set anyway can KILL THIS PROCESS "
                "rather than raise: measured 2026-08-30, forcing hip on a box "
                "with no ROCm aborts during import with no traceback, while "
                "forcing cuda on a box with no CUDA raises a normal exception "
                "at the first fit. The two are not alike and this refusal is "
                "the one that survives.\n"
                "If the device really is present and this probe is wrong, "
                "MOJOLEARN_VENDOR_FORCE=1 proceeds and accepts that risk. "
                "Unset MOJOLEARN_VENDOR to let the probe choose."
            )
        base = _vendor_base(pkg, forced)
        _VENDOR_SELECTED = forced
        _VENDOR_HOW = "MOJOLEARN_VENDOR in the environment"
        _LAYOUT = ("vendor", base)
        return _LAYOUT
    probe = _probe_box()
    hits = [v for v in present if probe[v]["found"]]
    if len(hits) == 1:
        base = _vendor_base(pkg, hits[0])
        _VENDOR_SELECTED = hits[0]
        _VENDOR_HOW = "the box probe (device nodes and driver libraries)"
        _LAYOUT = ("vendor", base)
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
        # NOT the basename: with the architecture axis the directory the
        # binaries load from is <vendor>/<arch>, and its basename is the
        # architecture. The vendor is what _layout() selected.
        expected = _VENDOR_SELECTED
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
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower()
    if mode not in _MODE_CODE:
        raise ImportError(
            f"mojolearn: MOJOLEARN_NUMERIC_MODE={mode!r}; it must be 'fast', "
            "'deterministic' or 'identical' (the default)"
        )
    return mode


# ===================================================================
# A CPU-ONLY INSTALL (DEVIATION 2615)
# ===================================================================
# The no-GPU refusal below is deliberate and stays the rule for every GPU
# binding. The ONE exception is a box where no GPU set loads but a CPU
# binding under mojolearn/host/ (`_mojolearn_byte_lm_host.so`, DEVIATION
# 2610, or `_mojolearn_forest_host.so`, the forest host lane) is built. Then
# the package imports, every GPU binding is a `_NoGpuBinding` stub that
# raises BY NAME on use with the original refusal attached, and `vendor()`
# answers 'cpu'. Nothing falls back: a GPU estimator on that box fails
# exactly as loudly as before, only at use instead of at import.
#
# THE HOST BINDING SET (the CPU training lane, 2026-09-13). `_HOST_MODULES`
# maps a `_MODULES` name to the host binding that exports the SAME function
# names the GPU binding exports for the fits it covers. On a CPU-only
# install `binding(name)` and the canonical `mojolearn.<name>` resolve a
# listed family to that binding when its file is built, through a
# `_HostBinding` proxy that raises BY NAME for any function the host
# binding lacks, so a lane with no host fit reads REFUSED and never a hash
# of something else. Two rules keep it honest: the host set loads only when
# `_CPU_ONLY` is not None, so a box with a GPU never serves host arithmetic
# under a GPU label, and every host binding is read back (`<prefix>_vendor()`
# must answer "cpu", `<prefix>_numeric_mode()` 1 and `<prefix>_column()`
# "cpu", the kernel matrix's CPU column) before a single function is served.

#: The refusal `select()` would have raised, when it installed the CPU-only
#: stubs instead; None on every install that loaded a GPU set.
_CPU_ONLY = None


def host_binding_path():
    """Where the CPU inference binding lives on this install. THROUGH
    `host_module_path`, like every other door onto a host binding: until
    lane/host-path-resolution (2026-09-16) this built its own path from the
    package directory and was one of four answers that did not follow
    MOJOLEARN_HOST_DIR."""
    return host_module_path("_mojolearn_byte_lm_host")


def forest_host_binding_path():
    """Where the CPU forest inference binding lives on this install (the
    forest host lane, 2026-09-13, `_forest_host.py`). Through
    `host_module_path`, as above."""
    return host_module_path("_mojolearn_forest_host")


def host_binding_built():
    """Whether ANY CPU binding is built. Any one is enough to turn the
    no-GPU refusal into by-name stubs, because any one is a surface that
    computes on this box."""
    return bool(host_families_built())


#: Names another directory of host bindings; how the CPU identity gate
#: (.github/workflows/cpu-identity-gate.yml) loads the set it built with
#: -D MOJOLEARN_HOST_SABOTAGE=1 without touching the production set.
_HOST_DIR_ENV = "MOJOLEARN_HOST_DIR"


def host_dir():
    """Where every CPU binding lives on this install, `mojolearn/host/`, or
    the directory MOJOLEARN_HOST_DIR names."""
    override = os.environ.get(_HOST_DIR_ENV, "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(_pkg_dir(), "host")


def host_module_path(basename):
    """The file a host binding of `basename` (`_mojolearn_<family>_host`)
    loads from.

    THE ONE RESOLUTION PATH. `load_host_module` below, the two helpers
    above, and the byte LM and forest INFERENCE doors
    (`_byte_lm_host.binary_path`, `_forest_host.binary_path`, each after its
    own named-binary variable) all answer from here, so one directory
    setting moves every host binding together. Two of them resolved
    independently until lane/host-path-resolution (2026-09-16), and under
    MOJOLEARN_HOST_DIR the training door opened while the inference door
    looked in the package directory: it refused there when the package had
    no host set, and silently served the package's own binding when it
    did."""
    return os.path.join(host_dir(), basename + ".so")


def host_families_built():
    """The basenames of every `_mojolearn_*_host.so` under mojolearn/host/,
    sorted. What `tools/identity_break.py` records as `host.families`, so a
    REFUSED cell on the CPU column is attributable to an unbuilt family
    rather than a bug."""
    d = host_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        f[:-3] for f in os.listdir(d)
        if f.startswith("_mojolearn_") and f.endswith("_host.so")
    )


#: `_MODULES` name -> the basename of the host binding under mojolearn/host/
#: that exports the SAME function names the GPU binding exports for the fits
#: it covers, plus `<prefix>_vendor()` answering "cpu", `<prefix>_numeric_mode()`
#: answering 1 and `<prefix>_column()` answering "cpu". THE TABLE IS READ FROM
#: THE MANIFEST (`host_surface.py`, the host surface manifest lane,
#: 2026-09-14): one file declares every host family, what it routes, which
#: lanes it covers for training and serves for inference, and what it
#: exports; `tests/test_host_surface.py` holds the bindings to it. Every
#: family not routed there refuses BY NAME on a CPU-only install. The three
#: bindings loaded by path, `_mojolearn_byte_lm_host`,
#: `_mojolearn_forest_host` and `_mojolearn_tokenizer_host`, export their
#: own names (`byte_lm_host_*`, `forest_host_*`, `tokenizer_host_*` and
#: `bpe_*`) for surfaces of their own (LanguageModelInference,
#: LanguageModelHostTrainer, HostForest, HostGBDT, BpeTokenizer), are
#: loaded in `_byte_lm_host.py`, `_forest_host.py`, `_gbdt_host.py` and
#: `tokenizer.py` (the last through `load_host_module` below), and are
#: deliberately NOT routed: mapping `_mojolearn_rf` to the forest host
#: binding would route RandomForestClassifier.predict to an entry with a
#: different address contract under the GPU entry's name, and the
#: tokenizer has no GPU binding to route from at all.
#:
#: What each routed binding carries, and what it leaves absent so the
#: refusal stays by name:
#:   _mojolearn -> _mojolearn_core_host: the base binding's HOST HELPERS
#:     (transpose_f32, cast_colmajor_f64_to_f32, cast_f64_to_f32,
#:     all_finite_*, gather_*, argmax_rows_*, so _buffer._native and _labels
#:     resolve on a CPU-only install) and the knn host inference lane's
#:     (2026-09-14) knn_search, knn_classify, knn_regress over
#:     core/knn_host_predict.mojo, and (workstream E batch 2, 2026-09-14)
#:     the training entry kmeans_fit over cluster/host/kmeans_oracle.mojo,
#:     and (the spectral-precomputed lane, 2026-09-14) the dense affinity's
#:     COO scan nonzero_f64_count and nonzero_f64_fill, and (batch 3,
#:     2026-09-14) the cosine, manhattan, chebyshev and minkowski metrics
#:     of the k-NN entries and the ball cover's radius_neighbors_count,
#:     radius_neighbors_fill and rbc_knn_search as an exhaustive scan.
#:   _mojolearn_linalg -> _mojolearn_linalg_host: gemm over
#:     gemm/host/gemm_oracle.mojo::gemm_oracle, the profile's definition.
#:   _mojolearn_estimators -> _mojolearn_estimators_host: kde_score_samples
#:     over kde/host/kde_oracle.mojo, the classical inference entries
#:     ols_predict, tsvd_transform, pca_transform, pca_whiten_transform,
#:     pca_whiten_inverse_transform, qn_decision_function, qn_sigmoid over
#:     core/classical_host_predict.mojo, and (workstream E, 2026-09-14) the
#:     training entries pca_fit and tsvd_fit over
#:     decomposition/host/pca_oracle.mojo and ols_fit and ridge_fit over
#:     glm/host/glm_oracle.mojo, dbscan_fit over
#:     dbscan/host/dbscan_oracle.mojo and (batch 2) qn_fit over
#:     glm/host/qn_oracle.mojo (the L-BFGS arm; batch 3, 2026-09-14, adds
#:     the OWL-QN arm, the softmax loss and DBSCAN's sample_weight;
#:     qn's sample_weight refuses by name), pca_fit_full over
#:     decomposition/host/pca_full_oracle.mojo (the pca-full-whiten lane,
#:     2026-09-14; the tall route only, a wide matrix refuses by name);
#:     inverse_transform is absent.
#:   _mojolearn_metrics -> _mojolearn_metrics_host (workstream E batch 2,
#:     2026-09-14): accuracy_score, adjusted_rand_score, entropy,
#:     mutual_info_score, homogeneity_score, completeness_score,
#:     v_measure_score, r2_score and silhouette over
#:     metrics/host/metrics_oracle.mojo, spectral_fit_predict_dataset
#:     over spectral/host/spectral_oracle.mojo (the spectral lane) and
#:     spectral_fit_predict_graph over the same oracle (the
#:     spectral-precomputed lane, 2026-09-14), umap_fit_transform,
#:     umap_transform and umap_numeric_mode over umap/host/umap_oracle.mojo
#:     (the umap lane, lane/cpu-training-umap-b, 2026-09-14: the IDENTICAL
#:     device epoch fold restated on the host), and (the
#:     metrics-classification lane, 2026-09-14) rand_score, the ranking and
#:     classification metrics, the three regression errors, kl_divergence
#:     and trustworthiness over metrics/host/classification_oracle.mojo.
#:   _mojolearn_preprocessing -> _mojolearn_preprocessing_host (workstream
#:     E batch 2, 2026-09-14): standard_fit, standard_transform, minmax_fit
#:     and minmax_transform over preprocessing/host/scaler_oracle.mojo, the
#:     whole GPU binding's surface.
#:   _mojolearn_tsa -> _mojolearn_tsa_host: holtwinters_fit and
#:     holtwinters_forecast over holtwinters/host/hw_oracle.mojo, and
#:     (batch 3, 2026-09-14) kpss_test over tsa/checks/kpss_oracle.mojo;
#:     select_d (ARIMA's) is absent.
#:   _mojolearn_solver -> _mojolearn_solver_host: cd_fit and cd_predict over
#:     solver/host/cd_oracle.mojo and gemm_oracle; agglomerative (phase 1b,
#:     2026-09-14) adds linkage_fit over hierarchy/checks/linkage_oracle.mojo
#:     (the pinned distances, Kruskal under the device's total order, the
#:     dendrogram and the cut).
#:   _mojolearn_svm -> _mojolearn_svm_host: svc_fit and svc_predict over
#:     svm/host/smo_oracle.mojo; iforest (phase 1b, 2026-09-14) adds
#:     iforest_run over isolation_forest/checks/if_oracle.mojo (the
#:     fit-on-every-call surface kept, DEVIATION 874); batch 3 (2026-09-14)
#:     adds svr_fit and svr_predict over the oracle's EPSILON_SVR arm.
#:   _mojolearn_trees -> _mojolearn_trees_host (et-clf, et-reg; phase 1b,
#:     2026-09-14): the eight et_*_fit entries over
#:     extratrees/estimator.mojo::fit_extra_trees_classifier_host_exact and
#:     fit_extra_trees_regressor_host_exact (the device trainer restated on
#:     the host, exact keys and quantized leaves), forest_export,
#:     forest_export_legacy, forest_export_release and et_predict over
#:     core/forest_host_predict.mojo; since 2026-09-15 the resident
#:     forest_prepare_gpu, forest_predict_resident_reuse_gpu and
#:     forest_release_gpu (inference_engine='parallel_groves') over
#:     core/forest_host_groves.mojo, with et_predict_gpu_parallel and the
#:     other resident arms absent.
#:   _mojolearn_rf -> _mojolearn_rf_host (rf-clf, rf-reg; workstream E
#:     batch 3, 2026-09-14): the eight rf_*_fit entries over
#:     ensemble/host/rf_oracle.mojo::rf_host_fit (the device trainer
#:     restated on the host), forest_export, forest_export_legacy,
#:     forest_export_release, and rf_predict_proba and rf_predict_reg over
#:     core/forest_host_predict.mojo; since 2026-09-15 the POISSON, GAMMA and
#:     INVERSE_GAUSSIAN criteria, rf_classifier_fit_weighted (the weighted
#:     bootstrap; weights without a bootstrap refuse by name) and the
#:     resident parallel_groves entries over core/forest_host_groves.mojo;
#:     the shard fits and the non-resident GPU engines refuse by name. This is its own family, not the forest host binding
#:     named above, because it exports the GPU binding's names.
#:   _mojolearn_gp -> _mojolearn_gp_host (gp, gp-matern12, gp-matern32,
#:     gp-matern52-ard; workstream E, 2026-09-14): gpr_fit and gpr_predict
#:     over gaussian_process/host/gpr_oracle.mojo (the kernel matrix, the
#:     Cholesky profile of cholesky/host/chol_oracle.mojo and gemm_oracle's
#:     posterior mean), and the Cholesky door's cholesky_factor,
#:     cholesky_solve and cholesky_profile_jitter over the same chol_oracle;
#:     gp_parallel_available is absent, so the ordered multi-GPU driver
#:     refuses by name.
#:   _mojolearn_gbdt -> _mojolearn_gbdt_host (gbdt-symmetric, gbdt-rmse,
#:     gbdt-depthwise, gbdt-lossguide; workstream E batch 3, 2026-09-14):
#:     gbdt_fit over gbdt/host/gbdt_oracle.mojo::gbdt_host_fit (the device
#:     trainer restated on the host for SymmetricTree, Logloss, Cosine and
#:     Newton leaves), gbdt/host/gbdt_oracle_rmse.mojo::gbdt_rmse_host_fit
#:     (the same tree with RMSE and the searcher's leaves) and
#:     gbdt/host/gbdt_oracle_depthwise.mojo (Depthwise with Cosine, Lossguide
#:     with NewtonL2); every other option value refuses by name inside
#:     gbdt_fit,
#:     gbdt_predict and gbdt_model_dim over the model text and
#:     core/gbdt_host_predict.mojo, and gbdt_sigmoid; gbdt_fit_ordered_rmse
#:     over gbdt/host/gbdt_oracle_ordered.mojo and
#:     gbdt_fit_two_level_feature_freq over
#:     gbdt/host/gbdt_oracle_feature_freq.mojo (gbdt-ordered-rmse,
#:     gbdt-feature-freq; lane/cpu-training-gbdt-ordered, 2026-09-15);
#:     gbdt_predict_multi and the adapters' binary transforms are absent. Its own family for the reason the rf family is: the
#:     forest host binding exports other names under another contract.
#:   _mojolearn_training -> _mojolearn_training_host (the mlp lane,
#:     2026-09-14): optimizer_step and ce_loss over
#:     training/checks/optimizer_oracle.mojo and loss_oracle.mojo (the
#:     normative answers of the optimizer and loss profiles) and
#:     mlp_bias_activation, mlp_relu_backward and mlp_sum_rows over
#:     training/host/mlp_oracle.mojo, so SmallMLPTrainer trains on the CPU;
#:     and (lane/cpu-training-misc, 2026-09-15) clip_grad_norm over the same
#:     optimizer oracle, and accumulate, accumulation_is_aligned and the
#:     embedding, RMSNorm and linear forward and backward over
#:     training/host/samba_ops_oracle.mojo; and (lane/cpu-training-samba,
#:     2026-09-15) neural_rng over core/philox_neural.mojo as
#:     tools/mamba_host_gen.py writes it out for the host
#:     (mamba/host/gen/philox_neural.mojo), so SambaStack trains on the CPU
#:     with the mamba and transformer families' blocks.
#:   _mojolearn_resample -> _mojolearn_resample_host (bootstrap,
#:     permutation-test, monte-carlo; lane/cpu-training-misc, 2026-09-15):
#:     bootstrap, permutation_test and monte_carlo_integrate over
#:     resample/host/resample_host.mojo (resample/estimator.mojo's entry
#:     points with the replicate and chunk kernels restated on the host);
#:     resample_ranges_parallel_available is absent, so the multi-GPU range
#:     drivers refuse by name.
#:   _mojolearn_arima -> _mojolearn_arima_host (arima, arima-011,
#:     arima-seasonal-c; workstream E, 2026-09-14): arima_fit,
#:     arima_predict and arima_forecast over arima/host/arima_oracle.mojo
#:     (estimate_x0, the Jones transform, the batched L-BFGS over the
#:     finite-difference Kalman likelihood, the undifferenced forecast), the
#:     GPU binding's whole surface; p, q or P above 1, any Q, d + D of 2,
#:     p + q + k of 0 refuse by name; an in-sample prediction (start < n_obs)
#:     runs since 2026-09-15. When this reference binding is not built (an
#:     installed wheel), `_HOST_INFERENCE_MODULES` below routes the family to
#:     `_mojolearn_forecast_host`, which carries predict and forecast only.
_HOST_MODULES = host_surface.routed_modules()

#: `_MODULES` name -> an INFERENCE-ONLY host binding that serves the route
#: when the reference binding above is not built (lane/inference-forecast-
#: umap-pca, 2026-09-15). The inference wheels ship
#: `_mojolearn_forecast_host` (ARIMA's `arima_predict` and `arima_forecast`,
#: no `arima_fit`) and not the reference `_mojolearn_arima_host`, so on an
#: installed CPU-only wheel `_mojolearn_arima` resolves here; a source build
#: that has the reference binding keeps it. An absent name still refuses by
#: name through `_HostBinding`.
_HOST_INFERENCE_MODULES = host_surface.inference_routes()

#: The env switch the CPU identity gate sets to load a host binding built
#: with `-D MOJOLEARN_HOST_SABOTAGE=1`; refused otherwise.
_HOST_ALLOW_SABOTAGE = "MOJOLEARN_HOST_ALLOW_SABOTAGE"

_HOST_MODULE_PREFIX = "mojolearn._host."


def _host_prefix(basename):
    """`_mojolearn_linalg_host` -> `linalg_host`, the read-back prefix."""
    return basename[len("_mojolearn_"):]


def load_host_module(basename):
    """Load (once per process, under `mojolearn._host.<basename>`, the same
    name `_byte_lm_host.py` and `_forest_host.py` use so one file is never
    initialized twice) and READ BACK a host binding: it must say it was
    compiled for the CPU (`<prefix>_vendor() == "cpu"`), IDENTICAL
    (`<prefix>_numeric_mode() == 1`) and as the kernel matrix's CPU column
    (`<prefix>_column() == "cpu"`, the comptime assert's witness), and a
    sabotage build is refused unless MOJOLEARN_HOST_ALLOW_SABOTAGE=1."""
    path = host_module_path(basename)
    if not os.path.exists(path):
        raise ImportError(
            f"mojolearn: {path} is not built. Build it with "
            f"bindings/build_{_host_prefix(basename)}.sh"
        )
    full = _HOST_MODULE_PREFIX + basename
    module = sys.modules.get(full)
    if module is None:
        loader = importlib.machinery.ExtensionFileLoader(full, path)
        spec = importlib.util.spec_from_loader(full, loader, origin=path)
        module = importlib.util.module_from_spec(spec)
        _exec_binding(loader, module)
        sys.modules[full] = module
    prefix = _host_prefix(basename)

    def read(fn):
        f = getattr(module, prefix + "_" + fn, None)
        if f is None:
            raise ImportError(
                f"mojolearn: {path} exports no {prefix}_{fn}(); a host binding "
                "must read back its vendor, numeric mode and column"
            )
        return f()

    said = str(read("vendor"))
    if said != "cpu":
        raise ImportError(
            f"mojolearn: {path} was compiled for {said!r}, not the CPU; a "
            "host binding is refused under any other vendor label"
        )
    compiled = _CODE_MODE.get(int(read("numeric_mode")), "unknown")
    if compiled != "identical":
        raise ImportError(
            f"mojolearn: {path} was compiled {compiled}; a host binding is "
            "IDENTICAL only. Rebuild it"
        )
    column = str(read("column"))
    if column != "cpu":
        raise ImportError(
            f"mojolearn: {path} was compiled as the {column!r} kernel-matrix "
            "column, not the CPU column; rebuild it with -D MOJOLEARN_COLUMN_CPU"
        )
    sabotage = getattr(module, prefix + "_sabotage", None)
    if sabotage is not None and bool(sabotage()) and os.environ.get(_HOST_ALLOW_SABOTAGE) != "1":
        raise ImportError(
            f"mojolearn: {path} is a SABOTAGE build and computes wrong answers "
            f"on purpose; it is refused outside the gate ({_HOST_ALLOW_SABOTAGE}=1)"
        )
    return module


def _no_cpu_implementation(name, item, reason, basename=None):
    """The by-name refusal of a CPU-only install, one sentence a caller can
    act on first, the original no-GPU refusal after it."""
    built = host_families_built()
    where = (
        f"the host binding {basename} is built but exports no {item}"
        if basename else
        f"no host binding covers {name}"
    )
    return (
        f"mojolearn: no CPU implementation of {name}.{item} yet; see "
        "SUPPORT_MATRIX.md (" + where + "; host "
        f"bindings built here: {', '.join(built) or 'none'}). This process "
        "loaded NO GPU binary set, so every GPU estimator, block and trainer "
        "without a host binding is unavailable here. The surfaces that "
        "compute on this box today are LanguageModelInference (byte LM "
        "forward pass on the CPU), LanguageModelHostTrainer (one byte LM "
        "training step on the CPU: forward, backward and the AdamW update), "
        "HostForest and HostGBDT (predict and predict_proba of a saved forest "
        "or GradientBoosting model on the CPU), and the saved-model inference "
        "of LinearRegression, Ridge, TruncatedSVD, LogisticRegression, PCA, "
        "ARIMA (predict and forecast), ExponentialSmoothing (forecast and "
        "predict) and UMAP (transform) "
        "(mojolearn.host_model, or the classes themselves on a CPU-only "
        "install), each only when its own host binding under mojolearn/host/ "
        "is built. "
        "Why no GPU set loaded:\n" + reason
    )


class _NoGpuBinding(type(sys)):
    """Stands in for a GPU binding with no host binding on a CPU-only
    install. Every attribute raises BY NAME."""

    def __init__(self, full, reason):
        super().__init__(full)
        self.__name = full.rsplit(".", 1)[-1]
        self.__reason = reason

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        raise ImportError(_no_cpu_implementation(self.__name, item, self.__reason))


class _HostBinding(type(sys)):
    """Stands in for a GPU binding whose family HAS a host binding on a
    CPU-only install. An attribute the host binding exports is served from
    it (loaded and read back on first use); one it lacks raises BY NAME, so
    a fit with no CPU implementation is a refusal and never a different
    routine under the same name."""

    def __init__(self, full, name, basename, reason):
        super().__init__(full)
        self.__name = name
        self.__basename = basename
        self.__reason = reason

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        module = load_host_module(self.__basename)
        fn = getattr(module, item, None)
        if fn is None:
            raise ImportError(
                _no_cpu_implementation(self.__name, item, self.__reason, self.__basename)
            )
        return fn


def _select_cpu_only(pkg, mode, reason):
    """Install the CPU-only set under the canonical names: a `_HostBinding`
    for every family whose host binding is listed in `_HOST_MODULES` and
    built, a `_NoGpuBinding` stub for the rest. Only the stubbed names are
    MISSING; a routed family is present on this box."""
    global _SELECTED, _CPU_ONLY
    for name in _MODULES:
        full = f"{pkg.__name__}.{name}"
        basename = _HOST_MODULES.get(name)
        if not (basename and os.path.exists(host_module_path(basename))):
            fallback = _HOST_INFERENCE_MODULES.get(name)
            basename = fallback if fallback and os.path.exists(host_module_path(fallback)) else None
        if basename:
            module = _HostBinding(full, name, basename, reason)
        else:
            module = _NoGpuBinding(full, reason)
            if name not in _MISSING:
                _MISSING.append(name)
        sys.modules[full] = module
        setattr(pkg, name, module)
    _CPU_ONLY = reason
    _SELECTED = mode
    return _SELECTED


def _cpu_only_binding(name, requested):
    """`binding()` on a CPU-only install: the module `_select_cpu_only`
    installed under the canonical name, IDENTICAL only. A tier other than
    identical is refused by name, because no host binding builds one."""
    if name not in _MODULES:
        raise ImportError(f"mojolearn: {name} is not a binding this package lists")
    if requested != "identical":
        raise ValueError(
            f"mojolearn: {name} has no {requested!r} tier on a CPU-only install; "
            "every host binding is IDENTICAL only. Drop numeric_mode= or pass "
            "numeric_mode='identical'."
        )
    pkg = __name__.rsplit(".", 1)[0]
    # A family served by a Python adapter over a host binding loaded by path
    # (host_surface.ADAPTED_MODULES; the byte LM trainer's single-device
    # entries over _mojolearn_byte_lm_host, lane/cpu-training-embedding-ivf,
    # 2026-09-15). Served only when that host binding is built; otherwise the
    # installed stub refuses by name as before.
    adapted = host_surface.ADAPTED_MODULES.get(name)
    if adapted is not None and os.path.exists(host_module_path(host_surface.family(adapted["family"])["binding"])):
        return importlib.import_module(f"{pkg}.{adapted['module']}").binding()
    module = sys.modules.get(f"{pkg}.{name}")
    if module is None:
        raise ImportError(
            f"mojolearn: {name} was not installed by the CPU-only selector; "
            "import mojolearn first"
        )
    return module


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
    # picked; that is the no-GPU refusal and it is deliberate. DEVIATION
    # 2615 turns it into by-name stubs only when the CPU binding is built.
    try:
        ident_dir = tier_dir(mode)
    except ImportError as exc:
        if not host_binding_built():
            raise
        return _select_cpu_only(pkg, mode, str(exc))
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
            full = f"{pkg.__name__}.{name}"
            if name in _IDENTICAL_ONLY:
                # BEFORE the on-disk check, on purpose. A fast `.so` of an
                # identical-only lane is never a legitimate artifact, only a
                # stale one (the lane's build script exits 2 on this tier),
                # and until DEVIATION 2490 a stale file here was imported
                # under the canonical name and ANSWERED. The stub wins over
                # whatever sits on disk.
                if full in sys.modules and not isinstance(sys.modules[full], _IdenticalOnlyTier):
                    del sys.modules[full]
                module = _IdenticalOnlyTier(full, name, "fast")
                sys.modules[full] = module
                setattr(pkg, name, module)
                _MISSING.append(name)
                continue
            if os.path.exists(os.path.join(pkg_dir, name + ".so")):
                continue
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
        if name in _IDENTICAL_ONLY and mode != "identical":
            module = _IdenticalOnlyTier(full, name, mode)
            sys.modules[full] = module
            setattr(pkg, name, module)
            continue
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
            _exec_binding(loader, module)
            # WHAT THE BINARY SAYS BEATS THE DIRECTORY IT SAT IN. Raises on
            # a vendor mismatch; see the module docstring.
            _check_vendor(module, name, path)
        sys.modules[full] = module
        setattr(pkg, name, module)
    if len(missing) == len(_MODULES):
        refusal = (
            f"mojolearn: MOJOLEARN_NUMERIC_MODE={mode} but no {mode} "
            f"binary exists under {ident_dir}. Build them with\n    "
            f"MOJOLEARN_NUMERIC_MODE={mode} bash bindings/build*.sh"
        )
        if host_binding_built():
            return _select_cpu_only(pkg, mode, refusal)
        raise ImportError(refusal)
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
        if name in _IDENTICAL_ONLY and self.mode != "identical":
            raise ImportError(
                f"mojolearn: {name} has no {self.mode!r} tier. "
                f"{_identical_only_reason(name)} See _IDENTICAL_ONLY."
            )
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
    mode = (default_mode() if mode is None else mode).strip().lower()
    if mode not in _MODE_CODE:
        raise ValueError(
            f"mojolearn: numeric_mode={mode!r}; it must be 'fast', "
            "'deterministic' or 'identical' (the default)"
        )
    if mode in _SETS:
        return _SETS[mode]
    # The vendor axis is folded in by tier_dir(): the same `<vendor>/` root
    # `select()` used, so a per-call `numeric_mode=` can never reach across
    # to the other vendor's set.
    tier_dir_ = tier_dir(mode)
    modules, missing = {}, []
    for name in _MODULES:
        # An identical-only lane is not "not built yet" in the lower tiers, it
        # is not offered there. Skipping it keeps `missing` meaning what the
        # _ModeSet error message says it means.
        if name in _IDENTICAL_ONLY and mode != "identical":
            continue
        path = os.path.join(tier_dir_, name + ".so")
        if not os.path.exists(path):
            missing.append(name)
            continue
        full = f"mojolearn._sets.{mode}.{name}"
        existing = sys.modules.get(full)
        if existing is not None:
            modules[name] = existing
            continue
        # select() may already have initialized this exact binary under its
        # canonical name. Reinitializing it under _sets registers Mojo-owned
        # Python types twice and aborts. Share only the same resolved file;
        # other tiers/vendors retain their separate loading and validation.
        canonical = sys.modules.get(f"mojolearn.{name}")
        canonical_path = vars(canonical).get('__file__') if canonical is not None else None
        if canonical_path and os.path.realpath(canonical_path) == os.path.realpath(path):
            _check_vendor(canonical, name, path)
            sys.modules[full] = canonical
            modules[name] = canonical
            continue
        loader = importlib.machinery.ExtensionFileLoader(full, path)
        spec = importlib.util.spec_from_loader(full, loader, origin=path)
        module = importlib.util.module_from_spec(spec)
        _exec_binding(loader, module)
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
    mode = (default_mode() if mode is None else mode).strip().lower()
    load_set(mode)
    prev = default_mode()
    _DEFAULT_MODE = mode
    return prev


def binding(name, mode=None):
    """Resolve a binding and reject a readable compiled-mode mismatch at call time."""
    requested = default_mode() if mode is None else mode
    if not isinstance(requested, str) or requested.strip().lower() not in _MODE_CODE:
        raise ValueError("numeric_mode must be fast, deterministic, identical or None")
    requested = requested.strip().lower()
    if name in _IDENTICAL_ONLY and requested != "identical":
        raise ValueError(
            f"mojolearn: {name} has no {requested!r} tier. "
            f"{_identical_only_reason(name)} "
            "Drop numeric_mode= (identical is the default) or pass "
            "numeric_mode='identical'."
        )
    # A CPU-ONLY INSTALL NEVER REACHES load_set: there is no tier directory
    # to open, and `_layout()` would raise the no-GPU refusal again. The
    # module the selector installed (a host binding proxy, or a stub that
    # refuses by name) is the answer, and it is IDENTICAL only.
    if _CPU_ONLY is not None:
        return _cpu_only_binding(name, requested)
    selected = load_set(requested)
    module = getattr(selected, name)
    # A correctly built GBDT sibling does not establish RF/ET (or any other
    # extension) mode. Check the actual called module, including cached sets.
    getter_name = _vendor_fn(name).removesuffix("_vendor") + "_numeric_mode"
    getter = getattr(module, getter_name, None)
    if getter is not None:
        compiled = _CODE_MODE.get(getter(), "unknown")
        if compiled != selected.mode:
            raise RuntimeError(
                f"mojolearn: {name} was compiled for {compiled}, but this call "
                f"requested {selected.mode}; rebuild the {selected.mode} binding"
            )
    return module


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


class _IdenticalOnlyTier(type(sys)):
    """Stands in for a neural binding under a tier that lane does not offer.

    Distinct from `_MissingUpperTier` on purpose. That one means "build it";
    this one means "there is nothing to build", and handing an operator
    `MOJOLEARN_NUMERIC_MODE=fast bash bindings/build_mamba.sh` would send them
    at a script that now exits 2. See `_IDENTICAL_ONLY`."""

    def __init__(self, full, name, mode):
        super().__init__(full)
        self.__name = name
        self.__mode = mode

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        raise ImportError(
            f"mojolearn: {self.__name} has no {self.__mode!r} tier. The neural "
            "lanes (transformer, mamba, training, byte LM) build IDENTICAL "
            "only: their fused kernels are gated on the identical contract, so "
            "the lower tiers ran the unfused path and were SLOWER than the "
            "default. Run with MOJOLEARN_NUMERIC_MODE=identical (the default) "
            "or pass numeric_mode='identical'."
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
        "_mojolearn_preprocessing": "build_preprocessing.sh",
        "_mojolearn_tsa": "build_tsa.sh",
        "_mojolearn_linalg": "build_linalg.sh",
        "_mojolearn_arima": "build_arima.sh",
        "_mojolearn_training": "build_training.sh",
        "_mojolearn_byte_lm": "build_byte_lm.sh",
        "_mojolearn_gp": "build_gp.sh",
        "_mojolearn_mamba": "build_mamba.sh",
        "_mojolearn_transformer": "build_transformer.sh",
        "_mojolearn_kernel_methods": "build_kernel_methods.sh",
        "_mojolearn_mixture": "build_mixture.sh",
        "_mojolearn_hdbscan": "build_hdbscan.sh",
        "_mojolearn_resample": "build_resample.sh",
        "_mojolearn_ivf": "build_ivf.sh",
        "_mojolearn_embedding": "build_embedding.sh",
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
    if _CPU_ONLY is not None:
        return "cpu"
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
        expected = _VENDOR_SELECTED
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
    if _CPU_ONLY is not None:
        return (
            "no GPU binary set loaded; the CPU bindings under mojolearn/host/ "
            f"only (DEVIATION 2615; built: {', '.join(host_families_built()) or 'none'})"
        )
    _layout()
    return _VENDOR_HOW
