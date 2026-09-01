# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn verify` -- turn the identity claim into a check.

WHAT THE CLAIM IS

mojolearn compiles one source for Metal, CUDA and HIP and, in IDENTICAL
mode, claims the FP32 result is the same bits on every one of them
(`IDENTITY_PATHS.md` is the enumeration of every pathway that can move a
bit and what IDENTICAL does about each). That claim currently lives in
forty markdown files and a directory of result artifacts, which means a
stranger has to read a ledger and believe it.

This module is the alternative. It runs ONE pinned fit, writes the stage
card that fit emits (`core/identity_trace.mojo`), and compares the card
against a reference card shipped in the wheel using the repository's own
comparator. A run either reproduces the reference bit for bit or it names
the first stage that did not.

WHAT ONE MACHINE CAN AND CANNOT PROVE, WHICH IS THE WHOLE HONESTY PROBLEM

A single GPU cannot check a cross-vendor property. What a local run proves
is that THIS build, on THIS device, reproduced a card that was produced
elsewhere. The cross-vendor content of the result is entirely in the
reference card's provenance, which is why the reference is required to
carry the run, the commit, the mode and the hardware that made it, and why
this command prints all four every time. A reference hash with no
provenance is a number, not evidence.

THE FAST BUILD MAKES NO IDENTITY CLAIM AT ALL

`_backend.py` selects the FAST extensions unless `MOJOLEARN_NUMERIC_MODE`
is `identical` at import time. The FAST arm is fast precisely because it
keeps the order-dependent operations IDENTICAL replaces (float atomics,
vendor transcendentals, whatever the compiler contracts), so two FAST runs
on two vendors are expected to differ and their agreement would be luck.
This command therefore REFUSES in FAST mode rather than reporting anything
green. That mirrors what the project's own round judge does with FAST
cards, which it records and never judges (`tools/e3_round_judge.sh`
section 7). DEVIATION 923.

DEVIATIONS RECORDED BY THIS FILE

  920  `_mojolearn` (the extension that runs this fixture) exposes no
       compile-time numeric-mode function, the way `_mojolearn_gbdt`
       exposes `gbdt_numeric_mode`. The arm of the binary that actually
       runs the fit is therefore INFERRED here from three legs rather
       than read from it. See `_mode_report`.
  921  The reference card is this command's own, not `e1u/kmeans.card`
       from a cross-vendor round. The E1U card is produced by
       `bench/unsupervised_trace_main.mojo`, which calls
       `kmeans_fit_main` with a PINNED `sum_scale` of 4096; the Python
       surface goes through `cluster/estimator.mojo::kmeans_fit`, which
       derives `sum_scale` from the data with `plan_sum_scale`. Different
       fixed-point scale, different accumulator bits, different card,
       even on one machine. Reusing the E1U card here would have been a
       comparison of two different computations.
  922  The fixture INPUT is gated against the E1U round's recorded input
       hashes before any fit runs. The bytes fitted here are the same
       bytes three vendors fitted on 2026-08-23, and that is checkable
       without a GPU.
  923  FAST mode refuses. See above.
  924  ONE COMPARATOR. This module imports `tools/identity_trace_diff.py`
       and refuses to run if it cannot find it. It does not contain a
       second card comparator, because two comparators that disagree is
       the exact class of defect this repository exists to avoid.
  925  A shipped placeholder reference exits NO-REFERENCE, never
       VERIFIED. A placeholder that silently passed would be worse than
       shipping nothing. The whole produce/confirm/install path is now
       three one-command steps -- `verify --emit-reference` on the
       producing box, `verify --confirm-reference` on the second box,
       `install-reference` for the pair -- and the install refuses the
       FILL-IN token, a profile mismatch, missing provenance, and a
       divergent pair (DEVIATION 927). docs/VERIFY.md, 'Regenerating the
       reference card'.
  926  `sample_weight` is pinned to None rather than an array of ones, so
       the weight bound is exactly `n_samples` and no host float
       reduction over caller weights enters the fixed-point scale.
  927  `install-reference` is refusal-first; see `cmd_install_reference`.
       The docs/VERIFY.md provenance block is PRINTED for a human to
       paste, never written by the command, so the doc stays audited by a
       reader rather than trusted to a tool.
  929  The compiled `.so` binding artifacts carry no build stamp, so
       nothing ties a binary to the commit that built it. Candidate cards
       and conformance bundles hash the loaded artifacts and say
       `source-commit unverified` outright (`binding_artifacts`). The
       closing mechanism is specced in docs/CONFORMANCE.md and owned by
       the bindings lane.

`_conformance.py` (DEVIATIONS 928 and 930) packages this fixture as a
portable conformance bundle; docs/CONFORMANCE.md is its document.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

# --------------------------------------------------------------------------
# EXIT CODES. People put this in continuous integration, so they are part of
# the interface and are documented in docs/VERIFY.md.
# --------------------------------------------------------------------------

EXIT_VERIFIED = 0
"""The card matched the reference, stage for stage."""

EXIT_MISMATCH = 1
"""The fit ran and the card did NOT match. A first diverging stage is named."""

EXIT_USAGE = 2
"""Bad arguments, or a broken input this command cannot interpret. Matches
`tools/identity_trace_diff.py`'s own convention, where 2 is usage/parse."""

EXIT_REFUSED_FAST = 3
"""This process loaded the FAST binaries. No identity claim applies to them
and nothing was judged."""

EXIT_CANNOT_RUN = 4
"""The check could not be attempted here. No GPU, no identical binaries, the
fit raised, or the trace never reached the binary."""

EXIT_NO_REFERENCE = 5
"""No usable reference card is present in this installation, or the one that
is present is still the shipped placeholder."""

_CODE_NAMES = {
    EXIT_VERIFIED: "VERIFIED",
    EXIT_MISMATCH: "MISMATCH",
    EXIT_USAGE: "USAGE",
    EXIT_REFUSED_FAST: "REFUSED (FAST BUILD)",
    EXIT_CANNOT_RUN: "CANNOT RUN",
    EXIT_NO_REFERENCE: "NO REFERENCE",
}


# --------------------------------------------------------------------------
# THE FIXTURE
#
# It is the E1U unsupervised k-means fixture, coordinate for coordinate:
# `bench/unsupervised_trace_main.mojo`, `_run_kmeans`. Reused rather than
# invented for three reasons.
#
#   1. IT IS ALREADY PINNED AND ALREADY CROSSED THREE VENDORS. Its input
#      hashes are recorded in every E1U leg's `kmeans.hashes`, and the two
#      constants below are copied from
#      bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/e1u/kmeans.hashes.
#      A new fixture would have had none of that behind it.
#   2. EVERY COORDINATE IS EXACT IN FLOAT32. `k / 256` for an integer k
#      below 2^16 has an exact binary representation, so no backend can
#      round the INPUT differently and a card difference is always a
#      difference in the computation. This is the property a cross-vendor
#      fixture actually needs.
#   3. IT REACHES THE MECHANISM RATHER THAN A CONVENIENCE. The k-means
#      centroid accumulation is IDENTITY_PATHS' REPLACE move in its purest
#      form: a float atomic sum swapped for a fixed-point Int32 accumulator
#      (rows 19-21, deviations 503/504/508). The card records
#      `sums_i32` and `weight_i32` per iteration, so the thing being
#      compared IS the thing that makes the claim true. A fixture that
#      exercised only elementwise arithmetic would agree everywhere and
#      prove nothing.
#
# WHY K-MEANS AND NOT GRADIENT BOOSTING. GradientBoosting is the flagship
# and it also emits a card (`gbdt/train.mojo`), but a GBDT fit large enough
# to be interesting takes far longer than the thirty seconds this command is
# budgeted, and its card runs to thousands of stages. The whole-library
# cross-vendor record is E3_RESULTS.md; this command is a doorway, not a
# replacement for it, and it says so in its own output.
# --------------------------------------------------------------------------

PROFILE = "mojolearn.identical.verify.kmeans.fp32.v1"
REFERENCE_NAME = "kmeans.identical.fp32.v1"

KM_N = 4096
KM_D = 8
KM_K = 8
KM_ITERS = 10
KM_TOL = 1e-4
KM_SEED = 7
KM_SALT = 7
KM_CENTROID_STRIDE = 37

#: FNV-1a64 over the raw little-endian bits of the fixture, in index order.
#: These are the numbers the E1U legs printed as `input.x` and
#: `input.centroids` on Apple, NVIDIA and AMD on 2026-08-23. They are a gate
#: (DEVIATION 922): a card comparison against a different fixture measures
#: nothing, and this is the cheapest place to find that out.
FIXTURE_X_FNV1A64 = 14006717752511810141
FIXTURE_C_FNV1A64 = 2727533609010192784

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _mix(i, f, salt):
    """`_mix` from `bench/unsupervised_trace_main.mojo`, digit for digit.

    Integer arithmetic only, so it is the same number in Mojo, in Python and
    on any machine. Written with explicit 64-bit masking rather than numpy
    unsigned types because numpy will silently promote a uint64 to float64
    when it meets a Python int, and a fixture generator that is approximately
    right is a fixture generator that is wrong.
    """
    h = ((i + 1) * 0x9E3779B97F4A7C15 + (f + salt) * 0xBF58476D1CE4E5B9) & _MASK64
    h ^= h >> 29
    h = (h * 0x94D049BB133111EB) & _MASK64
    return (h ^ (h >> 32)) & _MASK64


def _coord(i, f, salt):
    """`_coord`: a 16-bit integer over 256. Exact in Float32."""
    return (_mix(i, f, salt) & 0xFFFF) / 256.0


def build_fixture():
    """Return (X, init_centroids), float32 and C contiguous.

    The centroids are ROWS OF X, which is what `INIT_ARRAY` means upstream
    and what makes the start of the fit exact rather than sampled. Row
    `c * 37` for cluster c, exactly as the E1U driver picks them.
    """
    import numpy as np

    x = np.empty((KM_N, KM_D), dtype=np.float32)
    for i in range(KM_N):
        row = x[i]
        for f in range(KM_D):
            row[f] = _coord(i, f, KM_SALT)
    c = np.empty((KM_K, KM_D), dtype=np.float32)
    for k in range(KM_K):
        row = c[k]
        for f in range(KM_D):
            row[f] = _coord(k * KM_CENTROID_STRIDE, f, KM_SALT)
    return x, c


def _le_bytes(arr):
    """Raw little-endian bytes, in index order. The card's hash is over
    exactly these, so the comparison must be over exactly these."""
    return arr.astype("<f4", copy=False).tobytes()


# --------------------------------------------------------------------------
# THE COMPARATOR. Imported, never reimplemented (DEVIATION 924).
# --------------------------------------------------------------------------

_DIFF_ATTRS = ("parse_trace", "align", "run", "fnv1a64", "TraceError")
_DIFFER_CACHE = []
"""Resolved once. Loading the same file twice would give two module objects
and, on a bad day, two answers; and the fixture gate calls this before the
comparison does."""


def load_differ():
    """Return (module, path, how) for `tools/identity_trace_diff.py`.

    Search order, most explicit first.

      1. `MOJOLEARN_IDENTITY_TRACE_DIFF`, an explicit path. For a maintainer
         checking a candidate comparator against a shipped one.
      2. `mojolearn._identity_trace_diff`, the build-time copy. The wheel has
         no `tools/` directory, so the release build has to place one; see
         docs/VERIFY.md for the packaging stanza. It is a COPY OF ONE FILE
         made at build time, not a second implementation.
      3. `tools/identity_trace_diff.py` in a checkout above this package.
         This is the path a developer running from the repository takes.

    Raises RuntimeError when none is found. It does NOT fall back to a local
    comparator, because there is no local comparator to fall back to and
    writing one is the mistake this function exists to prevent.
    """
    if _DIFFER_CACHE:
        return _DIFFER_CACHE[0]

    explicit = os.environ.get("MOJOLEARN_IDENTITY_TRACE_DIFF", "").strip()
    tried = []
    if explicit:
        tried.append(explicit)
        mod = _load_by_path(explicit)
        if mod is not None:
            return _cache((mod, explicit, "MOJOLEARN_IDENTITY_TRACE_DIFF"))

    try:
        mod = importlib.import_module(
            __name__.rsplit(".", 1)[0] + "._identity_trace_diff")
    except ImportError:
        tried.append("<package>._identity_trace_diff (not in this install)")
    else:
        if _differ_is_whole(mod):
            return _cache(
                (mod, getattr(mod, "__file__", "<package>"), "wheel copy"))

    here = os.path.dirname(os.path.abspath(__file__))
    base = here
    for _ in range(6):
        base = os.path.dirname(base)
        if not base or base == os.path.dirname(base):
            break
        cand = os.path.join(base, "tools", "identity_trace_diff.py")
        tried.append(cand)
        mod = _load_by_path(cand)
        if mod is not None:
            return _cache((mod, cand, "checkout"))

    raise RuntimeError(
        "mojolearn verify cannot find tools/identity_trace_diff.py, which is "
        "the only card comparator this repository has. It will not compare "
        "cards with anything else. Looked at:\n  "
        + "\n  ".join(tried)
        + "\nPoint MOJOLEARN_IDENTITY_TRACE_DIFF at the file, or install a "
        "wheel built with the packaging step that copies it in (docs/VERIFY.md)."
    )


def _cache(resolved):
    _DIFFER_CACHE.append(resolved)
    return resolved


def _load_by_path(path):
    if not path or not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(
        "mojolearn_identity_trace_diff", path
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod if _differ_is_whole(mod) else None


def _differ_is_whole(mod):
    """A file at the right path is not the right file. Check the API this
    module actually uses before trusting a verdict from it."""
    return all(hasattr(mod, a) for a in _DIFF_ATTRS)


# --------------------------------------------------------------------------
# THE MODE GATE (DEVIATIONS 920 and 923)
# --------------------------------------------------------------------------


class ModeReport(object):
    __slots__ = ("requested", "loaded", "compiled", "binary", "ident_dir",
                 "under_identical", "problem")

    def __init__(self):
        self.requested = None
        self.loaded = None
        self.compiled = None      # what _mojolearn_gbdt says about ITSELF
        self.binary = None        # the .so that runs THIS fixture
        self.ident_dir = None
        self.under_identical = None
        self.problem = None

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def _mode_report():
    """Everything known about which arm this process is running.

    DEVIATION 920, stated plainly. The fixture runs through `_mojolearn`,
    and `_mojolearn` has no `numeric_mode()` of its own the way
    `_mojolearn_gbdt` has `gbdt_numeric_mode`. So the arm of the binary that
    does the work is not read from that binary. It is inferred from three
    legs, and all three have to agree.

      1. the selector's answer (`_backend.numeric_mode()`), which is what
         the process ASKED for and got;
      2. the gbdt binary's own compile-time answer, which `_backend`
         cross-checks against the selector and raises on. That binary is a
         SIBLING of the one running the fixture, loaded from the same
         directory in the same call, so it is evidence about the directory;
      3. the resolved path of the loaded `_mojolearn` extension. In
         identical mode it must sit under `<package>/identical/`. This is
         the only leg that touches the actual fixture binary.

    Three indirect legs is weaker than one direct read. The fix is four
    lines in `bindings/_mojolearn.mojo` and it is named in the report that
    accompanied this file rather than done here, because bindings are not
    this lane's to edit.
    """
    rep = ModeReport()
    rep.requested = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    if not rep.requested:
        rep.requested = "fast"

    try:
        from . import _backend
    except Exception as exc:            # pragma: no cover - import-time only
        rep.problem = "cannot import mojolearn._backend (%s)" % exc
        return rep

    try:
        rep.loaded = _backend.numeric_mode()
    except Exception as exc:
        # _backend raises when the loaded gbdt binary was compiled for the
        # other arm. That is exactly the mislabelled-measurement failure it
        # exists to prevent, and it must not be swallowed.
        rep.problem = str(exc)
        return rep

    pkg_name = __name__.rsplit(".", 1)[0]
    gb = sys.modules.get(pkg_name + "._mojolearn_gbdt")
    try:
        _readable = gb is not None and hasattr(gb, "gbdt_numeric_mode")
    except ImportError:
        # A stub raises ImportError from __getattr__, and `hasattr` only
        # swallows AttributeError. See `_backend.numeric_mode`.
        _readable = False
    if _readable:
        try:
            # Three tiers, so a NAME lookup rather than a boolean:
            # DETERMINISTIC reports 2 and the old spelling called it
            # "fast", which agrees with a fast selector and reports no
            # conflict while the caller holds the wrong arm.
            rep.compiled = _backend._CODE_MODE.get(
                gb.gbdt_numeric_mode(), "unknown"
            )
        except Exception:
            rep.compiled = None

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    rep.ident_dir = os.path.realpath(os.path.join(pkg_dir, "identical"))
    fixture_mod = sys.modules.get(pkg_name + "._mojolearn")
    rep.binary = getattr(fixture_mod, "__file__", None)
    if rep.binary:
        real = os.path.realpath(rep.binary)
        rep.under_identical = (
            real == rep.ident_dir
            or real.startswith(rep.ident_dir + os.sep)
        )

    if rep.loaded == "identical":
        if fixture_mod is not None and type(fixture_mod).__name__ == \
                "_MissingIdentical":
            # `_backend.select()` installs this stub when an identical binary
            # was not built, so the package still imports and every OTHER
            # family keeps working. It raises by name on first use. Reaching
            # it here means the extension this fixture needs does not exist
            # in the identical set, which is a build gap and not a result.
            rep.problem = (
                "MOJOLEARN_NUMERIC_MODE=identical, but the identical build of "
                "the extension this fixture runs on was never built; "
                "_backend installed its missing-binary stub instead. Build it "
                "with\n    MOJOLEARN_NUMERIC_MODE=identical bash "
                "bindings/build.sh"
            )
        elif rep.binary is None:
            rep.problem = (
                "the loaded numeric mode is identical but the extension that "
                "runs this fixture (<package>._mojolearn) has no resolvable "
                "file, so which binary it is cannot be established"
            )
        elif not rep.under_identical:
            rep.problem = (
                "the selector reports identical but the extension running "
                "this fixture is %s, which is NOT under %s. A binary is in "
                "the wrong directory; nothing here can be labelled."
                % (rep.binary, rep.ident_dir)
            )
    return rep


# --------------------------------------------------------------------------
# ENVIRONMENT. Everything that legitimately changes the answer, captured so a
# mismatch can be attributed instead of argued about.
# --------------------------------------------------------------------------


def _cmd(args, timeout=5):
    try:
        out = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except Exception:
        return None
    text = out.stdout.decode("utf-8", "replace").strip()
    return text or None


def describe_device():
    """A best-effort device name. Never fails, never blocks for long.

    This is a LABEL, not a measurement. Nothing in the verdict depends on
    it. It exists so that two people comparing a mismatch are talking about
    the same hardware.
    """
    system = platform.system()
    if system == "Darwin":
        model = _cmd(["sysctl", "-n", "hw.model"])
        chip = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        parts = [p for p in (chip, model) if p]
        return " / ".join(parts) if parts else "Apple (unidentified)"
    nv = _cmd(["nvidia-smi",
               "--query-gpu=name,driver_version",
               "--format=csv,noheader"])
    if nv:
        return "NVIDIA " + nv.splitlines()[0].strip()
    amd = _cmd(["rocm-smi", "--showproductname", "--csv"])
    if amd:
        lines = [ln for ln in amd.splitlines() if ln.strip()]
        return "AMD " + (lines[-1].strip() if lines else "(rocm-smi)")
    return "%s %s (no vendor tool answered)" % (system, platform.machine())


def _git_commit(start):
    """The commit, when this package sits in a checkout. `unknown` otherwise,
    and `unknown` is a legitimate answer that must not be dressed up: a wheel
    install has no `.git` and inventing a commit would be the worst kind of
    provenance."""
    base = start
    for _ in range(6):
        if os.path.isdir(os.path.join(base, ".git")):
            rev = _cmd(["git", "-C", base, "rev-parse", "HEAD"])
            if rev:
                dirty = _cmd(["git", "-C", base, "status", "--porcelain"])
                return rev.strip() + (" (WORKING TREE DIRTY)" if dirty else "")
            return None
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return None


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def environment(mode=None):
    """The block printed on every verdict."""
    here = os.path.dirname(os.path.abspath(__file__))
    if mode is None:
        mode = _mode_report()
    try:
        from ._version import __version__ as version
    except Exception:                    # pragma: no cover
        version = "unknown"
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:                    # pragma: no cover
        numpy_version = "unavailable"
    return {
        "profile": PROFILE,
        "mojolearn_version": version,
        "numpy_version": numpy_version,
        "python": "%d.%d.%d %s" % (sys.version_info[:3] + (platform.python_implementation(),)),
        "platform": "%s %s %s" % (platform.system(), platform.release(),
                                  platform.machine()),
        "device": describe_device(),
        "commit": _git_commit(here) or "unknown (no checkout beside this package)",
        "mode": mode,
        "vendor": _vendor_text(),
    }


def _vendor_text():
    """'cuda (the box probe ...)' and so on: what `_backend.vendor()` read
    back from the loaded binaries, and how the directory was chosen. Printed
    beside the numeric mode on every verdict since 2026-08-29, because on
    Linux the wheel carries two vendors' sets and a card must say which one
    produced it. A failure to establish it is reported, never hidden."""
    try:
        from . import _backend
        v = _backend.vendor()
        how = _backend.vendor_how()
    except Exception as exc:
        return "unknown (%s: %s)" % (type(exc).__name__, exc)
    if v is None:
        return "unknown (binaries predate the vendor read-back; %s)" % how
    return "%s (%s)" % (v, how)


# --------------------------------------------------------------------------
# THE REFERENCE CARD
# --------------------------------------------------------------------------

PLACEHOLDER_TOKEN = "FILL-IN"
"""Any reference card still carrying this string is a placeholder and is
refused (DEVIATION 925). The token is checked, not the filename, so copying
the placeholder into place under the real name does not defeat it."""


def reference_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "reference_cards")


def reference_path(name=REFERENCE_NAME):
    return os.path.join(reference_dir(), name + ".card")


def read_reference(name=REFERENCE_NAME):
    """Return (path, provenance_lines, n_records). Raises _NoReference."""
    path = reference_path(name)
    if not os.path.isfile(path):
        ph = path + ".PLACEHOLDER"
        if os.path.isfile(ph):
            raise _NoReference(
                "this installation ships NO reference card. What is present "
                "is the placeholder\n    %s\nwhich exists to make this "
                "failure loud rather than let a build pass with nothing "
                "behind it. docs/VERIFY.md, 'Regenerating the reference "
                "card', is the procedure." % ph)
        raise _NoReference(
            "no reference card at\n    %s\nand no placeholder beside it "
            "either. This installation was built without the packaging step "
            "that ships mojolearn/reference_cards/ (docs/VERIFY.md)." % path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise _NoReference("cannot read %s (%s)" % (path, exc))
    if PLACEHOLDER_TOKEN in text:
        raise _NoReference(
            "the reference card at\n    %s\nstill contains the placeholder "
            "token %r. It was copied into place but never generated. "
            "docs/VERIFY.md, 'Regenerating the reference card'."
            % (path, PLACEHOLDER_TOKEN))
    provenance = [ln[1:].strip() for ln in text.split("\n")
                  if ln.startswith("#")]
    records = [ln for ln in text.split("\n")
               if ln and not ln.startswith("#")]
    if not records:
        raise _NoReference(
            "the reference card at\n    %s\nholds no records. An empty card "
            "compares equal to nothing and must never be treated as a "
            "reference." % path)
    return path, provenance, len(records)


class _NoReference(Exception):
    pass


def parse_card_comments(path):
    """Return (kv, comment_lines, n_records) for a card file.

    `kv` maps `key` -> `value` for every `# key: value` comment line (first
    colon splits; later lines with a repeated key win, which cannot happen
    in cards this package emits). Free-form comment lines are carried in
    `comment_lines` untouched. Used by `install-reference` and by
    `conformance export --from-card`, both of which have to READ a card's
    stated provenance before trusting its records.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    kv = {}
    comments = []
    n_records = 0
    for ln in text.split("\n"):
        if ln.startswith("#"):
            body = ln[1:].strip()
            comments.append(body)
            if ":" in body:
                key, _, value = body.partition(":")
                key = key.strip()
                # Only plausible keys, so prose containing a colon does not
                # pollute the map ("READ THIS BEFORE..." lines).
                if key and " " not in key:
                    kv[key] = value.strip()
        elif ln.strip():
            n_records += 1
    return kv, comments, n_records


def binding_artifacts():
    """Every loaded mojolearn binding extension, hashed. DEVIATION 929.

    The compiled `.so` artifacts carry NO freshness signal: nothing ties one
    to the commit that built it, and a stale binary raises only when a
    signature happens to have drifted. So provenance records what CAN be
    established honestly -- the file's SHA-256, size and mtime at run time --
    and states `artifact_source_commit: "unverified"` outright rather than
    implying the repo commit covers the binaries. The closing mechanism (a
    compile-time commit+mode stamp read back at import, the way the vendor
    read-back already works) belongs to the bindings lane; see
    docs/CONFORMANCE.md, 'Artifact provenance'.
    """
    pkg = __name__.rsplit(".", 1)[0]
    out = []
    for name in sorted(sys.modules):
        if not name.startswith(pkg + "._mojolearn"):
            continue
        mod = sys.modules[name]
        path = getattr(mod, "__file__", None)
        if not path or not path.endswith(".so"):
            continue        # missing-binary stubs have no file; skip honestly
        try:
            st = os.stat(path)
        except OSError:
            continue
        out.append({
            "module": name.rsplit(".", 1)[1],
            "file": path,
            "sha256": _sha256(path),
            "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   time.gmtime(st.st_mtime)),
            "artifact_source_commit": "unverified",
        })
    return out


# --------------------------------------------------------------------------
# RUNNING THE FIXTURE
# --------------------------------------------------------------------------


class FixtureError(Exception):
    """The fit could not be run or produced no card. Maps to CANNOT RUN."""


def run_fixture(card_path, dump=None):
    """Fit the pinned k-means with the trace pointed at `card_path`.

    `dump` is a tag substring for MOJOLEARN_IDENTITY_TRACE_DUMP; None (the
    default, and what `verify` passes) clears it, keeping this function's
    original behavior. `conformance export` passes "." -- every tag this
    fixture emits contains a dot -- because a bundle needs the raw stage
    bytes, not only their hashes.

    ONE TRACED FIT PER PROCESS, which is the differ's contract and not a
    convenience: the tags carry `restartNN.iterMM.` prefixes that repeat
    across fits, and `core/identity_trace.mojo` RAISES on a duplicate tag.
    This function runs exactly one and restores the environment after it.

    The trace path reaches the binary through the ENVIRONMENT, which is how
    `IdentityTrace()` reads it and how a user would set it. CPython's
    `os.environ` assignment calls `putenv`, so the C `getenv` the Mojo side
    uses sees it. If that ever stops being true the card comes back empty
    and this function says so by name rather than reporting a mismatch.
    """
    import importlib as _il
    pkg = _il.import_module(__name__.rsplit(".", 1)[0])

    x, c = build_fixture()

    xh = _fnv_of(_le_bytes(x))
    ch = _fnv_of(_le_bytes(c))
    if xh != FIXTURE_X_FNV1A64 or ch != FIXTURE_C_FNV1A64:
        raise FixtureError(
            "THE FIXTURE IS NOT THE PINNED ONE (DEVIATION 922).\n"
            "  X          got %d, want %d\n"
            "  centroids  got %d, want %d\n"
            "Those two targets are what the E1U legs printed as input.x and "
            "input.centroids on Apple, NVIDIA and AMD. A card comparison "
            "against different input bytes measures nothing, so this stops "
            "here. Either the generator in _verify.py does not match "
            "bench/unsupervised_trace_main.mojo::_coord, or numpy produced "
            "something other than little-endian float32."
            % (xh, FIXTURE_X_FNV1A64, ch, FIXTURE_C_FNV1A64))

    saved_trace = os.environ.get("MOJOLEARN_IDENTITY_TRACE")
    saved_dump = os.environ.get("MOJOLEARN_IDENTITY_TRACE_DUMP")
    os.environ["MOJOLEARN_IDENTITY_TRACE"] = card_path
    if dump is None:
        # NO .bin DUMPS on the verify path. Cell-level diagnosis needs dumps
        # on BOTH sides and the reference ships without them (they would be
        # megabytes per card). A dump here would only slow the fit down and
        # mislead the differ's integrity step into checking one side of a
        # pair.
        os.environ.pop("MOJOLEARN_IDENTITY_TRACE_DUMP", None)
    else:
        os.environ["MOJOLEARN_IDENTITY_TRACE_DUMP"] = dump
    try:
        est = pkg.KMeans(
            n_clusters=KM_K,
            init="array",
            init_centroids=c,
            n_init=1,
            max_iter=KM_ITERS,
            tol=KM_TOL,
            random_state=KM_SEED,
        )
        started = time.time()
        try:
            # DEVIATION 926: sample_weight stays None. Unit weights make the
            # weight bound exactly n_samples, so `choose_scale` gets an
            # integer rather than a host float64 reduction over an array.
            est.fit(x)
        except Exception as exc:
            raise FixtureError(
                "the pinned fit raised, so there is nothing to compare.\n"
                "  %s: %s\n"
                "On a machine with no GPU this is the expected failure and "
                "the answer is CANNOT RUN, not MISMATCH."
                % (type(exc).__name__, exc))
        elapsed = time.time() - started
    finally:
        _restore("MOJOLEARN_IDENTITY_TRACE", saved_trace)
        _restore("MOJOLEARN_IDENTITY_TRACE_DUMP", saved_dump)

    if not os.path.isfile(card_path) or os.path.getsize(card_path) == 0:
        raise FixtureError(
            "the fit completed but wrote NO stage card to\n    %s\n"
            "MOJOLEARN_IDENTITY_TRACE did not reach the extension. Nothing "
            "was compared and nothing is claimed. This is a defect in the "
            "instrument, not a result about the hardware." % card_path)
    return est, elapsed, xh, ch


def _restore(key, value):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _fnv_of(data):
    """FNV-1a64 through the differ's implementation, so the fixture gate and
    the card comparison use ONE hash function. Falls back to a local loop
    only when the differ is unavailable, in which case nothing is being
    compared anyway."""
    try:
        differ, _, _ = load_differ()
    except RuntimeError:
        h = 0xCBF29CE484222325
        for b in data:
            h = ((h ^ b) * 0x100000001B3) & _MASK64
        return h
    return differ.fnv1a64(data)


# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------

_RULE = "=" * 74
_THIN = "-" * 74


def _kv(key, value):
    return "  %-16s %s" % (key, value)


def _proves_block():
    """Printed on EVERY verdict, green included. The point of this command
    is to move a claim from belief to property, so it must not itself
    overclaim, and the limits are not a footnote."""
    return [
        "WHAT THIS RUN PROVES",
        "  That this build, on this device, in this numeric mode, reproduced",
        "  the reference card stage for stage. The card is a sequence of",
        "  FNV-1a64 hashes over raw FP32 bit patterns at named points inside",
        "  one fit.",
        "",
        "WHAT IT DOES NOT PROVE",
        "  1. Nothing about a second vendor. One machine cannot check a",
        "     cross-vendor property. The cross-vendor content of this result",
        "     is entirely in where the reference card came from, printed",
        "     above, and in E3_RESULTS.md.",
        "  2. Nothing about stages the trace does not checkpoint, and not",
        "     that the computation was identical. Matching hashes mean the",
        "     buffers agreed at those checkpoints.",
        "  3. Nothing about the rest of the library. This is one k-means fit.",
        "     IDENTITY_PATHS.md enumerates the pathways; E3_RESULTS.md is the",
        "     whole-library record across three vendors.",
    ]


def _emit(lines):
    sys.stdout.write("\n".join(lines) + "\n")


def _env_lines(env, ref=None, comparator=None):
    mode = env["mode"]
    if mode.loaded is None:
        mode_text = "unknown (%s)" % (mode.problem or "not established")
    else:
        detail = "requested %s" % mode.requested
        if mode.compiled:
            detail += ", gbdt binary reports %s" % mode.compiled
        mode_text = "%s (%s)" % (mode.loaded.upper(), detail)
    lines = [
        _RULE,
        "mojolearn verify",
        _RULE,
        _kv("profile", env["profile"]),
        _kv("fixture", "k-means n=%d d=%d k=%d, init=array, max_iter=%d, "
                       "tol=%g, seed=%d" % (KM_N, KM_D, KM_K, KM_ITERS,
                                            KM_TOL, KM_SEED)),
        _kv("package", "mojolearn %s (numpy %s)" % (env["mojolearn_version"],
                                                    env["numpy_version"])),
        _kv("numeric mode", mode_text),
        _kv("vendor", env.get("vendor", "not established")),
        _kv("fixture binary", mode.binary or "not established"),
        _kv("python", env["python"]),
        _kv("host", env["platform"]),
        _kv("device", env["device"]),
        _kv("commit", env["commit"]),
    ]
    if comparator:
        lines.append(_kv("comparator", comparator[0]))
        lines.append(_kv("", "sha256 %s" % (comparator[1] or "unreadable")))
    if ref:
        lines.append(_kv("reference", ref["path"]))
        lines.append(_kv("", "sha256 %s" % (ref["sha256"] or "unreadable")))
        # THE PROVENANCE IS PRINTED IN FULL, every comment line the reference
        # carries. A reference hash with no provenance is a number, not
        # evidence, and burying it behind a flag would be the same thing.
        for ln in ref.get("provenance", []):
            if ln:
                lines.append(_kv("", ln))
    return lines


def _refused_fast(env):
    lines = _env_lines(env)
    lines += [
        _THIN,
        "REFUSED. This process loaded the FAST binaries.",
        "",
        "  The FAST arm makes NO cross-vendor claim and nothing about it was",
        "  judged. It is fast because it keeps exactly the order-dependent",
        "  operations the IDENTICAL arm replaces, so two FAST runs on two",
        "  vendors are expected to differ and any agreement between them is",
        "  luck rather than a property. Reporting a pass here would convert a",
        "  checkable property into a false comfort, which is the one thing",
        "  IDENTITY_PATHS.md says a toggle must never do.",
        "",
        "  To check the claim, set the mode BEFORE the package is imported.",
        "",
        "      MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify",
        "",
        "  The mode is read at import time by mojolearn/_backend.py, so",
        "  setting it inside an already-running interpreter does nothing.",
        _RULE,
        "RESULT: REFUSED (FAST BUILD). exit %d" % EXIT_REFUSED_FAST,
        _RULE,
    ]
    return lines


def _mode_problem(env):
    lines = _env_lines(env)
    lines += [
        _THIN,
        "CANNOT RUN. The numeric mode of this process cannot be established.",
        "",
        "  " + (env["mode"].problem or "unknown"),
        "",
        "  Nothing was fitted and nothing is claimed. A measurement whose arm",
        "  is unknown is a mislabelled measurement, which is worse than no",
        "  measurement.",
        _RULE,
        "RESULT: CANNOT RUN. exit %d" % EXIT_CANNOT_RUN,
        _RULE,
    ]
    return lines


# --------------------------------------------------------------------------
# THE COMMANDS
# --------------------------------------------------------------------------


def cmd_env(args):
    """Print the environment block and stop. No GPU is touched."""
    env = environment()
    if args.json:
        payload = dict(env)
        payload["mode"] = env["mode"].as_dict()
        _emit([json.dumps(payload, indent=2, sort_keys=True)])
        return EXIT_VERIFIED
    _emit(_env_lines(env) + [_RULE])
    return EXIT_VERIFIED


def cmd_check_fixture(args):
    """Regenerate the fixture and check it against the pinned input hashes.

    NO GPU IS NEEDED and no extension is called. This is the cheapest thing
    in the file and it is the first thing to run after any edit to `_mix` or
    `_coord`, because a wrong fixture makes every later verdict meaningless
    (DEVIATION 922).
    """
    try:
        x, c = build_fixture()
    except Exception as exc:
        _emit(["cannot build the fixture: %s: %s" % (type(exc).__name__, exc)])
        return EXIT_CANNOT_RUN
    xh = _fnv_of(_le_bytes(x))
    ch = _fnv_of(_le_bytes(c))
    ok = (xh == FIXTURE_X_FNV1A64 and ch == FIXTURE_C_FNV1A64)
    if args.json:
        _emit([json.dumps({
            "x_fnv1a64": xh, "x_expected": FIXTURE_X_FNV1A64,
            "centroids_fnv1a64": ch, "centroids_expected": FIXTURE_C_FNV1A64,
            "match": ok,
        }, indent=2, sort_keys=True)])
        return EXIT_VERIFIED if ok else EXIT_MISMATCH
    lines = [
        _RULE,
        "mojolearn verify --check-fixture",
        _RULE,
        "  The pinned E1U k-means fixture, rebuilt here and hashed. The two",
        "  expected values are what Apple, NVIDIA and AMD each printed as",
        "  input.x and input.centroids on 2026-08-23.",
        "",
        "  X          %20d   want %20d   %s"
        % (xh, FIXTURE_X_FNV1A64, "OK" if xh == FIXTURE_X_FNV1A64 else "WRONG"),
        "  centroids  %20d   want %20d   %s"
        % (ch, FIXTURE_C_FNV1A64, "OK" if ch == FIXTURE_C_FNV1A64 else "WRONG"),
        _RULE,
    ]
    lines.append("RESULT: %s. exit %d"
                 % ("FIXTURE OK" if ok else "FIXTURE WRONG",
                    EXIT_VERIFIED if ok else EXIT_MISMATCH))
    lines.append(_RULE)
    _emit(lines)
    return EXIT_VERIFIED if ok else EXIT_MISMATCH


def _fail(args, env, code, headline, detail, ref=None, comparator=None):
    """One shape for every non-comparing outcome, in both output modes.

    The JSON path is not a decoration. Somebody is going to put this in
    continuous integration and branch on the verdict, and a run that could
    not be attempted has to be distinguishable there from a run that failed,
    which is why these carry their own exit codes rather than folding into 1.
    """
    if args.json:
        _emit([_json_verdict(env, code, ref=ref, comparator=comparator,
                             note=headline + " " + detail)])
        return code
    _emit(_env_lines(env, ref=ref, comparator=comparator) + [
        _THIN,
        "%s %s" % (_CODE_NAMES[code] + ".", headline),
        "",
    ] + ["  " + ln for ln in detail.split("\n")] + [
        _RULE,
        "RESULT: %s. exit %d" % (_CODE_NAMES[code], code),
        _RULE,
    ])
    return code


def cmd_verify(args):
    """The whole check. See docs/VERIFY.md."""
    mode = _mode_report()
    env = environment(mode)

    if mode.problem is not None:
        if args.json:
            _emit([_json_verdict(env, EXIT_CANNOT_RUN, note=mode.problem)])
        else:
            _emit(_mode_problem(env))
        return EXIT_CANNOT_RUN

    if mode.loaded != "identical":
        if args.json:
            _emit([_json_verdict(
                env, EXIT_REFUSED_FAST,
                note="the FAST arm makes no cross-vendor claim; set "
                     "MOJOLEARN_NUMERIC_MODE=identical before import")])
        else:
            _emit(_refused_fast(env))
        return EXIT_REFUSED_FAST

    try:
        differ, differ_path, differ_how = load_differ()
    except RuntimeError as exc:
        return _fail(args, env, EXIT_CANNOT_RUN, "No comparator.", str(exc))
    comparator = (differ_path + "  (%s)" % differ_how, _sha256(differ_path))

    if getattr(args, "confirm_reference", None):
        # BEFORE --emit-reference: on the second box the two flags combine,
        # --emit-reference naming where the local candidate lands.
        return _confirm_reference(args, env, comparator)

    if args.emit_reference:
        return _emit_reference(args, env, comparator)

    try:
        ref_path, provenance, ref_stages = read_reference(args.reference_name)
    except _NoReference as exc:
        return _fail(
            args, env, EXIT_NO_REFERENCE, "Nothing to compare against.",
            str(exc) + "\n\nThis is not a result about the hardware. It is a "
            "build that shipped without the thing the check needs.",
            comparator=comparator)
    ref = {"path": ref_path, "sha256": _sha256(ref_path),
           "provenance": provenance, "stages": ref_stages}

    tmpdir = tempfile.mkdtemp(prefix="mojolearn-verify-")
    card = os.path.join(tmpdir, REFERENCE_NAME + ".local.card")
    try:
        _est, elapsed, xh, ch = run_fixture(card)
    except FixtureError as exc:
        return _fail(args, env, EXIT_CANNOT_RUN, "The fixture did not run.",
                     str(exc), ref=ref, comparator=comparator)

    header = _env_lines(env, ref=ref, comparator=comparator) + [
        _THIN,
        "  fixture X          %20d  matches the pinned E1U input" % xh,
        "  fixture centroids  %20d  matches the pinned E1U input" % ch,
        "  fit                %.2f s" % elapsed,
    ]

    # THE COMPARISON ITSELF IS THE DIFFER'S, verdict and all (DEVIATION 924).
    # `run` prints its report to stdout and returns the exit code. Under
    # --json that report would sit in front of the JSON object and break
    # every parser, so it is captured and dropped there and ONLY there; the
    # human path prints it verbatim, because the first diverging stage is
    # the whole value of a mismatch.
    argv = [ref_path, card, "--labels", "REFERENCE,THIS", "--no-verify-dumps"]
    if args.all:
        argv.append("--all")
    if args.json:
        import io
        buf, real = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            rc = differ.run(argv)
        finally:
            sys.stdout = real
        differ_report = buf.getvalue()
    else:
        differ_report = None
        _emit(header)
        rc = differ.run(argv)

    if rc == 0:
        try:
            recs, _c, _b = differ.parse_trace(card)
            n = len(recs)
        except Exception:
            n = ref_stages
        tail = [
            _RULE,
            "RESULT: VERIFIED. exit %d" % EXIT_VERIFIED,
            "  %d stages, every one equal in tag, dtype, count and hash." % n,
            "",
        ] + _proves_block() + [_RULE]
        code = EXIT_VERIFIED
        _cleanup(tmpdir, card, keep=args.keep)
        kept = card if args.keep else None
        if kept:
            tail.insert(3, "  card kept at %s" % kept)
    elif rc == 2:
        tail = [
            _RULE,
            "RESULT: CANNOT RUN. exit %d" % EXIT_CANNOT_RUN,
            "  The comparator could not parse one of the two cards. That is a",
            "  broken artifact, not a divergence, and no claim follows from",
            "  it either way.",
            _RULE,
        ]
        code = EXIT_CANNOT_RUN
        kept = card
    else:
        tail = [
            _RULE,
            "RESULT: MISMATCH. exit %d" % EXIT_MISMATCH,
            "  This build did NOT reproduce the reference bits. The first",
            "  diverging stage is named above; that name is the address of",
            "  the problem and is worth far more than the fact of the",
            "  difference. A final-output comparison could not have produced",
            "  it, which is why the card exists.",
            "",
            "  Before concluding anything about the hardware, check the four",
            "  things above that legitimately change the answer.",
            "    numeric mode   must be IDENTICAL on both sides",
            "    commit         must match the reference's",
            "    package        a different build is a different program",
            "    fixture hashes both must match the pinned values",
            "",
            "  Local card kept at",
            "    %s" % card,
            "  Re-run the full comparator over it, with more detail, with",
            "    python3 tools/identity_trace_diff.py --all \\",
            "        %s \\" % ref_path,
            "        %s" % card,
            "",
        ] + _proves_block() + [_RULE]
        code = EXIT_MISMATCH
        kept = card

    if args.json:
        _emit([_json_verdict(env, code, ref=ref, comparator=comparator,
                             elapsed=elapsed, local_card=kept,
                             report=differ_report)])
    else:
        _emit(tail)
    return code


def _cleanup(tmpdir, card, keep):
    if keep:
        return
    try:
        os.remove(card)
        os.rmdir(tmpdir)
    except OSError:
        pass


def _json_verdict(env, code, ref=None, comparator=None, elapsed=None,
                  note=None, local_card=None, report=None):
    payload = {
        "verdict": _CODE_NAMES.get(code, str(code)),
        "exit_code": code,
        "profile": env["profile"],
        "mojolearn_version": env["mojolearn_version"],
        "numeric_mode": env["mode"].as_dict(),
        "python": env["python"],
        "platform": env["platform"],
        "device": env["device"],
        "commit": env["commit"],
        "fixture": {
            "n": KM_N, "d": KM_D, "k": KM_K, "max_iter": KM_ITERS,
            "tol": KM_TOL, "seed": KM_SEED,
            "x_fnv1a64": FIXTURE_X_FNV1A64,
            "centroids_fnv1a64": FIXTURE_C_FNV1A64,
        },
        "one_machine_caveat": (
            "a single GPU cannot check a cross-vendor property; this says "
            "only that this build reproduced the reference card, whose "
            "provenance is where the cross-vendor content lives"),
    }
    if ref:
        payload["reference"] = {k: ref[k] for k in ("path", "sha256", "stages")}
        payload["reference"]["provenance"] = ref.get("provenance", [])
    if comparator:
        payload["comparator"] = {"path": comparator[0], "sha256": comparator[1]}
    if elapsed is not None:
        payload["fit_seconds"] = round(elapsed, 3)
    if local_card:
        payload["local_card"] = local_card
    if note:
        payload["note"] = note
    if report:
        # The comparator's own report, verbatim. It carries the first
        # diverging stage, which is the only part of a mismatch worth acting
        # on, so it is not dropped just because the caller asked for JSON.
        payload["comparator_report"] = report
    return json.dumps(payload, indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# THE MAINTAINER PATH: producing a reference card
# --------------------------------------------------------------------------


def _write_candidate(out_path, env, comparator):
    """Run the fixture once and write a provenance-stamped candidate card.

    The shared engine of `--emit-reference` (the producing box) and
    `--confirm-reference` (the second box): both need exactly the same
    artifact, a card whose comment head carries everything the eventual
    reader will be asked to trust. Returns (stages, elapsed). Raises
    FixtureError. The caller has already gated the numeric mode.
    """
    mode = env["mode"]
    tmpdir = tempfile.mkdtemp(prefix="mojolearn-emit-ref-")
    card = os.path.join(tmpdir, "candidate.card")
    _est, elapsed, xh, ch = run_fixture(card)

    with open(card, "r", encoding="utf-8") as fh:
        body = fh.read()
    stages = len([ln for ln in body.split("\n")
                  if ln and not ln.startswith("#")])

    stamped = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    head = [
        "# mojolearn verify reference card",
        "# profile: %s" % PROFILE,
        "# produced-by: python -m mojolearn verify --emit-reference",
        "# produced-at: %s" % stamped,
        "# mojolearn-version: %s" % env["mojolearn_version"],
        "# commit: %s" % env["commit"],
        "# numeric-mode: %s" % mode.loaded,
        "# fixture-binary: %s" % (mode.binary or "unknown"),
        "# host: %s" % env["platform"],
        "# device: %s" % env["device"],
        "# python: %s" % env["python"],
        "# fixture: kmeans n=%d d=%d k=%d init=array max_iter=%d tol=%g seed=%d"
        % (KM_N, KM_D, KM_K, KM_ITERS, KM_TOL, KM_SEED),
        "# fixture-input-x-fnv1a64: %d" % xh,
        "# fixture-input-centroids-fnv1a64: %d" % ch,
        "# stages: %d" % stages,
        "# fit-seconds: %.2f" % elapsed,
        "# comparator-at-emit: %s sha256 %s" % comparator,
    ]
    # DEVIATION 929. The card certifies runs made through whatever .so
    # artifacts sat on disk, and nothing ties those to a commit. Hash them
    # into the head, with the gap stated, so a stale-binary card is at
    # least a DIAGNOSABLE stale-binary card.
    for art in binding_artifacts():
        head.append("# artifact: %s sha256 %s size %d mtime %s "
                    "source-commit unverified"
                    % (os.path.basename(art["file"]), art["sha256"],
                       art["size"], art["mtime"]))
    head += [
        "#",
        "# READ THIS BEFORE TRUSTING THE NUMBERS BELOW. This card is one",
        "# machine's output. It becomes evidence for a CROSS-VENDOR claim",
        "# only when the same card has been produced on a second and a third",
        "# vendor and the three agree, which is E3_RESULTS.md's job. Record",
        "# the peer runs in docs/VERIFY.md when you install this file.",
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(head) + "\n" + body)
    _cleanup(tmpdir, card, keep=False)
    return stages, elapsed


def _emit_reference(args, env, comparator):
    """`verify --emit-reference PATH`. The regeneration procedure, in code.

    A reference hash nobody knows how to reproduce becomes untouchable and
    then wrong, so the procedure is a subcommand rather than a paragraph.
    It refuses on a FAST build for the same reason `verify` does, and it
    stamps the provenance the reader is going to be asked to trust.
    """
    mode = env["mode"]
    if mode.loaded != "identical":
        _emit(_env_lines(env) + [
            _THIN,
            "REFUSED. A reference card cannot be emitted from a FAST build.",
            "  The reference is the thing the claim rests on. Set",
            "  MOJOLEARN_NUMERIC_MODE=identical and run this again.",
            _RULE,
        ])
        return EXIT_REFUSED_FAST

    out = args.emit_reference
    try:
        stages, elapsed = _write_candidate(out, env, comparator)
    except FixtureError as exc:
        _emit(["CANNOT RUN. %s" % exc])
        return EXIT_CANNOT_RUN

    _emit(_env_lines(env, comparator=comparator) + [
        _THIN,
        "  wrote %s" % out,
        "  %d stages, fit %.2f s" % (stages, elapsed),
        "",
        "  THIS IS A CANDIDATE, NOT YET A REFERENCE. On a second vendor, at",
        "  the SAME commit, confirm it with ONE command:",
        "",
        "      MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify \\",
        "          --confirm-reference %s \\" % out,
        "          --emit-reference <local candidate path>",
        "",
        "  then install the agreed pair with",
        "",
        "      python -m mojolearn install-reference <produced> <confirmed>",
        "",
        "  A reference installed from one machine claims nothing across",
        "  vendors.",
        _RULE,
    ])
    return EXIT_VERIFIED


def _confirm_reference(args, env, comparator):
    """`verify --confirm-reference PEER.card [--emit-reference LOCAL.card]`.

    The second box's ONE command: run the fixture here, stamp a local
    candidate, and hand both cards to the one comparator. Exit 0 means the
    two vendors agree stage for stage and the pair is ready for
    `install-reference`; exit 1 means there is a finding to chase and
    NOTHING to install.
    """
    mode = env["mode"]
    if mode.loaded != "identical":
        _emit(_env_lines(env) + [
            _THIN,
            "REFUSED. A confirmation run must come from an IDENTICAL build,",
            "  for the same reason the producing run must.",
            _RULE,
        ])
        return EXIT_REFUSED_FAST

    peer = args.confirm_reference
    if not os.path.isfile(peer):
        _emit(["USAGE: --confirm-reference %s: not a file" % peer])
        return EXIT_USAGE
    kv, _comments, _n = parse_card_comments(peer)
    if PLACEHOLDER_TOKEN in open(peer, encoding="utf-8").read():
        _emit(["REFUSED: %s still carries the %r token. It is a placeholder,"
               % (peer, PLACEHOLDER_TOKEN),
               "not a produced card; there is nothing to confirm against."])
        return EXIT_USAGE
    if kv.get("profile") != PROFILE:
        _emit(["REFUSED: the peer card's profile line is %r; this build's "
               "profile is %r." % (kv.get("profile"), PROFILE),
               "Confirming across that mismatch would compare two different "
               "computations."])
        return EXIT_USAGE

    local = args.emit_reference or os.path.join(
        tempfile.mkdtemp(prefix="mojolearn-confirm-"), "local.candidate.card")
    try:
        stages, elapsed = _write_candidate(local, env, comparator)
    except FixtureError as exc:
        _emit(["CANNOT RUN. %s" % exc])
        return EXIT_CANNOT_RUN

    differ, _p, _h = load_differ()
    _emit(_env_lines(env, comparator=comparator) + [
        _THIN,
        "  local candidate %s (%d stages, fit %.2f s)" % (local, stages,
                                                          elapsed),
        "  peer card       %s" % peer,
        "",
    ])
    rc = differ.run([peer, local, "--labels", "PEER,LOCAL",
                     "--no-verify-dumps"])
    if rc == 0:
        _emit([
            _RULE,
            "CONFIRMED. The two vendors' cards agree stage for stage.",
            "  Install the pair (produced first, this confirmation second):",
            "",
            "      python -m mojolearn install-reference \\",
            "          %s \\" % peer,
            "          %s" % local,
            _RULE,
        ])
        return EXIT_VERIFIED
    _emit([
        _RULE,
        "NOT CONFIRMED (differ exit %d). There is NO reference to install;"
        % rc,
        "there is a finding to chase. The first diverging stage is named",
        "above. Local candidate kept at",
        "    %s" % local,
        _RULE,
    ])
    return EXIT_MISMATCH if rc == 1 else EXIT_CANNOT_RUN


# --------------------------------------------------------------------------
# INSTALLING THE REFERENCE (DEVIATION 927)
# --------------------------------------------------------------------------

_INSTALL_REQUIRED_KEYS = ("profile", "produced-at", "commit", "numeric-mode",
                          "host", "device", "mojolearn-version")


def cmd_install_reference(args):
    """`python -m mojolearn install-reference PRODUCED.card CONFIRMED.card`.

    Host-side only; no GPU, no extension import. Takes the two candidate
    cards -- the producing vendor's first, the confirming vendor's second --
    and, ONLY if every gate passes, installs the first into
    `reference_cards/` with the confirmation's provenance appended, removes
    the placeholder, and prints the filled docs/VERIFY.md template for a
    human to paste.

    DEVIATION 927, the refusals, each one a way a dishonest reference could
    otherwise come into being:

      - a card carrying the FILL-IN token (a placeholder wearing a real name)
      - a profile line that is absent or mismatched on either card
      - missing provenance keys (a hash with no provenance is a number)
      - the two cards' commits differing, or either being unknown -- the
        procedure REQUIRES the same commit on both boxes, and 'unknown'
        cannot be shown to satisfy that
      - a numeric-mode line that is not 'identical'
      - the two cards not comparing IDENTICAL under the one comparator

    WHY THE DOC IS PASTED BY A HAND AND NOT WRITTEN BY THIS COMMAND. The
    template block in docs/VERIFY.md is the human-audited statement of where
    the reference came from. A command that edits it would make the doc
    exactly as trustworthy as the command's inputs, unreviewed; a human
    pasting a printed block reads it on the way in. The card itself carries
    both runs' provenance either way, so the shipped artifact does not
    depend on the paste.

    Exit codes: 0 installed; 1 the two cards diverge; 2 refused (any gate
    above); 4 no comparator.
    """
    lines = []
    cards = {}
    for role, path in (("produced", args.produced),
                       ("confirmed", args.confirmed)):
        if not os.path.isfile(path):
            _emit(["USAGE: %s card %s: not a file" % (role, path)])
            return EXIT_USAGE
        text = open(path, encoding="utf-8").read()
        if PLACEHOLDER_TOKEN in text:
            _emit(["REFUSED: the %s card %s still carries the %r token."
                   % (role, path, PLACEHOLDER_TOKEN),
                   "A placeholder copied under a real name is exactly what "
                   "this gate exists for (DEVIATION 925)."])
            return EXIT_USAGE
        kv, comments, n_records = parse_card_comments(path)
        missing = [k for k in _INSTALL_REQUIRED_KEYS if not kv.get(k)]
        if missing:
            _emit(["REFUSED: the %s card %s carries no provenance for: %s."
                   % (role, path, ", ".join(missing)),
                   "A reference hash with no provenance is a number, not "
                   "evidence."])
            return EXIT_USAGE
        if kv["profile"] != PROFILE:
            _emit(["REFUSED: the %s card's profile line is %r; this package "
                   "verifies %r." % (role, kv["profile"], PROFILE),
                   "Installing across that mismatch would label one "
                   "computation with another's name."])
            return EXIT_USAGE
        if kv["numeric-mode"] != "identical":
            _emit(["REFUSED: the %s card states numeric-mode %r. Only an "
                   "identical-arm card can back the identity claim."
                   % (role, kv["numeric-mode"])])
            return EXIT_USAGE
        if n_records == 0:
            _emit(["REFUSED: the %s card holds no records." % role])
            return EXIT_USAGE
        cards[role] = {"path": path, "kv": kv, "comments": comments,
                       "records": n_records}

    pc = cards["produced"]["kv"]["commit"]
    cc = cards["confirmed"]["kv"]["commit"]
    if pc.startswith("unknown") or cc.startswith("unknown") or pc != cc:
        _emit(["REFUSED: the two cards must come from the SAME KNOWN commit.",
               "  produced   %s" % pc,
               "  confirmed  %s" % cc,
               "'Same commit' is the thing that makes their agreement a",
               "cross-vendor statement about one program; unknown cannot be",
               "shown to satisfy it. Re-emit from clean checkouts."])
        return EXIT_USAGE

    try:
        differ, differ_path, differ_how = load_differ()
    except RuntimeError as exc:
        _emit(["CANNOT RUN. No comparator.", str(exc)])
        return EXIT_CANNOT_RUN

    rc = differ.run([args.produced, args.confirmed,
                     "--labels", "PRODUCED,CONFIRMED", "--no-verify-dumps"])
    if rc != 0:
        _emit([
            _RULE,
            "NOT INSTALLED (differ exit %d). The two cards do not agree;" % rc,
            "there is no reference, there is a finding. The first diverging",
            "stage is named above.",
            _RULE,
        ])
        return EXIT_MISMATCH if rc == 1 else EXIT_CANNOT_RUN

    # Install: the produced card's bytes, with the confirmation's provenance
    # appended as comments (the differ skips comments, so `verify`'s
    # comparison is untouched). The shipped artifact then carries BOTH runs.
    name = args.reference_name
    dest = reference_path(name)
    os.makedirs(reference_dir(), exist_ok=True)
    produced_text = open(args.produced, encoding="utf-8").read()
    ckv = cards["confirmed"]["kv"]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    footer = [
        "#",
        "# confirmed-at: %s" % ckv["produced-at"],
        "# confirmed-on-device: %s" % ckv["device"],
        "# confirmed-on-host: %s" % ckv["host"],
        "# confirmed-commit: %s" % ckv["commit"],
        "# confirmed-with: tools/identity_trace_diff.py (%s, sha256 %s)"
        % (differ_how, _sha256(differ_path)),
        "# confirmed-result: IDENTICAL over %d stages"
        % cards["produced"]["records"],
        "# installed-by: python -m mojolearn install-reference, %s" % stamp,
    ]
    if not produced_text.endswith("\n"):
        produced_text += "\n"
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(produced_text + "\n".join(footer) + "\n")

    placeholder = dest + ".PLACEHOLDER"
    removed = False
    if os.path.isfile(placeholder):
        os.remove(placeholder)
        removed = True

    pkv = cards["produced"]["kv"]
    template = [
        "```",
        "reference   %s.card" % name,
        "profile     %s" % PROFILE,
        "produced    %s on %s, %s, mojolearn %s, commit %s"
        % (pkv["produced-at"], pkv["device"], pkv["host"],
           pkv["mojolearn-version"], pkv["commit"]),
        "confirmed   %s on %s, %s, same commit, cards compared with"
        % (ckv["produced-at"], ckv["device"], ckv["host"]),
        "            tools/identity_trace_diff.py, RESULT: IDENTICAL over "
        "%d stages" % cards["produced"]["records"],
        "stages      %d" % cards["produced"]["records"],
        "```",
    ]
    lines += [
        _RULE,
        "INSTALLED %s" % dest,
        "  sha256 %s" % _sha256(dest),
        ("  placeholder removed" if removed
         else "  (no placeholder was present)"),
        _THIN,
        "NOW DO THESE TWO THINGS, in this order.",
        "",
        "1. Paste this filled block into docs/VERIFY.md under 'Where the",
        "   reference card came from', replacing the placeholder note and",
        "   template there, and record the change in CHANGELOG.md:",
        "",
    ] + ["   " + t for t in template] + [
        "",
        "2. Close the loop on the machine that produced the card:",
        "",
        "       MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify",
        "",
        "   It must print RESULT: VERIFIED. A producer that cannot reproduce",
        "   its own card is a much more interesting problem than a version",
        "   bump.",
        _RULE,
    ]
    _emit(lines)
    return EXIT_VERIFIED
