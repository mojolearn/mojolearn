#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OUR side of the classical identity-cost grid, driven through the PUBLIC
PYTHON SURFACE, one lane per process.

    MOJOLEARN_NUMERIC_MODE=identical \\
      python bench/speed/classical_py_speed_arm.py --lane kmeans --size large
    MOJOLEARN_SPEED_LANE=svm MOJOLEARN_SPEED_SIZE=wide \\
      python bench/speed/classical_py_speed_arm.py

THE QUESTION THIS FILE ANSWERS
-------------------------------
What a user of `import mojolearn` pays, in the numeric mode the package
was imported under, for the same classical fit the NVIDIA-native opponent
in `tools/speed_cuml_arm.py` runs on the same bytes, at the LARGE and
WIDE shapes of the 2026-09-07 NVIDIA identity-cost grid.

It is the third `ours` driver for the classical family and it is NOT a
replacement for the other two. `bench/speed/classical_speed_main.mojo`
times the Mojo entry points directly and covers every lane, including the
eight that have no Python door; this file times the SAME kernels through
`python/mojolearn/`, which is what a pip user actually calls and which
carries costs the Mojo driver does not (an int32 label copy, a Fortran
transpose, a square root over the k-NN result). Those costs are LEFT IN
and NAMED, one DEVIATION each, because a benchmark that deletes a cost the
user cannot delete is measuring something nobody can buy (the rule
`bench/speed/forest_speed_arm.py` sets as DEVIATION 1840).

THE MODE OF OUR ARM IS NOT CHOSEN HERE (DEVIATION 2162). No `numeric_mode=`
is passed to any estimator and no `set_numeric_mode` is called. The tier is
whatever `MOJOLEARN_NUMERIC_MODE` selected at import, READ BACK from the
package with `mojolearn.numeric_mode()` -- which cross-checks the loaded
binary's own compile-time answer -- and printed on the header as
`mode=FAST|DETERMINISTIC|IDENTICAL`. Where the estimator carries
`numeric_mode_used()`, that per-instance read-back is compared against the
header's and a disagreement is a NOTE, never silently one or the other.
The identity-cost grid runs this file under `identical`; a header that says
anything else is a run of a different question, and a NOTE says so.

THE TWO SIDES RUN ON THE SAME BYTES (DEVIATION 2161). This file generates
NO data. It imports `fixture(lane, size)` from `tools/speed_cuml_arm.py`
and takes the arrays, the shape tag and the hyperparameters from it, so
both arms see identical bytes and print identical `shape=` tags by
construction rather than by two transcriptions agreeing. If that import
fails -- the vendor module loads `bench/bench_sklearn.py`, which imports
scikit-learn at module top (DEVIATION 2173) -- this arm prints ONE
`FSPEED-REFUSED` naming what was missing and exits zero. It never invents
a fixture.

WHAT IS TIMED. The same region the vendor arm times for the lane: the
constructor plus `fit` where the opponent times a fit, the query where
the opponent times a query (`knn`, `kde`), fit plus predict where the
opponent does both (`gp`). Everything else -- the fixture, the float32
C-order preparation, an index `fit` for a search lane -- happens before
the clock. One untimed warm-up, `rounds` timed rounds, a hash per round.

SYNCHRONIZATION (DEVIATION 2163). Every binding this file reaches returns
HOST arrays -- `kmeans_fit` writes `centers` and `labels`, `knn_search`
writes `dist` and `ind`, `svc_fit` writes `dual` and `support` -- so the
call cannot return before the device finished producing them. The
synchronization proof is the return itself, and `_sync_ours` is a named
no-op that says so, exactly as `forest_speed_arm._our_sync` does. It is
called inside the timer anyway so the region is spelled the same way on
every arm.

THE HASH (DEVIATION 2164). sha256 over dtype, shape and bytes, truncated
to 16 hex digits, per output array, folded across the lane's outputs --
`tools/speed_gbdt_arm.py::hash_predictions`, the recipe the forest arm and
gbm-bench use. It is NOT the vendor arm's FNV-1a64 and it is NOT the Mojo
driver's; no line here gates a hash and none is comparable across arms.
It is a WITHIN-ARM probe: under `identical` or `deterministic` two rounds
that print different hashes are the finding this column exists to show,
and under `fast` they are expected. The quantity hashed is the vendor
arm's quantity for the lane (k-means centers and labels, PCA components
and spectra, OLS coefficients, k-NN distances and indices, ...). Two lanes
where the vendor prints `hash=-` for a CROSS-arm reason (`spectral`,
label numbering; `metrics`, eleven floats of two widths) are hashed here
anyway, because the within-arm question is the one this grid asks
(DEVIATION 2170); the NOTE on those rows says so.

ACCURACY. `tools/speed_cuml_arm.py` emits no `FSPEED-ACC` line for any
classical lane, by its own design ("a speed harness that also scores
accuracy invites a reader to trade one against the other"), so this arm
emits none either -- with three exceptions since 2026-09-08 (DEVIATION
2239): the `logistic` lane scores log loss on the held-out last 10% of
rows, the `arima` lane scores in-sample one-step RMSE, and the
`holtwinters` lane (DEVIATION 2270) scores the RMSE of `forecast(h)`,
`h = 2 * seasonal_periods`, from an UNTIMED fit on the first `n - h`
observations of every series against the held-out last `h` (the timed
fit stays on the full series), EACH THROUGH THE VENDOR MODULE'S OWN
HELPER (`binary_log_loss`, `arima_insample_rmse`,
`holtwinters_holdout_rmse`), imported by name, so the two arms score with
one piece of arithmetic. A lane that scores returns a third element,
`after()`, which `main` calls once after the race and prints as
`FSPEED-ACC lane= arm=ours metric= value=`; a helper that is missing is a
refusal before the race, because a row with a speed and no score is not
the row this grid asks for. holtwinters' `after()` also NOTEs the batch
means of the fitted `alpha_` / `beta_` / `gamma_` of that holdout fit, as
the vendor arm does for cuML (which exposes none, and says so), and a
fixture too short for the holdout is a NOTE, not a refusal.

THE gp LANE AT scale (DEVIATION 2271). The vendor module's `scale` tier
for `gp` carries alpha = 2**-20, the one positive jitter this arm's
`identical` profile accepts (DEVIATION 2169), so the tier is the
identical-mode profile and this arm no longer refuses it; `wide` keeps
alpha 0.1 and is refused under `identical` exactly as before. If the
20,000 x 20,000 float32 RBF Gram does not factor at that jitter, the
lane's `call` raises `Refuse` with the factorization error's text and
`main` prints it as one `FSPEED-REFUSED` on this arm, as it already does
for any exception inside the race.

THE FIFTH SIZE, scale (DEVIATION 2236). `tools/speed_cuml_arm.py`'s
`scale` tier (DEVIATION 2230) is a narrow, generated fixture above the
paper's floor for the lanes whose `large` was a tiny dump, and IS `large`
for every other lane. This file only forwards the size; the tag and the
bytes are the vendor module's, as at every other size.

NO PYTHON SURFACE (DEVIATION 2171). Eight of the twenty-four vendor lanes
have no public Python door in `python/mojolearn/` -- `ivf`, `hdbscan`,
`cholesky`, `gmm`, `krr`, `nystroem`, `rbfsampler`, `resample`. Each
prints exactly one `FSPEED-REFUSED ... reason=NO-PYTHON-SURFACE: <what was
looked for>` and exits zero; the Mojo driver covers them. That refusal is
printed BEFORE `mojolearn` is imported, so it is the same line on a box
where the package will not import.

THE OUTPUT CONTRACT is `tools/fast_speed_table.py`'s, unchanged:

    FSPEED-HEADER family=classical lane=<l> arm=ours mode=<M> \\
        device=<name> rounds=<n> size=<shipped|smoke|large|wide>
    FSPEED-WARMUP lane=<l> arm=ours shape=<tag> ms=<float>
    FSPEED lane=<l> arm=ours shape=<tag> round=<i> ms=<float> hash=<16 hex|->
    FSPEED-NOTE lane=<l> arm=ours <free text>
    FSPEED-REFUSED lane=<l> arm=ours reason=<one line>
    FSPEED-ACC lane=<l> arm=ours metric=<name> value=<float>   (logistic,
        arima, DEVIATION 2239; holtwinters, DEVIATION 2270)

`arm=ours` is the label the brief fixes and the Mojo driver already uses,
so a run directory that holds BOTH drivers' logs for one lane needs the
leg body to name the files apart; the first NOTE of every run carries
`surface=python` so the two are never confused once opened (DEVIATION
2160).

Environment (each a fallback for the flag of the same name):
    MOJOLEARN_SPEED_LANE      the lane
    MOJOLEARN_SPEED_ROUNDS    timed rounds (default 5, the vendor arm's)
    MOJOLEARN_SPEED_SIZE      shipped | smoke | large | wide | scale
    MOJOLEARN_NUMERIC_MODE    read by mojolearn at import, never by this file

DEVIATION numbers 2160-2189 are this file's, plus 2236-2239 from the
2026-09-08 block (2230-2249, shared with the vendor module) and 2270-2271
from the later block of the same day (2270-2274, also shared); the ones
spent are listed at `DEVIATIONS` below so the next reader can see the gaps.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

# `tools/` is not a package and never has been, so both helper modules are
# imported by name with that directory on the path, exactly as
# `bench/speed/forest_speed_arm.py` does. `python/` goes on too so an
# in-repo, not-yet-installed `mojolearn` is importable the way
# `bench/external/run_gbm_bench.sh` arranges it. The repository root is two
# levels up (bench/speed/ -> bench/ -> root).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAMILY = "classical"
ARM = "ours"

SIZES = ("shipped", "smoke", "large", "wide", "scale")

#: The twenty-four lanes the vendor arm knows, in its order (logistic and
#: arima since 2026-09-08, DEVIATIONS 2237-2238).
VENDOR_LANES = (
    "kmeans", "dbscan", "pca", "ols", "knn", "cd", "kde", "linkage", "svm",
    "metrics", "ivf", "hdbscan", "cholesky", "gmm", "gp", "krr", "nystroem",
    "rbfsampler", "resample", "spectral", "holtwinters", "kpss",
    "logistic", "arima",
)

#: DEVIATION 2171. Lanes with no public Python door, and WHAT WAS LOOKED
#: FOR, so the refusal is a finding about the surface and not a shrug.
#: Verified against `python/mojolearn/__init__.py`'s `__all__` and every
#: `_*_impl.py` on 2026-09-07; the Mojo lane directories exist for all
#: eight (`ivf/`, `hdbscan/`, `cholesky/`, `mixture/`, `kernel_methods/`,
#: `resample/`), so these are unexported lanes, not missing ones.
NO_PYTHON_SURFACE = {
    "ivf": "NO-PYTHON-SURFACE: looked for an IVF-Flat index in "
           "mojolearn.neighbors (NearestNeighbors accepts algorithm= "
           "'brute'/'auto'/'rbc' only, all EXACT; no ivf_flat build/search "
           "is exported from bindings/_mojolearn.mojo); the Mojo driver "
           "covers ivf/",
    "hdbscan": "NO-PYTHON-SURFACE: looked for mojolearn.HDBSCAN and a "
               "cluster/density module class; none in __init__.__all__, "
               "no _hdbscan_impl.py; the Mojo driver covers hdbscan/",
    "cholesky": "NO-PYTHON-SURFACE: looked for mojolearn.linalg.cholesky / "
                "cho_solve; mojolearn.linalg exports matmul only "
                "(_linalg_impl.py), and the Cholesky the GP wraps is not "
                "reachable as a factorization; the Mojo driver covers "
                "cholesky/",
    "gmm": "NO-PYTHON-SURFACE: looked for mojolearn.GaussianMixture; "
           "mixture/ has no _impl.py and no binding export; the Mojo "
           "driver covers mixture/",
    "krr": "NO-PYTHON-SURFACE: looked for mojolearn.KernelRidge; "
           "kernel_methods/ has no Python class; the Mojo driver covers it",
    "nystroem": "NO-PYTHON-SURFACE: looked for mojolearn.Nystroem; "
                "kernel_methods/ has no Python class; the Mojo driver "
                "covers it",
    "rbfsampler": "NO-PYTHON-SURFACE: looked for mojolearn.RBFSampler; "
                  "kernel_methods/ has no Python class; the Mojo driver "
                  "covers it",
    "resample": "NO-PYTHON-SURFACE: looked for a bootstrap in "
                "mojolearn (resample/ has no Python door); the Mojo driver "
                "covers resample/",
}

#: The DEVIATION numbers this file spends, so a reader can see the gaps
#: (2175-2189 are unspent as of 2026-09-07).
DEVIATIONS = {
    2160: "arm=ours is shared with the Mojo driver; surface=python NOTE",
    2161: "fixture bytes and shape tag come from the vendor module, never "
          "regenerated here",
    2162: "mode is read back from mojolearn.numeric_mode(), never passed",
    2163: "sync is the host return; _sync_ours is a named no-op",
    2164: "hash is sha256/16 per hash_predictions, within-arm only",
    2165: "kmeans: labels.astype(int32) happens inside fit, inside the timer",
    2166: "knn: the Python surface returns sqrt distances and an int64 "
          "index cast, both inside the timer; the Mojo driver returns "
          "squared distances",
    2167: "cd: Lasso.fit copies X to Fortran order inside the timer",
    2168: "kpss: kpss_test wants (n_obs, n_series) and transposes to "
          "series-major with a copy inside the timer",
    2169: "gp: float32 against the vendor's float64; under identical the "
          "ridge alpha must be +0.0 or 2**-20 (DEVIATION 1637) and any "
          "other fixture alpha is refused with the bound quoted",
    2170: "spectral and metrics are hashed here where the vendor prints "
          "hash=- for a cross-arm reason",
    2171: "eight lanes refused NO-PYTHON-SURFACE before mojolearn imports",
    2172: "a hyperparameter the fixture params lack falls back to the "
          "vendor arm's transcribed constant, with a NOTE naming it",
    2173: "importing the vendor module pulls in scikit-learn via "
          "bench/bench_sklearn.py; the refusal names it",
    2174: "dbscan runs algorithm='rbc' (the class default and the Mojo "
          "driver's EPS_NN_RBC) at the class's max_iterations=None fixed "
          "point where the Mojo entry defaults to 200",
    2236: "size=scale is accepted and forwarded to the vendor module's "
          "fixture(); nothing else about the tier is decided here",
    2237: "logistic: mojolearn.LogisticRegression(qn) on rows [0, n_fit), "
          "hash(coef_, intercept_); predict_proba on the held-out rows",
    2238: "arima: mojolearn.ARIMA(order=(p,d,q), trend='c'|'n' from the "
          "fixture's fit_intercept) on series-major y, hash(ar_, ma_, "
          "sigma2_); predict(0, n_obs) for the score",
    2239: "FSPEED-ACC is emitted by an `after()` hook through the vendor "
          "module's helper (binary_log_loss, arima_insample_rmse), never "
          "a local scorer",
    2270: "holtwinters: FSPEED-ACC holdout_rmse_h<2*seasonal_periods> from "
          "an untimed fit on the first n-h observations, scored by the "
          "vendor module's holtwinters_holdout_rmse; the timed fit is "
          "untouched; alpha_/beta_/gamma_ batch means NOTEd; a fixture "
          "too short for the holdout is a NOTE, not a refusal",
    2271: "gp: the vendor module's scale tier carries alpha=2**-20, the "
          "identical-mode jitter, so this arm runs it; a Gram that does "
          "not factor is a Refuse carrying the factorization error",
}


# --------------------------------------------------------------------------
# The output contract.
# --------------------------------------------------------------------------

def emit_header(lane, mode, device, n_rounds, size):
    print("FSPEED-HEADER family=%s lane=%s arm=%s mode=%s device=%s "
          "rounds=%d size=%s" % (FAMILY, lane, ARM, mode, device, n_rounds,
                                 size), flush=True)


def emit_warmup(lane, shape, ms):
    print("FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.6f"
          % (lane, ARM, shape, ms), flush=True)


def emit_round(lane, shape, idx, ms, digest):
    print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.6f hash=%s"
          % (lane, ARM, shape, idx, ms, digest or "-"), flush=True)


def emit_note(lane, text):
    print("FSPEED-NOTE lane=%s arm=%s %s"
          % (lane, ARM, " ".join(str(text).split())), flush=True)


def emit_refused(lane, reason):
    print("FSPEED-REFUSED lane=%s arm=%s reason=%s"
          % (lane, ARM, " ".join(str(reason).split())), flush=True)


def emit_acc(lane, metric, value):
    """DEVIATION 2239. The vendor arm's `acc` line, same format."""
    print("FSPEED-ACC lane=%s arm=%s metric=%s value=%.6f"
          % (lane, ARM, metric, float(value)), flush=True)


class Refuse(Exception):
    """A refusal with a one-line reason; caught at the top and printed as
    exactly one FSPEED-REFUSED. Raised for a shape outside a documented
    bound (quoting the bound), a fixture key this file cannot find, and a
    hyperparameter that the estimator's contract cannot honor."""


# --------------------------------------------------------------------------
# Hash, sync, device, mode.
# --------------------------------------------------------------------------

def _hash_predictions():
    """`tools/speed_gbdt_arm.py::hash_predictions`, imported by name so the
    recipe cannot drift from the forest arm's. The fallback spells the same
    recipe (dtype, shape, bytes; sha256; 16 hex) for a checkout where
    `tools/` is unreadable, and says so once."""
    try:
        from speed_gbdt_arm import hash_predictions       # noqa: PLC0415
        return hash_predictions, True
    except Exception:                                     # noqa: BLE001
        def hash_predictions(vec):
            if vec is None:
                return None
            arr = np.ascontiguousarray(vec)
            h = hashlib.sha256()
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
            return h.hexdigest()[:16]
        return hash_predictions, False


HASH_ONE, HASH_IMPORTED = _hash_predictions()


def hash_outputs(*arrays):
    """One 16-hex digest over the lane's outputs, in order (DEVIATION 2164).

    Each array is hashed by `hash_predictions` (dtype + shape + bytes), and
    the per-array digests are folded by one more sha256 so that a lane with
    two outputs prints one column. `None` anywhere is `-`: an output that
    could not be brought back is not hashed as an empty one."""
    parts = []
    for a in arrays:
        if a is None:
            return "-"
        parts.append(HASH_ONE(np.asarray(a)))
    if len(parts) == 1:
        return parts[0]
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()[:16]


def _sync_ours():
    """A named no-op (DEVIATION 2163).

    Every binding this file reaches returns HOST arrays, so the call cannot
    return before the device finished producing them; the return is the
    synchronization. Spelled out and called inside the timer so that the
    timed region reads the same way on every arm in the family."""
    return None


def device_string():
    """The vendor arm's spelling first (`nvidia-smi` name, spaces to
    underscores, via `speed_gbdt_arm.device_string`), so the two headers of
    one lane carry the same token; the platform as the fallback."""
    try:
        from speed_gbdt_arm import device_string as ds    # noqa: PLC0415
        return ds()
    except Exception:                                     # noqa: BLE001
        import platform                                   # noqa: PLC0415
        return (platform.system() + "_" + platform.machine()).replace(" ", "_")


def mode_label(mojolearn):
    """`mojolearn.numeric_mode()` upper-cased (DEVIATION 2162). That
    function reads the CURRENT default tier and cross-checks it against the
    loaded gbdt binary's compile-time answer; a mismatch RAISES there, and
    this file lets it, because a mislabelled arm is worse than a missing
    one. The environment variable is compared beside it only to NOTE a
    disagreement, never to override."""
    loaded = str(mojolearn.numeric_mode()).strip().lower()
    label = {"fast": "FAST", "deterministic": "DETERMINISTIC",
             "identical": "IDENTICAL"}.get(loaded)
    if label is None:
        raise Refuse("mojolearn.numeric_mode() returned %r, which is not one "
                     "of fast/deterministic/identical" % (loaded,))
    return label, loaded


# --------------------------------------------------------------------------
# The fixture, from the vendor module (DEVIATION 2161).
# --------------------------------------------------------------------------

def load_fixture(lane, size):
    """`tools/speed_cuml_arm.fixture(lane, size)` -> (arrays, shape_tag,
    params). Imported by name with `tools/` on the path; an ImportError is
    a refusal naming the module that failed, because the vendor module
    loads `bench/bench_sklearn.py` at import and THAT imports scikit-learn
    at module top (DEVIATION 2173), so on a Python without scikit-learn
    this arm cannot obtain the fixture at all."""
    try:
        import speed_cuml_arm as vendor                    # noqa: PLC0415
    except Exception as exc:                               # noqa: BLE001
        raise Refuse(
            "cannot import tools/speed_cuml_arm.py, which owns the fixture "
            "contract (it loads bench/bench_sklearn.py, which imports "
            "scikit-learn at module top): %s: %s"
            % (exc.__class__.__name__, exc))
    fixture = getattr(vendor, "fixture", None)
    if fixture is None:
        raise Refuse(
            "tools/speed_cuml_arm.py exports no fixture(lane, size); this "
            "arm never regenerates data, so it cannot run until the vendor "
            "module carries the fixture contract")
    try:
        out = fixture(lane, size)
    except Exception as exc:                               # noqa: BLE001
        raise Refuse("fixture(%r, %r) raised %s: %s"
                     % (lane, size, exc.__class__.__name__, exc))
    try:
        arrays, shape_tag, params = out
    except Exception:                                      # noqa: BLE001
        raise Refuse("fixture(%r, %r) returned %r, not an (arrays, "
                     "shape_tag, params) triple" % (lane, size, type(out)))
    if not isinstance(arrays, dict) or not isinstance(params, dict):
        raise Refuse("fixture(%r, %r) returned arrays=%s params=%s; both "
                     "must be dicts" % (lane, size, type(arrays).__name__,
                                        type(params).__name__))
    return arrays, str(shape_tag), params


def vendor_helper(name):
    """A scorer from `tools/speed_cuml_arm.py`, by name (DEVIATION 2239).
    The vendor arm scores its own row with the same function, so the
    accuracy column compares two fits and never two scorers. Missing is a
    refusal that names it."""
    try:
        import speed_cuml_arm as vendor                    # noqa: PLC0415
    except Exception as exc:                               # noqa: BLE001
        raise Refuse("cannot import tools/speed_cuml_arm.py for its %s: %s: "
                     "%s" % (name, exc.__class__.__name__, exc))
    f = getattr(vendor, name, None)
    if not callable(f):
        raise Refuse("tools/speed_cuml_arm.py exports no %s; the FSPEED-ACC "
                     "line is computed by ONE helper on both arms or not at "
                     "all, and this lane does not run without it" % name)
    return f


def _need(arrays, lane, *names):
    """The first of `names` present in the fixture's arrays, or a refusal
    that lists what IS there, so a contract mismatch is one readable line."""
    for n in names:
        if n in arrays:
            return arrays[n]
    raise Refuse("fixture for %s carries no array named %s (it has: %s)"
                 % (lane, "/".join(names), ",".join(sorted(arrays)) or "-"))


def _param(params, lane, names, fallback=None, cast=None):
    """A hyperparameter from the fixture's params (DEVIATION 2172).

    The vendor arm's value wins whenever it is present. When it is absent
    and a `fallback` is given, the fallback is the CONSTANT the vendor arm
    hard-codes for the lane, transcribed here, and a NOTE names it so a
    reader can see which knob was inferred. No fallback means the knob is
    load-bearing and its absence is a refusal."""
    for n in names:
        if n in params:
            v = params[n]
            return cast(v) if cast is not None else v
    if fallback is None:
        raise Refuse("fixture params for %s carry no %s (they have: %s)"
                     % (lane, "/".join(names), ",".join(sorted(params)) or "-"))
    emit_note(lane, "params lack %s; using the vendor arm's transcribed "
                    "constant %r (DEVIATION 2172)" % ("/".join(names), fallback))
    return fallback


def _f32_matrix(a, name, lane, cols=None):
    """A C-contiguous float32 2-D array. A flat dump is reshaped by `cols`
    when the caller knows it; anything else that is not 2-D is refused."""
    x = np.asarray(a)
    if x.ndim == 1 and cols is not None:
        x = x.reshape(-1, int(cols))
    if x.ndim != 2:
        raise Refuse("%s: fixture array %s is %d-D, need 2-D"
                     % (lane, name, x.ndim))
    return np.ascontiguousarray(x, dtype=np.float32)


def _f32_vector(a, name, lane):
    x = np.asarray(a)
    if x.ndim != 1:
        x = x.reshape(-1)
    return np.ascontiguousarray(x, dtype=np.float32)


def _i32_vector(a):
    return np.ascontiguousarray(np.asarray(a).reshape(-1), dtype=np.int32)


# --------------------------------------------------------------------------
# The round loop.
# --------------------------------------------------------------------------

def race(lane, shape, n_rounds, size, mode, device, call, notes=()):
    """One untimed warm-up plus `n_rounds` timed calls of `call`, which
    returns the outputs to hash (a tuple) or `None`. The header goes out
    first, then the lane's notes, then the rounds, exactly as the vendor
    arm orders them, so a log of either arm reads the same way."""
    emit_header(lane, mode, device, n_rounds, size)
    for n in notes:
        emit_note(lane, n)
    hashes = []
    for r in range(n_rounds + 1):
        t0 = time.perf_counter()
        out = call()
        _sync_ours()
        ms = (time.perf_counter() - t0) * 1000.0
        digest = "-" if out is None else hash_outputs(*out)
        if r == 0:
            emit_warmup(lane, shape, ms)
        else:
            emit_round(lane, shape, r, ms, digest)
            hashes.append(digest)
    for h in hashes[1:]:
        if h != hashes[0] and "-" not in (h, hashes[0]):
            emit_note(lane, "hash moved across rounds: %s %s" % (hashes[0], h))
            break


def _tier_check(lane, est, loaded):
    """Compare the estimator's own tier read-back with the header's. A
    class without the mixin (AgglomerativeClustering, SpectralClustering,
    ExponentialSmoothing) has nothing to read and is skipped by name."""
    used = getattr(est, "numeric_mode_used", None)
    if used is None:
        return
    try:
        tier = str(used()).lower()
    except Exception as exc:                               # noqa: BLE001
        emit_note(lane, "numeric_mode_used() raised %s: %s"
                  % (exc.__class__.__name__, exc))
        return
    if tier != loaded:
        emit_note(lane, "TIER MISMATCH: %s.numeric_mode_used()=%s but "
                        "mojolearn.numeric_mode()=%s; the header carries "
                        "the package's answer" % (type(est).__name__, tier,
                                                  loaded))


# --------------------------------------------------------------------------
# GROUP 1: the five splitmix64 lanes.
# --------------------------------------------------------------------------

def lane_kmeans(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    X = _f32_matrix(_need(arrays, lane, "X", "x"), "X", lane)
    n, d = X.shape
    k = _param(params, lane, ("n_clusters", "k", "km_k"), fallback=64, cast=int)
    init = _f32_matrix(_need(arrays, lane, "init", "init_centroids",
                             "centroids"), "init", lane, cols=d)
    iters = _param(params, lane, ("max_iter", "km_iter", "iters"),
                   fallback=20, cast=int)
    tol = _param(params, lane, ("tol",), fallback=1e-7, cast=float)
    if k > n:
        raise Refuse("KMeans bound: n_clusters=%d exceeds n_samples=%d "
                     "(cluster.py KMeans.fit refuses n_clusters > n)" % (k, n))
    if init.shape != (k, d):
        raise Refuse("KMeans bound: init_centroids must be (%d, %d), fixture "
                     "gives %s" % (k, d, init.shape))
    probe = ml.KMeans(n_clusters=k, init="array", init_centroids=init,
                      n_init=1, max_iter=iters, tol=tol)
    _tier_check(lane, probe, ctx.loaded)

    def call():
        # Constructed inside the clock, matching the vendor arm, whose
        # cuml.KMeans does no device work in __init__ either.
        km = ml.KMeans(n_clusters=k, init="array", init_centroids=init,
                       n_init=1, max_iter=iters, tol=tol)
        km.fit(X)
        return (km.cluster_centers_, km.labels_)

    return call, [
        "init='array' with the fixture's centroids, n_init=1, max_iter=%d, "
        "tol=%g, metric L2Expanded: the same knobs the vendor arm passes "
        "to cuml.KMeans" % (iters, tol),
        "KMeans.fit converts the uint32 device labels to int32 on the host "
        "INSIDE fit, so that %d-element copy is inside the timer; it is a "
        "cost of this surface and it stays (DEVIATION 2165)" % n,
    ]


def lane_dbscan(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    X = _f32_matrix(_need(arrays, lane, "X", "x"), "X", lane)
    eps = _param(params, lane, ("eps",), fallback=0.35, cast=float)
    min_samples = _param(params, lane, ("min_samples", "min_pts"),
                         fallback=5, cast=int)
    if eps <= 0:
        raise Refuse("DBSCAN bound: eps must be positive, fixture gives %r"
                     % eps)
    if min_samples < 1:
        raise Refuse("DBSCAN bound: min_samples must be at least 1, fixture "
                     "gives %r" % min_samples)
    _tier_check(lane, ml.DBSCAN(eps=eps, min_samples=min_samples), ctx.loaded)

    def call():
        db = ml.DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean",
                       algorithm="rbc")
        db.fit(X)
        return (db.labels_,)

    return call, [
        "eps=%g min_samples=%d metric=euclidean algorithm='rbc': the class "
        "default and the arm classical_speed_main.mojo times "
        "(dbscan_fit_impl's EPS_NN_RBC default); max_iterations is the "
        "class's None (the fixed point) where the Mojo entry defaults to "
        "200 (DEVIATION 2174)" % (eps, min_samples),
        "labels only, matching the vendor's calc_core_sample_indices=False",
    ]


def lane_pca(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    X = _f32_matrix(_need(arrays, lane, "X", "x"), "X", lane)
    n, d = X.shape
    nc = _param(params, lane, ("n_components", "pca_comp", "comp"),
                fallback=8, cast=int)
    if n < 2 or d < 2:
        raise Refuse("PCA bound: at least 2 rows and 2 features "
                     "(decomposition.py PCA.fit), fixture is %dx%d" % (n, d))
    if nc < 1 or nc > d:
        raise Refuse("PCA bound: n_components must be in [1, %d] "
                     "(decomposition.py _component_count), fixture asks %d"
                     % (d, nc))
    _tier_check(lane, ml.PCA(n_components=nc, svd_solver="jacobi"), ctx.loaded)

    def call():
        p = ml.PCA(n_components=nc, svd_solver="jacobi")
        p.fit(X)
        return (p.components_, p.explained_variance_, p.singular_values_)

    return call, [
        "svd_solver='jacobi': the covariance plus device Jacobi arm, the "
        "route the vendor arm asks cuml.PCA for first (its NOTE names which "
        "solver actually ran on its side)",
    ]


def lane_ols(ml, ctx):
    lane, arrays = ctx.lane, ctx.arrays
    A = _f32_matrix(_need(arrays, lane, "A", "X", "x", "a"), "A", lane)
    b = _f32_vector(_need(arrays, lane, "b", "y"), "b", lane)
    if b.shape[0] != A.shape[0]:
        raise Refuse("LinearRegression bound: X and y lengths differ "
                     "(%d vs %d)" % (A.shape[0], b.shape[0]))
    _tier_check(lane, ml.LinearRegression(fit_intercept=False), ctx.loaded)

    def call():
        m = ml.LinearRegression(fit_intercept=False)
        m.fit(A, b)
        return (m.coef_,)

    return call, [
        "fit_intercept=False, the normal-equations eigen route (cuML's "
        "algorithm='eig', lstsqEig), matching the vendor arm's "
        "algorithm=eig fit_intercept=False",
    ]


def lane_knn(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    idx = _f32_matrix(_need(arrays, lane, "index", "idx", "X", "x"),
                      "index", lane)
    qry = _f32_matrix(_need(arrays, lane, "queries", "qry", "query", "Q"),
                      "queries", lane, cols=idx.shape[1])
    k = _param(params, lane, ("k", "n_neighbors", "knn_k"), fallback=10,
               cast=int)
    n_index = idx.shape[0]
    if k < 1 or k > n_index:
        raise Refuse("NearestNeighbors bound: n_neighbors must be in [1, %d] "
                     "(neighbors.py kneighbors), fixture asks %d"
                     % (n_index, k))
    if ctx.loaded == "identical" and k > 256:
        raise Refuse("NearestNeighbors bound under NUMERIC_IDENTICAL: "
                     "n_neighbors above 256 is refused (DEVIATION 500, the "
                     "identical selector's rank pass gives one thread per "
                     "output slot); fixture asks %d" % k)
    nn = ml.NearestNeighbors(n_neighbors=k, metric="euclidean",
                             algorithm="brute")
    _tier_check(lane, nn, ctx.loaded)
    # THE INDEX FIT IS OUTSIDE THE CLOCK, matching the vendor arm, whose
    # cuml.NearestNeighbors.fit runs before its race too. There is nothing
    # to build on either side: both are brute force.
    nn.fit(idx)
    _sync_ours()

    def call():
        d, i = nn.kneighbors(qry)
        return (d, i)

    return call, [
        "algorithm='brute' metric='euclidean' k=%d; the search is the timed "
        "region and the index fit is outside it on both arms" % k,
        "THIS SURFACE RETURNS EUCLIDEAN (sqrt) DISTANCES and casts the "
        "uint32 index to int64, both inside kneighbors and so inside the "
        "timer; the vendor arm's note that 'our arm returns SQUARED ones' "
        "describes the Mojo driver, not this one (DEVIATION 2166)",
    ]


# --------------------------------------------------------------------------
# GROUP 2: the dumped-fixture lanes.
# --------------------------------------------------------------------------

def lane_cd(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    d = params.get("d")
    X = _f32_matrix(_need(arrays, lane, "x", "X"), "x", lane, cols=d)
    y = _f32_vector(_need(arrays, lane, "y"), "y", lane)
    alpha = _param(params, lane, ("alpha",), cast=float)
    max_iter = _param(params, lane, ("max_iter",), cast=int)
    tol = _param(params, lane, ("tol",), cast=float)
    if alpha < 0.0:
        raise Refuse("Lasso bound: alpha >= 0 (cuML's guard, "
                     "_solver_impl.py), fixture gives %r" % alpha)
    if y.shape[0] != X.shape[0]:
        raise Refuse("Lasso bound: X and y lengths differ (%d vs %d)"
                     % (X.shape[0], y.shape[0]))
    probe = ml.Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter,
                     tol=tol, selection="cyclic")
    _tier_check(lane, probe, ctx.loaded)

    def call():
        m = ml.Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter,
                     tol=tol, selection="cyclic")
        m.fit(X, y)
        return (m.coef_,)

    return call, [
        "alpha=%g max_iter=%d tol=%g fit_intercept=True selection=cyclic, "
        "all explicit, the vendor arm's values" % (alpha, max_iter, tol),
        "Lasso.fit copies X to Fortran order INSIDE fit (cdFit reads "
        "column-major), so that copy is inside the timer; it is a cost of "
        "this surface and it stays (DEVIATION 2167)",
    ]


def lane_kde(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    d = params.get("d")
    train = _f32_matrix(_need(arrays, lane, "train", "X", "x"), "train",
                        lane, cols=d)
    query = _f32_matrix(_need(arrays, lane, "query", "queries", "Q"), "query",
                        lane, cols=train.shape[1])
    weights = arrays.get("weights", arrays.get("sample_weight"))
    if weights is not None:
        weights = _f32_vector(weights, "weights", lane)
    bw = _param(params, lane, ("bandwidth", "bw"), cast=float)
    if not (bw > 0.0):
        raise Refuse("KernelDensity bound: bandwidth must be positive "
                     "(density.py), fixture gives %r" % bw)
    kd = ml.KernelDensity(bandwidth=bw, kernel="gaussian", metric="euclidean")
    _tier_check(lane, kd, ctx.loaded)
    # `fit` stores the training set; the SCORE is the timed call on both
    # arms, and the vendor's fit runs before its race too.
    kd.fit(train, sample_weight=weights)
    _sync_ours()

    def call():
        return (kd.score_samples(query),)

    return call, [
        "kernel=gaussian metric=euclidean bandwidth=%g, %s; fit is outside "
        "the clock and score_samples is the timed region on both arms"
        % (bw, "weighted" if weights is not None else "unweighted"),
    ]


def lane_linkage(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    d = params.get("d")
    X = _f32_matrix(_need(arrays, lane, "x", "X"), "x", lane, cols=d)
    n = X.shape[0]
    k = _param(params, lane, ("n_clusters", "k"), cast=int)
    limit = getattr(ml, "_hierarchy_impl", None)
    limit = getattr(limit, "PAIRWISE_MAX_ROWS", 46340)
    if n < 2:
        raise Refuse("AgglomerativeClustering bound: n_rows >= 2, fixture "
                     "has %d" % n)
    if n > limit:
        raise Refuse("AgglomerativeClustering bound: n_rows <= "
                     "PAIRWISE_MAX_ROWS=%d (the dense m x m connectivity "
                     "matrix overflows cuVS's int past that), fixture has %d"
                     % (limit, n))
    if k < 1 or k > n:
        raise Refuse("AgglomerativeClustering bound: 1 <= n_clusters <= "
                     "n_rows=%d, fixture asks %d" % (n, k))
    _tier_check(lane, ml.AgglomerativeClustering(n_clusters=k), ctx.loaded)

    def call():
        m = ml.AgglomerativeClustering(n_clusters=k, linkage="single",
                                       connectivity="pairwise",
                                       metric="euclidean")
        m.fit(X)
        return (m.labels_,)

    return call, [
        "linkage=single connectivity=pairwise metric=euclidean, the vendor "
        "arm's configuration; cuML implements single linkage only",
    ]


def lane_svm(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    d = params.get("d")
    X = _f32_matrix(_need(arrays, lane, "x", "X"), "x", lane, cols=d)
    y = _f32_vector(_need(arrays, lane, "y"), "y", lane)
    C = _param(params, lane, ("C",), cast=float)
    gamma = _param(params, lane, ("gamma",), cast=float)
    tol = _param(params, lane, ("tol",), cast=float)
    steps = _param(params, lane, ("nochange_steps",), cast=int)
    classes = np.unique(y)
    if classes.shape[0] != 2:
        raise Refuse("SVC bound: binary classification only (_svm_impl.py "
                     "_as_labels), fixture has %d classes" % classes.shape[0])
    if not np.isfinite(C) or C <= 0.0:
        raise Refuse("SVC bound: C must be finite and positive (DEVIATION "
                     "636), fixture gives %r" % C)
    if not np.isfinite(tol) or tol <= 0.0:
        raise Refuse("SVC bound: tol must be finite and positive (DEVIATION "
                     "636), fixture gives %r" % tol)
    if not (gamma >= 0.0) or not np.isfinite(gamma):
        raise Refuse("SVC bound: gamma must be a finite float >= 0, fixture "
                     "gives %r" % gamma)
    probe = ml.SVC(kernel="rbf", C=C, gamma=gamma, tol=tol, nochange_steps=steps)
    _tier_check(lane, probe, ctx.loaded)

    def call():
        m = ml.SVC(kernel="rbf", C=C, gamma=gamma, tol=tol,
                   nochange_steps=steps)
        m.fit(X, y)
        return (m.dual_coef_, m.support_)

    return call, [
        "FIT ONLY. kernel=rbf C=%g gamma=%g tol=%g nochange_steps=%d, all "
        "explicit; cache_size is the class default (the PREDICTION buffer "
        "here, DEVIATION 871) and is not a parameter of the answer"
        % (C, gamma, tol, steps),
    ]


def lane_metrics(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    mt = ml.metrics
    yt = _i32_vector(_need(arrays, lane, "y_true"))
    yp = _i32_vector(_need(arrays, lane, "y_pred"))
    y = _f32_vector(_need(arrays, lane, "y"), "y", lane)
    yhat = _f32_vector(_need(arrays, lane, "y_hat"), "y_hat", lane)
    p = _f32_vector(_need(arrays, lane, "p"), "p", lane)
    q = _f32_vector(_need(arrays, lane, "q"), "q", lane)
    sil_x = _f32_matrix(_need(arrays, lane, "sil_x"), "sil_x", lane,
                        cols=params.get("d_sil"))
    sil_l = _i32_vector(_need(arrays, lane, "sil_labels"))
    tr_x = _f32_matrix(_need(arrays, lane, "trust_x"), "trust_x", lane,
                       cols=params.get("m_trust"))
    tr_e = _f32_matrix(_need(arrays, lane, "trust_emb"), "trust_emb", lane,
                       cols=params.get("d_trust"))
    k_tru = _param(params, lane, ("k_trust", "n_neighbors"), cast=int)
    for name in ("accuracy_score", "adjusted_rand_score", "entropy",
                 "mutual_info_score", "homogeneity_score",
                 "completeness_score", "v_measure_score", "r2_score",
                 "kl_divergence", "silhouette_score", "trustworthiness"):
        if not hasattr(mt, name):
            raise Refuse("mojolearn.metrics has no %s; the vendor arm times "
                         "eleven and an arm timing ten is not a comparison"
                         % name)

    def call():
        vals = [
            mt.accuracy_score(yt, yp),
            mt.adjusted_rand_score(yt, yp),
            mt.entropy(yt),
            mt.mutual_info_score(yt, yp),
            mt.homogeneity_score(yt, yp),
            mt.completeness_score(yt, yp),
            mt.v_measure_score(yt, yp),
            mt.r2_score(y, yhat),
            mt.kl_divergence(p, q),
            mt.silhouette_score(sil_x, sil_l),
            mt.trustworthiness(tr_x, tr_e, n_neighbors=k_tru),
        ]
        return (np.asarray([float(v) for v in vals], dtype=np.float64),)

    return call, [
        "ELEVEN metrics, the same eleven the vendor arm and the Mojo lane "
        "time; rand_index is in neither",
        "hashed here as one float64 vector of the eleven answers where the "
        "vendor prints hash=- (its reason is cross-arm width, and this "
        "column is within-arm; DEVIATION 2170)",
    ]


def lane_gp(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    d = params.get("d")
    X = _f32_matrix(_need(arrays, lane, "x", "X"), "x", lane, cols=d)
    y = _f32_vector(_need(arrays, lane, "y"), "y", lane)
    Xs = _f32_matrix(_need(arrays, lane, "x_star", "X_star", "query"),
                     "x_star", lane, cols=X.shape[1])
    ls = np.asarray(_need(arrays, lane, "length_scale"), dtype=np.float64)
    ls = ls.reshape(-1)
    alpha = _param(params, lane, ("alpha",), cast=float)
    if ls.shape[0] not in (1, X.shape[1]):
        raise Refuse("RBF bound: length_scale has 1 entry (isotropic) or "
                     "n_features=%d entries (ARD), fixture has %d"
                     % (X.shape[1], ls.shape[0]))
    if ctx.loaded == "identical" and alpha not in (0.0, 2.0 ** -20):
        raise Refuse("GaussianProcessRegressor bound under "
                     "numeric_mode='identical': alpha (the Cholesky "
                     "profile's jitter) must be +0.0 or 2**-20 (DEVIATION "
                     "1637, gp_validate_alpha); fixture gives %r "
                     "(DEVIATION 2169)" % alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise Refuse("GaussianProcessRegressor bound: alpha must be finite "
                     "and non-negative (DEVIATION 1768), fixture gives %r"
                     % alpha)
    kernel_ls = float(ls[0]) if ls.shape[0] == 1 else ls.tolist()
    probe = ml.GaussianProcessRegressor(kernel=ml.RBF(length_scale=kernel_ls),
                                        alpha=alpha)
    _tier_check(lane, probe, ctx.loaded)
    n_train = X.shape[0]

    def call():
        m = ml.GaussianProcessRegressor(kernel=ml.RBF(length_scale=kernel_ls),
                                        alpha=alpha, optimizer=None,
                                        normalize_y=False)
        try:
            m.fit(X, y)
            mean, std = m.predict(Xs, return_std=True)
        except Exception as exc:                           # noqa: BLE001
            # DEVIATION 2271: a Gram that does not factor at this alpha
            # (the scale tier's 2**-20 on a 20,000 x 20,000 float32 RBF
            # Gram is the case this was written for) is a refusal that
            # carries the factorization error, on this arm's label; main
            # prints it as one FSPEED-REFUSED and the process lives.
            raise Refuse("the %dx%d float32 RBF Gram with alpha=%.9g did "
                         "not factor (or fit/predict raised): %s: %s"
                         % (n_train, n_train, alpha, exc.__class__.__name__,
                            " ".join(str(exc).split())))
        return (mean, std)

    return call, [
        "kernel=RBF(length_scale from the fixture), alpha=%g, optimizer="
        "None, normalize_y=False on both arms; fit plus predict(return_std="
        "True) inside the clock, matching the vendor%s"
        % (alpha, " (alpha=2**-20 is the identical-mode jitter the scale "
                  "tier carries, DEVIATION 2271)"
                  if alpha == 2.0 ** -20 else ""),
        "this arm is float32 end to end; the vendor's sklearn-cpu fallback "
        "is float64 (a difference that cannot be turned off on either side, "
        "DEVIATION 2169) and its gpytorch-gpu incumbent is float32 with the "
        "solver forced to Cholesky (DEVIATION 2231)",
    ]


def lane_spectral(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    d = params.get("d")
    X = _f32_matrix(_need(arrays, lane, "x", "X"), "x", lane, cols=d)
    n = X.shape[0]
    k = _param(params, lane, ("n_clusters",), cast=int)
    nc = _param(params, lane, ("n_components",), cast=int)
    nnb = _param(params, lane, ("n_neighbors",), cast=int)
    n_init = _param(params, lane, ("n_init",), cast=int)
    seed = _param(params, lane, ("seed", "random_state"), cast=int)
    if k > n:
        raise Refuse("SpectralClustering bound: n_clusters=%d exceeds "
                     "n_samples=%d" % (k, n))
    if nc >= n:
        raise Refuse("SpectralClustering bound: 1 <= n_components < "
                     "n_samples (cuVS's RAFT_EXPECTS), fixture asks %d on %d "
                     "rows" % (nc, n))
    if nnb > n:
        raise Refuse("SpectralClustering bound: n_neighbors=%d exceeds "
                     "n_samples=%d" % (nnb, n))
    if seed < 0 or seed >= 2 ** 32:
        raise Refuse("SpectralClustering bound: 0 <= random_state < 2**32, "
                     "fixture gives %d" % seed)
    if not np.isfinite(X).all():
        raise Refuse("SpectralClustering bound: X must be finite")
    _tier_check(lane, ml.SpectralClustering(n_clusters=k), ctx.loaded)

    def call():
        m = ml.SpectralClustering(n_clusters=k, n_components=nc,
                                  affinity="nearest_neighbors",
                                  n_neighbors=nnb, n_init=n_init,
                                  random_state=seed)
        m.fit(X)
        return (m.labels_,)

    return call, [
        "affinity=nearest_neighbors n_neighbors=%d n_clusters=%d "
        "n_components=%d n_init=%d random_state=%d, the vendor arm's knobs; "
        "eigen_tol is the class default %s (cuVS's struct default; 'auto' "
        "is refused, DEVIATION 890)"
        % (nnb, k, nc, n_init, seed,
           getattr(getattr(ml, "_spectral_impl", None), "DEFAULT_EIGEN_TOL",
                   "1e-5")),
        "labels_ hashed here where the vendor prints hash=- (its reason, "
        "arbitrary label numbering, is cross-arm; this column is within-arm; "
        "DEVIATION 2170)",
    ]


def lane_holtwinters(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    batch = _param(params, lane, ("batch_size", "ts_num"), cast=int)
    n = params.get("n")
    y = np.asarray(_need(arrays, lane, "y"))
    if y.ndim == 1:
        if n is None:
            raise Refuse("holtwinters: y is flat and params carry no n to "
                         "reshape it by")
        y = y.reshape(int(batch), int(n))
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.shape[0] != batch:
        raise Refuse("ExponentialSmoothing bound: endog is (ts_num, n) with "
                     "each series in a ROW; fixture is %s for ts_num=%d"
                     % (y.shape, batch))
    n = y.shape[1]
    freq = _param(params, lane, ("frequency", "seasonal_periods"), cast=int)
    sp = _param(params, lane, ("start_periods",), cast=int)
    eps = _param(params, lane, ("eps",), cast=float)
    if freq < 2 or sp < 2 or freq < sp:
        raise Refuse("ExponentialSmoothing bound: seasonal_periods >= "
                     "start_periods >= 2 (holtwinters.pyx's validation, run "
                     "on the Mojo side by name), fixture gives "
                     "seasonal_periods=%d start_periods=%d" % (freq, sp))
    if n < sp * freq:
        raise Refuse("ExponentialSmoothing bound: n >= start_periods * "
                     "seasonal_periods = %d, fixture has n=%d"
                     % (sp * freq, n))
    if not (eps > 0.0):
        raise Refuse("ExponentialSmoothing bound: eps > 0, fixture gives %r"
                     % eps)
    if not np.isfinite(y).all():
        raise Refuse("ExponentialSmoothing bound: endog must be finite "
                     "(DEVIATION 664 refuses a non-finite value by name)")
    # DEVIATION 2270. The scorer is the vendor module's, looked up BEFORE
    # the race so a missing helper is a refusal and never a scoreless row;
    # the holdout is sliced here, outside the clock, and the timed fit
    # below stays on the full-length `y` exactly as before.
    holdout_rmse = vendor_helper("holtwinters_holdout_rmse")
    h = 2 * freq
    y_fit = np.ascontiguousarray(y[:, :n - h])
    y_hold = np.ascontiguousarray(y[:, n - h:])

    def call():
        m = ml.ExponentialSmoothing(y, seasonal="additive",
                                    seasonal_periods=freq, start_periods=sp,
                                    ts_num=batch, eps=eps)
        m.fit()
        return (np.ascontiguousarray(m.get_level(), dtype=np.float32),)

    def after():
        # DEVIATION 2270. One more fit on the first n - h observations of
        # every series, outside the clock, then forecast(h) against the
        # held-out block through the vendor module's helper. The class's
        # forecast(h) with index=None returns cuML's (h, ts_num) block for
        # ts_num > 1 and a flat array of h for one series (_tsa_impl.py);
        # both are brought to series-major (ts_num, h) here.
        if n - h < sp * freq:
            emit_note(lane, "FSPEED-ACC skipped: holding out h=2*seasonal_"
                            "periods=%d of n=%d leaves %d < start_periods*"
                            "seasonal_periods=%d observations, below the "
                            "estimator's own bound (DEVIATION 2270)"
                      % (h, n, n - h, sp * freq))
            return []
        m = ml.ExponentialSmoothing(y_fit, seasonal="additive",
                                    seasonal_periods=freq, start_periods=sp,
                                    ts_num=batch, eps=eps)
        m.fit()
        fc = np.asarray(m.forecast(h), dtype=np.float64)
        if fc.ndim == 1:
            fc = fc.reshape(h, batch)
        if fc.shape == (h, batch):
            fc = fc.T
        elif fc.shape != (batch, h):
            raise Refuse("ExponentialSmoothing.forecast(%d) returned %s; "
                         "expected (h, ts_num)=(%d, %d) or (ts_num, h)"
                         % (h, fc.shape, h, batch))
        fc = np.ascontiguousarray(fc)
        emit_note(lane, "holdout: the last h=%d observations of each of the "
                        "%d series are held out; the score is the RMSE of "
                        "forecast(%d) from a fit on the first %d "
                        "observations, over every series and horizon, "
                        "outside the clock (DEVIATION 2270)"
                  % (h, batch, h, n - h))
        parts = []
        for name in ("alpha_", "beta_", "gamma_"):
            v = getattr(m, name, None)
            if v is None:
                parts.append("%s: no accessor on ExponentialSmoothing"
                             % name)
                continue
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            parts.append("%s(%s) mean=%.6g min=%.6g max=%.6g"
                         % (name[:-1], name, float(v.mean()), float(v.min()),
                            float(v.max())))
        emit_note(lane, "fitted smoothing parameters of the holdout "
                        "(n-h=%d) fit: %s (DEVIATION 2270)"
                  % (n - h, "; ".join(parts)))
        return [("holdout_rmse_h%d" % h, holdout_rmse(y_hold, fc))]

    return call, [
        "seasonal=additive seasonal_periods=%d start_periods=%d ts_num=%d "
        "eps=%g, all explicit, the vendor arm's; the fixture is series-major "
        "(ts_num x n), which is this class's own (ts_num, n) layout, so no "
        "transpose is needed" % (freq, sp, batch, eps),
        "get_level() is hashed, as the vendor hashes cuML's get_level(); "
        "FSPEED-ACC is the vendor module's holtwinters_holdout_rmse on "
        "forecast(%d) from an untimed fit on the first %d observations "
        "(DEVIATION 2270)" % (h, n - h),
    ], after


def lane_kpss(ml, ctx):
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    batch = _param(params, lane, ("batch_size", "n_series"), cast=int)
    n_obs = params.get("n_obs")
    y = np.asarray(_need(arrays, lane, "y"))
    if y.ndim == 1:
        if n_obs is None:
            raise Refuse("kpss: y is flat and params carry no n_obs to "
                         "reshape it by")
        y = y.reshape(int(batch), int(n_obs))
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.shape[0] != batch:
        raise Refuse("kpss_test: fixture y is %s for batch_size=%d; expected "
                     "(batch_size, n_obs) series-major" % (y.shape, batch))
    d, D, s = 1, 0, 0
    if d + D > 2:
        raise Refuse("kpss_test bound: d + D <= 2")
    if not np.isfinite(y).all():
        raise Refuse("kpss_test bound: y must be finite (non-finite values "
                     "are refused by name)")
    # `kpss_test` takes cuML's (n_obs, n_series) layout, each series in a
    # COLUMN. The fixture is series-major, so the view handed in is `y.T`,
    # and `_series_major` transposes it back with a copy INSIDE the timer
    # (DEVIATION 2168). Held here so the fixture array outlives every call.
    y_cols = y.T

    def call():
        return (ml.kpss_test(y_cols, d=d, D=D, s=s, pval_threshold=0.05),)

    return call, [
        "kpss_test at d=1 D=0 s=0 pval_threshold=0.05, the same batch of "
        "%d series the vendor arm tests" % batch,
        "the (n_obs, n_series) layout this surface takes is transposed back "
        "to series-major with a copy inside kpss_test, so that copy is "
        "inside the timer; it is a cost of this surface and it stays "
        "(DEVIATION 2168)",
        "the hash is over the bool stationarity flags this function returns",
    ]


def lane_logistic(ml, ctx):
    """DEVIATION 2237. `mojolearn.LogisticRegression`, cuML's QN solver
    ported, with the five knobs the fixture fixes passed explicitly on
    both arms. Returns a third element, `after`, that scores the held-out
    rows through the vendor module's `binary_log_loss` (DEVIATION 2239)."""
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    X = _f32_matrix(_need(arrays, lane, "x", "X"), "x", lane,
                    cols=params.get("cols"))
    y = _f32_vector(_need(arrays, lane, "y"), "y", lane)
    n = X.shape[0]
    n_fit = _param(params, lane, ("n_fit",), cast=int)
    penalty = _param(params, lane, ("penalty",), fallback="l2", cast=str)
    C = _param(params, lane, ("C",), cast=float)
    max_iter = _param(params, lane, ("max_iter",), cast=int)
    tol = _param(params, lane, ("tol",), cast=float)
    fit_intercept = bool(_param(params, lane, ("fit_intercept",), cast=int))
    log_loss = vendor_helper("binary_log_loss")
    if y.shape[0] != n:
        raise Refuse("LogisticRegression bound: X and y lengths differ "
                     "(%d vs %d)" % (n, y.shape[0]))
    if n_fit < 2 or n_fit >= n:
        raise Refuse("logistic: n_fit=%d must leave at least one held-out "
                     "row of %d and at least two fitted" % (n_fit, n))
    if C <= 0.0:
        raise Refuse("LogisticRegression bound: C must be positive "
                     "(linear_model.py), fixture gives %r" % C)
    if penalty not in ("l1", "l2", "elasticnet", None):
        raise Refuse("LogisticRegression bound: penalty %r not supported"
                     % penalty)
    Xf, yf = np.ascontiguousarray(X[:n_fit]), np.ascontiguousarray(y[:n_fit])
    Xh, yh = np.ascontiguousarray(X[n_fit:]), np.ascontiguousarray(y[n_fit:])
    classes = np.unique(yf)
    if classes.shape[0] != 2:
        raise Refuse("LogisticRegression bound: binary only (softmax is not "
                     "ported, glm/NOT_IMPLEMENTED.tsv), fitted rows carry %d "
                     "classes" % classes.shape[0])
    kw = dict(penalty=penalty, C=C, max_iter=max_iter, tol=tol,
              fit_intercept=fit_intercept, solver="qn")
    _tier_check(lane, ml.LogisticRegression(**kw), ctx.loaded)
    last = {}

    def call():
        m = ml.LogisticRegression(**kw)
        m.fit(Xf, yf)
        last["m"] = m
        return (m.coef_, m.intercept_)

    def after():
        m = last.get("m")
        if m is None:
            return []
        proba = np.asarray(m.predict_proba(Xh))[:, 1]
        return [("holdout_logloss_n%d" % yh.shape[0], log_loss(proba, yh))]

    return call, [
        "penalty=%s C=%g max_iter=%d tol=%g fit_intercept=%s solver=qn, all "
        "explicit, the vendor arm's values (linesearch_max_iter=50 and "
        "lbfgs_memory=5 are the shared defaults); the fit is rows [0, %d) "
        "and rows [%d, %d) are held out for the log loss"
        % (penalty, C, max_iter, tol, fit_intercept, n_fit, n_fit, n),
        "hash(coef_, intercept_); FSPEED-ACC is the vendor module's "
        "binary_log_loss on predict_proba[:, 1] of the held-out rows, "
        "outside the clock (DEVIATION 2239)",
    ], after


def lane_arima(ml, ctx):
    """DEVIATION 2238. `mojolearn.ARIMA(order=(p, d, q))` on the
    series-major fixture, which is this class's own layout; `trend` is
    spelled from the fixture's `fit_intercept` so both arms carry the same
    `k`. Returns `after`, which scores `predict(0, n_obs)` through the
    vendor module's `arima_insample_rmse` (DEVIATION 2239)."""
    lane, arrays, params = ctx.lane, ctx.arrays, ctx.params
    batch = _param(params, lane, ("batch_size",), cast=int)
    n_obs = params.get("n_obs")
    y = np.asarray(_need(arrays, lane, "y"))
    if y.ndim == 1:
        if n_obs is None:
            raise Refuse("arima: y is flat and params carry no n_obs to "
                         "reshape it by")
        y = y.reshape(int(batch), int(n_obs))
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.shape[0] != batch:
        raise Refuse("ARIMA: fixture y is %s for batch_size=%d; expected "
                     "(batch_size, n_obs) series-major" % (y.shape, batch))
    n_obs = y.shape[1]
    p = _param(params, lane, ("p",), cast=int)
    d = _param(params, lane, ("d",), cast=int)
    q = _param(params, lane, ("q",), cast=int)
    maxiter = _param(params, lane, ("maxiter", "max_iter"), fallback=1000,
                     cast=int)
    fit_intercept = bool(_param(params, lane, ("fit_intercept",), cast=int))
    rmse = vendor_helper("arima_insample_rmse")
    if p > 8 or q > 8 or d > 2 or p + q == 0:
        raise Refuse("ARIMA bound: p <= 8, q <= 8, d <= 2 and at least one "
                     "parameter (arima_common.mojo::validate_order, cuML's "
                     "words), fixture asks (%d, %d, %d)" % (p, d, q))
    if not np.isfinite(y).all():
        raise Refuse("ARIMA bound: y must be finite (batched_arima.mojo "
                     "refuses a non-finite value by name)")
    # cuML's fit_intercept is this class's `trend`: 'c' is True, 'n' is
    # False; `trend=None` would resolve by statsmodels' rule (no intercept
    # once differenced), so it is spelled out rather than left to a rule.
    trend = "c" if fit_intercept else "n"
    _tier_check(lane, ml.ARIMA(order=(p, d, q), trend=trend, maxiter=maxiter),
                ctx.loaded)
    last = {}

    def call():
        m = ml.ARIMA(order=(p, d, q), seasonal_order=(0, 0, 0, 0),
                     trend=trend, method="ml", maxiter=maxiter)
        m.fit(y)
        last["m"] = m
        return (m.ar_, m.ma_, m.sigma2_)

    def after():
        m = last.get("m")
        if m is None:
            return []
        pred = m.predict(0, n_obs)              # (batch_size, n_obs)
        return [("insample_rmse_skip%d" % d, rmse(pred, y, d))]

    return call, [
        "order=(%d,%d,%d) seasonal_order=(0,0,0,0) trend=%r (cuML's "
        "fit_intercept=%s, k=%d) method=ml maxiter=%d, the vendor arm's "
        "values; y is series-major (batch_size x n_obs), this class's own "
        "layout, so no transpose on this side; this arm is float32 and "
        "cuML's ARIMA is float64 only"
        % (p, d, q, trend, fit_intercept, 1 if fit_intercept else 0, maxiter),
        "hash(ar_, ma_, sigma2_), the three blocks of params_ this order "
        "has; FSPEED-ACC is the vendor module's arima_insample_rmse on "
        "predict(0, n_obs) skipping the first %d differenced observations, "
        "outside the clock (DEVIATION 2239)" % d,
    ], after


LANES = {
    "kmeans": lane_kmeans,
    "dbscan": lane_dbscan,
    "pca": lane_pca,
    "ols": lane_ols,
    "knn": lane_knn,
    "cd": lane_cd,
    "kde": lane_kde,
    "linkage": lane_linkage,
    "svm": lane_svm,
    "metrics": lane_metrics,
    "gp": lane_gp,
    "spectral": lane_spectral,
    "holtwinters": lane_holtwinters,
    "kpss": lane_kpss,
    "logistic": lane_logistic,
    "arima": lane_arima,
}

#: What each lane calls in `python/mojolearn/`, for `--list-arms` and the
#: report. Fit and hash columns mirror `tools/speed_cuml_arm.py`'s lane.
OUR_ENTRY_POINTS = {
    "kmeans": "mojolearn.KMeans(init='array', n_init=1).fit -> "
              "hash(cluster_centers_, labels_)",
    "dbscan": "mojolearn.DBSCAN(algorithm='rbc').fit -> hash(labels_)",
    "pca": "mojolearn.PCA(svd_solver='jacobi').fit -> hash(components_, "
           "explained_variance_, singular_values_)",
    "ols": "mojolearn.LinearRegression(fit_intercept=False).fit -> "
           "hash(coef_)",
    "knn": "mojolearn.NearestNeighbors(algorithm='brute').kneighbors -> "
           "hash(dist, ind); fit outside the clock",
    "cd": "mojolearn.Lasso(selection='cyclic').fit -> hash(coef_)",
    "kde": "mojolearn.KernelDensity(kernel='gaussian').score_samples -> "
           "hash(log density); fit outside the clock",
    "linkage": "mojolearn.AgglomerativeClustering(linkage='single', "
               "connectivity='pairwise').fit -> hash(labels_)",
    "svm": "mojolearn.SVC(kernel='rbf').fit -> hash(dual_coef_, support_)",
    "metrics": "mojolearn.metrics.{eleven} -> hash(float64[11])",
    "gp": "mojolearn.GaussianProcessRegressor(RBF).fit + predict("
          "return_std=True) -> hash(mean, std)",
    "spectral": "mojolearn.SpectralClustering(affinity='nearest_neighbors')"
                ".fit -> hash(labels_)",
    "holtwinters": "mojolearn.ExponentialSmoothing(seasonal='additive')"
                   ".fit -> hash(get_level()); FSPEED-ACC holdout RMSE of "
                   "forecast(2 * seasonal_periods)",
    "kpss": "mojolearn.kpss_test(d=1) -> hash(stationary flags)",
    "logistic": "mojolearn.LogisticRegression(penalty='l2', C=1.0, "
                "max_iter=100, tol=1e-4, fit_intercept=True).fit -> "
                "hash(coef_, intercept_); FSPEED-ACC holdout log loss",
    "arima": "mojolearn.ARIMA(order=(1,1,1), trend='n').fit -> hash(ar_, "
             "ma_, sigma2_); FSPEED-ACC in-sample one-step RMSE",
}


class _Ctx(object):
    def __init__(self, lane, arrays, params, loaded):
        self.lane = lane
        self.arrays = arrays
        self.params = params
        self.loaded = loaded


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="classical_py_speed_arm",
        description="mojolearn's PUBLIC PYTHON SURFACE on the classical "
                    "lanes, in the numeric mode MOJOLEARN_NUMERIC_MODE "
                    "selected at import (read back and printed on the "
                    "header), on the vendor arm's own fixture bytes; one "
                    "lane per process",
    )
    p.add_argument("--lane", default=os.environ.get("MOJOLEARN_SPEED_LANE", ""),
                   help="one of: %s (env MOJOLEARN_SPEED_LANE)"
                        % " ".join(VENDOR_LANES))
    p.add_argument("--rounds", type=int,
                   default=int(os.environ.get("MOJOLEARN_SPEED_ROUNDS", "5")),
                   help="timed rounds after one untimed warm-up (env "
                        "MOJOLEARN_SPEED_ROUNDS, default 5)")
    p.add_argument("--size", choices=SIZES,
                   default=os.environ.get("MOJOLEARN_SPEED_SIZE", "shipped"),
                   help="shipped | smoke | large | wide | scale (env "
                        "MOJOLEARN_SPEED_SIZE)")
    p.add_argument("--list-arms", action="store_true",
                   help="print the lane's public entry point and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    lane = args.lane.strip()
    if lane not in VENDOR_LANES:
        raise SystemExit("--lane / MOJOLEARN_SPEED_LANE must be one of: %s; "
                         "got %r" % (" ".join(VENDOR_LANES), lane))
    if args.rounds < 1:
        raise SystemExit("--rounds / MOJOLEARN_SPEED_ROUNDS must be >= 1")
    if args.size not in SIZES:
        raise SystemExit("--size / MOJOLEARN_SPEED_SIZE must be one of %s"
                         % "/".join(SIZES))

    if args.list_arms:
        print("ours  %s" % OUR_ENTRY_POINTS.get(lane, NO_PYTHON_SURFACE.get(lane)))
        return 0

    # DEVIATION 2171: the eight lanes with no Python door refuse BEFORE
    # anything is imported, so the line is the same on every box.
    if lane in NO_PYTHON_SURFACE:
        emit_refused(lane, NO_PYTHON_SURFACE[lane])
        return 0

    try:
        arrays, shape_tag, params = load_fixture(lane, args.size)
    except Refuse as exc:
        emit_refused(lane, exc)
        return 0

    # THE IMPORT IS THE MODE CHOICE, and it happens here, after the fixture,
    # so a fixture refusal never depends on whether the package imports.
    try:
        import mojolearn                                   # noqa: PLC0415
    except Exception as exc:                               # noqa: BLE001
        emit_refused(lane, "import mojolearn failed: %s: %s"
                     % (exc.__class__.__name__, " ".join(str(exc).split())))
        return 0
    try:
        mode, loaded = mode_label(mojolearn)
    except Refuse as exc:
        emit_refused(lane, exc)
        return 0
    except Exception as exc:                               # noqa: BLE001
        emit_refused(lane, "mojolearn.numeric_mode() raised %s: %s (a tier "
                           "read-back that fails is a mislabelled arm, and "
                           "a mislabelled arm does not run)"
                     % (exc.__class__.__name__, " ".join(str(exc).split())))
        return 0

    device = device_string()
    notes = [
        "surface=python: this is the python/mojolearn/ ours arm (DEVIATION "
        "2160); the Mojo driver's arm=ours rows time the entry points "
        "directly",
        "tier read back from mojolearn.numeric_mode()=%s; "
        "MOJOLEARN_NUMERIC_MODE=%r in the environment%s (DEVIATION 2162)"
        % (loaded, os.environ.get("MOJOLEARN_NUMERIC_MODE", ""),
           "" if os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
           == loaded else " -- DISAGREES with the read-back; the header "
                          "carries the read-back"),
        "hash is sha256/16 over dtype+shape+bytes (hash_predictions%s), "
        "within-arm only, never gated (DEVIATION 2164)"
        % ("" if HASH_IMPORTED else ", re-spelled locally: tools/ was not "
                                    "importable"),
    ]
    if loaded != "identical":
        notes.append("mode=%s is NOT the identity-cost grid's question, which "
                     "runs this arm under MOJOLEARN_NUMERIC_MODE=identical; "
                     "this row is a %s-mode row and the table flags it" % (mode, mode))
    try:
        vendor_api = mojolearn.vendor()
        arch = mojolearn.gpu_arch()
        notes.append("binaries: vendor=%s gpu_arch=%s (read back from the "
                     "loaded set)" % (vendor_api, arch))
    except Exception as exc:                               # noqa: BLE001
        notes.append("mojolearn.vendor()/gpu_arch() raised %s: %s"
                     % (exc.__class__.__name__, " ".join(str(exc).split())))

    ctx = _Ctx(lane, arrays, params, loaded)
    try:
        prepared = LANES[lane](mojolearn, ctx)
        # DEVIATION 2239: a scoring lane returns (call, notes, after).
        call, lane_notes = prepared[0], prepared[1]
        after = prepared[2] if len(prepared) > 2 else None
    except Refuse as exc:
        emit_refused(lane, exc)
        return 0
    except Exception as exc:                               # noqa: BLE001
        # A LANE THAT DIES IS A REFUSAL, NOT A CRASH. The box is rented and
        # the leg body runs one process per lane; a traceback that took the
        # exit code with it would look the same as a lane nobody ran.
        emit_refused(lane, "%s while preparing: %s"
                     % (exc.__class__.__name__, " ".join(str(exc).split())))
        return 0

    try:
        race(lane, shape_tag, args.rounds, args.size, mode, device, call,
             notes=notes + list(lane_notes))
    except Exception as exc:                               # noqa: BLE001
        emit_refused(lane, "raised at run time: %s: %s"
                     % (exc.__class__.__name__, " ".join(str(exc).split())))
        return 0
    if after is not None:
        # DEVIATION 2239: the score, outside the clock, after the rounds,
        # exactly where the vendor arm prints its own.
        try:
            for metric, value in after():
                emit_acc(lane, metric, value)
        except Exception as exc:                           # noqa: BLE001
            emit_refused(lane, "FSPEED-ACC not emitted: %s: %s"
                         % (exc.__class__.__name__,
                            " ".join(str(exc).split())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
