#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Linux release smoke: every estimator family, one tier, one vendor.

    MOJOLEARN_NUMERIC_MODE=<tier> python3 packaging/linux/smoke.py \\
        --vendor cuda|hip --json <out.json>

`packaging/macos/smoke.py` fits nine families and asserts the tier read
back from the binary. This does the same on Linux and adds the vendor
axis, and it takes its fits from `tools/repeat_run_stability.py`'s lane
registry rather than re-typing them: that registry already covers every
public estimator (both forests, the three boosting policies, k-means,
k-NN x3, DBSCAN, PCA, tSVD, OLS, ridge, logistic, isolation forest, lasso,
elastic net, SVC, KDE, agglomerative, spectral, Holt-Winters, the metrics
and the GEMM), it has run on all three vendors, and a fit that works there
and not here is a packaging defect, which is the only kind this smoke is
looking for.

THREE ASSERTIONS PER RUN, and a lane that refuses is a FAILURE here, not a
row: this is a release gate, not a measurement.

ONE EXCEPTION, AND IT IS AN ASSERTION RATHER THAN AN EXEMPTION. `gemm-pinned`
asks `linalg.matmul(..., identical=True)`, and under `fast` and
`deterministic` the library REFUSES it by name, because those tiers make no
cross-vendor claim and returning a product anyway would be the library
asserting something it cannot honour. `repeat_run_stability.py`'s own lane
says so: "REFUSED under fast, which is the correct answer and is reported
rather than skipped."

The first NVIDIA wheel leg (2026-08-30) failed the fast and deterministic
smokes on exactly this, 28 of 29 lanes green, and the one red one was the
library behaving correctly. So `DESIGNED_REFUSALS` below inverts the test for
those pairs instead of skipping them: the lane must RAISE, and a lane that
SUCCEEDS is the failure. A skip would have hidden a library that quietly
started honouring an identity request under a tier that cannot keep it.

  1. `mojolearn.numeric_mode()` reads back the tier that was asked for.
  2. `mojolearn.vendor()` reads back the vendor the box is (`--vendor`),
     and `_backend.vendor_how()` says how it was decided.
  3. `vendor_used()` on one estimator per binding agrees with (2). This
     touches every binding through the parameter path, not only the
     canonical import path.

Exit 0 only when every lane fitted and every read-back agreed.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time
import traceback


#: (lane, tier) pairs where the library is REQUIRED to refuse. See the
#: module docstring. `gemm-pinned` calls matmul(identical=True); only the
#: `identical` tier may answer it.
DESIGNED_REFUSALS = {
    ("gemm-pinned", "fast"),
    ("gemm-pinned", "deterministic"),
}


def load_lanes(repo):
    p = pathlib.Path(repo) / "tools" / "repeat_run_stability.py"
    spec = importlib.util.spec_from_file_location("repeat_run_stability", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.LANES


# One class per binding THAT CARRIES NumericModeMixin, constructed only (no
# fit): `vendor_used()` resolves the binding through `_bind()` and reads the
# constant out of it. The solver, metrics, tsa and linalg bindings have no
# mixin class (their impl modules carry private loaders), so those four are
# read through `_backend.binding(name)` in the loop below, which is the same
# read on the same tier.
PER_BINDING = {
    "_mojolearn": lambda ml: ml.KMeans(n_clusters=2),
    "_mojolearn_gbdt": lambda ml: ml.GradientBoosting(n_estimators=1),
    "_mojolearn_estimators": lambda ml: ml.PCA(n_components=1),
    "_mojolearn_rf": lambda ml: ml.RandomForestClassifier(n_estimators=1),
    "_mojolearn_trees": lambda ml: ml.ExtraTreesRegressor(n_estimators=1),
    "_mojolearn_svm": lambda ml: ml.SVC(),
}
ALL_BINDINGS = (
    "_mojolearn", "_mojolearn_gbdt", "_mojolearn_estimators", "_mojolearn_rf",
    "_mojolearn_trees", "_mojolearn_svm", "_mojolearn_solver",
    "_mojolearn_metrics", "_mojolearn_tsa", "_mojolearn_linalg",
)


#: Lines MAX appends to a message that carry no cause. A message made only
#: of these is a message with its cause thrown away.
_NOISE = (
    "To get more accurate error information",
    "set MODULAR_DEBUG",
)


def _cause(exc, full):
    """The most informative single line of a raised message.

    Walks the exception's own `str()` first and falls back to the traceback,
    taking the FIRST line that is not blank, not a traceback frame and not
    one of MAX's trailing hints. If every line is noise the whole trimmed
    message is returned rather than a hint, because a gate that reports only
    "set MODULAR_DEBUG" has reported nothing."""
    for text in (str(exc), full):
        for ln in text.splitlines():
            s = ln.strip()
            if not s or s.startswith(("File \"", "Traceback", "^", "~", "|")):
                continue
            if any(n in s for n in _NOISE):
                continue
            return s
    return _trim(full)


def _trim(text, head=24, tail=8):
    """The whole message when it is small, otherwise its head AND its tail.
    Never the tail alone."""
    lines = text.strip().splitlines()
    if len(lines) <= head + tail:
        return "\n".join(lines)
    return "\n".join(lines[:head] + [f"    ... {len(lines) - head - tail} lines elided ..."] + lines[-tail:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True, choices=("cuda", "hip", "metal"))
    ap.add_argument("--arch", default="", help="the architecture the set was "
                    "built for (sm_90a, gfx942, ...); asserted against "
                    "gpu_arch() when given")
    ap.add_argument("--json", default="")
    ap.add_argument("--repo", default=os.environ.get(
        "MOJOLEARN_REPO", str(pathlib.Path(__file__).resolve().parents[2])))
    ap.add_argument("--lanes", default="", help="comma list; default all")
    a = ap.parse_args()

    want_mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower() or "fast"
    report = {"vendor_wanted": a.vendor, "mode_wanted": want_mode,
              "python": sys.version.split()[0], "lanes": {}, "ok": False}
    failures = []

    import mojolearn as ml
    from mojolearn import _backend
    report["mojolearn_file"] = ml.__file__
    report["version"] = ml.__version__

    mode = ml.numeric_mode()
    report["mode_loaded"] = mode
    if mode != want_mode:
        failures.append(f"numeric_mode() read back {mode!r}, asked for {want_mode!r}")

    vendor = ml.vendor()
    report["vendor_loaded"] = vendor
    report["vendor_how"] = _backend.vendor_how()
    if vendor != a.vendor:
        failures.append(f"vendor() read back {vendor!r}, the box is {a.vendor!r}")

    # THE ARCHITECTURE AXIS (2026-08-30): the selector must have opened the
    # architecture directory this leg built, chosen from the DEVICE, not
    # from the directory being the only one there.
    report["arch_loaded"] = _backend.gpu_arch()
    report["arch_how"] = _backend.gpu_arch_how()
    if a.arch and _backend.gpu_arch() != a.arch:
        failures.append(
            f"gpu_arch() read back {_backend.gpu_arch()!r} "
            f"({_backend.gpu_arch_how()}), the leg built {a.arch!r}")

    per = {}
    for name in ALL_BINDINGS:
        try:
            per[name] = _backend.read_vendor(_backend.binding(name))
        except Exception as exc:
            per[name] = f"REFUSED {type(exc).__name__}: {exc}"[:200]
        if per[name] != a.vendor:
            failures.append(f"{name}: read-back = {per[name]!r}")
    report["vendor_per_binding"] = per
    used = {}
    for name, ctor in PER_BINDING.items():
        try:
            used[name] = ctor(ml).vendor_used()
        except Exception as exc:
            used[name] = f"REFUSED {type(exc).__name__}: {exc}"[:200]
        if used[name] != a.vendor:
            failures.append(f"{name}: vendor_used() = {used[name]!r}")
    report["vendor_used_per_estimator"] = used

    lanes = load_lanes(a.repo)
    names = [n for n in sorted(lanes) if not a.lanes or n in a.lanes.split(",")]
    for name in names:
        t0 = time.time()
        must_refuse = (name, mode) in DESIGNED_REFUSALS
        try:
            h = lanes[name](ml)
            if must_refuse:
                # The library answered a question this tier is not allowed
                # to answer. That is worse than a lane failing.
                msg = (f"lane {name} RETURNED a result under mode={mode}, "
                       f"where it must refuse: this tier makes no "
                       f"cross-vendor claim and the pinned product asserts "
                       f"one")
                report["lanes"][name] = {"ok": False, "error": msg,
                                         "designed_refusal": True,
                                         "seconds": round(time.time() - t0, 2)}
                failures.append(msg)
                print(f"  {name:<16} FAIL {msg[:120]}")
            else:
                report["lanes"][name] = {"ok": True, "hash": h,
                                         "seconds": round(time.time() - t0, 2)}
                print(f"  {name:<16} ok   {h}  {time.time()-t0:6.1f}s")
        except Exception as exc:
            # NEVER `splitlines()[-1]`. MAX raises a multi-line message whose
            # CAUSE is on the first line and which then repeats
            # "To get more accurate error information, set MODULAR_DEBUG=..."
            # one to three times at the end. Taking the last line recorded
            # that hint for all 27 failing lanes of the 2026-08-30 A40 leg
            # and threw the actual reason away, so the board said only that
            # something went wrong and not one word about what. This is the
            # third time today a gate has been caught reading a tail: see
            # sabotage.py's six-line window and audit.sh's grep -c.
            full = traceback.format_exc()
            report_err = _trim(full)
            short = _cause(exc, full)
            if must_refuse:
                report["lanes"][name] = {"ok": True, "refused": short,
                                         "refused_full": report_err,
                                         "designed_refusal": True,
                                         "seconds": round(time.time() - t0, 2)}
                print(f"  {name:<16} ok   REFUSED as designed under {mode}")
            else:
                report["lanes"][name] = {"ok": False, "error": short,
                                         "error_full": report_err,
                                         "seconds": round(time.time() - t0, 2)}
                failures.append(f"lane {name}: {short}")
                print(f"  {name:<16} FAIL {short[:160]}")

    report["failures"] = failures
    report["ok"] = not failures
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"smoke v{ml.__version__} mode={mode} vendor={vendor} "
          f"({report['vendor_how']}) lanes={len(names)} "
          f"{'PASS' if not failures else 'FAIL'}")
    for f in failures:
        print("  ", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
