#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane gbdt-fairness: try to BREAK the GBDT speed claim, not to confirm it.

The claim under test (bench/OPPONENT_REFERENCE.md, September 12, pod
u4elzj1eo486ps): our IDENTICAL arm at 1,000,000 taxi rows fits 100 symmetric
trees in 310.1 ms against CatBoost GPU's 709.0, a ratio of 0.437x. CatBoost's
CUDA learner is a production implementation, so the ratio is not credible on
its face and every subcommand here exists to attack one way it could be wrong.

OUTCOME (2026-09-12, merge 6a779d0e). The claim partly broke. Our model
reproduced bit for bit, but CatBoost ran 624-636 ms on a second pod against the
709.0 recorded above, so the published row was timed against a CatBoost sample
at the slow end: taxi symmetric is 0.525x and taxi depthwise 0.591x. The
leading hypothesis below (`race`) is DEAD -- a real-drain arm came out 1%
FASTER. What `decompose` found is the real story: ours is 86.1 ms fixed +
2.297 ms/tree against CatBoost's 346.0 ms fixed + 2.809 ms/tree, so the
advantage is 4.0x before the first tree and only 1.22x per tree, and it shrinks
to about 0.75x at CatBoost's own 1000-iteration default.

Note on framing: do NOT describe any of this as a price paid for bitwise
determinism. That quantity is not measurable here -- it would require our
deterministic arm against our own OPTIMIZED non-deterministic arm of the same
code, which does not exist -- and the framing is withdrawn project-wide.

    python3 tools/gbdt_fairness_probe.py race      --dataset taxi
    python3 tools/gbdt_fairness_probe.py models    --dataset taxi
    python3 tools/gbdt_fairness_probe.py decompose --dataset taxi
    python3 tools/gbdt_fairness_probe.py flags     --dataset taxi
    python3 tools/gbdt_fairness_probe.py b2b       --dataset taxi

WHAT EACH SUBCOMMAND ATTACKS

`race`  THE LEADING HYPOTHESIS: that `bench/speed/forest_speed_arm.py::_our_sync`
        is a no-op and our clock therefore stops while kernels are still in
        flight, while CatBoost's `_blocking` no-op is honest because its fit
        really is host-synchronous. That asymmetry alone would produce this
        ratio. A THIRD arm, `ours-sync`, is our arm with a REAL device drain
        (torch, cudart through ctypes, and cupy where present -- all of them,
        so no CUDA-context subtlety can let work slip past one of them) and is
        interleaved with the other two in ONE process and one heat window.
        If `ours` and `ours-sync` differ, the shipped number is inflated by
        exactly that difference.

`models` THE CHECK THAT DOES NOT EXIST TODAY: nothing in the harness compares
        the PRODUCED models, only the configs. If our ensemble is smaller or
        shallower we are solving a cheaper problem and the ratio is
        meaningless. Dumps tree count, per-tree leaf counts and total leaves
        for BOTH libraries from fits on the same rows.

`decompose` Whether the cell is a comparison of BOOSTING or of STARTUP. Each
        arm is timed at 1, 10 and 100 trees. The slope over trees is the
        per-tree cost and the intercept is what the arm pays before it boosts
        at all (pool build, quantization, CUDA setup). An arm that wins only
        on the intercept is not a faster learner.

`flags` Whether a pinned value puts CatBoost on a slow path it would not
        normally take. `bootstrap_type='No'` and `boosting_type='Plain'` are
        matched on both arms, but a setting that is "matched" and pathological
        for one of them is not a fair comparison. CatBoost is re-timed under
        its own defaults, one pinned value released at a time.

`b2b`   A context-free check on the sync question that needs no CUDA API at
        all: N fits back to back with no scoring and no sync between them. If
        our per-fit number hides in-flight work, the device cannot outrun the
        host forever and the back-to-back average exceeds the per-fit median.

Every subcommand prints lines beginning `FAIR ` so the Mac can parse one log.
Nothing here writes to bench/results; the leg fetches the logs home.
"""

import argparse
import ctypes
import json
import os
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python"),
           os.path.join(_ROOT, "bench", "speed")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec            # noqa: E402
import forest_speed_arm as fsa           # noqa: E402


# --------------------------------------------------------------------------
# The device drain our arm does not do.
# --------------------------------------------------------------------------

def build_device_sync():
    """A REAL drain, through every mechanism this box offers.

    Three, not one, and all of them are called. `cudaDeviceSynchronize` waits
    on the calling thread's CURRENT context, so a single mechanism could in
    principle drain a context that is not the one MAX enqueued into and look
    like a no-op for the wrong reason. Calling torch's, cudart's and cupy's
    drains together removes that argument from the result. The list of what
    was actually armed is printed once, because a sync that silently did
    nothing would turn this whole experiment into a confirmation of the thing
    it is supposed to attack."""
    backends = []

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
            backends.append(("torch", torch.cuda.synchronize))
    except Exception as exc:                       # noqa: BLE001
        print("FAIR sync-backend torch UNAVAILABLE: %s" % exc)

    try:
        lib = None
        for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            import torch as _t                      # the image ships one
            import glob
            cands = glob.glob(os.path.join(os.path.dirname(_t.__file__),
                                           "lib", "libcudart*"))
            if cands:
                lib = ctypes.CDLL(cands[0])
        if lib is not None:
            fn = lib.cudaDeviceSynchronize
            fn.restype = ctypes.c_int

            def _cudart():
                rc = fn()
                if rc != 0:
                    raise RuntimeError("cudaDeviceSynchronize -> %d" % rc)

            _cudart()
            backends.append(("cudart", _cudart))
    except Exception as exc:                       # noqa: BLE001
        print("FAIR sync-backend cudart UNAVAILABLE: %s" % exc)

    try:
        import cupy
        cupy.cuda.runtime.deviceSynchronize()
        backends.append(("cupy", cupy.cuda.runtime.deviceSynchronize))
    except Exception as exc:                       # noqa: BLE001
        print("FAIR sync-backend cupy UNAVAILABLE: %s" % exc)

    if not backends:
        raise SystemExit("FAIR FATAL: no device drain available; a no-op sync "
                         "would make this experiment meaningless")
    print("FAIR sync-backends armed=%s" % ",".join(n for n, _ in backends),
          flush=True)

    def drain():
        for _, fn in backends:
            fn()

    return drain


# --------------------------------------------------------------------------
# Shared setup.
# --------------------------------------------------------------------------

def load(dataset, rows, lane):
    size = spec.size_tag()
    data = spec.load_with_fallback(dataset, size, rows)
    # A fallback to the synthetic fixture would make every number below a
    # statement about a different problem. The pointwise lane already lost a
    # leg this way (OPPONENT_REFERENCE: "labeled the rows shape=synthclf-...").
    if not data.tag.startswith(dataset):
        raise SystemExit("FAIR FATAL: asked for %r and the loader produced %r; "
                         "the fixture is not present on this box"
                         % (dataset, data.tag))
    cfg = spec.lane_config(lane, size)
    spec.prepare_cuml_labels(data)
    fsa.prepare_our_inputs(data)
    print("FAIR dataset=%s shape=%s task=%s rows=%d features=%d"
          % (dataset, data.tag, data.task, data.X_train.shape[0],
             data.X_train.shape[1]), flush=True)
    return data, cfg, size


def our_estimator(cfg, data, **over):
    import mojolearn
    params = dict(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"], l2_leaf_reg=cfg["l2"],
        border_count=cfg["borders"], random_state=cfg["seed"],
        bootstrap_type="No", grow_policy=cfg["grow_policy"],
        loss="RMSE" if data.task == "regression" else "Logloss",
    )
    if cfg["grow_policy"] == "Lossguide":
        params["max_leaves"] = cfg["max_leaves"]
    params.update(over)
    return mojolearn.GradientBoosting(**params)


def catboost_estimator(cfg, data, **over):
    import catboost
    p = dict(
        iterations=cfg["n_estimators"], depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"], l2_leaf_reg=cfg["l2"],
        border_count=cfg["borders"], random_seed=cfg["seed"],
        bootstrap_type="No", boosting_type="Plain",
        grow_policy=cfg["grow_policy"], task_type="GPU", devices="0",
        verbose=False, allow_writing_files=False,
    )
    if cfg["grow_policy"] == "Lossguide":
        p["max_leaves"] = cfg["max_leaves"]
    p.update(over)
    for k in [k for k, v in p.items() if v is None]:
        del p[k]
    if data.task == "regression":
        return catboost.CatBoostRegressor(loss_function="RMSE", **p)
    return catboost.CatBoostClassifier(loss_function="Logloss", **p)


def timed(fn, drain, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        drain()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def report(tag, ms):
    print("FAIR TIME %s n=%d median=%.1f min=%.1f max=%.1f all=%s"
          % (tag, len(ms), statistics.median(ms), min(ms), max(ms),
             ",".join("%.1f" % v for v in ms)), flush=True)
    return statistics.median(ms)


# --------------------------------------------------------------------------
# race: the missing-sync hypothesis, inside the real harness.
# --------------------------------------------------------------------------

def cmd_race(args):
    lane = args.lane
    data, cfg, size = load(args.dataset, args.rows, lane)
    drain = build_device_sync()

    arms = fsa.build_ours(lane, cfg, data)
    if not arms:
        raise SystemExit("FAIR FATAL: our arm refused to build")

    synced = fsa.build_ours(lane, cfg, data)
    if not synced:
        raise SystemExit("FAIR FATAL: the synced copy of our arm refused")
    synced[0].name = "ours-sync"
    synced[0].sync = drain
    # Same library string, so the harness's own same-library agreement check
    # (DEVIATION 1839) compares the two arms' accuracy and fails loudly if the
    # drain somehow changed the answer.
    synced[0].library = "mojolearn"

    devices, _auto = spec.resolve_devices("auto", lane)
    opponents = [a for a in spec.build_opponents(lane, cfg, data, devices)
                 if a.name == "catboost-gpu"]
    if not opponents:
        raise SystemExit("FAIR FATAL: catboost-gpu did not build")

    every = arms + synced + opponents
    print("FAIR race arms=%s rounds=%d"
          % (",".join(a.name for a in every), args.rounds), flush=True)
    spec.run(lane, every, data, args.rounds, size, rotate_order=True)


# --------------------------------------------------------------------------
# models: are the two ensembles the same size?
# --------------------------------------------------------------------------

def _our_model_shape(model):
    counts = np.asarray(model.get_tree_leaf_counts())
    text = str(model.model_)
    declared = None
    for line in text.splitlines():
        if line.startswith("trees "):
            declared = int(line.split()[1])
            break
    return dict(library="mojolearn", trees=int(counts.size),
                trees_declared=declared,
                leaves_total=int(counts.sum()),
                leaves_per_tree_min=int(counts.min()),
                leaves_per_tree_max=int(counts.max()),
                leaves_per_tree_mean=float(counts.mean()),
                model_text_bytes=len(text))


def _catboost_model_shape(model):
    out = dict(library="catboost")
    try:
        out["trees"] = int(model.tree_count_)
    except Exception as exc:                       # noqa: BLE001
        out["trees"] = "UNAVAILABLE: %s" % exc
    counts = None
    for name in ("get_tree_leaf_counts", "_get_tree_leaf_counts"):
        try:
            counts = np.asarray(getattr(model, name)())
            break
        except Exception:                          # noqa: BLE001
            continue
    try:
        out["leaf_values"] = int(np.asarray(model.get_leaf_values()).size)
    except Exception:                              # noqa: BLE001
        out["leaf_values"] = None
    if counts is not None and counts.size:
        out.update(leaves_total=int(counts.sum()),
                   leaves_per_tree_min=int(counts.min()),
                   leaves_per_tree_max=int(counts.max()),
                   leaves_per_tree_mean=float(counts.mean()),
                   leaves_total_source="get_tree_leaf_counts")
    elif out["leaf_values"]:
        # CatBoost 1.2's Python surface does not always expose per-tree leaf
        # counts. `get_leaf_values()` is flat and holds ONE value per leaf for
        # a single-dimension objective (Logloss, RMSE), so its size is the
        # leaf total. It is labeled with the surface that answered rather than
        # silently merged, because the two are not the same measurement and a
        # reader comparing model sizes is entitled to know which one this is.
        out["leaves_total"] = out["leaf_values"]
        out["leaves_total_source"] = "get_leaf_values"
    else:
        out["leaves_total"] = "UNAVAILABLE"
    return out


def cmd_models(args):
    lane = args.lane
    data, cfg, _size = load(args.dataset, args.rows, lane)
    drain = build_device_sync()

    ours = our_estimator(cfg, data)
    t0 = time.perf_counter()
    ours.fit(data._ours_X, data._ours_y)
    drain()
    print("FAIR models-fit ours ms=%.1f" % ((time.perf_counter() - t0) * 1000.0))
    shape = _our_model_shape(ours)
    print("FAIR MODEL %s" % json.dumps(shape, sort_keys=True), flush=True)

    cat = catboost_estimator(cfg, data)
    t0 = time.perf_counter()
    cat.fit(data.X_train, data.y_train)
    drain()
    print("FAIR models-fit catboost ms=%.1f" % ((time.perf_counter() - t0) * 1000.0))
    cshape = _catboost_model_shape(cat)
    print("FAIR MODEL %s" % json.dumps(cshape, sort_keys=True), flush=True)

    # The comparison the harness never made, stated as a verdict rather than
    # left for a reader to compute from two JSON blobs.
    try:
        ratio = cshape["leaves_total"] / float(shape["leaves_total"])
        print("FAIR MODEL-VERDICT ours_leaves=%d catboost_leaves=%d "
              "catboost/ours=%.4f ours_trees=%s catboost_trees=%s"
              % (shape["leaves_total"], cshape["leaves_total"], ratio,
                 shape["trees"], cshape["trees"]), flush=True)
    except Exception as exc:                       # noqa: BLE001
        print("FAIR MODEL-VERDICT UNCOMPUTABLE: %s" % exc, flush=True)


# --------------------------------------------------------------------------
# decompose: startup cost against per-tree cost.
# --------------------------------------------------------------------------

def cmd_decompose(args):
    lane = args.lane
    data, cfg, _size = load(args.dataset, args.rows, lane)
    drain = build_device_sync()
    ladder = [int(v) for v in args.trees.split(",")]

    # one untimed warm-up per library, the harness's own contract
    our_estimator(cfg, data, n_estimators=1).fit(data._ours_X, data._ours_y)
    drain()
    catboost_estimator(cfg, data, iterations=1).fit(data.X_train, data.y_train)
    drain()

    medians = {}
    for n in ladder:
        m = timed(lambda: our_estimator(cfg, data, n_estimators=n)
                  .fit(data._ours_X, data._ours_y), drain, args.reps)
        medians[("ours", n)] = report("ours trees=%d" % n, m)
        m = timed(lambda: catboost_estimator(cfg, data, iterations=n)
                  .fit(data.X_train, data.y_train), drain, args.reps)
        medians[("catboost", n)] = report("catboost trees=%d" % n, m)

    lo, hi = min(ladder), max(ladder)
    for who in ("ours", "catboost"):
        a, b = medians[(who, lo)], medians[(who, hi)]
        per_tree = (b - a) / float(hi - lo)
        intercept = a - per_tree * lo
        print("FAIR DECOMPOSE %s fixed_ms=%.1f per_tree_ms=%.3f "
              "at%d=%.1f at%d=%.1f" % (who, intercept, per_tree, lo, a, hi, b),
              flush=True)
    ours_fixed = medians[("ours", lo)]
    cat_fixed = medians[("catboost", lo)]
    print("FAIR DECOMPOSE-VERDICT one_tree_ratio=%.4f full_ratio=%.4f"
          % (ours_fixed / cat_fixed,
             medians[("ours", hi)] / medians[("catboost", hi)]), flush=True)


# --------------------------------------------------------------------------
# flags: is the pinned config a slow path for CatBoost?
# --------------------------------------------------------------------------

CATBOOST_VARIANTS = (
    ("pinned", {}),
    ("bootstrap_default", {"bootstrap_type": None}),
    ("bootstrap_bernoulli", {"bootstrap_type": "Bernoulli", "subsample": 0.8}),
    ("boosting_default", {"boosting_type": None}),
    ("borders_128_gpu_default", {"border_count": 128}),
    ("borders_and_bootstrap_default", {"border_count": 128,
                                       "bootstrap_type": None}),
    ("all_natural", {"bootstrap_type": None, "boosting_type": None,
                     "border_count": None}),
)


def cmd_flags(args):
    lane = args.lane
    data, cfg, _size = load(args.dataset, args.rows, lane)
    drain = build_device_sync()

    catboost_estimator(cfg, data, iterations=1).fit(data.X_train, data.y_train)
    drain()

    base = None
    for name, over in CATBOOST_VARIANTS:
        def fit(over=over):
            catboost_estimator(cfg, data, **over).fit(data.X_train, data.y_train)
        med = report("catboost-%s" % name, timed(fit, drain, args.reps))
        if name == "pinned":
            base = med
        print("FAIR FLAGS %s median_ms=%.1f vs_pinned=%.4f over=%s"
              % (name, med, med / base if base else float("nan"),
                 json.dumps(over, sort_keys=True)), flush=True)


# --------------------------------------------------------------------------
# b2b: the sync question without any CUDA API.
# --------------------------------------------------------------------------

def cmd_b2b(args):
    lane = args.lane
    data, cfg, _size = load(args.dataset, args.rows, lane)
    drain = build_device_sync()

    our_estimator(cfg, data).fit(data._ours_X, data._ours_y)
    drain()

    per = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        our_estimator(cfg, data).fit(data._ours_X, data._ours_y)
        per.append((time.perf_counter() - t0) * 1000.0)
    per_median = report("ours no-sync per-fit", per)

    drain()
    t0 = time.perf_counter()
    for _ in range(args.reps):
        our_estimator(cfg, data).fit(data._ours_X, data._ours_y)
    drain()
    total = (time.perf_counter() - t0) * 1000.0
    avg = total / args.reps
    print("FAIR B2B n=%d total_ms=%.1f average_ms=%.1f per_fit_median_ms=%.1f "
          "average/median=%.4f" % (args.reps, total, avg, per_median,
                                   avg / per_median), flush=True)


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(prog="gbdt_fairness_probe")
    p.add_argument("cmd", choices=("race", "models", "decompose", "flags", "b2b"))
    p.add_argument("--dataset", default="taxi")
    p.add_argument("--lane", default="gbdt-symmetric")
    p.add_argument("--rows", type=int, default=1000000)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--trees", default="1,10,100")
    args = p.parse_args(argv)
    print("FAIR start cmd=%s lane=%s dataset=%s rows=%d mode=%s"
          % (args.cmd, args.lane, args.dataset, args.rows,
             os.environ.get("MOJOLEARN_NUMERIC_MODE", "unset")), flush=True)
    {"race": cmd_race, "models": cmd_models, "decompose": cmd_decompose,
     "flags": cmd_flags, "b2b": cmd_b2b}[args.cmd](args)
    print("FAIR done cmd=%s" % args.cmd, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
