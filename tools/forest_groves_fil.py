#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cuML FIL opponent arm on the same box and the same rows
(lane/forest-groves-cpu-and-speed, 2026-09-17). Runs on the pod's system
python with `cuml-cu12` and `treelite` installed; imports nothing from
mojolearn (our archive is read with numpy).

Two arms per model:

  fil-ours   OUR trained forest (the saved archive's five arrays) rebuilt as a
             treelite model through treelite.model_builder and loaded into
             cuml.fil.ForestInference: the same trees, the same leaves, FIL's
             traversal and summation. `<=` routes left as ours does; averaging
             is treelite's average_tree_output. FIL's bits are its own (its
             association is device dependent); the max absolute difference
             against our saved groves prediction is reported, never a claim of
             equal bits.
  cuml-rf    cuML's own RandomForest fit on the same training rows with the same
             hyperparameters (trees, depth, n_bins 128, the same max_features),
             timed through its cached nvForest model, the cell the existing
             21.09 ms against 10.70 ms figure was taken with. Independent trees.

Both are timed as our arms are: one warmup, --rounds single calls, --rounds
blocks of --calls calls, the spread gate, the hashes of every output. Never
write "we are faster" from this file's numbers; quote both sides.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

SPREAD_GATE = 1.10


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _scalar(z, key):
    v = z[key]
    return str(v.item()) if hasattr(v, "item") else str(v)


def read_archive(path):
    z = np.load(path, allow_pickle=False)
    fmt = _scalar(z, "format")
    est = _scalar(z, "estimator")
    meta = z["meta"].astype(np.int64)
    out = dict(format=fmt, estimator=est, n_features=int(meta[0]), n_trees=int(meta[1]), num_outputs=int(meta[2]),
               offsets=z["offsets"].astype(np.int32), colid=z["colid"].astype(np.int32),
               quesval=z["quesval"].astype(np.float32), left=z["left_child"].astype(np.int32),
               leaves=z["leaves"].astype(np.float32), classes=(z["classes"] if "classes" in z.files else None))
    return out


def treelite_from_archive(a):
    """Our flat forest as a treelite model: every node under its local id,
    `feature <= threshold` left, the leaf vector (classifier) or scalar."""
    import treelite
    from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation
    k = a["num_outputs"]
    classifier = a["classes"] is not None
    n_trees = a["n_trees"]
    if classifier:
        meta = Metadata(num_feature=a["n_features"], task_type="kMultiClf", average_tree_output=True,
                        num_target=1, num_class=[k], leaf_vector_shape=(1, k))
        ann = TreeAnnotation(num_tree=n_trees, target_id=[0] * n_trees, class_id=[-1] * n_trees)
        base = [0.0] * k
    else:
        meta = Metadata(num_feature=a["n_features"], task_type="kRegressor", average_tree_output=True,
                        num_target=1, num_class=[1], leaf_vector_shape=(1, 1))
        ann = TreeAnnotation(num_tree=n_trees, target_id=[0] * n_trees, class_id=[0] * n_trees)
        base = [0.0]
    b = ModelBuilder(threshold_type="float32", leaf_output_type="float32", metadata=meta,
                     tree_annotation=ann, postprocessor=PostProcessorFunc(name="identity"), base_scores=base)
    off, col, thr, left, leaves = a["offsets"], a["colid"], a["quesval"], a["left"], a["leaves"].reshape(-1, k)
    t0 = time.perf_counter()
    for t in range(n_trees):
        lo, hi = int(off[t]), int(off[t + 1])
        b.start_tree()
        for node in range(lo, hi):
            local = node - lo
            b.start_node(local)
            l = int(left[node])
            if l == -1:
                if classifier:
                    b.leaf([float(v) for v in leaves[node]])
                else:
                    b.leaf(float(leaves[node, 0]))
            else:
                b.numerical_test(feature_id=int(col[node]), threshold=float(thr[node]), default_left=True,
                                 opname="<=", left_child_key=l, right_child_key=l + 1)
            b.end_node()
        b.end_tree()
    model = b.commit()
    return model, time.perf_counter() - t0


def timed(method, x, rounds, calls):
    def digest(r):
        r = np.asarray(r)
        return _sha(r.tobytes()) + f":{r.dtype}:{tuple(r.shape)}"

    def call():
        r = method(x)
        if hasattr(r, "get"):
            r = r.get()
        elif hasattr(r, "to_numpy"):
            r = r.to_numpy()
        return np.asarray(r)

    t0 = time.perf_counter()
    warm = call()
    warm_ms = (time.perf_counter() - t0) * 1000
    hashes = [digest(warm)]
    single = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        r = call()
        single.append((time.perf_counter() - t0) * 1000)
        hashes.append(digest(r))
    blocks = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        kept = [call() for _ in range(calls)]
        blocks.append((time.perf_counter() - t0) * 1000 / calls)
        hashes.extend(digest(r) for r in kept)
        del kept
    ss, bs = max(single) / min(single), max(blocks) / min(blocks)
    return warm, dict(warmup_ms=warm_ms, single_ms=single, single_median_ms=statistics.median(single), single_spread=ss,
                      single_stable=ss <= SPREAD_GATE and len(single) >= 5, calls_per_block=calls,
                      block_ms_per_call=blocks, block_median_ms=statistics.median(blocks), block_spread=bs,
                      block_stable=bs <= SPREAD_GATE and len(blocks) >= 5, hash=hashes[0],
                      hashes_equal=len(set(hashes)) == 1, output_shape=list(np.asarray(warm).shape),
                      output_dtype=str(np.asarray(warm).dtype))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="our saved archive (.npz)")
    ap.add_argument("--x", required=True, help="the fixed prediction rows (.npy)")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--json", required=True)
    ap.add_argument("--ours", default="", help="dir holding <model>.groves.pred.npy from forest_groves_speed.py")
    ap.add_argument("--no-treelite", action="store_true")
    ap.add_argument("--no-cuml-rf", action="store_true")
    args = ap.parse_args()
    import cuml
    import cupy
    import treelite
    a = read_archive(args.model)
    x = np.ascontiguousarray(np.load(args.x), dtype=np.float32)
    stem = os.path.basename(args.model)[:-4]
    classifier = a["classes"] is not None
    out = dict(model=os.path.abspath(args.model), x=os.path.abspath(args.x), rows=int(x.shape[0]),
               features=int(x.shape[1]), estimator=a["estimator"], n_trees=a["n_trees"], outputs=a["num_outputs"],
               nodes=int(a["colid"].size), cuml=cuml.__version__, treelite=treelite.__version__,
               gpu=cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), arms={})
    ours = None
    if args.ours:
        p = os.path.join(args.ours, stem + ".groves.pred.npy")
        if os.path.exists(p):
            ours = np.load(p)
    if not args.no_treelite:
        from cuml.fil import ForestInference
        tl, build_s = treelite_from_archive(a)
        fil = ForestInference.load_from_treelite_model(tl, is_classifier=classifier, output_type="numpy")
        method = fil.predict_proba if classifier else fil.predict
        pred, rec = timed(method, x, args.rounds, args.calls)
        rec["treelite_build_s"] = round(build_s, 1)
        if ours is not None:
            p = np.asarray(pred, dtype=np.float64).reshape(len(pred), -1)
            o = np.asarray(ours, dtype=np.float64).reshape(len(ours), -1)
            if p.shape == o.shape:
                rec["max_abs_diff_vs_our_groves"] = float(np.max(np.abs(p - o)))
                rec["rows_bits_differ_vs_our_groves"] = int(np.count_nonzero(
                    (np.asarray(pred, dtype=np.float32).reshape(len(pred), -1) != np.asarray(ours, dtype=np.float32).reshape(len(ours), -1)).any(axis=1)))
            else:
                rec["shape_vs_our_groves"] = [list(p.shape), list(o.shape)]
        out["arms"]["fil-ours"] = rec
        print(f"fil-ours {stem} single={rec['single_median_ms']:.2f}ms spread={rec['single_spread']:.3f} "
              f"block{args.calls}={rec['block_median_ms']:.2f}ms/call spread={rec['block_spread']:.3f} "
              f"maxdiff={rec.get('max_abs_diff_vs_our_groves')}", flush=True)
        del fil, tl
    if not args.no_cuml_rf and a["estimator"].startswith("RandomForest"):
        from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
        dataset = os.path.basename(args.x)[2:-4]
        xt = np.load(os.path.join(os.path.dirname(args.x), f"xtrain_{dataset}.npy"))
        yt = np.load(os.path.join(os.path.dirname(args.x), f"ytrain_{dataset}.npy"))
        with open(os.path.join(os.path.dirname(args.model), "manifest.json")) as fh:
            info = json.load(fh)["models"][stem]
        cfg = info["config"]
        common = dict(n_estimators=a["n_trees"], max_depth=info["depth"], n_bins=int(cfg.get("n_bins", 128)),
                      max_features=float(cfg["max_features"]) if str(cfg.get("max_features", "")).replace(".", "", 1).isdigit() else cfg.get("max_features", "sqrt"),
                      min_samples_leaf=int(cfg.get("min_samples_leaf", 1)), min_samples_split=int(cfg.get("min_samples_split", 2)),
                      bootstrap=bool(cfg.get("bootstrap", True)), random_state=int(cfg.get("seed", 7)))
        t0 = time.perf_counter()
        if classifier:
            m = RandomForestClassifier(split_criterion=0, **common)
            m.fit(xt, yt.astype(np.int32))
        else:
            m = RandomForestRegressor(split_criterion=2, **common)
            m.fit(xt, yt)
        cupy.cuda.runtime.deviceSynchronize()
        fit_s = time.perf_counter() - t0
        method = m.predict_proba if classifier else m.predict
        _, rec = timed(method, x, args.rounds, args.calls)
        rec["fit_s"] = round(fit_s, 1)
        rec["hyperparameters"] = {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in common.items()}
        out["arms"]["cuml-rf"] = rec
        print(f"cuml-rf {stem} fit={fit_s:.1f}s single={rec['single_median_ms']:.2f}ms spread={rec['single_spread']:.3f} "
              f"block{args.calls}={rec['block_median_ms']:.2f}ms/call spread={rec['block_spread']:.3f}", flush=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
