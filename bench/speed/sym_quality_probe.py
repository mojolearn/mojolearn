"""lane/sym-quality probe: why does our symmetric GBDT score below CatBoost
on the board's gbm-bench settings, and why do 500 and 1000 trees give the
same AUC?

    python bench/speed/sym_quality_probe.py ours  taxi 1000 [key=value ...]
    python bench/speed/sym_quality_probe.py cat   taxi 1000 [key=value ...]

`ours` fits mojolearn.GradientBoosting with the board's gbdt-symmetric
parameters (bench/speed/forest_speed_arm.our_gbdt_arm), then prints the AUC
and logloss of the model truncated to every 100 trees, and the mean |leaf|
per 100-tree block. `cat` fits CatBoost with the board's parameters
(tools/speed_gbdt_arm.catboost_arms) and prints its staged AUC every 100
trees. key=value overrides one estimator keyword on that arm (A/B).
Diagnosis only; nothing here is timed.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import speed_gbdt_arm as spec  # noqa: E402


def _val(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"None": None, "True": True, "False": False}.get(v, v)


def _truncate(text, k):
    out = []
    for line in text.splitlines():
        f = line.split()
        if f and f[0] == "trees":
            out.append("trees %d" % k)
            continue
        if f and f[0] in ("tree", "ntree", "split", "leaf") and len(f) > 1:
            try:
                if int(f[1]) >= k:
                    continue
            except ValueError:
                pass
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main():
    which, ds, n_trees = sys.argv[1], sys.argv[2], int(sys.argv[3])
    extra = dict(a.split("=", 1) for a in sys.argv[4:])
    extra = {k: _val(v) for k, v in extra.items()}
    data = spec.load_dataset(ds, "shipped")
    cfg = spec.lane_config("gbdt-symmetric", "shipped")
    spw = spec.scale_pos_weight_for(cfg, data)
    print("PROBE data=%s train=%s test=%s pos_train=%.4f spw=%s extra=%s" % (
        ds, data.X_train.shape, data.X_test.shape, float(np.mean(data.y_train)), spw, extra),
        flush=True)
    xtr = np.ascontiguousarray(data.X_train, dtype=np.float32)
    ytr = np.ascontiguousarray(data.y_train, dtype=np.float32)
    xte = np.ascontiguousarray(data.X_test, dtype=np.float32)
    yte = data.y_test
    if which == "ours":
        import mojolearn
        params = dict(
            n_estimators=n_trees, max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
            l2_leaf_reg=cfg["l2"], border_count=cfg["borders"], random_state=cfg["seed"],
            bootstrap_type="No", grow_policy=cfg["grow_policy"], loss="Logloss",
            max_leaves=cfg["max_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
            random_strength=cfg["random_strength"], score_function=cfg["score_function"],
            leaf_estimation_method=cfg["leaf_estimation_method"],
            leaf_estimation_iterations=cfg["leaf_estimation_iterations"],
            feature_border_type=cfg["feature_border_type"], nan_mode=cfg["nan_mode"],
            boosting_type=cfg["boosting_type"], boost_from_average=False,
        )
        if spw is not None:
            params["class_weights"] = [1.0, spw]
        params.update(extra)
        m = mojolearn.GradientBoosting(**params)
        t0 = time.time()
        m.fit(xtr, ytr)
        print("PROBE ours fit_s=%.1f version=%s" % (time.time() - t0, mojolearn.__version__), flush=True)
        for attr in ("learning_rate_", "n_trees_", "best_iteration_", "tree_count_"):
            if hasattr(m, attr):
                print("PROBE ours %s=%r" % (attr, getattr(m, attr)), flush=True)
        counts = [int(c) for c in np.asarray(m.get_tree_leaf_counts()).reshape(-1)]
        leaves = np.asarray(m.get_leaf_values(), dtype=np.float64).reshape(-1)
        print("PROBE ours trees=%d leaves=%d" % (len(counts), int(sum(counts))), flush=True)
        off = [0]
        for c in counts:
            off.append(off[-1] + c)
        depths = [c.bit_length() - 1 for c in counts]
        for b in range(0, len(counts), 100):
            blk = depths[b:b + 100]
            print("PROBE ours block %4d depth-hist %s" % (b, {d: blk.count(d) for d in sorted(set(blk))}), flush=True)
        for b in range(0, len(counts), 100):
            e = min(b + 100, len(counts))
            v = leaves[off[b]:off[e]]
            print("PROBE ours block %4d-%4d mean|leaf|=%.3e max|leaf|=%.3e" % (
                b, e, float(np.mean(np.abs(v))), float(np.max(np.abs(v)))), flush=True)
        full = m.model_
        head = [ln for ln in full.splitlines()
                if not ln.split() or ln.split()[0] not in ("split", "leaf")]
        print("PROBE ours model-head:\n" + "\n".join(head[:12]), flush=True)
        for k in list(range(100, len(counts) + 1, 100)):
            m._release_resident()
            m.model_ = _truncate(full, k)
            p = np.asarray(m.predict_proba(xte))[:, 1]
            print("PROBE ours trees=%4d auc=%.6f logloss=%.6f" % (
                k, spec.auc(yte, p), spec.logloss(yte, p)), flush=True)
        m._release_resident()
        m.model_ = full
    else:
        import catboost
        p = spec.catboost_tree_params(dict(cfg, n_estimators=n_trees), "CPU")
        p["boost_from_average"] = False
        if spw is not None:
            p["scale_pos_weight"] = spw
        p.update(extra)
        m = catboost.CatBoostClassifier(loss_function="Logloss", **p)
        t0 = time.time()
        m.fit(xtr, ytr)
        print("PROBE cat fit_s=%.1f version=%s trees=%d" % (
            time.time() - t0, catboost.__version__, m.tree_count_), flush=True)
        allp = m.get_all_params()
        for k in ("random_strength", "score_function", "leaf_estimation_method",
                  "leaf_estimation_iterations", "leaf_estimation_backtracking",
                  "border_count", "feature_border_type", "l2_leaf_reg", "model_size_reg",
                  "boost_from_average", "scale_pos_weight", "class_weights",
                  "bootstrap_type", "sampling_frequency", "model_shrink_rate",
                  "fold_len_multiplier", "approx_on_full_history", "nan_mode",
                  "border_count", "penalties_coefficient", "rsm", "sparse_features_conflict_fraction",
                  "use_best_model", "od_type", "min_data_in_leaf", "max_leaves", "depth",
                  "learning_rate", "iterations", "boosting_type", "random_score_type"):
            if k in allp:
                print("PROBE cat param %s=%r" % (k, allp[k]), flush=True)
        for i, pr in enumerate(m.staged_predict_proba(xte, eval_period=100)):
            k = (i + 1) * 100
            q = pr[:, 1]
            print("PROBE cat trees=%4d auc=%.6f logloss=%.6f" % (
                k, spec.auc(yte, q), spec.logloss(yte, q)), flush=True)


if __name__ == "__main__":
    main()
