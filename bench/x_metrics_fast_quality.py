# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's FAST quality check (phase C): every case of the speed
board, at SEEDS seeded 200k-row samples of each of the two R2 datasets,
against scikit-learn in float64 on the same float32 inputs. Writes one JSON
of relative errors; `--compare old.json new.json` is the paired verdict:
a case is WORSE when its new error exceeds max(old error * 1.01, 1e-7).

    MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python:/root/skl \\
        python bench/x_metrics_fast_quality.py --out new.json
    python bench/x_metrics_fast_quality.py --compare old.json new.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _flat(v):
    if isinstance(v, tuple):
        v = v[:3]
        return np.concatenate([np.ravel(np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, dtype=np.float64))
                               for x in v])
    return np.ravel(np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float64))


def _rel(a, b):
    a, b = _flat(a), _flat(b)
    if a.shape != b.shape:
        return float("inf")
    d = np.abs(a - b)
    return float(np.max(d / np.maximum(np.abs(b), 1e-12))) if d.size else 0.0


def cases(d):
    import mojolearn.metrics as M
    import sklearn.metrics as K
    y2 = np.stack([d["yr"], d["yr"] * 2], axis=1)
    p2 = np.stack([d["pr"], d["pr"] * 2], axis=1)
    f64 = lambda x: np.asarray(x, dtype=np.float64)
    return [
        ("median_absolute_error_w", lambda m: m.median_absolute_error(d["yr"], d["pr"], sample_weight=d["sw"]), K.median_absolute_error),
        ("mean_pinball_loss", lambda m: m.mean_pinball_loss(d["yr"], d["pr"], alpha=0.3), None),
        ("explained_variance_score", lambda m: m.explained_variance_score(d["fare"], d["pred"]), None),
        ("mean_tweedie_deviance", lambda m: m.mean_tweedie_deviance(d["yr"], d["pr"], power=1.5), None),
        ("d2_absolute_error_score", lambda m: m.d2_absolute_error_score(d["fare"], d["pred"]), None),
        ("r2_score_w_mo", lambda m: m.r2_score(y2, p2, sample_weight=d["sw"], multioutput="raw_values"), None),
        ("mean_squared_log_error", lambda m: m.mean_squared_log_error(d["yr"], d["pr"]), None),
        ("balanced_accuracy_score", lambda m: m.balanced_accuracy_score(d["ct"], d["cp"]), None),
        ("matthews_corrcoef", lambda m: m.matthews_corrcoef(d["ct"], d["cp"]), None),
        ("cohen_kappa_score", lambda m: m.cohen_kappa_score(d["ct"], d["cp"], weights="quadratic"), None),
        ("prfs_w", lambda m: m.precision_recall_fscore_support(d["ct"], d["cp"], average=None, sample_weight=d["sw"])[:3], None),
        ("roc_auc_max_fpr", lambda m: m.roc_auc_score(d["hy"], d["hs"], max_fpr=0.5), None),
        ("roc_auc_w", lambda m: m.roc_auc_score(d["hy"], d["hs"], sample_weight=d["sw"]), None),
        ("average_precision_score", lambda m: m.average_precision_score(d["hy"], d["hs"]), None),
        ("roc_auc_ovr", lambda m: m.roc_auc_score(d["ct"], d["proba"], multi_class="ovr"), None),
        ("log_loss_w", lambda m: m.log_loss(d["ct"], d["proba"], sample_weight=d["sw"]), None),
        ("brier_score_loss", lambda m: m.brier_score_loss(d["hy"], d["hs"]), None),
        ("top_k_accuracy_score", lambda m: m.top_k_accuracy_score(d["ct"], d["proba"], k=2), None),
        ("adjusted_mutual_info_score", lambda m: m.adjusted_mutual_info_score(d["ct"], d["cp"]), None),
        ("calinski_harabasz_score", lambda m: m.calinski_harabasz_score(d["X"], d["ct"]), None),
        ("davies_bouldin_score", lambda m: m.davies_bouldin_score(d["X"], d["ct"]), None),
    ], M, K, f64


def sample(full, seed, rows):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(full["yr"]), size=rows, replace=False))
    out = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(full["yr"]) else v) for k, v in full.items()}
    return out


def run(args):
    import x_metrics_speed as B
    raw = B._load(args.pool)
    res = {}
    # two datasets: primary taxi (taxi regression / multiclass / clustering,
    # HIGGS binary) and primary higgs (the other way round); each seed
    # re-samples the rows
    for primary in ("taxi", "higgs"):
      full = B.data(args.pool, primary, raw)
      for seed in range(args.seeds):
        d = sample(full, 1000 + seed, args.rows)
        table, M, K, f64 = cases(d)
        for name, ours, _ in table:
            theirs = getattr(K, name) if hasattr(K, name) else None
            fn = {
                "median_absolute_error_w": lambda: K.median_absolute_error(f64(d["yr"]), f64(d["pr"]), sample_weight=f64(d["sw"])),
                "r2_score_w_mo": lambda: K.r2_score(f64(np.stack([d["yr"], d["yr"] * 2], 1)), f64(np.stack([d["pr"], d["pr"] * 2], 1)), sample_weight=f64(d["sw"]), multioutput="raw_values"),
                "prfs_w": lambda: K.precision_recall_fscore_support(d["ct"], d["cp"], average=None, sample_weight=f64(d["sw"]))[:3],
                "roc_auc_max_fpr": lambda: K.roc_auc_score(d["hy"], f64(d["hs"]), max_fpr=0.5),
                "roc_auc_w": lambda: K.roc_auc_score(d["hy"], f64(d["hs"]), sample_weight=f64(d["sw"])),
                "roc_auc_ovr": lambda: K.roc_auc_score(d["ct"], f64(d["proba"]), multi_class="ovr"),
                "log_loss_w": lambda: K.log_loss(d["ct"], f64(d["proba"]), sample_weight=f64(d["sw"])),
                "cohen_kappa_score": lambda: K.cohen_kappa_score(d["ct"], d["cp"], weights="quadratic"),
                "mean_pinball_loss": lambda: K.mean_pinball_loss(f64(d["yr"]), f64(d["pr"]), alpha=0.3),
                "mean_tweedie_deviance": lambda: K.mean_tweedie_deviance(f64(d["yr"]), f64(d["pr"]), power=1.5),
                "top_k_accuracy_score": lambda: K.top_k_accuracy_score(d["ct"], f64(d["proba"]), k=2),
                "explained_variance_score": lambda: K.explained_variance_score(f64(d["fare"]), f64(d["pred"])),
                "d2_absolute_error_score": lambda: K.d2_absolute_error_score(f64(d["fare"]), f64(d["pred"])),
                "mean_squared_log_error": lambda: K.mean_squared_log_error(f64(d["yr"]), f64(d["pr"])),
                "brier_score_loss": lambda: K.brier_score_loss(d["hy"], f64(d["hs"])),
                "average_precision_score": lambda: K.average_precision_score(d["hy"], f64(d["hs"])),
                "calinski_harabasz_score": lambda: K.calinski_harabasz_score(f64(d["X"]), d["ct"]),
                "davies_bouldin_score": lambda: K.davies_bouldin_score(f64(d["X"]), d["ct"]),
            }.get(name, (lambda: theirs(d["ct"], d["cp"])) if theirs else None)
            err = _rel(ours(M), fn())
            res["%s/%s/seed%d" % (name, primary, seed)] = err
            print("XMQ %-28s %-5s seed %d rel_err %.3e" % (name, primary, seed, err), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)


def compare(old_path, new_path):
    old = json.load(open(old_path))
    new = json.load(open(new_path))
    worse = []
    for k in sorted(new):
        o, n = old.get(k), new[k]
        if o is None or not (n <= max(o * 1.01, 1e-7)):
            worse.append((k, o, n))
    for k, o, n in worse:
        print("WORSE %s old %s new %s" % (k, o, n))
    print("XMQ-VERDICT %s: %d cases, %d worse, max new rel err %.3e" %
          ("PASS" if not worse else "FAIL", len(new), len(worse), max(new.values())))
    return 1 if worse else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--pool", type=int, default=2_000_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="x_metrics_fast_quality.json")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    run(a)


if __name__ == "__main__":
    main()
