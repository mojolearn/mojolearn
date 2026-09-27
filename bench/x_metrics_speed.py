# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's GPU speed board (phase C, lane `metrics`).

Times every x_metrics entry family at a realistic shape on the two R2
datasets (taxi: regression, multiclass and clustering; HIGGS: binary
ranking), under the numeric mode of the environment (MOJOLEARN_NUMERIC_MODE).
Each case runs once to load, then REPS timed runs; the minimum is reported.
Every case also prints a digest of its result, so a before and an after on
IDENTICAL show the same bits by eye (the lane check proves it by column).

    python bench/x_metrics_speed.py [--rows 1000000] [--reps 3] [--only name,...]

Data: GBM_BENCH_DATA (default ~/datasets/gbm-bench), staged from R2 with
`tools/dataset_store.sh stage` (taxi/taxi_speed.npz, higgs/higgs_speed.npz).
Lines: `XMSPEED <case> <seconds> <digest>`.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _root():
    return os.environ.get("GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench"))


def _digest(v):
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
            return
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                walk(x[k])
            return
        if isinstance(x, str):
            h.update(x.encode())
            return
        try:
            a = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)
        except Exception:
            h.update(repr(x).encode())
            return
        if a.dtype == object:
            h.update(repr(a.tolist()).encode())
        else:
            h.update(np.ascontiguousarray(a).tobytes())
    walk(v)
    return h.hexdigest()[:16]


def data(n):
    taxi = np.load(os.path.join(_root(), "taxi", "taxi_speed.npz"))
    higgs = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))
    tx = np.asarray(taxi["x"][:n], dtype=np.float32)
    fare = np.asarray(taxi["fare"][:n], dtype=np.float32)
    # a fixed least-squares predictor of the fare from the taxi features
    A = np.concatenate([tx[:, :8].astype(np.float64), np.ones((n, 1))], axis=1)
    coef = np.linalg.lstsq(A, fare.astype(np.float64), rcond=None)[0]
    pred = (A @ coef).astype(np.float32)
    yr = np.abs(fare) + np.float32(0.5)
    pr = np.abs(pred) + np.float32(0.5)
    # multiclass: fare quintiles, true vs predicted
    edges = np.quantile(fare, [0.2, 0.4, 0.6, 0.8])
    ct = np.searchsorted(edges, fare).astype(np.int64)
    cp = np.searchsorted(edges, pred).astype(np.int64)
    # a 5-class score matrix: softmax of minus the distance to each bin centre
    centres = np.quantile(fare, [0.1, 0.3, 0.5, 0.7, 0.9]).astype(np.float32)
    logits = -np.abs(pred[:, None] - centres[None, :])
    logits = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(logits)
    proba = (proba / proba.sum(axis=1, keepdims=True)).astype(np.float32)
    hx = np.asarray(higgs["x"][:n], dtype=np.float32)
    hy = np.asarray(higgs["y"][:n]).astype(np.int64)
    # a fixed binary score: a logistic fit on the HIGGS high-level features
    z = hx[:, 21:28].astype(np.float64)
    z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-12)
    w = np.linalg.lstsq(np.concatenate([z, np.ones((n, 1))], axis=1), hy * 2.0 - 1.0, rcond=None)[0]
    s = 1.0 / (1.0 + np.exp(-(np.concatenate([z, np.ones((n, 1))], axis=1) @ w) * 2.0))
    hs = s.astype(np.float32)
    sw = (np.abs(tx[:, 0]) % 3 + 1).astype(np.float32)
    return dict(yr=yr, pr=pr, fare=fare, pred=pred, ct=ct, cp=cp, proba=proba, hy=hy, hs=hs, sw=sw,
                X=tx[:, :4].copy())


def cases(d):
    import mojolearn.metrics as M
    import mojolearn.model_selection as S
    n = len(d["yr"])
    y2 = np.stack([d["yr"], d["yr"] * 2], axis=1)
    p2 = np.stack([d["pr"], d["pr"] * 2], axis=1)
    return [
        ("median_absolute_error", lambda: M.median_absolute_error(d["yr"], d["pr"])),
        ("median_absolute_error_w", lambda: M.median_absolute_error(d["yr"], d["pr"], sample_weight=d["sw"])),
        ("mean_pinball_loss", lambda: M.mean_pinball_loss(d["yr"], d["pr"], alpha=0.3)),
        ("explained_variance_score", lambda: M.explained_variance_score(d["fare"], d["pred"])),
        ("mean_tweedie_deviance", lambda: M.mean_tweedie_deviance(d["yr"], d["pr"], power=1.5)),
        ("max_error", lambda: M.max_error(d["fare"], d["pred"])),
        ("d2_absolute_error_score", lambda: M.d2_absolute_error_score(d["fare"], d["pred"])),
        ("r2_score_w_mo", lambda: M.r2_score(y2, p2, sample_weight=d["sw"], multioutput="raw_values")),
        ("mean_squared_log_error", lambda: M.mean_squared_log_error(d["yr"], d["pr"])),
        ("balanced_accuracy_score", lambda: M.balanced_accuracy_score(d["ct"], d["cp"])),
        ("matthews_corrcoef", lambda: M.matthews_corrcoef(d["ct"], d["cp"])),
        ("cohen_kappa_score", lambda: M.cohen_kappa_score(d["ct"], d["cp"], weights="quadratic")),
        ("prfs_w", lambda: M.precision_recall_fscore_support(d["ct"], d["cp"], average=None, sample_weight=d["sw"])),
        ("hamming_loss", lambda: M.hamming_loss(d["ct"], d["cp"])),
        ("roc_curve", lambda: M.roc_curve(d["hy"], d["hs"])),
        ("roc_curve_w", lambda: M.roc_curve(d["hy"], d["hs"], sample_weight=d["sw"])),
        ("average_precision_score", lambda: M.average_precision_score(d["hy"], d["hs"])),
        ("roc_auc_max_fpr", lambda: M.roc_auc_score(d["hy"], d["hs"], max_fpr=0.5)),
        ("roc_auc_ovr", lambda: M.roc_auc_score(d["ct"], d["proba"], multi_class="ovr")),
        ("log_loss_w", lambda: M.log_loss(d["ct"], d["proba"], sample_weight=d["sw"])),
        ("brier_score_loss", lambda: M.brier_score_loss(d["hy"], d["hs"])),
        ("top_k_accuracy_score", lambda: M.top_k_accuracy_score(d["ct"], d["proba"], k=2)),
        ("adjusted_mutual_info_score", lambda: M.adjusted_mutual_info_score(d["ct"], d["cp"])),
        ("calinski_harabasz_score", lambda: M.calinski_harabasz_score(d["X"], d["ct"])),
        ("davies_bouldin_score", lambda: M.davies_bouldin_score(d["X"], d["ct"])),
        ("kfold_shuffle", lambda: [t[1][:5].tolist() if hasattr(t[1], "tolist") else list(t[1])[:5]
                                   for t in S.KFold(5, shuffle=True, random_state=0).split(np.empty((n, 1)))]),
        ("stratified_kfold_shuffle", lambda: [len(t[1]) for t in
                                              S.StratifiedKFold(5, shuffle=True, random_state=0).split(np.empty((n, 1)), d["ct"])]),
        ("shuffle_split", lambda: [len(t[1]) for t in S.ShuffleSplit(5, test_size=0.2, random_state=0).split(np.empty((n, 1)))]),
        ("train_test_split", lambda: train_test(S, d)),
    ]


def train_test(S, d):
    a, b = S.train_test_split(d["yr"], test_size=0.25, random_state=0)
    return [np.asarray(a)[:10], np.asarray(b)[:10]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    d = data(a.rows)
    print("XMSPEED-DATA rows=%d load_s=%.2f mode=%s" % (a.rows, time.time() - t0,
                                                         os.environ.get("MOJOLEARN_NUMERIC_MODE", "default")), flush=True)
    only = set(x for x in a.only.split(",") if x)
    total = 0.0
    for name, fn in cases(d):
        if only and name not in only:
            continue
        try:
            v = fn()
            best = float("inf")
            for _ in range(a.reps):
                t = time.perf_counter()
                v = fn()
                best = min(best, time.perf_counter() - t)
            total += best
            print("XMSPEED %-28s %9.4f %s" % (name, best, _digest(v)), flush=True)
        except Exception as e:  # a case that fails is reported, never hidden
            print("XMSPEED %-28s   FAILED %s: %s" % (name, type(e).__name__, str(e)[:160]), flush=True)
    print("XMSPEED-TOTAL %.4f" % total, flush=True)


if __name__ == "__main__":
    main()
