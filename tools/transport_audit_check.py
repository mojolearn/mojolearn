#!/usr/bin/env python3
"""Cloud-only transport audit: multi-GPU drivers with inputs above 1 MiB, one device against two.

On two MI300X a kernel can read a device-to-device copy's target before the
copy lands (bench/results/multi_gpu/2026-09-15/peer-copy-mi300x/). This asks
the drivers that the identity_break par lanes only reach at small shapes
(DBSCAN, the query and reference-sharded neighbors, HDBSCAN, SVM kernel rows,
coordinate descent, Gaussian process rows, the isolation forest, full PCA's
TSQR panels, the wide Gram outputs and the Cholesky trailing rows) the same
question at shapes whose device buffers pass 1 MiB: every output of the
driver on devices (0,) must equal, byte for byte, its output on (0, 1) in the
same process. It prints one line per case and exits 1 if any case differs.
MOJOLEARN_TRANSPORT_AUDIT_CASES=a,b runs a subset, and
MOJOLEARN_TRANSPORT_AUDIT_SABOTAGE=1 flips one bit of each case's first
two-device output so every case must read DIFFERS.

    RUNPOD_POD_ID=... MOJOLEARN_NUMERIC_MODE=identical python3 tools/transport_audit_check.py
"""
import os
import sys

import numpy as np


def _data(n, d, seed):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((n, d)).astype(np.float32))


def _labels(X):
    return np.ascontiguousarray((X[:, 0] + X[:, 1] > 0).astype(np.int32))


def _mib(*arrays):
    return sum(np.asarray(a).nbytes for a in arrays) / 1048576.0


def case_dbscan(ml, dev):
    from mojolearn.parallel_classical import fit_dbscan
    X = _data(40000, 8, 1)
    return _mib(X), [fit_dbscan(ml.DBSCAN(eps=0.9, min_samples=5), X, devices=dev).labels_]


def case_svm(ml, dev):
    from mojolearn.parallel_classical import fit_svm, predict_svm
    X = _data(6000, 16, 2)
    y = _labels(X)
    kw = dict(C=1.0, kernel="rbf", max_iter=200)
    m = fit_svm(ml.SVC(**kw), X[:4096], y[:4096], devices=dev)
    q = np.ascontiguousarray(X[4096:6000])
    return _mib(X[:4096]), [predict_svm(m, q, devices=dev, method="decision_function"),
                            predict_svm(m, q, devices=dev, method="predict")]


def case_queries(ml, dev):
    from mojolearn.parallel_neighbors import ParallelQueries
    X = _data(80000, 16, 3)
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:64000], _labels(X[:64000]))
    q = np.ascontiguousarray(X[64000:])
    with ParallelQueries(m, devices=dev, rows_per_shard=4096) as pq:
        d, i = pq.query(q, method="kneighbors")
    return _mib(X[:64000]), [d, i]


def case_reference(ml, dev):
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    X = _data(80000, 16, 4)
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:64000], _labels(X[:64000]))
    q = np.ascontiguousarray(X[64000:68096])
    with ReferenceShardedNeighbors(m, devices=dev, reference_rows_per_shard=16000,
                                   query_rows_per_shard=1024) as rs:
        d, i = rs.kneighbors(q)
    return _mib(X[:64000]), [d, i]


def case_hdbscan(ml, dev):
    from mojolearn.parallel_classical import fit_hdbscan
    X = _data(12000, 4, 5)
    m = fit_hdbscan(ml.HDBSCAN(min_cluster_size=5), X, devices=dev)
    return _mib(X), [m.labels_, m.core_distances_]


def case_cd(ml, dev):
    from mojolearn.parallel_classical import fit_coordinate_descent
    X = _data(80000, 16, 6)
    y = np.ascontiguousarray((X @ np.arange(1, 17, dtype=np.float32) / 16).astype(np.float32))
    m = fit_coordinate_descent(ml.Lasso(alpha=0.01, max_iter=200), X, y, devices=dev)
    return _mib(X), [m.coef_, np.asarray(m.intercept_)]


def case_gp(ml, dev):
    from mojolearn.parallel_classical import fit_gaussian_process, predict_gaussian_process
    X = _data(1400, 4, 7)
    y = np.ascontiguousarray(np.sin(X[:, 0]).astype(np.float32))
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    m = fit_gaussian_process(ml.GaussianProcessRegressor(kernel=k), X[:1024], y[:1024], devices=dev)
    mean, std = predict_gaussian_process(m, X[1024:], devices=dev, return_std=True)
    return 1024 * 1024 * 4 / 1048576.0, [m.alpha_, m.L_, mean, std]


def case_iforest(ml, dev):
    from mojolearn.parallel_ensemble import fit_isolation_forest, score_isolation_forest
    X = _data(160000, 16, 8)
    m = fit_isolation_forest(ml.IsolationForest(n_estimators=16, random_state=5), X, devices=dev)
    return _mib(X), [score_isolation_forest(m, X, devices=dev, method="score_samples")]


def case_pca_full(ml, dev):
    from mojolearn.parallel_classical import fit_gram_estimator
    X = _data(40000, 33, 9)
    m = fit_gram_estimator(ml.PCA(n_components=3, svd_solver="full", numeric_mode="identical"), X, devices=dev)
    return _mib(X), [m.components_, m.singular_values_, m.transform(X[:256])]


def case_gram_wide(ml, dev):
    from mojolearn.parallel_classical import fit_gram_estimator
    X = _data(8000, 257, 10)
    m = fit_gram_estimator(ml.PCA(n_components=3, numeric_mode="identical"), X, devices=dev)
    return _mib(X), [m.components_, m.explained_variance_]


def case_cholesky(ml, dev):
    from mojolearn.parallel_classical import fit_cholesky, solve_cholesky
    rng = np.random.default_rng(11)
    n = 4500
    M = rng.standard_normal((n, 64)).astype(np.float64)
    A = np.ascontiguousarray((M @ M.T / 64 + np.eye(n)).astype(np.float32))
    A = np.ascontiguousarray(np.tril(A) + np.tril(A, -1).T)
    B = np.ascontiguousarray(rng.standard_normal((n, 3)).astype(np.float32))
    c = fit_cholesky(ml.Cholesky(), A, devices=dev)
    return _mib(A), [c.L_, np.int64(c.info_), solve_cholesky(c, B, devices=dev)]


CASES = [("dbscan", case_dbscan), ("svm", case_svm), ("queries-knn", case_queries),
         ("reference-knn", case_reference), ("hdbscan", case_hdbscan), ("cd", case_cd),
         ("gp", case_gp), ("iforest", case_iforest), ("pca-full", case_pca_full),
         ("gram-wide", case_gram_wide), ("cholesky", case_cholesky)]


def main():
    if not os.environ.get("RUNPOD_POD_ID"):
        raise SystemExit("RunPod required; no local execution")
    import mojolearn as ml
    only = set(filter(None, os.environ.get("MOJOLEARN_TRANSPORT_AUDIT_CASES", "").split(",")))
    failures = 0
    for name, fn in CASES:
        if only and name not in only:
            continue
        try:
            mib, one = fn(ml, (0,))
            _, two = fn(ml, (0, 1))
        except Exception as e:  # a refusal or a crash is reported, not hidden
            print(f"TRANSPORT {name} ERROR {type(e).__name__}: {e}", flush=True)
            failures += 1
            continue
        diffs = []
        for k, (a, b) in enumerate(zip(one, two)):
            ba, bb = np.asarray(a).tobytes(), np.asarray(b).tobytes()
            if os.environ.get("MOJOLEARN_TRANSPORT_AUDIT_SABOTAGE") == "1" and k == 0 and bb:
                # Check-only: flip one bit of the two-device output so the comparison is seen to fail.
                bb = bytes([bb[0] ^ 1]) + bb[1:]
            if ba != bb:
                diffs.append(f"output {k}: {sum(x != y for x, y in zip(ba, bb)) + abs(len(ba) - len(bb))} of {max(len(ba), len(bb))} bytes")
        verdict = "EQUAL" if not diffs else "DIFFERS " + "; ".join(diffs)
        print(f"TRANSPORT {name} input {mib:.2f} MiB outputs {len(one)} {verdict}", flush=True)
        failures += bool(diffs)
    print(f"TRANSPORT cases that differ or failed: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
