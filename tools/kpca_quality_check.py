#!/usr/bin/env python3
"""Quality of our KernelPCA (FAST) against scikit-learn on the board's kernel-pca rows.

Loads the istella (or --dataset) block exactly as tools/bench_board_algos.py
does for the kernel-pca lane (same stride rows, n_components=8, kernel='rbf',
gamma=1/d, random_state=7), fits mojolearn.KernelPCA and
sklearn.decomposition.KernelPCA on the same X, transforms the same Xq and
prints one line:

  KPCA-Q eig_rel_err=<max |ours-sk|/sk, sorted eigenvalues>
         subspace_cos=<mean |cos| of the principal angles between transform(Xq)s>
         n=<fit rows>

Run with PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=fast for the FAST arm.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import bench_board_algos as bba  # noqa: E402

LANE = "kernel-pca"


def _data_dir():
    for b in ("board-0834", "board-0833"):
        d = os.path.join(os.path.expanduser("~"), b, "cache", "algos-data", "rows-full")
        if os.path.isdir(d):
            return d
    raise SystemExit("KPCA-Q error=no algos-data/rows-full under ~/board-0834 or ~/board-0833")


def _np_arr(a):
    if hasattr(a, "to_numpy"):
        a = a.to_numpy()
    try:
        return np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray(a.to_list() if hasattr(a, "to_list") else a.tolist(), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="istella")
    ap.add_argument("--data", default=None)
    ap.add_argument("--rows", type=int, default=20_000, help="cap on fit and query rows")
    a = ap.parse_args()

    B, _rec = bba._load_block(LANE, a.dataset, a.data or _data_dir())
    D = bba.lane_arrays(LANE, B)
    X = np.ascontiguousarray(D["X"][: a.rows])
    Xq = np.ascontiguousarray(D["Xq"][: a.rows])
    D = dict(D, X=X, Xq=Xq)

    mk_o, _, po = bba._est_factory(LANE, "ours-fast", D)
    mk_s, _, ps = bba._est_factory(LANE, "sklearn-cpu", D)
    ours = mk_o()
    ours.fit(X)
    To = _np_arr(ours.transform(Xq))
    eo = np.sort(_np_arr(ours.eigenvalues_).reshape(-1))[::-1]
    sk = mk_s()
    sk.fit(X)
    Ts = np.asarray(sk.transform(Xq), dtype=np.float64)
    es = np.sort(np.asarray(sk.eigenvalues_, dtype=np.float64).reshape(-1))[::-1]

    k = min(eo.size, es.size)
    eig = float(np.max(np.abs(eo[:k] - es[:k]) / np.maximum(np.abs(es[:k]), 1e-30))) if k else float("nan")
    cos = bba._subspace(np.nan_to_num(To), np.nan_to_num(Ts))   # sign/rotation invariant
    print("KPCA-Q eig_rel_err=%.3e subspace_cos=%.6f n=%d nq=%d d=%d k_ours=%d k_sk=%d gamma=%.6g "
          "mode=%s nan_ours=%d" % (eig, cos, X.shape[0], Xq.shape[0], X.shape[1], eo.size, es.size,
                                   po.get("gamma", float("nan")),
                                   os.environ.get("MOJOLEARN_NUMERIC_MODE", "default"),
                                   int(np.isnan(To).sum())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
