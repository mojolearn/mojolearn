# Lane linear-cluster-istella (2026-09-11): pick DBSCAN's eps by a STATED RULE
# rather than by taste, and show what that eps does before any race runs.
#
#   python3 dbscan_eps.py <dataset> <data dir> [min_samples]
#
# THE RULE. eps is a quantile of the min_samples-th nearest neighbor distance
# measured ON THE RACE BLOCK ITSELF (all its rows as the index, a stride sample
# of 20,000 of them as the queries, cuML brute force). The 50th percentile is
# the candidate; the 25th and 75th are printed beside it, and each candidate is
# run through cuML DBSCAN on a 200,000-row stride subsample so the cluster
# count, the noise share and the fit seconds are known before the full race.
# The eps that goes into MOJOLEARN_CTD_DBSCAN_<DATASET> is the one whose
# subsample run is not degenerate (not one cluster holding everything, not
# everything noise), and the same number is used by every arm on that dataset.
import json
import os
import sys
import time

import numpy as np

ds, data = sys.argv[1], sys.argv[2]
min_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10

with np.load(os.path.join(data, "dbscan-%s.npz" % ds)) as z:
    X = np.ascontiguousarray(z["X"], dtype=np.float32)
print("block %s rows=%d cols=%d" % (ds, X.shape[0], X.shape[1]), flush=True)

import cupy as cp  # noqa: E402
from cuml.cluster import DBSCAN  # noqa: E402
from cuml.neighbors import NearestNeighbors  # noqa: E402

xd = cp.asarray(X)
qn = min(20_000, X.shape[0])
qi = (np.arange(qn, dtype=np.int64) * X.shape[0]) // qn
q = xd[cp.asarray(qi)]
nn = NearestNeighbors(n_neighbors=min_samples, algorithm="brute", metric="euclidean",
                      output_type="cupy")
nn.fit(xd)
t0 = time.perf_counter()
dist, _ = nn.kneighbors(q)
cp.cuda.runtime.deviceSynchronize()
kth = cp.asnumpy(dist[:, -1]).astype(np.float64)
print("kth_nn(k=%d) seconds=%.1f min=%.4f p25=%.4f p50=%.4f p75=%.4f max=%.4f"
      % (min_samples, time.perf_counter() - t0, kth.min(), np.percentile(kth, 25),
         np.percentile(kth, 50), np.percentile(kth, 75), kth.max()), flush=True)

sn = min(200_000, X.shape[0])
si = (np.arange(sn, dtype=np.int64) * X.shape[0]) // sn
xs = xd[cp.asarray(si)]
out = {"dataset": ds, "rows": int(X.shape[0]), "cols": int(X.shape[1]),
       "min_samples": min_samples, "rule": "quantile of the k-th NN distance on the block",
       "candidates": []}
for name in ("p25", "p50", "p75"):
    eps = float(np.percentile(kth, {"p25": 25, "p50": 50, "p75": 75}[name]))
    eps = float("%.3g" % eps)
    t0 = time.perf_counter()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", algorithm="brute",
                calc_core_sample_indices=False, output_type="cupy")
    db.fit(xs)
    cp.cuda.runtime.deviceSynchronize()
    secs = time.perf_counter() - t0
    lab = cp.asnumpy(db.labels_).astype(np.int64)
    row = {"quantile": name, "eps": eps, "subsample_rows": int(sn),
           "n_clusters": int(np.unique(lab[lab >= 0]).shape[0]),
           "noise_fraction": float((lab < 0).mean()),
           "largest_cluster_fraction": float(np.bincount(lab[lab >= 0]).max() / lab.shape[0])
           if (lab >= 0).any() else 0.0,
           "subsample_fit_seconds": secs}
    out["candidates"].append(row)
    print("candidate %s eps=%g clusters=%d noise=%.3f largest=%.3f subsample_fit_s=%.1f"
          % (name, eps, row["n_clusters"], row["noise_fraction"],
             row["largest_cluster_fraction"], secs), flush=True)
print("JSON " + json.dumps(out, sort_keys=True), flush=True)
