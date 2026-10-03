#!/usr/bin/env python3
"""Same-bits check of the x_cluster post-processing the board's ID lines do
not reach (cgr2-cluster, 2026-10-03): AgglomerativeClustering with a
connectivity matrix (connected, and with several components joined), OPTICS
dbscan extraction and a precomputed matrix, MeanShift without bin seeding
(the bandwidth estimate, every row a seed) and with cluster_all=False,
AffinityPropagation on a precomputed S with positive entries, the
BayesianGaussianMixture 'random', 'k-means++' and 'random_from_data' starts,
and weighted k-means++ (MiniBatchKMeans with sample weights).

Each case fits a fixed seeded fixture twice, in separate subprocesses: once
with MOJOLEARN_VENDOR unset (the device column) and once with
MOJOLEARN_VENDOR=cpu (the host column), digests the fitted attributes'
words, and prints

    IDCHECK <case> device=<digest> host=<digest> MATCH|DIFFER

The device digest is comparable across boxes (NVIDIA = AMD = Apple = host).
Usage: python tools/xcluster_post_idcheck.py [case ...]
"""
import hashlib
import json
import os
import subprocess
import sys

CASES = ["agglo-conn-ward", "agglo-conn-average", "agglo-conn-complete", "agglo-conn-single",
         "agglo-split-average", "agglo-split-ward", "optics-dbscan", "optics-precomputed",
         "meanshift-nobin", "meanshift-noclusterall", "ap-precomputed-pos",
         "bgmm-random", "bgmm-kpp", "bgmm-rfd", "minibatch-kpp-w"]


def _blobs(n, d=4, k=6, seed=11, spread=0.6):
    import numpy as np
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-6, 6, (k, d)).astype(np.float32)
    lab = rng.integers(0, k, n)
    X = (centers[lab] + spread * rng.standard_normal((n, d))).astype(np.float32)
    return X


def _dist(X):
    """Euclidean distances by column-ordered float64 adds (no BLAS)."""
    import numpy as np
    n, d = X.shape
    acc = np.zeros((n, n), dtype=np.float64)
    for j in range(d):
        c = X[:, j].astype(np.float64)
        acc += (c[:, None] - c[None, :]) ** 2
    return np.sqrt(acc).astype(np.float32)


def _band_graph(X, split=1):
    """Rows ordered by their first feature; each joined to the next 1, 2 and
    5 in that order, inside its part when `split` > 1 (several components)."""
    import numpy as np
    n = X.shape[0]
    order = np.argsort(X[:, 0], kind="stable")
    part = np.zeros(n, dtype=np.int64)
    part[order] = (np.arange(n) * split) // n
    A = np.zeros((n, n), dtype=np.int32)
    for s in (1, 2, 5):
        a = order[:-s]
        b = order[s:]
        keep = part[a] == part[b]
        A[a[keep], b[keep]] = 1
    return A


def _digest(est, names):
    import numpy as np
    h = hashlib.sha256()
    for nm in names:
        v = np.asarray(getattr(est, nm))
        h.update(v.astype(np.float32 if v.dtype.kind == "f" else np.int64).tobytes())
    return h.hexdigest()[:16]


def _run_case(case):
    import numpy as np
    import mojolearn as ml
    if case.startswith("agglo-"):
        X = _blobs(1500)
        A = _band_graph(X, split=3 if "split" in case else 1)
        e = ml.AgglomerativeClustering(n_clusters=8, linkage=case.rsplit("-", 1)[1], connectivity=A,
                                       compute_distances=True).fit(X)
        return _digest(e, ["children_", "labels_", "distances_"])
    if case == "optics-dbscan":
        X = _blobs(2000)
        e = ml.OPTICS(min_samples=10, cluster_method="dbscan", eps=0.8).fit(X)
        return _digest(e, ["ordering_", "reachability_", "core_distances_", "predecessor_", "labels_"])
    if case == "optics-precomputed":
        X = _blobs(1200)
        e = ml.OPTICS(min_samples=8, metric="precomputed").fit(_dist(X))
        return _digest(e, ["ordering_", "reachability_", "core_distances_", "predecessor_", "labels_"])
    if case == "meanshift-nobin":
        X = _blobs(800)
        e = ml.MeanShift().fit(X)
        return _digest(e, ["cluster_centers_", "labels_", "n_iter_"])
    if case == "meanshift-noclusterall":
        X = _blobs(3000, spread=1.5)
        e = ml.MeanShift(bandwidth=1.2, bin_seeding=True, cluster_all=False).fit(X)
        return _digest(e, ["cluster_centers_", "labels_", "n_iter_"])
    if case == "ap-precomputed-pos":
        X = _blobs(400)
        S = -_dist(X)
        S = (S + np.float32(0.5)).astype(np.float32)  # some positive entries
        e = ml.AffinityPropagation(affinity="precomputed", random_state=0).fit(S)
        return _digest(e, ["cluster_centers_indices_", "labels_", "n_iter_"])
    if case.startswith("bgmm-"):
        X = _blobs(4000)
        init = {"bgmm-random": "random", "bgmm-kpp": "k-means++", "bgmm-rfd": "random_from_data"}[case]
        e = ml.BayesianGaussianMixture(n_components=6, init_params=init, max_iter=60, random_state=3).fit(X)
        return _digest(e, ["weights_", "means_", "lower_bound_", "n_iter_"])
    if case == "minibatch-kpp-w":
        X = _blobs(20000)
        sw = np.random.default_rng(5).uniform(0.2, 3.0, X.shape[0]).astype(np.float32)
        e = ml.MiniBatchKMeans(n_clusters=8, init="k-means++", n_init=1, random_state=4).fit(X, sample_weight=sw)
        return _digest(e, ["cluster_centers_", "labels_", "inertia_"])
    raise ValueError(case)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        try:
            print(json.dumps({"digest": _run_case(sys.argv[2])}))
        except Exception as ex:  # reported as the digest, so a refusal shows
            print(json.dumps({"digest": "ERROR:" + type(ex).__name__ + ":" + str(ex)[:120]}))
        return 0
    cases = sys.argv[1:] or CASES
    bad = 0
    for case in cases:
        out = {}
        for col in ("device", "host"):
            env = dict(os.environ)
            env.pop("MOJOLEARN_VENDOR", None)
            if col == "host":
                env["MOJOLEARN_VENDOR"] = "cpu"
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", case],
                               env=env, capture_output=True, text=True, timeout=900)
            dig = "ERROR:rc=%d" % p.returncode
            for line in p.stdout.splitlines()[::-1]:
                if line.startswith("{"):
                    dig = json.loads(line)["digest"]
                    break
            if dig.startswith("ERROR"):
                sys.stderr.write(p.stderr[-2000:])
            out[col] = dig
        ok = out["device"] == out["host"] and not out["device"].startswith("ERROR")
        bad += 0 if ok else 1
        print(f"IDCHECK {case} device={out['device']} host={out['host']} {'MATCH' if ok else 'DIFFER'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
