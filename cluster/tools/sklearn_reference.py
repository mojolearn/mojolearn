#!/usr/bin/env python3
"""Write a k-means fixture and the answer scikit-learn gives for it.

The oracle is scikit-learn, never our own previous output. A check against
what we produced last time is a ratchet: it locks in whatever was wrong the
first time and reports green forever.

WHY SCIKIT-LEARN AND NOT CUVS
-----------------------------
cuVS is the thing being ported and would be the better oracle, but its GPU
arms do not run on Apple silicon, which is the entire premise of this work.
scikit-learn runs everywhere and implements the same algorithm.

WHAT TO COMPARE, AND WHAT NOT TO
--------------------------------
Compare INERTIA. Do not compare centroid arrays or label arrays.

1. Labels are only defined up to a permutation of the clusters.
2. Empty clusters are handled DIFFERENTLY on purpose. cuVS keeps the old
   centroid (`finalize_centroids`); scikit-learn relocates it onto the point
   furthest from its centroid. Both are correct, and from the same seed they
   diverge permanently.
3. scikit-learn's default is `n_init=10`; cuVS's is `n_init=1`. Comparing
   both at their defaults compares ten restarts against one, which measures
   the restart count and nothing else. This script pins both to the same
   value and says which.

So the test is: over several seeds, is our inertia within tolerance of
scikit-learn's, and is it never systematically worse. A single seed proves
nothing about either implementation.
"""

import argparse
import json
import pathlib

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("cluster/reference"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    x, _, _ = make_blobs(
        n_samples=args.rows,
        n_features=args.features,
        centers=args.clusters,
        random_state=0,
        return_centers=True,
    )
    x = np.ascontiguousarray(x, dtype=np.float32)
    x.tofile(args.out / "x.f32")

    results = []
    for seed in range(args.seeds):
        # n_init=1 to match cuVS's default. Stated, not silent.
        km = KMeans(
            n_clusters=args.clusters,
            init="k-means++",
            n_init=1,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=seed,
        ).fit(x)
        results.append(
            {
                "seed": seed,
                "inertia": float(km.inertia_),
                "n_iter": int(km.n_iter_),
                "empty_clusters": int(
                    args.clusters - len(np.unique(km.labels_))
                ),
            }
        )

    inertias = [r["inertia"] for r in results]
    manifest = {
        "rows": args.rows,
        "features": args.features,
        "clusters": args.clusters,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "n_init": 1,
        "note": (
            "n_init pinned to 1 on BOTH sides. sklearn's own default is 10 "
            "and cuVS's is 1; comparing defaults compares restart counts."
        ),
        "compare": "inertia only; not labels, not centroids",
        "sklearn_runs": results,
        "inertia_min": min(inertias),
        "inertia_max": max(inertias),
        "inertia_spread_rel": (max(inertias) - min(inertias)) / min(inertias),
    }
    (args.out / "kmeans_reference.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"wrote {args.out}/x.f32 and kmeans_reference.json")
    print(
        "inertia over {} seeds: {:.6g} .. {:.6g}  (spread {:.2%})".format(
            args.seeds,
            manifest["inertia_min"],
            manifest["inertia_max"],
            manifest["inertia_spread_rel"],
        )
    )
    print(
        "READ THIS SPREAD FIRST. It is how much the SAME implementation "
        "moves on seed alone, and any tolerance we hold ourselves to has to "
        "be wider than it or the check is measuring luck."
    )


if __name__ == "__main__":
    main()
