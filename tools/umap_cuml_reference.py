#!/usr/bin/env python3
"""cuML UMAP fit_transform at the UMAP phase-price shapes; one JSON out.

The opponent's FAST arm for `bench/umap_phase_price_main.mojo`: same dyadic
fixture (salt 0), n_neighbors, n_components=2, n_epochs, spectral init.
One warmup then `--rounds` timed fits, host numpy in and out. cuML's UMAP is
approximate by default (NN-descent above its brute-force threshold, GPU
stochastic optimizer), so only the time is comparable, not the embedding.

Only the main lane runs this, on the rented box.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from knn_cuml_reference import coordinate_block, nvidia_smi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[20000, 100000])
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cupy as cp
    import cuml
    from cuml.manifold import UMAP

    name, driver = nvidia_smi()
    env = {"gpu": name, "driver": driver, "cuml": cuml.__version__, "cupy": cp.__version__,
           "numpy": np.__version__, "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
           "arm": "cuml-UMAP-fit_transform-fast",
           "note": "cuML default build_algo (brute force below its threshold, NN-descent above), GPU optimizer; no deterministic configuration"}
    results = []
    for n in args.rows:
        x = np.ascontiguousarray(coordinate_block(n, args.features, 0))
        times = []
        for r in range(args.rounds + 1):
            model = UMAP(n_neighbors=args.neighbors, n_components=2, n_epochs=args.epochs,
                         init="spectral", random_state=0, output_type="numpy")
            t0 = time.perf_counter()
            emb = model.fit_transform(x)
            cp.cuda.runtime.deviceSynchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            if r > 0:
                times.append(ms)
            if not np.all(np.isfinite(emb)):
                raise SystemExit("cuML UMAP produced a non-finite embedding")
        rec = {"rows": n, "features": args.features, "neighbors": args.neighbors,
               "epochs": args.epochs, "rounds": args.rounds, "fit_transform_ms": times,
               "median_ms": statistics.median(times)}
        results.append(rec)
        print("CUML_UMAP rows=%d median_ms=%.1f samples=%s" % (n, rec["median_ms"], ["%.1f" % t for t in times]), flush=True)
    with open(args.out, "w") as fh:
        json.dump({"environment": env, "results": results}, fh, indent=1)
    print("CUML UMAP REFERENCE DONE", args.out)


if __name__ == "__main__":
    main()
