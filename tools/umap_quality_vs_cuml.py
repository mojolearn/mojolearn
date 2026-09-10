#!/usr/bin/env python3
"""neighborhood_quality(k) of OUR IDENTICAL UMAP embedding beside cuML's, same bytes.

Reads the embedding `bench/umap_phase_price_main.mojo` dumped
(MOJOLEARN_UMAP_DUMP, little-endian Float32, rows x 2) for the dyadic
fixture (salt 0), scores it with `tools/nvidia_public_compare.py`'s
`neighborhood_quality` (trustworthiness and k-neighbour retention, exact
ranks, so a few thousand rows at most), fits cuML UMAP on the same rows with
the same n_neighbors / n_epochs / spectral init, and scores that. One JSON.

Only the main lane runs this, on the rented box, inside the cuML venv.
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from knn_cuml_reference import coordinate_block, nvidia_smi  # noqa: E402
from nvidia_public_compare import neighborhood_quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ours-dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    x = np.ascontiguousarray(coordinate_block(args.rows, args.features, 0))
    ours = np.fromfile(args.ours_dump, dtype="<f4")
    if ours.size != args.rows * 2:
        raise SystemExit("ours dump has %d values, expected %d" % (ours.size, args.rows * 2))
    ours = ours.reshape(args.rows, 2)
    q_ours = neighborhood_quality(x, ours, args.k)
    print("QUALITY ours trustworthiness=%.6f retention=%.6f k=%d rows=%d" % (
        q_ours["trustworthiness"], q_ours["neighbor_retention"], args.k, args.rows), flush=True)

    import cupy as cp
    import cuml
    from cuml.manifold import UMAP

    name, driver = nvidia_smi()
    model = UMAP(n_neighbors=args.neighbors, n_components=2, n_epochs=args.epochs,
                 init="spectral", random_state=0, output_type="numpy")
    t0 = time.perf_counter()
    emb = model.fit_transform(x)
    cp.cuda.runtime.deviceSynchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    q_cuml = neighborhood_quality(x, np.asarray(emb, dtype=np.float32), args.k)
    print("QUALITY cuml trustworthiness=%.6f retention=%.6f k=%d rows=%d fit_ms=%.1f" % (
        q_cuml["trustworthiness"], q_cuml["neighbor_retention"], args.k, args.rows, ms), flush=True)
    with open(args.out, "w") as fh:
        json.dump({
            "environment": {"gpu": name, "driver": driver, "cuml": cuml.__version__,
                            "cupy": cp.__version__, "numpy": np.__version__},
            "rows": args.rows, "features": args.features, "neighbors": args.neighbors,
            "epochs": args.epochs, "k": args.k,
            "ours_identical": q_ours, "cuml_fast": q_cuml, "cuml_fit_ms_one_round": ms,
        }, fh, indent=1)
    print("UMAP QUALITY VS CUML DONE", args.out)


if __name__ == "__main__":
    main()
