#!/usr/bin/env python3
"""Time and hash a production CPU GaussianMixture fit.

Build ``bindings/build_mixture_host.sh`` into a fresh directory and pass that
directory here.  The default case is the million-row qualification workload.
Run baseline and candidate in interleaved processes; ``/usr/bin/time -l`` adds
peak RSS on macOS.
"""
import argparse
import hashlib
import os
import statistics
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("host_dir")
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--features", type=int, default=8)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3)
    a = p.parse_args()
    os.environ["MOJOLEARN_HOST_DIR"] = a.host_dir

    import mojolearn
    from mojolearn._cpu_reference import reference_training

    i = np.arange(a.rows, dtype=np.float32)
    x = np.empty((a.rows, a.features), dtype=np.float32)
    x[:, 0] = (i % 997) / 997
    for j in range(1, a.features):
        x[:, j] = np.sin(i * np.float32(0.0001 * (j + 1))) + np.float32(j * 0.01)

    elapsed, hashes = [], []
    state = None
    for _ in range(a.rounds):
        start = time.perf_counter()
        with reference_training():
            model = mojolearn.GaussianMixture(
                n_components=a.components, max_iter=2,
                init_params="random", random_state=17,
            ).fit(x)
        elapsed.append(time.perf_counter() - start)
        state = b"".join(np.asarray(v).tobytes() for v in (
            model.weights_, model.means_, model.covariances_,
            model.precisions_cholesky_, model.log_det_chol_,
        ))
        hashes.append(hashlib.sha256(state).hexdigest())
    assert len(set(hashes)) == 1
    print({"rows": a.rows, "features": a.features,
           "components": a.components, "seconds": elapsed,
           "median_seconds": statistics.median(elapsed),
           "sha256": hashes[0], "n_iter": model.n_iter_,
           "converged": model.converged_,
           "lower_bound_bits": np.float32(model.lower_bound_).view(np.uint32).item()})


if __name__ == "__main__":
    main()
