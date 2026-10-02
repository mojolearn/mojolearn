"""GPU-path stage profile of Isomap and LocallyLinearEmbedding at the board's
10,000-row shape (lane hr2-graph-embed). Never run with MOJOLEARN_VENDOR=cpu.
Prints the wall time of each fit and the top cumulative Python frames."""
import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

assert os.environ.get("MOJOLEARN_VENDOR", "") != "cpu", "GPU path only"
import mojolearn as ml

rng = np.random.default_rng(7)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
X = rng.standard_normal((n, 16)).astype(np.float32)
X[n // 2:n // 2 + 50] = X[:50]          # duplicate rows, as taxi has
for name, est in (("isomap", ml.Isomap(n_neighbors=10, n_components=2)),
                  ("lle", ml.LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=7))):
    est.fit(X[:2000])                    # warm the bindings
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    est.fit(X)
    pr.disable()
    dt = time.perf_counter() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
    print("STAGES %s n=%d fit_s=%.3f" % (name, n, dt))
    for line in s.getvalue().splitlines():
        if "_expansion_decomp" in line or "x_decomp" in line or "{method" in line:
            print("  " + line.strip()[:160])
