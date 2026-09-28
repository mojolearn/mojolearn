# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bench/decomp_eigh_width.py -- the x_decomp eigh (jacobi_eigh_kernel) at
launch widths 256 / 512 / 1024 (MOJOLEARN_XD_EIGH_WIDTH), one timed call
each after a warm one, with a sha256 of (w, V): every width must print the
same hash (the kernel is launch-width invariant under IDENTICAL). The
256-wide (old) launch is timed only up to OLD_MAX rows."""
import os, sys, time, hashlib
sys.path.insert(0, os.environ.get("ML_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")))
import numpy as np
from mojolearn._expansion_decomp import _M, _Kit
from mojolearn import _backend

k = _Kit(_backend.default_mode())
OLD_MAX = int(os.environ.get("OLD_MAX", "800"))
rng = np.random.default_rng(7)
for n in [int(x) for x in os.environ.get("NS", "128,400,800,1500").split(",")]:
    B = rng.standard_normal((n, n)).astype(np.float32)
    S = ((B + B.T) * 0.5).astype(np.float32)
    A = _M.of(S.ravel().tolist(), n, n)
    for wdt in ("256", "512", "1024"):
        if wdt == "256" and n > OLD_MAX:
            continue
        os.environ["MOJOLEARN_XD_EIGH_WIDTH"] = wdt
        try:
            if n <= 400:
                k.eigh(A)
            t = time.perf_counter(); w, V = k.eigh(A); dt = time.perf_counter() - t
            h = hashlib.sha256(np.asarray(w.s, np.float32).tobytes() + np.asarray(V.s, np.float32).tobytes()).hexdigest()[:16]
            print(f"eigh n={n:5d} width {wdt:>4s} {dt:9.3f} s  hash {h}", flush=True)
        except Exception as e:
            print(f"eigh n={n:5d} width {wdt:>4s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
os.environ.pop("MOJOLEARN_XD_EIGH_WIDTH", None)
