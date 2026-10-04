#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Where mojolearn.resample.resample's time goes at the board shape (lane
apple-fast-w2-clres). One process, main's code, the installed FAST binding:
one warm call, then ONE timed call of each piece, in this order:

  draw   resample_indices(n)               device draw + 4 MB download + Array
  intp   numpy.asarray(idx, dtype=intp)    the int32 -> intp copy
  gX     X[ix]                             the row gather of X (as _take)
  gy     y[ix]
  full   resample(X, y, **board params)    the whole call the board times
  mean   Xr.astype(float64).mean(0)        the board's quality fold inside its timed fit

usage: w2_resample_probe.py taxi|istella   (MOJOLEARN_NUMERIC_MODE=fast)
Prints one W2RS-PROBE line. Diagnostic only, not an A/B and not a board cell.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np


def main(ds):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("w2rs_bba", here / "bench_board_algos.py")
    bba = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bba)
    data = next(str(p) for p in (Path.home() / b / "cache/algos-data/rows-full"
                                 for b in ("board-0834", "board-0833")) if p.is_dir())
    B, _ = bba._load_block("resample", ds, data)
    D = bba.lane_arrays("resample", B)
    X = D["X"]
    y = np.ascontiguousarray(D["y"], dtype=np.float32)
    from mojolearn import resample as R
    kw = dict(bba.LANES["resample"]["params"])
    R.resample(X, y, **kw)                      # warm: binding load, pipeline compile
    t = {}

    def tick(name, fn):
        s = time.perf_counter()
        v = fn()
        t[name] = (time.perf_counter() - s) * 1e3
        return v

    n = X.shape[0]
    idx = tick("draw", lambda: R.resample_indices(n, None, True, kw["random_state"]))
    ix = tick("intp", lambda: np.asarray(idx, dtype=np.intp))
    tick("gX", lambda: X[ix])
    tick("gy", lambda: y[ix])
    Xr, _ = tick("full", lambda: R.resample(X, y, **kw))
    tick("mean", lambda: Xr.astype(np.float64).mean(0))
    print("W2RS-PROBE ds=%s X=%sx%s %s %s" % (ds, X.shape[0], X.shape[1] if X.ndim > 1 else 1, X.dtype,
          " ".join("%s=%.1f" % kv for kv in t.items())))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
