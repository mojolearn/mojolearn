#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""VAR board-row overhead split (lane w2-ts), unscored, FAST Metal only.

The board's VAR fit is ~5 ms for one OLS over a (1392, 16) lagged design,
whose device work (8 queued launches, one wait) should be about 1 ms. This
splits one board-shaped `VAR(endog).fit(maxlags=2, trend='c')` into:
the `var_fit` binding call itself, and the Python around it (binding
lookup, VARResults construction), plus `forecast(48)`. Five warm calls,
each printed (milliseconds), so the manager can see which side the
remaining milliseconds live on before any kernel is changed.

Usage (M3, branch tree):
  MOJOLEARN_NUMERIC_MODE=fast ~/board-0834/cache/venv/bin/python \\
      tools/var_overhead_diag.py taxi-hourly
"""
import argparse
import os
import sys
import time
from pathlib import Path

H = 48


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("taxi-hourly", "synthetic"))
    a = ap.parse_args()
    if os.environ.get("MOJOLEARN_NUMERIC_MODE") != "fast":
        raise SystemExit("REFUSING: set MOJOLEARN_NUMERIC_MODE=fast")
    import numpy as np
    import mojolearn as ml
    from mojolearn import _backend
    from mojolearn import _x_sequence_var as xv
    root = None
    for b in ("board-0834", "board-0833"):
        p = Path.home() / b / "cache" / "algos-data" / "rows-full"
        if p.is_dir():
            root = p
            break
    if root is None:
        raise SystemExit("REFUSING: no ~/board-083x/cache/algos-data/rows-full")
    with np.load(root / ("ts-%s.npz" % a.dataset)) as z:
        Y = np.ascontiguousarray(z["Y"], dtype=np.float32)[:, :-H]
    endog = np.ascontiguousarray(Y[:16].T)
    T = {}
    real_binding = _backend.binding

    class Timed:
        def __init__(self, mod):
            self._m = mod

        def __getattr__(self, name):
            fn = getattr(self._m, name)
            if name not in ("var_fit", "var_forecast"):
                return fn

            def call(*args):
                t0 = time.perf_counter()
                try:
                    return fn(*args)
                finally:
                    T[name] = T.get(name, 0.0) + time.perf_counter() - t0
            return call

    def binding(name, mode=None):
        t0 = time.perf_counter()
        m = real_binding(name, mode)
        T["lookup"] = T.get("lookup", 0.0) + time.perf_counter() - t0
        return Timed(m)

    xv._backend.binding = binding
    try:
        for rep in range(6):
            T.clear()
            t0 = time.perf_counter()
            est = ml.VAR(endog).fit(maxlags=2, method="ols", ic=None, trend="c")
            t1 = time.perf_counter()
            fc = est.forecast(np.ascontiguousarray(endog[-est.k_ar:]), H)
            t2 = time.perf_counter()
            if rep == 0:
                continue  # warm-up
            fit_ms, bind_ms = 1e3 * (t1 - t0), 1e3 * T.get("var_fit", 0.0)
            print("VAR_DIAG ds=%s rep=%d fit_ms=%.3f var_fit_binding_ms=%.3f python_ms=%.3f lookup_ms=%.3f "
                  "forecast_ms=%.3f var_forecast_binding_ms=%.3f fc_finite=%s" % (
                      a.dataset, rep, fit_ms, bind_ms, fit_ms - bind_ms, 1e3 * T.get("lookup", 0.0),
                      1e3 * (t2 - t1), 1e3 * T.get("var_forecast", 0.0), bool(np.isfinite(fc).all())))
    finally:
        xv._backend.binding = real_binding
    return 0


if __name__ == "__main__":
    sys.exit(main())
