#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AutoARIMA board-row diagnostic (lane w2-ts): where the time goes, and
whether taxi-hourly's forecast RMSE gap is an order-selection effect.

Unscored, one run per variant, FAST Metal only, no opponent. Reads the
board's own ts block (<data>/ts-<dataset>.npz, last 48 hours held out) and
runs the board's call (search s=1, d 0..1, p/q 0..3, aicc, then fit and
forecast(48)), timing select_d, the grouped order search, the refit and
the forecast separately (perf_counter around the bindings). Then reports
the refit's chosen-order groups (size, n_iter max/median, retcodes) and
the held-out RMSE, overall, per d and per chosen order.

--search-maxiter N adds a second variant whose SEARCH runs N iterations
(the default search stops at 20, so its log-likelihoods, and therefore the
AICc order choice, come from unconverged fits): if that variant's RMSE
drops toward the opponent, the gap is order selection, not the refit.

Usage (M3, branch tree):
  MOJOLEARN_NUMERIC_MODE=fast ~/board-0834/cache/venv/bin/python \\
      tools/arima_phase_diag.py taxi-hourly [--search-maxiter 200]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

H = 48


def _data_root():
    for b in ("board-0834", "board-0833"):
        p = Path.home() / b / "cache" / "algos-data" / "rows-full"
        if p.is_dir():
            return p
    raise SystemExit("REFUSING: no ~/board-083x/cache/algos-data/rows-full")


def run(Y, hold, search_maxiter):
    import numpy as np
    import mojolearn as ml
    from mojolearn import _x_sequence_autoarima as aa
    from mojolearn import _arima_impl as ai

    T = Counter()
    orig_select_d = aa.select_d

    def timed_select_d(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_select_d(*a, **k)
        finally:
            T["select_d"] += time.perf_counter() - t0

    aa.select_d = timed_select_d
    orig_fit = ai.ARIMA.fit

    def timed_fit(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig_fit(self, *a, **k)
        finally:
            T["arima_fit_calls_s"] += time.perf_counter() - t0
            T["arima_fit_calls_n"] += 1

    ai.ARIMA.fit = timed_fit
    try:
        t0 = time.perf_counter()
        m = ml.AutoARIMA(Y)
        m.search(s=1, d=range(0, 2), p=range(0, 4), q=range(0, 4), P=range(1), D=range(1), Q=range(1),
                 ic="aicc", fit_intercept="auto", maxiter=search_maxiter)
        t1 = time.perf_counter()
        fit_calls_search = T["arima_fit_calls_n"]
        m.fit()
        t2 = time.perf_counter()
        fc = np.asarray(m.forecast(H), dtype=np.float64)
        t3 = time.perf_counter()
    finally:
        aa.select_d = orig_select_d
        ai.ARIMA.fit = orig_fit
    err = fc - hold.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    groups = []
    for (order, sorder, k), ids, f in zip(m.models, m._ids, m._fitted):
        it = np.asarray(f.n_iter_)
        rc = np.asarray(f.retcode_)
        groups.append(dict(order=list(order), k=int(k), n=len(ids), n_iter_max=int(it.max()),
                           n_iter_med=float(np.median(it)), retcodes=dict(Counter(int(v) for v in rc)),
                           rmse=float(np.sqrt(np.mean(err[np.asarray(ids)] ** 2)))))
    d = np.asarray(m.d_)
    per_d = {int(v): dict(n=int((d == v).sum()), rmse=float(np.sqrt(np.mean(err[d == v] ** 2))))
             for v in np.unique(d)}
    worst = np.argsort(-np.sqrt(np.mean(err ** 2, axis=1)))[:5]
    return dict(search_maxiter=search_maxiter, ms=dict(
        total=1e3 * (t3 - t0), search=1e3 * (t1 - t0), select_d=1e3 * T["select_d"],
        search_after_select_d=1e3 * (t1 - t0 - T["select_d"]), refit=1e3 * (t2 - t1),
        forecast=1e3 * (t3 - t2)), per_order_fit_calls_in_search=int(fit_calls_search),
        per_order_fit_calls_in_refit=int(T["arima_fit_calls_n"] - fit_calls_search),
        rmse=rmse, per_d=per_d, n_groups=len(groups), groups=groups,
        worst_series=[dict(i=int(i), order=[int(v) for v in m.order_[i]],
                           rmse=float(np.sqrt(np.mean(err[i] ** 2)))) for i in worst])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("taxi-hourly", "synthetic"))
    ap.add_argument("--search-maxiter", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if os.environ.get("MOJOLEARN_NUMERIC_MODE") != "fast":
        raise SystemExit("REFUSING: set MOJOLEARN_NUMERIC_MODE=fast")
    import numpy as np
    import mojolearn as ml
    b = ml.ARIMA()._extension()
    if str(b.arima_vendor()) != "metal" or int(b.arima_numeric_mode()) != 0:
        raise SystemExit("REFUSING: needs the FAST Metal arima binding")
    with np.load(_data_root() / ("ts-%s.npz" % a.dataset)) as z:
        Yall = np.ascontiguousarray(z["Y"], dtype=np.float32)
    Y, hold = np.ascontiguousarray(Yall[:, :-H]), Yall[:, -H:]
    flags = dict(fit_groups=bool(getattr(b, "arima_fit_groups_enabled", lambda: False)()),
                 order_batch=bool(getattr(b, "arima_order_batch_enabled", lambda: False)()))
    run(Y[:2], hold[:2], 20)  # warm-up: kernel/pipeline creation, not reported
    out = dict(dataset=a.dataset, shape=list(Y.shape), binding=flags, variants=[run(Y, hold, 20)])
    if a.search_maxiter:
        out["variants"].append(run(Y, hold, a.search_maxiter))
    for v in out["variants"]:
        print("ARIMA_DIAG ds=%s search_maxiter=%d total_ms=%.0f select_d_ms=%.0f search_ms=%.0f refit_ms=%.0f "
              "forecast_ms=%.0f groups=%d rmse=%.4f per_d=%s" % (
                  a.dataset, v["search_maxiter"], v["ms"]["total"], v["ms"]["select_d"],
                  v["ms"]["search_after_select_d"], v["ms"]["refit"], v["ms"]["forecast"], v["n_groups"],
                  v["rmse"], json.dumps(v["per_d"], sort_keys=True)))
    path = Path(a.out or (Path.home() / "mq" / "out" / ("arima-diag-%s.json" % a.dataset)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print("ARIMA_DIAG_JSON " + str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
