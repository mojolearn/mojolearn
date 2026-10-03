"""Prophet FAST fit diagnostic (lane apple-fast-prophetfix).

Loads the board's prophet data exactly as tools/bench_board_algos.py does
(_load_block + lane_arrays, the lane's params, ds hourly from 2024-01-01),
fits mojolearn.ProphetForecaster once, and prints grep-sized lines:
PROPHET-DIAG per a few series and one summary per dataset.

    MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python python tools/prophet_diag.py [--data DIR] [--datasets a,b]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_board_algos as bba  # noqa: E402


def _default_data():
    for b in ("board-0834", "board-0833"):
        d = os.path.join(os.path.expanduser("~"), b, "cache", "algos-data", "rows-full")
        if os.path.isdir(d):
            return d
    return None


def _fmt(v, n=5):
    return "[" + ",".join("%.6g" % x for x in np.asarray(v).ravel()[:n]) + "]"


def run(ds_name, data):
    B, _rec = bba._load_block("prophet", ds_name, data)
    D = bba.lane_arrays("prophet", B)
    Y32 = D["Yfit"]
    Yh = D["Yhold"].astype(np.float64)
    h = bba.TS_H
    p = bba.LANES["prophet"]["params"]
    _name, cls = bba._ours_class("prophet")
    ds = np.datetime64("2024-01-01T00", "h") + np.arange(Y32.shape[1] + h).astype("timedelta64[h]")
    m = cls(**p).fit(ds[:Y32.shape[1]], Y32)
    info = np.asarray(m.info_, dtype=np.float64)
    prm = np.asarray(m.params_, dtype=np.float64)
    S = len(m.changepoints_t_)
    it = info[:, 2]
    obj = info[:, 1]
    fc = np.asarray(m.predict(ds[Y32.shape[1]:]), dtype=np.float64)
    rmse = float(np.sqrt(np.mean((fc - Yh) ** 2)))
    print("PROPHET-DIAG ds=%s B=%d N=%d P=%d S=%d K=%d mode=%s" % (
        ds_name, Y32.shape[0], Y32.shape[1], prm.shape[1], S, prm.shape[1] - 3 - S,
        os.environ.get("MOJOLEARN_NUMERIC_MODE", "")), flush=True)
    for b in sorted(set([0, 1, 2, Y32.shape[0] // 2, Y32.shape[0] - 1])):
        r = prm[b]
        print("PROPHET-DIAG ds=%s series=%d y_scale=%.6g objective=%.6g n_iter=%d k=%.6g m=%.6g "
              "delta=%s log_sigma=%.6g beta=%s" % (
                  ds_name, b, info[b, 0], obj[b], int(it[b]), r[0], r[1], _fmt(r[2:2 + S]),
                  r[2 + S], _fmt(r[3 + S:])), flush=True)
    print("PROPHET-DIAG-SUMMARY ds=%s n_iter min/median/max=%d/%d/%d objective first5=%s "
          "nan_objective=%d nan_params=%d inf_params=%d zero_beta_series=%d forecast_rmse=%.6g" % (
              ds_name, int(it.min()), int(np.median(it)), int(it.max()), _fmt(obj),
              int(np.isnan(obj).sum()), int(np.isnan(prm).sum()), int(np.isinf(prm).sum()),
              int(np.all(prm[:, 3 + S:] == 0.0, axis=1).sum()), rmse), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_default_data())
    ap.add_argument("--datasets", default="synthetic,taxi-hourly")
    a = ap.parse_args()
    if not a.data:
        print("PROPHET-DIAG error=no board data dir", flush=True)
        return 2
    rc = 0
    for d in a.datasets.split(","):
        try:
            run(d, a.data)
        except Exception as exc:  # noqa: BLE001
            print("PROPHET-DIAG ds=%s error=%s: %s" % (d, type(exc).__name__, exc), flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
