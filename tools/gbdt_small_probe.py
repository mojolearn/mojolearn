# gbdt_small_probe.py <data dir>: GradientBoosting on small pools (<= 200k cells, the old host-route sizes), GPU default; warm + 3 timed; prediction digest.
import sys, time, hashlib, numpy as np
from mojolearn.ensemble import GradientBoosting
D = sys.argv[1]
CASES = [("reg", "taxi", 2000, "RMSE", None), ("reg", "taxi", 18000, "RMSE", None), ("reg", "taxi", 18000, "RMSE", "Plain"),
         ("cls", "taxi", 18000, "Logloss", None), ("reg", "istella", 900, "RMSE", None), ("cls", "istella", 900, "Logloss", None)]
for blk, ds, n, loss, bt in CASES:
    z = np.load(f"{D}/{blk}-{ds}.npz"); X, y, Xq = z["X"][:n], z["y"][:n], z["Xq"][:5000]
    kw = dict(loss=loss, n_estimators=200, random_state=7)
    if bt: kw["boosting_type"] = bt
    ts = []
    for i in range(4):
        m = GradientBoosting(**kw); t = time.perf_counter(); m.fit(X, y); ts.append((time.perf_counter() - t) * 1e3)
    p = np.ascontiguousarray(np.asarray(m.predict(Xq), dtype=np.float32))
    print(f"GBDTSMALL {blk}-{ds} n={n} loss={loss} bt={bt or 'auto'} median_ms={np.median(ts[1:]):.1f} warm_ms={ts[0]:.1f} digest={hashlib.sha256(p.tobytes()).hexdigest()[:16]}", flush=True)
