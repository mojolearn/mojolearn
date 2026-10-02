"""Time `IsolationForest.score_samples` on a board dataset (lane
apple-fast-trees-io, the IF_QUERY_RAW arm). Every scoring call rebuilds the
forest (DEVIATION 874), so the clock holds one fit plus one query upload and
score; the fit is the same in both arms of tools/aft_if_score_ab.sh, so the
A/B difference is the query path. Prints one AFT-IFSCORE line.

    python tools/aft_if_score.py <taxi|istella> <label>
Env: AFT_IFSCORE_ROUNDS (default 3), AFT_IFSCORE_QUERY=train|test (default
train: the largest query the board has), GBM_BENCH_DATA as the board."""
import hashlib
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec  # noqa: E402
import mojolearn as ml  # noqa: E402

ds = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "?"
rounds = int(os.environ.get("AFT_IFSCORE_ROUNDS", "3"))
which = os.environ.get("AFT_IFSCORE_QUERY", "train")
data = spec.load_with_fallback(ds, "shipped", None)
cfg = spec.lane_config("iforest", "shipped")
x = np.ascontiguousarray(data.X_train, dtype=np.float32)
q = np.ascontiguousarray(data.X_test if which == "test" else data.X_train, dtype=np.float32)
m = ml.IsolationForest(
    n_estimators=cfg["n_estimators"], max_samples=cfg["max_samples"],
    max_features=cfg["max_features"], bootstrap=cfg["bootstrap"],
    contamination="auto", random_state=cfg["seed"],
).fit(x)
m.score_samples(q[:1])  # warm-up: the first call compiles and loads
ms, h = [], None
for _ in range(rounds):
    t0 = time.perf_counter()
    s = np.asarray(m.score_samples(q), dtype=np.float32)
    ms.append((time.perf_counter() - t0) * 1000.0)
    h = hashlib.sha256(s.tobytes()).hexdigest()[:16]
print("AFT-IFSCORE arm=%s ds=%s query=%s n_query=%d n_cols=%d rounds=%d median_ms=%.1f all=%s hash=%s"
      % (label, ds, which, q.shape[0], q.shape[1], rounds, float(np.median(ms)),
         [round(v) for v in ms], h), flush=True)
