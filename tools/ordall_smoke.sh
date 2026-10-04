#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ordall_smoke.sh: import smoke for the Apple FAST Ordered bundle (ORD_ALL,
# lane/apple-fast-ordall). Fits an Ordered GBDT on 2000x40 random data (wide:
# the bundle runs, 40 > 32 features) and on 2000x8 (narrow: main's path),
# prints auc on the training rows. Run in a built tree.
set -u
cd "$(dirname "$0")/.."
PY=${ORDALL_PY:-$HOME/board-0834/cache/venv/bin/python}
PYTHONPATH=$PWD/python MOJOLEARN_NUMERIC_MODE=${MOJOLEARN_NUMERIC_MODE:-fast} "$PY" - <<'PYEOF'
import numpy as np
import mojolearn as ml
from sklearn.metrics import roc_auc_score
r = np.random.default_rng(0)
bad = 0
for d in (40, 8):
    X = r.random((2000, d), dtype=np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.1 * r.random(2000)) > 0.8).astype(np.int32)
    m = ml.GradientBoostingClassifier(n_estimators=50, max_depth=6,
                                      boosting_type='Ordered', random_state=0).fit(X, y)
    p = np.asarray(m.predict_proba(X))[:, 1]
    auc = float(roc_auc_score(y, p))
    ok = np.isfinite(p).all() and auc > 0.9
    bad += not ok
    print("ORDALL-SMOKE d=%d auc=%.4f %s" % (d, auc, "ok" if ok else "FAIL"))
print("ORDALL-SMOKE status=%s" % ("ok" if bad == 0 else "FAIL"))
raise SystemExit(bad)
PYEOF
