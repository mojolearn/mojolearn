#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# tshap_smoke.sh: import smoke for TreeExplainer in FAST mode (lane fix-treeshap).
# Fits a small RandomForestRegressor, explains 100 rows, prints the max
# additivity error |sum(phi) + expected_value - predict|. Run in a built tree.
set -u
cd "$(dirname "$0")/../python"
PY=${TSHAP_PY:-$HOME/board-0834/cache/venv/bin/python}
MOJOLEARN_NUMERIC_MODE=fast "$PY" - <<'PYEOF'
import numpy as np
import mojolearn as ml
from mojolearn._expansion_trees import TreeExplainer
r = np.random.default_rng(0)
X = r.random((2000, 8), dtype=np.float32)
y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
m = ml.RandomForestRegressor(n_estimators=10, max_depth=6, random_state=0).fit(X, y)
e = TreeExplainer(m, X[::20])
p = np.asarray(e.shap_values(X[:100]))
pr = np.asarray(m.predict(X[:100]))
err = float(np.abs(p.sum(1) + e.expected_value - pr).max())
print("TSHAP-SMOKE status=%s additivity=%.3g" % ("ok" if err < 1e-3 else "FAIL", err))
PYEOF
