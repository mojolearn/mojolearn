#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# miv_smoke.sh: import smoke, FAST mode, for the lane/apple-fast-miv defaults
# (MI_REG_TIES, MI_CLF_RANKMAJOR): SelectKBest with mutual_info_regression /
# mutual_info_classif on a tie-heavy fixture. MIV-SMOKE status=ok|FAIL. Times nothing.
set -u
cd "$(dirname "$0")/../python"
PY=${MIV_PY:-$HOME/board-0834/cache/venv/bin/python}
PYTHONPATH=$PWD MOJOLEARN_NUMERIC_MODE=fast "$PY" - <<'PYEOF'
import traceback
import numpy as np
from mojolearn._expansion_prep import SelectKBest, mutual_info_classif, mutual_info_regression

r = np.random.default_rng(0)
n, d = 4000, 12
X = np.empty((n, d), dtype=np.float32)
X[:, ::2] = r.integers(0, 4, (n, d // 2))
X[:, 1::2] = r.standard_normal((n, d // 2))
y = (X[:, 0] + X[:, 1] + 0.3 * r.standard_normal(n)).astype(np.float32)
yc = (y > np.median(y)).astype(np.int32)
bad = []
for name, fn, t in (("mi-reg", mutual_info_regression, y), ("mi-clf", mutual_info_classif, yc)):
    try:
        s = np.asarray(fn(X, t, random_state=0), dtype=np.float64)
        sel = SelectKBest(fn, k=4).fit(X, t)
        top = set(np.argsort(-s)[:2].tolist())
        ok = np.isfinite(s).all() and top == {0, 1} and np.asarray(sel.get_support()).sum() == 4
        print("MIV %s %s top2=%s" % (name, "ok" if ok else "FAIL", sorted(top)))
    except Exception:
        traceback.print_exc(); ok = False; print("MIV %s FAIL raised" % name)
    if not ok:
        bad.append(name)
print("MIV-SMOKE status=%s" % ("FAIL " + " ".join(bad) if bad else "ok"))
PYEOF
