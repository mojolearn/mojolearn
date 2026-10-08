# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every trees-family entry point called TWICE in one process, on the GPU
bindings and on the CPU host bindings, with the second call's bits equal to
the first's; the GPU and CPU column digests are recorded side by side.

The directive it answers (CURRENT DIRECTIVES, 2026-09-27): x_cluster and
x_neighbors hung on the SECOND GPU call in a process because each call built
a DeviceContext whose buffers outlived it. The lane checks call each entry
point once per process, so they cannot see that. This test covers the RF,
ExtraTrees, IsolationForest, GBDT (SymmetricTree Logloss and RMSE,
Depthwise, Lossguide, MultiRMSE, OrderedRMSE, the FeatureFreq estimator),
the host forest/GBDT readers (`host_predict` on a saved file) and every
xtrees ensemble, CART and SHAP entry, at the configurations their identity
lanes record (so the CPU column answers every one).

Each column runs in its own subprocess (the CPU one with MOJOLEARN_VENDOR=cpu
and the host bindings the lane check builds under python/mojolearn/host),
with a timeout, so a hang fails the test instead of stalling the run.
Runs on the lane's pod:
`python -m pytest python/mojolearn/tests/test_trees_repeat.py`."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

PKG = Path(__file__).resolve().parents[1]

_SCRIPT = r'''
import hashlib
import os
import tempfile
import numpy as np
import mojolearn as ml

rng = np.random.default_rng(0)
X = rng.standard_normal((240, 8)).astype(np.float32)
Xh = rng.standard_normal((40, 8)).astype(np.float32)
yc = ((X[:, 0] + X[:, 1] > 0).astype(np.int64) + (X[:, 2] > 0.5).astype(np.int64))
yr = (X[:, 0] * 2.0 + X[:, 3] - X[:, 4] * X[:, 5]).astype(np.float32)
Y3 = np.ascontiguousarray(np.stack([yr, X[:, 1], X[:, 2] - yr], axis=1).astype(np.float32))
C = np.ascontiguousarray(np.column_stack([
    (X[:, 4] > 0).astype(np.float32),
    (2 * (X[:, 5] > 0).astype(np.int32) + (X[:, 6] > 0).astype(np.int32)).astype(np.float32),
    X]).astype(np.float32))
Ch = np.ascontiguousarray(np.column_stack([
    (Xh[:, 4] > 0).astype(np.float32),
    (2 * (Xh[:, 5] > 0).astype(np.int32) + (Xh[:, 6] > 0).astype(np.int32)).astype(np.float32),
    Xh]).astype(np.float32))
perm = np.argsort(rng.random(X.shape[0]), kind="stable")


def gb(**kw):
    """The recorded GBDT configuration (identity_break._gbdt_legacy_kw)."""
    kw.setdefault("learning_rate", 0.03)
    kw.setdefault("random_strength", 0.0)
    kw.setdefault("bootstrap_type", "No")
    return ml.GradientBoosting(**kw)


def calls():
    out = []
    # the RF and ET device builders, and their predict
    for cls, y in ((ml.RandomForestClassifier, yc), (ml.ExtraTreesClassifier, yc)):
        m = cls(n_estimators=4, max_depth=6, random_state=7).fit(X, y)
        out += [m.predict(Xh), m.predict_proba(Xh)]
    for cls in (ml.RandomForestRegressor, ml.ExtraTreesRegressor):
        out += [cls(n_estimators=4, max_depth=6, random_state=7).fit(X, yr).predict(Xh)]
    iso = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    out += [iso.score_samples(Xh), iso.predict(Xh)]
    # GBDT: every fit arm the CPU host binding restates, and predict/multi
    s = gb(n_estimators=6, max_depth=4, loss="Logloss").fit(X, (yc > 0).astype(np.int64))
    out += [s.predict(Xh), s.predict_proba(Xh)]
    out += [gb(n_estimators=6, max_depth=4, loss="RMSE").fit(X, yr).predict(Xh)]
    out += [gb(n_estimators=6, max_depth=4, grow_policy="Depthwise", loss="Logloss")
            .fit(X, (yc > 0).astype(np.int64)).predict(Xh)]
    out += [gb(n_estimators=6, max_leaves=16, grow_policy="Lossguide", loss="Logloss")
            .fit(X, (yc > 0).astype(np.int64)).predict(Xh)]
    out += [gb(n_estimators=4, max_depth=4, loss="MultiRMSE").fit(X, Y3).predict(Xh)]
    out += [ml.OrderedRMSE(n_estimators=4, max_depth=4).fit(X, yr, permutation=perm).predict(Xh)]
    out += [ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(C, yr).predict(Ch)]
    # CTR categoricals (column 1 has four categories, above one_hot_max_size):
    # the CPU column trains every permutation's ordered Borders columns
    ctr = gb(n_estimators=4, max_depth=4, loss="Logloss", cat_features=[0, 1]).fit(C, (yc > 0).astype(np.int64))
    out += [ctr.predict(Ch), ctr.predict_proba(Ch)]
    # the host readers on saved files (host forest and host GBDT inference)
    with tempfile.TemporaryDirectory() as d:
        rf = ml.RandomForestClassifier(n_estimators=4, max_depth=6, random_state=7).fit(X, yc)
        p = os.path.join(d, "rf.npz"); rf.save(p)
        out += [ml.host_predict(p, Xh), ml.host_predict_proba(p, Xh)]
        g = os.path.join(d, "gb.npz"); s.save(g)
        out += [ml.host_predict(g, Xh)]
    # xtrees: CART, the ensembles, DART and SHAP
    dt = ml.DecisionTreeClassifier(max_depth=6, criterion="entropy", random_state=7).fit(X, yc)
    out += [dt.predict(Xh), dt.predict_proba(Xh)]
    out += [ml.DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=7).fit(X, yr).predict(Xh)]
    out += [ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=4), n_estimators=3, max_samples=0.8,
                                 max_features=0.75, random_state=7).fit(X, yc).predict_proba(Xh)]
    out += [ml.AdaBoostClassifier(ml.DecisionTreeClassifier(max_depth=2), n_estimators=3, learning_rate=0.8,
                                  random_state=7).fit(X, yc).predict_proba(Xh)]
    out += [ml.AdaBoostRegressor(ml.DecisionTreeRegressor(max_depth=3), n_estimators=3, loss="square",
                                 random_state=7).fit(X, yr).predict(Xh)]
    dart = ml.DARTRegressor(n_estimators=4, num_leaves=15, max_depth=5, min_child_samples=5, drop_rate=0.5,
                            skip_drop=0.0, reg_lambda=1.0, drop_seed=3, random_state=7).fit(X, yr)
    out += [dart.predict(Xh)]
    out += [ml.RandomTreesEmbedding(n_estimators=3, max_depth=3, random_state=7).fit(X).transform(Xh)]
    out += [ml.VotingRegressor([("a", ml.DecisionTreeRegressor(max_depth=3)),
                                ("b", ml.DecisionTreeRegressor(max_depth=5))]).fit(X, yr).predict(Xh)]
    out += [ml.OneVsRestClassifier(ml.DecisionTreeClassifier(max_depth=3)).fit(X, yc).predict_proba(Xh)]
    out += [ml.TreeExplainer(dart, data=X[:64]).shap_values(Xh[:8])]
    h = hashlib.sha256()
    for a in out:
        a = np.asarray(a.toarray() if hasattr(a, "toarray") else a)
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


first = calls()
second = calls()
print("FIRST", first)
print("SECOND", second)
assert first == second, "a second call in the same process moved the bits"
'''


def _run(env_extra):
    env = dict(os.environ, PYTHONPATH=str(PKG.parent), MOJOLEARN_NUMERIC_MODE="identical")
    env.update(env_extra)
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True,
                          text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return [l for l in proc.stdout.splitlines() if l.startswith("FIRST")][0].split()[1]


def test_every_entry_point_twice_on_gpu_and_cpu():
    # Each column asserts its own second call equals its first (_SCRIPT).
    # Identity is required across NVIDIA and AMD only (2026-10-07); a GPU
    # column and the host column are recorded side by side, never required to
    # agree, so the pair is printed rather than asserted.
    gpu = _run({})
    cpu = _run({"MOJOLEARN_VENDOR": "cpu", "MOJOLEARN_HOST_DIR": str(PKG / "host")})
    print("trees_repeat digests: gpu", gpu, "cpu", cpu, "agree" if gpu == cpu else "differ")
