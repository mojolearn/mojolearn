"""DOES THE HIGGS AUC GAP GROW WITH DEPTH? The falsifiable half of archive/reference/PORTING.md 140.

The config hypothesis died on 2026-08-31: matching CatBoost's regularization
moved CatBoost UP and widened the gap. The surviving suspect is item 140's
Logloss Newton leaf walk, where their CPU freezes at six accepted steps and
ours stalls later. Its blast radius is stated as "leaves still moving at step
six -- extreme leaves, which are exactly the ones a deep tree makes many of".

That is a PREDICTION and it can be falsified: if the leaf walk is the cause,
the gap must GROW WITH DEPTH. If the gap is flat in depth, item 140 is not
the driver and the search moves elsewhere.

Depth and tree count are varied SEPARATELY so a growth in one is not read as
the other. Both arms matched throughout (bootstrap_type='No',
random_strength=0, boosting_type='Plain' on their side, which is what ours
already runs), so this sweep is about mechanism, not configuration.
"""
import json, time, sys, os, hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mojolearn
import catboost as cb
print(f"mojolearn {mojolearn.__version__} {mojolearn.vendor()}, catboost {cb.__version__}", flush=True)

d = np.load("/Users/andrewhendel/datasets/gbm-bench/higgs/HIGGS.f32.npy", mmap_mode="r")
y = np.ascontiguousarray(d[:, 0], dtype=np.float32)
X = np.ascontiguousarray(d[:, 1:], dtype=np.float32)
del d
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=77, test_size=0.2)
del X, y
w = float(len(ytr)) / float((ytr == 1).sum())
print(f"train {Xtr.shape} test {Xte.shape} pos_weight {w:.4f}\n", flush=True)

MATCHED = {"bootstrap_type": "No", "random_strength": 0.0, "boosting_type": "Plain"}
rows = []
for n_est, depth in ((100, 6), (100, 8), (500, 6), (500, 8)):
    r = {"n_estimators": n_est, "max_depth": depth}
    t = time.time()
    m = mojolearn.GradientBoosting(n_estimators=n_est, max_depth=depth, learning_rate=0.1,
                                   l2_leaf_reg=1.0, border_count=254, loss="Logloss",
                                   class_weights=[1.0, w], random_state=0)
    m.fit(Xtr, ytr)
    r["ours_fit_s"] = time.time() - t
    p = m.predict_proba(Xte)[:, 1]
    r["ours_auc"] = float(roc_auc_score(yte, p))
    r["ours_sha"] = hashlib.sha256(np.ascontiguousarray(p, dtype=np.float32)).hexdigest()[:12]

    t = time.time()
    c = cb.CatBoost(dict(iterations=n_est, depth=depth, learning_rate=0.1, reg_lambda=1.0,
                         objective="Logloss", scale_pos_weight=w, random_seed=0,
                         task_type="CPU", verbose=0, **MATCHED))
    c.fit(Xtr, ytr)
    r["cat_fit_s"] = time.time() - t
    r["cat_auc"] = float(roc_auc_score(yte, c.predict(Xte)))
    r["delta"] = r["ours_auc"] - r["cat_auc"]
    rows.append(r)
    print(f"n={n_est:4d} depth={depth}  ours {r['ours_auc']:.8f} ({r['ours_fit_s']:6.1f}s)  "
          f"cat {r['cat_auc']:.8f} ({r['cat_fit_s']:6.1f}s)  DELTA {r['delta']:+.5f}", flush=True)

print("\n==== DOES THE GAP GROW WITH DEPTH? ====")
for n_est in (100, 500):
    a = [r for r in rows if r["n_estimators"] == n_est]
    d6 = [r for r in a if r["max_depth"] == 6][0]["delta"]
    d8 = [r for r in a if r["max_depth"] == 8][0]["delta"]
    print(f"  at {n_est:4d} trees: depth 6 {d6:+.5f} -> depth 8 {d8:+.5f}   change {d8-d6:+.5f}")
print("==== AND WITH TREE COUNT? ====")
for depth in (6, 8):
    a = [r for r in rows if r["max_depth"] == depth]
    n1 = [r for r in a if r["n_estimators"] == 100][0]["delta"]
    n5 = [r for r in a if r["n_estimators"] == 500][0]["delta"]
    print(f"  at depth {depth}: 100 trees {n1:+.5f} -> 500 trees {n5:+.5f}   change {n5-n1:+.5f}")
json.dump(rows, open(sys.argv[1], "w"), indent=2)
