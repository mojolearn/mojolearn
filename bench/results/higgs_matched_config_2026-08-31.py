"""THE RULING ON THE -0.00843 HIGGS AUC GAP, owed since 2026-08-22.

The published gap compares OUR unregularized arm against CatBoost running its
SHIPPED DEFAULTS: MVS at subsample 0.8 with random_strength 1.0. That is a
different estimator, not a tuning difference, and it favours them.

MVS cannot be matched from our side -- we do not implement it. So the match is
made by turning THEIR regularization off, which is what the FAST speed lane
already does on both arms.

Two arms, one process, same split, interleaved. Arm A is the published
config (theirs regularized). Arm B pins bootstrap_type='No',
random_strength=0.0 and boosting_type='Plain' on CatBoost ONLY -- ours is
byte-identical between the two arms, and is fitted ONCE and scored twice, so
any movement in our column would be a bug in this script.
"""
import json, time, sys, os, hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# EVERY IMPORT THAT CAN FAIL COMES BEFORE THE LOAD. The first attempt at this
# script spent three minutes parsing 11M rows and THEN died on
# `ModuleNotFoundError: mojolearn`, because the bench env does not carry the
# package and PYTHONPATH must point at the repository's python/. A load that
# precedes its own preconditions turns a one-second error into a three-minute
# one.
import mojolearn
import catboost as cb
print(f"mojolearn {mojolearn.__version__} {mojolearn.vendor()}, catboost {cb.__version__}", flush=True)

CACHE = "/Users/andrewhendel/datasets/gbm-bench/higgs/HIGGS.f32.npy"
t0 = time.time()
if os.path.exists(CACHE):
    print("loading HIGGS from the parsed cache ...", flush=True)
    d = np.load(CACHE, mmap_mode="r")
else:
    print("parsing HIGGS.csv.gz (once; cached after) ...", flush=True)
    import pandas as pd
    d = pd.read_csv("/Users/andrewhendel/datasets/gbm-bench/higgs/HIGGS.csv.gz",
                    header=None, dtype=np.float32).values
    np.save(CACHE, d)
y, X = np.ascontiguousarray(d[:, 0], dtype=np.float32), np.ascontiguousarray(d[:, 1:], dtype=np.float32)
del d
print(f"  {X.shape} in {time.time()-t0:.0f} s", flush=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=77, test_size=0.2)
del X, y
w = float(len(ytr)) / float((ytr == 1).sum())
print(f"  train {Xtr.shape} test {Xte.shape} pos_weight {w:.4f}", flush=True)

N, DEPTH, LR, L2, BORDERS = 500, 8, 0.1, 1.0, 254
out = {"n_estimators": N, "max_depth": DEPTH, "pos_weight": w}

print("\nOURS (unchanged between arms) ...", flush=True)
t = time.time()
m = mojolearn.GradientBoosting(n_estimators=N, max_depth=DEPTH, learning_rate=LR,
                               l2_leaf_reg=L2, border_count=BORDERS,
                               loss="Logloss", class_weights=[1.0, w],
                               random_state=0)
m.fit(Xtr, ytr)
out["ours_fit_s"] = time.time() - t
p = m.predict_proba(Xte)[:, 1]
out["ours_auc"] = float(roc_auc_score(yte, p))
out["ours_pred_sha"] = hashlib.sha256(np.ascontiguousarray(p, dtype=np.float32)).hexdigest()[:16]
print(f"  AUC {out['ours_auc']:.16f}  fit {out['ours_fit_s']:.1f}s  sha {out['ours_pred_sha']}", flush=True)

for tag, extra in (("published", {}),
                   ("matched", {"bootstrap_type": "No", "random_strength": 0.0,
                                "boosting_type": "Plain"})):
    print(f"\nCATBOOST CPU [{tag}] {extra} ...", flush=True)
    t = time.time()
    c = cb.CatBoost(dict(iterations=N, depth=DEPTH, learning_rate=LR, reg_lambda=L2,
                         objective="Logloss", scale_pos_weight=w, random_seed=0,
                         task_type="CPU", verbose=0, **extra))
    c.fit(Xtr, ytr)
    ft = time.time() - t
    raw = c.predict(Xte)
    auc = float(roc_auc_score(yte, raw))
    rp = c.get_all_params()
    out[f"cat_{tag}_auc"] = auc
    out[f"cat_{tag}_fit_s"] = ft
    out[f"cat_{tag}_resolved"] = {k: rp.get(k) for k in
        ("bootstrap_type", "subsample", "random_strength", "boosting_type", "border_count")}
    out[f"delta_{tag}"] = out["ours_auc"] - auc
    print(f"  AUC {auc:.16f}  fit {ft:.1f}s", flush=True)
    print(f"  resolved {out[f'cat_{tag}_resolved']}", flush=True)
    print(f"  DELTA ours-theirs {out[f'delta_{tag}']:+.5f}", flush=True)

print("\n==== RULING ====")
print(f"published config  delta {out['delta_published']:+.5f}")
print(f"matched   config  delta {out['delta_matched']:+.5f}")
json.dump(out, open(sys.argv[1], "w"), indent=2)
