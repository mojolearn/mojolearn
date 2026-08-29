import faulthandler, sys, time
import numpy as np
faulthandler.dump_traceback_later(240, exit=True)
import mojolearn as ml
rng = np.random.default_rng(0)
X = rng.standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 3] + 0.5 * X[:, 4] > 0).astype(np.int32)
def step(name, fn):
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(240, exit=True)
    t = time.time(); r = fn(); print(f"STEP {name}: ok {time.time()-t:.2f}s", flush=True); return r
print("mode", ml.numeric_mode(), flush=True)
m = step("rf_fit", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
p = step("rf_predict", lambda: m.predict(X))
pp = step("rf_predict_proba", lambda: m.predict_proba(X))
m2 = step("rf_fit_again", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
pp2 = step("rf_predict_proba_again", lambda: m2.predict_proba(X))
print("proba equal across fits:", np.array_equal(pp, pp2), flush=True)
print("RF_STEPS DONE", flush=True)
