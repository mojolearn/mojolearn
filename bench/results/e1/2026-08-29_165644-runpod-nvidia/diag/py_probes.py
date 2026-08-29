import faulthandler, gc, sys, time
import numpy as np
faulthandler.dump_traceback_later(110, exit=True)
import mojolearn as ml
which = sys.argv[1]
rng = np.random.default_rng(0)
X = rng.standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 3] + 0.5 * X[:, 4] > 0).astype(np.int32)
def step(name, fn):
    faulthandler.cancel_dump_traceback_later(); faulthandler.dump_traceback_later(110, exit=True)
    t = time.time(); r = fn(); print(f"STEP {name}: ok {time.time()-t:.2f}s", flush=True); return r
if which == "if_small":      # the probe's shape through the BINDING
    step("if_small_fit", lambda: ml.IsolationForest(n_estimators=4, max_samples=32, random_state=5).fit(X[:64, :4]))
elif which == "if_default":  # the estimator defaults through the binding (the fixed build)
    m = step("if_default_fit", lambda: ml.IsolationForest(n_estimators=16, random_state=5).fit(X))
    step("if_default_score", lambda: m.score_samples(X))
    step("if_default_fit_again", lambda: ml.IsolationForest(n_estimators=16, random_state=5).fit(X))
elif which == "rf_gc":       # second fit after the first model is gone
    m = step("rf_fit1", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    del m; gc.collect()
    step("rf_fit2_after_gc", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
elif which == "rf_reg":      # the regressor twice
    step("rfreg_fit1", lambda: ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, X[:, 0]))
    step("rfreg_fit2", lambda: ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, X[:, 0]))
elif which == "rf_small":    # tiny second fit
    step("rf_fit1", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("rf_fit2_small", lambda: ml.RandomForestClassifier(n_estimators=2, max_depth=3, random_state=7).fit(X[:500], y[:500]))
elif which == "rf_then_et":  # a different binding's context after rf
    step("rf_fit1", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("et_fit_after_rf", lambda: ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
elif which == "et_twice":    # sequential contexts in a lane that passes
    step("et_fit1", lambda: ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("et_fit2", lambda: ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("rf_fit_after_et", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
print("PY_PROBE DONE", flush=True)
