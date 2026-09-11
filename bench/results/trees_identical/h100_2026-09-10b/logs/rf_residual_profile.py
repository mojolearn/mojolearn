"""RF Python-side residual profile (scratch, pod only): times the pieces of
RandomForestClassifier.fit at HIGGS 1M the way the speed harness calls it
(C-order float32 X, float32 y). IDENTICAL tier. Not a certifiable timing."""
import os, sys, time
sys.path.insert(0, "/root/mojolearn/tools"); sys.path.insert(0, "/root/mojolearn/bench/speed")
sys.path.insert(0, "/root/mojolearn/python")
os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
import numpy as np
import speed_gbdt_arm as spec
import forest_speed_arm as fsa
import mojolearn
from mojolearn import randomforest as rfmod
from mojolearn._labels import encode_labels
from mojolearn._buffer import as_f32_colmajor, addr_ro
from mojolearn._forest_protocol import _forest_fit_function, _forest_fit_arrays

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
data = spec.load_with_fallback("higgs", spec.size_tag(), rows)
cfg = spec.lane_config("rf", spec.size_tag())
fsa.prepare_our_inputs(data)
X, y = data._ours_X, data._ours_y
print("X", X.shape, X.dtype, X.flags["C_CONTIGUOUS"], "y", y.shape, y.dtype)

def pc(): return time.perf_counter()

for r in range(reps):
    arms = fsa.our_rf_arm("rf", cfg, data)
    model = arms.make()
    t = {}
    t0 = pc()
    model._refresh_config(); model._capture_fit_mode()
    t["config"] = pc() - t0
    t1 = pc(); classes, y32 = encode_labels(y); t["encode_labels"] = pc() - t1
    model.classes_ = classes; model.n_classes_ = int(len(classes))
    t2 = pc(); binding = model._bind("_mojolearn_rf"); fit_fn = _forest_fit_function(binding, "rf_classifier_fit"); t["bind"] = pc() - t2
    t3 = pc(); Xf, copied = as_f32_colmajor(X, name="X"); t["as_f32_colmajor"] = pc() - t3
    n_rows, n_features = Xf.shape
    params = model._fit_params(n_rows, n_features, model.n_classes_)
    t4 = pc(); xa = addr_ro(Xf, name="X"); ya = addr_ro(y32, name="y"); t["addr"] = pc() - t4
    t5 = pc(); out = fit_fn(xa, ya, params, model._cfg["criterion"]); t["native_fit_fn"] = pc() - t5
    t6 = pc(); del Xf; t["del_Xf"] = pc() - t6
    t7 = pc(); arrays = _forest_fit_arrays(out); t["_forest_fit_arrays"] = pc() - t7
    t8 = pc(); del out; t["del_out"] = pc() - t8
    total = pc() - t0
    print("rep", r, "total_ms=%.1f" % (total * 1e3), " ".join("%s=%.1f" % (k, v * 1e3) for k, v in t.items()))
    del arrays, model
# and the whole fit as the harness calls it, for the same process
for r in range(2):
    model = fsa.our_rf_arm("rf", cfg, data).make()
    t0 = pc(); model.fit(X, y); print("harness-style fit ms=%.1f" % ((pc() - t0) * 1e3))
    del model
