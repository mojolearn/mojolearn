"""Scratch, pod only: split the RF fit's Python export wrapper from the native
call at HIGGS 1M (IDENTICAL tier). Not a certifiable timing."""
import os, sys, time
sys.path.insert(0, "/root/mojolearn/tools"); sys.path.insert(0, "/root/mojolearn/bench/speed"); sys.path.insert(0, "/root/mojolearn/python")
os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
import speed_gbdt_arm as spec, forest_speed_arm as fsa
import mojolearn
from mojolearn._labels import encode_labels
from mojolearn._buffer import as_f32_colmajor, addr_ro, empty
from mojolearn import _forest_protocol as fp
pc = time.perf_counter
data = spec.load_with_fallback("higgs", spec.size_tag(), 1000000)
cfg = spec.lane_config("rf", spec.size_tag()); fsa.prepare_our_inputs(data)
X, y = data._ours_X, data._ours_y
for r in range(3):
    model = fsa.our_rf_arm("rf", cfg, data).make()
    model._refresh_config(); model._capture_fit_mode()
    classes, y32 = encode_labels(y); model.classes_ = classes; model.n_classes_ = len(classes)
    native = model._bind("_mojolearn_rf"); entry = getattr(native, "rf_classifier_fit_export")
    Xf, _ = as_f32_colmajor(X, name="X"); n_rows, n_features = Xf.shape
    params = model._fit_params(n_rows, n_features, model.n_classes_)
    t0 = pc(); desc = entry(addr_ro(Xf, name="X"), addr_ro(y32, name="y"), params, model._cfg["criterion"]); t_native = pc() - t0
    handle, trees, nodes, outputs, meta = desc
    dtypes = ('<i4', '<i4', '<f4', '<i4', '<f4'); sizes = (trees + 1, nodes, nodes, nodes, nodes * outputs)
    t1 = pc(); arrays = tuple(empty((s,), d) for s, d in zip(sizes, dtypes)); t_alloc = pc() - t1
    t2 = pc(); native.forest_export(handle, *(fp._addr(a) for a in arrays), [trees, nodes, outputs]); t_copy = pc() - t2
    t3 = pc(); native.forest_export_release(handle); t_release = pc() - t3
    t4 = pc(); del Xf; t_del = pc() - t4
    print("rep %d trees=%d nodes=%d native_ms=%.1f export_alloc_ms=%.1f export_copy_ms=%.1f export_release_ms=%.1f del_Xf_ms=%.1f" % (r, trees, nodes, t_native*1e3, t_alloc*1e3, t_copy*1e3, t_release*1e3, t_del*1e3))
    del arrays, model
