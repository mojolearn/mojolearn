"""RandomForest and ExtraTrees (classifier and regressor) refuse a non-finite
training cell with ValueError, for C-order and F-order X, and a finite fit
after it works (lane apple-fast-rfet-scan, FAST on Apple device scan).
Prints whether the binding scans on device."""
import numpy as np
import mojolearn as ml

rng = np.random.default_rng(0)
x = rng.standard_normal((5000, 8)).astype(np.float32)
yc = (x[:, 0] > 0).astype(np.int64)
yr = x[:, 1].astype(np.float32)
cases = [
    ("rf-clf", ml.RandomForestClassifier, yc),
    ("rf-reg", ml.RandomForestRegressor, yr),
    ("et-clf", ml.ExtraTreesClassifier, yc),
    ("et-reg", ml.ExtraTreesRegressor, yr),
]
for name, cls, y in cases:
    m = cls(n_estimators=4, max_depth=6, random_state=0, device="gpu")
    for order in ("C", "F"):
        for bad in (np.nan, np.inf, -np.inf):
            xb = np.array(x, order=order, copy=True)
            xb[4321, 5] = bad
            try:
                m.fit(xb, y)
                print("AFT-REFUSAL FAIL", name, order, bad, "no error")
            except ValueError as e:
                print("AFT-REFUSAL ok", name, order, bad, str(e)[:70])
            except Exception as e:  # wrong type is a failure
                print("AFT-REFUSAL FAIL", name, order, bad, type(e).__name__, str(e)[:120])
    p = np.asarray(m.fit(x, y).predict(x[:5]))
    print("AFT-REFUSAL finite fit ok", name, p.round(4).tolist())
for b, f in (("_mojolearn_rf", "rf_device_finite_scan"), ("_mojolearn_trees", "trees_device_finite_scan")):
    from mojolearn import _backend
    mod = _backend.binding(b)
    print("AFT-REFUSAL device_scan", f, getattr(mod, f, lambda: "absent")())
