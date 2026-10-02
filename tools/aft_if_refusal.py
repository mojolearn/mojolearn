"""IsolationForest refuses a non-finite training cell with ValueError, and a
finite fit after it works (lane apple-fast-trees2, FAST on Apple device scan)."""
import numpy as np
import mojolearn as ml

rng = np.random.default_rng(0)
x = rng.standard_normal((5000, 8)).astype(np.float32)
m = ml.IsolationForest(n_estimators=10, random_state=0)
for bad in (np.nan, np.inf):
    y = x.copy()
    y[4321, 5] = bad
    try:
        m.fit(y)
        print("AFT-REFUSAL FAIL no error for", bad)
    except ValueError as e:
        print("AFT-REFUSAL ok", bad, str(e)[:80], "fitted_attr=%s" % hasattr(m, "_x"))
s = m.fit(x).score_samples(x[:5])
print("AFT-REFUSAL finite fit ok", np.asarray(s).round(4).tolist())
