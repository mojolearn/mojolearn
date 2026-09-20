"""Values, not hashes: (A) the Plain held-out curve, (B) RMSE on Depthwise and
Lossguide. Run on a CPU-only process and on a GPU process; diff at home."""
import sys

import numpy as np

sys.path.insert(0, "tools")
import identity_break as ib  # noqa: E402
import mojolearn as ml  # noqa: E402

X, yc, yr = ib.fixture("base")[:3]
Xh = ib.heldout("base")
ych = ib.labels_for(Xh, ib.HELDOUT_SEED)[0]
print("vendor", ml.vendor())


def hx(v):
    return " ".join(float(x).hex() for x in v)


# ---- A: the lane's first fit, fitted three times in one process ----
kw = dict(n_estimators=6, max_depth=6, loss="Logloss", learning_rate=0.03,
          random_strength=0.0, bootstrap_type="No", leaf_estimation_iterations=10,
          use_best_model=False)
for rep in range(3):
    if rep == 2:
        # an unrelated fit in between, so the allocator's state differs
        ml.GradientBoosting(n_estimators=3, max_depth=4, loss="RMSE", learning_rate=0.03,
                            random_strength=0.0, bootstrap_type="No").fit(X[:7000], yr[:7000])
    m = ml.GradientBoosting(**kw).fit(X, yc, eval_set=(Xh, ych))
    print("A rep", rep, "test", [round(float(v), 6) for v in m.test_loss_curve_])
    print("A rep", rep, "test_hex", hx(m.test_loss_curve_))
    print("A rep", rep, "learn_hex", hx(m.loss_curve_))
# the same held-out rows as the LEARN set: its learn curve is the held-out
# curve's arithmetic on a cursor that is filled
t = ml.GradientBoosting(**kw).fit(Xh, ych, eval_set=(Xh, ych))
print("A self test_hex ", hx(t.test_loss_curve_))
print("A self learn_hex", hx(t.loss_curve_))

# ---- B: RMSE on the non-symmetric driver, two trees ----
Xs, ys = X[:4000], yr[:4000]
for name, k in (("Depthwise", dict(grow_policy="Depthwise")),
                ("Depthwise-nobfa", dict(grow_policy="Depthwise", boost_from_average=False)),
                ("Lossguide", dict(grow_policy="Lossguide")),
                ("Symmetric", dict(bootstrap_type="No", random_strength=0.0))):
    g = ml.GradientBoosting(n_estimators=2, loss="RMSE", **k).fit(Xs, ys)
    print("B", name, "lr", g.learning_rate_, "loss_hex", hx(g.loss_curve_))
    for line in str(g.model_).split("\n"):
        if line.startswith(("bias", "ntree", "node", "leaf", "weight", "loss ", "tree", "split")):
            print("B", name, "|", line)
