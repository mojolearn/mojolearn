"""Reference check: mojolearn GaussianProcessClassifier (IDENTICAL, the loaded
binding) against scikit-learn 1.9.0's GaussianProcessClassifier(optimizer=None)
on the same kernel, the identity lanes' own fixtures (256 training rows of
four columns, 64 held-out rows). sk_ref.npz was written by sk_ref.py in the
bench env. We do not tune to match; this reports the difference."""
import sys
import numpy as np

W = "/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-gpc"
sys.path.insert(0, W + "/python")
sys.path.insert(0, W + "/tools")
import mojolearn as ml
from mojolearn._cpu_reference import reference_training
import identity_break as ib

ref = np.load(W + "/../gpc/sk_ref.npz")
worst = {"bin": 0.0, "mc": 0.0}
lml_worst = {"bin": 0.0, "mc": 0.0}
agree = {"bin": [0, 0], "mc": [0, 0]}
print(f"vendor {ml.vendor()} mode {ml.numeric_mode()}")
print("fixture  lane  max|proba diff|  |lml diff|  label agreement  n_iter")
with reference_training():
    for fx in ib.FIXTURES:
        X, yc, yr = ib.fixture(fx)
        q = ib.heldout(fx)[:64, :4]
        for name, kernel, y in (
            ("bin", ml.ConstantKernel(1.0) * ml.RBF(1.0), yc[:256]),
            ("mc", ml.ConstantKernel(2.0) * ml.Matern(1.0, nu=1.5), ib._gpc_three_classes(X)),
        ):
            m = ml.GaussianProcessClassifier(kernel=kernel).fit(X[:256, :4], y)
            p = np.asarray(m.predict_proba(q))
            d = float(np.abs(p - ref[f"{fx}__{name}_proba"]).max())
            dl = abs(m.log_marginal_likelihood_value_ - float(ref[f"{fx}__{name}_lml"][0]))
            same = int((np.asarray(m.predict(q)) == ref[f"{fx}__{name}_pred"]).sum())
            worst[name] = max(worst[name], d)
            lml_worst[name] = max(lml_worst[name], dl)
            agree[name][0] += same
            agree[name][1] += 64
            print(f"{fx:13s} {name:4s} {d:.3e}  {dl:.3e}  {same}/64  {m.n_iter_}")
for name in ("bin", "mc"):
    print(f"WORST {name}: max|proba diff| {worst[name]:.3e}, max|lml diff| {lml_worst[name]:.3e}, "
          f"labels {agree[name][0]}/{agree[name][1]}")
