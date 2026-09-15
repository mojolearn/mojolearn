import sys, numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-gpc/tools")
import identity_break as ib
from sklearn.gaussian_process import GaussianProcessClassifier as G
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Matern
out = {}
for fx in ib.FIXTURES:
    X, yc, yr = ib.fixture(fx)
    Xh = ib.heldout(fx)
    Xt = X[:256, :4].astype(np.float64); q = Xh[:64, :4].astype(np.float64)
    m = G(kernel=C(1.0, "fixed") * RBF(1.0, "fixed"), optimizer=None).fit(Xt, yc[:256])
    out[f"{fx}__bin_proba"] = m.predict_proba(q); out[f"{fx}__bin_pred"] = m.predict(q)
    out[f"{fx}__bin_lml"] = np.array([m.log_marginal_likelihood_value_])
    y3 = ib._gpc_three_classes(X) if hasattr(ib, "_gpc_three_classes") else None
    if y3 is None:
        s = (X[:256, 3] + np.float32(0.5) * X[:256, 4]).astype(np.float32)
        y3 = np.empty(256, dtype=np.int64); y3[np.argsort(s, kind="stable")] = (np.arange(256) * 3) // 256
    m3 = G(kernel=C(2.0, "fixed") * Matern(1.0, "fixed", nu=1.5), optimizer=None).fit(Xt, y3)
    out[f"{fx}__mc_proba"] = m3.predict_proba(q); out[f"{fx}__mc_pred"] = m3.predict(q)
    out[f"{fx}__mc_lml"] = np.array([m3.log_marginal_likelihood_value_])
    print(fx, "ok", m.log_marginal_likelihood_value_, m3.log_marginal_likelihood_value_)
np.savez("/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/gpc/sk_ref.npz", **out)
