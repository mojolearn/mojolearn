"""Before/after digests of the member image limits lane mf-img3 closed (2026-11-23), host column (and the fast
binding where one is installed). One line per configuration and output: name and sha256 of the output bytes (dtype
and shape included); a refusal is digested by its words. Base: origin/lane/member-fits 256d6473c1 (these
configurations took the estimators' own doors: the admits refused them), head lane/mf-img3 (native members, and the
host QN class-code query). Test tool, never product runtime.

Configurations:
  * `krr_fast_*`: KernelRidge(numeric_mode='fast') in BaggingRegressor and IterativeImputer (item 1: the image's
    own provider table runs the fast door's bodies).
  * `host_lr_k3_*`, `host_svc_k3_*`: LogisticRegression / LinearSVC with three classes, plain predict (the doors now
    take the class-code query on the host image) and inside CalibratedClassifierCV (item 2: LinearSVC is a member on
    the host column).
  * `gp_saved_*`: GaussianProcessRegressor over a loaded kernel (a `_SavedKernel`, the "nodes" layout) in
    BaggingRegressor (item 3).
  * `gbdt_*`: GradientBoosting with cat_features, one_hot_features, loss='MultiRMSE', loss='QueryRMSE' (no group
    ids: the door's fit without group_id), and GradientBoostingClassifier(numeric_mode='fast') in
    CalibratedClassifierCV (item 4).

EXPECTED: every line equal. The member runs the bodies the door ran, in the same order, on the same rows. Where a
line differs, stop and send it back, unless it is one of these, stated:
  * `gbdt_cat_*`: equal when the base door received the caller's float32 X (every caller here passes float32);
    the keys of a float32 value are the door's own Float32 keys.
  * `*_raised` lines whose words change from a door refusal to the member's refusal of the same input.
Configurations are independent: one process each across the host cores.

    python tools/mf_img3_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import tempfile

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed, n, k, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, s).astype(np.int64), s.astype(np.float32)


def _krr_fast(kernel):
    import mojolearn
    from mojolearn.impute import IterativeImputer
    X, _, s = _data(3, 90, 2)
    out = []
    tag = f"krr_fast_{kernel}"
    try:
        m = mojolearn.BaggingRegressor(mojolearn.KernelRidge(kernel=kernel, alpha=0.5, numeric_mode="fast"),
                                       n_estimators=3, random_state=0).fit(X, s)
        out.append((tag + "_bag_predict", _digest(m.predict(X))))
        for i, est in enumerate(m.estimators_):
            out.append((f"{tag}_bag{i}_dual", _digest(est.dual_coef_)))
        Xn = X.copy()
        Xn[np.random.default_rng(1).random(Xn.shape) < 0.1] = np.nan
        imp = IterativeImputer(estimator=mojolearn.KernelRidge(kernel=kernel, numeric_mode="fast"), max_iter=2,
                               random_state=0).fit(Xn)
        out.append((tag + "_imp_transform", _digest(imp.transform(Xn))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _host_codes(name):
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    X, y, _ = _data(5, 150, 3)
    tag = f"host_{name}_k3"
    out = []

    def member():
        if name == "lr":
            return mojolearn.LogisticRegression(max_iter=200, device="cpu")
        return mojolearn.LinearSVC(C=0.5, device="cpu")

    try:
        m = member().fit(X, y)
        out.append((tag + "_predict", _digest(m.predict(X))))
        out.append((tag + "_score", repr(float(m.score(X, y)))))
        cal = CalibratedClassifierCV(member(), cv=3).fit(X, y)
        out.append((tag + "_cal_proba", _digest(cal.predict_proba(X))))
        for p, (est, _) in enumerate(cal.calibrated_classifiers_):
            out.append((f"{tag}_cal{p}_coef", _digest(est.coef_)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _gp_saved(_):
    import mojolearn
    X, _, s = _data(9, 60, 2, d=2)
    tag = "gp_saved"
    out = []
    try:
        base = mojolearn.GaussianProcessRegressor(alpha=1e-2).fit(X, s)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gp.mojolearn")
            base.save(path)
            loaded = type(base).load(path)
        kernel = loaded.kernel_
        out.append((tag + "_kernel_class", type(kernel).__name__))
        m = mojolearn.BaggingRegressor(mojolearn.GaussianProcessRegressor(kernel=kernel, alpha=1e-2),
                                       n_estimators=2, random_state=0).fit(X, s)
        out.append((tag + "_bag_predict", _digest(m.predict(X))))
        for i, est in enumerate(m.estimators_):
            out.append((f"{tag}_bag{i}_alpha", _digest(est.alpha_)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


def _gbdt(name):
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    X, y, s = _data(13, 120, 2)
    Xc = X.copy()
    Xc[:, 0] = np.floor(np.abs(X[:, 0]) * 3)  # numeric categories
    Xc[::17, 0] = np.nan  # the missing category
    tag = f"gbdt_{name}"
    out = []
    try:
        if name == "cat":
            est = mojolearn.GradientBoosting(loss="RMSE", cat_features=[0], n_estimators=20, random_state=0)
            m = mojolearn.BaggingRegressor(est, n_estimators=2, random_state=0).fit(Xc, s)
            out.append((tag + "_predict", _digest(m.predict(Xc))))
            for i, e in enumerate(m.estimators_):
                out.append((f"{tag}{i}_map", _digest(e.category_map_)))
                out.append((f"{tag}{i}_model", hashlib.sha256(e.model_.encode()).hexdigest()[:24]))
        elif name == "hot":
            est = mojolearn.GradientBoosting(loss="RMSE", one_hot_features=[0], n_estimators=20, random_state=0)
            m = mojolearn.BaggingRegressor(est, n_estimators=2, random_state=0).fit(Xc, s)
            out.append((tag + "_predict", _digest(m.predict(Xc))))
        elif name == "multirmse":
            Y = np.stack([s, -2 * s + 1], axis=1).astype(np.float32)
            est = mojolearn.GradientBoosting(loss="MultiRMSE", n_estimators=20, random_state=0)
            m = mojolearn.BaggingRegressor(est, n_estimators=2, random_state=0).fit(X, Y)
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif name == "queryrmse":
            est = mojolearn.GradientBoosting(loss="QueryRMSE", n_estimators=20, random_state=0)
            m = mojolearn.BaggingRegressor(est, n_estimators=2, random_state=0).fit(X, s)
            out.append((tag + "_predict", _digest(m.predict(X))))
        else:
            cal = CalibratedClassifierCV(mojolearn.GradientBoostingClassifier(n_estimators=20, numeric_mode="fast"),
                                         cv=3).fit(X, y)
            out.append((tag + "_proba", _digest(cal.predict_proba(X))))
            out.append((tag + "_predict", _digest(cal.predict(X))))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ([(_krr_fast, k) for k in ("linear", "rbf", "poly")] + [(_host_codes, n) for n in ("lr", "svc")]
         + [(_gp_saved, None)] + [(_gbdt, n) for n in ("cat", "hot", "multirmse", "queryrmse", "fastclf")])


def _run(index):
    unit, arg = UNITS[index]
    return unit(arg)


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(UNITS))):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
