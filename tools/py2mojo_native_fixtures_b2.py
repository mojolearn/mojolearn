# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and scikit-learn reference values of the batch 2 Mojo checks (lane py2mojo-B, 2026-11-22;
docs/plans/PY2MOJO_CONVERSION.md, batch 2: the x_prep classifier, label, encoder, binning, selection and target
encoder tests).

The original tests draw their inputs from NumPy's PCG64 streams and compare against scikit-learn at run time; Mojo
can do neither, so both are stored as raw little-endian words (`.f32`, `.i32`, `.f64`), written by `_put` of
`tools/py2mojo_native_fixtures.py`. Every array is the one the original test built (same seed, draw order and dtype)
and every reference is the scikit-learn call the original test made; the originating test is named beside each set.
Class labels are stored as their codes (the position in the sorted classes, which is what the door's label encoding
hands the programs).

    checks/native_tests/fixtures/prep_nb/   prep_naive_bayes.mojo   (test_x_prep_naive_bayes.py)
    checks/native_tests/fixtures/prep_da/   prep_discriminant.mojo  (test_x_prep_discriminant.py)
    checks/native_tests/fixtures/prep_lab/  prep_labels.mojo        (test_x_prep_labels.py, test_x_prep_parity.py's
                                            test_multilabel_binarizer)
    checks/native_tests/fixtures/prep_kb/   prep_kbins.mojo         (test_x_prep_kbins.py, test_x_prep_parity.py's
                                            test_kbins_quantile_methods and test_kbins_sample_weight)

Generated with numpy 2.5.3 and scikit-learn 1.9.1 (the repository's pixi `test` environment, 2026-11-22). Regenerate
and repack:

    .pixi/envs/test/bin/python tools/py2mojo_native_fixtures_b2.py && \
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402


# ---------------------------------------------------------------- naive Bayes (test_x_prep_naive_bayes.py)
def _nb_data(seed=0, n=600, d=6):
    # test_x_prep_naive_bayes.py::_data (the labels "a", "b", "c" are the codes 0, 1, 2)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0.2, 1.0, d)).astype(np.float32)
    X[:, 5] = np.float32(1.0)
    return X, y


def _nb_refs(out, tag, r, Xh, attrs):
    for a in attrs:
        out.append(_put("prep_nb", f"{tag}-{a.rstrip('_')}.f64", getattr(r, a), np.float64))
    out.append(_put("prep_nb", f"{tag}-proba.f64", r.predict_proba(Xh), np.float64))
    out.append(_put("prep_nb", f"{tag}-logproba.f64", r.predict_log_proba(Xh), np.float64))
    out.append(_put("prep_nb", f"{tag}-predict.i32", r.predict(Xh), np.int32))


DISCRETE = ("feature_count_", "feature_log_prob_", "class_log_prior_")
# test_multinomial, test_bernoulli, test_complement: the keyword sets in order
NB_MNB_KW = (dict(alpha=1.0), dict(alpha=0.3, fit_prior=False))
NB_BNB_KW = (dict(), dict(binarize=0.5, alpha=2.0))
NB_CNB_KW = (dict(), dict(alpha=0.4, norm=True))


def nb_plain():
    import sklearn.naive_bayes as sknb
    X, y = _nb_data()
    Xh, _ = _nb_data(1)
    out = [_put("prep_nb", "x0.f32", X, np.float32), _put("prep_nb", "y0.i32", y, np.int32),
           _put("prep_nb", "xh1.f32", Xh, np.float32)]
    # test_gaussian
    _nb_refs(out, "gnb", sknb.GaussianNB().fit(X, y), Xh, ("theta_", "var_", "class_prior_"))
    Xa, Xha = np.abs(X), np.abs(Xh)
    for k, kw in enumerate(NB_MNB_KW):   # test_multinomial
        _nb_refs(out, f"mnb{k}", sknb.MultinomialNB(**kw).fit(Xa, y), Xha, DISCRETE)
    for k, kw in enumerate(NB_BNB_KW):   # test_bernoulli (X as drawn)
        _nb_refs(out, f"bnb{k}", sknb.BernoulliNB(**kw).fit(X, y), Xh, DISCRETE)
    for k, kw in enumerate(NB_CNB_KW):   # test_complement
        _nb_refs(out, f"cnb{k}", sknb.ComplementNB(**kw).fit(Xa, y), Xha, DISCRETE)
    return out


def nb_weighted():
    import sklearn.naive_bayes as sknb
    # test_sample_weight_and_min_categories, the GaussianNB and discrete half
    X, y = _nb_data(5)
    Xh, _ = _nb_data(6)
    w = np.random.default_rng(7).uniform(0.1, 3.0, len(y)).astype(np.float32)
    out = [_put("prep_nb", "x5.f32", X, np.float32), _put("prep_nb", "y5.i32", y, np.int32),
           _put("prep_nb", "xh6.f32", Xh, np.float32), _put("prep_nb", "w7.f32", w, np.float32)]
    _nb_refs(out, "wgnb", sknb.GaussianNB().fit(X, y, sample_weight=w), Xh,
             ("theta_", "var_", "class_prior_", "class_count_"))
    Xa, Xha = np.abs(X), np.abs(Xh)
    for tag, nm in (("wmnb", "MultinomialNB"), ("wcnb", "ComplementNB"), ("wbnb", "BernoulliNB")):
        r = getattr(sknb, nm)().fit(Xa, y, sample_weight=w)
        _nb_refs(out, tag, r, Xha, ("feature_count_", "class_count_", "feature_log_prob_"))
    return out


# ---------------------------------------------------------------- discriminant analysis (test_x_prep_discriminant.py)
def _da_data(seed=0, n=500, d=5, k=3):
    # test_x_prep_discriminant.py::_data
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    A = rng.standard_normal((d, d)) * 0.5 + np.eye(d)
    X = (rng.standard_normal((n, d)) @ A + y[:, None] * np.linspace(0.5, 1.5, d)).astype(np.float32)
    return X, y


def _da_put(out, name, array, dtype=np.float64):
    out.append(_put("prep_da", name, array, dtype))


def _da_inputs(out, tag, X, y, Xh):
    _da_put(out, f"{tag}-x.f32", X, np.float32)
    _da_put(out, f"{tag}-y.i32", y, np.int32)
    _da_put(out, f"{tag}-xh.f32", Xh, np.float32)


def _lda_refs(out, tag, r, Xh, transform):
    Xd = Xh.astype(np.float64)
    _da_put(out, f"{tag}-coef.f64", r.coef_)
    _da_put(out, f"{tag}-intercept.f64", r.intercept_)
    _da_put(out, f"{tag}-proba.f64", r.predict_proba(Xd))
    _da_put(out, f"{tag}-predict.i32", r.predict(Xd), np.int32)
    if hasattr(r, "covariance_"):
        _da_put(out, f"{tag}-cov.f64", r.covariance_)
    if transform:
        _da_put(out, f"{tag}-evr.f64", r.explained_variance_ratio_)
        b = r.transform(Xd)
        _da_put(out, f"{tag}-tw.i32", [b.shape[1]], np.int32)
        _da_put(out, f"{tag}-transform.f64", b)
    if len(r.classes_) == 2:
        _da_put(out, f"{tag}-decision.f64", r.decision_function(Xd))


def _qda_refs(out, tag, r, Xh):
    Xd = Xh.astype(np.float64)
    s = np.asarray([np.asarray(a) for a in r.scalings_])
    assert s.shape == (len(r.classes_), Xh.shape[1]), s.shape
    _da_put(out, f"{tag}-scalings.f64", s)
    _da_put(out, f"{tag}-proba.f64", r.predict_proba(Xd))
    _da_put(out, f"{tag}-predict.i32", r.predict(Xd), np.int32)
    if getattr(r, "covariance_", None) is not None:
        _da_put(out, f"{tag}-cov.f64", np.asarray([np.asarray(c) for c in r.covariance_]))
    if len(r.classes_) == 2:
        _da_put(out, f"{tag}-decision.f64", r.decision_function(Xd))


def da_lda():
    import sklearn.discriminant_analysis as skda
    out = []
    for k in (2, 3):   # test_lda
        X, y = _da_data(k=k)
        Xh, _ = _da_data(1, k=k)
        _da_inputs(out, f"lda{k}", X, y, Xh)
        r = skda.LinearDiscriminantAnalysis().fit(X.astype(np.float64), y)
        _lda_refs(out, f"lda{k}", r, Xh, True)
    return out


def da_qda():
    import sklearn.discriminant_analysis as skda
    out = []
    for k, reg in ((2, 0.0), (3, 0.1)):   # test_qda
        X, y = _da_data(2, k=k)
        Xh, _ = _da_data(3, k=k)
        _da_inputs(out, f"qda{k}", X, y, Xh)
        r = skda.QuadraticDiscriminantAnalysis(reg_param=reg).fit(X.astype(np.float64), y)
        _qda_refs(out, f"qda{k}", r, Xh)
    return out


def da_priors():
    import sklearn.discriminant_analysis as skda
    import sklearn.naive_bayes as sknb
    # test_priors (scikit-learn renormalizes LDA's [1, 2, 1] with a warning)
    X, y = _da_data(4, k=3)
    Xh, _ = _da_data(5, k=3)
    out = []
    _da_inputs(out, "pri", X, y, Xh)
    pairs = (("pri-gnb", sknb.GaussianNB(priors=[0.2, 0.5, 0.3]), X, Xh),
             ("pri-mnb", sknb.MultinomialNB(class_prior=[0.1, 0.6, 0.3]), np.abs(X), np.abs(Xh)),
             ("pri-lda", skda.LinearDiscriminantAnalysis(priors=[1.0, 2.0, 1.0]), X, Xh),
             ("pri-qda", skda.QuadraticDiscriminantAnalysis(priors=[0.25, 0.25, 0.5]), X, Xh))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag, r, Xa, Xt in pairs:
            r.fit(Xa.astype(np.float64), y)
            _da_put(out, f"{tag}-proba.f64", r.predict_proba(Xt.astype(np.float64)))
    return out


# test_solvers_and_shrinkage, the LDA keyword sets in order (shrinkage 'auto' is -1 in the Mojo table)
DA_LDA_KW = (dict(solver="lsqr"), dict(solver="lsqr", shrinkage="auto"), dict(solver="lsqr", shrinkage=0.3),
             dict(solver="eigen"), dict(solver="eigen", shrinkage="auto"), dict(solver="eigen", shrinkage=0.6),
             dict(solver="svd", store_covariance=True))
DA_QDA_KW = (dict(solver="eigen"), dict(solver="eigen", shrinkage="auto", store_covariance=True),
             dict(solver="eigen", shrinkage=0.2), dict(solver="svd", reg_param=0.1, store_covariance=True))


def da_solvers():
    import sklearn.discriminant_analysis as skda
    out = []
    for k in (2, 3):
        X, y = _da_data(6, k=k)
        X[:, 4] = X[:, 3] * 0.7 + X[:, 4] * 0.3         # a correlated pair: shrinkage matters
        Xh, _ = _da_data(7, k=k)
        _da_inputs(out, f"sol{k}", X, y, Xh)
        for j, kw in enumerate(DA_LDA_KW):
            r = skda.LinearDiscriminantAnalysis(**kw).fit(X.astype(np.float64), y)
            # the original compares explained_variance_ratio_ and transform for 'eigen' when every ratio is at most 1
            eigen_view = kw["solver"] == "eigen" and np.abs(r.explained_variance_ratio_).max() <= 1
            _lda_refs(out, f"sol{k}-lda{j}", r, Xh, eigen_view)
            _da_put(out, f"sol{k}-lda{j}-view.i32", [1 if eigen_view else 0], np.int32)
    X, y = _da_data(8, k=3)
    Xh, _ = _da_data(9, k=3)
    _da_inputs(out, "solq", X, y, Xh)
    for j, kw in enumerate(DA_QDA_KW):
        r = skda.QuadraticDiscriminantAnalysis(**kw).fit(X.astype(np.float64), y)
        _qda_refs(out, f"solq-qda{j}", r, Xh)
    return out


# ---------------------------------------------------------------- KBinsDiscretizer (test_x_prep_kbins.py)
def _kb_data(seed=0, n=500, d=4):
    # test_x_prep_kbins.py::_data
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 1] = np.round(X[:, 1])          # ties: some quantile bins collapse
    X[:, 2] = np.float32(1.5)            # constant
    return X


# test_strategies, the keyword sets in order (scikit-learn 1.9's quantile default is averaged_inverted_cdf)
KB_KW = (dict(strategy="uniform"), dict(strategy="quantile"), dict(strategy="quantile", quantile_method="linear"),
         dict(strategy="kmeans"))


def _kb_edges(out, tag, r):
    out.append(_put("prep_kb", f"{tag}-nbins.i32", r.n_bins_, np.int32))
    out.append(_put("prep_kb", f"{tag}-edges.f64", np.concatenate([np.asarray(e) for e in r.bin_edges_]),
                    np.float64))


def kbins():
    from sklearn.preprocessing import KBinsDiscretizer as SkKB
    X = _kb_data()
    out = [_put("prep_kb", "x0.f32", X, np.float32)]
    for k, kw in enumerate(KB_KW):
        r = SkKB(n_bins=5, encode="ordinal", **kw).fit(X.astype(np.float64))
        _kb_edges(out, f"s{k}", r)
        Z = r.transform(X.astype(np.float64))
        out.append(_put("prep_kb", f"s{k}-transform.f64", Z, np.float64))
        out.append(_put("prep_kb", f"s{k}-inverse.f64", r.inverse_transform(Z), np.float64))
    # test_onehot
    X1 = _kb_data(1)
    out.append(_put("prep_kb", "x1.f32", X1, np.float32))
    r = SkKB(n_bins=4, encode="onehot-dense", strategy="uniform").fit(X1.astype(np.float64))
    _kb_edges(out, "oh", r)
    want = r.transform(X1.astype(np.float64))
    out.append(_put("prep_kb", "oh-shape.i32", want.shape, np.int32))
    out.append(_put("prep_kb", "oh-transform.f64", want, np.float64))
    out.append(_put("prep_kb", "oh-inverse.f64", r.inverse_transform(want), np.float64))
    return out


# test_x_prep_parity.py::test_kbins_quantile_methods, the methods in order (x n_bins 3, 5, 7)
KB_METHODS = ("inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
              "median_unbiased", "normal_unbiased", "linear", "averaged_inverted_cdf")
# test_x_prep_parity.py::test_kbins_sample_weight, the keyword sets in order (x the three weights, x n_bins 3, 5)
KB_WKW = (dict(strategy="quantile"), dict(strategy="quantile", quantile_method="inverted_cdf"),
          dict(strategy="uniform"), dict(strategy="kmeans"))


def kbins_parity():
    import sklearn.preprocessing as sk
    out = []
    rng = np.random.default_rng(8)
    X = np.round(rng.standard_normal((299, 3)) * 4).astype(np.float32) / np.float32(4)
    out.append(_put("prep_kb", "qm-x.f32", X, np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, meth in enumerate(KB_METHODS):
            for nb in (3, 5, 7):
                r = sk.KBinsDiscretizer(n_bins=nb, encode="ordinal", quantile_method=meth).fit(X)
                _kb_edges(out, f"qm{i}-{nb}", r)
        rng = np.random.default_rng(9)
        X = np.round(rng.standard_normal((301, 3)) * 4).astype(np.float32) / np.float32(4)
        X[:, 2] = rng.standard_normal(301).astype(np.float32)
        out.append(_put("prep_kb", "sw-x.f32", X, np.float32))
        weights = (rng.integers(0, 4, 301).astype(np.float32), rng.random(301).astype(np.float32),
                   np.ones(301, np.float32))
        for s, sw in enumerate(weights):
            out.append(_put("prep_kb", f"sw{s}-w.f32", sw, np.float32))
            for j, kw in enumerate(KB_WKW):
                for nb in (3, 5):
                    r = sk.KBinsDiscretizer(n_bins=nb, encode="ordinal", **kw).fit(X, sample_weight=sw)
                    _kb_edges(out, f"sw{s}-{j}-{nb}", r)
    return out


# ---------------------------------------------------------------- labels (test_x_prep_labels.py)
def labels():
    import sklearn.preprocessing as skp
    out = []
    # test_label_encoder, the numeric label vectors (the str vector's encoding is the public boundary's)
    rng = np.random.default_rng(0)
    ys = (rng.integers(-5, 9, 200) * 3, rng.standard_normal(100).astype(np.float32))
    for i, y in enumerate(ys):
        r = skp.LabelEncoder().fit(y)
        out.append(_put("prep_lab", f"le{i}-y.{'i64' if i == 0 else 'f32'}", y, np.int64 if i == 0 else np.float32))
        out.append(_put("prep_lab", f"le{i}-k.i32", [len(r.classes_)], np.int32))
        out.append(_put("prep_lab", f"le{i}-classes.f64", r.classes_, np.float64))
        out.append(_put("prep_lab", f"le{i}-transform.i32", r.transform(y), np.int32))
    # test_label_binarizer: the same rng draws in the same order (the str case draws its noise too)
    rng = np.random.default_rng(1)
    ys = (rng.integers(0, 4, 50) * 2, rng.integers(0, 2, 30), np.array(["x", "y", "x"]), np.array([5, 5]))
    for i, y in enumerate(ys):
        for k, kw in enumerate((dict(), dict(neg_label=-1, pos_label=3))):
            r = skp.LabelBinarizer(**kw).fit(y)
            T = r.transform(y)
            Y = T.astype(np.float64)
            Y2 = Y + rng.standard_normal(Y.shape) * 0.3
            if y.dtype.kind == "U":
                continue
            tag = f"lb{i}{k}"
            out.append(_put("prep_lab", f"{tag}-y.i64", y, np.int64))
            out.append(_put("prep_lab", f"{tag}-shape.i32", [len(y), T.shape[1], len(r.classes_)], np.int32))
            out.append(_put("prep_lab", f"{tag}-classes.f64", r.classes_, np.float64))
            out.append(_put("prep_lab", f"{tag}-transform.i32", T, np.int32))
            out.append(_put("prep_lab", f"{tag}-y2.f32", Y2, np.float32))
            out.append(_put("prep_lab", f"{tag}-inv-y.f64", r.inverse_transform(Y), np.float64))
            out.append(_put("prep_lab", f"{tag}-inv-y2.f64", r.inverse_transform(Y2.astype(np.float32)), np.float64))
            out.append(_put("prep_lab", f"{tag}-inv-y2t.f64", r.inverse_transform(Y2.astype(np.float32), threshold=0.4),
                            np.float64))
    # test_multilabel_binarizer, the int label sets (the str sets are the public boundary's)
    ys = [[1, 3], [2], [], [3, 1, 7], [7, 7]]
    r = skp.MultiLabelBinarizer().fit(ys)
    out.append(_put("prep_lab", "mlb-classes.f64", r.classes_, np.float64))
    out.append(_put("prep_lab", "mlb-transform.i32", r.transform(ys), np.int32))
    # test_x_prep_parity.py::test_multilabel_binarizer: a multilabel indicator Y, three label pairs, S drawn per pair
    rng = np.random.default_rng(3)
    Y = (rng.random((50, 4)) > 0.6).astype(np.int64)
    out.append(_put("prep_lab", "ind-y.f32", Y, np.float32))
    for k, kw in enumerate((dict(), dict(neg_label=-1, pos_label=2), dict(neg_label=-2, pos_label=0))):
        r = skp.LabelBinarizer(**kw).fit(Y)
        assert r.y_type_ == "multilabel-indicator"
        S = rng.standard_normal((50, 4)).astype(np.float32)
        out.append(_put("prep_lab", f"ind{k}-transform.i32", r.transform(Y), np.int32))
        out.append(_put("prep_lab", f"ind{k}-s.f32", S, np.float32))
        out.append(_put("prep_lab", f"ind{k}-inv.i32", r.inverse_transform(S), np.int32))
        out.append(_put("prep_lab", f"ind{k}-inv3.i32", r.inverse_transform(S, threshold=0.3), np.int32))
    return out


SETS = (nb_plain, nb_weighted, da_lda, da_qda, da_priors, da_solvers, kbins, kbins_parity, labels)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
