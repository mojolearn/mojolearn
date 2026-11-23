# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and scikit-learn reference values of the CategoricalNB and naive Bayes partial_fit Mojo checks
(lane bm-rest-nb, 2026-11-23; checks/native_tests/prep_naive_bayes.mojo `test_categorical`,
`test_categorical_sample_weight`, `test_nb_partial_fit`).

The original tests draw their inputs from NumPy's PCG64 streams and compare against scikit-learn at run time; Mojo
can do neither, so both are stored as raw little-endian words (`.f32`, `.i32`, `.f64`), written by `_put` of
`tools/py2mojo_native_fixtures.py`. Every array is the one the original test built (same seed, draw order and dtype)
and every reference is the scikit-learn call the original test made. Class labels are stored as their codes (the
labels are the integers 0, 1, 2, so the codes are the labels). A CategoricalNB per-feature block list
(feature_log_prob_, category_count_) is stored as its blocks concatenated in feature order, with n_categories_
beside it.

    checks/native_tests/fixtures/prep_nbx/   prep_naive_bayes.mojo
        cat-*     test_x_prep_naive_bayes.py::test_categorical
        wcat*-*   test_x_prep_naive_bayes.py::test_sample_weight_and_min_categories, the CategoricalNB half
        pf-*      test_x_prep_parity.py::test_nb_partial_fit

Regenerate and repack (the repository's pixi `test` environment):

    .pixi/envs/test/bin/python tools/py2mojo_native_fixtures_bm_rest_nb.py && \\
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures/prep_nbx

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

FAMILY = "prep_nbx"


def _blocks(blocks):
    return np.concatenate([np.asarray(b, dtype=np.float64).ravel() for b in blocks])


def cat_plain():
    import sklearn.naive_bayes as sknb
    # test_categorical
    rng = np.random.default_rng(4)
    n, d = 500, 4
    y = rng.integers(0, 3, n)
    X = np.clip(rng.integers(0, 4, (n, d)) + (y[:, None] > 0) * rng.integers(0, 2, (n, d)), 0, 4).astype(np.float32)
    Xh = np.clip(rng.integers(0, 5, (100, d)), 0, 4).astype(np.float32)
    Xh = np.minimum(Xh, X.max(axis=0))
    r = sknb.CategoricalNB(alpha=0.5).fit(X, y)
    return [_put(FAMILY, "cat-x.f32", X, np.float32), _put(FAMILY, "cat-y.i32", y, np.int32),
            _put(FAMILY, "cat-xh.f32", Xh, np.float32),
            _put(FAMILY, "cat-ncat.i32", r.n_categories_, np.int32),
            _put(FAMILY, "cat-flp.f64", _blocks(r.feature_log_prob_), np.float64),
            _put(FAMILY, "cat-proba.f64", r.predict_proba(Xh), np.float64),
            _put(FAMILY, "cat-predict.i32", r.predict(Xh), np.int32)]


# test_sample_weight_and_min_categories: the CategoricalNB keyword sets in order
WCAT_KW = (dict(), dict(min_categories=6), dict(min_categories=[2, 7, 5]))


def cat_weighted():
    import sklearn.naive_bayes as sknb
    rng = np.random.default_rng(8)
    Xc = rng.integers(0, 4, (500, 3)).astype(np.float32)
    yc = rng.integers(0, 3, 500)
    wc = rng.uniform(0.1, 2.0, 500)
    # the door admits the float64 weights as float32 words (admit_weight_words)
    out = [_put(FAMILY, "wcat-x.f32", Xc, np.float32), _put(FAMILY, "wcat-y.i32", yc, np.int32),
           _put(FAMILY, "wcat-w.f32", wc, np.float32)]
    for k, kw in enumerate(WCAT_KW):
        r = sknb.CategoricalNB(**kw).fit(Xc, yc, sample_weight=wc)
        out.append(_put(FAMILY, f"wcat{k}-ncat.i32", r.n_categories_, np.int32))
        out.append(_put(FAMILY, f"wcat{k}-flp.f64", _blocks(r.feature_log_prob_), np.float64))
        out.append(_put(FAMILY, f"wcat{k}-proba.f64", r.predict_proba(Xc[:50]), np.float64))
    return out


PF_NAMES = ("MultinomialNB", "ComplementNB", "BernoulliNB", "CategoricalNB", "GaussianNB")


def nb_partial():
    import sklearn.naive_bayes as sknb
    # test_nb_partial_fit
    rng = np.random.default_rng(7)
    X = rng.standard_normal((600, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5)
    Xa = np.abs(X)
    Xc = np.clip(np.floor(Xa * 2), 0, 4).astype(np.float32)
    Xc[:200] = np.minimum(Xc[:200], 2)
    w = (Xa[:, 2] + 0.5).astype(np.float32)
    sel = np.flatnonzero(y[:200] != 2)
    out = [_put(FAMILY, "pf-x.f32", X, np.float32), _put(FAMILY, "pf-xc.f32", Xc, np.float32),
           _put(FAMILY, "pf-y.i32", y, np.int32), _put(FAMILY, "pf-w.f32", w, np.float32),
           _put(FAMILY, "pf-sel.i32", sel, np.int32)]
    data_of = {"MultinomialNB": Xa, "ComplementNB": Xa, "BernoulliNB": X, "CategoricalNB": Xc, "GaussianNB": X}
    for name in PF_NAMES:
        data = data_of[name]
        r = getattr(sknb, name)()
        r.partial_fit(data[sel], y[sel], classes=[0, 1, 2])
        r.partial_fit(data[200:450], y[200:450])
        r.partial_fit(data[450:], y[450:], sample_weight=w[450:])
        tag = name.lower()
        out.append(_put(FAMILY, f"pf-{tag}-class_count.f64", r.class_count_, np.float64))
        out.append(_put(FAMILY, f"pf-{tag}-proba.f64", r.predict_proba(data), np.float64))
        if name == "CategoricalNB":
            out.append(_put(FAMILY, f"pf-{tag}-ncat.i32", r.n_categories_, np.int32))
            out.append(_put(FAMILY, f"pf-{tag}-category_count.f64", _blocks(r.category_count_), np.float64))
        elif name == "GaussianNB":
            out.append(_put(FAMILY, f"pf-{tag}-theta.f64", r.theta_, np.float64))
        else:
            out.append(_put(FAMILY, f"pf-{tag}-feature_count.f64", r.feature_count_, np.float64))
    return out


SETS = (cat_plain, cat_weighted, nb_partial)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
