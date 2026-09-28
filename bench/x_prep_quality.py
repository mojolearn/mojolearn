# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's FAST quality check (the paired rule: FAST quality never
worse, 5+ seeds on 2+ datasets, against scikit-learn in float64).

For every estimator the FAST device folds reach (x_prep/fastred.mojo:
col_stats, class_stats, PowerTransformer's pt_fold, IterativeImputer's
ii_mean / ii_gram), fit ours and scikit-learn on the
same rows (seed s draws ROWS rows of the dataset) and report ours' error
against the reference:

  transformers   max |ours - ref| / max(1, max |ref|) over the output
  VarianceThreshold  the same over variances_
  classifiers    predict_proba max |ours - ref| and the agreement of predict

Run it twice in one process's environment (MOJOLEARN_NUMERIC_MODE=fast): with
MOJOLEARN_XPREP_FAST_FOLDS=0 (the row-order units: the before arm) and =1
(the threadgroup folds: the after arm). Lines:
`XPQ <arm> <dataset> <seed> <case> <metric> <value>`.

    python bench/x_prep_quality.py --arm after [--seeds 5] [--rows 200000]
scikit-learn comes from PYTHONPATH (the steward job installs it in ~/skl).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x_prep_speed import load  # noqa: E402


def _np(a):
    return np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a, dtype=np.float64)


def _err(ours, ref):
    o, r = _np(ours), np.asarray(ref, dtype=np.float64)
    m = np.isfinite(r)
    return float(np.max(np.abs(o[m] - r[m])) / max(1.0, float(np.max(np.abs(r[m])))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--dataset", default="taxi,higgs")
    ap.add_argument("--only", default="", help="comma separated cases (default: all)")
    a = ap.parse_args()
    import mojolearn as ml
    import sklearn.preprocessing as skp
    import sklearn.feature_selection as skf
    import sklearn.naive_bayes as sknb
    import sklearn.discriminant_analysis as skda
    for ds in a.dataset.split(","):
        full = load(ds, 1_000_000)
        n_all = full["raw"].shape[0]
        for seed in range(a.seeds):
            idx = np.sort(np.random.default_rng(1000 + seed).choice(n_all, a.rows, replace=False))
            b = {k: v[idx] for k, v in full.items()}
            X, y = b["raw"], b["y"]
            X64 = X.astype(np.float64)

            def out(case, metric, v):
                print(f"XPQ {a.arm} {ds} {seed} {case} {metric} {v:.6g}", flush=True)

            m = ml.MaxAbsScaler().fit(X)
            out("maxabs-scaler", "transform_err", _err(m.transform(X), skp.MaxAbsScaler().fit(X64).transform(X64)))
            m = ml.VarianceThreshold(threshold=0.0).fit(X)
            out("variance-threshold", "variances_err", _err(m.variances_, skf.VarianceThreshold().fit(X64).variances_))
            m = ml.PowerTransformer().fit(X)
            r = skp.PowerTransformer().fit(X64)
            out("power-transformer", "lambda_err", _err(m.lambdas_, r.lambdas_))
            out("power-transformer", "transform_err", _err(m.transform(X), r.transform(X64)))
            m = ml.SimpleImputer(strategy="mean").fit(b["nan"])
            out("simple-imputer-mean", "transform_err",
                _err(m.transform(b["nan"]), __import__("sklearn.impute", fromlist=["x"]).SimpleImputer(
                    strategy="mean").fit(b["nan"].astype(np.float64)).transform(b["nan"].astype(np.float64))))
            sc_o = ml.f_classif(X, y)[0]
            out("f-classif", "scores_err", _err(sc_o, skf.f_classif(X64, y)[0]))
            nb = b["nan"][:20_000]
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer as SkII
            kw = dict(max_iter=10, tol=1e-3, random_state=7, sample_posterior=False, imputation_order="ascending")
            miss = np.isnan(nb)
            oi = _np(ml.IterativeImputer(**kw).fit_transform(nb))
            ri = SkII(**kw).fit_transform(nb.astype(np.float64))
            out("iterative-imputer", "imputed_err", _err(oi[miss], ri[miss]))
            for name, ours, ref, blk in (("gaussian-nb", ml.GaussianNB(), sknb.GaussianNB(), "raw"),
                                         ("qda", ml.QuadraticDiscriminantAnalysis(reg_param=1e-3),
                                          skda.QuadraticDiscriminantAnalysis(reg_param=1e-3), "raw"),
                                         ("multinomial-nb", ml.MultinomialNB(), sknb.MultinomialNB(), "counts"),
                                         ("lda", ml.LinearDiscriminantAnalysis(), skda.LinearDiscriminantAnalysis(), "raw")):
                Xb = b[blk]
                ours.fit(Xb, y)
                ref.fit(Xb.astype(np.float64), y)
                out(name, "proba_err", _err(ours.predict_proba(Xb), ref.predict_proba(Xb.astype(np.float64))))
                for attr in ("means_", "theta_", "var_"):
                    if hasattr(ref, attr) and getattr(ours, attr, None) is not None:
                        out(name, attr.rstrip("_") + "_err", _err(getattr(ours, attr), getattr(ref, attr)))
                out(name, "predict_agree", float(np.mean(_np(ours.predict(Xb)) == ref.predict(Xb.astype(np.float64)))))
                out(name, "accuracy", float(np.mean(_np(ours.predict(Xb)) == y)))


if __name__ == "__main__":
    main()
