# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's GPU speed board (phase C, lane `prep`).

Times every prep-family estimator (preprocessing, feature selection,
imputers, encoders, naive Bayes, LDA/QDA) at the board's shapes
(tools/bench_board_algos.py, lane prep) on the two R2 datasets, under the
numeric mode of the environment (MOJOLEARN_NUMERIC_MODE). Each case runs once
to load, then REPS timed runs; the minimum is reported, split into the time
inside the binding (`x_prep_run`, the GPU program) and the rest (Python:
arena layout, label encoding, read back). Every case prints a digest of its
outputs, so a before and an after on IDENTICAL show the same bits by eye (the
lane check proves it by column).

    python bench/x_prep_speed.py [--dataset taxi,higgs] [--rows 1000000] [--reps 3]
                                 [--only name,...] [--profile]

--profile re-runs every program of the last timed run stage by stage (prefix
k of its stages, each prefix timed once; stage k's time is T(k) - T(k-1)) and
prints one XPPROF line per stage, so the hot stage of each case is named.

Data: GBM_BENCH_DATA (default ~/datasets/gbm-bench), staged from R2 with
`tools/dataset_store.sh stage` (taxi/taxi_speed.npz, higgs/higgs_speed.npz).
Lines: `XPSPEED <dataset> <case> <rows> <total_s> <binding_s> <programs> <digest>`
and `XPPROF <dataset> <case> <program> <stage> <op> <units> <seconds>`.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _root():
    return os.environ.get("GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench"))


def _np(a):
    if a is None:
        return None
    if isinstance(a, list):
        return np.asarray([float(v) if isinstance(v, (int, float)) else hash(v) for v in a])
    return np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a)


def _digest(*vs):
    h = hashlib.sha256()
    for v in vs:
        a = _np(v)
        if a is None:
            h.update(b"-")
            continue
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def load(dataset, n, seed=7):
    """dict of blocks: raw (float32, 16 columns), y (binary int64), cat
    (small integer codes as float32), nonneg, counts, yreg, nan (raw with 10%
    NaN by a seed mask)."""
    if dataset == "taxi":
        z = np.load(os.path.join(_root(), "taxi", "taxi_speed.npz"))
        x = np.asarray(z["x"][:n], dtype=np.float32)
        y = np.asarray(z["card"][:n]).astype(np.int64)
        yreg = np.asarray(z["fare"][:n], dtype=np.float32)
        cat = x[:, [0, 1, 3, 4, 5, 6, 7, 8, 9]]
        raw = x
    else:
        z = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))
        x = np.asarray(z["x"][:n], dtype=np.float32)
        y = np.asarray(z["y"][:n]).astype(np.int64)
        yreg = x[:, 3].copy()
        raw = np.ascontiguousarray(x[:, :16])
        binned = np.floor(np.clip(x[:, [1, 2, 6, 7, 10]], -3, 3) * 3)
        cat = np.concatenate([np.round(x[:, [8, 12, 16, 20]] * 4), binned], axis=1)
    rng = np.random.default_rng(seed)
    nan = raw.copy()
    nan[rng.random(nan.shape) < 0.1] = np.nan
    nonneg = raw - raw.min(0)
    counts = np.floor(np.abs(raw - np.median(raw, 0)) / (raw.std(0) + 1e-6) * 3)
    c = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    return dict(raw=c(raw), y=y, cat=c(cat), nonneg=c(nonneg), counts=c(counts), yreg=c(yreg), nan=c(nan),
                raw16=c(raw[:, :16]))


def cases(ml):
    """name -> (block keys, rows cap or None, fn(blocks) -> outputs)."""
    C = {}

    def tr(name, cls, kw, blk="raw", cap=None, infer=True):
        def run(b):
            m = getattr(ml, cls)(**kw).fit(b[blk])
            return [m.transform(b[blk])] if infer else [getattr(m, "scale_", None)]
        C[name] = (cap, run)

    tr("robust-scaler", "RobustScaler", {})
    tr("maxabs-scaler", "MaxAbsScaler", {})
    tr("quantile-transformer", "QuantileTransformer",
       dict(n_quantiles=1000, output_distribution="uniform", subsample=None, random_state=7))
    tr("power-transformer", "PowerTransformer", dict(method="yeo-johnson", standardize=True))
    tr("normalizer", "Normalizer", dict(norm="l2"))
    tr("binarizer", "Binarizer", dict(threshold=0.0))
    tr("poly-features", "PolynomialFeatures", dict(degree=2, include_bias=False), blk="raw16")
    tr("spline", "SplineTransformer", dict(n_knots=5, degree=3), blk="raw16")
    tr("kbins", "KBinsDiscretizer", dict(n_bins=16, encode="ordinal", strategy="quantile",
                                         quantile_method="linear", subsample=None))
    tr("onehot", "OneHotEncoder", dict(handle_unknown="ignore", sparse_output=False), blk="cat")
    tr("ordinal", "OrdinalEncoder", dict(handle_unknown="use_encoded_value", unknown_value=-1), blk="cat")
    tr("variance-threshold", "VarianceThreshold", dict(threshold=0.01))

    def target_encoder(b):
        m = ml.TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=7)
        return [m.fit_transform(b["cat"], b["y"]), m.transform(b["cat"])]
    C["target-encoder"] = (None, target_encoder)

    def simple_imputer(b):
        m = ml.SimpleImputer(strategy="median").fit(b["nan"])
        return [m.transform(b["nan"])]
    C["simple-imputer"] = (None, simple_imputer)

    def simple_imputer_mean(b):
        m = ml.SimpleImputer(strategy="mean").fit(b["nan"])
        return [m.transform(b["nan"])]
    C["simple-imputer-mean"] = (None, simple_imputer_mean)

    def iterative_imputer(b):
        m = ml.IterativeImputer(max_iter=10, tol=1e-3, random_state=7, sample_posterior=False,
                                imputation_order="ascending")
        return [m.fit_transform(b["nan"])]
    C["iterative-imputer"] = (100_000, iterative_imputer)

    def label_encoder(b):
        y = b["cat"][:, 4].astype(np.int64)
        m = ml.LabelEncoder().fit(y)
        return [m.transform(y)]
    C["label-encoder"] = (None, label_encoder)

    def label_binarizer(b):
        y = b["cat"][:, 4].astype(np.int64)
        m = ml.LabelBinarizer().fit(y)
        return [m.transform(y)]
    C["label-binarizer"] = (None, label_binarizer)

    def multilabel(b):
        cat = b["cat"].astype(np.int64)
        off = np.concatenate([[0], np.cumsum(cat.max(0) - cat.min(0) + 1)[:-1]])
        sets = [tuple(r) for r in (cat - cat.min(0) + off).tolist()]
        m = ml.MultiLabelBinarizer().fit(sets)
        return [m.transform(sets)]
    C["multilabel-binarizer"] = (200_000, multilabel)

    for nm, fn, blk, yk in (("select-f-classif", "f_classif", "raw", "y"),
                            ("select-chi2", "chi2", "nonneg", "y"),
                            ("select-f-regression", "f_regression", "raw", "yreg")):
        def sel(b, fn=fn, blk=blk, yk=yk):
            X = b[blk]
            m = ml.SelectKBest(getattr(ml, fn), k=X.shape[1] // 2).fit(X, b[yk])
            return [m.scores_, m.transform(X)]
        C[nm] = (None, sel)

    def mi(b):
        return [ml.mutual_info_classif(b["raw"], b["y"], random_state=7)]
    C["mutual-info-classif"] = (100_000, mi)

    def clf(name, cls, kw, blk):
        def run(b):
            m = getattr(ml, cls)(**kw).fit(b[blk], b["y"])
            return [m.predict_proba(b[blk]), m.predict(b[blk])]
        C[name] = (None, run)

    clf("gaussian-nb", "GaussianNB", {}, "raw")
    clf("bernoulli-nb", "BernoulliNB", {}, "raw")
    clf("categorical-nb", "CategoricalNB", {}, "cat")
    clf("multinomial-nb", "MultinomialNB", dict(alpha=1.0), "counts")
    clf("complement-nb", "ComplementNB", dict(alpha=1.0), "counts")
    clf("lda", "LinearDiscriminantAnalysis", dict(solver="svd"), "raw")
    clf("qda", "QuadraticDiscriminantAnalysis", dict(reg_param=1e-3), "raw")
    return C


class _Spy:
    """Wraps _Prog.run: the binding time of every program and its stages."""

    def __init__(self, xp):
        self.xp, self.progs, self.binding_s = xp, [], 0.0
        orig = xp._Prog.run
        spy = self

        def run(prog, mode):
            t0 = time.perf_counter()
            out = orig(prog, mode)
            spy.binding_s += time.perf_counter() - t0
            spy.progs.append((prog, mode, time.perf_counter() - t0))
            return out
        self.orig = orig
        xp._Prog.run = run

    def reset(self):
        self.progs, self.binding_s = [], 0.0


def profile(xp, spy, dataset, name, top=6, budget=20.0):
    """Prefix timing of the slowest programs of the last run."""
    inv = {v: k for k, v in xp._OPS.items()}
    ranked = sorted(enumerate(spy.progs), key=lambda t: -t[1][2])[:top]
    for pi, (prog, mode, wall) in ranked:
        stages = list(prog._stages)
        if wall * len(stages) / 2 > budget:
            # the prefix sum costs ~ stages/2 program runs: too dear here
            print(f"XPPROF {dataset} {name} {pi} skipped {len(stages)} {wall:.4f}", flush=True)
            continue
        prev = 0.0
        for k in range(1, len(stages) + 1):
            prog._stages = stages[:k]
            t0 = time.perf_counter()
            spy.orig(prog, mode)
            tk = time.perf_counter() - t0
            s = stages[k - 1]
            print(f"XPPROF {dataset} {name} {pi} {k - 1} {inv.get(s[0], s[0])} {s[1]} {tk - prev:.4f}", flush=True)
            prev = tk
        prog._stages = stages
        print(f"XPPROF {dataset} {name} {pi} total - - {wall:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taxi,higgs")
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    import mojolearn as ml
    from mojolearn import _expansion_prep as xp
    spy = _Spy(xp)
    only = [s for s in a.only.split(",") if s]
    skip = set(s for s in a.skip.split(",") if s)
    print(f"XPINFO mode={os.environ.get('MOJOLEARN_NUMERIC_MODE', 'identical')} rows={a.rows} reps={a.reps}",
          flush=True)
    for ds in a.dataset.split(","):
        t0 = time.perf_counter()
        blocks = load(ds, a.rows)
        print(f"XPINFO {ds} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        C = cases(ml)
        for name in (only or list(C)):
            if name in skip:
                continue
            cap, fn = C[name]
            n = min(a.rows, cap) if cap else a.rows
            b = {k: v[:n] for k, v in blocks.items()} if n < a.rows else blocks
            try:
                out = fn(b)                               # load / warm
                best = None
                for _ in range(a.reps):
                    spy.reset()
                    t0 = time.perf_counter()
                    out = fn(b)
                    t = time.perf_counter() - t0
                    if best is None or t < best[0]:
                        best = (t, spy.binding_s, len(spy.progs))
                print(f"XPSPEED {ds} {name} {n} {best[0]:.4f} {best[1]:.4f} {best[2]} {_digest(*out)}", flush=True)
                if a.profile:
                    profile(xp, spy, ds, name)
            except Exception as e:  # one case failing never hides the others
                print(f"XPERROR {ds} {name} {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
