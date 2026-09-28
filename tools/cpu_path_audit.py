#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""THE CPU PATH AUDIT (lane cpu, phase 1, 2026-09-27).

Does every public name of mojolearn train and infer on a CPU-only install,
and does the CPU give the GPU's bits? This tool answers it BY RUNNING the
public surface, not by reading the manifest: a family declared in
host_surface.py whose host binding lacks one function, or a Python door that
refuses on the CPU inside `fit`, reads as a gap here and nowhere else.

    # on a GPU box, with the GPU and host bindings built:
    python3 tools/cpu_path_audit.py record --out gpu.jsonl
    python3 tools/cpu_path_audit.py record --cpu --out cpu.jsonl
    python3 tools/cpu_path_audit.py diff gpu.jsonl cpu.jsonl [--markdown]

`record` runs every probe in child processes (a crash in one binding costs
that name, recorded CRASH, never the run). Each probe fits the public name on
a small non-uniform fixture and calls every inference method it has; every
stage is recorded OK (with the sha256 of its outputs), REFUSED (a by-name
refusal: NotImplementedError, or an ImportError naming no CPU
implementation) or ERROR. `--cpu` sets the CPU arm's environment the lane
check uses (tools/algos_lane_check.arm_env). `diff` reads the two records and
prints one row per name: the GPU and CPU status of each stage and whether the
CPU bits EQUAL the GPU's.

A probe is a sanity sweep of the DEFAULT path (default options, one small
shape). It is not an identity lane: admission is the verifier's
(tools/identity_break.py lanes, `algos_lane_check.sh`), and the lanes that
reach a name are the selector's (tools/lane_select.py).
"""
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Submodules whose own `__all__` holds public callables a user reaches as
#: `mojolearn.<module>.<name>`.
SUBMODULES = ("metrics", "linalg", "manifold", "resample", "hdbscan", "model_selection",
              "training", "parallel_forecasting", "parallel_gaussian_process",
              "parallel_model_selection")

#: Names that are not algorithms (types, configs, version helpers, modules,
#: result records): listed in the table as NOT AN ALGORITHM, never probed.
NOT_ALGORITHMS = {
    "__version__", "numeric_mode", "set_numeric_mode", "vendor", "gpu_arch", "gpu_arch_how",
    "Array", "ByteLanguageModelConfig", "LanguageModelConfig", "SambaConfig",
    "Mamba1State", "Mamba2State", "Mamba3State", "TransformerState",
    "ConstantKernel", "Matern", "RBF", "WhiteKernel",
    "linalg.numeric_mode", "linalg.profile", "linalg.require_identical", "linalg.PROFILE",
    "linalg.PROFILE_FAMILY", "linalg.PROFILE_VERSION", "linalg.PROFILE_BF16", "linalg.PROFILE_INT8",
    "resample.BootstrapResult", "resample.PermutationTestResult", "resample.MonteCarloResult",
    "resample.STATISTICS", "resample.METHODS", "resample.ALTERNATIVES", "resample.INTEGRANDS",
    "training.numeric_mode_used", "training.vendor_used", "training.accumulation_is_aligned",
    "model_selection.split_descriptor",
}

CLASSIFIER_HINTS = ("Classifier", "SVC", "Logistic", "NB", "Discriminant", "Perceptron",
                    "NearestCentroid", "LabelPropagation", "LabelSpreading", "GaussianProcessClassifier",
                    "OneVsRest", "Calibrated")
REGRESSOR_HINTS = ("Regressor", "Regression", "Ridge", "Lasso", "ElasticNet", "SVR", "Lars",
                   "GradientBoosting", "OrderedRMSE", "ExperimentalTwoLevel", "QNRegressor",
                   "PLS", "CCA", "Huber", "Bayesian", "ARD", "Isotonic")
NONNEG_HINTS = ("MultinomialNB", "ComplementNB", "CategoricalNB", "NMF", "LatentDirichlet",
                "chi2", "AdditiveChi2", "SkewedChi2", "AlternatingLeastSquares", "PoissonRegressor",
                "GammaRegressor", "TweedieRegressor")
INFER_METHODS = ("predict", "predict_proba", "predict_log_proba", "decision_function", "transform",
                 "score_samples", "kneighbors", "radius_neighbors", "inverse_transform", "apply")
FITTED_ATTRS = ("labels_", "embedding_", "coef_", "intercept_", "cluster_centers_", "components_")

#: Constructor arguments the generic probe passes (session 2, 2026-09-28): a
#: name's REQUIRED parameters, or an option without which its default path
#: cannot answer the probe (prediction_data for predict on the density
#: clusterers, novelty for LocalOutlierFactor.predict, an int random_state
#: where None is refused by name). Values are callables of (ml, F) so an
#: estimator argument is built fresh per probe.
CTOR = {
    "AvgPool1d": lambda ml, F: dict(kernel_size=2),
    "MaxPool1d": lambda ml, F: dict(kernel_size=2),
    "AvgPool2d": lambda ml, F: dict(kernel_size=2),
    "MaxPool2d": lambda ml, F: dict(kernel_size=2),
    "BatchNorm1d": lambda ml, F: dict(num_features=3),
    "BatchNorm2d": lambda ml, F: dict(num_features=3),
    "Conv1d": lambda ml, F: dict(in_channels=3, out_channels=2, kernel_size=3),
    "Conv2d": lambda ml, F: dict(in_channels=3, out_channels=2, kernel_size=3),
    "BasicBlock": lambda ml, F: dict(inplanes=3, planes=3),
    "CNNClassifier": lambda ml, F: dict(input_shape=(3, 8, 8)),
    "IVFPQIndex": lambda ml, F: dict(n_lists=4, n_probes=2),
    "IVFSQIndex": lambda ml, F: dict(n_lists=4, n_probes=2),
    "IVFRaBitQIndex": lambda ml, F: dict(n_lists=4, n_probes=2),
    "CCA": lambda ml, F: dict(n_components=1),
    "PLSCanonical": lambda ml, F: dict(n_components=1),
    "GaussianRandomProjection": lambda ml, F: dict(n_components=3, random_state=0),
    "SparseRandomProjection": lambda ml, F: dict(n_components=3, random_state=0),
    "PolynomialCountSketch": lambda ml, F: dict(random_state=0),
    "SkewedChi2Sampler": lambda ml, F: dict(random_state=0),
    "SGDClassifier": lambda ml, F: dict(loss="log_loss"),
    "LocalOutlierFactor": lambda ml, F: dict(novelty=True),
    "AgglomerativeClustering": lambda ml, F: dict(prediction_data=True),
    "DBSCAN": lambda ml, F: dict(prediction_data=True),
    "SpectralClustering": lambda ml, F: dict(prediction_data=True),
    "ExperimentalTwoLevelFeatureFreq": lambda ml, F: dict(sources=(0, 5)),
    "SelectKBest": lambda ml, F: dict(k=3),
    "MultiOutputRegressor": lambda ml, F: dict(estimator=ml.Ridge()),
    "MultiOutputClassifier": lambda ml, F: dict(estimator=ml.LogisticRegression()),
    "OneVsRestClassifier": lambda ml, F: dict(estimator=ml.LogisticRegression()),
    "RFE": lambda ml, F: dict(estimator=ml.LinearRegression()),
    "VotingRegressor": lambda ml, F: dict(estimators=[("r", ml.Ridge()), ("o", ml.LinearRegression())]),
    "VotingClassifier": lambda ml, F: dict(estimators=[("l", ml.LogisticRegression()),
                                                       ("k", ml.KNeighborsClassifier())]),
    "StackingRegressor": lambda ml, F: dict(estimators=[("r", ml.Ridge()), ("o", ml.LinearRegression())]),
    "StackingClassifier": lambda ml, F: dict(estimators=[("l", ml.LogisticRegression()),
                                                         ("k", ml.KNeighborsClassifier())]),
    "SparseCoder": lambda ml, F: dict(dictionary=F["dictionary"]),
}

#: Input shapes other than (rows, features) (session 2): image ops take
#: (N, C, H, W), the 1-D ops (N, C, L), the recurrent estimators
#: (samples, timesteps, features), IsotonicRegression a 1-D X.
INPUTS = {
    **{n: "img" for n in ("AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "AvgPool2d", "MaxPool2d",
                          "BatchNorm2d", "Conv2d", "Dropout2d", "BasicBlock", "CNNClassifier")},
    **{n: "seq1d" for n in ("AvgPool1d", "MaxPool1d", "Conv1d", "BatchNorm1d")},
    **{n: "rnn" for n in ("LSTMClassifier", "LSTMRegressor", "GRUClassifier", "GRURegressor",
                          "RNNClassifier", "RNNRegressor")},
    "IsotonicRegression": "x1",
    "PoissonRegressor": "ypos", "GammaRegressor": "ypos",
    "OneHotEncoder": "cat", "OrdinalEncoder": "cat",
}

#: Inference methods a probe skips because the default fit cannot answer
#: them by DESIGN (predict_proba of an RMSE booster); the rest are probed.
SKIP_METHODS = {"GradientBoosting": ("predict_proba",)}


# ------------------------------------------------------------------ fixtures
def fixtures():
    import numpy as np
    rng = np.random.default_rng(20260927)
    X = rng.standard_normal((160, 6)).astype(np.float32)
    X[:, 2] *= 4.0
    X[:, 5] = np.round(X[:, 5])              # ties
    X[::17, 1] = 0.0
    w = np.array([1.5, -2.0, 0.5, 0.0, 3.0, -1.0], np.float32)
    yr = (X @ w + 0.1 * rng.standard_normal(160)).astype(np.float32)
    yc = (yr > np.median(yr)).astype(np.int32)
    Xh = rng.standard_normal((40, 6)).astype(np.float32)
    t = np.arange(96, dtype=np.float64)
    series = (10 + 0.1 * t + np.sin(t * 2 * np.pi / 12) + 0.05 * rng.standard_normal(96)).astype(np.float64)
    d = rng.standard_normal((4, 6)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    img = rng.standard_normal((8, 3, 8, 8)).astype(np.float32)
    seq1d = rng.standard_normal((8, 3, 16)).astype(np.float32)
    rnn = rng.standard_normal((40, 5, 6)).astype(np.float32)
    cat = np.round(np.abs(X[:, :3])).astype(np.int64)
    return dict(X=X, yr=yr, yc=yc, Xh=Xh, Xp=np.abs(X) + np.float32(0.01),
                Xhp=np.abs(Xh) + np.float32(0.01), series=series, rng=rng, dictionary=d,
                img=img, img_y=(np.arange(8) % 2).astype(np.int32), seq1d=seq1d,
                rnn=rnn, rnn_yc=(rnn[:, -1, 0] > 0).astype(np.int32), rnn_yr=rnn[:, -1, 0].copy(),
                cat=cat, cat_h=cat[:40].copy())


def digest(value):
    """sha256 over the bytes of a value's arrays (in order), or None."""
    import numpy as np
    h = hashlib.sha256()
    seen = [0]

    def feed(v, depth=0):
        if depth > 4 or v is None:
            return
        if isinstance(v, (bytes, bytearray)):
            h.update(bytes(v)); seen[0] += 1; return
        if isinstance(v, (str,)):
            h.update(v.encode()); seen[0] += 1; return
        if isinstance(v, (tuple, list)) and not (v and isinstance(v[0], (int, float)) and len(v) > 0):
            for item in v:
                feed(item, depth + 1)
            return
        if isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); feed(v[k], depth + 1)
            return
        try:
            a = np.asarray(v)
        except Exception:
            a = None
        if a is not None and a.dtype != object:
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes()); seen[0] += 1
            return
        for attr in ("statistic", "pvalue", "value", "estimate", "confidence_interval", "labels_"):
            if hasattr(v, attr):
                feed(getattr(v, attr), depth + 1)
    feed(value)
    return h.hexdigest()[:16] if seen[0] else None


def classify_exc(exc):
    text = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:240]
    kind = "REFUSED" if isinstance(exc, NotImplementedError) or "no CPU implementation" in text \
        or "CPU-only install" in text or "no cpu" in text.lower() else "ERROR"
    return kind, f"{type(exc).__name__}: {text}"


# ------------------------------------------------------------------ recipes
#: name -> callable(ml, F) returning an ordered dict of stage -> callable().
#: A recipe exists only where the generic probe cannot guess the call.
RECIPES = {}


def recipe(*names):
    def deco(fn):
        for n in names:
            RECIPES[n] = fn
        return fn
    return deco


_LABEL_PAIR = ("rand_score", "adjusted_rand_score", "mutual_info_score", "fowlkes_mallows_score",
               "homogeneity_score", "completeness_score", "v_measure_score",
               "homogeneity_completeness_v_measure", "accuracy_score", "precision_score",
               "recall_score", "f1_score", "confusion_matrix")


def _pred_labels(F):
    import numpy as np
    p = F["yc"].copy()
    p[::7] = 1 - p[::7]
    return p


@recipe(*("metrics." + n for n in _LABEL_PAIR))
def _r_label_pair(ml, F):
    fn = resolve(ml, _r_label_pair.name)
    return {"call": lambda: fn(F["yc"], _pred_labels(F))}


@recipe("metrics.entropy")
def _r_entropy(ml, F):
    return {"call": lambda: ml.metrics.entropy(F["yc"])}


@recipe("metrics.r2_score", "metrics.mean_squared_error", "metrics.mean_absolute_error",
        "metrics.root_mean_squared_error")
def _r_reg_pair(ml, F):
    pred = F["yr"] + F["X"][:, 0] * 0.25
    return {"call": lambda: getattr(ml.metrics, _r_reg_pair.name.split(".")[1])(F["yr"], pred)}


@recipe("metrics.kl_divergence")
def _r_kl(ml, F):
    import numpy as np
    P = np.abs(F["X"][:, 0]) + 0.1
    Q = np.abs(F["X"][:, 1]) + 0.1
    P = (P / P.sum()).astype(np.float32)
    Q = (Q / Q.sum()).astype(np.float32)
    return {"call": lambda: ml.metrics.kl_divergence(P, Q)}


@recipe("metrics.silhouette_score", "metrics.silhouette_samples")
def _r_sil(ml, F):
    return {"call": lambda: getattr(ml.metrics, _r_sil.name.split(".")[1])(F["X"], F["yc"])}


@recipe("metrics.trustworthiness")
def _r_trust(ml, F):
    return {"call": lambda: ml.metrics.trustworthiness(F["X"], F["X"][:, :2].copy(), n_neighbors=5)}


@recipe("metrics.log_loss", "metrics.roc_auc_score", "metrics.precision_recall_curve")
def _r_prob(ml, F):
    import numpy as np
    s = (1.0 / (1.0 + np.exp(-F["yr"] / 4.0))).astype(np.float32)
    return {"call": lambda: getattr(ml.metrics, _r_prob.name.split(".")[1])(F["yc"], s)}


@recipe("linalg.matmul", "matmul")
def _r_matmul(ml, F):
    fn = resolve(ml, _r_matmul.name)
    return {"call": lambda: fn(F["X"], F["Xh"].T.copy())}


@recipe("linalg.matmul_bf16")
def _r_matmul_bf16(ml, F):
    # the bf16 profile takes bf16 BITS (to_bf16's uint16 buffer)
    return {"call": lambda: ml.linalg.matmul_bf16(ml.linalg.to_bf16(F["X"]),
                                                  ml.linalg.to_bf16(F["Xh"].T.copy()))}


@recipe("linalg.matmul_int8")
def _r_matmul_int8(ml, F):
    return {"call": lambda: ml.linalg.matmul_int8(F["X"], F["Xh"].copy())}


@recipe("linalg.qr", "linalg.svdvals")
def _r_qr(ml, F):
    fn = resolve(ml, _r_qr.name)
    return {"call": lambda: fn(F["X"][:24].copy())}


@recipe("linalg.eigh")
def _r_eigh(ml, F):
    a = (F["X"].T @ F["X"]).astype(F["X"].dtype)
    return {"call": lambda: ml.linalg.eigh(a)}


@recipe("linalg.Cholesky", "Cholesky")
def _r_chol(ml, F):
    import numpy as np
    a = (F["X"].T @ F["X"] + 6 * np.eye(6)).astype(np.float32)
    st = {}

    def fit():
        st["c"] = ml.Cholesky()
        return st["c"].fit(a)
    return {"fit": fit, "solve": lambda: st["c"].solve(F["X"][:6].T.copy())}


@recipe("linalg.to_bf16", "linalg.quantize_int8")
def _r_quant(ml, F):
    fn = resolve(ml, _r_quant.name)
    return {"call": lambda: fn(F["X"])}


@recipe("linalg.from_bf16")
def _r_from_bf16(ml, F):
    return {"call": lambda: ml.linalg.from_bf16(ml.linalg.to_bf16(F["X"]))}


@recipe("linalg.dequantize_int8")
def _r_dequant(ml, F):
    return {"call": lambda: ml.linalg.dequantize_int8(*ml.linalg.quantize_int8(F["X"]))}


@recipe("resample.bootstrap")
def _r_boot(ml, F):
    return {"call": lambda: ml.resample.bootstrap(F["yr"], n_resamples=199)}


@recipe("resample.permutation_test")
def _r_perm(ml, F):
    return {"call": lambda: ml.resample.permutation_test(F["yr"][:80], F["yr"][80:], n_resamples=199)}


@recipe("resample.monte_carlo_integrate")
def _r_mc(ml, F):
    return {"call": lambda: ml.resample.monte_carlo_integrate("product", [0.0, 0.0], [1.0, 2.0], 4096)}


@recipe("hdbscan.approximate_predict", "hdbscan.membership_vector", "hdbscan.all_points_membership_vectors")
def _r_hdb_fn(ml, F):
    st = {}
    fn = resolve(ml, _r_hdb_fn.name)

    def fit():
        st["c"] = ml.HDBSCAN(min_cluster_size=8, prediction_data=True)
        return st["c"].fit(F["X"]).labels_
    call = (lambda: fn(st["c"])) if _r_hdb_fn.name.endswith("all_points_membership_vectors") \
        else (lambda: fn(st["c"], F["Xh"]))
    return {"fit": fit, "call": call}


@recipe("kpss_test")
def _r_kpss(ml, F):
    return {"call": lambda: ml.kpss_test(F["series"][None, :].copy())}


@recipe("select_d")
def _r_select_d(ml, F):
    return {"call": lambda: ml.select_d(F["series"][None, :].copy())}


@recipe("ExponentialSmoothing")
def _r_es(ml, F):
    st = {}

    def fit():
        st["m"] = ml.ExponentialSmoothing(F["series"], seasonal_periods=12)
        st["m"].fit()
        return st["m"].forecast(1)
    return {"fit": fit, "forecast": lambda: st["m"].forecast(6),
            "predict": lambda: st["m"].predict(0, 100)}


@recipe("ARIMA")
def _r_arima(ml, F):
    st = {}

    def fit():
        st["m"] = ml.ARIMA(order=(1, 1, 1))
        return st["m"].fit(F["series"])
    return {"fit": fit, "forecast": lambda: st["m"].forecast(6),
            "predict": lambda: st["m"].predict(0, 100)}


@recipe("Embedding")
def _r_emb(ml, F):
    import numpy as np
    ids = (np.arange(40) * 7 % 13).astype(np.int64)
    st = {}

    def fit():
        w = F["rng"].standard_normal((13, 8)).astype(np.float32)
        st["e"] = ml.Embedding.from_pretrained(w)
        return st["e"].forward(ids)
    return {"forward": fit,
            "backward": lambda: st["e"].backward(ids, np.ones((40, 8), np.float32))}


@recipe("IVFIndex")
def _r_ivf(ml, F):
    st = {}

    def fit():
        st["i"] = ml.IVFIndex(n_lists=4, n_probes=2, n_neighbors=5)
        return st["i"].fit(F["X"])
    return {"fit": fit, "search": lambda: st["i"].search(F["Xh"])}


@recipe("manifold.spectral_embedding")
def _r_spec(ml, F):
    return {"call": lambda: ml.manifold.spectral_embedding(F["X"], n_components=2, random_state=0,
                                                           n_neighbors=10)}


@recipe("cross_val_score", "model_selection.cross_val_score")
def _r_cvs(ml, F):
    fn = resolve(ml, _r_cvs.name)
    return {"call": lambda: fn(ml.Ridge(), F["X"], F["yr"], cv=3)}


@recipe("Theta", "OptimizedTheta", "DynamicTheta", "DynamicOptimizedTheta", "AutoTheta",
        "CrostonClassic", "CrostonOptimized", "CrostonSBA", "ETS", "DampedETS")
def _r_statsforecast(ml, F):
    # statsforecast-shaped: fit(y), predict(h) (session 2; the generic
    # probe's predict(Xh) was a probe bug, not a CPU gap)
    st = {}
    cls = resolve(ml, _r_statsforecast.name)
    y = F["series"].astype("float32") if "Croston" not in _r_statsforecast.name \
        else (abs(F["series"] - 10.0) * (F["series"] > 11.0)).astype("float32")

    def fit():
        st["m"] = cls()
        st["m"].fit(y)
        return st["m"].predict(6)
    return {"fit": fit, "predict": lambda: st["m"].predict(12)}


@recipe("AutoARIMA")
def _r_autoarima(ml, F):
    st = {}

    def fit():
        st["m"] = ml.AutoARIMA(F["series"])
        return st["m"].fit()
    return {"fit": fit, "forecast": lambda: st["m"].forecast(6)}


@recipe("STL")
def _r_stl(ml, F):
    return {"fit": lambda: ml.STL(F["series"], period=12).fit()}


@recipe("VAR")
def _r_var(ml, F):
    import numpy as np
    endog = np.stack([F["series"], np.roll(F["series"], 3) * 0.5 + 1.0], axis=1)
    st = {}

    def fit():
        st["m"] = ml.VAR(endog)
        st["r"] = st["m"].fit(maxlags=2)
        return st["r"]
    return {"fit": fit}


@recipe("GARCH")
def _r_garch(ml, F):
    import numpy as np
    r = np.diff(np.log(F["series"])).astype(np.float64)
    st = {}

    def fit():
        st["m"] = ml.GARCH()
        return st["m"].fit(r)
    return {"fit": fit, "forecast": lambda: st["m"].forecast(horizon=5)}


@recipe("ProphetForecaster")
def _r_prophet(ml, F):
    import numpy as np
    t = np.arange(len(F["series"]), dtype=np.float64)
    st = {}

    def fit():
        st["m"] = ml.ProphetForecaster()
        return st["m"].fit(t, F["series"])
    return {"fit": fit, "predict": lambda: st["m"].predict(t + 12.0)}


@recipe("PageRank", "Louvain")
def _r_graph(ml, F):
    import numpy as np
    x = F["X"][:40]
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    a = (d < np.quantile(d, 0.15)).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    a = np.maximum(a, a.T)
    cls = resolve(ml, _r_graph.name)
    return {"fit": lambda: cls().fit(a)}


@recipe("LabelEncoder")
def _r_labelenc(ml, F):
    st = {}

    def fit():
        st["e"] = ml.LabelEncoder().fit(F["yc"])
        return st["e"].classes_
    return {"fit": fit, "transform": lambda: st["e"].transform(F["yc"]),
            "inverse_transform": lambda: st["e"].inverse_transform(st["e"].transform(F["yc"]))}


@recipe("LabelBinarizer")
def _r_labelbin(ml, F):
    st = {}

    def fit():
        st["e"] = ml.LabelBinarizer().fit(F["yc"])
        return st["e"].classes_
    return {"fit": fit, "transform": lambda: st["e"].transform(F["yc"]),
            "inverse_transform": lambda: st["e"].inverse_transform(st["e"].transform(F["yc"]))}


@recipe("parallel_model_selection.cross_val_score")
def _r_pcvs(ml, F):
    return {"call": lambda: ml.parallel_model_selection.cross_val_score(
        ml.Ridge(), F["X"], F["yr"], cv=3, devices=[0])}


def _named_recipes():
    """Bind each recipe to the name it was looked up under (several names
    share one recipe body)."""
    out = {}
    for n, fn in RECIPES.items():
        def make(ml, F, n=n, fn=fn):
            fn.name = n
            return fn(ml, F)
        out[n] = make
    return out


def _generic(ml, name, F):
    import numpy as np
    obj = resolve(ml, name)
    if not inspect.isclass(obj):
        return None
    bare = name.split(".")[-1]
    nonneg = any(h in name for h in NONNEG_HINTS)
    X, Xh = (F["Xp"], F["Xhp"]) if nonneg else (F["X"], F["Xh"])
    yc, yr = F["yc"], F["yr"]
    kind = INPUTS.get(bare)
    if kind == "img":
        X, Xh, yc, yr = F["img"], F["img"], F["img_y"], F["img_y"].astype(np.float32)
    elif kind == "seq1d":
        X, Xh = F["seq1d"], F["seq1d"]
    elif kind == "rnn":
        X, Xh, yc, yr = F["rnn"], F["rnn"], F["rnn_yc"], F["rnn_yr"]
    elif kind == "x1":
        X, Xh = F["X"][:, 0].copy(), F["Xh"][:, 0].copy()
    elif kind == "ypos":
        yr = (np.exp(F["yr"] / 8.0) + np.float32(0.1)).astype(np.float32)
    elif kind == "cat":
        X, Xh = F["cat"], F["cat_h"]
    if any(h in name for h in CLASSIFIER_HINTS):
        targets = [yc, yr, None]
    elif any(h in name for h in REGRESSOR_HINTS):
        targets = [yr, yc, None]
    else:
        targets = [None, yr, yc]
    ctor = CTOR.get(bare)
    state = {}

    def fit():
        # The FIRST error that is not a call-signature mismatch is the one
        # reported: a TypeError from inside a fit must not be masked by the
        # next target shape's "missing argument".
        first = None
        for y in targets:
            est = obj(**ctor(ml, F)) if ctor else obj()
            try:
                out = est.fit(X) if y is None else est.fit(X, y)
            except TypeError as exc:
                if first is None or "positional argument" in str(first):
                    first = exc
                continue
            state["est"] = est
            got = [getattr(est, a) for a in FITTED_ATTRS if hasattr(est, a)]
            return got or out
        raise first

    def infer(m):
        est = state["est"]
        if m == "inverse_transform" and callable(getattr(est, "transform", None)):
            # an inverse takes the transform's output, never raw features
            return est.inverse_transform(est.transform(Xh))
        return getattr(est, m)(Xh)

    stages = {"fit": fit}
    for m in INFER_METHODS:
        if m in SKIP_METHODS.get(bare, ()):
            continue
        if callable(getattr(obj, m, None)):
            stages[m] = (lambda m=m: infer(m))
    if not callable(getattr(obj, "fit", None)):
        if callable(getattr(obj, "fit_predict", None)):
            stages = {"fit_predict": lambda: obj().fit_predict(X)}
        elif callable(getattr(obj, "fit_transform", None)):
            stages = {"fit_transform": lambda: obj().fit_transform(X)}
        else:
            return None
    return stages


def resolve(ml, name):
    obj = ml
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def public_names(ml):
    names = list(ml.__all__)
    for mod in SUBMODULES:
        sub = getattr(ml, mod, None)
        for n in getattr(sub, "__all__", ()):
            names.append(f"{mod}.{n}")
    return names


# ------------------------------------------------------------------ child
def run_child(names, out):
    import numpy as np  # noqa: F401
    import mojolearn as ml
    from mojolearn import _backend
    F = fixtures()
    recipes = _named_recipes()
    with open(out, "a") as fh:
        for name in names:
            fh.write(json.dumps({"start": name}) + "\n"); fh.flush()
            row = {"name": name, "vendor": _backend.vendor(),
                   "cpu_only": _backend._CPU_ONLY is not None, "stages": {}}
            t0 = time.time()
            try:
                if name in NOT_ALGORITHMS or name in {m for m in SUBMODULES} or \
                        inspect.ismodule(resolve(ml, name)):
                    row["kind"] = "not an algorithm"
                    stages = {}
                else:
                    make = recipes.get(name)
                    stages = make(ml, F) if make else _generic(ml, name, F)
                    row["kind"] = "recipe" if make else "generic"
                    if stages is None:
                        row["kind"] = "no probe"
                        stages = {}
            except Exception as exc:
                kind, text = classify_exc(exc)
                row["stages"]["setup"] = {"status": kind, "why": text}
                stages = {}
            for stage, call in stages.items():
                try:
                    value = call()
                    row["stages"][stage] = {"status": "OK", "sha": digest(value)}
                except Exception as exc:
                    kind, text = classify_exc(exc)
                    row["stages"][stage] = {"status": kind, "why": text,
                                            "where": traceback.format_exc().strip().splitlines()[-3][:200]}
                    if stage in ("fit", "setup"):
                        break
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row) + "\n"); fh.flush()


def cpu_env():
    sys.path.insert(0, str(ROOT / "tools"))
    import algos_lane_check as alc
    return alc.arm_env("cpu")


def gpu_env():
    sys.path.insert(0, str(ROOT / "tools"))
    import algos_lane_check as alc
    return alc.arm_env("gpu")


def record(args):
    env = cpu_env() if args.cpu else gpu_env()
    env.setdefault("PYTHONPATH", str(ROOT / "python"))
    if args.names:
        names = args.names.split(",")
    else:
        code = ("import json, sys; sys.path.insert(0, %r); import cpu_path_audit as a, mojolearn as ml; "
                "print(json.dumps(a.public_names(ml)))" % str(ROOT / "tools"))
        names = json.loads(subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                          capture_output=True, text=True).stdout.strip().splitlines()[-1])
    out = Path(args.out)
    if out.exists() and not args.resume:
        out.unlink()
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            if "name" in rec:
                done.add(rec["name"])
    todo = [n for n in names if n not in done]
    while todo:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_child", "--out", str(out),
                                 "--names", ",".join(todo)], env=env)
        # A STALLED NAME IS A RESULT, NOT A HANG OF THE AUDIT: when the
        # record has not grown for --stall seconds, the child is killed and
        # the name it started reads TIMEOUT.
        size, since, stalled = -1, time.time(), False
        while proc.poll() is None:
            time.sleep(2)
            now = out.stat().st_size if out.exists() else 0
            if now != size:
                size, since = now, time.time()
            elif time.time() - since > args.stall:
                proc.kill()
                proc.wait()
                stalled = True
        started, finished = None, set()
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            if "start" in rec:
                started = rec["start"]
            else:
                finished.add(rec["name"])
        rest = [n for n in todo if n not in finished]
        if not rest:
            break
        crashed = started if started in rest else rest[0]
        status, why = (("TIMEOUT", f"no progress for {args.stall}s; killed") if stalled
                       else ("CRASH", f"child exited {proc.returncode}"))
        with open(out, "a") as fh:
            fh.write(json.dumps({"name": crashed, "kind": "crash", "stages": {
                "process": {"status": status, "why": why}}}) + "\n")
        todo = [n for n in rest if n != crashed]
    return 0


# ------------------------------------------------------------------ diff
def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if "name" in rec:
            rows[rec["name"]] = rec
    return rows


def summarize(g, c):
    """(gpu, cpu, bits) summaries of one name."""
    def st(rec):
        if rec is None:
            return "MISSING"
        if rec.get("kind") in ("not an algorithm", "no probe"):
            return rec["kind"].upper()
        stages = rec.get("stages", {})
        bad = [f"{k} {v['status']}" for k, v in stages.items() if v["status"] != "OK"]
        return "OK " + "+".join(stages) if not bad else "; ".join(bad)
    bits = ""
    if g and c:
        gs, cs = g.get("stages", {}), c.get("stages", {})
        both = [k for k in gs if gs[k]["status"] == "OK" and cs.get(k, {}).get("status") == "OK"
                and gs[k].get("sha") and cs[k].get("sha")]
        if both:
            diff = [k for k in both if gs[k]["sha"] != cs[k]["sha"]]
            bits = "EQUAL" if not diff else "DIFFER: " + ",".join(diff)
    return st(g), st(c), bits


def diff(args):
    g, c = load(args.gpu), load(args.cpu)
    names = list(dict.fromkeys(list(g) + list(c)))
    rows = []
    for n in names:
        gs, cs, bits = summarize(g.get(n), c.get(n))
        why = ""
        rec = c.get(n) or {}
        for k, v in rec.get("stages", {}).items():
            if v["status"] != "OK":
                why = v.get("why", "")
                break
        rows.append((n, gs, cs, bits, why))
    if args.markdown:
        print("| name | GPU | CPU | CPU bits vs GPU | first CPU failure |")
        print("|---|---|---|---|---|")
        for r in rows:
            print("| " + " | ".join(x.replace("|", "/") for x in r) + " |")
    else:
        for r in rows:
            print("\t".join(r))
    return 0


def _stage_text(rec):
    """(fit, infer) for one probe record: the fit-like stage and the rest."""
    if rec is None:
        return "-", "-"
    if rec.get("kind") in ("not an algorithm", "no probe"):
        return "-", "-"
    stages = rec.get("stages", {})
    if "process" in stages:
        return stages["process"]["status"], "-"
    fitlike = [k for k in stages if k in ("fit", "fit_predict", "fit_transform", "setup", "forward")]
    other = [k for k in stages if k not in fitlike]

    def sayit(keys):
        if not keys:
            return "-"
        bad = [f"{k} {stages[k]['status']}" for k in keys if stages[k]["status"] != "OK"]
        return "OK" if not bad else ", ".join(bad)
    if not fitlike and other:
        return sayit(other), "-"
    return sayit(fitlike), sayit(other)


def table(args):
    """The markdown audit table: one row per public name, from the two probe
    records, the selector's lane -> names map and the host manifest."""
    sys.path.insert(0, str(ROOT / "python" / "mojolearn"))
    import host_surface as hs
    g, c = load(args.gpu), load(args.cpu)
    lane_names = json.loads(Path(args.lane_names).read_text())
    public = set(hs.public_reference_lanes())
    pending = hs.PUBLIC_PENDING_LANES
    covered = set()
    for fam in hs.FAMILIES:
        covered |= set(fam.get("training_lanes", ())) | set(fam.get("inference_lanes", ()))
    gaps = {}
    for line in (Path(args.gaps).read_text().splitlines() if args.gaps else ()):
        if line.strip() and not line.startswith("#"):
            name, _, why = line.partition("\t")
            gaps[name.strip()] = why.strip()
    print("| name | GPU probe | CPU fit | CPU infer | CPU = GPU bits | lanes (CPU column) | status | why |")
    print("|---|---|---|---|---|---|---|---|")
    for name in list(dict.fromkeys(list(g) + list(c))):
        gr, cr = g.get(name), c.get(name)
        if (cr or gr or {}).get("kind") == "not an algorithm":
            continue
        bare = name.split(".")[-1]
        lanes = sorted(l for l, names in lane_names.items() if bare in names and not l.startswith("par-"))
        cov = [l for l in lanes if l in covered or l in hs.PUBLIC_HOST_ONLY_LANES]
        adm = [l for l in cov if l in public]
        pend = sorted({pending[l] for l in cov if l in pending})
        gs = summarize(gr, cr)
        fit, infer = _stage_text(cr)
        gfit, ginfer = _stage_text(gr)
        gtxt = "OK" if gfit == "OK" and ginfer in ("OK", "-") else (gfit if gfit != "OK" else ginfer)
        if name in gaps:
            status = gaps[name]
        elif adm:
            status = "ADMITTED"
        elif cov and pend:
            status = "PENDING (" + ", ".join(pend) + ")"
        elif cov:
            status = "COVERED, not admitted"
        else:
            status = "NO LANE"
        why = ""
        if cr:
            for k, v in cr.get("stages", {}).items():
                if v["status"] != "OK":
                    why = v.get("why", "")[:140]
                    break
        lane_txt = (f"{len(cov)}: " + ", ".join(cov[:3]) + (" ..." if len(cov) > 3 else "")) if cov else "none"
        row = (name, gtxt, fit, infer, gs[2] or "-", lane_txt, status, why)
        print("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--names")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--stall", type=int, default=300,
                   help="seconds without progress before a name reads TIMEOUT")
    ch = sub.add_parser("_child")
    ch.add_argument("--out", required=True)
    ch.add_argument("--names", required=True)
    d = sub.add_parser("diff")
    d.add_argument("gpu")
    d.add_argument("cpu")
    d.add_argument("--markdown", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("gpu")
    t.add_argument("cpu")
    t.add_argument("--lane-names", required=True,
                   help="lane -> touched names JSON (lane_select._seed_names per lane)")
    t.add_argument("--gaps", help="TSV name<TAB>status overriding the derived status")
    args = ap.parse_args(argv)
    if args.cmd == "table":
        return table(args)
    if args.cmd == "_child":
        run_child(args.names.split(","), args.out)
        return 0
    return record(args) if args.cmd == "record" else diff(args)


if __name__ == "__main__":
    sys.exit(main())
