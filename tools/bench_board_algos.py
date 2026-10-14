#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `algos` family of tools/bench_board.py: every algorithm of the
algorithm expansion (the nine lane tables, their Additions and the long
tail) against its opponents on the same
box, interleaved round by round, quality beside every time.

    python3 tools/bench_board_algos.py prep --data DIR --lanes sgd-clf,ivf-pq --datasets taxi
    python3 tools/bench_board_algos.py race --lane sgd-clf --dataset taxi --data DIR \\
        --arms ours,sklearn-cpu,cuml-gpu --rounds 5 --out DIR --work DIR
    python3 tools/bench_board_algos.py table        # the race table, as JSON

It speaks tools/bench_board_more.py's worker protocol (one persistent worker
per arm, one warm-up then `--rounds` timed rounds, arm order rotated every
round, outputs saved after the clock for a float64 NumPy quality pass in the
conductor) with two additions:

  * TRAINING AND INFERENCE, both timed. A worker's round reports `ms` (the
    lane's fit: `fit`, `fit_transform`, a build, a train step) and, where the
    lane has one, `infer_ms` (predict / transform / search / forward on the
    held-out rows with the model of that round). The race JSON carries the
    inference timings as an `infer` record in tools/bench_board_infer.py's
    classical shape, so bench_board.py renders them as inference cells.
  * SKIPPED: not built yet. Our side calls the public class by name
    (`mojolearn.<Name>`, the first of the lane's candidate names that the
    installed wheel exports). While a lane has not merged its class the `ours`
    arms answer `{"event": "skipped"}` and the board prints
    "SKIPPED: not built yet", never an error; the opponents still race. A class
    that IS exported but lacks a method the board calls refuses by name
    ("CONTRACT: ...").
  * OUR SEPARATE UPLOAD CLOCK (2026-10-08). After each round's clocks close,
    OUR arm (ours, ours-fast) uploads the host arrays its timed call was handed
    once more, timed on its own (`_ours_upload_probe`), and its span carries
    `upload_ms_separate` (+ bytes, inputs, scope) and the derived
    `fit_minus_upload_ms`: a kernel-only reading beside the whole-operation
    one, for ratios against GPU arms whose inputs went up before their clock.
    The scored clock is unchanged; opponent arms are never probed.

OUR SIDE'S CONTRACT (what the board calls; the lanes' classes are
scikit-learn shaped unless a line below says otherwise)
----------------------------------------------------------------------------
  estimators   Cls(**params).fit(X[, y]); predict / predict_proba / transform /
               fit_predict / fit_transform as scikit-learn names them; fitted
               attributes with scikit-learn's names (components_, means_, ...)
  forecasters  Cls(**params).fit(Y) with Y (n_series, n_obs) float32, then
               .predict(h)["mean"] -> (n_series, h)   (statsforecast's shape:
               Theta, CrostonClassic, ETS); AutoARIMA is cuML's shape,
               Cls(Y).search(...); .fit(); .forecast(h); GARCH is arch's
               shape, Cls(p, q, mean, dist).fit(Y, horizon=h), .forecast(h)
               (variances), .loglikelihood_ (per series)
  graph        Cls(**params).fit(indptr, indices) of a symmetric CSR graph,
               then .scores_ (PageRank) or .labels_ (components, Louvain)
  ANN          Cls(**params).fit(index).search(queries) -> (dist, ind), the
               IVFIndex shape
  layers       Cls(**ctor); .load_state_dict({torch name: ndarray}) when the
               board compares outputs; y = layer(x) (or .forward(x));
               dx = layer.backward(dy) for the training column
  optimizers   Cls([param ndarray], **hyper).step([grad ndarray]) in place
  ALS          Cls(**params).fit(csr user x item counts); .user_factors_,
               .item_factors_
  explainers   Cls(model, ...).shap_values(X)   (the shap package's shape)
  linalg       mojolearn.linalg.<name>(...) as numpy/scipy name it
  functions    mojolearn.resample.bootstrap / permutation_test / resample and
               mojolearn.cross_val_score, SciPy's and scikit-learn's call shapes

Every lane's settings, datasets, rows, quality metric and each mismatch that
cannot be avoided are in LANES below; tools/bench_board.py copies them into
every cell's `settings`. The algorithm lanes do not touch this file (lane
bench owns it); a class exported under another name is added to its
candidates here.

DATA (prep, untimed, once per box and row cap; nothing is downloaded)
--------------------------------------------------------------------
taxi and Istella-S through the classical2 prep (R2 keys
gbm-bench/taxi/taxi_speed.npz, gbm-bench/istella/istella_speed.npz) plus the
blocks below; text from the R2 corpora corpus/enwik8/input.txt and
corpus/pile_github/input.txt. Blocks with "synthetic" data are built from
seed 7 where the workload has no dataset kind (dense linear algebra, a layer's
input tensor, an optimizer's gradients) or as the second kind beside a real
one (time series). The R2 store holds NO image set (CIFAR/ImageNet) and NO
implicit-feedback set (MovieLens/Last.fm): the CNN layers run on seeded
tensors and ALS on taxi zone-hour x dropoff-zone trip counts and text counts.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shlex
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SEED = 7
FAMILY = "algos"
TAB = ("taxi", "istella")
KNN_K = 10

# shapes (the board; `--rows N` caps them for a smoke)
FIT_ROWS = 1_000_000          # the classical2 cls/reg blocks' fit rows
EVAL_ROWS = 100_000
SUB = {"quad": 10_000, "cubic": 3_000, "knn": 200_000, "mid": 100_000, "small": 20_000,
       "tiny": 5_000}
# Explicit full-workload input/measurement support, only for the tree families
# requested by the Apple FAST experiment programme. Default boards keep their
# historical blocks and subsets. Source added uncompiled/unverified/unmeasured.
AFT_FULL_TREE_LANES = frozenset((
    "decision-tree-clf", "decision-tree-reg", "bagging-clf", "bagging-reg",
    "adaboost-clf", "adaboost-reg", "dart", "dart-reg",
    "random-trees-embedding", "tree-shap",
))
AFT_ALGOS_BOUNDARY = (
    "public constructor/model preparation, fit or explanation, required synchronization, "
    "inference or transform and consumed outputs; dataset loading/CPU input normalization "
    "and startup probe excluded; preparation, fit/explanation and inference timings separate"
)
GRAPH_NODES = 100_000
GRAPH_SMALL = 20_000          # the graph-algorithm races: ours takes a dense adjacency
TS_H = 48                     # held-out hours / points per series
TS_SERIES = 64
TEXT_DOC_BYTES = 2048
TEXT_BINS = 4096
TAXI_CAT_COLUMNS = (0, 3, 4, 5, 6)
TAXI_PU, TAXI_DO, TAXI_HOUR, TAXI_WDAY, TAXI_DAY = 5, 6, 7, 8, 9

CORPUS_KEYS = {"enwik8": "corpus/enwik8/input.txt",
               "pile_github": "corpus/pile_github/input.txt"}

#: The opponent pins of this family, per vendor (installed beside the trees
#: and rapids sets when the family is planned). Versions resolved on the lane's
#: NVIDIA pod on 2026-09-27 (Python 3.11).
PINS_COMMON = ["statsmodels==0.15.0", "statsforecast==2.1.1", "arch==8.0.0", "prophet==1.4.0",
               "networkx==3.6.1", "shap==0.51.0", "implicit==0.7.3",
               "torch-geometric==2.8.0.post1", "gpytorch==1.15.2", "faiss-cpu==1.15.1",
               # the bpe-encode and bpe-train opponent (Hugging Face tokenizers)
               "tokenizers==0.23.2",
               # FAIRNESS FIX (2026-10-13 audit): the moe lane's opponent is transformers' own
               # MixtralSparseMoeBlock (bench_board_neural.TRANSFORMERS_PIN; torchvision for
               # resnet-block comes with torch, tools/bench_board.py DEFAULT_TORCHVISION_SPEC).
               # Results raced before this change were wrong for this lane; apply extra scrutiny.
               "transformers==5.17.0"]
PINS = {"apple": list(PINS_COMMON), "amd": list(PINS_COMMON),
        "nvidia": list(PINS_COMMON)}
#: installed from the rapids index with the pinned cuml/cuvs set
RAPIDS_EXTRA = {"nvidia": ["cugraph-cu12==26.8.0"]}
#: An arm whose library cannot share the board's venv gets its own (clean)
#: venv, created by tools/bench_board.py and passed as --arm-python ARM=PY.
#: implicit-gpu (2026-09-29): the PyPI implicit 0.7.3 GPU extension links
#: CUDA 13 (libcublas.so.13, librmm from rmm-cu13) and was built against rmm
#: 26.4 (26.8 lacks its rmm::bad_alloc typeinfo; 25.12 its device_buffer
#: constructor), while the RAPIDS sets are cu12; cuda-toolkit 13.0 matches
#: the pods' 580 driver. The worker preloads the CUDA 13 libraries through
#: cuda.pathfinder (the extension carries no RUNPATH to them).
ARM_VENVS = {"nvidia": {"implicit-gpu": [
    "implicit==0.7.3", "rmm-cu13==26.4.0", "librmm-cu13==26.4.0",
    "cuda-toolkit[cublas,curand,cudart]==13.0.3", "numpy==2.4.6", "scipy==1.17.1",
    "threadpoolctl==3.7.0"]}}
#: the index the arm venvs install from besides PyPI
ARM_VENV_INDEX = "https://pypi.nvidia.com"


def _preload_cuda_libs(names):
    """Load CUDA libraries from the venv's wheels (cuda.pathfinder) so an
    extension without a RUNPATH to them resolves; a missing loader or library
    leaves it to the arm's own refusal."""
    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib
    except ImportError:
        return
    for n in names:
        try:
            load_nvidia_dynamic_lib(n)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# THE TABLE. One entry per algorithm (a class pair such as SGDClassifier /
# SGDRegressor is two entries). Keys:
#   xlane   the expansion lane that owns the class
#   ours    candidate public names, first exported wins
#   kind    the worker that runs it (est, ts, graph, ann, layer, optim, linalg,
#           als, shap)
#   task    clf reg outlier transform cluster embed semi select impute forecast ...
#   block   the prep block the rows come from; datasets the board datasets
#   sk / cuml / other   opponent specs: "module:Class" or a builder name
#   params  constructor arguments, the SAME for ours and scikit-learn unless
#           sk_params / cuml_params say otherwise (then `mism` says why)
#   sub     stride subsets {array: rows} for the O(n^2) / O(n^3) lanes
#   quality the conductor's metric; infer the timed inference call or None
# ---------------------------------------------------------------------------

def _E(name, **kw):
    """A nested estimator, resolved per library (ours: mojolearn.<name>)."""
    return {"__est__": name, "kw": kw}


_STD = "standardized by the fit rows"
_RAW = "raw (Istella sentinel cleaned), not scaled"
_NONNEG = "standardized, then shifted by the fit rows' column minimum to be nonnegative"

LANES = {}


def _add(slug, **spec):
    assert slug not in LANES, slug
    spec.setdefault("datasets", TAB)
    spec.setdefault("kind", "est")
    spec.setdefault("params", {})
    spec.setdefault("mism", [])
    spec.setdefault("sub", {})
    spec.setdefault("cuml", None)
    spec.setdefault("other", {})
    # FAIRNESS FIX (2026-10-13 audit): an opponent arm that does different work from ours (a
    # different algorithm) raced as if comparable. `refused` maps such an arm to its reason:
    # opponents() leaves it out, lane_config() prints it, and build() refuses it by name if a
    # caller asks for it anyway. Results raced before this change were wrong for those lanes;
    # apply extra scrutiny.
    spec.setdefault("refused", {})
    if isinstance(spec["ours"], str):
        spec["ours"] = (spec["ours"],)
    LANES[slug] = spec


# ---- lane linear ----------------------------------------------------------
# cuML benchmark MBSGDClassifier / MBSGDRegressor cuml_args eta0=0.005, epochs=100
# (tools/bench_board_harness.py), on every arm: max_iter=100 is epochs=100, and
# learning_rate is pinned 'constant' (cuML's default, the schedule eta0 sets)
# FAIRNESS FIX (2026-10-13 audit): ours' batch_size was left at the library default (4096,
# python/mojolearn/_expansion_linear.py) and never set by the harness; cuML's fit_intercept,
# l1_ratio, power_t and n_iter_no_change were left at cuML's defaults; scikit-learn's per-sample
# SGD raced ours' minibatch SGD as if it were the same algorithm, and the stored lane note said
# ours was per-sample. Now every SGD setting is pinned explicitly on ours and cuML (batch_size
# 4096 on both), and the scikit-learn arm is refused (a different algorithm). Results raced
# before this change were wrong for this lane; apply extra scrutiny.
SGD_BATCH = 4096
_SGD = dict(penalty="l2", alpha=1e-4, max_iter=100, tol=None, shuffle=True, random_state=SEED,
            learning_rate="constant", eta0=0.005, batch_size=SGD_BATCH, fit_intercept=True,
            l1_ratio=0.15, n_iter_no_change=5, average=False, early_stopping=False,
            warm_start=False)
_SGD_CUML = dict(penalty="l2", alpha=1e-4, l1_ratio=0.15, fit_intercept=True, epochs=100,
                 batch_size=SGD_BATCH, learning_rate="constant", eta0=0.005, tol=0.0,
                 shuffle=True, n_iter_no_change=5)
_SGD_SK_REFUSED = ("different algorithm: scikit-learn SGD updates per sample and has no "
                   "batch_size; ours and cuML MBSGD take minibatch steps of %d rows" % SGD_BATCH)
_SGD_MISM = ["minibatch SGD of %d rows a step on ours and cuML MBSGD (batch_size pinned on "
             "both); scikit-learn's SGD is per-sample and is refused, not raced" % SGD_BATCH,
             "cuML reads epochs=100 and tol=0.0 (no early stop); ours max_iter=100 and tol=None",
             "intercept: cuML MBSGD centers X and y before the epochs (GLM preProcessData) and "
             "recovers the intercept; ours learns the intercept in the SGD steps"]
_add("sgd-clf", xlane="linear", ours="SGDClassifier", task="clf", block="cls",
     sk="sklearn.linear_model:SGDClassifier",
     params=dict(_SGD, loss="hinge", power_t=0.5, class_weight=None),
     cuml="cuml.linear_model:MBSGDClassifier",
     cuml_params=dict(_SGD_CUML, loss="hinge", power_t=0.5),
     refused={"sklearn-cpu": _SGD_SK_REFUSED}, mism=list(_SGD_MISM))
_add("sgd-reg", xlane="linear", ours="SGDRegressor", task="reg", block="reg",
     sk="sklearn.linear_model:SGDRegressor",
     params=dict(_SGD, loss="squared_error", power_t=0.25),
     cuml="cuml.linear_model:MBSGDRegressor",
     cuml_params=dict(_SGD_CUML, loss="squared_loss", power_t=0.25),
     refused={"sklearn-cpu": _SGD_SK_REFUSED}, mism=list(_SGD_MISM))
# FAIRNESS FIX (2026-10-13 audit): the race declared solver='lbfgs' on both arms, but ours'
# GLMs run Newton-Cholesky with a line search for either solver name
# (python/mojolearn/_expansion_linear.py _GLMBase.fit), so scikit-learn ran L-BFGS against
# ours' Newton. Both arms now take solver='newton-cholesky' (scikit-learn has it since 1.2).
# Results raced before this change were wrong for these lanes; apply extra scrutiny.
GLM_SOLVER = "newton-cholesky"
for _slug, _cls, _kw, _why in (
        ("poisson", "PoissonRegressor", {}, "y = the reg target (taxi fare > 0; Istella grade >= 0)"),
        ("gamma", "GammaRegressor", {}, "y = the reg target + 1 (Gamma needs y > 0; Istella grades start at 0)"),
        ("tweedie", "TweedieRegressor", {"power": 1.5, "link": "log"}, "y = the reg target (>= 0)")):
    _add(_slug, xlane="linear", ours=_cls, task="reg", block="reg", target=_slug,
         sk="sklearn.linear_model:" + _cls,
         params=dict(_kw, alpha=1e-4, max_iter=100, tol=1e-4, solver=GLM_SOLVER),
         notes=[_why, "solver='newton-cholesky' on both arms: ours runs Newton-Cholesky with a "
                "line search whatever the name (before 2026-10-13 scikit-learn raced 'lbfgs')"])
_add("huber", xlane="linear", ours="HuberRegressor", task="reg", block="reg",
     sk="sklearn.linear_model:HuberRegressor",
     params=dict(epsilon=1.35, alpha=1e-4, max_iter=100, tol=1e-5))
_add("bayesian-ridge", xlane="linear", ours="BayesianRidge", task="reg", block="reg",
     sk="sklearn.linear_model:BayesianRidge", params=dict(max_iter=300, tol=1e-3))
_add("ard", xlane="linear", ours="ARDRegression", task="reg", block="reg", sub={"X": SUB["mid"]},
     sk="sklearn.linear_model:ARDRegression", params=dict(max_iter=300, tol=1e-3))
# FAIRNESS FIX (2026-10-13 audit): the race declared eps=2.22e-16 on every arm and read it back
# from ours' get_params, but ours never passes eps to its kernel (python/mojolearn/
# _expansion_linear.py _lars_fit; "Accepted for compatibility; ignored"), and precompute was left
# at each library's 'auto'. Now precompute=True is pinned on every arm (ours always runs the Gram
# form, x_linear/lars.mojo; cuML and scikit-learn 'auto' resolve to it at n_rows > n_cols) and
# the lane says what ours really does with eps: it has no eps floor and drops a candidate whose
# Cholesky pivot is non-positive or whose root is < 1e-7 (x_linear/lars.mojo:204). cuML floors
# the new diagonal at eps and then drops it below 1e-7 (raft cholesky_r1_update.cuh:127,
# cuml lars_impl.cuh:509), scikit-learn takes max(sqrt|c - v|, eps) and drops it below 1e-7, so
# any eps in [0, 1e-7) is ours' rule on both; 2**-52 (both libraries' default) is kept there.
# Results raced before this change were wrong for this lane; apply extra scrutiny.
LARS_EPS = 2.220446049250313e-16
_add("lars", xlane="linear", ours="Lars", task="reg", block="reg",
     sk="sklearn.linear_model:Lars", params=dict(n_nonzero_coefs=500, fit_intercept=True, eps=LARS_EPS,
                                                 precompute=True, random_state=SEED),
     cuml="cuml.experimental.linear_model:Lars",   # eps set on every arm: a default is not a matched value
     cuml_params=dict(n_nonzero_coefs=500, fit_intercept=True, eps=LARS_EPS, precompute=True),
     mism=["eps: ours does not read it (its kernel has no eps floor; a pivot <= 0 or with root "
           "< 1e-7 drops the candidate); cuML and scikit-learn floor the pivot at eps and drop it "
           "below the same 1e-7, so eps = 2**-52 < 1e-7 gives them ours' rule. scikit-learn alone "
           "takes |c - v| and keeps a candidate whose pivot square is negative but at least 1e-14 "
           "in size, where ours and cuML drop it",
           "precompute=True on every arm: ours always factors the Gram (it ignores the argument); "
           "cuML and scikit-learn build it from X once"])
if os.environ.get("MOJOLEARN_BENCH_LARS_STABLE") == "1":
    # A separate, opt-in recipe: never rewrite the historical LARS setting or
    # claim its failed quality was repaired under the original configuration.
    # sklearn documents increasing eps for ill-conditioned Cholesky factors.
    # sqrt(binary64 epsilon) reserves half the precision for their diagonal
    # stabilization; this hypothesis applies to any shape, not a board size.
    # Audit: our Lars stores eps/precompute but never forwards either to Mojo.
    # This recipe is sklearn-only; matching kwargs cannot establish matching
    # numerical behavior. Original published receipts remain unchanged.
    _LARS_STABLE_EPS = 2.0 ** -26
    _add("lars-stable", xlane="linear", ours=(), task="reg", block="reg",
         sk="sklearn.linear_model:Lars",
         params=dict(n_nonzero_coefs=500, fit_intercept=True,
                     eps=_LARS_STABLE_EPS, random_state=SEED),
         notes=["Sklearn-only stabilization experiment: eps=sqrt(binary64 machine epsilon). "
                "Other arms unsupported: ours stores but ignores eps/precompute, cuML behavior "
                "is not qualified. Original lars recipe and failure evidence remain unchanged. "
                "Require finite held-out quality and R2 >= 0 (constant-mean baseline) "
                "before accepting this experiment's quality; no default promotion."])
if os.environ.get("MOJOLEARN_BENCH_LARS_DIRECT") == "1":
    # One supported sklearn numerical-route hypothesis, not a parameter search.
    # Direct X correlations avoid the cached full Gram product, while retaining
    # the LARS objective, data, stopping settings, and original machine eps.
    # sklearn still factors an active Gram; this is not a QR stability claim.
    # This applies to any shape. Other implementations are deliberately absent:
    # ours accepts precompute/eps but does not forward them to its native solver.
    _add("lars-direct", xlane="linear", ours=(), task="reg", block="reg",
         sk="sklearn.linear_model:Lars",
         params=dict(n_nonzero_coefs=500, fit_intercept=True, precompute=False,
                     eps=2.0 ** -52, random_state=SEED),
         notes=["Sklearn-only direct-X LARS route: precompute=False, original eps=2**-52. "
                "No row, feature, iteration, target, seed or objective change. "
                "Our native solver ignores precompute/eps; other arms unsupported here. "
                "Finite predictions and R2>=0 required; no default promotion or A/B claim."])
# FAIRNESS FIX (2026-10-13 audit): as on lars, ours' LassoLars never reads eps and always
# factors the Gram, and precompute was left at each library's 'auto'. precompute=True is
# pinned on both arms and the lane says what ours does with eps (LARS_EPS above). Results raced before this change were wrong for this lane; apply extra scrutiny.
_add("lasso-lars", xlane="linear", ours="LassoLars", task="reg", block="reg",
     sk="sklearn.linear_model:LassoLars",
     params=dict(alpha=0.01, max_iter=500, random_state=SEED, precompute=True),
     mism=["eps: ours does not read it (no eps floor; a pivot <= 0 or with root < 1e-7 drops "
           "the candidate); scikit-learn floors the pivot at eps = 2**-52 and drops it below the "
           "same 1e-7, so its rule is ours'", "precompute=True on both: ours always factors the "
           "Gram (it ignores the argument)"])
# FAIRNESS FIX (2026-10-13 audit): ours accepts solver='highs' and always runs ADMM, while
# scikit-learn solved the HiGHS LP: a different algorithm, and ours silently ignored the
# declared solver. No library has an ADMM quantile regression, so the scikit-learn arm is
# refused and the lane has no opponent. Results raced before this change were wrong for this lane; apply extra scrutiny.
_add("quantile", xlane="linear", ours="QuantileRegressor", task="reg", block="reg",
     sub={"X": SUB["mid"]}, sk="sklearn.linear_model:QuantileRegressor",
     params=dict(quantile=0.5, alpha=1e-4, solver="highs"),
     refused={"sklearn-cpu": "different algorithm: ours runs ADMM whatever solver it is given "
                             "(max_iter=5000, tol=1e-4); scikit-learn's QuantileRegressor has "
                             "only linear-programming solvers (HiGHS, interior point)"},
     mism=["solver: ours accepts 'highs' and always runs ADMM (max_iter=5000, tol=1e-4); no "
           "library implements an ADMM quantile regression, so no opponent races"])
# FAIRNESS FIX (2026-10-13 audit): ours' Perceptron and passive-aggressive pair default to
# minibatch steps of 256 rows, while scikit-learn updates per sample: a different algorithm.
# Ours now races batch_size=0, its documented per-sample form (scikit-learn's); `ours_extra`
# carries it to our arm only (scikit-learn has no batch_size). Results raced before this change were wrong for this lane; apply extra scrutiny.
_add("perceptron", xlane="linear", ours="Perceptron", task="clf", block="cls",
     sk="sklearn.linear_model:Perceptron", params=dict(max_iter=20, tol=None, random_state=SEED),
     ours_extra=dict(batch_size=0),
     mism=["ours batch_size=0 (per-sample updates, scikit-learn's form; ours defaults to 256)"])
_add("pa-clf", xlane="linear", ours="PassiveAggressiveClassifier", task="clf", block="cls",
     sk="sklearn.linear_model:PassiveAggressiveClassifier",
     params=dict(C=1.0, max_iter=20, tol=None, random_state=SEED), ours_extra=dict(batch_size=0),
     mism=["ours batch_size=0 (per-sample updates, scikit-learn's form; ours defaults to 256)"])
_add("pa-reg", xlane="linear", ours="PassiveAggressiveRegressor", task="reg", block="reg",
     sk="sklearn.linear_model:PassiveAggressiveRegressor",
     params=dict(C=1.0, max_iter=20, tol=None, random_state=SEED), ours_extra=dict(batch_size=0),
     mism=["ours batch_size=0 (per-sample updates, scikit-learn's form; ours defaults to 256)"])
_add("ridge-clf", xlane="linear", ours="RidgeClassifier", task="clf", block="cls",
     sk="sklearn.linear_model:RidgeClassifier", params=dict(alpha=1.0, random_state=SEED))
# FAIRNESS FIX (2026-10-13 audit): ours' batch_size was left at its default (None: minibatches of
# 4096 rows in IDENTICAL mode, per-sample otherwise) while scikit-learn's SGDOneClassSVM is
# per-sample SGD: two algorithms (istella flagged 0.00044 vs 0.0933). Ours is pinned to its
# documented per-sample fit (batch_size=0) through `ours_extra` (scikit-learn has no batch_size, so
# it cannot be a shared param). Results raced before this change were wrong for this lane; apply extra
# scrutiny.
_add("sgd-ocsvm", xlane="linear", ours="SGDOneClassSVM", task="outlier", block="cls",
     sk="sklearn.linear_model:SGDOneClassSVM", params=dict(nu=0.1, max_iter=20, tol=None,
                                                          random_state=SEED),
     ours_extra=dict(batch_size=0),
     mism=["ours takes batch_size=0 (its per-sample fit, scikit-learn's algorithm); scikit-learn "
           "SGDOneClassSVM has no batch_size and is always per-sample"])
_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
_add("ridge-cv", xlane="linear", ours="RidgeCV", task="reg", block="reg",
     sk="sklearn.linear_model:RidgeCV", params=dict(alphas=_ALPHAS, cv=5),
     notes=["cv=5: k-fold (cross_val_score's fold order), not the leave-one-out GCV of cv=None"])
# FAIRNESS FIX (2026-10-13 audit): precompute was left at each library's 'auto' (scikit-learn
# resolves it to the Gram at n_samples > n_features; ours ignores it and always sweeps per-fold
# centered Grams, x_linear/enetcv_fast.mojo). precompute=True is pinned on both arms. Results raced before this change were wrong for this lane; apply extra scrutiny.
_CV_MISM = ["precompute=True on both: ours ignores the argument and always sweeps each fold's "
            "centered Gram (x_linear/enetcv_fast.mojo); scikit-learn's Gram coordinate descent"]
_add("lasso-cv", xlane="linear", ours="LassoCV", task="reg", block="reg",
     sk="sklearn.linear_model:LassoCV",
     params=dict(alphas=_ALPHAS[:4], cv=5, max_iter=1000, tol=1e-4, random_state=SEED,
                 precompute=True),
     sk_params=dict(alphas=_ALPHAS[:4], cv=5, max_iter=1000, tol=1e-4, random_state=SEED, n_jobs=-1,
                    precompute=True), mism=list(_CV_MISM))
_add("enet-cv", xlane="linear", ours="ElasticNetCV", task="reg", block="reg",
     sk="sklearn.linear_model:ElasticNetCV",
     params=dict(l1_ratio=0.5, alphas=_ALPHAS[:4], cv=5, max_iter=1000, tol=1e-4, random_state=SEED,
                 precompute=True),
     sk_params=dict(l1_ratio=0.5, alphas=_ALPHAS[:4], cv=5, max_iter=1000, tol=1e-4,
                    random_state=SEED, n_jobs=-1, precompute=True), mism=list(_CV_MISM))
_add("logreg-cv", xlane="linear", ours="LogisticRegressionCV", task="clf", block="cls",
     sk="sklearn.linear_model:LogisticRegressionCV",
     params=dict(Cs=[0.1, 1.0, 10.0], cv=5, max_iter=1000, random_state=SEED),
     sk_params=dict(Cs=[0.1, 1.0, 10.0], cv=5, max_iter=1000, random_state=SEED, n_jobs=-1))
_add("isotonic", xlane="linear", ours="IsotonicRegression", task="reg", block="reg", iso=True,
     sk="sklearn.isotonic:IsotonicRegression", params=dict(out_of_bounds="clip"),
     notes=["X = the one fit-row column most correlated with y (1-D, as isotonic regression takes)"])

# ---- lane cluster ---------------------------------------------------------
# FAIRNESS FIX (2026-10-13 audit): init_size (the rows the k-means++ start is drawn from) was left
# at each library's default (scikit-learn 3 x batch_size; ours' None, unverified to be the same);
# it is pinned to 3 x batch_size on both. n_iter and n_steps of every arm are recorded
# (_ITER_LANES): early stopping by max_no_improvement can end the arms at different steps.
# Results raced before this change were wrong for this lane; apply extra scrutiny.
_add("minibatch-kmeans", xlane="cluster", ours="MiniBatchKMeans", task="cluster", block="cls",
     sk="sklearn.cluster:MiniBatchKMeans",
     params=dict(n_clusters=8, batch_size=4096, max_iter=100, n_init=1, init_size=3 * 4096,
                 random_state=SEED))
_add("bisecting-kmeans", xlane="cluster", ours="BisectingKMeans", task="cluster", block="cls",
     sk="sklearn.cluster:BisectingKMeans", params=dict(n_clusters=8, random_state=SEED))
_add("meanshift", xlane="cluster", ours="MeanShift", task="cluster", block="cls",
     sub={"X": SUB["quad"]}, bandwidth=True, sk="sklearn.cluster:MeanShift",
     params=dict(bin_seeding=True), sk_params=dict(bin_seeding=True, n_jobs=-1),
     notes=["bandwidth = the 0.3 quantile of pairwise distances over the first 1,000 fit rows, "
            "computed once in the conductor and passed to every arm"])
# FAIRNESS FIX (2026-10-13 audit): algorithm='auto' and leaf_size=30 were declared on both, but
# ours ignores them and always computes the full distance matrix (a brute-force search), while
# scikit-learn's 'auto' builds a KD/ball tree: different neighbour searches. Both now run
# algorithm='brute' (ours' own method; scikit-learn's brute NearestNeighbors). Results raced
# before this change were wrong for this lane; apply extra scrutiny.
_add("optics", xlane="cluster", ours="OPTICS", task="cluster", block="cls", sub={"X": SUB["quad"]},
     sk="sklearn.cluster:OPTICS", params=dict(min_samples=10, xi=0.05, algorithm="brute"),
     sk_params=dict(min_samples=10, xi=0.05, algorithm="brute", n_jobs=-1),
     mism=["algorithm='brute' on both: ours always computes the full n x n distance matrix "
           "(it accepts and ignores algorithm and leaf_size); scikit-learn's brute "
           "NearestNeighbors for the core distances"])
_add("affinity-prop", xlane="cluster", ours="AffinityPropagation", task="cluster", block="cls",
     sub={"X": SUB["tiny"]}, sk="sklearn.cluster:AffinityPropagation",
     params=dict(damping=0.5, max_iter=200, convergence_iter=15, random_state=SEED))
_add("bayesian-gmm", xlane="cluster", ours="BayesianGaussianMixture", task="gmm", block="reg",
     sub={"X": SUB["mid"], "Xq": SUB["small"]}, sk="sklearn.mixture:BayesianGaussianMixture",
     params=dict(n_components=8, covariance_type="full", max_iter=100, tol=1e-3, reg_covar=1e-6,
                 init_params="kmeans", random_state=SEED),
     dataset_params={"istella": dict(reg_covar=3e-3)},
     notes=["reg_covar 3e-3 on Istella-S (taxi keeps 1e-6), for every arm on every vendor: at 1e-6 "
            "every arm refused on the constant-dropped Istella rows (ours and ours-fast on 0.8.29 "
            "with the GEMM moments, and scikit-learn: ill-defined empirical covariance, "
            "m3ultra-b 2026-09-29; again ours and scikit-learn on the 0.8.34 L40S and MI300X boards), and 3e-3 is the smallest value at which the arms fitted "
            "(tools/gmm_istella_probe.py), the value classical2/gmm uses on Istella-S (GMM_REG_COVAR)",
            "constant columns dropped: the columns constant on the fit rows (Istella-S: 20 of 220 on the 100,000 fit rows) are removed from X and Xq before the clock, the same for every arm; a full covariance over them is singular, and on the raw float32 rows ours, scikit-learn float32 and both Bayesian mixtures refused (ill-defined empirical covariance) where only scikit-learn float64 fitted (m3ultra-b, 2026-09-29, tools/gmm_istella_probe.py)"])

# ---- lane neighbors + kernel ---------------------------------------------
_add("lof", xlane="neighbors", ours="LocalOutlierFactor", task="outlier", block="cls",
     sub={"X": SUB["knn"]}, fit_predict=True, sk="sklearn.neighbors:LocalOutlierFactor",
     params=dict(n_neighbors=20, algorithm="brute"),
     sk_params=dict(n_neighbors=20, algorithm="brute", n_jobs=-1))
_add("nearest-centroid", xlane="neighbors", ours="NearestCentroid", task="clf", block="cls",
     sk="sklearn.neighbors:NearestCentroid")
# FAIRNESS FIX (2026-10-13 audit): ours' only search is a random ball cover (it refuses
# 'brute', 'kd_tree' and 'ball_tree'); scikit-learn's 'auto' builds a KD/ball tree in fit and
# queries it: different algorithms (the DBSCAN ruling: rbc against brute is not the same race),
# and the fit column timed a tree build against ours' no-op. No library implements a random ball
# cover radius search on the CPU (cuML's rbc has no radius query), so scikit-learn is refused and
# the lane has no opponent. Results raced before this change were wrong for this lane; apply
# extra scrutiny.
_add("radius-neighbors", xlane="neighbors", ours=("RadiusNeighbors",), task="radius",
     block="cls", sub={"X": SUB["knn"], "Xq": SUB["small"]}, radius=True,
     refused={"sklearn-cpu": "a different algorithm: ours' radius search is a random ball cover "
                             "(it refuses brute and the trees); scikit-learn's is a KD/ball tree "
                             "or brute force, and no library implements a random ball cover "
                             "radius search for this device (2026-10-13 fairness audit)"},
     sk="sklearn.neighbors:NearestNeighbors",
     params=dict(metric="euclidean", algorithm="auto", p=2),
     sk_params=dict(metric="euclidean", algorithm="auto", p=2, n_jobs=-1),
     notes=["radius = the 0.1% quantile of the pairwise distances of 1,000 fit rows (about 200 "
            "neighbours per query), the same value on both arms; fit(X) then "
            "radius_neighbors(Xq) (the inference column)"],
     mism=["algorithm='auto' on both: ours' auto is its random ball cover (exact, triangle-"
           "inequality pruning), scikit-learn's picks a KD/ball tree or brute force; both return "
           "the exact neighbour set"])
_add("ocsvm", xlane="neighbors", ours="OneClassSVM", task="outlier", block="cls",
     sub={"X": SUB["quad"], "Xq": SUB["quad"]}, gamma=True, sk="sklearn.svm:OneClassSVM",
     params=dict(kernel="rbf", nu=0.1, tol=1e-3), sk_params=dict(kernel="rbf", nu=0.1, tol=1e-3,
                                                                 cache_size=2000))
_add("kernel-pca", xlane="neighbors", ours="KernelPCA", task="transform", block="cls",
     sub={"X": SUB["quad"], "Xq": SUB["quad"]}, gamma=True, quality="subspace",
     sk="sklearn.decomposition:KernelPCA", params=dict(n_components=8, kernel="rbf", random_state=SEED),
     sk_params=dict(n_components=8, kernel="rbf", random_state=SEED, n_jobs=-1),
     notes=["eigen_solver='auto' picks a randomized/ARPACK solver at this size: random_state=7 "
            "on every arm"])
_add("poly-count-sketch", xlane="neighbors", ours="PolynomialCountSketch", task="transform",
     block="reg", sub={"X": SUB["mid"], "Xq": 1000}, gamma=True, quality="kernel-poly",
     sk="sklearn.kernel_approximation:PolynomialCountSketch",
     params=dict(degree=2, coef0=0, n_components=256, random_state=SEED))
_add("additive-chi2", xlane="neighbors", ours="AdditiveChi2Sampler", task="transform",
     block="nonneg", sub={"X": SUB["mid"], "Xq": 1000}, quality="kernel-achi2",
     sk="sklearn.kernel_approximation:AdditiveChi2Sampler", params=dict(sample_steps=2))
_add("skewed-chi2", xlane="neighbors", ours="SkewedChi2Sampler", task="transform",
     block="nonneg", sub={"X": SUB["mid"], "Xq": 1000}, quality="kernel-schi2",
     sk="sklearn.kernel_approximation:SkewedChi2Sampler",
     params=dict(skewedness=1.0, n_components=256, random_state=SEED))
_add("label-propagation", xlane="neighbors", ours="LabelPropagation", task="semi", block="cls",
     sub={"X": SUB["knn"], "Xq": SUB["small"]}, sk="sklearn.semi_supervised:LabelPropagation",
     params=dict(kernel="knn", n_neighbors=7, max_iter=1000, tol=1e-3),
     sk_params=dict(kernel="knn", n_neighbors=7, max_iter=1000, tol=1e-3, n_jobs=-1),
     notes=["90% of the fit labels hidden (-1) by a fixed stride: every 10th row keeps its label"])
_add("label-spreading", xlane="neighbors", ours="LabelSpreading", task="semi", block="cls",
     sub={"X": SUB["knn"], "Xq": SUB["small"]}, sk="sklearn.semi_supervised:LabelSpreading",
     params=dict(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3),
     sk_params=dict(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3, n_jobs=-1),
     notes=["90% of the fit labels hidden (-1) by a fixed stride"])
_add("knn-imputer", xlane="neighbors", ours="KNNImputer", task="impute", block="raw",
     sub={"X": SUB["mid"], "Xq": SUB["small"]}, sk="sklearn.impute:KNNImputer",
     params=dict(n_neighbors=5, weights="uniform"),
     notes=["10% of the cells of X and Xq set to NaN by a seed-7 mask; quality on those cells"])
_GRAPH_MISM = ("ours takes the CSR graph as an (indptr, indices, n) tuple and uploads it inside the "
               "clock (the 2026-10-08 CSR entries; before, a dense n x n adjacency built before the clock); "
               "networkx and cuGraph take the graph itself")
_add("pagerank", xlane="neighbors", ours=("PageRank",), kind="graph", task="pagerank",
     block="graphs", params=dict(alpha=0.85, tol=1e-6, max_iter=100),
     other={"networkx-cpu": "networkx", "cugraph-gpu": "cugraph"}, mism=[_GRAPH_MISM],
     notes=["graph: the symmetric 10-nearest-neighbour graph of %d stride rows of the cls block "
            "(float64 brute force in prep), unweighted" % GRAPH_SMALL,
            # FAIRNESS FIX (2026-10-13 audit): cuGraph stopped below its unscaled tol
            # (1e-6), ours and networkx below n x tol (0.02): 20,000x tighter on
            # cuGraph. One rule now on every arm, and cuGraph's graph is stored
            # transposed as its pagerank asks. Results raced before this change were
            # wrong for this lane; apply extra scrutiny.
            "stopping rule on every arm: the L1 change of the scores below n x tol, at most "
            "max_iter=100 iterations (ours and networkx: tol=1e-6 is per node, networkx "
            "err < N * tol; cuGraph's tol is the unscaled L1 bound, so it is given n x tol)",
            "cuGraph: from_cudf_edgelist(renumber=False, store_transposed=True) before the clock "
            "(pagerank's documented layout; without it cuGraph transposes on every call)"])
# FAIRNESS FIX (2026-10-13 audit): connected-components raced a 20,000-node
# graph (the `graphs` block, sized for ours' former dense adjacency), where
# cuGraph's time was its per-call Python/cudf overhead (renumbering on) and
# networkx on the CPU beat it. It now races the 100,000-node `graph` block (its
# k=2 graph; ours takes the CSR since 2026-10-01, so the dense limit is gone)
# with cuGraph's documented fast path (renumber=False for 0..n-1 vertex ids,
# the Graph built before its clock; _build_graph). Results raced before this
# change were wrong for this lane; apply extra scrutiny.
_add("connected-components", xlane="neighbors", ours=("connected_components", "ConnectedComponents"),
     kind="graph", task="components", block="graph",
     other={"networkx-cpu": "networkx", "cugraph-gpu": "cugraph"}, mism=[_GRAPH_MISM],
     notes=["the symmetric 2-nearest-neighbour graph of %d stride rows of the cls block "
            "(float64 brute force in prep; sparse enough to have more than one component), "
            "unweighted" % GRAPH_NODES,
            "cuGraph: Graph(directed=False).from_cudf_edgelist(renumber=False) built before "
            "the clock (vertex ids are already 0..n-1); ours' CSR arrays are built before its "
            "clock and uploaded inside it"])
# FAIRNESS FIX (2026-10-13 audit): the modularity-gain threshold was each library's default
# (1e-7 on ours, networkx and cuGraph, but never passed); it is now pinned to 1e-7 on every
# arm (_build_graph). Results raced before this change were wrong for this lane; apply extra
# scrutiny.
_add("louvain", xlane="neighbors", ours=("Louvain",), kind="graph", task="louvain", block="graphs",
     params=dict(resolution=1.0, threshold=1e-7, seed=SEED),
     other={"networkx-cpu": "networkx", "cugraph-gpu": "cugraph"},
     mism=["networkx louvain_communities(seed=7) and cuGraph louvain (max_level=100) are "
           "order-dependent; ours pins a colour-batched move order (a fixed hash colouring, ties to the smallest community id)", _GRAPH_MISM])
_add("svgp", xlane="neighbors", ours=("SVGP", "SparseVariationalGP"), kind="svgp", task="reg",
     block="reg", sub={"X": SUB["mid"], "Xq": SUB["small"]},
     params=dict(n_inducing=512, kernel_variance=1.0, lengthscale=1.0, noise_variance=1.0,
                 jitter=1e-6),
     other={"gpytorch-gpu": "gpytorch", "gpytorch-cpu": "gpytorch"},
     notes=["zero mean, RBF kernel (one lengthscale), Gaussian likelihood, inducing points = 512 "
            "stride rows (fixed), kernel variance, lengthscale and noise variance all 1.0 and "
            "NOT trained on any arm: ours sets q(u) to its optimum for the given "
            "hyperparameters (Titsias's collapsed bound, DEVIATION 5205); gpytorch is the same "
            "model through InducingPointKernel (SGPR, the collapsed bound) in an ExactGP with "
            "ZeroMean, the same fixed hyperparameters, its prediction cache built in the fit clock",
            # FAIRNESS FIX (2026-10-13 audit): the note above claimed the same model while
            # GPyTorch's default added the FITC diagonal to the train covariance; see _build_svgp.
            # Results raced before this change were wrong for this lane; apply extra scrutiny.
            "gpytorch runs with gpytorch.settings.sgpr_diagonal_correction(False) in both clocks: "
            "its default (True) adds diag(K_ff - Q_ff) to the train covariance in eval mode, a "
            "FITC-style posterior and not the Titsias one ours computes (before 2026-10-13 it "
            "raced with the correction on)"],
     mism=["no seed on any arm: nothing is drawn (fixed inducing points, closed form)",
           "jitter: ours 1e-6 on K_uu; gpytorch adds its own Cholesky jitter (1e-6 in float32) "
           "only when a factorization fails",
           "precision and outputs: ours solves the M x M inducing system in float-float and its fit "
           "also forms q_sqrt and the ELBO; gpytorch float32, its fit clock builds the predictive "
           "mean cache only (the covariance cache would add an n x M^2 product ours does not do)"])

# ---- lane decomp + linalg -------------------------------------------------
# FAIRNESS FIX (2026-10-13 audit): the per-batch algorithm differed and was undeclared (mismatches
# were []): ours takes each batch's thin SVD as the eigendecomposition of the d x d Gram matrix of
# the stacked batch (python/mojolearn/_expansion_decomp.py `_gram_svd`, IncrementalPCA._partial),
# while cuML (cuml/decomposition/incremental_pca.py: cp.linalg.svd(X, full_matrices=False)) and
# scikit-learn (scipy.linalg.svd) run a full thin SVD of the stacked (n_components + batch + 1) x d
# matrix, several times the work. Neither library has a Gram/eigh option and ours has no SVD
# option, so the algorithms cannot be matched: the difference is now declared and both opponents
# are refused (`refused`: not planned, and a named arm refuses with this reason). The race has no
# opponent until they can be matched. Results raced before this change were wrong for this lane;
# apply extra scrutiny.
_IPCA_REFUSED = ("fairness audit 2026-10-13: a different per-batch algorithm, the work "
                 "differs. Ours: thin SVD of each stacked batch through eigh of its d x d Gram "
                 "matrix; this library: a full thin SVD (LAPACK/cuSOLVER gesvd-family) of the "
                 "stacked batch. No setting on either side selects the other's algorithm, and no "
                 "library offers ours' per-batch algorithm (cuML, scikit-learn and dask-ml "
                 "IncrementalPCA all take a thin SVD; torch has no IncrementalPCA)")
_add("incremental-pca", xlane="decomp", ours="IncrementalPCA", task="transform", block="tsvd",
     quality="pca", sk="sklearn.decomposition:IncrementalPCA",
     params=dict(n_components=10, batch_size=65536),     # cuML benchmark IncrementalPCA
     cuml="cuml.decomposition:IncrementalPCA", cuml_params=dict(n_components=10, batch_size=65536),
     refused={"cuml-gpu": _IPCA_REFUSED, "sklearn-cpu": _IPCA_REFUSED},
     mism=["per-batch SVD: ours eigh of the d x d Gram matrix of the stacked batch (Jacobi eigh); "
           "cuML cp.linalg.svd and scikit-learn scipy.linalg.svd of the stacked batch: different "
           "algorithms, so both opponents are refused"])
# FAIRNESS FIX (2026-10-13 audit): commit fcef3cd62 gave cuML HOST input in the two random-projection
# lanes only (cuml_host_input=True), so cuML's host-side input check/transfer sat inside its clock
# (gaussian-rp Istella fit 31.4 ms with host input against 2.4 ms with device input on the same
# L40S), unlike every other cuML lane. The flag is removed: cuML gets its input on the device before
# its clock, as in every other cuML lane (ours includes its upload: the disclosed clock asymmetry).
# Results raced before this change were wrong for this lane; apply extra scrutiny.
_add("gaussian-rp", xlane="decomp", ours="GaussianRandomProjection", task="transform",
     block="tsvd", quality="distortion", sk="sklearn.random_projection:GaussianRandomProjection",
     params=dict(n_components=10, random_state=SEED),   # cuML benchmark GaussianRandomProjection
     cuml="cuml.random_projection:GaussianRandomProjection",
     notes=["n_components = 10, the cuML benchmark's",
            "cuML's X and Xq are on the device before its clock, as in every cuML lane"],
     mism=["the random matrix: each library draws its own from its own generator seeded 7"])
_add("sparse-rp", xlane="decomp", ours="SparseRandomProjection", task="transform", block="tsvd",
     quality="distortion", sk="sklearn.random_projection:SparseRandomProjection",
     params=dict(n_components=10, density="auto", random_state=SEED),  # cuML benchmark
     cuml="cuml.random_projection:SparseRandomProjection",
     notes=["n_components = 10, the cuML benchmark's",
            "cuML's X and Xq are on the device before its clock, as in every cuML lane"],
     mism=["the random matrix: each library draws its own from its own generator seeded 7 "
           "(cuML a fixed nonzero count density x n_features x n_components, scikit-learn a "
           "binomial count per row)"])
_add("nmf", xlane="decomp", ours="NMF", task="transform", block="nonneg", quality="nmf",
     sk="sklearn.decomposition:NMF",
     params=dict(n_components=8, init="nndsvda", solver="mu", max_iter=200, tol=1e-4,
                 random_state=SEED))
_add("fastica", xlane="decomp", ours="FastICA", task="transform", block="tsvd", quality="ica",
     sk="sklearn.decomposition:FastICA",
     params=dict(n_components=8, whiten="unit-variance", max_iter=200, tol=1e-4, random_state=SEED))
# FAIRNESS FIX (2026-10-13 audit): the lane declared svd_method='randomized', which ours accepts and
# ignores (python/mojolearn/_expansion_decomp.py FactorAnalysis: "both svd_method values use an
# exact SVD at every iteration"), while scikit-learn ran its randomized SVD of the full scaled X at
# every EM iteration: a different, approximate algorithm on one side. Both arms now take
# svd_method='lapack', the exact SVD ours really computes. Results raced before this change were
# wrong for this lane; apply extra scrutiny.
_add("factor-analysis", xlane="decomp", ours="FactorAnalysis", task="transform", block="cls",
     quality="fa", sk="sklearn.decomposition:FactorAnalysis",
     params=dict(n_components=8, max_iter=1000, tol=1e-2, svd_method="lapack",
                 random_state=SEED),
     mism=["exact SVD on both arms: ours reduces the centred X by one QR and takes the SVD of the "
           "d x d scaled R per EM iteration (the same singular values and vectors as the scaled X's); "
           "scikit-learn svd_method='lapack' takes scipy.linalg.svd of the full scaled X per iteration",
           "precision: scikit-learn FactorAnalysis converts X to float64 (validate_data dtype=float64, "
           "its only path); ours float32",
           "the same stopping rule on both: the total log-likelihood gain below tol=1e-2, max_iter=1000"])
# FAIRNESS FIX (2026-10-13 audit): the CPU opponent of lu-solve, lstsq, cholesky, qr, eigh and svd
# was numpy.linalg, which computes every float32 input in float64 (numpy/linalg/_linalg.py
# _commonType: "always double", the 'dd->d' gufunc signatures) and rounds the result back: double the
# arithmetic and a different precision from ours' float32. scipy.linalg, the same LAPACK through
# its public API, keeps float32 (the s-routines): the CPU arm is now scipy-cpu with numpy's drivers
# (getrf/getrs, gelsd, potrf, geqrf + orgqr, syevd, gesdd) and check_finite=False, as numpy checks
# nothing. Results raced before this change were wrong for these lanes; apply extra scrutiny.
_add("lu-solve", xlane="decomp", ours=("solve", "linalg.solve"), kind="linalg", task="lu",
     block="dense", datasets=("synthetic",),
     other={"scipy-cpu": "scipy", "torch-gpu": "torch", "cupy-gpu": "cupy"},
     notes=["A = N(0,1) 8192 x 8192 + 2 sqrt(8192) I (seed 7), B 8192 x 64; LU with partial pivoting then "
            "the two triangular solves (scipy.linalg.solve = LAPACK sgetrf/sgetrs in float32)",
            "dense linear algebra has no dataset kind: one seeded matrix, not taxi/Istella"])
_add("lstsq", xlane="decomp", ours=("lstsq", "linalg.lstsq"), kind="linalg", task="lstsq", block="reg",
     other={"scipy-cpu": "scipy", "torch-gpu": "torch", "cupy-gpu": "cupy"},
     notes=["min ||X b - y|| on the reg block's fit rows (1,000,000 x d)"],
     # FAIRNESS FIX (2026-10-13 audit): torch's GPU lstsq (driver 'gels' only on CUDA and ROCm: QR
     # without pivoting, full rank assumed) returned NaN on the rank-deficient Istella block and
     # still counted as the fastest opponent. A non-finite torch solution now fails its round (an
     # invalid result is not a result); the difference is declared. Results raced before this
     # change were wrong for this lane; apply extra scrutiny.
     mism=["torch-gpu: torch.linalg.lstsq on CUDA/ROCm has only driver='gels' (QR without "
           "pivoting, full rank assumed); ours, scipy (gelsd) and CuPy are rank-revealing; a "
           "non-finite torch solution (rank-deficient X, e.g. Istella) refuses the arm"])
_add("randomized-svd", xlane="decomp", ours=("randomized_svd", "linalg.randomized_svd"),
     kind="linalg", task="rsvd", block="tsvd",
     other={"sklearn-cpu": "sklearn", "torch-gpu": "torch"},
     # FAIRNESS FIX (2026-10-13 audit): power_iteration_normalizer was left at scikit-learn's
     # default 'auto' (LU for n_iter > 2), while ours re-orthonormalizes by QR every half
     # iteration whatever it is given and torch.svd_lowrank does the same: a different
     # normalization on one arm. It is pinned to 'QR' on ours and scikit-learn (torch's own
     # rule). Results raced before this change were wrong for this lane; apply extra scrutiny.
     params=dict(n_components=8, n_oversamples=10, n_iter=4, random_state=SEED,
                 power_iteration_normalizer="QR"),
     mism=["torch-gpu is torch.svd_lowrank(q = n_components + n_oversamples, clipped to "
           "min(X.shape), niter=4), its randomized range finder with a QR step every half "
           "iteration, as ours and scikit-learn's power_iteration_normalizer='QR'"])
# the rest of mojolearn's dense linear algebra (README family "linear algebra"),
# raced from 2026-09-29: numpy/scipy on the CPU, torch on this box's GPU, CuPy on NVIDIA
_add("lu-factor", xlane="decomp", ours=("lu_factor", "linalg.lu_factor"), kind="linalg",
     task="lufac", quality="lu", block="dense", datasets=("synthetic",),
     other={"scipy-cpu": "scipy", "torch-gpu": "torch", "cupy-gpu": "cupy"},
     notes=["the lu-solve lane's system (A 8192 x 8192 + 2 sqrt(8192) I, B 8192 x 64, seed 7) "
            "through the two-call form: lu_solve(lu_factor(A), B) (scipy.linalg's; torch.linalg."
            "lu_factor + lu_solve; cupyx.scipy.linalg's)"])
_add("cholesky", xlane="decomp", ours=("Cholesky", "linalg.Cholesky"), kind="linalg",
     task="chol", block="sym", datasets=("synthetic",), params=dict(jitter=0.0),
     other={"scipy-cpu": "scipy", "torch-gpu": "torch", "cupy-gpu": "cupy"},
     notes=["A = (M + M^T) / 2 + 2 sqrt(n) I, M N(0,1) 8192 x 8192 seed 7 (symmetric positive "
            "definite); ours Cholesky(jitter=0.0).fit(A) (no ridge, as scipy.linalg.cholesky)"])
_add("qr", xlane="decomp", ours=("linalg.qr",), kind="linalg", task="qr", block="reg",
     params=dict(mode="reduced"),
     other={"scipy-cpu": "scipy", "torch-gpu": "torch", "cupy-gpu": "cupy"},
     notes=["qr(X, mode='reduced') of the reg block's fit rows (1,000,000 x d) on every arm "
            "(Householder, Q and R formed)"])
_add("eigh", xlane="decomp", ours=("linalg.eigh",), kind="linalg", task="eigh", block="sym",
     datasets=("synthetic",), params=dict(UPLO="L"),
     other={"scipy-cpu": "scipy", "torch-gpu": "torch", "cupy-gpu": "cupy"},
     notes=["eigh(A, UPLO='L') of the cholesky lane's symmetric matrix at n = 4096: every "
            "eigenvalue and eigenvector"])
_add("svd", xlane="decomp", ours=("linalg.svd",), kind="linalg", task="svd", block="reg",
     params=dict(full_matrices=False),
     other={"scipy-cpu": "scipy", "torch-gpu": "torch", "cupy-gpu": "cupy"},
     notes=["svd(X, full_matrices=False) of the reg block's fit rows (1,000,000 x d): U, S, "
            "Vh on every arm"])
_add("cca", xlane="decomp", ours="CCA", task="transform", block="cls", quality="cca", xy_split=True,
     sk="sklearn.cross_decomposition:CCA",
     params=dict(n_components=2, max_iter=500, tol=1e-6, scale=True),
     notes=["X = the first half of the columns, Y = the second half"],
     # FAIRNESS FIX (2026-10-13 audit): the precision difference was undeclared, and scale was
     # each library's default (True on both, now explicit). Results raced before this change
     # were wrong for this lane; apply extra scrutiny.
     mism=["precision: scikit-learn's _PLS family converts X and Y to float64 (validate_data "
           "dtype=np.float64, its only path); ours float32"])
# FAIRNESS FIX (2026-10-13 audit): the precision difference was undeclared, and scale (and tol on
# pls) were each library's default (the same values, now explicit). Results raced before this
# change were wrong for these lanes; apply extra scrutiny.
_add("pls-canonical", xlane="decomp", ours="PLSCanonical", task="transform", block="cls",
     quality="cca", xy_split=True, sk="sklearn.cross_decomposition:PLSCanonical",
     params=dict(n_components=2, max_iter=500, tol=1e-6, scale=True),
     notes=["X = the first half of the columns, Y = the second half"],
     mism=["precision: scikit-learn's _PLS family converts X and Y to float64 (validate_data "
           "dtype=np.float64, its only path); ours float32"])
_add("pls", xlane="decomp", ours="PLSRegression", task="reg", block="reg",
     sk="sklearn.cross_decomposition:PLSRegression",
     params=dict(n_components=4, max_iter=500, tol=1e-6, scale=True),
     mism=["precision: scikit-learn's _PLS family converts X and Y to float64 (validate_data "
           "dtype=np.float64, its only path); ours float32"])
_add("sparse-pca", xlane="decomp", ours="SparsePCA", task="transform", block="cls",
     sub={"X": SUB["small"], "Xq": SUB["small"]}, quality="recon",
     sk="sklearn.decomposition:SparsePCA",
     params=dict(n_components=8, alpha=1.0, max_iter=100, tol=1e-6, random_state=SEED),
     sk_params=dict(n_components=8, alpha=1.0, max_iter=100, tol=1e-6, random_state=SEED, n_jobs=-1))
_add("mb-sparse-pca", xlane="decomp", ours="MiniBatchSparsePCA", task="transform", block="cls",
     sub={"X": SUB["mid"], "Xq": SUB["small"]}, quality="recon",
     sk="sklearn.decomposition:MiniBatchSparsePCA",
     params=dict(n_components=8, alpha=1.0, batch_size=1024, max_iter=10, random_state=SEED),
     sk_params=dict(n_components=8, alpha=1.0, batch_size=1024, max_iter=10, random_state=SEED,
                    n_jobs=-1))
_add("dict-learning", xlane="decomp", ours="DictionaryLearning", task="transform", block="cls",
     sub={"X": SUB["small"], "Xq": SUB["tiny"]}, quality="recon",
     sk="sklearn.decomposition:DictionaryLearning",
     params=dict(n_components=16, alpha=1.0, max_iter=100, tol=1e-8, fit_algorithm="cd",
                 transform_algorithm="lasso_cd", random_state=SEED))
_add("lda", xlane="decomp", ours="LatentDirichletAllocation", task="transform", block="counts",
     datasets=("text", "taxi-zones"), quality="perplexity",
     sk="sklearn.decomposition:LatentDirichletAllocation",
     params=dict(n_components=16, learning_method="batch", max_iter=20, random_state=SEED),
     sk_params=dict(n_components=16, learning_method="batch", max_iter=20, random_state=SEED,
                    n_jobs=-1))
_add("classical-mds", xlane="decomp", ours="ClassicalMDS", task="embed", block="manifold",
     sub={"X": SUB["tiny"]}, sk="sklearn.manifold:ClassicalMDS", params=dict(n_components=2),
     # FAIRNESS FIX (2026-10-13 audit): with the pinned scikit-learn 1.7.2 (no ClassicalMDS) the
     # opponent was this harness's assembly of classical MDS (-0.5 D^2 handed to
     # KernelPCA(kernel='precomputed')), not a library's ClassicalMDS. The assembly is no longer
     # an opponent: the arm runs only when the installed scikit-learn exports
     # sklearn.manifold.ClassicalMDS (1.8 and later) and refuses by name otherwise. Results
     # raced before this change were wrong for this lane; apply extra scrutiny.
     notes=["the opponent is scikit-learn's own sklearn.manifold.ClassicalMDS (scikit-learn 1.8 "
            "and later); with an older scikit-learn (the pinned 1.7.2) the arm refuses by name: "
            "a KernelPCA assembly by this harness is not an opponent"])
_add("mb-dict-learning", xlane="decomp", ours="MiniBatchDictionaryLearning", task="transform",
     block="cls", sub={"X": SUB["mid"], "Xq": SUB["tiny"]}, quality="recon",
     sk="sklearn.decomposition:MiniBatchDictionaryLearning",
     params=dict(n_components=16, alpha=1.0, batch_size=256, max_iter=10,
                 transform_algorithm="lasso_cd", random_state=SEED),
     sk_params=dict(n_components=16, alpha=1.0, batch_size=256, max_iter=10,
                    transform_algorithm="lasso_cd", random_state=SEED, n_jobs=-1))
# FAIRNESS FIX (2026-10-13 audit): scikit-learn's SparseCoder ran transform (OMP over 100,000
# rows) at its default n_jobs=None, one core; sk_extra gives it every core. Results raced before this change were wrong for this lane; apply extra scrutiny.
_add("sparse-coder", xlane="decomp", ours=("SparseCoder",), task="transform", block="cls",
     sub={"X": SUB["tiny"], "Xq": SUB["mid"]}, quality="vs-sklearn", dictionary=True,
     sk="sklearn.decomposition:SparseCoder", sk_extra=dict(n_jobs=-1),
     params=dict(transform_algorithm="omp", transform_n_nonzero_coefs=4, transform_alpha=None,
                 split_sign=False, positive_code=False, transform_max_iter=1000),
     notes=["dictionary = 64 stride rows of the fit block, each scaled to unit norm, the same "
            "array on both arms; SparseCoder has no fit (fit returns self), so the work is "
            "transform(Xq), the inference column: OMP with 4 nonzero coefficients per row",
            "mojolearn.sparse_encode is the same encoder as a function (SparseCoder.transform "
            "calls it); this lane is its race"])
# FAIRNESS FIX (2026-10-13 audit): eigen_solver='auto' with tol=0 was declared on both, but ours
# ignores tol and max_iter and solves exactly, while scikit-learn's 'auto' runs ARPACK at this
# size: different eigensolvers. Both now run eigen_solver='dense' (ours' exact solve;
# scikit-learn's full eigendecomposition). Results raced before this change were wrong for this
# lane; apply extra scrutiny.
_add("isomap", xlane="decomp", ours="Isomap", task="embed", block="manifold",
     sub={"X": SUB["quad"]}, sk="sklearn.manifold:Isomap",
     params=dict(n_neighbors=10, n_components=2, eigen_solver="dense"),
     sk_params=dict(n_neighbors=10, n_components=2, eigen_solver="dense", n_jobs=-1),
     mism=["eigen_solver='dense' on both (ours ignores tol and max_iter; the dense solve reads "
           "neither)", "path_method: ours always runs Dijkstra; scikit-learn 'auto'"])
_add("mds", xlane="decomp", ours="MDS", task="embed", block="manifold", sub={"X": SUB["tiny"]},
     sk="sklearn.manifold:MDS", params=dict(n_components=2, n_init=1, max_iter=300, eps=1e-3,
                                             random_state=SEED),
     sk_params=dict(n_components=2, n_init=1, max_iter=300, eps=1e-3, random_state=SEED, n_jobs=-1),
     mism=["metric MDS on Euclidean distances on both, spelled differently: ours "
           "metric='euclidean', metric_mds=True, init='random' (scikit-learn 1.9's names); the "
           "pinned scikit-learn 1.7.2 metric=True, dissimilarity='euclidean' and a random start "
           "from random_state; each draws its own start"])
# FAIRNESS FIX (2026-10-13 audit): eigen_solver='auto' with tol=1e-6 and max_iter=100 was declared
# on both, but ours ignores tol (its iterative route stops on its own subspace tolerance, 1e-5)
# while scikit-learn's ARPACK uses tol=1e-6: different stopping rules. Both now run
# eigen_solver='dense' (ours' full exact solve; scikit-learn's dense eigh), which reads neither.
# Results raced before this change were wrong for this lane; apply extra scrutiny.
_add("lle", xlane="decomp", ours="LocallyLinearEmbedding", task="embed", block="manifold",
     sub={"X": SUB["quad"]}, sk="sklearn.manifold:LocallyLinearEmbedding",
     params=dict(n_neighbors=10, n_components=2, eigen_solver="dense", random_state=SEED),
     sk_params=dict(n_neighbors=10, n_components=2, eigen_solver="dense", random_state=SEED,
                    n_jobs=-1),
     mism=["eigen_solver='dense' on both; ours ignores tol and neighbors_algorithm (exact k-NN)"])
_add("elliptic-envelope", xlane="decomp", ours="EllipticEnvelope", task="outlier", block="cls",
     sub={"X": SUB["mid"]}, sk="sklearn.covariance:EllipticEnvelope",
     params=dict(contamination=0.1, random_state=SEED))
_add("min-cov-det", xlane="decomp", ours="MinCovDet", task="covariance", block="cls",
     sub={"X": SUB["mid"]}, sk="sklearn.covariance:MinCovDet", params=dict(random_state=SEED))
_add("als", xlane="decomp", ours=("ImplicitALS", "AlternatingLeastSquares", "ALS"), kind="als",
     task="als", block="counts", datasets=("taxi-zones", "text"),
     params=dict(factors=64, regularization=0.01, alpha=1.0, iterations=15, use_cg=False, cg_steps=3,
                 calculate_training_loss=False, random_state=SEED),
     other={"implicit-cpu": "implicit", "implicit-gpu": "implicit"},
     mism=["implicit-gpu has only the conjugate-gradient solver (use_cg ignored there); ours and "
           "implicit-cpu solve each least-squares step exactly (use_cg=False)",
           "each library draws its own initial factors from random_state=7"],
     notes=["users x items = taxi (day, hour, pickup zone) x dropoff zone trip counts, and text "
            "documents x byte-bigram bins; one held-out interaction per user row for recall@10"])

# ---- lane prep (preprocessing, naive Bayes, discriminant analysis) --------
_CUP = "cuml.preprocessing:"
for _slug, _cls, _kw, _blk, _cu in (
        ("robust-scaler", "RobustScaler", {}, "raw", True),
        ("maxabs-scaler", "MaxAbsScaler", {}, "raw", True),
        ("quantile-transformer", "QuantileTransformer",
         # subsample 10**9 on every arm: every row (above the 1,000,000 fit rows, so no draw),
         # the value cuML needs (it takes no None); None on ours and scikit-learn only refused
         # the race (a library default is not a matched value, L40S 0.8.34 board)
         dict(n_quantiles=1000, output_distribution="uniform", subsample=10 ** 9, random_state=SEED),
         "raw", True),
        ("power-transformer", "PowerTransformer", dict(method="yeo-johnson", standardize=True),
         "raw", True),
        ("normalizer", "Normalizer", dict(norm="l2"), "raw", True),
        ("binarizer", "Binarizer", dict(threshold=0.0), "cls", True),
        ("poly-features", "PolynomialFeatures", dict(degree=2, include_bias=False), "raw16", True),
        ("spline", "SplineTransformer", dict(n_knots=5, degree=3), "raw16", False),
        ("kbins", "KBinsDiscretizer", dict(n_bins=16, encode="ordinal", strategy="quantile",
                                           quantile_method="linear", subsample=None,
                                           random_state=SEED), "raw", True),
        ("onehot", "OneHotEncoder", dict(handle_unknown="ignore", sparse_output=False), "cat", True),
        ("ordinal", "OrdinalEncoder", dict(handle_unknown="use_encoded_value", unknown_value=-1),
         "cat", False),
        ("variance-threshold", "VarianceThreshold", dict(threshold=0.01), "raw", False),
        # the two scalers of mojolearn.preprocessing (public since 0.8; raced from 2026-09-29)
        ("minmax-scaler", "MinMaxScaler", dict(feature_range=(0, 1), clip=False), "raw", True),
        ("standard-scaler", "StandardScaler", dict(with_mean=True, with_std=True), "raw", True)):
    # cuML takes no subsample and no quantile_method (its quantile edges are np.percentile's
    # linear ones, the method ours and scikit-learn are set to above)
    _cukw = {k: v for k, v in _kw.items() if k not in ("subsample", "quantile_method")
             and not (_cls == "KBinsDiscretizer" and k == "random_state")}
    if _cls == "QuantileTransformer":
        _cukw["subsample"] = _kw["subsample"]      # the same 10**9 (every row) as ours and scikit-learn
    if _cls == "MinMaxScaler" and _kw.get("clip") is False:
        # cuML 26.08 has no clip argument and always extrapolates outside the
        # fitted range, matching clip=False. Keep True unsupported, and retain
        # this fixed behavior in bench_board_params' estimator read-back.
        _cukw.pop("clip")
    _add(_slug, xlane="prep", ours=_cls, task="transform", block=_blk, quality="vs-sklearn",
         sk=("sklearn.feature_selection:" if _cls == "VarianceThreshold" else "sklearn.preprocessing:")
         + _cls, params=_kw, cuml=(_CUP + _cls) if _cu else None, cuml_params=_cukw,
         notes=(["the first 16 columns (PolynomialFeatures of Istella's 220 would be 24,000+ "
                 "columns)"] if _blk == "raw16" else [])
         + (["quantile_method='linear' on ours and scikit-learn: the pinned scikit-learn 1.7.2's "
             "default, and cuML's np.percentile edges (ours defaults to 1.9's "
             "'averaged_inverted_cdf')"] if _slug == "kbins" else [])
         + (["cuML 26.08 MinMaxScaler has fixed unclipped transform behavior, equivalent to "
             "clip=False; the unsupported constructor keyword is omitted"]
            if _slug == "minmax-scaler" else [])
         # FAIRNESS FIX (2026-10-13 audit): VarianceThreshold's precision difference was
         # undeclared. Results raced before this change were wrong for this lane; apply extra
         # scrutiny.
         + (["precision: scikit-learn VarianceThreshold.fit converts X to float64 "
             "(validate_data dtype=np.float64, its only path); ours float32"]
            if _slug == "variance-threshold" else []),
         # FAIRNESS FIX (2026-10-13 audit): the lane raced ours' and scikit-learn's lambda searches
         # as one algorithm; they are not, and no setting of either matches the other, so the
         # difference is declared (no harness fix). Results raced before this change were wrong
         # for this lane; apply extra scrutiny.
         mism=(["lambda search: ours golden-section, 48 steps in [-8, 8], float32 "
                "(x_prep/transform.mojo PT_ITERS); scikit-learn and cuML scipy.optimize.brent "
                "(bracket (-2, 2), tol 1.48e-8); the lambdas and outputs differ (rel 0.1 on "
                "Istella)"] if _slug == "power-transformer" else []))
_add("target-encoder", xlane="prep", ours="TargetEncoder", task="transform", block="cat", supervised=True,
     quality="vs-sklearn", sk="sklearn.preprocessing:TargetEncoder",
     # cuML benchmark TargetEncoder: shared smooth=0.0; cpu_args cv=4, random_state=42;
     # cuml_args n_folds=4, seed=42, split_method='interleaved',
     # multi_feature_mode='independent' (the lane's seed is 42, bench_board_params.LANE_SEED)
     params=dict(target_type="binary", smooth=0.0, cv=4, shuffle=True, random_state=42),
     cuml="cuml.preprocessing:TargetEncoder",
     cuml_params=dict(n_folds=4, smooth=0.0, seed=42, split_method="interleaved",
                      multi_feature_mode="independent"),
     mism=["fold assignment: cuML 'interleaved' (row i in fold i mod 4, the cuML benchmark's "
           "cuml_args); scikit-learn and ours a KFold shuffled by seed 42 (its cpu_args)"])
_add("simple-imputer", xlane="prep", ours="SimpleImputer", task="impute", block="raw",
     sk="sklearn.impute:SimpleImputer", params=dict(strategy="median"),
     cuml="cuml.preprocessing:SimpleImputer",
     notes=["10% of the cells set to NaN by a seed-7 mask; quality on those cells"])
_add("iterative-imputer", xlane="prep", ours="IterativeImputer", task="impute", block="raw32",
     sub={"X": SUB["mid"], "Xq": SUB["small"]}, sk="sklearn.impute:IterativeImputer",
     params=dict(max_iter=10, tol=1e-3, random_state=SEED, sample_posterior=False,
                 imputation_order="ascending"),
     notes=["the first 32 columns; 10% of cells NaN by a seed-7 mask"])
_add("label-encoder", xlane="prep", ours="LabelEncoder", task="labels", block="cat",
     sk="sklearn.preprocessing:LabelEncoder", cuml="cuml.preprocessing:LabelEncoder",
     notes=["y = the widest categorical column (taxi pickup zone)"])
_add("label-binarizer", xlane="prep", ours="LabelBinarizer", task="labels", block="cat",
     sk="sklearn.preprocessing:LabelBinarizer", cuml="cuml.preprocessing:LabelBinarizer")
_add("multilabel-binarizer", xlane="prep", ours="MultiLabelBinarizer", task="multilabel",
     block="cat", sub={"X": SUB["knn"], "Xq": SUB["small"]},
     sk="sklearn.preprocessing:MultiLabelBinarizer",
     notes=["each row's label set = its categorical codes, offset per column"])
for _slug, _fn, _blk, _k in (("select-f-classif", "f_classif", "cls", 0.5),
                             ("select-chi2", "chi2", "nonneg", 0.5),
                             ("select-f-regression", "f_regression", "reg", 0.5),
                             ("select-mutual-info", "mutual_info_classif", "cls", 0.5),
                             ("select-r-regression", "r_regression", "reg", 0.5),
                             ("select-mutual-info-reg", "mutual_info_regression", "reg", 0.5)):
    _add(_slug, xlane="prep", ours="SelectKBest", task="select", block=_blk, score_func=_fn,
         sub=({"X": SUB["mid"]} if _fn.startswith("mutual") else {}),
         sk="sklearn.feature_selection:SelectKBest", params=dict(k="half"),
         notes=["k = d // 2; score_func = %s (ours: mojolearn.%s)" % (_fn, _fn)])
_add("rfe", xlane="prep", ours="RFE", task="select", block="cls", sub={"X": SUB["knn"]}, quality="select-exact",
     sk="sklearn.feature_selection:RFE",
     params=dict(estimator=_E("LogisticRegression", max_iter=200), n_features_to_select="half",
                 step=0.1),
     mism=["nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion",
           "jaccard_vs_sklearn compares two selections, neither exact: on istella (200k standardized rows) neither solver converges in 200 iterations (scikit-learn warns on all 12 fits); against an exact RFE (newton-cholesky, float64, tol 1e-10) scikit-learn's selection scores 0.789 and ours 0.746 at max_iter=200, and 0.760 and 0.930 once each solver reports convergence (evidence: rfe-istella-20261012)"])
# FAIRNESS FIX (2026-10-13 audit): cuML's naive Bayes arms got only alpha (or nothing) and ran
# fit_prior, binarize, var_smoothing and norm at cuML's defaults. Each setting cuML has is now
# pinned to ours' value (cuML has no force_alpha; alpha=1.0 never reaches its 1e-10 floor).
# Results raced before this change were wrong for these lanes; apply extra scrutiny.
_NB_CUML = {"GaussianNB": dict(var_smoothing=1e-9),
            "BernoulliNB": dict(alpha=1.0, binarize=0.0, fit_prior=True),
            "CategoricalNB": dict(alpha=1.0, fit_prior=True),
            "MultinomialNB": dict(alpha=1.0, fit_prior=True),
            "ComplementNB": dict(alpha=1.0, fit_prior=True, norm=False)}
for _slug, _cls, _blk, _cu in (("gaussian-nb", "GaussianNB", "cls", True),
                               ("bernoulli-nb", "BernoulliNB", "cls", True),
                               ("categorical-nb", "CategoricalNB", "cat", True)):
    _add(_slug, xlane="prep", ours=_cls, task="clf", block=_blk, sk="sklearn.naive_bayes:" + _cls,
         cuml=("cuml.naive_bayes:" + _cls) if _cu else None, cuml_params=dict(_NB_CUML[_cls]),
         mism=(["min_categories = every code seen in X or Xq on ours and scikit-learn; "
                "cuML has no min_categories option"] if _cls == "CategoricalNB" else []))
for _slug, _cls in (("multinomial-nb", "MultinomialNB"), ("complement-nb", "ComplementNB")):
    _add(_slug, xlane="prep", ours=_cls, task="clf", block="countclf",
         datasets=("text", "taxi", "istella"), sk="sklearn.naive_bayes:" + _cls,
         cuml="cuml.naive_bayes:" + _cls, params=dict(alpha=1.0), cuml_params=dict(_NB_CUML[_cls]),
         notes=["text: byte-bigram counts of 2 KB documents, label = which corpus (enwik8 or "
                "pile_github); taxi/Istella: the nonnegative block"])
_add("lda-clf", xlane="prep", ours="LinearDiscriminantAnalysis", task="clf", block="cls",
     sk="sklearn.discriminant_analysis:LinearDiscriminantAnalysis", params=dict(solver="svd"))
_add("qda", xlane="prep", ours="QuadraticDiscriminantAnalysis", task="clf", block="cls",
     sk="sklearn.discriminant_analysis:QuadraticDiscriminantAnalysis", params=dict(reg_param=1e-3))

# ---- resampling and model selection (mojolearn.resample, cross_val_score) --
# public since 0.8 and raced from 2026-09-29: function calls, so kind "fn" (one
# call per round on each arm, the result's numbers saved for the quality pass)
_add("bootstrap", xlane="resample", ours=("resample.bootstrap",), kind="fn", task="bootstrap",
     block="reg", sub={"X": SUB["small"]},
     params=dict(statistic="mean", n_resamples=9999, confidence_level=0.95, method="percentile",
                 alternative="two-sided", random_state=SEED),
     other={"scipy-cpu": "scipy"},
     notes=["the mean of the reg target over 20,000 stride fit rows, 9,999 resamples, a 95% "
            "percentile interval; scipy.stats.bootstrap((y,), np.mean, vectorized, batch 250)"],
     mism=["each library draws its resamples from its own generator seeded 7 (ours: the Philox "
           "position map; scipy: numpy default_rng(7)), so the intervals agree to Monte Carlo "
           "error, not bit for bit"])
_add("permutation-test", xlane="resample", ours=("resample.permutation_test",), kind="fn",
     task="permutation", block="reg", sub={"X": SUB["small"], "Xq": SUB["small"]},
     params=dict(statistic="diff_means", n_resamples=9999, alternative="two-sided",
                 random_state=SEED, permutation_type="independent"),
     other={"scipy-cpu": "scipy"},
     notes=["x = the reg target on 20,000 stride fit rows, y = on 20,000 stride held-out rows; "
            "difference of means, 9,999 independent permutations; scipy.stats.permutation_test("
            "(x, y), vectorized, batch 250)"],
     mism=["each library draws its permutations from its own generator seeded 7, so the p-values "
           "agree to Monte Carlo error, not bit for bit"])
_add("resample", xlane="resample", ours=("resample.resample",), kind="fn", task="resample",
     block="reg", params=dict(replace=True, n_samples=None, random_state=SEED),
     other={"sklearn-cpu": "sklearn"},
     notes=["resample(X, y) of the reg block's 1,000,000 fit rows with replacement "
            "(sklearn.utils.resample); the gather of both arrays is timed"],
     mism=["the drawn rows are each library's own (ours the Philox position map, scikit-learn "
           "numpy RandomState(7)); quality is the resampled column means against the "
           "population's"])
_add("cross-val-score", xlane="model_selection", ours=("cross_val_score",
                                                        "model_selection.cross_val_score"),
     kind="fn", task="cv", block="reg",
     params=dict(estimator=_E("LinearRegression"), cv=5, scoring="r2"),
     other={"sklearn-cpu": "sklearn"},
     notes=["5 unshuffled k-fold splits (KFold, no seed on either side) of the reg block's "
            "1,000,000 fit rows, LinearRegression refit per fold, R2 on the held-out fold"])

# ---- lane sequence --------------------------------------------------------
_TORCH = "torch"      # opponents: torch at every fast setting of the box (bench_board_neural)
SEQ_T = 24
for _cell in ("LSTM", "GRU", "RNN"):
    for _task in ("clf", "reg"):
        _add("%s-%s" % (_cell.lower(), _task), xlane="sequence",
             ours=("%s%s" % (_cell, "Classifier" if _task == "clf" else "Regressor"),),
             kind="seqmodel", task=_task, block="seqwin", datasets=("taxi-hourly", "synthetic"),
             cell=_cell, torch=True,
             params=dict(hidden_size=64, num_layers=1, optimizer="adam", learning_rate=1e-3,
                         optimizer_options=dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
                         batch_size=256, max_epochs=2, shuffle=True, random_state=SEED,
                         **({"nonlinearity": "tanh"} if _cell == "RNN" else {})),
             notes=["windows of %d steps of the time-series block (each series z-scored by its fit "
                    "part), X (n, %d, 1); target: the next value (reg) or next value above the "
                    "window mean (clf); windows ending in the first 80%% of time fit, the rest "
                    "held out" % (SEQ_T, SEQ_T),
                    # FAIRNESS FIX (2026-10-13 audit): this note claimed the torch arm's numpy
                    # default_rng(7) weights and orders were ours' (false since 86b2bc254 /
                    # 4bdc5487e). It now says what the arm loads (ours-init). Results raced
                    # before this change were wrong for this lane; apply extra scrutiny.
                    "torch arm: nn.%s(1, 64, batch_first) + Linear head on the last hidden state, "
                    "Adam lr 1e-3 betas (0.9, 0.999) eps 1e-8 no weight decay, batch 256, 2 "
                    "epochs; its initial weights and per-epoch row orders are ours', exported "
                    "untimed by `bench_board_algos.py ours-init` under our interpreter (ours' "
                    "InitStream draws and its Feistel epoch orders) and loaded into the torch "
                    "module; without the export the torch arm is refused" % _cell])
_add("layernorm", xlane="sequence", ours=("LayerNorm", "layer_norm_forward"), kind="layer",
     task="layernorm",
     block="tensor", datasets=("synthetic",), torch=True,
     params=dict(normalized_shape=1024, eps=1e-5, elementwise_affine=True, bias=True),
     notes=["x (16384, 1024) N(0,1) seed 7"])
_add("moe", xlane="sequence", ours=("MoEBlock",), kind="layer",
     task="moe", block="tensor", datasets=("synthetic",), torch=True, forward_only=True,
     params=dict(hidden_size=1024, intermediate_size=2816, num_experts=8, top_k=2,
                 norm_topk_prob=True),
     # FAIRNESS FIX (2026-10-13 audit): the torch arm is transformers' own MixtralSparseMoeBlock now
     # (it was the harness's transcription). Results raced before this change were wrong for this
     # lane; apply extra scrutiny.
     notes=["x (8192, 1024); the torch arm is transformers' own MixtralSparseMoeBlock (softmax "
            "router, top-2, renormalized weights, SwiGLU experts, no jitter), the fixture's "
            "per-expert weights mapped into its layout",
            "forward only on every arm (ours exports no MoEBlock backward, "
            "sequence/NOT_IMPLEMENTED.tsv): the fit column times one forward, torch under "
            "no_grad; ours loads torch's weights in HF's fused layout (router (E, D), "
            "gate_up_proj (E, 2F, D) = [w1; w3], down_proj (E, D, F) = w2)"])
#: every optimizer hyperparameter passed explicitly to ours and torch.optim (ours'
#: defaults, which are torch's)
_OPT_HYPER = {
    "RMSprop": dict(alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False),
    "Adagrad": dict(lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10),
    "Adamax": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
    "NAdam": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=4e-3,
                  decoupled_weight_decay=False),
    "Adafactor": dict(beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, weight_decay=0.0, maximize=False),
    "Lion": dict(betas=(0.9, 0.99), weight_decay=0.0),
    "LAMB": dict(betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01),
    # mojolearn.training's three (the LM and Samba trainers' optimizers; top-level
    # mojolearn.SGD / Adam / AdamW), raced from 2026-09-29
    "SGD": dict(momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False, maximize=False),
    "Adam": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, maximize=False),
    "AdamW": dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, maximize=False),
}
for _slug, _cls, _torch in (("rmsprop", "RMSprop", "RMSprop"), ("adagrad", "Adagrad", "Adagrad"),
                            ("adamax", "Adamax", "Adamax"), ("nadam", "NAdam", "NAdam"),
                            ("adafactor", "Adafactor", "Adafactor"), ("lion", "Lion", None),
                            ("lamb", "LAMB", None)):
    _add(_slug, xlane="sequence", ours=_cls, kind="optim", task="optim", block="optim",
         datasets=("synthetic",), torch_opt=_torch, params=dict(lr=1e-3, **_OPT_HYPER[_cls]),
         notes=["one fp32 parameter of 16,777,216 values and 10 seed-7 gradients; timed = the 10 "
                "steps; quality = the parameter after them vs torch eager"])
for _slug, _cls in (("sgd", "SGD"), ("adam", "Adam"), ("adamw", "AdamW")):
    _add(_slug, xlane="training", ours=(_cls, "training." + _cls), kind="optim", task="optim",
         block="optim", datasets=("synthetic",), torch_opt=_cls,
         params=dict(lr=1e-3, **_OPT_HYPER[_cls]),
         notes=["one fp32 parameter of 16,777,216 values and 10 seed-7 gradients; timed = the 10 "
                "steps; quality = the parameter after them vs torch eager",
                "SGD with momentum 0.9 (coupled L2 off); Adam coupled and AdamW decoupled weight "
                "decay at torch's own defaults (0 and 0.01), every hyperparameter explicit on "
                "both sides"])
_add("mlp-clf", xlane="sequence", ours="MLPClassifier", task="clf", block="cls",
     sk="sklearn.neural_network:MLPClassifier",
     params=dict(hidden_layer_sizes=(256, 256), solver="adam", batch_size=4096, max_iter=5,
                 learning_rate_init=1e-3, random_state=SEED, shuffle=True, tol=0.0,
                 n_iter_no_change=1000))
_add("mlp-reg", xlane="sequence", ours="MLPRegressor", task="reg", block="reg",
     sk="sklearn.neural_network:MLPRegressor",
     params=dict(hidden_layer_sizes=(256, 256), solver="adam", batch_size=4096, max_iter=5,
                 learning_rate_init=1e-3, random_state=SEED, shuffle=True, tol=0.0,
                 n_iter_no_change=1000))
_TS = ("taxi-hourly", "synthetic")
_TSNOTE = ("taxi-hourly: hourly pickup counts of the 64 busiest pickup zones over January and "
           "February 2024 (1,440 hours; the last 48 held out), from the taxi npz; synthetic: 64 "
           "seed-7 series of the same length")
# FAIRNESS FIX (2026-10-13 audit): (1) the cuML arm cast Y to float64 (cuML AutoARIMA is float64
# only: cuml/tsa/auto_arima.pyx 26.08 check_array(endog, dtype="float64"), "Float64 only for now")
# on an L40S with 1/64-rate FP64, while ours runs float32 (it refuses float64): no same-precision
# comparison exists, so cuml-gpu is refused (`refused`; _build_ts also probes the dtype). (2) The
# search space differed: ours and cuML searched k = 1 only for d <= 1 (fit_intercept='auto'),
# statsforecast every order with AND without the constant (statsforecast/arima.py search_arima:
# `for K in range(max_K + 1)`), twice the candidates; ours (and a future cuML arm) now pass
# fit_intercept=[0, 1], statsforecast's grid. (3) The optimizer budget was implicit: maxiter
# (search and fit) and method are now pinned on ours and cuML (both were 1000 by default in
# cuML 26.08, ours searched at 1000 explicitly). Results raced before this change were wrong for
# this lane; apply extra scrutiny.
_ARIMA_CUML_REFUSED = ("fairness audit 2026-10-13: cuML AutoARIMA computes in float64 "
                       "only (cuml/tsa/auto_arima.pyx, cuML 26.08: check_array(endog, "
                       "dtype='float64')), ours in float32 only; FP64 runs at 1/64 rate on the "
                       "L40S, so there is no same-precision comparison")
_add("autoarima", xlane="sequence", ours="AutoARIMA", kind="ts", task="forecast", block="ts",
     datasets=_TS, sf="AutoARIMA", params=dict(max_p=3, max_q=3, max_d=1, seasonal=False, ic="aicc",
                               stepwise=False, allow_intercept=True,
                               fit_intercept=[0, 1], maxiter=1000, method="ml"),
     other={"statsforecast-cpu": "statsforecast", "cuml-gpu": "cuml"},
     refused={"cuml-gpu": _ARIMA_CUML_REFUSED}, notes=[_TSNOTE,
     "ours (and cuML when it can run float32): search(s=1, d=0..1, p=0..3, q=0..3, no seasonal "
     "part, fit_intercept=[0, 1], ic='aicc', maxiter=1000, method='ml') then fit(maxiter=1000, "
     "method='ml'), d chosen by the KPSS test; statsforecast AutoARIMA(season_length=1, "
     "seasonal=False, max_p=3, max_q=3, max_d=1, max_order=6, stepwise=False, ic='aicc', "
     "allowmean=True, allowdrift=True, approximation=False): the same exhaustive grid, every "
     "order with and without the mean (d = 0) or drift (d = 1), d by KPSS"],
     mism=["the likelihood optimizer and its stopping rule are each library's own (ours L-BFGS-B "
           "ML, maxiter 1000; statsforecast 2.1.1 CSS-ML, scipy BFGS with optim_control "
           "maxiter 100; it has no maxiter argument on AutoARIMA)",
           "precision: statsforecast computes in float64 on the CPU (its only precision), ours "
           "float32 on the GPU",
           "the no-constant white-noise / random-walk order (0, d, 0) without constant: "
           "statsforecast fits it, ours (cuML's rule) skips p + q + k = 0",
           "ours adopts each chosen order's search fit (same start, optimizer and maxiter) instead "
           "of refitting it in fit(); statsforecast also keeps the search's fit",
           "KPSS d choice: ours cuML's KPSS rule, statsforecast its ndiffs (KPSS, alpha 0.05)"])
_add("stl", xlane="sequence", ours="STL", kind="ts", task="decompose", block="ts", datasets=_TS,
     params=dict(period=24, robust=False, seasonal=7, trend=None, low_pass=None, seasonal_deg=1,
                 trend_deg=1, low_pass_deg=1, seasonal_jump=1, trend_jump=1, low_pass_jump=1), other={"statsmodels-cpu": "statsmodels"}, notes=[_TSNOTE])
_add("var", xlane="sequence", ours="VAR", kind="ts", task="var", block="ts", datasets=_TS,
     params=dict(maxlags=2, method="ols", ic=None, trend="c"), other={"statsmodels-cpu": "statsmodels"},
     notes=[_TSNOTE, "one VAR over 16 series jointly (the first 16), lag order 2"])
_add("theta", xlane="sequence", ours=("Theta",), kind="ts", task="forecast", sf="Theta",
     block="ts", datasets=_TS, params=dict(season_length=24, decomposition_type="multiplicative"),
     other={"statsforecast-cpu": "statsforecast", "statsmodels-cpu": "statsmodels"},
     notes=[_TSNOTE, "statsforecast Theta (the standard theta model, STM) on both sides; "
                     "statsmodels-cpu is ThetaModel(period=24), its own theta method"])
_add("croston", xlane="sequence", ours=("CrostonClassic",), kind="ts", task="forecast",
     sf="CrostonClassic", block="tsi", datasets=("taxi-hourly", "synthetic"), params={},
     mism=["CrostonClassic has no tuning parameter and no seed on either side (smoothing 0.1, "
           "fixed)"],
     other={"statsforecast-cpu": "statsforecast"},
     notes=["intermittent series: taxi-hourly's 64 pickup zones with 30-70% zero hours; "
            "synthetic Bernoulli(0.3) x Poisson(3) demand, seed 7"])
#: the rest of statsforecast's theta family and Croston's two variants, one lane
#: each, ours and statsforecast with the same arguments (2026-09-29)
for _slug, _cls, _kw, _why in (
        ("optimized-theta", "OptimizedTheta", {}, "OTM: theta optimized by MSE"),
        ("dynamic-theta", "DynamicTheta", {}, "DSTM: the dynamic standard theta model"),
        ("dynamic-optimized-theta", "DynamicOptimizedTheta", {},
         "DOTM: the dynamic optimized theta model"),
        ("auto-theta", "AutoTheta", {"model": None},
         "model=None on both: STM, OTM, DSTM and DOTM fitted, the least in-sample MSE kept")):
    _add(_slug, xlane="sequence", ours=(_cls,), kind="ts", task="forecast", sf=_cls, block="ts",
         datasets=_TS, params=dict(season_length=24, decomposition_type="multiplicative", **_kw),
         other={"statsforecast-cpu": "statsforecast"},
         notes=[_TSNOTE, "statsforecast %s(season_length=24, decomposition_type="
                         "'multiplicative'%s) on both sides; %s" % (
                             _cls, ", model=None" if _kw else "", _why)])
for _slug, _cls, _why in (("croston-optimized", "CrostonOptimized",
                           "the smoothing of demand and of the intervals optimized separately"),
                          ("croston-sba", "CrostonSBA",
                           "Syntetos-Boylan: CrostonClassic's forecast times 0.95")):
    _add(_slug, xlane="sequence", ours=(_cls,), kind="ts", task="forecast", sf=_cls, block="tsi",
         datasets=("taxi-hourly", "synthetic"), params={},
         mism=["%s has no tuning parameter and no seed on either side" % _cls],
         other={"statsforecast-cpu": "statsforecast"},
         notes=["intermittent series: taxi-hourly's 64 pickup zones with 30-70% zero hours; "
                "synthetic Bernoulli(0.3) x Poisson(3) demand, seed 7", _why])
_add("damped-ets", xlane="sequence", ours=("ETS",), kind="ts", task="forecast", sf="AutoETS",
     block="ts", datasets=_TS, params=dict(season_length=1, model="AAN", damped=True),
     other={"statsmodels-cpu": "statsmodels", "statsforecast-cpu": "statsforecast"},
     notes=[_TSNOTE, "ETS(A,Ad,N), Holt's damped additive trend, no season: ours refuses seasonal "
                     "ETS by name, so every arm fits the non-seasonal model; statsforecast "
                     "AutoETS(model='AAN', damped=True), statsmodels ExponentialSmoothing("
                     "trend='add', damped_trend=True, seasonal=None)"])
_add("garch", xlane="sequence", ours="GARCH", kind="ts", task="garch", block="tsr", datasets=_TS,
     params=dict(p=1, o=0, q=1, power=2.0, mean="Constant", dist="normal"),
     other={"arch-cpu": "arch"},
     notes=["taxi-hourly: first differences of log(1 + count); synthetic: GARCH(1,1) returns "
            "omega 0.1 alpha 0.1 beta 0.8, seed 7",
            "arch_model(vol='GARCH', p=1, o=0, q=1, mean='Constant', dist='normal', "
            "rescale=False) per series (ours refuses rescaling); the forecast is the "
            "conditional variance, h steps"])
_add("prophet", xlane="sequence", ours=("ProphetForecaster",), kind="ts",
     task="forecast", block="ts", datasets=_TS,
     params=dict(growth="linear", n_changepoints=25, changepoint_range=0.8,
                 daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False,
                 seasonality_mode="additive", seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0, changepoint_prior_scale=0.05, max_iter=10000),
     other={"prophet-cpu": "prophet"},
     notes=[_TSNOTE, "prophet's shape on both sides: fit(ds, y), predict(future ds), ds hourly "
                     "from 2024-01-01; ours fits the batch of series at once, prophet one "
                     "series per job; linear growth, additive seasonality, 25 changepoints over "
                     "the first 80%, daily (order 4) and weekly (order 3) seasonality, no "
                     "yearly; the iteration cap is prophet's (Stan iter=1e4) on both"],
     mism=["prophet fits by Stan's L-BFGS (MAP); ours by its own L-BFGS; parity is at a "
           "tolerance, the forecast RMSE is the comparable number",
           "prophet uncertainty_samples=0: ours computes no intervals, so prophet's 1,000 "
           "sampled intervals (prophet only, numpy's unseeded RNG) are switched off"])

# ---- lane trees -----------------------------------------------------------
_add("decision-tree-clf", xlane="trees", ours="DecisionTreeClassifier", task="clf", block="cls",
     sk="sklearn.tree:DecisionTreeClassifier", params=dict(max_depth=16, max_features=1.0, random_state=SEED),
     cuml="cuml.ensemble:RandomForestClassifier",
     cuml_params=dict(n_estimators=1, bootstrap=False, max_features=1.0, max_depth=16, n_bins=128,
                      random_state=SEED),
     mism=["cuml-gpu is cuML's forest with one tree, no bootstrap, every feature (no GPU "
           "single-tree class exists), n_bins=128 as ours",
           "ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
_add("decision-tree-reg", xlane="trees", ours="DecisionTreeRegressor", task="reg", block="reg",
     sk="sklearn.tree:DecisionTreeRegressor", params=dict(max_depth=16, max_features=1.0, random_state=SEED),
     cuml="cuml.ensemble:RandomForestRegressor",
     cuml_params=dict(n_estimators=1, bootstrap=False, max_features=1.0, max_depth=16, n_bins=128,
                      random_state=SEED),
     mism=["cuml-gpu is cuML's forest with one tree (see decision-tree-clf)", "ours' DecisionTree* is its forest builder with one tree: it splits on n_bins quantile bins per feature (128; 256 in FAST, rf_dt_default_bins) (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
_add("bagging-clf", xlane="trees", ours="BaggingClassifier", task="clf", block="cls",
     sk="sklearn.ensemble:BaggingClassifier",
     params=dict(estimator=_E("DecisionTreeClassifier", max_depth=12, random_state=SEED),
                 n_estimators=10, random_state=SEED), sk_extra=dict(n_jobs=-1),
     mism=["ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
_add("bagging-reg", xlane="trees", ours="BaggingRegressor", task="reg", block="reg",
     sk="sklearn.ensemble:BaggingRegressor",
     params=dict(estimator=_E("DecisionTreeRegressor", max_depth=12, random_state=SEED),
                 n_estimators=10, random_state=SEED), sk_extra=dict(n_jobs=-1),
     mism=["ours' DecisionTree* is its forest builder with one tree: it splits on n_bins quantile bins per feature (128; 256 in FAST, rf_dt_default_bins) (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
_add("adaboost-clf", xlane="trees", ours="AdaBoostClassifier", task="clf", block="cls",
     sk="sklearn.ensemble:AdaBoostClassifier",
     params=dict(estimator=_E("DecisionTreeClassifier", max_depth=3, random_state=SEED),
                 n_estimators=50, learning_rate=1.0, algorithm="SAMME", random_state=SEED),
     mism=["ours' DecisionTree* is its forest builder with one tree: it splits on n_bins=128 quantile bins per feature (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
_add("adaboost-reg", xlane="trees", ours="AdaBoostRegressor", task="reg", block="reg",
     sk="sklearn.ensemble:AdaBoostRegressor",
     params=dict(estimator=_E("DecisionTreeRegressor", max_depth=3, random_state=SEED),
                 n_estimators=50, learning_rate=1.0, loss="linear", random_state=SEED),
     mism=["ours' DecisionTree* is its forest builder with one tree: it splits on n_bins quantile bins per feature (128; 256 in FAST, rf_dt_default_bins) (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
#: DART: ours takes LightGBM's DART parameter set; every one is passed explicitly
#: to ours and LightGBM, and the ones XGBoost has to XGBoost (_build_dart).
_DART = dict(n_estimators=200, learning_rate=0.1, max_depth=8, num_leaves=255, drop_rate=0.1,
             skip_drop=0.5, max_drop=50, xgboost_dart_mode=False, uniform_drop=False,
             min_child_samples=20, reg_lambda=0.0, reg_alpha=0.0, max_bin=255, max_delta_step=0.0,
             subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
             drop_seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED, random_state=SEED)
_DART_MISM = (
    "LightGBM boosting='dart' grows leaf-wise (num_leaves=255, max_depth=8) as ours; XGBoost "
    "booster='dart' tree_method='hist' grows depth-wise (max_depth=8, no leaf cap)",
    "XGBoost has no max_drop (ours and LightGBM 50), no min_child_samples (ours and LightGBM "
    "20 rows; XGBoost min_child_weight=1, a hessian sum), no uniform_drop / xgboost_dart_mode "
    "(XGBoost sample_type='uniform', normalize_type='tree') and no drop_seed: its drops come "
    "from random_state=7; each library's drop RNG is its own",
    "max_bin 255 on every arm (XGBoost's default is 256, set to 255 here)")
_add("dart", xlane="trees", ours=("DARTClassifier", "DartClassifier"), kind="dart", task="clf",
     block="cls",
     params=dict(_DART),
     other={"lightgbm-cpu": "lightgbm", "xgboost-cpu": "xgboost", "xgboost-gpu": "xgboost"},
     mism=list(_DART_MISM))
_add("dart-reg", xlane="trees", ours=("DARTRegressor",), kind="dart", task="reg", block="reg",
     params=dict(_DART),
     other={"lightgbm-cpu": "lightgbm", "xgboost-cpu": "xgboost", "xgboost-gpu": "xgboost"},
     mism=list(_DART_MISM))
_add("random-trees-embedding", xlane="trees", ours="RandomTreesEmbedding", task="transform",
     block="cls", quality="shape", sk="sklearn.ensemble:RandomTreesEmbedding",
     params=dict(n_estimators=10, max_depth=5, random_state=SEED, sparse_output=False),
     sk_params=dict(n_estimators=10, max_depth=5, random_state=SEED, sparse_output=False, n_jobs=-1))
_add("voting-clf", xlane="trees", ours="VotingClassifier", task="clf", block="cls",
     sk="sklearn.ensemble:VotingClassifier",
     params=dict(estimators=[("lr", _E("LogisticRegression", max_iter=200)), ("nb", _E("GaussianNB")),
                             ("dt", _E("DecisionTreeClassifier", max_depth=8, random_state=SEED))],
                 voting="soft"),
     mism=["nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion"])
_add("voting-reg", xlane="trees", ours="VotingRegressor", task="reg", block="reg",
     sk="sklearn.ensemble:VotingRegressor",
     params=dict(estimators=[("ridge", _E("Ridge", alpha=1.0)), ("lasso", _E("Lasso", alpha=0.01, tol=1e-3, max_iter=1000)),
                             ("dt", _E("DecisionTreeRegressor", max_depth=8, random_state=SEED))]),
     mism=["nested DecisionTreeRegressor(max_depth=8): ours is its forest builder with one tree, splitting on n_bins quantile bins per feature (128; 256 in FAST, rf_dt_default_bins) (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
_add("stacking-clf", xlane="trees", ours="StackingClassifier", task="clf", block="cls",
     sk="sklearn.ensemble:StackingClassifier",
     params=dict(estimators=[("nb", _E("GaussianNB")),
                             ("dt", _E("DecisionTreeClassifier", max_depth=8, random_state=SEED))],
                 final_estimator=_E("LogisticRegression", max_iter=200), cv=5),
     mism=["nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion"])
_add("stacking-reg", xlane="trees", ours="StackingRegressor", task="reg", block="reg",
     sk="sklearn.ensemble:StackingRegressor",
     params=dict(estimators=[("lasso", _E("Lasso", alpha=0.01, tol=1e-3, max_iter=1000)),
                             ("dt", _E("DecisionTreeRegressor", max_depth=8, random_state=SEED))],
                 final_estimator=_E("Ridge", alpha=1.0), cv=5),
     mism=["nested DecisionTreeRegressor(max_depth=8): ours is its forest builder with one tree, splitting on n_bins quantile bins per feature (128; 256 in FAST, rf_dt_default_bins) (ours only; not a scikit-learn parameter), scikit-learn on exact thresholds"])
_add("multioutput-clf", xlane="trees", ours="MultiOutputClassifier", task="multiclf", block="cls",
     sk="sklearn.multioutput:MultiOutputClassifier",
     params=dict(estimator=_E("LogisticRegression", max_iter=200)),
     notes=["targets: y and (last column > 0); that column is removed from X"],
     mism=["nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion"])
_add("multioutput-reg", xlane="trees", ours="MultiOutputRegressor", task="multireg", block="reg",
     sk="sklearn.multioutput:MultiOutputRegressor", params=dict(estimator=_E("Ridge", alpha=1.0)),
     notes=["targets: y and the last column; that column is removed from X"])
_add("ovr", xlane="trees", ours="OneVsRestClassifier", task="clf", block="mc",
     sk="sklearn.multiclass:OneVsRestClassifier",
     params=dict(estimator=_E("LogisticRegression", max_iter=200)),
     notes=["taxi: the tip-share band (4 classes); Istella-S: the relevance grade (5 classes)"],
     mism=["nested LogisticRegression(max_iter=200): ours is L-BFGS (cuML's quasi-Newton, memory 5, solver='qn'), scikit-learn solver='lbfgs' (scipy, memory 10); C=1.0 and tol=1e-4 are both libraries' defaults; each stops by its own criterion"])
_add("calibrated", xlane="trees", ours="CalibratedClassifierCV", task="clf", block="cls",
     sk="sklearn.calibration:CalibratedClassifierCV",
     params=dict(estimator=_E("GaussianNB"), method="sigmoid", cv=5, ensemble=True))
# FAIRNESS FIX (2026-10-13 audit, second pass), the trees expansion lanes above:
# - scikit-learn's trees search EXACT thresholds; ours' DecisionTree* (alone, or nested in
#   bagging, AdaBoost, voting and stacking) splits on 128 quantile bins per feature (256 in FAST).
#   That is a different split algorithm (a sort per node against a histogram pass) and a
#   different model, raced as if comparable; the scikit-learn arm of each such lane is refused
#   (cuML's one-tree forest, the same binned algorithm, still races decision-tree-* on NVIDIA).
# - voting-reg and stacking-reg passed random_state=7 to the nested Lasso, which ours refuses
#   (cyclic coordinate descent draws nothing) so ours never ran; scikit-learn's cyclic Lasso
#   reads none either, so it is dropped on both arms (in the _add calls above).
# - scikit-learn's meta-estimators ran at n_jobs None, one job, while its forest and bagging
#   arms already take every core: calibrated, multioutput-* and ovr now pass n_jobs=-1.
# - dart / dart-reg: XGBoost's DART grows depth-wise with no max_drop, no min_child_samples
#   (min_child_weight 1.0 instead) and its own drop schedule, against ours' and LightGBM's
#   leaf-wise DART at the same 20-row floor: different settings and growth; refused.
# Results raced before this change were wrong for these lanes; apply extra scrutiny.
_DT_EXACT_REFUSED = ("different algorithm: scikit-learn's trees search exact thresholds; ours' "
                     "DecisionTree splits on 128 quantile bins per feature (256 in FAST)")
for _lane in ("decision-tree-clf", "decision-tree-reg", "bagging-clf", "bagging-reg",
              "adaboost-clf", "adaboost-reg", "voting-clf", "voting-reg", "stacking-clf",
              "stacking-reg"):
    LANES[_lane]["refused"]["sklearn-cpu"] = _DT_EXACT_REFUSED
for _lane in ("calibrated", "multioutput-clf", "multioutput-reg", "ovr"):
    LANES[_lane].setdefault("sk_extra", {})["n_jobs"] = -1
_DART_XGB_REFUSED = ("different settings and growth: XGBoost's DART grows depth-wise with no "
                     "max_drop and no min_child_samples (min_child_weight 1.0); ours and LightGBM "
                     "grow leaf-wise DART at max_drop 50 and a 20-row floor")
for _lane in ("dart", "dart-reg"):
    for _arm in ("xgboost-cpu", "xgboost-gpu"):
        LANES[_lane]["refused"][_arm] = _DART_XGB_REFUSED
del _lane, _arm

# FAIRNESS FIX (2026-10-13 audit): tree-shap raced ours explaining its own RandomForestRegressor
# with 100-row background covers against XGBoost (and shap, LightGBM) explaining their own GBDT with
# training covers: different models, different covers. No opponent here can explain ours' forest with
# ours' covers through its public API (ours' TreeExplainer takes mojolearn models only), so every
# opponent arm is REFUSED by name in _build_shap; `explained_model` names the model so a copy of the
# old opponent cells is withheld by the board's settings check. Results raced before this change were
# wrong for this lane; apply extra scrutiny.
_add("tree-shap", xlane="trees", ours="TreeExplainer", kind="shap", task="tree-shap", block="reg",
     sub={"X": SUB["mid"], "Xq": SUB["quad"]},
     params=dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                 explained_model="ours' RandomForestRegressor, covers = 100 background rows"),
     other={"shap-cpu": "shap", "xgboost-cpu": "xgboost", "xgboost-gpu": "xgboost",
            "lightgbm-cpu": "lightgbm"},
     notes=["ours explains its RandomForestRegressor (100 trees, depth 6) fit before the clock, node "
            "covers from 100 background rows; timed = the SHAP values of 10,000 rows. Every opponent "
            "is refused: none explains that model with those covers (2026-10-13 audit)"])
# FAIRNESS FIX (2026-10-13 audit, second pass): the tree-shap refusal also goes through the
# shared `refused` table, so the opponents are planned out and listed as refused by name.
# Results raced before this change were wrong for this lane; apply extra scrutiny.
for _arm in ("shap-cpu", "xgboost-cpu", "xgboost-gpu", "lightgbm-cpu"):
    LANES["tree-shap"]["refused"][_arm] = (
        "a different model: ours explains its RandomForestRegressor with node covers from 100 "
        "background rows; this arm explains its own library's GBDT with training covers")
del _arm
# lane shap-xvendor-identity (2026-10-08): ours explains this library's device ridge.
# FAIRNESS FIX (2026-10-13 audit): the opponents explained a different model (the NumPy closed form
# of the ridge, host-BLAS coefficients) and cuML evaluated it as a float64 cupy GEMV (FP64 on the
# L40S), while ours explained its own float32 device RidgeCV. Every arm now explains ONE float32
# linear model whose coefficients the harness solves in a fixed order (_shared_ridge_f32, the same
# bits on every box): ours through mojolearn.RidgeCV's device predict holding those coefficients,
# cuML through a float32 cupy GEMV, shap through a float32 NumPy GEMV. Results raced before this
# change were wrong for this lane; apply extra scrutiny.
_SHAP_RIDGE_MODEL = ("one float32 ridge (alpha 1, centered) on every arm, coefficients solved once "
                     "in a fixed order by the harness")
_add("kernel-shap", xlane="trees", ours="KernelExplainer", kind="shap", task="kernel-shap",
     block="reg", sub={"X": SUB["mid"], "Xq": 100},
     params=dict(n_background=100, nsamples=2048, link="identity", l1_reg=False,
                 explained_model=_SHAP_RIDGE_MODEL),
     other={"shap-cpu": "shap", "cuml-gpu": "cuml"},
     notes=["the model is a ridge (alpha 1, centered) solved before the clock in a fixed order on the "
            "host (no BLAS), its coefficients rounded to float32 (its exact SHAP values are known: "
            "w_j (x_j - E x_j)); background = 100 stride rows; every arm evaluates that one float32 model: "
            "ours mojolearn.RidgeCV(alphas=[1.0]).predict on the device holding those coefficients, cuML a "
            "float32 cupy GEMV, shap a float32 NumPy GEMV"])
_add("permutation-shap", xlane="trees", ours="PermutationExplainer", kind="shap",
     task="permutation-shap", block="reg", sub={"X": SUB["mid"], "Xq": 100},
     params=dict(n_background=100, npermutations=10, explained_model=_SHAP_RIDGE_MODEL),
     other={"shap-cpu": "shap", "cuml-gpu": "cuml"},
     notes=["the ridge model as kernel-shap"])

# ---- lane cnn -------------------------------------------------------------
_CNN = "seeded N(0,1) tensors (no image set is in the R2 store)"
_CONV = dict(stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros")
_add("conv1d", xlane="cnn", ours="Conv1d", kind="layer", task="conv1d", block="tensor",
     datasets=("synthetic",), torch=True,
     params=dict(in_channels=128, out_channels=128, kernel_size=3, padding=1, **_CONV),
     notes=["x (64, 128, 4096), " + _CNN])
_add("conv2d", xlane="cnn", ours="Conv2d", kind="layer", task="conv2d", block="tensor",
     datasets=("synthetic",), torch=True,
     params=dict(in_channels=64, out_channels=64, kernel_size=3, padding=1, **_CONV),
     notes=["x (64, 64, 56, 56), " + _CNN])
_add("maxpool2d", xlane="cnn", ours="MaxPool2d", kind="layer", task="maxpool2d", block="tensor",
     datasets=("synthetic",), torch=True,
     params=dict(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     notes=["x (64, 64, 112, 112), " + _CNN])
_add("avgpool2d", xlane="cnn", ours="AvgPool2d", kind="layer", task="avgpool2d", block="tensor",
     datasets=("synthetic",), torch=True,
     params=dict(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None),
     notes=["x (64, 64, 112, 112), " + _CNN])
_add("maxpool1d", xlane="cnn", ours="MaxPool1d", kind="layer", task="maxpool1d", block="tensor",
     datasets=("synthetic",), torch=True,
     params=dict(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
     notes=["x (64, 64, 4096), " + _CNN])
_add("avgpool1d", xlane="cnn", ours="AvgPool1d", kind="layer", task="avgpool1d", block="tensor",
     datasets=("synthetic",), torch=True,
     params=dict(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True),
     notes=["x (64, 64, 4096), " + _CNN])
_add("batchnorm1d", xlane="cnn", ours="BatchNorm1d", kind="layer", task="batchnorm1d",
     block="tensor", datasets=("synthetic",), torch=True,
     params=dict(num_features=256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
     notes=["x (64, 256, 1024), training-mode statistics, " + _CNN])
_add("cnn-clf", xlane="cnn", ours="CNNClassifier", kind="cnnclf", task="clf", block="images",
     datasets=("synthetic",), torch=True,
     params=dict(input_shape=(1, 28, 28), conv_channels=(8, 16), kernel_size=3, pool_size=2,
                 learning_rate=0.01, momentum=0.9, weight_decay=0.0, dampening=0.0, nesterov=False,
                 optimizer="sgd", batch_size=128, max_iter=2, shuffle=True, random_state=SEED),
     notes=["seed-7 synthetic 1 x 28 x 28 images of 10 classes (a class template plus N(0, 1) "
            "noise), 20,000 fit and 5,000 held out: the R2 store holds no image set, so this is "
            "the one kind",
            # FAIRNESS FIX (2026-10-13 audit): this note claimed the torch arm's numpy weights
            # and orders were ours' (false since 86b2bc254). It now says what the arm loads
            # (ours-init). Results raced before this change were wrong for this lane; apply
            # extra scrutiny.
            "torch arm: the same net (Conv2d-ReLU-MaxPool2d per entry of conv_channels, then "
            "Linear), SGD lr 0.01 momentum 0.9 dampening 0 no nesterov no weight decay, cross "
            "entropy, the same epochs and batch size; its initial weights and per-epoch row "
            "orders are ours', exported untimed by `bench_board_algos.py ours-init` under our "
            "interpreter (ours' InitStream draws and its Feistel epoch orders) and loaded into "
            "the torch module; without the export the torch arm is refused"])
_add("batchnorm2d", xlane="cnn", ours="BatchNorm2d", kind="layer", task="batchnorm2d",
     block="tensor", datasets=("synthetic",), torch=True,
     params=dict(num_features=64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
     notes=["x (64, 64, 56, 56), training-mode statistics, " + _CNN])
_add("dropout2d", xlane="cnn", ours="Dropout2d", kind="layer", task="dropout2d", block="tensor",
     datasets=("synthetic",), torch=True, params=dict(p=0.1), quality="dropout",
     notes=["x (64, 64, 56, 56); each library's own RNG; score channel-wise inverted dropout and the declared drop probability"])
_add("global-avgpool", xlane="cnn", ours=("AdaptiveAvgPool2d", "GlobalAvgPool2d"), kind="layer",
     task="gap", block="tensor", datasets=("synthetic",), torch=True, params=dict(output_size=1),
     notes=["x (64, 512, 7, 7)"])
_add("global-maxpool", xlane="cnn", ours=("AdaptiveMaxPool2d", "GlobalMaxPool2d"), kind="layer",
     task="gmp", block="tensor", datasets=("synthetic",), torch=True, params=dict(output_size=1),
     notes=["x (64, 512, 7, 7)"])
_add("resnet-block", xlane="cnn", ours=("BasicBlock", "ResNetBasicBlock"), kind="layer",
     task="resnet", block="tensor", datasets=("synthetic",), torch=True,
     params=dict(inplanes=64, planes=64),
     # FAIRNESS FIX (2026-10-13 audit): the torch arm is torchvision's own BasicBlock now (it was
     # the harness's transcription). Results raced before this change were wrong for this lane;
     # apply extra scrutiny.
     notes=["torch arm: torchvision.models.resnet.BasicBlock (stride 1, no downsample), torch's "
            "seeded init via the neural fixture; ours loads the same weights",
            "torchvision BasicBlock semantics (conv3x3-BN-ReLU-conv3x3-BN + identity, ReLU), "
            "x (64, 64, 56, 56)"])
# FAIRNESS FIX (2026-10-13 audit): the torch GCNConv rebuilt the normalized graph on every call
# (cached=False) while ours caches it; the torch arm also skipped grad_x on both graph lanes
# while ours computes it; ours' output was never comparable (its own random weights). Now: PyG's
# cached=True (torch only; ours always caches), x.requires_grad on both lanes, and ours loads
# the fixture weights (torch's seeded init) so its y is compared with torch's. The notes below
# say so, which also changes the store's settings key. Results raced before this change were
# wrong for this lane; apply extra scrutiny.
GCN_TORCH_KW = {"cached": True}
_GRAPH_FAIR = ("both arms compute the full backward of forward + backward(dy): the weight and "
               "bias gradients and grad_x (torch: x.requires_grad); ours loads torch's seeded "
               "initial weights from the neural fixture, so its y is compared with torch's "
               "(2026-10-13 fairness audit)")
_add("gcn", xlane="cnn", ours=("GCNConv", "GCN"), kind="layer", task="gcn", block="graph",
     torch=True, params=dict(out_channels=128, improved=False, add_self_loops=True, normalize=True,
                             bias=True),
     notes=["the kNN graph and its node features; PyG GCNConv (normalize=True, self loops)",
            "the normalized graph is cached on both arms after the first (warm-up) call: ours' "
            "GCNConv caches it per edge_index, torch's is PyG GCNConv(cached=True) (a torch-only "
            "argument; ours has no `cached`)", _GRAPH_FAIR])
_add("graphsage", xlane="cnn", ours=("SAGEConv", "GraphSAGE"), kind="layer", task="sage",
     block="graph", torch=True,
     params=dict(out_channels=128, aggr="mean", normalize=False, root_weight=True, project=False,
                 bias=True),
     notes=["the kNN graph and its node features; PyG SAGEConv (mean aggregation)",
            "ours keeps its CSR view of edge_index between calls; PyG aggregates from edge_index "
            "by scatter and needs no such view", _GRAPH_FAIR])

_add("embedding", xlane="embedding", ours=("Embedding",), kind="layer", task="embedding",
     block="tensor", datasets=("synthetic",), torch=True,
     params=dict(num_embeddings=32768, embedding_dim=1024, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False),
     notes=["ids (64, 512) uniform over the 32,768 rows, seed 7; the table is torch's "
            "nn.Embedding seeded N(0,1) init, handed to ours as weight= (ours refuses to draw "
            "one); training column = forward + the dense (V, d) weight gradient of dy",
            "fp32 torch arms only: a gather has no matmul for TF32 and autocast leaves "
            "nn.Embedding in fp32"])

# ---- lane ann -------------------------------------------------------------
_ANN = ("the classical knn lane's block (tools/knn_datasets.real_block, the classical2 ivf "
        "block): 400,000 index rows, 4,000 queries, raw")
# FAIRNESS FIX (2026-10-13 audit): the opponents' coarse-quantizer training
# sets were their library defaults (cuVS ivf_sq 262,144 rows against ours' all
# 400,000 at de2b2b739). Declared here, set in _build_ann. Results raced before
# this change were wrong for these lanes; apply extra scrutiny.
_ANN_TRAIN = ("coarse k-means training rows: ours' count on every arm (ours every s-th row, "
              "ivf_trainset_stride: 200,000 of 400,000; cuVS kmeans_trainset_fraction = that "
              "count / n (ivf_pq), max_train_points_per_cluster = count // n_lists (ivf_sq, "
              "199,680 rows); FAISS max_points_per_centroid = count // n_lists, its seeded "
              "sample); 20 k-means iterations on every arm. cuVS ivf_sq trains its "
              "scalar-quantizer ranges on that sample, ours over every row")
# FAIRNESS FIX (2026-10-13 audit): AMD raced faiss on the CPU though hipVS, AMD's port of cuVS
# with the same ivf_pq API, is a library IVF-PQ for the AMD GPU; hipvs-gpu is its arm (AMD only,
# the cuvs-gpu code path with the same pinned settings). Results raced before this change were
# wrong for this lane; apply extra scrutiny.
_add("ivf-pq", xlane="ann", ours=("IVFPQIndex",), kind="ann", task="ivf-pq", block="ivf",
     params=dict(n_lists=1024, n_probes=32, n_neighbors=10, pq_bits=8, kmeans_n_iters=20,
                 random_state=SEED),
     other={"faiss-cpu": "faiss", "cuvs-gpu": "cuvs", "hipvs-gpu": "cuvs"},
     notes=[_ANN, "pq_dim = d when d <= 16 else d // 4 (taxi 11, Istella 55); no refine"], mism=[_ANN_TRAIN])
# FAIRNESS FIX (2026-10-13 audit): faiss-cpu raced HNSW (IndexHNSWFlat) against ours' CAGRA: a
# different graph index. It is refused; the CAGRA opponents are cuVS on NVIDIA and hipVS (AMD's
# cuVS port, the same cagra API) on AMD; Apple has none. Results raced before this change were
# wrong for this lane; apply extra scrutiny.
_add("cagra", xlane="ann", ours=("CAGRAIndex", "CagraIndex"), kind="ann", task="cagra", block="ivf",
     params=dict(graph_degree=32, intermediate_graph_degree=64, itopk_size=64, n_neighbors=10,
                 random_state=SEED),
     other={"faiss-cpu": "faiss", "cuvs-gpu": "cuvs", "hipvs-gpu": "cuvs"},
     refused={"faiss-cpu": "a different algorithm: faiss-cpu's graph index is HNSW "
                           "(IndexHNSWFlat), not CAGRA; FAISS builds CAGRA only on an NVIDIA GPU "
                           "(2026-10-13 fairness audit)"},
     notes=[_ANN], mism=["cuvs-gpu (NVIDIA) and hipvs-gpu (AMD, hipVS) are CAGRA itself"])
_add("tsne", xlane="ann", ours="TSNE", task="embed", block="manifold",
     sk="sklearn.manifold:TSNE",
     params=dict(n_components=2, perplexity=30.0, early_exaggeration=12.0, max_iter=1000,
                 learning_rate="auto", init="seeded", random_state=SEED),
     sk_params=dict(n_components=2, perplexity=30.0, early_exaggeration=12.0, max_iter=1000,
                    learning_rate="auto", init="seeded", random_state=SEED, n_jobs=-1,
                    method="barnes_hut", metric="euclidean", n_iter_without_progress=1000,
                    min_grad_norm=0.0),
     cuml="cuml.manifold:TSNE",
     # FAIRNESS FIX (2026-10-13 audit): cuML raced method='fft' (an O(n)
     # approximation of the repulsion; ours computes it exactly, O(n^2)) with
     # learning_rate_method='adaptive', which at n = 20,000 rewrote its rate to
     # 6,667, its early exaggeration to 24 and its k-NN to 78 (t_sne.pyx
     # _init_params) against ours' 416.7, 12 and 91, and it could stop early on
     # min_grad_norm=1e-7. cuML now runs its own exact algorithm (method='exact':
     # attractive forces over the sparse k-NN P, exact all-pairs repulsion,
     # exact_tsne.cuh, the algorithm ours implements), adaptive off, ours'
     # resolved rate, 12, 91 neighbours ("tsne_auto" / "tsne_k" resolved in
     # _derived_params), ours' 250 exaggeration steps and 0.5 / 0.8 momentum,
     # and min_grad_norm=0.0 so it runs exactly max_iter steps as ours does.
     # Results raced before this change were wrong for this lane; apply extra
     # scrutiny.
     # FAIRNESS FIX (2026-10-13 audit): scikit-learn raced Barnes-Hut t-SNE (an approximate,
     # O(n log n) repulsion) against ours' exact repulsion; its method='exact' is another
     # algorithm again (a dense all-pairs P, not the sparse k-NN P). No CPU library implements
     # ours' algorithm, so scikit-learn is refused; cuML (method='exact') is the opponent on
     # NVIDIA, and AMD and Apple have none. Results raced before this change were wrong for this
     # lane; apply extra scrutiny.
     refused={"sklearn-cpu": "a different algorithm: scikit-learn's t-SNE is Barnes-Hut "
                             "(approximate repulsion) or exact over a dense all-pairs P; ours "
                             "is exact repulsion over the sparse k-NN P (cuML method='exact' is "
                             "the same algorithm, NVIDIA only) (2026-10-13 fairness audit)"},
     cuml_params=dict(n_components=2, perplexity=30.0, early_exaggeration=12.0,
                      late_exaggeration=1.0, max_iter=1000, learning_rate_method="none",
                      learning_rate="tsne_auto", n_neighbors="tsne_k", method="exact",
                      exaggeration_iter=250, pre_momentum=0.5, post_momentum=0.8,
                      min_grad_norm=0.0, perplexity_max_iter=100, metric="euclidean",
                      square_distances=True, init="random", random_state=SEED),
     mism=["gradients: ours and cuML exact repulsion over k-NN affinities (cuML method='exact'; "
           "no Barnes-Hut atomics under IDENTICAL); scikit-learn Barnes-Hut (angle=0.5, "
           "scikit-learn only)",
           "learning rate: ours and scikit-learn 'auto' = max(n / early_exaggeration / 4, 50); "
           "cuML that value with learning_rate_method='none', which doubles it after the 250 "
           "exaggeration steps (t_sne.pyx post_learning_rate = 2 x learning_rate; no Python "
           "switch). Every arm runs the same number of steps",
           "k-NN: 91 = min(n - 1, floor(3 x perplexity) + 1) neighbours on ours and "
           "scikit-learn (its rule), cuML n_neighbors=91 by its own exact k-NN",
           "init: ours and scikit-learn start from the SAME array, ours' 'random' rule "
           "(default_rng(7).uniform(-5e-5, 5e-5, (n, 2)) float32); cuML takes only 'random' "
           "or 'pca' and draws its own start",
           "early stop off on every arm: scikit-learn n_iter_without_progress=1000, "
           "min_grad_norm=0.0; cuML min_grad_norm=0.0; ours runs exactly max_iter steps. "
           "n_iter is in every quality cell"])
_add("ivf-sq", xlane="ann", ours=("IVFSQIndex",), kind="ann", task="ivf-sq", block="ivf",
     params=dict(n_lists=1024, n_probes=32, n_neighbors=10, kmeans_n_iters=20, random_state=SEED),
     other={"faiss-cpu": "faiss", "cuvs-gpu": "cuvs"}, notes=[_ANN, "8-bit scalar quantizer"], mism=[_ANN_TRAIN])
_add("ivf-rabitq", xlane="ann", ours=("IVFRaBitQIndex",), kind="ann", task="ivf-rabitq",
     block="ivf", params=dict(n_lists=1024, n_probes=32, n_neighbors=10, kmeans_n_iters=20,
                              random_state=SEED),
     other={"faiss-cpu": "faiss"}, notes=[_ANN])
_add("ivf-refine", xlane="ann", ours=("IVFPQIndex",), kind="ann", task="ivf-refine", block="ivf",
     params=dict(n_lists=1024, n_probes=32, n_neighbors=10, pq_bits=8, kmeans_n_iters=20,
                 refine_ratio=4, random_state=SEED),
     other={"faiss-cpu": "faiss", "cuvs-gpu": "cuvs"},
     notes=[_ANN, "IVF-PQ search of 4k candidates then an exact re-rank to k"], mism=[_ANN_TRAIN])
_add("ivf-filter", xlane="ann", ours=("IVFPQIndex",), kind="ann", task="ivf-filter", block="ivf",
     params=dict(n_lists=1024, n_probes=32, n_neighbors=10, pq_bits=8, kmeans_n_iters=20,
                 random_state=SEED),
     other={"faiss-cpu": "faiss"},
     notes=[_ANN, "IVF-PQ search with a sample filter (ours search(filter=), faiss "
            "IDSelectorBatch): every odd index row is excluded; recall against the filtered "
            "brute force"])

#: Every parameter that ours and scikit-learn BOTH have, set EXPLICITLY to the
#: same value on both (ours' default, which is scikit-learn 1.7.2's; read from
#: the two signatures, 2026-09-29), so no arm runs on a library default. None
#: values are not listed (None is the same documented setting on both).
_SHARED = {
    'sgd-clf': dict(l1_ratio=0.15, fit_intercept=True, epsilon=0.1, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, n_iter_no_change=5, average=False),
    'sgd-reg': dict(l1_ratio=0.15, fit_intercept=True, epsilon=0.1, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, n_iter_no_change=5, average=False),
    'poisson': dict(fit_intercept=True, solver=GLM_SOLVER),   # FAIRNESS FIX (2026-10-13 audit): see GLM_SOLVER
    'gamma': dict(fit_intercept=True, solver=GLM_SOLVER),   # FAIRNESS FIX (2026-10-13 audit): see GLM_SOLVER
    'tweedie': dict(fit_intercept=True, solver=GLM_SOLVER),   # FAIRNESS FIX (2026-10-13 audit): see GLM_SOLVER
    'huber': dict(fit_intercept=True),
    'bayesian-ridge': dict(alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True),
    'ard': dict(alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, threshold_lambda=10000.0, fit_intercept=True),
    'lars': dict(eps=2.220446049250313e-16),
    'lasso-lars': dict(fit_intercept=True, eps=2.220446049250313e-16, positive=False),
    'quantile': dict(fit_intercept=True),
    'perceptron': dict(alpha=0.0001, l1_ratio=0.15, fit_intercept=True, shuffle=True, eta0=1.0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5),
    'pa-clf': dict(fit_intercept=True, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, loss='hinge', average=False),
    'pa-reg': dict(fit_intercept=True, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, loss='epsilon_insensitive', epsilon=0.1, average=False),
    'ridge-clf': dict(fit_intercept=True, tol=0.0001, solver='auto', positive=False),
    'sgd-ocsvm': dict(fit_intercept=True, shuffle=True, learning_rate='optimal', eta0=0.0, power_t=0.5, average=False),
    'ridge-cv': dict(fit_intercept=True, alpha_per_target=False),
    'lasso-cv': dict(eps=0.001, fit_intercept=True, positive=False, selection='cyclic'),
    'enet-cv': dict(eps=0.001, fit_intercept=True, positive=False, selection='cyclic'),
    'logreg-cv': dict(fit_intercept=True, dual=False, penalty='l2', solver='lbfgs', tol=0.0001, refit=True, intercept_scaling=1.0),
    'isotonic': dict(increasing=True),
    'minibatch-kmeans': dict(init='k-means++', tol=0.0, max_no_improvement=10, reassignment_ratio=0.01),
    'bisecting-kmeans': dict(init='random', n_init=1, max_iter=300, tol=0.0001, algorithm='lloyd', bisecting_strategy='biggest_inertia'),
    'meanshift': dict(min_bin_freq=1, cluster_all=True, max_iter=300),
    'optics': dict(max_eps=float('inf'), metric='minkowski', p=2, cluster_method='xi', predecessor_correction=True, algorithm='auto', leaf_size=30),
    'affinity-prop': dict(affinity='euclidean'),
    'bayesian-gmm': dict(n_init=1, weight_concentration_prior_type='dirichlet_process'),
    'lof': dict(leaf_size=30, metric='minkowski', p=2, contamination='auto', novelty=False),
    'nearest-centroid': dict(metric='euclidean', priors='uniform'),
    'ocsvm': dict(degree=3, gamma='scale', coef0=0.0, shrinking=True, max_iter=-1),
    'kernel-pca': dict(degree=3, coef0=1, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, iterated_power='auto', remove_zero_eig=False),
    'poly-count-sketch': dict(gamma=1.0),
    'label-propagation': dict(gamma=20),
    'label-spreading': dict(gamma=20),
    'knn-imputer': dict(metric='nan_euclidean', add_indicator=False, keep_empty_features=False),
    'incremental-pca': dict(whiten=False),
    'gaussian-rp': dict(eps=0.1, compute_inverse_components=False),
    'sparse-rp': dict(eps=0.1, dense_output=False, compute_inverse_components=False),
    'nmf': dict(beta_loss='frobenius', alpha_W=0.0, alpha_H='same', l1_ratio=0.0, shuffle=False),
    'fastica': dict(algorithm='parallel', fun='logcosh', whiten_solver='svd'),
    'factor-analysis': dict(iterated_power=3),
    'cca': dict(scale=True),
    'pls-canonical': dict(scale=True, algorithm='nipals'),
    'pls': dict(scale=True, tol=1e-06),
    'sparse-pca': dict(ridge_alpha=0.01, method='lars'),
    'mb-sparse-pca': dict(ridge_alpha=0.01, shuffle=True, method='lars', tol=0.001, max_no_improvement=10),
    'dict-learning': dict(split_sign=False, positive_code=False, positive_dict=False, transform_max_iter=1000),
    'lda': dict(learning_decay=0.7, learning_offset=10.0, batch_size=128, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100),
    'mb-dict-learning': dict(fit_algorithm='lars', shuffle=True, split_sign=False, positive_code=False, positive_dict=False, transform_max_iter=1000, tol=0.001, max_no_improvement=10),
    'isomap': dict(eigen_solver='auto', tol=0, path_method='auto', neighbors_algorithm='auto', metric='minkowski', p=2),
    'mds': dict(normalized_stress='auto'),
    'lle': dict(reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto'),
    'robust-scaler': dict(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), unit_variance=False),
    'quantile-transformer': dict(ignore_implicit_zeros=False),
    'poly-features': dict(interaction_only=False),
    'spline': dict(knots='uniform', extrapolation='constant', include_bias=True),
    'onehot': dict(categories='auto'),
    'ordinal': dict(categories='auto'),
    'target-encoder': dict(categories='auto', smooth='auto'),
    'simple-imputer': dict(add_indicator=False, keep_empty_features=False),
    'iterative-imputer': dict(initial_strategy='mean', skip_complete=False, min_value=-float('inf'), max_value=float('inf'), add_indicator=False, keep_empty_features=False),
    'label-binarizer': dict(neg_label=0, pos_label=1),
    'gaussian-nb': dict(var_smoothing=1e-09),
    'bernoulli-nb': dict(alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True),
    'categorical-nb': dict(alpha=1.0, force_alpha=True, fit_prior=True),
    'multinomial-nb': dict(force_alpha=True, fit_prior=True),
    'complement-nb': dict(force_alpha=True, fit_prior=True, norm=False),
    'lda-clf': dict(tol=0.0001),
    'qda': dict(tol=0.0001),
    'mlp-clf': dict(activation='relu', alpha=0.0001, learning_rate='constant', power_t=0.5, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, max_fun=15000),
    'mlp-reg': dict(activation='relu', alpha=0.0001, learning_rate='constant', power_t=0.5, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, max_fun=15000),
    'decision-tree-clf': dict(criterion='gini', splitter='best', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, ccp_alpha=0.0),
    'decision-tree-reg': dict(criterion='squared_error', splitter='best', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, ccp_alpha=0.0),
    'bagging-clf': dict(max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False),
    'bagging-reg': dict(max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False),
    'random-trees-embedding': dict(min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0),
    'voting-clf': dict(flatten_transform=True),
    'stacking-clf': dict(stack_method='auto', passthrough=False),
    'stacking-reg': dict(passthrough=False),
    'tsne': dict(early_exaggeration=12.0),
}


# ---- the extra lanes (tools/bench_board_extra.py, 2026-09-29): QNRegressor,
# kpss_test / select_d, clip_grad_norm_, cross_entropy, the BPE tokenizer, the
# six learning-rate schedules and johnson_lindenstrauss_min_dim
def _load_extra():
    spec = importlib.util.spec_from_file_location(
        "bba_extra", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_board_extra.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EXTRA = _load_extra()
_EXTRA.register(_add)


def _merge_shared():
    for lane, extra in _SHARED.items():
        spec = LANES[lane]
        if spec.get("cuml") and "cuml_params" not in spec:
            spec["cuml_params"] = dict(spec["params"])   # cuML keeps its own argument set
        spec["params"] = dict(extra, **spec["params"])
        if "sk_params" in spec:
            spec["sk_params"] = dict(extra, **spec["sk_params"])
    # params and sk_params must agree on every parameter ours takes (sk_params
    # may only ADD execution settings such as n_jobs or a scikit-learn-only
    # solver choice named in the lane's mismatches)
    for lane, spec in LANES.items():
        skp = spec.get("sk_params")
        if skp is None:
            continue
        for k, v in spec["params"].items():
            if k not in skp or skp[k] != v:
                raise AssertionError("LANES[%r]: params[%r]=%r but sk_params has %r"
                                     % (lane, k, v, skp.get(k, "<unset>")))


_merge_shared()

# FAIRNESS FIX (2026-10-13 audit): the CPU time-series opponents mapped their per-series fits
# through a joblib process pool only (see _ts_map); each now has a `-serial` twin, an in-process
# loop, so the board races the library's faster path. Prophet (seconds per series) and AutoARIMA
# (the search, about a second per series) keep the pool only; VAR is one joint fit. Results raced
# before this change were wrong for these lanes; apply extra scrutiny.
for _slug, _s in LANES.items():
    if _s.get("kind") != "ts" or _slug in ("prophet", "autoarima") or _s.get("task") == "var":
        continue
    for _arm, _lib in list(_s["other"].items()):
        if _lib in ("statsforecast", "statsmodels", "arch") and not _arm.endswith("-serial"):
            _s["other"][_arm + "-serial"] = _lib
# FAIRNESS FIX (2026-10-13 audit): the precision difference of the CPU time-series libraries was
# undeclared. They compute in float64 (their only path); ours float32. Declared on every lane that
# races them. Results raced before this change were wrong for these lanes; apply extra scrutiny.
for _slug, _s in LANES.items():
    _libs = sorted({_l for _l in _s.get("other", {}).values()
                    if _l in ("statsforecast", "statsmodels", "arch")})
    if _libs and not any(str(_m).startswith("precision:") for _m in _s["mism"]):
        _s["mism"].append("precision: %s compute%s in float64 on the CPU (the only path); ours "
                          "float32" % (" and ".join(_libs), "" if len(_libs) > 1 else "s"))
LANE_ORDER = tuple(LANES)

#: Algorithms of the expansion that are NOT a race, with the reason (the board
#: prints this; nothing is dropped silently).
NOT_RACED = {
    "LinearSVC / LinearSVR": "already public and raced as classical2/linearsvc and "
                             "classical2/linearsvr",
    "SpectralEmbedding": "already public and raced as classical2/spectral-embedding",
    "DampedETS": "ETS(model='AAN', damped=True) by construction (error='A'), the model the "
                 "damped-ets lane races",
    "sparse_encode": "the encoder SparseCoder.transform calls, raced as algos/sparse-coder",
    "resample_indices": "the index draw inside resample(), raced as algos/resample",
    "monte_carlo_integrate": "no pinned library has a plain seeded Monte Carlo integrator "
                             "(numpy and torch have none; scipy.integrate.qmc_quad is "
                             "quasi-Monte Carlo over a QMCEngine); no opponent",
}


# ---------------------------------------------------------------------------
# Rosters per vendor
# ---------------------------------------------------------------------------

#: torch settings per vendor, bench_board_neural's (eager/compile x fp32/tf32/bf16)
TORCH_GPU = {"nvidia": ("eager-fp32", "compile-fp32", "eager-tf32", "compile-tf32",
                        "eager-bf16", "compile-bf16"),
             "amd": ("eager-fp32", "compile-fp32", "eager-bf16", "compile-bf16"),
             "apple": ("eager-fp32", "compile-fp32", "eager-bf16", "compile-bf16")}

#: the torch optimizers with a fused GPU kernel (fused=True) raced as torch-fused-fp32
FUSED_TORCH_OPTIMIZERS = ("SGD", "Adam", "AdamW")

_NVIDIA_ONLY = ("cuml-gpu", "cuvs-gpu", "cugraph-gpu", "cupy-gpu", "implicit-gpu", "xgboost-gpu")
#: FAIRNESS FIX (2026-10-13 audit): hipVS (AMD's cuVS port, `pip install amd-hipvs`, the
#: `cuvs` API) is the AMD GPU opponent of the vector-search lanes it implements (IVF-PQ,
#: CAGRA); before, AMD raced only faiss on the CPU. Results raced before this change were wrong
#: for these lanes; apply extra scrutiny.
_AMD_ONLY = ("hipvs-gpu",)


def opponents(vendor, lane):
    s = LANES[lane]
    arms = []
    if s["kind"] == "extra":
        return _EXTRA.opponents(vendor, lane, s, TORCH_GPU[vendor])
    if s["kind"] in ("layer", "seqmodel", "cnnclf"):
        arms = ["torch-" + t for t in TORCH_GPU[vendor]]
        if s["task"] == "dropout2d":
            arms = [a for a in arms if "bf16" not in a]
        if s["task"] == "embedding":          # a gather: no matmul for TF32, autocast keeps fp32
            arms = [a for a in arms if a.endswith("fp32")]
        return tuple(arms)
    if s["kind"] == "optim":
        if not s.get("torch_opt"):
            return ()
        # FAIRNESS FIX (2026-10-13 audit): torch's fused optimizer kernel (fused=True, its fastest
        # path for SGD / Adam / AdamW) was never raced; torch-fused-fp32 races it (refused by name
        # where the device has none). Results raced before this change were wrong for this lane;
        # apply extra scrutiny.
        fused = ("fused-fp32",) if s["torch_opt"] in FUSED_TORCH_OPTIMIZERS else ()
        return tuple("torch-" + t for t in ("eager-fp32", "compile-fp32") + fused)
    if s.get("sk"):
        arms.append("sklearn-cpu")
    for arm in s["other"]:
        arms.append(arm)
    if s.get("cuml"):
        arms.append("cuml-gpu")
    out = []
    for a in arms:
        if a in _NVIDIA_ONLY and vendor != "nvidia":
            continue
        if a in _AMD_ONLY and vendor != "amd":   # FAIRNESS FIX (2026-10-13 audit): hipVS
            continue
        if a in s.get("refused", {}):     # FAIRNESS FIX (2026-10-13 audit): see _add's `refused`
            continue
        if a not in out:
            out.append(a)
    return tuple(out)


def datasets_of(lane):
    return LANES[lane]["datasets"]


def block_of(lane):
    return LANES[lane]["block"]


def has_infer(lane):
    s = LANES[lane]
    if s["kind"] in ("layer", "ann", "svgp", "seqmodel", "cnnclf"):
        return True
    if s["kind"] in ("est", "dart"):
        if s.get("fit_predict") or s["task"] in ("embed", "covariance", "select"):
            return False
        return True
    # time series: the libraries fit and forecast in one call (statsforecast's
    # forecast(), one joblib task per series), so the forecast is inside the
    # fit clock on every arm; ALS: quality from the factors, no inference call
    return False


def infer_call(lane):
    s = LANES[lane]
    if s["kind"] == "layer":
        return "forward(x) (no autograd)"
    if s["kind"] == "ann":
        return "search(queries)"
    if s["kind"] == "als":
        return "recommend top-10 for every user row"
    if s["kind"] == "ts":
        return "forecast(%d)" % TS_H
    if s["kind"] in ("svgp", "seqmodel", "cnnclf"):
        return "predict(Xq)"
    t = s["task"]
    if t == "radius":
        return "radius_neighbors(Xq)"
    if t in ("clf", "reg", "semi", "outlier", "multiclf", "multireg", "gmm"):
        return "predict(Xq)"
    if t == "cluster":
        return "predict(Xq) or fit labels"
    return "transform(Xq)"


def not_planned(vendor):
    out = []
    if vendor != "nvidia":
        out.append("cuML, cuVS, cuGraph, CuPy, implicit-gpu and xgboost-gpu arms: CUDA only; "
                   "the CPU libraries and torch on this box's GPU are the arms")
    out.append("Lion and LAMB: torch.optim has neither; ours races alone (no pinned reference "
               "implementation is fast code)")
    out.append("torch-*-bf16 on optimizers: the fp32 parameter and state are the workload; "
               "torch-*-tf32: no matmul in an optimizer step")
    out.append("openTSNE and hnswlib: not pinned; scikit-learn/cuML t-SNE and FAISS HNSW stand in")
    out.append("pmdarima: statsforecast AutoARIMA (compiled) is the CPU AutoARIMA arm")
    out.append("torch-*-bf16 and torch-*-tf32 on embedding: a gather has no matmul for TF32 and "
               "autocast leaves nn.Embedding in fp32, so those arms would repeat the fp32 ones")
    if vendor == "nvidia":
        out.append("cuVS on ivf-filter: cuVS's Python IVF-PQ search takes no sample filter (only "
                   "IVF-Flat, CAGRA and brute force do); ours filters IVF-PQ, so faiss-cpu "
                   "IndexIVFPQ with an IDSelectorBatch is the matched arm")
    for name, why in NOT_RACED.items():
        out.append("%s: %s" % (name, why))
    # FAIRNESS FIX (2026-10-13 audit): the opponents refused as different work, by name
    for lane in LANE_ORDER:
        for arm, why in sorted(LANES[lane].get("refused", {}).items()):
            if arm in _NVIDIA_ONLY and vendor != "nvidia":
                continue
            out.append("%s on %s: refused (%s)" % (arm, lane, why))
    return out


def lane_config(lane):
    s = LANES[lane]
    cfg = {"expansion_lane": s["xlane"], "ours_class": " or ".join("mojolearn." + n for n in s["ours"]),
           "kind": s["kind"], "task": s["task"], "block": s["block"],
           "datasets": list(s["datasets"]), "params": _jsonable(s["params"]),
           "timed_fit": fit_text(lane),
           "timed_infer": infer_call(lane) if has_infer(lane) else None,
           "quality": quality_text(lane), "notes": list(s.get("notes", [])),
           "mismatches": list(s["mism"])}
    if s.get("sk"):
        cfg["sklearn"] = s["sk"]
    if s.get("cuml"):
        cfg["cuml"] = s["cuml"]
    if s["sub"]:
        cfg["stride_subsets"] = dict(s["sub"])
    if s.get("refused"):                  # FAIRNESS FIX (2026-10-13 audit): see _add's `refused`
        cfg["refused_opponents"] = dict(s["refused"])
    if s.get("ours_extra"):               # FAIRNESS FIX (2026-10-13 audit): see _est_factory
        cfg["ours_only_params"] = _jsonable(s["ours_extra"])
    if s.get("dataset_params"):
        cfg["dataset_params"] = _jsonable(s["dataset_params"])
    return cfg


def _jsonable(v):
    if isinstance(v, dict):
        if "__est__" in v:
            return "%s(%s)" % (v["__est__"], ", ".join("%s=%r" % kv for kv in sorted(v["kw"].items())))
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def fit_text(lane):
    s = LANES[lane]
    k, t = s["kind"], s["task"]
    if k == "extra":
        return s["fit_text"]
    if k == "layer":
        return "forward + backward(dy)"
    if k == "seqmodel":
        return "fit(X (n, T, 1), y): 2 epochs"
    if k == "cnnclf":
        return "fit(X (n, 1, 28, 28), y): 2 epochs"
    if k == "optim":
        return "10 optimizer steps"
    if k == "ann":
        return "index build"
    if k == "graph":
        return "the graph algorithm on the CSR graph"
    if k == "linalg":
        return {"lu": "LU factor + solve", "lstsq": "least squares", "rsvd": "randomized SVD",
                "lufac": "lu_factor(A) then lu_solve(., B)", "chol": "Cholesky factor",
                "qr": "QR (reduced)", "eigh": "symmetric eigendecomposition",
                "svd": "thin SVD"}[t]
    if k == "ts":
        return "fit + forecast (or decomposition) of every series"
    if k == "shap":
        return "SHAP values of Xq (model fit before the clock)"
    if k == "fn":
        return {"bootstrap": "bootstrap(y): every resample and the interval",
                "permutation": "permutation_test(x, y): every permutation and the p-value",
                "resample": "resample(X, y): the index draw and both gathers",
                "cv": "cross_val_score(LinearRegression(), X, y, cv=5): 5 fits and scores"}[t]
    if s.get("fit_predict"):
        return "fit_predict(X)"
    if t in ("embed",):
        return "fit_transform(X)"
    return "fit(X%s)" % (", y" if t in ("clf", "reg", "semi", "select", "multiclf", "multireg")
                         or s.get("supervised") else "")


QUALITY_TEXT = {
    "clf": "held-out accuracy (and log loss when the arm has predict_proba)",
    "multiclf": "held-out accuracy per target, mean",
    "reg": "held-out R2 and RMSE", "multireg": "held-out R2 per target, mean",
    "semi": "accuracy on the held-out rows",
    "outlier": "fraction flagged on Xq; Jaccard of the flagged set against scikit-learn's",
    "cluster": "cluster count, silhouette on up to 10,000 rows, ARI vs ours",
    "gmm": "held-out mean log-likelihood from each arm's weights, means and covariances",
    "embed": "trustworthiness k=15 over every embedded row",
    "impute": "RMSE over the masked cells; max abs diff vs scikit-learn",
    "select": "selected-feature Jaccard against scikit-learn's selection",
    "select-exact": "selected-feature Jaccard against an exact RFE (scikit-learn RFE with "
                    "LogisticRegression(solver='newton-cholesky', tol=1e-10) in float64, untimed) "
                    "and against scikit-learn's selection",
    "covariance": "relative Frobenius difference of covariance_ vs scikit-learn",
    "labels": "exact agreement with scikit-learn's output",
    "radius": "neighbours found in total; fraction of queries whose neighbour count equals "
              "scikit-learn's",
    "multilabel": "exact agreement with scikit-learn's output",
    "vs-sklearn": "max abs and relative Frobenius difference of transform(Xq) vs scikit-learn",
    "subspace": "RBF feature-distance relative stress over seeded query pairs (lower is better); "
                "principal-angle cosine vs scikit-learn when a stored reference is available",
    "pca": "explained-variance fraction of the centered rows by the components",
    "distortion": "mean |projected / original squared distance - 1| over 2,000 row pairs",
    "nmf": "relative reconstruction error ||X - W H|| / ||X|| on the fit rows",
    "ica": "mean |excess kurtosis| of the recovered sources on Xq",
    "fa": "held-out mean Gaussian log-likelihood (W W^T + diag(psi))",
    "cca": "mean correlation of the paired canonical scores on Xq",
    "recon": "relative reconstruction error of Xq from the arm's code and dictionary",
    "perplexity": "held-out perplexity from transform(Xq) and the normalized topic-word matrix",
    "shape": "output columns and nonzeros per row",
    "kernel-poly": "relative Frobenius error of Z Z^T against the exact (gamma x.y)^2 kernel",
    "kernel-achi2": "relative Frobenius error of Z Z^T against the exact additive chi2 kernel",
    "kernel-schi2": "relative Frobenius error of Z Z^T against the exact skewed chi2 kernel",
    "forecast": "forecast RMSE over the held-out points",
    "decompose": "relative difference of trend + seasonal vs statsmodels STL",
    "var": "forecast RMSE over the held-out points",
    "garch": "mean log-likelihood per series",
    "pagerank": "L1 distance of the scores vs networkx's",
    "components": "component count; ARI vs networkx's labels",
    "louvain": "modularity of the partition (conductor, float64)",
    "layer": "max relative difference of the forward output vs torch eager fp32 (same weights)",
    "dropout": "channel-wise inverted-dropout contract error and dropped-channel binomial deviation",
    "optim": "max relative difference of the parameter after 10 steps vs torch eager fp32",
    "ann": "recall@10 against a float64 NumPy brute force",
    "lu": "relative residual ||A x - B|| / ||B||", "lstsq": "relative residual vs numpy's",
    "rsvd": "relative rank-8 reconstruction error",
    "chol": "relative residual ||L L^T - A|| / ||A||",
    "qr": "relative difference of R^T R vs X^T X (float64) and max |diag R| / its float64 value",
    "eigh": "relative residual ||A V - V diag(w)|| / ||A|| and max eigenvalue error vs float64",
    "svd": "max relative singular-value error vs float64 and ||X - U S Vh|| / ||X|| on 100,000 "
           "rows",
    "als": "recall@10 of the held-out interaction per user row",
    "tree-shap": "max additivity error |sum(phi) + base - margin|",
    "bootstrap": "interval endpoints and standard error; their relative difference vs scipy's",
    "permutation": "statistic and p-value; |p - scipy's p|",
    "resample": "max |resampled column mean - population mean| / column std",
    "cv": "mean fold R2; max |fold score - scikit-learn's|",
    "kernel-shap": "relative error vs the exact linear-model SHAP values",
    "permutation-shap": "relative error vs the exact linear-model SHAP values",
}


QUALITY_TEXT.update(_EXTRA.QUALITY_TEXT)


def quality_kind(lane):
    s = LANES[lane]
    if s.get("quality"):
        return s["quality"]
    if s["kind"] in ("layer",):
        return "layer"
    if s["kind"] in ("optim", "ann", "als"):
        return s["kind"]
    return s["task"]


def quality_text(lane):
    return QUALITY_TEXT.get(quality_kind(lane), quality_kind(lane))


BLOCK_ROWS = {
    "cls": "%d fit + %d held-out stride rows, binary target, %s" % (FIT_ROWS, EVAL_ROWS, _STD),
    "reg": "%d fit + %d held-out stride rows, real target, %s" % (FIT_ROWS, EVAL_ROWS, _STD),
    "nonneg": "%d fit + %d held-out stride rows, %s" % (FIT_ROWS, EVAL_ROWS, _NONNEG),
    "raw": "%d fit + %d held-out stride rows, %s" % (FIT_ROWS, EVAL_ROWS, _RAW),
    "raw16": "%d fit + %d held-out stride rows, first 16 columns, %s" % (FIT_ROWS, EVAL_ROWS, _RAW),
    "raw32": "%d fit + %d held-out stride rows, first 32 columns, %s" % (FIT_ROWS, EVAL_ROWS, _RAW),
    "cat": "%d fit + %d held-out rows of integer category codes" % (FIT_ROWS, EVAL_ROWS),
    "mc": "%d fit + %d held-out stride rows, multiclass target, %s" % (FIT_ROWS, EVAL_ROWS, _STD),
    "tsvd": "1,000,000 stride rows (the last 10%% held out), raw",
    "manifold": "20,000 stride rows, %s" % _STD,
    "ivf": "400,000 index rows, 4,000 queries, raw",
    "graph": "%d nodes, symmetric 10-NN graph, node features and labels" % GRAPH_NODES,
    "graphs": "%d nodes, symmetric 10-NN graph (and a 2-NN graph)" % GRAPH_SMALL,
    "ts": "%d series x 1,440 points, the last %d held out" % (TS_SERIES, TS_H),
    "tsi": "%d intermittent series x 1,440 points, the last %d held out" % (TS_SERIES, TS_H),
    "tsr": "%d return series x 1,439 points, the last %d held out" % (TS_SERIES, TS_H),
    "counts": "count matrix, every 10th row held out",
    "countclf": "count features with a class label, every 10th row held out",
    "bytes": "64 x 256 bytes of enwik8",
    "tensor": "a seeded tensor (see notes)",
    "corpus": "the first 4 MiB of enwik8 as 2,048-character documents",
    "images": "20,000 fit + 5,000 held-out seeded 1 x 28 x 28 images, 10 classes",
    "seqwin": "64 series -> windows of 24 steps, fit = the first 80% of time", "optim": "16,777,216 parameters x 10 steps",
    "dense": "8192 x 8192 system, 64 right-hand sides",
    "sym": "a seed-7 symmetric positive definite matrix (8192 x 8192; 4096 for eigh)",
}


def rows_text(lane, rows=None):
    s = LANES[lane]
    t = BLOCK_ROWS.get(s["block"], s["block"])
    if s["sub"]:
        t += "; stride subsets %s" % ", ".join("%s %d" % kv for kv in sorted(s["sub"].items()))
    if rows:
        t += "; SMOKE cap %d" % rows
    return t


def r2_keys(lane, dataset):
    """The R2 keys a (lane, dataset) reads (bench/results/dataset_store/manifest.tsv)."""
    s = LANES[lane]
    if dataset == "taxi" or dataset in ("taxi-hourly", "taxi-zones"):
        return ["gbm-bench/taxi/taxi_speed.npz"] if s["block"] not in ("dense", "sym", "tensor", "optim") else []
    if dataset == "istella":
        return ["gbm-bench/istella/istella_speed.npz"]
    if dataset == "text":
        return [CORPUS_KEYS["enwik8"], CORPUS_KEYS["pile_github"]]
    if dataset == "enwik8":
        return [CORPUS_KEYS["enwik8"]]
    return []


def corpus_path(key):
    """Where a staged corpus key lives on this box (tools/dataset_store.sh puts
    keys outside $HOME under ~/r2-stage/<key>)."""
    home = os.path.expanduser("~")
    cands = []
    if os.environ.get("MOJOLEARN_CORPUS_ROOT"):
        cands.append(os.path.join(os.environ["MOJOLEARN_CORPUS_ROOT"], key))
    # tools/dataset_store.sh maps a key under the staging checkout's training/
    # to the same path under the box's home; ~/r2-stage/<key> otherwise
    import glob
    cands += [os.path.join(REPO, "training", key), os.path.join(home, "r2-stage", key),
              os.path.join(home, "CascadeProjects", "mojolearn", "training", key)]
    cands += sorted(glob.glob(os.path.join(home, "mojolearn-wt", "*", "training", key)))
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def source_exports():
    """{name} in the source tree's expansion `__all__` lists (read with ast, no
    import): what the lanes have merged so far. The board itself asks the
    INSTALLED wheel; this is only the dry run's hint."""
    import ast
    names = set()
    d = os.path.join(REPO, "python", "mojolearn")
    # the expansion doors, plus the package's own __all__ and the public
    # submodules whose functions the lanes name as `module.function`
    core = ("__init__.py", "linalg.py", "resample.py", "training.py", "model_selection.py",
            "preprocessing.py", "embedding.py", "tokenizer.py")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not ((f.startswith("_expansion_") and f.endswith(".py")) or f in core):
            continue
        try:
            tree = ast.parse(open(os.path.join(d, f)).read())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__"
                                                    for t in node.targets):
                try:
                    names.update(ast.literal_eval(node.value))
                except ValueError:
                    pass
    return names


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location("bba_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MODS = {}


def _BP():
    """tools/bench_board_params.py, the board's one parameter check."""
    return _tool("bench_board_params")


def _tool(name):
    if name not in _MODS:
        _MODS[name] = _load(name)
    return _MODS[name]


def _np():
    import numpy as np
    return np


def _cap(n, cap, floor=0):
    return n if not cap else max(min(floor, n), min(n, cap))


def _imp(path):
    mod, _, cls = path.partition(":")
    if cls == "IterativeImputer":
        importlib.import_module("sklearn.experimental.enable_iterative_imputer")
    return getattr(importlib.import_module(mod), cls)


def _stride(a, m):
    np = _np()
    m = min(int(m), a.shape[0])
    idx = (np.arange(m, dtype=np.int64) * a.shape[0]) // m
    return np.ascontiguousarray(a[idx])


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------

#: this family's block -> the classical2 block it is (or is derived from)
MORE_BLOCK = {"cls": "cls", "reg": "reg", "manifold": "manifold", "tsvd": "tsvd", "ivf": "ivf",
              "nonneg": "cls", "graph": "cls", "graphs": "cls", "countclf": "cls"}
MORE_LANE_OF = {"cls": "logreg", "reg": "ridge", "manifold": "umap", "tsvd": "tsvd", "ivf": "ivf"}


def block_file(lane, dataset):
    """The npz/json basename a (lane, dataset) reads."""
    b = block_of(lane)
    if _aft_full_tree_workload(lane):
        return "aft-full-%s-%s" % (b, dataset)
    if b in ("dense", "sym", "tensor", "optim", "images", "corpus"):
        return None
    if b == "seqwin":
        return "ts-%s" % dataset
    if b in ("ts", "tsi", "tsr"):
        return "%s-%s" % (b, dataset)
    if dataset == "text":
        return "text"                      # the labelled document counts
    if dataset == "taxi-zones":
        return "zones"
    if b in ("bytes", "graph", "graphs"):
        return "%s-%s" % (b, dataset)
    if b in ("raw16", "raw32"):
        return "raw-%s" % dataset
    return "%s-%s" % (MORE_BLOCK.get(b, b), dataset)


def _tab_loader(ctd, harness, ds, kind):
    np = _np()
    if ds == "taxi":
        if kind == "mc":
            d = harness.load_taxi("shipped", multiclass=True)
        elif kind == "cat":
            d = harness.load_taxi("shipped", categorical=True)
            return d.X_train, d.X_test, np.asarray(d.y_train), np.asarray(d.y_test)
        else:
            d = harness.load_taxi("shipped", regression=(kind == "reg"))
        return (ctd._taxi_numeric(harness, d.X_train), ctd._taxi_numeric(harness, d.X_test),
                np.asarray(d.y_train), np.asarray(d.y_test))
    d = harness.load_istella("shipped", regression=True)
    ytr, yte = np.asarray(d.y_train), np.asarray(d.y_test)
    if kind in ("cls", "cat"):
        ytr, yte = (ytr > 0).astype(np.float32), (yte > 0).astype(np.float32)
    return d.X_train, d.X_test, ytr, yte


def taxi_hourly(x):
    """(zones, hours) float32 pickup counts over Jan+Feb 2024 from the taxi
    feature rows. The month is recovered from (day of month, weekday): 1 Jan
    2024 is weekday 0 in the decode's convention and 1 Feb is weekday 3; a
    row whose pair matches neither month is dropped."""
    np = _np()
    day = x[:, TAXI_DAY].astype(np.int64)
    wd = x[:, TAXI_WDAY].astype(np.int64)
    hr = x[:, TAXI_HOUR].astype(np.int64)
    pu = x[:, TAXI_PU].astype(np.int64)
    jan = (wd == (day - 1) % 7) & (day >= 1) & (day <= 31)
    feb = (wd == (day - 1 + 3) % 7) & (day >= 1) & (day <= 29)
    keep = (jan | feb) & (pu >= 0) & (pu < 300) & (hr >= 0) & (hr < 24)
    dayidx = np.where(feb, 31 + day - 1, day - 1)
    h = (dayidx * 24 + hr)[keep]
    z = pu[keep]
    counts = np.zeros((300, 60 * 24), dtype=np.float64)
    np.add.at(counts, (z, h), 1.0)
    return counts


def taxi_zone_matrix(x):
    """rows (day, hour, pickup zone) with a trip x dropoff zone trip counts."""
    np = _np()
    c = taxi_hourly(x)  # validates the month recovery the same way
    del c
    day = x[:, TAXI_DAY].astype(np.int64)
    wd = x[:, TAXI_WDAY].astype(np.int64)
    feb = (wd == (day - 1 + 3) % 7)
    jan = (wd == (day - 1) % 7)
    keep = (jan | feb) & (x[:, TAXI_PU] >= 0) & (x[:, TAXI_DO] >= 0) & (x[:, TAXI_DO] < 300)
    dayidx = np.where(feb, 31 + day - 1, day - 1)
    key = ((dayidx * 24 + x[:, TAXI_HOUR].astype(np.int64)) * 300 + x[:, TAXI_PU].astype(np.int64))[keep]
    do = x[:, TAXI_DO].astype(np.int64)[keep]
    ukey, row = np.unique(key, return_inverse=True)
    M = np.zeros((ukey.shape[0], 300), dtype=np.float32)
    np.add.at(M, (row, do), 1.0)
    used = M.sum(axis=0) > 0
    return np.ascontiguousarray(M[:, used])


def text_counts(path, doc_bytes=TEXT_DOC_BYTES, bins=TEXT_BINS, max_docs=None):
    """(docs, bins) float32 byte-bigram counts of consecutive doc_bytes chunks."""
    np = _np()
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.shape[0] // doc_bytes
    if max_docs:
        n = min(n, max_docs)
    raw = raw[:n * doc_bytes].reshape(n, doc_bytes).astype(np.int64)
    ids = (raw[:, :-1] * 256 + raw[:, 1:]) % bins
    out = np.zeros((n, bins), dtype=np.float32)
    step = 4096
    for s in range(0, n, step):
        e = min(n, s + step)
        flat = (ids[s:e] + (np.arange(e - s)[:, None] * bins)).reshape(-1)
        out[s:e] = np.bincount(flat, minlength=(e - s) * bins).reshape(e - s, bins)
    return out


def knn_graph(X, k=KNN_K, chunk=512):
    """Symmetric unweighted kNN graph (CSR indptr, indices), float64 brute
    force, ties to the lower index."""
    np = _np()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    sq = (X * X).sum(axis=1)
    nbr = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = sq[s:e, None] + sq[None, :] - 2.0 * (X[s:e] @ X.T)
        d[np.arange(e - s), np.arange(s, e)] = np.inf
        part = np.argpartition(d, k, axis=1)[:, :k + 1]
        dd = np.take_along_axis(d, part, axis=1)
        order = np.lexsort((part, dd), axis=1)[:, :k]
        nbr[s:e] = np.take_along_axis(part, order, axis=1)
    src = np.repeat(np.arange(n), k)
    dst = nbr.reshape(-1)
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    key = np.unique(a * n + b)
    a, b = key // n, key % n
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, a + 1, 1)
    return np.cumsum(indptr), b.astype(np.int64)


def synthetic_series(kind, n_series, n_obs, seed=SEED):
    np = _np()
    rng = np.random.default_rng(seed)
    if kind == "intermittent":
        return (rng.random((n_series, n_obs)) < 0.3) * rng.poisson(3.0, (n_series, n_obs)).astype(np.float64)
    if kind == "garch":
        e = rng.standard_normal((n_series, n_obs + 200))
        r = np.zeros_like(e)
        s2 = np.full(n_series, 1.0)
        for t in range(1, n_obs + 200):
            s2 = 0.1 + 0.1 * r[:, t - 1] ** 2 + 0.8 * s2
            r[:, t] = np.sqrt(s2) * e[:, t]
        return r[:, 200:]
    more = _tool("bench_board_more")
    return more.seasonal_series(n_series, n_obs).astype(np.float64)


def _aft_full_tree_workload(lane):
    enabled = os.environ.get("MOJOLEARN_AFT_FULL_TREE_WORKLOAD") == "1"
    if enabled:
        raise ValueError("integration benchmark freeze forbids removing existing tree caps; coverage remains pending")
    if enabled and lane not in AFT_FULL_TREE_LANES:
        raise ValueError("full-tree-workload is restricted to the named tree estimator lanes")
    return enabled


def _aft_prepare_full_tree_blocks(args, lanes, datasets):
    """CPU-only input step using the existing complete saved loader splits.

    Distinct names prevent replacing capped historical board inputs. The same
    sentinel cleanup, train-statistic standardization, targets and taxi numeric
    feature schema remain; only row sampling is removed. No corpus download,
    model fitting, benchmark or quality admission happens in this input step.
    """
    if any(lane not in AFT_FULL_TREE_LANES for lane in lanes):
        raise ValueError("full-tree-workload prep accepts tree lanes only")
    if args.max_rows or os.environ.get("MOJOLEARN_ALGOS_SMOKE_ROWS"):
        raise ValueError("full-tree-workload cannot accept a row cap")
    if any(ds not in TAB for ds in datasets):
        raise ValueError("full tree prep supports existing taxi and istella saved splits only")
    np = _np()
    ctd = _tool("classical_two_datasets")
    harness = ctd._module("speed_gbdt_arm")
    for block in sorted({block_of(lane) for lane in lanes}):
        if block not in ("cls", "reg"):
            raise ValueError("full tree prep supports cls/reg blocks only")
        for dataset in datasets:
            name = "aft-full-%s-%s" % (block, dataset)
            if any(os.path.exists(os.path.join(args.data, name + suffix))
                   for suffix in (".npz", ".json")):
                raise FileExistsError("retain existing full input artifacts; choose a new data directory: " + name)
            xtr, xte, ytr, yte = _tab_loader(ctd, harness, dataset, block)
            source_dimensions = {"train": list(xtr.shape), "test": list(xte.shape)}
            source_arrays = {"X": ctd.sha256_array(xtr), "Xq": ctd.sha256_array(xte),
                             "y": ctd.sha256_array(ytr), "yq": ctd.sha256_array(yte)}
            X, bad_x = ctd.clean_sentinel(xtr)
            Xq, bad_q = ctd.clean_sentinel(xte)
            X, Xq = ctd.standardize(X, Xq)
            y = np.ascontiguousarray(ytr, dtype=np.float32)
            yq = np.ascontiguousarray(yte, dtype=np.float32)
            if block == "cls" and len(np.unique(y)) != 2:
                raise RuntimeError("full classifier block must retain both original classes")
            ctd._write_block(args.data, name, {"X": X, "y": y, "Xq": Xq, "yq": yq}, {
                "rule": "tools/bench_board_algos.py prep --full-tree-workload",
                "full_tree_workload": True, "smoke_max_rows": None, "seed": SEED,
                "block": block, "dataset": dataset, "data_root": harness.data_root(),
                "loader": "bench_board_algos._tab_loader existing shipped split, no rows_cap",
                "source_dimensions": source_dimensions, "source_array_sha256": source_arrays,
                "fit_rows": "all loader train rows", "eval_rows": "all loader held-out rows",
                "target": "binary" if block == "cls" else "real",
                "scaling": "standardized by all fit rows using existing float64 mean/std",
                "feature_schema": "existing taxi numeric columns or all Istella features",
                "sentinel_cells_replaced": {"X": bad_x, "Xq": bad_q},
                "intrinsic_caps": {"FIT_ROWS": None, "EVAL_ROWS": None, "lane_subsets": None},
                "coverage": "complete existing saved dataset split; source hashes retained, no external release/version claim",
            })
    return 0


def prep(args):
    np = _np()
    ctd = _tool("classical_two_datasets")
    more = _tool("bench_board_more")
    lanes = [l for l in args.lanes.split(",") if l]
    for l in lanes:
        if l not in LANES:
            raise SystemExit("unknown lane %r" % l)
    datasets = [d for d in args.datasets.split(",") if d]
    cap = int(args.max_rows) if args.max_rows else None
    os.makedirs(args.data, exist_ok=True)
    if getattr(args, "full_tree_workload", False):
        raise ValueError("integration benchmark freeze forbids --full-tree-workload; retain the saved race")
        return _aft_prepare_full_tree_blocks(args, lanes, datasets)
    base = {"rule": "tools/bench_board_algos.py prep", "smoke_max_rows": cap, "seed": SEED}
    need = set()
    for l in lanes:
        for ds in datasets_of(l):
            if ds in datasets or ds not in TAB:
                need.add(("ts" if block_of(l) == "seqwin" else block_of(l), ds))

    def have(name):
        return os.path.exists(os.path.join(args.data, name + ".json"))

    # the classical2 blocks
    mlanes = sorted({MORE_LANE_OF[MORE_BLOCK[b]] for b, ds in need
                     if b in MORE_BLOCK and ds in TAB and not have("%s-%s" % (MORE_BLOCK[b], ds))})
    mds = sorted({ds for b, ds in need if b in MORE_BLOCK and ds in TAB})
    if mlanes:
        more.prep(argparse.Namespace(data=args.data, lanes=",".join(mlanes), datasets=",".join(mds),
                                     max_rows=args.max_rows))
    harness = None
    for b, ds in sorted(need):
        t0 = time.perf_counter()
        name = None
        if b in ("raw", "raw16", "raw32", "cat", "mc") and ds in TAB:
            name = "%s-%s" % ("raw" if b.startswith("raw") else b, ds)
            if have(name):
                continue
            harness = harness or ctd._module("speed_gbdt_arm")
            xtr, xte, ytr, yte = _tab_loader(ctd, harness, ds, "cls" if b != "mc" else "mc")
            fi = ctd.stride_rows(xtr.shape[0], _cap(FIT_ROWS, cap, 256))
            ei = ctd.stride_rows(xte.shape[0], _cap(EVAL_ROWS, cap, 256))
            if b == "cat":
                if ds == "taxi":
                    c = harness.load_taxi("shipped", categorical=True)
                    X = np.asarray(c.X_train)[fi][:, TAXI_CAT_COLUMNS]
                    Xq = np.asarray(c.X_test)[ei][:, TAXI_CAT_COLUMNS]
                    y, yq = np.asarray(c.y_train)[fi], np.asarray(c.y_test)[ei]
                    how = "taxi id columns %s" % (TAXI_CAT_COLUMNS,)
                else:
                    Xf, _ = ctd.clean_sentinel(xtr[fi])
                    Xe, _ = ctd.clean_sentinel(xte[ei])
                    cols = np.argsort(-Xf.var(axis=0), kind="stable")[:8]
                    X = np.empty((Xf.shape[0], 8))
                    Xq = np.empty((Xe.shape[0], 8))
                    for j, c in enumerate(cols):
                        edges = np.quantile(Xf[:, c].astype(np.float64), np.linspace(0, 1, 17)[1:-1])
                        X[:, j] = np.searchsorted(edges, Xf[:, c], side="right")
                        Xq[:, j] = np.searchsorted(edges, Xe[:, c], side="right")
                    y, yq = ytr[fi], yte[ei]
                    how = "Istella's 8 highest-variance columns quantile-binned to 16 codes"
                arrays = {"X": np.ascontiguousarray(X.astype(np.float32)),
                          "Xq": np.ascontiguousarray(Xq.astype(np.float32)),
                          "y": np.asarray(y, dtype=np.float32), "yq": np.asarray(yq, dtype=np.float32)}
                rec = dict(base, block="cat", dataset=ds, columns=how)
            else:
                X, bx = ctd.clean_sentinel(xtr[fi])
                Xq, bq = ctd.clean_sentinel(xte[ei])
                if b == "mc":
                    X, Xq = ctd.standardize(X, Xq)
                arrays = {"X": np.ascontiguousarray(X, dtype=np.float32),
                          "Xq": np.ascontiguousarray(Xq, dtype=np.float32),
                          "y": np.ascontiguousarray(np.asarray(ytr)[fi], dtype=np.float32),
                          "yq": np.ascontiguousarray(np.asarray(yte)[ei], dtype=np.float32)}
                rec = dict(base, block=name.split("-")[0], dataset=ds,
                           scaling=_STD if b == "mc" else _RAW,
                           sentinel_cells_replaced={"X": bx, "Xq": bq})
            ctd._write_block(args.data, name, arrays, rec)
        elif b in ("ts", "tsi", "tsr"):
            name = "%s-%s" % (b, ds)
            if have(name):
                continue
            if ds == "taxi-hourly":
                harness = harness or ctd._module("speed_gbdt_arm")
                raw = np.load(os.path.join(harness.data_root(), "taxi", "taxi_speed.npz"))["x"]
                C = taxi_hourly(raw)
                vol = C.sum(axis=1)
                if b == "tsi":
                    zfrac = (C == 0).mean(axis=1)
                    ok = np.nonzero((zfrac >= 0.3) & (zfrac <= 0.7))[0]
                    zones = ok[np.argsort(-vol[ok], kind="stable")][:TS_SERIES]
                else:
                    zones = np.argsort(-vol, kind="stable")[:TS_SERIES]
                Y = C[np.sort(zones)]
                if b == "tsr":
                    Y = np.diff(np.log1p(Y), axis=1)
                how = "taxi pickup counts per zone per hour, zones %s" % ",".join(map(str, np.sort(zones)[:8]))
            else:
                n = 60 * 24
                Y = synthetic_series({"tsi": "intermittent", "tsr": "garch"}.get(b, "seasonal"),
                                     TS_SERIES, n)
                how = "seed-7 synthetic (%s)" % {"tsi": "intermittent", "tsr": "garch"}.get(b, "seasonal")
            if cap:
                Y = Y[:, -max(TS_H * 8, min(Y.shape[1], cap)):]
            ctd._write_block(args.data, name, {"Y": np.ascontiguousarray(Y, dtype=np.float32)},
                             dict(base, block=b, dataset=ds, source=how, h=TS_H))
        elif b in ("counts", "countclf"):
            if ds in TAB:
                continue           # the nonneg derivation of the cls block
            name = "text" if ds == "text" else "zones"
            if have(name):
                continue
            if ds == "text":
                docs = []
                for lab, key in enumerate(("enwik8", "pile_github")):
                    p = corpus_path(CORPUS_KEYS[key])
                    if not os.path.isfile(p):
                        raise SystemExit("REFUSING: %s missing (R2 key %s; stage it with "
                                         "tools/dataset_store.sh stage)" % (p, CORPUS_KEYS[key]))
                    docs.append((text_counts(p, max_docs=cap), lab))
                X = np.concatenate([d for d, _ in docs])
                y = np.concatenate([np.full(d.shape[0], lab, dtype=np.float32) for d, lab in docs])
                held = (np.arange(X.shape[0]) % 10) == 0
                arrays = {"X": X[~held], "y": y[~held], "Xq": X[held], "yq": y[held]}
                how = "byte-bigram counts (mod %d) of %d-byte documents of enwik8 (0) and pile_github (1)" % (
                    TEXT_BINS, TEXT_DOC_BYTES)
            else:
                harness = harness or ctd._module("speed_gbdt_arm")
                raw = np.load(os.path.join(harness.data_root(), "taxi", "taxi_speed.npz"))["x"]
                if cap:
                    raw = raw[:max(cap * 20, 20000)]
                M = taxi_zone_matrix(raw)
                held = (np.arange(M.shape[0]) % 10) == 0
                arrays = {"X": M[~held], "Xq": M[held]}
                how = "taxi (day, hour, pickup zone) rows x dropoff zone trip counts"
            ctd._write_block(args.data, name, arrays, dict(base, block=b, dataset=ds, source=how))
        elif b == "bytes":
            name = "bytes-%s" % ds
            if have(name):
                continue
            p = corpus_path(CORPUS_KEYS["enwik8"])
            if not os.path.isfile(p):
                raise SystemExit("REFUSING: %s missing (R2 key %s)" % (p, CORPUS_KEYS["enwik8"]))
            raw = np.fromfile(p, dtype=np.uint8, count=64 * 256 * 4)
            ctd._write_block(args.data, name, {"bytes": raw}, dict(base, block="bytes", dataset=ds))
        elif b in ("graph", "graphs") and ds in TAB:
            name = "%s-%s" % (b, ds)
            if have(name):
                continue
            with np.load(os.path.join(args.data, "cls-%s.npz" % ds)) as z:
                X, y = z["X"], z["y"]
            n = _cap(GRAPH_NODES if b == "graph" else GRAPH_SMALL, cap, 512)
            X, y = _stride(X, n), _stride(y, n)
            indptr, indices = knn_graph(X, KNN_K)
            ip2, ix2 = knn_graph(X, 2)
            ctd._write_block(args.data, name, {"X": X, "y": y, "indptr": indptr, "indices": indices,
                                               "indptr2": ip2, "indices2": ix2},
                             dict(base, block="graph", dataset=ds, nodes=n, k=KNN_K))
        else:
            continue
        print("ALGOS-PREP block=%s dataset=%s seconds=%.1f" % (b, ds, time.perf_counter() - t0),
              flush=True)
    return 0


# ---------------------------------------------------------------------------
# a lane's arrays (the SAME rows for every arm and for the quality pass)
# ---------------------------------------------------------------------------

def lane_arrays(lane, B):
    np = _np()
    s = LANES[lane]
    b = s["block"]
    if b in ("ts", "tsi", "tsr"):
        Y = B["Y"]
        return {"Yfit": np.ascontiguousarray(Y[:, :-TS_H]), "Yhold": np.ascontiguousarray(Y[:, -TS_H:])}
    if b == "images":
        cap = B.get("_cap") or os.environ.get("MOJOLEARN_ALGOS_SMOKE_ROWS")
        n_fit, n_q = (min(20_000, int(cap)), min(5_000, int(cap))) if cap else (20_000, 5_000)
        X, y = synthetic_images(n_fit + n_q)
        return {"X": X[:n_fit], "y": y[:n_fit], "Xq": X[n_fit:], "yq": y[n_fit:]}
    if b == "seqwin":
        Y = B["Y"].astype(np.float64)
        n = Y.shape[1]
        cut = int(n * 0.8)
        mu = Y[:, :cut].mean(1, keepdims=True)
        sd = np.maximum(Y[:, :cut].std(1, keepdims=True), 1e-9)
        Z = ((Y - mu) / sd).astype(np.float32)
        T = SEQ_T
        idx = np.arange(T, n)                     # the target step of each window
        W = np.stack([Z[:, i - T:i] for i in idx], axis=1)      # (series, windows, T)
        tgt = np.stack([Z[:, i] for i in idx], axis=1)
        clf = s["task"] == "clf"
        yall = (tgt > W.mean(-1)).astype(np.float32) if clf else tgt
        fit = idx < cut
        X = np.ascontiguousarray(W[:, fit].reshape(-1, T, 1))
        Xq = np.ascontiguousarray(W[:, ~fit].reshape(-1, T, 1))
        return {"X": X, "y": np.ascontiguousarray(yall[:, fit].reshape(-1)),
                "Xq": Xq, "yq": np.ascontiguousarray(yall[:, ~fit].reshape(-1)),
                **({"_cap": B["_cap"]} if "_cap" in B else {})}
    if b in ("ivf", "bytes") or s["kind"] == "graph":
        return dict(B)
    D = {k: B[k] for k in ("X", "y", "Xq", "yq", "_cap") if k in B}
    if b == "graph":
        D.update({k: B[k] for k in ("indptr", "indices")})
        return D
    if "Xq" not in D and "X" in D:          # tsvd / manifold blocks hold X only
        n = D["X"].shape[0]
        cut = max(n - n // 10, 1)
        D["Xq"] = np.ascontiguousarray(D["X"][cut:])
        D["X"] = np.ascontiguousarray(D["X"][:cut]) if s["task"] != "embed" else D["X"]
    if b == "nonneg" or (b == "countclf" and "X" in D and float(D["X"].min()) < 0):
        mn = D["X"].min(axis=0)
        D["X"] = np.ascontiguousarray(D["X"] - mn)
        D["Xq"] = np.ascontiguousarray(np.maximum(D["Xq"] - mn, 0))
    if b in ("raw16",):
        D["X"], D["Xq"] = (np.ascontiguousarray(D[k][:, :16]) for k in ("X", "Xq"))
    if b in ("raw32",):
        D["X"], D["Xq"] = (np.ascontiguousarray(D[k][:, :32]) for k in ("X", "Xq"))
    # The explicit tree-only path consumes the complete prepared train/query
    # matrices. This is a harness input policy, never a kernel shape route.
    subsets = {} if _aft_full_tree_workload(lane) else s["sub"]
    for k, m in subsets.items():
        if k in D:
            D[k] = _stride(D[k], m)
            yk = {"X": "y", "Xq": "yq"}[k]
            if yk in D:
                D[yk] = _stride(D[yk], m)
    t = s["task"]
    if lane == "bayesian-gmm":             # the gmm lane's rule, the same columns for every arm
        D = _tool("bench_board_more").drop_constant_columns(D)
    if s["kind"] == "shap" or t in ("impute",):
        pass
    if t == "impute":
        rng = np.random.default_rng(SEED)
        for k in ("X", "Xq"):
            m = rng.random(D[k].shape) < 0.1
            D[k + "_true"] = D[k].copy()
            D[k] = np.where(m, np.nan, D[k]).astype(np.float32)
    if t == "semi":
        y = D["y"].copy()
        y[np.arange(y.shape[0]) % 10 != 0] = -1
        D["y_semi"] = y
    if t in ("multiclf", "multireg"):
        last = D["X"][:, -1]
        lq = D["Xq"][:, -1]
        D["X"], D["Xq"] = D["X"][:, :-1].copy(), D["Xq"][:, :-1].copy()
        if t == "multiclf":
            D["Y"] = np.stack([D["y"], (last > 0).astype(np.float32)], axis=1)
            D["Yq"] = np.stack([D["yq"], (lq > 0).astype(np.float32)], axis=1)
        else:
            D["Y"] = np.stack([D["y"], last], axis=1)
            D["Yq"] = np.stack([D["yq"], lq], axis=1)
    if t in ("labels", "multilabel"):
        col = int(np.argmax([len(np.unique(D["X"][:, j])) for j in range(D["X"].shape[1])]))
        D["lab"] = D["X"][:, col].astype(np.int64)
        labq = D["Xq"][:, col].astype(np.int64)
        D["labq"] = np.ascontiguousarray(labq[np.isin(labq, D["lab"])])   # seen labels only
    if s.get("iso"):
        Xc = D["X"].astype(np.float64)
        yc = D["y"].astype(np.float64)
        c = np.nan_to_num(np.abs([np.corrcoef(Xc[:, j], yc)[0, 1] if Xc[:, j].std() > 0 else 0
                                  for j in range(Xc.shape[1])]))
        j = int(np.argmax(c))
        D["X"] = np.ascontiguousarray(D["X"][:, j])
        D["Xq"] = np.ascontiguousarray(D["Xq"][:, j])
    if s.get("target") == "gamma":
        D["y"] = D["y"] + (1.0 if D["y"].min() <= 0 else 0.0)
        D["yq"] = D["yq"] + (1.0 if D["yq"].min() <= 0 else 0.0)
    if s.get("target") in ("poisson", "tweedie"):
        D["y"], D["yq"] = np.maximum(D["y"], 0), np.maximum(D["yq"], 0)
    return D


def _derived_params(lane, D, params):
    """Resolve the table's data-dependent placeholders ('half', gamma)."""
    np = _np()
    s = LANES[lane]
    p = dict(params)
    # a per-dataset value (s["dataset_params"]), the same for every arm: this
    # process's dataset is the one _load_block read
    p.update(s.get("dataset_params", {}).get(_DATASET, {}))
    X = D.get("X")
    d = X.shape[1] if X is not None and X.ndim == 2 else 1
    for k, v in list(p.items()):
        if v == "half":
            p[k] = max(1, d // 2)
    if s.get("gamma"):
        p["gamma"] = 1.0 / d
    if s.get("bandwidth"):
        A = X[:1000].astype(np.float64)
        dd = np.sqrt(np.maximum((A * A).sum(1)[:, None] + (A * A).sum(1)[None, :] - 2 * A @ A.T, 0))
        p["bandwidth"] = float(np.quantile(dd[np.triu_indices(A.shape[0], 1)], 0.3))
    if s.get("radius"):                 # radius-neighbors: ~0.1% of the rows inside the ball
        A = X[:1000].astype(np.float64)
        dd = np.sqrt(np.maximum((A * A).sum(1)[:, None] + (A * A).sum(1)[None, :] - 2 * A @ A.T, 0))
        p["radius"] = float(np.quantile(dd[np.triu_indices(A.shape[0], 1)], 0.001))
    if s.get("dictionary"):             # sparse-coder: 64 unit-norm stride rows of the fit block
        A = _stride(X.astype(np.float64), 64)
        A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        p["dictionary"] = np.ascontiguousarray(A, dtype=np.float32)
    if lane == "poly-count-sketch":
        p["gamma"] = 1.0 / d
    if lane == "categorical-nb":
        p["min_categories"] = (np.maximum(X.max(axis=0), D["Xq"].max(axis=0)).astype(np.int64)
                               + 1).tolist()
    if p.get("init") == "seeded":       # t-SNE: ours' 'random' start, handed to every arm that takes one
        p["init"] = np.random.default_rng(SEED).uniform(-5e-5, 5e-5, (X.shape[0], 2)).astype(np.float32)
    # FAIRNESS FIX (2026-10-13 audit): cuML t-SNE ran its adaptive schedule
    # (rate n / 3, exaggeration 24, 78 neighbours at n = 20,000). Its rate and
    # k-NN count are now ours' own resolved values: learning_rate 'auto' as ours
    # resolves it (python/mojolearn/_expansion_ann.py TSNE.fit:
    # max(n / early_exaggeration / 4, 50)) and k = min(n - 1, int(3 x
    # perplexity) + 1) (x_ann/tsne_core.mojo tsne_nn, scikit-learn's rule).
    # Results raced before this change were wrong for this lane; apply extra
    # scrutiny.
    if p.get("learning_rate") == "tsne_auto":
        p["learning_rate"] = max(X.shape[0] / float(p["early_exaggeration"]) / 4.0, 50.0)
    if p.get("n_neighbors") == "tsne_k":
        p["n_neighbors"] = min(X.shape[0] - 1, int(3.0 * float(p["perplexity"])) + 1)
    return p


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

class Skipped(Exception):
    """Our class is not exported by the installed wheel."""


class Runner:
    def __init__(self, info, fit, outputs, infer=None, sync=None, record=None, model_for_hash=None,
                 upload_inputs=None):
        self.model_for_hash = model_for_hash
        #: OUR arm only: the host arrays the timed call is handed, [(name, array)], for the
        #: separate upload probe (`_ours_upload_probe`). None = the builder declared none.
        self.upload_inputs = upload_inputs
        self.info, self._fit, self._out, self._infer, self._sync = info, fit, outputs, infer, sync
        #: what this arm was constructed with, for tools/bench_board_params.py:
        #: BP.arm_record(constructed object), or a declared {"__library__": ...}
        #: dict of the values really passed (a function-call arm)
        # FAIRNESS FIX (2026-10-13 audit): resolve a real library callable
        # before the worker's JSON transport turns it into a string repr.
        self.record = _BP().arm_record(record) if record is not None else None

    def sync_for_receipt(self):
        if self._sync:
            self._sync()

    def fit(self):
        self._fit()
        if self._sync:
            self._sync()

    def infer(self):
        if self._infer is None:
            return False
        self._infer()
        if self._sync:
            self._sync()
        return True

    def outputs(self):
        return self._out()


OURS_ARMS = ("ours", "ours-fast")


def _with_upload(runner, inputs):
    """Declare the host arrays OUR arm's timed call is handed (see `_ours_upload_probe`)."""
    runner.upload_inputs = [(k, v) for k, v in inputs if v is not None]
    return runner


def _probe_words(np, name, a):
    """The 4-byte words one handed host array goes up as, with no arithmetic on its values.
    A floating array goes up as float32 (the words our public calls upload, as in
    tools/classical_two_datasets.py `_ours_upload_probe`); any other dtype as its own bytes.
    A scipy sparse matrix goes up as its CSR parts. Harness glue, outside every clock."""
    if hasattr(a, "detach") and hasattr(a, "numpy"):          # a CPU torch tensor (the layer lanes)
        a = a.detach().numpy()
    if hasattr(a, "tocsr") and hasattr(a, "indptr"):          # scipy CSR / CSC
        return [("%s.%s" % (name, part), np.ascontiguousarray(getattr(a, part), dtype=np.float32)
                 if part == "data" else np.ascontiguousarray(getattr(a, part)))
                for part in ("data", "indices", "indptr")]
    a = np.asarray(a)
    if a.dtype.kind == "f":
        a = np.ascontiguousarray(a, dtype=np.float32)  # glue: the words the fit uploads
    else:
        a = np.ascontiguousarray(a)
    return [(name, a)]


def _ours_upload_probe(runner):
    """A SEPARATE host-to-device upload of OUR arm's inputs, timed on its own and
    OUTSIDE every clock (lane algos-upload-probe, 2026-10-08; the algos twin of
    tools/classical_two_datasets.py `_ours_upload_probe`).

    Ours uploads its host inputs inside its clock and the torch/cupy/gpytorch GPU
    arms put theirs on the device before theirs, so our algos cells had only the
    whole-operation clock (the MIXED ratios in tools/board_clock_audit.py). This
    takes a kernel-only reading beside it: the host arrays the builder declared
    the timed call is handed (`Runner.upload_inputs`) go up once more through the
    resident-array binding (`x_cnn_res_upload`: an enqueue_copy and a device
    synchronize per array), after the round's scored call, its outputs and every
    clock of the round are done. NOTHING MOVES: the scored clock and the
    whole-operation clock are exactly what they were; this is a second
    measurement beside them. The device buffer is allocated and freed outside the
    timed window. No arithmetic on data: a device copy call and a clock. When the
    binding or the inputs are not available the record says why, never a guess."""
    if (runner.info or {}).get("input_home") == "device":
        return {"unavailable": "the inputs are on the device before the clock (input_home=device)"}
    inputs = getattr(runner, "upload_inputs", None)
    if not inputs:
        return {"unavailable": "the builder declared no host inputs for this arm"}
    np = _np()
    try:
        import mojolearn as ml
        b = ml._backend.binding("_mojolearn_x_cnn", ml._backend.default_mode())
    except Exception as exc:  # noqa: BLE001
        return {"unavailable": "no resident-array binding: %r" % (exc,)}
    parts = []
    for name, a in inputs:
        parts.extend(_probe_words(np, name, a))
    sizes = [(name, a, int(a.nbytes) // 4) for name, a in parts]
    sizes = [x for x in sizes if x[2] > 0]
    if not sizes:
        return {"unavailable": "empty inputs"}
    try:
        h = int(b.x_cnn_res_alloc(max(w for _, _, w in sizes)))
    except Exception as exc:  # noqa: BLE001
        return {"unavailable": "res_alloc failed: %r" % (exc,)}
    try:
        t0 = time.perf_counter()
        for _, a, words in sizes:
            b.x_cnn_res_upload(h, a.ctypes.data, words)
        ms = (time.perf_counter() - t0) * 1000.0
    finally:
        try:
            b.x_cnn_res_free(h)
        except Exception:  # noqa: BLE001
            pass
    return {"ms": ms, "bytes": sum(w * 4 for _, _, w in sizes),
            "inputs": [name for name, _, _ in sizes],
            "scope": "host_to_device_upload_separate", "outside_scored_clock": True}


def _ours_class(lane):
    import mojolearn as ml
    s = LANES[lane]
    for name in s["ours"]:
        obj = ml
        ok = True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return name, obj
    raise Skipped("SKIPPED: not built yet (mojolearn %s exports none of %s)" % (
        getattr(ml, "__version__", "?"), ", ".join(s["ours"])))


# The binding each function/optimizer lane's mode is read back from when there is no estimator to ask:
# these modules do not load "_mojolearn_x_<xlane>" (resample.py, model_selection.py _SPLIT_BINDING,
# _training_impl.py _EXT_NAME).
_READBACK_BINDING = {"resample": "_mojolearn_resample", "model_selection": "_mojolearn_x_metrics",
                     "training": "_mojolearn_training", "embedding": "_mojolearn_embedding"}


# THE BINDINGS A RACE NEEDS (2026-10-09, lane/board-quality-bf16). A race of `ours` loads the lane's device binding;
# when the race holds no torch-eager-fp32 arm (the board default ARMS=ours: opponents are stored, never re-raced),
# the layer and optimizer lanes' quality is the untimed own-host reference (bench_board_host_quality.enrich), which
# loads the lane's HOST twin under MOJOLEARN_VENDOR=cpu. An lq RACE of main builds only the bindings a branch changed
# (none), and the box's retry loop builds a binding only when the RACE LOG names it, so before this table no
# x_sequence / x_cnn / embedding / training host twin was ever built and all 17 neural-lane cells carried "Own host
# reference failed". The race now names a missing host twin in its log BEFORE the timed rounds
# (`host_reference_preflight`), and names any further one the reference refused after them.
# `python3 tools/bench_board_algos.py builds --lanes a,b [--arms ours]` prints the BUILDS= list a caller passes.
HOST_REFERENCE_KINDS = ("layer", "optim")   # quality_kind values whose ours-only quality is the host reference
QUALITY_REFERENCE_ARM = "torch-eager-fp32"  # with this arm in the race, quality() compares against it instead


def lane_binding(lane):
    """The lane's device binding basename (`_mojolearn_x_cnn`, `_mojolearn_embedding`, ...)."""
    s = LANES[lane]
    return s.get("binding") or _READBACK_BINDING.get(s["xlane"], "_mojolearn_x_" + s["xlane"])


def needs_host_reference(lane, arms):
    """True when this race's quality for ours can only come from the own-host reference."""
    arms = list(arms)
    return (any(a in OURS_ARMS for a in arms) and QUALITY_REFERENCE_ARM not in arms
            and LANES[lane]["kind"] != "extra" and quality_kind(lane) in HOST_REFERENCE_KINDS)


def race_builds(lane, arms=("ours",), repo=REPO):
    """bindings/build_*.sh basenames (no .sh) a race of `lane` with `arms` needs: the device binding, plus its
    host twin when the quality is the own-host reference. Scripts the tree does not hold are left out."""
    base = lane_binding(lane)[len("_mojolearn_"):]
    want = ["build_" + base]
    if needs_host_reference(lane, arms):
        want += ["build_%s" % b[len("_mojolearn_"):] for b in host_reference_bindings(lane)]
    return [b for b in want if os.path.exists(os.path.join(repo, "bindings", b + ".sh"))]


# The base binding's host twin. A layer or optimizer constructor draws its seeded init through `_buffer.InitStream`
# (`_mojolearn.uniform_init_f32` / `normal_init_f32`), and the CPU vendor serves `_mojolearn` from this twin
# (host_surface family "core"). Without it the host reference of every lane whose class draws an init refused
# "no CPU implementation of _mojolearn.uniform_init_f32 ... no GPU set is loaded, on request" (resnet-block, nv2 v0140 /
# amd a1721, 2026-10-12); batchnorm draws nothing, so its cells passed in the same jobs.
CORE_HOST_BINDING = "_mojolearn_core_host"


def host_reference_bindings(lane):
    """Host binding basenames the own-host reference of `lane` loads: the lane's twin, then the core twin."""
    return [lane_binding(lane) + "_host", CORE_HOST_BINDING]


def _host_binding_path(basename):
    """Where the ours interpreter's loader looks for a host binding (python/mojolearn/_backend.py
    host_module_path: $MOJOLEARN_HOST_DIR, else <package>/host)."""
    d = os.environ.get("MOJOLEARN_HOST_DIR", "").strip() or os.path.join(REPO, "python", "mojolearn", "host")
    return os.path.join(d, basename + ".so")


def _say_needs_host(lane, ds, basename, when):
    # The exact phrase the box retry loop greps: "needs python/mojolearn/<...>_mojolearn_<x>.so"
    print("ALGOS-HOST-REFERENCE lane=%s dataset=%s %s: needs python/mojolearn/host/%s.so "
          "(bindings/build_%s.sh; the untimed own-host quality reference)"
          % (lane, ds, when, basename, basename[len("_mojolearn_"):]), flush=True)


def host_reference_preflight(lane, ds, arms, work):
    """Before any worker starts: when the quality needs the host twin and it is not built, name it in the log and
    stop (exit 4) so the box job builds it and races again; nothing is timed twice. Asked once per work dir: when
    the twin is still missing on the next try (its build failed), the race runs and the quality says why.
    MOJOLEARN_ALGOS_HOST_PREFLIGHT=0 turns it off; an installed-wheel race (MOJOLEARN_BENCH_INSTALLED=1) skips it,
    the wheel's host directory is not the tree's."""
    if os.environ.get("MOJOLEARN_ALGOS_HOST_PREFLIGHT", "1") == "0" or os.environ.get("MOJOLEARN_BENCH_INSTALLED") == "1":
        return None
    if not needs_host_reference(lane, arms):
        return None
    # one twin per ask (the box loop builds the last `needs` line it reads), in host_reference_bindings order
    for basename in host_reference_bindings(lane):
        if not os.path.exists(os.path.join(REPO, "bindings", "build_%s.sh" % basename[len("_mojolearn_"):])):
            continue
        if os.path.exists(_host_binding_path(basename)):
            continue
        marker = os.path.join(work, "host-preflight-%s.asked" % basename)
        if os.path.exists(marker):
            # the scorer preflight (scorer_preflight, own host reference) refuses the race when it cannot load
            print("ALGOS-HOST-REFERENCE lane=%s dataset=%s %s still missing after one ask; the scorer preflight "
                  "decides" % (lane, ds, basename), flush=True)
            continue
        with open(marker, "w") as fh:
            fh.write(now_utc() + "\n")
        _say_needs_host(lane, ds, basename, "preflight")
        return 4
    return None


def _ours_info(lane, est=None):
    import mojolearn as ml
    more = _tool("bench_board_more")
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower()
    info = {"library": "mojolearn", "version": getattr(ml, "__version__", "unknown"),
            "numeric_mode_env": os.environ.get("MOJOLEARN_NUMERIC_MODE"), "device": "gpu",
            "module_path": getattr(ml, "__file__", None), "pre_clock_fit": False,
            "input_home": "host"}
    try:
        mode, how = more._mode_readback(ml, est, LANES[lane].get("binding") or _READBACK_BINDING.get(
            LANES[lane]["xlane"], "_mojolearn_x_" + LANES[lane]["xlane"]))
    except Exception as exc:  # noqa: BLE001
        mode, how = "unknown", "readback failed (%r)" % (exc,)
    info.update(numeric_mode_used=mode, numeric_mode_how=how)
    try:
        info["vendor_used"] = ml.vendor()
    except Exception as exc:  # noqa: BLE001
        info["vendor_used"] = "unavailable (%r)" % (exc,)
    if mode not in (want, "unknown"):
        raise RuntimeError("REFUSED: ours is not %s: the binary reads back %r" % (want.upper(), mode))
    return info


def _resolve(v, lib):
    """Nested estimator placeholders -> instances of `lib`'s class."""
    if isinstance(v, dict) and "__est__" in v:
        name = v["__est__"]
        if lib == "ours":
            import mojolearn as ml
            if not hasattr(ml, name):
                raise Skipped("SKIPPED: not built yet (mojolearn has no %s, a base estimator)" % name)
            cls = getattr(ml, name)
        else:
            cls = _imp(SK_BASE[name])
        return cls(**v["kw"])
    if isinstance(v, list):
        return [_resolve(x, lib) for x in v]
    if isinstance(v, tuple):
        return tuple(_resolve(x, lib) for x in v)
    return v


class _SkClassicalMDS(object):
    """Classical MDS from scikit-learn parts: -0.5 D^2 double-centred and its
    top eigenpairs (KernelPCA precomputed, dense eigh); for a scikit-learn
    without ClassicalMDS."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X):
        from sklearn.decomposition import KernelPCA
        from sklearn.metrics import pairwise_distances
        D2 = pairwise_distances(X, metric="sqeuclidean", n_jobs=-1)
        return KernelPCA(n_components=self.n_components, kernel="precomputed",
                         eigen_solver="dense").fit_transform(-0.5 * D2)

    def __repr__(self):
        return "sklearn KernelPCA(precomputed) on -0.5 D^2 (ClassicalMDS)"


SK_BASE = {"LogisticRegression": "sklearn.linear_model:LogisticRegression",
           "Ridge": "sklearn.linear_model:Ridge", "Lasso": "sklearn.linear_model:Lasso",
           "GaussianNB": "sklearn.naive_bayes:GaussianNB",
           "DecisionTreeClassifier": "sklearn.tree:DecisionTreeClassifier",
           "DecisionTreeRegressor": "sklearn.tree:DecisionTreeRegressor",
           "LinearRegression": "sklearn.linear_model:LinearRegression"}


def _host(a):
    return _tool("classical_two_datasets")._to_host(a)


def _arr(a, dtype=None):
    np = _np()
    if hasattr(a, "to_numpy") and not hasattr(a, "get"):   # cudf / pandas first
        a = a.to_numpy()
    a = _host(a)
    if hasattr(a, "toarray"):
        a = a.toarray()
    return np.asarray(a, dtype=dtype) if dtype else np.asarray(a)


def _cuml_up(arrays, library="cuml"):
    ctd = _tool("classical_two_datasets")
    dev, info = ctd._cuml_setup(arrays, library=library)
    return dev, info, ctd._cupy_sync


_NEURAL_AB_ORIGINAL_LANES = {}


def _configure_neural_ab(path, lane):
    """Apply an explicitly selected neural source recipe before any builder.

    This reads settings only: requested compilation flags are retained as
    provenance, never treated as evidence about the installed binary. The
    descriptor copy is local to this process, shared by ours and opponents.
    """
    for name, descriptor in _NEURAL_AB_ORIGINAL_LANES.items():
        LANES[name] = descriptor
    _NEURAL_AB_ORIGINAL_LANES.clear()
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict) or config.get("schema") != 1 or config.get("mode") != "identical":
        raise ValueError("neural A/B config requires schema=1 and mode='identical'")
    if not isinstance(config.get("id"), str) or not isinstance(config.get("arm"), str):
        raise ValueError("neural A/B config requires string id and arm")
    descriptor = LANES[lane]
    if (descriptor["kind"] not in ("seqmodel", "cnnclf", "layer", "optim")
            and lane not in ("mlp-clf", "mlp-reg")):
        raise ValueError("--neural-ab-config accepts neural lanes only")
    definitions = config.get("compile_defines", [])
    if (not isinstance(definitions, list)
            or any(not isinstance(v, str) or "=" not in v or v.startswith("-D") for v in definitions)):
        raise ValueError("neural A/B compile_defines must be bare NAME=value strings")
    environment = config.get("environment", {})
    unset = config.get("environment_unset", [])
    if (not isinstance(environment, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items())):
        raise ValueError("neural A/B environment requires string names and values")
    if not isinstance(unset, list) or any(not isinstance(k, str) for k in unset):
        raise ValueError("neural A/B environment_unset requires names")
    if environment.get("MOJOLEARN_NUMERIC_MODE", "identical") != "identical":
        raise ValueError("neural A/B mode must remain IDENTICAL")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("neural A/B runtime must be an object")
    if runtime.get("numeric_mode", "identical") != "identical":
        raise ValueError("neural A/B runtime numeric_mode must be identical")
    if runtime.get("operation", "model_default") != "model_default" or runtime.get("owned_weights"):
        raise ValueError("explicit session/owned operations require bench_board_neural.py")
    settings = runtime.get("estimator_settings", {})
    if not isinstance(settings, dict) or set(settings) - {"pool_size"}:
        raise ValueError("algos neural A/B supports estimator_settings.pool_size only")
    selected = dict(descriptor)
    selected["params"] = dict(descriptor["params"])
    if "pool_size" in settings:
        pool = settings["pool_size"]
        if lane != "cnn-clf" or not isinstance(pool, int) or isinstance(pool, bool) or pool < 1:
            raise ValueError("pool_size requires a positive integer on cnn-clf")
        selected["params"]["pool_size"] = pool
    if config["id"] == "NN45" and settings.get("pool_size") != 1:
        raise ValueError("NN45 needs the declared unpooled CNN recipe: pool_size=1 in both arms")
    if "training" in runtime:
        if lane not in ("batchnorm1d", "batchnorm2d", "resnet-block") or type(runtime["training"]) is not bool:
            raise ValueError("training is supported only for neural batchnorm/ResNet layers")
        selected["_neural_ab_training"] = runtime["training"]
    if "graphsage_aggregator" in runtime:
        if lane != "graphsage" or runtime["graphsage_aggregator"] != "mean":
            raise ValueError("NN48 GraphSAGE recipe requires mean aggregation")
        selected["params"]["aggr"] = "mean"
    for name in unset:
        os.environ.pop(name, None)
    os.environ.update(environment)
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    # Complete model construction/preparation stays inside the existing whole
    # operation clock. No dataset size, intrinsic cap or dataset source changes.
    os.environ["MOJOLEARN_BENCH_WHOLE_OPERATION"] = "1"
    if os.environ.get("MOJOLEARN_ALGOS_SMOKE_ROWS"):
        raise ValueError("neural A/B recipe refuses inherited smoke-row caps")
    _NEURAL_AB_ORIGINAL_LANES[lane] = descriptor
    LANES[lane] = selected
    return config


def _neural_ab_layer_mode(layer, descriptor):
    if "_neural_ab_training" in descriptor:
        layer.train(descriptor["_neural_ab_training"])


def build(lane, arm, D):
    kind = LANES[lane]["kind"]
    fn = {"est": _build_est, "dart": _build_dart, "seqmodel": _build_seqmodel,
          "cnnclf": _build_cnnclf, "ts": _build_ts, "graph": _build_graph, "ann": _build_ann,
          "layer": _build_layer, "optim": _build_optim, "linalg": _build_linalg,
          "als": _build_als, "shap": _build_shap, "svgp": _build_svgp, "fn": _build_fn,
          "extra": lambda l, a, d: _EXTRA.build(l, a, d, globals())}[kind]
    why = LANES[lane].get("refused", {}).get(arm)
    if why:                               # FAIRNESS FIX (2026-10-13 audit): see _add's `refused`
        raise RuntimeError("REFUSED: %s is not raced on %s: %s" % (arm, lane, why))
    if arm in OURS_ARMS:
        _ours_class(lane)                     # Skipped when not exported
    return fn(lane, arm, D)


# ---- estimators (the scikit-learn-shaped lanes) ----------------------------

def _est_factory(lane, arm, D):
    s = LANES[lane]
    if arm in OURS_ARMS:
        name, cls = _ours_class(lane)
        params = _derived_params(lane, D, s["params"])
        params = {k: _resolve(v, "ours") for k, v in params.items()}
        for k in s.get("ours_drop", ()):     # an option ours does not take yet (named in mism)
            params.pop(k, None)
        # FAIRNESS FIX (2026-10-13 audit): an option only ours has, set so ours runs the
        # opponent's algorithm (e.g. batch_size=0, the per-sample form); named in the lane's
        # mism. Results raced before this change were wrong for those lanes; apply extra scrutiny.
        params.update(s.get("ours_extra", {}))
        if s.get("score_func"):
            import mojolearn as ml
            if not hasattr(ml, s["score_func"]):
                raise Skipped("SKIPPED: not built yet (mojolearn has no %s)" % s["score_func"])
            f = getattr(ml, s["score_func"])
            if s["score_func"].startswith("mutual"):
                import functools            # the same noise seed as scikit-learn's arm below
                f = functools.partial(f, random_state=SEED)
                f.__name__ = s["score_func"]
            params["score_func"] = f
        return (lambda: cls(**params)), "mojolearn." + name, params
    if arm == "sklearn-cpu":
        try:
            cls = _imp(s["sk"])
        except (ImportError, AttributeError):
            if lane != "classical-mds":
                raise
            # FAIRNESS FIX (2026-10-13 audit): the fallback raced _SkClassicalMDS, this
            # harness's assembly of classical MDS from KernelPCA, as the opponent. Without
            # scikit-learn's own ClassicalMDS (1.8 and later) the arm now refuses by name.
            # Results raced before this change were wrong for this lane; apply extra scrutiny.
            raise RuntimeError("REFUSED: sklearn-cpu is not raced on classical-mds: the installed "
                               "scikit-learn has no sklearn.manifold.ClassicalMDS (1.8 and later); "
                               "a KernelPCA assembly by this harness is not an opponent "
                               "(2026-10-13 fairness audit)")
        params = _derived_params(lane, D, s.get("sk_params", s["params"]))
        params.update(s.get("sk_extra", {}))
        params = {k: _resolve(v, "sklearn") for k, v in params.items()}
        if s.get("score_func"):
            import functools
            import sklearn.feature_selection as fs
            f = getattr(fs, s["score_func"])
            if s["score_func"].startswith("mutual"):
                # FAIRNESS FIX (2026-10-13 audit): mutual_info_* ran at its default n_jobs=None,
                # one core; n_jobs=-1 gives it every core (a scikit-learn >= 1.5 argument).
                # Results raced before this change were wrong for these lanes; apply extra
                # scrutiny.
                f = functools.partial(f, random_state=SEED, n_jobs=-1)
                f.__name__ = s["score_func"]
            params["score_func"] = f
        return (lambda: cls(**params)), s["sk"], params
    if arm == "cuml-gpu":
        cls = _imp(s["cuml"])
        params = _derived_params(lane, D, s.get("cuml_params", s["params"]))
        params.pop("min_categories", None)        # cuML CategoricalNB has no such option
        return (lambda: cls(**params)), s["cuml"], params
    raise SystemExit("no arm %r for lane %r" % (arm, lane))


def _build_est(lane, arm, D):
    np = _np()
    s = LANES[lane]
    t = s["task"]
    make, what, params = _est_factory(lane, arm, D)
    S = {}
    sync = None
    # FAIRNESS FIX (2026-10-13 audit): a `cuml_host_input` branch here handed cuML the numpy arrays
    # in the two random-projection lanes only (fcef3cd62), putting cuML's host-side input handling
    # inside its clock unlike every other cuML lane. The branch is removed; every cuML lane gets
    # its inputs on the device before the clock (the upload is timed apart, upload_ms_untimed), and
    # a lane that still sets the flag is refused. Results raced before this change were wrong for
    # gaussian-rp and sparse-rp; apply extra scrutiny.
    if s.get("cuml_host_input"):
        raise SystemExit("lane %r sets cuml_host_input, removed by the 2026-10-13 fairness audit: "
                         "cuML takes its inputs on the device as in every cuML lane" % lane)
    if arm == "cuml-gpu":
        keys = [k for k in ("X", "y", "Xq") if k in D]
        dev, info, sync = _cuml_up({k: D[k] for k in keys})
        if t == "clf" and "y" in dev:
            import cupy as cp
            dev["y"] = dev["y"].astype(cp.int32)
    else:
        dev = D
        info = (_ours_info(lane) if arm in OURS_ARMS
                else _tool("bench_board_more")._sk_info())
    info["config"] = "%s(%s)" % (what, ", ".join("%s=%r" % (k, v) for k, v in sorted(params.items())
                                                if k != "score_func"))
    X, Xq = dev.get("X"), dev.get("Xq")
    if s["block"] == "countclf" and X is not None and t == "clf":
        # lane apple-fast-nb: the count block as scipy CSR, converted ONCE here outside the clock,
        # for every arm whose estimator takes sparse input: sklearn's MultinomialNB / ComplementNB
        # natively, ours when the FAST x_prep binding was built on Apple with
        # NB_TEXT_CSR (default; `_nb_csr_ready`). A build that cannot take sparse (ours without
        # the export; cuML, whose arrays went up dense through _cuml_up) keeps the dense block.
        sparse_ok = arm == "sklearn-cpu"
        if arm in OURS_ARMS:
            _, cls = _ours_class(lane)
            ready = getattr(cls, "_nb_csr_ready", None)
            sparse_ok = ready is not None and ready()
        if sparse_ok:
            import scipy.sparse as sp
            X, Xq = sp.csr_matrix(X), sp.csr_matrix(Xq)
    y = dev.get("y")
    if t == "semi":
        y = D["y_semi"]
    if t in ("multiclf", "multireg"):
        y = D["Y"]
    if t in ("labels", "multilabel"):
        lab, labq = D["lab"], D["labq"]
        if t == "multilabel":
            offs = np.cumsum([0] + [int(D["X"][:, j].max()) + 1 for j in range(D["X"].shape[1] - 1)])
            sets = [set((D["X"][i].astype(np.int64) + offs).tolist()) for i in range(D["X"].shape[0])]
            setsq = [set((D["Xq"][i].astype(np.int64) + offs).tolist()) for i in range(D["Xq"].shape[0])]
        if arm == "cuml-gpu":
            import cudf
            lab, labq = cudf.Series(lab), cudf.Series(labq)
    if s.get("xy_split"):
        h = D["X"].shape[1] // 2
        Xa, Ya, Xqa, Yqa = X[:, :h], X[:, h:], Xq[:, :h], Xq[:, h:]

    lars_float64 = lane in ("lars", "lars-stable", "lars-direct") and arm == "sklearn-cpu"
    if lars_float64:
        # sklearn Lars explicitly retains consistent input precision for its
        # Gram/Cholesky solver. Its float32 ill-conditioned recovery overflowed
        # on full data; float64 is the supported same-value path. Widen all
        # original float32 rows/labels inside the fit clock, and queries inside
        # the inference clock. Keep eps, iterations and every estimator option.
        info.update(input_dtype=str(X.dtype), target_input_dtype=str(y.dtype),
                    query_input_dtype=str(Xq.dtype), fit_dtype="float64", infer_dtype="float64",
                    input_conversion_inside_clock=True,
                    fit_rows=int(X.shape[0]), eval_rows=int(Xq.shape[0]),
                    precision_reason="Lossless float32-to-float64 input widening for sklearn Lars Gram/Cholesky stability; original failed attempt retained")

    def fit():
        est = make()
        if lars_float64:
            est.fit(np.ascontiguousarray(X, dtype=np.float64),
                    np.ascontiguousarray(y, dtype=np.float64))
        elif t in ("labels",):
            S["out"] = est.fit_transform(lab)
        elif t == "multilabel":
            S["out"] = est.fit_transform(sets)
        elif s.get("xy_split"):
            est.fit(Xa, Ya)
        elif s.get("fit_predict"):
            S["out"] = est.fit_predict(X)
        elif t == "embed":
            S["out"] = est.fit_transform(X)
        elif t in ("clf", "reg", "semi", "select", "multiclf", "multireg") or s.get("supervised"):
            est.fit(X, y)
        else:
            est.fit(X)
        S["est"] = est

    def infer():
        est = S["est"]
        if lars_float64:
            S["pred"] = est.predict(np.ascontiguousarray(Xq, dtype=np.float64))
        elif t in ("clf", "reg", "semi", "outlier", "multiclf", "multireg"):
            S["pred"] = est.predict(Xq)
            if t == "clf" and hasattr(est, "predict_proba") and quality_kind(lane) == "clf":
                try:
                    S["proba"] = est.predict_proba(Xq)
                except Exception:  # noqa: BLE001  (hinge SGD, Perceptron: no probabilities)
                    S["proba"] = None
        elif t == "gmm":
            S["pred"] = est.predict(Xq)
        elif t == "cluster":
            S["pred"] = est.predict(Xq) if hasattr(est, "predict") else None
        elif t == "radius":                 # neighbour count per query (ragged rows)
            _dist, ind = est.radius_neighbors(Xq)
            S["pred"] = [int(np.shape(r)[0]) for r in ind]
        elif t in ("labels",):
            S["pred"] = est.transform(labq)
        elif t == "multilabel":
            S["pred"] = est.transform(setsq)
        elif s.get("xy_split"):
            S["pred"] = est.transform(Xqa, Yqa)
        else:
            S["pred"] = est.transform(Xq)

    def outputs():
        est = S["est"]
        o = {}
        qk = quality_kind(lane)
        if "pred" in S and S["pred"] is not None and qk not in ("cca",):
            o["pred"] = _arr(S["pred"], np.float64)
        if S.get("proba") is not None:
            P = _arr(S["proba"], np.float64)
            o["proba1"] = P[:, -1] if P.ndim == 2 and P.shape[1] == 2 else P
        if "out" in S:
            o["out"] = _arr(S["out"], np.float64)
        if lane in _ITER_LANES:
            # FAIRNESS FIX (2026-10-13 audit): the iterations (and mini-batch steps) each
            # arm of an iterative lane really ran were not recorded, so an opponent that
            # iterated more, or an early stop on one side, could not be seen. n_iter_ and
            # n_steps_ of every arm now reach the quality cell. Results raced before this
            # change were wrong for these lanes; apply extra scrutiny.
            for attr, key in (("n_iter_", "n_iter"), ("n_steps_", "n_steps")):
                v = getattr(est, attr, None)
                if v is not None:
                    o[key] = np.array([int(np.asarray(_arr(v, np.int64)).reshape(-1)[0])],
                                      dtype=np.int64)
        if qk == "cca":
            a, b_ = S["pred"]
            o["xs"], o["ys"] = _arr(a, np.float64), _arr(b_, np.float64)
        if qk in ("cluster",):
            o["labels"] = _arr(getattr(est, "labels_", S.get("out", [])), np.int64).reshape(-1)
        if qk == "gmm":
            o.update(weights=_arr(est.weights_, np.float64), means=_arr(est.means_, np.float64),
                     covariances=_arr(est.covariances_, np.float64))
        if qk in ("pca", "nmf", "recon", "fa"):
            o["components"] = _arr(est.components_, np.float64)
        if qk == "fa":
            o["noise_variance"] = _arr(est.noise_variance_, np.float64)
            o["mean"] = _arr(est.mean_, np.float64)
        if qk == "pca":
            o["mean"] = _arr(getattr(est, "mean_", np.zeros(o["components"].shape[1])), np.float64)
        if qk == "nmf":
            o["W"] = _arr(est.transform(X), np.float64)
        if qk == "recon" and hasattr(est, "mean_"):
            o["mean"] = _arr(est.mean_, np.float64)
        if qk == "perplexity":
            o["components"] = _arr(est.components_, np.float64)
        if qk in ("select", "select-exact"):
            if hasattr(est, "get_support"):
                sup = _arr(est.get_support(), np.int64)
            elif hasattr(est, "support_"):
                sup = _arr(est.support_, np.int64)
            elif hasattr(est, "ranking_"):
                sup = (_arr(est.ranking_, np.int64) == 1).astype(np.int64)
            else:                                   # scores_ top k, ties to the lower index
                sc = _arr(est.scores_, np.float64)
                k = int(params.get("k") or params.get("n_features_to_select"))
                sup = np.zeros(sc.shape[0], dtype=np.int64)
                sup[np.argsort(-sc, kind="stable")[:k]] = 1
            o["support"] = sup
        if qk == "covariance":
            o["covariance"] = _arr(est.covariance_, np.float64)
        if qk == "outlier" and s.get("fit_predict"):
            o["pred"] = o.pop("out")
        return o

    runner = Runner(info, fit, outputs, infer if has_infer(lane) else None, sync,
                    record=_BP().arm_record(make()), model_for_hash=lambda: S.get("est"))
    if arm in OURS_ARMS:
        # the host arrays fit() hands our call (the separate upload probe; never timed in the clock)
        if t == "labels":
            ups = [("lab", lab)]
        elif t == "multilabel":
            ups = []                                  # Python sets: no array goes up as handed
        elif t in ("clf", "reg", "semi", "select", "multiclf", "multireg") or s.get("supervised"):
            ups = [("X", X), ("y", y)]
        else:
            ups = [("X", X)]
        _with_upload(runner, ups)
    return runner


# ---- LSTM / GRU / RNN estimators -------------------------------------------

def synthetic_images(n, n_classes=10, shape=(1, 28, 28), seed=SEED):
    """(X (n, C, H, W) float32, y) : a seed-7 template per class plus N(0, 1) noise."""
    np = _np()
    rng = np.random.default_rng(seed)
    templates = rng.standard_normal((n_classes,) + shape).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    X = templates[y] + rng.standard_normal((n,) + shape).astype(np.float32)
    return np.ascontiguousarray(X, dtype=np.float32), y.astype(np.float32)


# ---- ours' initial weights and row orders, exported for the torch twin --------
# FAIRNESS FIX (2026-10-13 audit): since 86b2bc254 / 4bdc5487e ours draws its initial weights in
# Mojo (`_buffer.InitStream`) and every epoch's row order on the executor (the integer Feistel
# permutation `op_mlp_perm` / x_cnn `epoch_rows_at`), while the torch twins of the RNN/LSTM/GRU
# and CNN lanes still drew numpy default_rng(7) weights and orders and the lane notes claimed
# they were ours'. The parity rule is the same initial weight bytes and the same batches in the
# same order. The conductor now runs `ours-init` under OUR interpreter before the race: it builds
# ours' estimator with the lane's parameters and runs the same steps its fit runs first (the
# InitStream draws, then the schedule seed), writes those weights and every epoch's row order
# (the Feistel permutation below, checked against ours' own `x_cnn_epoch_rows` entry when the
# binding has it) to one npz, and every torch arm loads it. No export, no torch arm: the arm is
# refused by name. Results raced before this change were wrong for this lane; apply extra
# scrutiny.

#: the ours-init npz this worker reads (worker() sets it from --ours-init); None: not exported
_OURS_INIT = None
_M64 = (1 << 64) - 1
OURS_INIT_KINDS = ("seqmodel", "cnnclf")
#: the neural-network lane kinds whose torch arms follow bench_board_neural's compile-timeout rule
NEURAL_KINDS = ("layer", "seqmodel", "cnnclf")


def _export_ours_init(args, lane, ds, tag, env_extra):
    """Run `ours-init` under OUR interpreter (untimed, before any worker): (npz path, None), or
    (None, the refusal reason)."""
    import subprocess
    path = os.path.abspath(os.path.join(args.work, "%s-ours-init.npz" % tag))
    log = os.path.join(args.out, "%s-ours-init.log" % tag)
    cmd = shlex.split(args.ours_python) + [os.path.abspath(__file__), "ours-init", "--lane", lane,
                                           "--dataset", ds, "--data", args.data, "--out", path]
    env = _worker_env("ours")
    env.update(env_extra)
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO,
                                timeout=args.ready_seconds).returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        rc = "%s: %s" % (type(exc).__name__, exc)
    if rc == 0 and os.path.isfile(path):
        print("ALGOS-OURS-INIT lane=%s dataset=%s path=%s" % (lane, ds, path), flush=True)
        return path, None
    return None, ("REFUSED: ours' initial weights and row orders could not be exported (ours-init "
                  "exit %s, see %s), so the torch twin cannot start from ours' weights or follow "
                  "ours' batches (2026-10-13 fairness audit)" % (rc, log))


def epoch_key(seed, epoch):
    """splitmix64's output at state seed + (epoch + 1) * golden: x_cnn/ops.mojo `epoch_key` and
    sequence/mlp.mojo `mlp_epoch_key` (the same words)."""
    z = (int(seed) + (int(epoch) + 1) * 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _mix32(x):
    """murmur3's 32-bit finalizer on a Python int (x_cnn/ops.mojo `_perm_mix`)."""
    x &= 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return x ^ (x >> 16)


def feistel_order(n, seed, epoch, shuffle=True):
    """Epoch `epoch`'s row order over n rows (int64), as ours builds it: the position itself
    without shuffle, else six Feistel rounds on the two ceil(bits/2)-bit halves of the position
    (round key mix(k0 + round * 0x9E3779B9) ^ k1, round function mix(r + rk) & mask), cycle-walked
    into [0, n) (sequence/mlp.mojo `op_mlp_perm`, x_cnn/ops.mojo `epoch_row_val`). Integers only."""
    np = _np()
    n = int(n)
    if not shuffle:
        return np.arange(n, dtype=np.int64)
    bits = 1
    while (1 << bits) < n:
        bits += 1
    h = (bits + 1) // 2
    mask = np.uint32((1 << h) - 1)
    sh = np.uint32(h)
    key = epoch_key(seed, epoch)
    k0, k1 = key & 0xFFFFFFFF, (key >> 32) & 0xFFFFFFFF
    rks = [np.uint32(_mix32(k0 + rd * 0x9E3779B9) ^ k1) for rd in range(6)]
    c1, c2, s16, s13 = np.uint32(0x85EBCA6B), np.uint32(0xC2B2AE35), np.uint32(16), np.uint32(13)

    def mix(v):
        v = v ^ (v >> s16)
        v = v * c1
        v = v ^ (v >> s13)
        v = v * c2
        return v ^ (v >> s16)

    def perm(x):
        left, right = (x >> sh) & mask, x & mask
        for rk in rks:
            f = mix((right + rk).astype(np.uint32)) & mask
            left, right = right, left ^ f
        return ((left << sh) | right).astype(np.uint32)
    with np.errstate(over="ignore"):
        x = perm(np.arange(n, dtype=np.uint32))
        bad = x >= np.uint32(n)
        while bad.any():
            x[bad] = perm(x[bad])
            bad = x >= np.uint32(n)
    return x.astype(np.int64)


def _ours_epoch_rows(n, seed, epoch):
    """Ours' own epoch order (the x_cnn binding's host entry `x_cnn_epoch_rows`), or None
    when this build has no such entry. Used only to check feistel_order."""
    np = _np()
    try:
        from mojolearn import _expansion_cnn as xc
        b = xc._Layer()._binding()
        if not hasattr(b, "x_cnn_epoch_rows"):
            return None
        out = np.empty(int(n), dtype=np.int32)
        b.x_cnn_epoch_rows(out.ctypes.data, [int(n), int(epoch), 1, int(seed) & 0xFFFFFFFF,
                                             (int(seed) >> 32) & 0xFFFFFFFF])
        return out.astype(np.int64)
    except Exception:  # noqa: BLE001  (no binding on this interpreter: unchecked, said so)
        return None


def ours_initial_state(lane, D):
    """Under OUR interpreter: ({name: initial tensor}, orders (epochs, n) int64, meta) of ours'
    estimator for this lane, exactly as its fit starts. RNN/LSTM/GRU: names are its
    `state_dict()` names (`weight_ih_l0` ..., `head.weight`, `head.bias`); CNN: conv<i>.weight /
    conv<i>.bias per conv block and head.weight / head.bias."""
    np = _np()
    import mojolearn as ml
    from mojolearn._buffer import InitStream
    s = LANES[lane]
    make, what, params = _est_factory(lane, "ours", D)
    est = make()
    X, y = D["X"], np.asarray(D["y"])
    n = int(X.shape[0])
    if s["kind"] == "seqmodel":
        # _x_sequence_rnn._RecurrentBase._fit's first steps, in its order
        est._check_X(X, True)
        est.n_outputs_ = (int(np.unique(y).shape[0]) if s["task"] == "clf"
                          else int(np.asarray(y, np.float32).reshape(n, -1).shape[1]))
        rng = InitStream(None if est.random_state is None else int(est.random_state))
        est.params_ = est._init_params(rng)
        state = {k: np.ascontiguousarray(v, np.float32) for k, v in est.state_dict().items()}
        epochs, _bs, shuffle, seed = est._schedule(n, rng)
    elif s["kind"] == "cnnclf":
        # _expansion_cnn.CNNClassifier.fit's first steps: the layers (their InitStream draws),
        # then one splitmix64 seed for every epoch's order
        est._build(int(np.unique(y).shape[0]))
        convs = [layer for layer in est.layers_ if hasattr(layer, "weight_")]
        state = {}
        for i, cv in enumerate(convs):
            state["conv%d.weight" % i] = np.ascontiguousarray(cv.weight_, np.float32)
            state["conv%d.bias" % i] = np.ascontiguousarray(cv.bias_, np.float32)
        state["head.weight"] = np.ascontiguousarray(est.head_.weight_, np.float32)
        state["head.bias"] = np.ascontiguousarray(est.head_.bias_, np.float32)
        epochs, shuffle = int(est.max_iter), bool(est.shuffle)
        seed = int(InitStream(est.random_state).seed)
    else:
        raise ValueError("ours-init: lane %s (kind %s) has no exported init" % (lane, s["kind"]))
    orders = np.stack([feistel_order(n, seed, e, shuffle) for e in range(epochs)]) if epochs else \
        np.zeros((0, n), np.int64)
    check = "not shuffled" if not shuffle else "unchecked (no x_cnn_epoch_rows entry)"
    if shuffle and epochs:
        got = _ours_epoch_rows(n, seed, 0)
        if got is not None:
            if not np.array_equal(got, orders[0]):
                raise RuntimeError("ours-init: feistel_order differs from ours' x_cnn_epoch_rows "
                                   "(epoch 0); the torch twin cannot follow ours' batches")
            check = "epoch 0 equal to ours' x_cnn_epoch_rows"
    meta = {"lane": lane, "kind": s["kind"], "n_rows": n, "epochs": int(epochs),
            "shuffle": bool(shuffle), "order_seed": str(seed), "order_check": check,
            "estimator": what, "mojolearn": getattr(ml, "__version__", "?"),
            "tensors": {k: list(v.shape) for k, v in sorted(state.items())},
            "rule": "ours' initial weights (InitStream) and row orders, as its fit draws them"}
    return state, orders, meta


def write_ours_init(lane, D, path):
    np = _np()
    state, orders, meta = ours_initial_state(lane, D)
    tmp = path + ".tmp.npz"
    np.savez(tmp, orders=orders, meta=np.array(json.dumps(meta, sort_keys=True)),
             **{"state/" + k: v for k, v in state.items()})
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        meta["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    return meta


def ours_init_main(args):
    """`ours-init`: write ours' initial weights and row orders for one race (OUR interpreter)."""
    B, _rec = _load_block(args.lane, args.dataset, args.data)
    D = lane_arrays(args.lane, B)
    meta = write_ours_init(args.lane, D, args.out)
    print("OURS-INIT " + json.dumps(meta, sort_keys=True), flush=True)
    return 0


def _read_ours_init(lane, n_rows, epochs):
    """The torch twin's view of the export: (state {name: array}, orders [(n,) int64], meta).
    Raises (the arm refuses at ready) when there is none or it does not fit this race."""
    np = _np()
    if not _OURS_INIT or not os.path.isfile(_OURS_INIT):
        raise RuntimeError("REFUSED: ours' initial weights and row orders were not exported for "
                           "%s (no --ours-init), so this torch twin cannot start from ours' weights "
                           "or follow ours' batches (2026-10-13 fairness audit)" % lane)
    with np.load(_OURS_INIT) as z:
        meta = json.loads(str(z["meta"]))
        state = {k[len("state/"):]: np.ascontiguousarray(z[k]) for k in z.files if k.startswith("state/")}
        orders = np.ascontiguousarray(z["orders"])
    if meta.get("lane") != lane or int(meta.get("n_rows", -1)) != int(n_rows) \
            or tuple(orders.shape) != (int(epochs), int(n_rows)):
        raise RuntimeError("REFUSED: the ours-init export (%s) does not fit this race (lane %s, %d "
                           "rows, %d epochs)" % (meta, lane, n_rows, epochs))
    with open(_OURS_INIT, "rb") as fh:
        meta["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    return state, [orders[e] for e in range(orders.shape[0])], meta


def _build_cnnclf(lane, arm, D):
    np = _np()
    s = LANES[lane]
    p = s["params"]
    X, y, Xq = D["X"], D["y"], D["Xq"]
    if arm in OURS_ARMS:
        return _build_est(lane, arm, D)
    import torch
    nn = torch.nn
    setting = arm[len("torch-"):]
    mode, prec = setting.split("-")
    dev = _torch_device(torch)
    torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
    torch.backends.cudnn.allow_tf32 = prec == "tf32"
    dt = torch.bfloat16 if prec == "bf16" else None
    Xt, Xqt = torch.from_numpy(X).to(dev), torch.from_numpy(Xq).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    # FAIRNESS FIX (2026-10-13 audit): the weights and row orders here were numpy
    # default_rng(7) draws (conv i from seed + 101 i, the head from seed + 997, one permutation
    # per epoch) that ours stopped using in 86b2bc254 (InitStream weights, device epoch
    # orders), so the two arms started from different weights and trained on different
    # batches. They are now ours' own, exported by `ours-init` (_read_ours_init); without the
    # export this arm refuses. Results raced before this change were wrong for this lane; apply
    # extra scrutiny.
    init_state, init_orders, init_meta = _read_ours_init(lane, X.shape[0], p["max_iter"])
    orders = [torch.from_numpy(o).to(dev) for o in init_orders]
    n_classes = int(np.unique(y).shape[0])
    S = {}

    def ours_init(m):
        """ours' initial weights, from ours' exported init (conv<i>.weight / .bias, head.*)."""
        convs = [mm for mm in m if isinstance(mm, nn.Conv2d)]
        lin = [mm for mm in m if isinstance(mm, nn.Linear)][0]
        pairs = [(cv.weight, "conv%d.weight" % i) for i, cv in enumerate(convs)] \
            + [(cv.bias, "conv%d.bias" % i) for i, cv in enumerate(convs)] \
            + [(lin.weight, "head.weight"), (lin.bias, "head.bias")]
        if len(convs) * 2 + 2 != len(init_state):
            raise RuntimeError("REFUSED: ours-init has %d tensors, the torch net %d"
                               % (len(init_state), len(convs) * 2 + 2))
        with torch.no_grad():
            for t, name in pairs:
                v = init_state[name]
                if tuple(v.shape) != tuple(t.shape):
                    raise RuntimeError("REFUSED: ours-init %s is %s, the torch net's %s"
                                       % (name, tuple(v.shape), tuple(t.shape)))
                t.copy_(torch.from_numpy(v))

    def net():
        c, h, w = p["input_shape"]
        layers = []
        for co in p["conv_channels"]:
            layers += [nn.Conv2d(c, co, p["kernel_size"], padding=p["kernel_size"] // 2), nn.ReLU()]
            if h >= p["pool_size"] and w >= p["pool_size"] and p["pool_size"] > 1:
                layers.append(nn.MaxPool2d(p["pool_size"]))
                h, w = h // p["pool_size"], w // p["pool_size"]
            c = co
        return nn.Sequential(*layers, nn.Flatten(), nn.Linear(c * h * w, n_classes))

    info = {"library": "torch", "version": torch.__version__, "device": "gpu" if dev != "cpu" else "cpu",
            "pre_clock_fit": False, "input_home": "device", "setting": setting,
            "config": "Conv2d-ReLU-MaxPool2d x %s + Linear, SGD lr %g momentum %g, batch %d, %d "
                      "epochs, ours' exported initial weights and row orders, %s %s on %s"
                      % (p["conv_channels"], p["learning_rate"], p["momentum"], p["batch_size"],
                         p["max_iter"], mode, prec, dev),
            "ours_init": init_meta}
    ours_init(net())                      # untimed: the export fits this net, or the arm refuses now
    lossf = nn.CrossEntropyLoss()

    def run(m, x):
        if dt is not None:
            with torch.autocast(device_type=dev, dtype=dt):
                return m(x)
        return m(x)

    # FAIRNESS FIX (2026-10-13 audit): every fit built and torch.compile'd a new module, so the
    # compile arm compiled (or failed dynamo's guards and recompiled) inside every timed round.
    # The module and its compiled callable are built ONCE here (the untimed warm-up round
    # compiles); each fit puts ours' initial weights back into the same parameters in place (the
    # host-to-device copy stays inside the clock, as before) and builds a fresh SGD over them.
    # Results raced before this change were wrong for this lane; apply extra scrutiny.
    torch.manual_seed(SEED)
    m_once = net().to(dev)
    fwd_once = torch.compile(m_once) if mode == "compile" else m_once
    info["build_once"] = "module and compiled callable built once; initial weights reloaded per fit"

    def fit():
        torch.manual_seed(SEED)
        m = m_once
        ours_init(m)
        fwd = fwd_once
        opt = torch.optim.SGD(m.parameters(), lr=p["learning_rate"], momentum=p["momentum"],
                              dampening=p["dampening"], nesterov=p["nesterov"],
                              weight_decay=p["weight_decay"])
        bs = p["batch_size"]
        for order in orders:
            for st in range(0, X.shape[0], bs):
                b = order[st:st + bs]
                opt.zero_grad(set_to_none=True)
                lossf(run(fwd, Xt[b]).float(), yt[b]).backward()
                opt.step()
        _torch_sync(torch, dev)
        S["fwd"] = fwd

    def infer():
        with torch.no_grad():
            out = torch.cat([run(S["fwd"], Xqt[i:i + 4096]).float() for i in range(0, Xqt.shape[0], 4096)])
        S["pred"] = out.argmax(1)
        _torch_sync(torch, dev)
    rec = dict(__library__="torch", seed=p["random_state"], learning_rate=p["learning_rate"],
               momentum=p["momentum"], dampening=p["dampening"], nesterov=p["nesterov"],
               weight_decay=p["weight_decay"], batch_size=p["batch_size"], max_iter=p["max_iter"],
               shuffle=p["shuffle"], kernel_size=p["kernel_size"], pool_size=p["pool_size"],
               conv_channels=list(p["conv_channels"]), optimizer="sgd")
    return Runner(info, fit, lambda: {"pred": _torch_metric_array(S["pred"])}, infer, record=rec)


def _build_seqmodel(lane, arm, D):
    np = _np()
    s = LANES[lane]
    p = s["params"]
    X, y, Xq = D["X"], D["y"], D["Xq"]
    clf = s["task"] == "clf"
    S = {}
    if arm in OURS_ARMS:
        return _build_est(lane, arm, D)
    import torch
    nn = torch.nn
    setting = arm[len("torch-"):]
    mode, prec = setting.split("-")
    dev = _torch_device(torch)
    torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
    torch.backends.cudnn.allow_tf32 = prec == "tf32"
    dt = torch.bfloat16 if prec == "bf16" else None
    Xt, Xqt = torch.from_numpy(X).to(dev), torch.from_numpy(Xq).to(dev)
    yt = torch.from_numpy(y.astype(np.int64) if clf else y.astype(np.float32)).to(dev)
    cell = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[s["cell"]]
    ckw = {"nonlinearity": p["nonlinearity"]} if s["cell"] == "RNN" else {}
    oo = p["optimizer_options"]
    n_out = int(np.unique(y).shape[0]) if clf else 1
    # FAIRNESS FIX (2026-10-13 audit): the weights and row orders here were numpy
    # default_rng(7) draws (uniform(+-1/sqrt(H)) per tensor, then one permutation per epoch)
    # that ours stopped using in 86b2bc254 / 4bdc5487e (InitStream weights, executor Feistel
    # orders): the arms started from different weights and trained on different batches while
    # the lane notes said they were ours'. They are now ours' own, exported by `ours-init`
    # (_read_ours_init); without the export this arm refuses. Results raced before this change
    # were wrong for this lane; apply extra scrutiny.
    init_state, init_orders, init_meta = _read_ours_init(lane, X.shape[0], p["max_epochs"])
    init = {(k if k.startswith("head.") else "rnn." + k): torch.from_numpy(v)
            for k, v in init_state.items()}
    orders = [torch.from_numpy(o).to(dev) for o in init_orders]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = cell(X.shape[2], p["hidden_size"], num_layers=p["num_layers"], batch_first=True,
                            **ckw)
            self.head = nn.Linear(p["hidden_size"], n_out)

        def forward(self, x):
            out = self.rnn(x)[0]
            return self.head(out[:, -1])

    info = {"library": "torch", "version": torch.__version__, "device": "gpu" if dev != "cpu" else "cpu",
            "pre_clock_fit": False, "input_home": "device", "setting": setting,
            "config": "nn.%s(1, %d, batch_first) + Linear, Adam lr %g betas %s eps %g wd %g, batch %d, "
                      "%d epochs, ours' exported seed-%d initial weights and row orders, %s %s on %s"
                      % (s["cell"], p["hidden_size"], p["learning_rate"], oo["betas"], oo["eps"],
                         oo["weight_decay"], p["batch_size"], p["max_epochs"], p["random_state"], mode,
                         prec, dev),
            "ours_init": init_meta}
    Net().load_state_dict(init)           # untimed, strict: the export fits this net or the arm refuses now
    lossf = nn.CrossEntropyLoss() if clf else nn.MSELoss()
    rec = dict(__library__="torch", seed=p["random_state"], hidden_size=p["hidden_size"],
               num_layers=p["num_layers"], learning_rate=p["learning_rate"], batch_size=p["batch_size"],
               max_epochs=p["max_epochs"], shuffle=p["shuffle"], optimizer="adam",
               betas=list(oo["betas"]), eps=oo["eps"], weight_decay=oo["weight_decay"], **ckw)

    def run(net, x):
        if dt is not None:
            with torch.autocast(device_type=dev, dtype=dt):
                return net(x)
        return net(x)

    # FAIRNESS FIX (2026-10-13 audit): every fit built and torch.compile'd a new module, so the
    # compile arm compiled (or recompiled) inside every timed round. The module and its compiled
    # callable are built ONCE here (the untimed warm-up round compiles); each fit loads ours'
    # initial weights into the same parameters in place (load_state_dict copies; the
    # host-to-device copy stays inside the clock) and builds a fresh Adam over them. Results raced
    # before this change were wrong for this lane; apply extra scrutiny.
    torch.manual_seed(SEED)
    net_once = Net().to(dev)
    fwd_once = torch.compile(net_once) if mode == "compile" else net_once
    info["build_once"] = "module and compiled callable built once; initial weights reloaded per fit"

    def fit():
        torch.manual_seed(SEED)
        net = net_once
        net.load_state_dict(init)
        fwd = fwd_once
        opt = torch.optim.Adam(net.parameters(), lr=p["learning_rate"], betas=tuple(oo["betas"]),
                               eps=oo["eps"], weight_decay=oo["weight_decay"])
        bs = p["batch_size"]
        for order in orders:
            for st in range(0, X.shape[0], bs):
                b = order[st:st + bs]
                opt.zero_grad(set_to_none=True)
                out = run(fwd, Xt[b]).float()
                loss = lossf(out, yt[b]) if clf else lossf(out[:, 0], yt[b])
                loss.backward()
                opt.step()
        _torch_sync(torch, dev)
        S["net"], S["fwd"] = net, fwd

    def infer():
        with torch.no_grad():                    # in chunks, as ours' predict_chunk does
            out = torch.cat([run(S["fwd"], Xqt[i:i + 4096]).float()
                             for i in range(0, Xqt.shape[0], 4096)])
        S["pred"] = out.argmax(1) if clf else out[:, 0]
        _torch_sync(torch, dev)
    return Runner(info, fit, lambda: {"pred": _torch_metric_array(S["pred"])}, infer, record=rec)


# ---- DART -----------------------------------------------------------------

def _build_dart(lane, arm, D):
    np = _np()
    s = LANES[lane]
    p = s["params"]
    X, y, Xq = D["X"], D["y"], D["Xq"]
    S = {}
    if arm in OURS_ARMS:
        return _build_est(lane, arm, D)
    if arm == "lightgbm-cpu":
        import lightgbm as lgb
        kw = dict(boosting_type="dart", n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                  max_depth=p["max_depth"], num_leaves=p["num_leaves"], drop_rate=p["drop_rate"],
                  skip_drop=p["skip_drop"], max_drop=p["max_drop"],
                  xgboost_dart_mode=p["xgboost_dart_mode"], uniform_drop=p["uniform_drop"],
                  min_child_samples=p["min_child_samples"], reg_lambda=p["reg_lambda"],
                  reg_alpha=p["reg_alpha"], max_bin=p["max_bin"], max_delta_step=p["max_delta_step"],
                  subsample=p["subsample"], subsample_freq=p["subsample_freq"],
                  colsample_bytree=p["colsample_bytree"], drop_seed=p["drop_seed"],
                  feature_fraction_seed=p["feature_fraction_seed"], bagging_seed=p["bagging_seed"],
                  random_state=p["random_state"], n_jobs=-1, verbose=-1)
        make = lambda: (lgb.LGBMClassifier if s["task"] == "clf" else lgb.LGBMRegressor)(**kw)  # noqa: E731
        info = {"library": "lightgbm", "version": lgb.__version__, "device": "cpu"}
    else:
        import xgboost as xgb
        gpu = arm == "xgboost-gpu"
        kw = dict(booster="dart", n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                  max_depth=p["max_depth"], rate_drop=p["drop_rate"], skip_drop=p["skip_drop"],
                  sample_type="uniform", normalize_type="tree", one_drop=False,
                  reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"], max_bin=p["max_bin"],
                  max_delta_step=p["max_delta_step"], subsample=p["subsample"],
                  colsample_bytree=p["colsample_bytree"], min_child_weight=1.0,
                  tree_method="hist", device="cuda" if gpu else "cpu",
                  random_state=p["random_state"], n_jobs=-1)
        make = lambda: (xgb.XGBClassifier if s["task"] == "clf" else xgb.XGBRegressor)(**kw)  # noqa: E731
        info = {"library": "xgboost", "version": xgb.__version__, "device": "gpu" if gpu else "cpu"}
    info.update(pre_clock_fit=False, input_home="host",
                config="%s(%s)" % (arm, ", ".join("%s=%r" % kv for kv in sorted(kw.items()))))
    yi = y.astype(np.int32) if s["task"] == "clf" else y

    def fit():
        S["m"] = make().fit(X, yi)

    def infer():
        S["proba"] = (S["m"].predict_proba(Xq) if s["task"] == "clf" else S["m"].predict(Xq))

    def outputs():
        P = np.asarray(S["proba"], dtype=np.float64)
        if s["task"] != "clf":
            return {"pred": P.reshape(-1)}
        return {"pred": (P[:, 1] >= 0.5).astype(np.float64), "proba1": P[:, 1]}
    return Runner(info, fit, outputs, infer, record=_BP().arm_record(make()))


# ---- time series ----------------------------------------------------------

def _joblib_map(fn, rows):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=-1, backend="loky")(delayed(fn)(r) for r in rows)


def _ts_map(arm):
    """How a CPU time-series arm maps its per-series fit over the batch.

    FAIRNESS FIX (2026-10-13 audit): every CPU time-series opponent (statsforecast, statsmodels,
    arch) was timed through a joblib loky process pool, one task per series, so on cheap models
    (Croston: 64 sub-millisecond forecasts) the pool's dispatch, not the library, filled its
    clock (statsforecast CrostonClassic 54-92 ms against ours 2-5 ms). Each such arm now has a
    `-serial` twin that loops over the series in its own process; the board keeps the faster.
    Results raced before this change were wrong for these lanes; apply extra scrutiny."""
    if arm.endswith("-serial"):
        return lambda fn, rows: [fn(r) for r in rows]
    return _joblib_map


_STL_NAMES = ("period", "seasonal", "trend", "low_pass", "seasonal_deg", "trend_deg",
              "low_pass_deg", "robust", "seasonal_jump", "trend_jump", "low_pass_jump")
_PROPHET_NAMES = ("growth", "n_changepoints", "changepoint_range", "yearly_seasonality",
                  "weekly_seasonality", "daily_seasonality", "seasonality_mode",
                  "seasonality_prior_scale", "holidays_prior_scale", "changepoint_prior_scale")


def _STL_KW(params):
    """STL's arguments, the same on ours and statsmodels (every one explicit)."""
    return {k: params[k] for k in _STL_NAMES}


def _PROPHET_KW(params):
    """prophet.Prophet's arguments: the lane's, plus uncertainty_samples=0 (ours
    computes no intervals) and mcmc_samples=0 (MAP, as ours)."""
    kw = {k: params[k] for k in _PROPHET_NAMES}
    kw.update(uncertainty_samples=0, mcmc_samples=0)
    return kw


def _sf_autoarima_kw(p):
    """statsforecast AutoARIMA over ours' / cuML's exhaustive grid."""
    # FAIRNESS FIX (2026-10-13 audit): statsforecast's fitting method and d test were left at its
    # defaults (None -> 'CSS-ML' at approximation=False; 'kpss'). Both are now passed explicitly
    # (the same values, so the same computation); ours is pinned to method='ml' and test='kpss'.
    # Results raced before this change were wrong for this lane; apply extra scrutiny.
    return dict(season_length=1, seasonal=False, max_p=p["max_p"], max_q=p["max_q"], max_d=p["max_d"],
                max_order=p["max_p"] + p["max_q"], stepwise=p["stepwise"], ic=p["ic"],
                allowmean=p["allow_intercept"], allowdrift=p["allow_intercept"], approximation=False,
                method="CSS-ML", test="kpss")


def _ts_one(lib, task, params, y, h):
    """One series through one library: (forecast[h], extra float)."""
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    if lib == "statsmodels":
        if task == "decompose":
            from statsmodels.tsa.seasonal import STL
            r = STL(y, **_STL_KW(params)).fit()
            return np.asarray(r.trend + r.seasonal)
        if task == "forecast" and "damped" in params:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None,
                                     initialization_method="estimated").fit()
            return np.asarray(m.forecast(h))
        from statsmodels.tsa.forecasting.theta import ThetaModel
        # FAIRNESS FIX (2026-10-13 audit): the deseasonalization settings were statsmodels'
        # defaults, unnamed. They are passed explicitly (the same values, so the same work):
        # deseasonalize after the seasonality test, method 'auto' (multiplicative for a positive
        # series, else additive: the rule ours and statsforecast apply to
        # decomposition_type='multiplicative'). Results raced before this change were wrong for
        # this lane; apply extra scrutiny.
        m = ThetaModel(y, period=params.get("season_length", 24), deseasonalize=True,
                       use_test=True, method="auto").fit()
        return np.asarray(m.forecast(h))
    if lib == "arch":
        from arch import arch_model
        r = arch_model(y, vol="GARCH", p=params["p"], o=params["o"], q=params["q"],
                       power=params["power"], mean=params["mean"], dist=params["dist"],
                       rescale=False).fit(disp="off")
        f = r.forecast(horizon=h, reindex=False).variance.values[-1]
        return np.concatenate([[r.loglikelihood], f])
    if lib == "prophet":
        import logging
        import pandas as pd
        from prophet import Prophet
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        ds = pd.date_range("2024-01-01", periods=len(y) + h, freq="h")
        m = Prophet(**_PROPHET_KW(params))
        m.fit(pd.DataFrame({"ds": ds[:len(y)], "y": y}), iter=int(params["max_iter"]))
        return m.predict(pd.DataFrame({"ds": ds[len(y):]}))["yhat"].to_numpy()
    raise ValueError(lib)


def _build_ts(lane, arm, D):
    np = _np()
    s = LANES[lane]
    t = s["task"]
    Y = D["Yfit"].astype(np.float64)
    h = TS_H
    S = {}
    sync = None
    p = s["params"]
    if arm in OURS_ARMS:
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        info["config"] = "mojolearn.%s(%s).fit(Y (%d, %d))" % (name, p, Y.shape[0], Y.shape[1])
        Y32 = D["Yfit"]
        # prophet's time axis, as the prophet arm builds it: hourly from 2024-01-01
        ds = np.datetime64("2024-01-01T00", "h") + np.arange(Y32.shape[1] + h).astype("timedelta64[h]")
        # autoarima: the search's maxiter, the lane's pinned maxiter; the env switch of the
        # 2026-10-04 quality A/B (below) restores 20 and the params record says so
        _sm = 20 if os.environ.get("MOJOLEARN_AUTOARIMA_SEARCH_QOLD") == "1" else int(p.get("maxiter", 1000))

        def fit():
            if lane == "autoarima":           # the cuML shape: construct on the batch, search, fit
                m = cls(Y32)
                # QUALITY FIX (lane apple-fast-q-misc, 2026-10-04; old: env
                # MOJOLEARN_AUTOARIMA_SEARCH_QOLD=1). The search used cuML's
                # default maxiter=20, so every candidate order was scored on a
                # 20-step likelihood fit while statsforecast scores each candidate
                # at convergence; the IC then compared unconverged fits (audit:
                # taxi-hourly forecast_rmse FAST 74.66, IDENTICAL 73.63,
                # statsforecast 68.21). The candidates now get the fit's own
                # maxiter (1000), the opponent's rule.
                # FAIRNESS FIX (2026-10-13 audit): the search space and budget were implicit:
                # fit_intercept='auto' (k = 1 only, half of statsforecast's grid) and fit() at its
                # default maxiter. Now the lane's pinned values: fit_intercept=[0, 1] (every order
                # with and without the constant, statsforecast's grid), maxiter and method on both
                # search and fit, test='kpss'. Results raced before this change were wrong for
                # this lane; apply extra scrutiny.
                m.search(s=1, d=range(0, p["max_d"] + 1), p=range(0, p["max_p"] + 1),
                         q=range(0, p["max_q"] + 1), P=range(1), D=range(1), Q=range(1), ic=p["ic"],
                         fit_intercept=list(p["fit_intercept"]), test="kpss", maxiter=_sm,
                         method=p["method"])
                m.fit(maxiter=int(p["maxiter"]), method=p["method"])
                S["est"] = m
            elif lane == "stl":               # statsmodels' shape: STL(endog, period).fit() -> result
                S["est"] = cls(Y32, **_STL_KW(p)).fit()
            elif lane == "var":               # statsmodels' shape: VAR(endog (n_obs, K)).fit(maxlags)
                S["est"] = cls(np.ascontiguousarray(Y32[:16].T)).fit(
                    maxlags=p["maxlags"], method=p["method"], ic=p["ic"], trend=p["trend"])
            elif t == "garch":                # arch's shape; the variance forecast horizon is fixed at fit
                S["est"] = cls(**p).fit(Y32, horizon=h)
            elif lane == "prophet":           # prophet's shape: fit(ds, y), then predict(future ds)
                S["est"] = cls(**p).fit(ds[:Y32.shape[1]], Y32)
            else:
                S["est"] = cls(**p).fit(Y32)

        def infer():
            e = S["est"]
            if t == "decompose":
                tr = getattr(e, "trend", None)
                if tr is None or np.ndim(tr) == 0:
                    tr, se = e.trend_, e.seasonal_
                else:
                    se = e.seasonal
                S["fc"] = _arr(tr, np.float64) + _arr(se, np.float64)
            elif lane == "var":
                S["fc"] = _arr(e.forecast(np.ascontiguousarray(Y32[:16].T[-e.k_ar:]), h), np.float64).T
            else:
                if lane == "autoarima" or t == "garch":
                    fc = e.forecast(h)
                elif lane == "prophet":
                    fc = e.predict(ds[Y32.shape[1]:])
                else:                          # statsforecast's shape: predict(h) -> {"mean": ...}
                    fc = e.predict(h)
                    if isinstance(fc, dict):
                        fc = fc["mean"]
                fc = _arr(fc, np.float64)
                if fc.ndim == 2 and fc.shape[0] == h and fc.shape[1] == Y32.shape[0] != h:
                    fc = fc.T                  # (h, batch), cuML's layout
                S["fc"] = fc

        def outputs():
            o = {"forecast": _arr(S["fc"], np.float64)} if t != "decompose" else {
                "components": _arr(S["fc"], np.float64)}
            if t == "garch":
                o["llf"] = _arr(S["est"].loglikelihood_, np.float64).reshape(-1)
            return o
        rec = {"__library__": "mojolearn"}
        if lane == "autoarima":
            # FAIRNESS FIX (2026-10-13 audit): the record said fit_intercept='auto' and carried no
            # optimizer budget. It now carries what the calls above are given (fit_intercept and
            # maxiter, compared by tools/bench_board_params.py against a cuML arm). Results raced
            # before this change were wrong for this lane; apply extra scrutiny.
            rec.update(p, maxiter=_sm, fit_maxiter=int(p["maxiter"]), test="kpss",
                       search="s=1, d 0..%d, p 0..%d, q 0..%d, fit_intercept=%r, KPSS"
                       % (p["max_d"], p["max_p"], p["max_q"], list(p["fit_intercept"])))
        elif lane not in ("stl", "var"):
            try:
                rec = _BP().arm_record(cls(**p))
            except Exception:  # noqa: BLE001  (a constructor that needs data: declare)
                rec.update(p)
        else:
            rec.update(p)
        fit0 = fit

        def fit():                        # the forecast is inside the clock, as for every arm
            fit0()
            infer()
        return Runner(info, fit, outputs, None, record=rec)
    lib = s["other"][arm]
    if lib == "statsforecast":
        import statsforecast
        from statsforecast import models as sfm
        info = {"library": "statsforecast", "version": statsforecast.__version__, "device": "cpu",
                "pre_clock_fit": False, "input_home": "host"}
        # the lane's statsforecast class (LANES[lane]["sf"]) with the lane's own
        # arguments: the theta family and Croston take ours' params as they are
        if lane == "autoarima":
            kw = _sf_autoarima_kw(p)
        elif lane == "damped-ets":
            kw = dict(season_length=p["season_length"], model=p["model"], damped=p["damped"])
        else:
            kw = dict(p)
        ctor = getattr(sfm, s["sf"])
        mk = lambda: ctor(**kw)  # noqa: E731
        info["config"] = "%s(%s)" % (ctor.__name__, ", ".join("%s=%r" % i for i in sorted(kw.items())))
        rec = dict(kw, __library__="statsforecast")

        def one(y):
            return mk().forecast(y=y, h=h)["mean"]

        info["config"] += ("; one process, the series in a loop" if arm.endswith("-serial")
                           else "; one series per joblib loky task, n_jobs=-1")
        _map = _ts_map(arm)   # FAIRNESS FIX (2026-10-13 audit): see _ts_map

        def fit():
            S["fc"] = np.stack(_map(one, list(Y)))
        return Runner(info, fit, lambda: {"forecast": S["fc"]}, None, record=rec)
    if lib == "cuml":
        # FAIRNESS FIX (2026-10-13 audit): this arm cast Y to float64 (Y above) for cuML, whose
        # AutoARIMA is float64 only, on an L40S with 1/64-rate FP64, while ours runs float32; and
        # it left maxiter, method and the constant grid at cuML's defaults. The lane refuses this
        # arm (LANES["autoarima"]["refused"]); should that entry go, this arm still runs only if
        # cuML keeps float32 input as float32 (probed here, outside the clock), and then with
        # ours' float32 series and the lane's pinned search (fit_intercept, maxiter, method).
        # Results raced before this change were wrong for this lane; apply extra scrutiny.
        from cuml.tsa.auto_arima import AutoARIMA
        Yt32 = np.ascontiguousarray(D["Yfit"].T, dtype=np.float32)
        try:
            got = str(np.asarray(AutoARIMA(Yt32, output_type="numpy").d_y).dtype)
        except Exception as exc:  # noqa: BLE001  (a refusal of float32, or no dtype to read)
            got = "unverified (%s: %s)" % (type(exc).__name__, exc)
        if got != "float32":
            raise Skipped("REFUSED: cuML AutoARIMA does not compute in float32 (it holds float32 "
                          "input as %s); ours is float32 only, so there is no same-precision "
                          "comparison" % got)
        dev, info, sync = _cuml_up({"Yt": Yt32})
        info["config"] = ("cuml.tsa.auto_arima.AutoARIMA(Yt float32).search(s=1, d=range(%d), "
                          "p=range(%d), q=range(%d), fit_intercept=%r, ic=%r, maxiter=%d, "
                          "method=%r); fit(maxiter=%d, method=%r)"
                          % (p["max_d"] + 1, p["max_p"] + 1, p["max_q"] + 1, list(p["fit_intercept"]),
                             p["ic"], p["maxiter"], p["method"], p["maxiter"], p["method"]))

        def fit():
            m = AutoARIMA(dev["Yt"], output_type="numpy")
            m.search(s=1, d=range(0, p["max_d"] + 1), p=range(0, p["max_p"] + 1),
                     q=range(0, p["max_q"] + 1), P=range(1), D=range(1), Q=range(1), ic=p["ic"],
                     fit_intercept=list(p["fit_intercept"]), test="kpss", maxiter=int(p["maxiter"]),
                     method=p["method"])
            m.fit(maxiter=int(p["maxiter"]), method=p["method"])
            S["fc"] = m.forecast(h)
        rec = dict(p, __library__="cuml", fit_maxiter=int(p["maxiter"]), test="kpss")
        return Runner(info, fit, lambda: {"forecast": _arr(S["fc"], np.float64).T}, None, sync,
                      record=rec)
    import importlib as il
    modname = {"statsmodels": "statsmodels", "arch": "arch", "prophet": "prophet"}[lib]
    mod = il.import_module(modname)
    info = {"library": lib, "version": getattr(mod, "__version__", "unknown"), "device": "cpu",
            "pre_clock_fit": False, "input_home": "host",
            "config": "%s per series, joblib loky n_jobs=-1, params %s" % (lib, p)}
    if t == "var":
        from statsmodels.tsa.api import VAR
        info["config"] = "statsmodels VAR(Y[:16].T).fit(maxlags=%d, method=%r, ic=%r, trend=%r)" % (
            p["maxlags"], p["method"], p["ic"], p["trend"])

        def fit():
            r = VAR(Y[:16].T).fit(maxlags=p["maxlags"], method=p["method"], ic=p["ic"], trend=p["trend"])
            S["fc"] = r.forecast(Y[:16].T[-r.k_ar:], h).T
        return Runner(info, fit, lambda: {"forecast": S["fc"]}, None,
                      record=dict(p, __library__="statsmodels"))
    task = "decompose" if t == "decompose" else "forecast"
    _map = _ts_map(arm)       # FAIRNESS FIX (2026-10-13 audit): see _ts_map
    if arm.endswith("-serial"):
        info["config"] = "%s per series in one process (a loop), params %s" % (lib, p)

    def fit():
        S["res"] = np.stack(_map(lambda y: _ts_one(lib, task, p, y, h), list(Y)))

    def outputs():
        r = S["res"]
        if t == "decompose":
            return {"components": r}
        if t == "garch":
            return {"llf": r[:, 0], "forecast": r[:, 1:]}
        return {"forecast": r}
    rec = {"__library__": lib}
    if lib == "prophet":
        rec.update(_PROPHET_KW(p), max_iter=p["max_iter"])
    elif lib == "statsmodels" and t == "decompose":
        rec.update(_STL_KW(p))
    elif lib == "statsmodels" and "damped" in p:
        rec.update(trend="add", damped_trend=True, seasonal=None, initialization_method="estimated")
    elif lib == "statsmodels":
        rec.update(period=p.get("season_length", 24), method="theta (statsmodels ThetaModel defaults)")
    else:
        rec.update(p)
    return Runner(info, fit, outputs, None, record=rec)


# ---- graph ----------------------------------------------------------------

def _build_graph(lane, arm, D):
    np = _np()
    s = LANES[lane]
    t = s["task"]
    ip, ix = (D["indptr2"], D["indices2"]) if t == "components" else (D["indptr"], D["indices"])
    n = ip.shape[0] - 1
    S = {}
    sync = None
    p = s["params"]
    if arm in OURS_ARMS:
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        # lane/neural-pass69 (2026-10-01) for components, lane gap-graph
        # (2026-10-08, docs/plans/gaps-2026-10-08.md 5.1/5.2) for pagerank
        # and louvain: the sparse graph, as the opponents receive it (scipy's
        # csgraph, networkx and cuGraph read the CSR); the (indptr, indices,
        # n) tuple is PageRank's and Louvain's CSR entry too (`knn_graph`
        # writes each row's columns ascending, which the device checks). No
        # dense adjacency anywhere.
        csr = (np.ascontiguousarray(ip, dtype=np.int32), np.ascontiguousarray(ix, dtype=np.int32), n)
        info["pre_clock_fit"] = False
        info["config"] = "mojolearn.%s(%s) on the CSR adjacency (%d nodes, %d edges)" % (name, p, n, ix.shape[0])

        def fit():
            if t == "components":
                S["lab"] = cls(csr, directed=False)[1]
            else:
                S["e"] = cls(**p).fit(csr)

        def outputs():
            if t == "components":
                return {"labels": _arr(S["lab"], np.int64)}
            e = S["e"]
            sc = getattr(e, "pagerank_", None)
            if t == "pagerank":
                return {"scores": _arr(sc if sc is not None else e.scores_, np.float64)}
            return {"labels": _arr(e.labels_, np.int64)}
        rec = dict(p, __library__="mojolearn")
        if t == "pagerank":
            # FAIRNESS FIX (2026-10-13 audit): the stopping threshold (ours stops
            # when the L1 change < n x tol, _expansion_neighbors.py PageRank),
            # declared so the parameter check compares it with cuGraph's. Results
            # raced before this change were wrong for this lane; apply extra
            # scrutiny.
            rec["l1_stop_threshold"] = n * float(p["tol"])
        return Runner(info, fit, outputs, record=rec)
    lib = s["other"][arm]
    if lib == "networkx":
        import networkx as nx
        import scipy.sparse as sp
        A = sp.csr_matrix((np.ones(ix.shape[0]), ix, ip), shape=(n, n))
        G = nx.from_scipy_sparse_array(A)
        info = {"library": "networkx", "version": nx.__version__, "device": "cpu",
                "pre_clock_fit": False, "input_home": "host"}

        def fit():
            if t == "pagerank":
                pr = nx.pagerank(G, alpha=p["alpha"], tol=p["tol"], max_iter=p["max_iter"])
                S["o"] = {"scores": np.array([pr[i] for i in range(n)])}
            else:
                # FAIRNESS FIX (2026-10-13 audit): louvain's threshold is passed (it was
                # networkx's default). Results raced before this change were wrong for this
                # lane; apply extra scrutiny.
                comms = (nx.connected_components(G) if t == "components"
                         else nx.community.louvain_communities(G, resolution=p["resolution"],
                                                               threshold=p["threshold"], seed=SEED))
                lab = np.empty(n, dtype=np.int64)
                for c, members in enumerate(comms):
                    lab[list(members)] = c
                S["o"] = {"labels": lab}
        info["config"] = {"pagerank": "nx.pagerank(alpha=0.85, tol=1e-6, max_iter=100)",
                          "components": "nx.connected_components",
                          "louvain": "nx.community.louvain_communities(resolution=1.0, threshold=1e-7, "
                                     "seed=7)"}[t]
        rec = dict(p, __library__="networkx")
        if t == "pagerank":
            rec["l1_stop_threshold"] = n * float(p["tol"])   # nx.pagerank: err < N * tol
        return Runner(info, fit, lambda: S["o"], record=rec)
    import cudf
    import cugraph
    dev, info, sync = _cuml_up({})
    info.update(library="cugraph", version=cugraph.__version__)
    src = np.repeat(np.arange(n), np.diff(ip))
    df = cudf.DataFrame({"src": src.astype(np.int32), "dst": ix.astype(np.int32)})
    G = cugraph.Graph(directed=False)
    # FAIRNESS FIX (2026-10-13 audit): the Graph was built with cuGraph's
    # defaults (renumber=True, store_transposed=False), so every timed call paid
    # an unrenumber of its result and pagerank a transpose of the graph: per-call
    # overhead that dominated the 20,000-node races (networkx on the CPU beat
    # cuGraph on connected components). The vertex ids are already 0..n-1, so
    # renumber=False (cuGraph's documented case for it), and pagerank's graph is
    # stored transposed as cuGraph asks. Still built before the clock. Results
    # raced before this change were wrong for these lanes; apply extra scrutiny.
    G.from_cudf_edgelist(df, source="src", destination="dst", renumber=False,
                         store_transposed=(t == "pagerank"))
    # FAIRNESS FIX (2026-10-13 audit): cuGraph's pagerank tol is the unscaled L1
    # bound (it stopped 20,000x tighter than ours at 20,000 nodes); it is given
    # n x tol, the bound ours and networkx apply. Results raced before this
    # change were wrong for this lane; apply extra scrutiny.
    pr_tol = n * float(p["tol"]) if t == "pagerank" else None
    info["config"] = {"pagerank": "cugraph.pagerank(alpha=%r, tol=%r (= n x %r), max_iter=%r) on "
                                  "Graph(renumber=False, store_transposed=True)"
                                  % (p.get("alpha"), pr_tol, p.get("tol"), p.get("max_iter")),
                      "components": "cugraph.connected_components on Graph(renumber=False)",
                      "louvain": "cugraph.louvain(resolution=1.0, max_level=100, threshold=1e-7) "
                                 "on Graph(renumber=False)"}[t]

    def fit():
        if t == "pagerank":
            S["df"] = cugraph.pagerank(G, alpha=p["alpha"], tol=pr_tol, max_iter=p["max_iter"])
        elif t == "components":
            S["df"] = cugraph.connected_components(G)
        else:
            # FAIRNESS FIX (2026-10-13 audit): threshold passed (it was cuGraph's default)
            S["df"], _ = cugraph.louvain(G, resolution=p["resolution"], max_level=100,
                                         threshold=p["threshold"])

    def outputs():
        df = S["df"].to_pandas().sort_values("vertex")
        col = {"pagerank": "pagerank", "components": "labels", "louvain": "partition"}[t]
        full = np.zeros(n) if t == "pagerank" else np.arange(n, dtype=np.int64) + n
        full[df["vertex"].to_numpy()] = df[col].to_numpy()
        return {"scores": full.astype(np.float64)} if t == "pagerank" else {"labels": full.astype(np.int64)}
    # FAIRNESS FIX (2026-10-13 audit): record the actual called library
    # routine; a bare library-name string cannot establish an opponent.
    graph_call = {"pagerank": cugraph.pagerank, "components": cugraph.connected_components,
                  "louvain": cugraph.louvain}[t]
    rec = dict(p, __library__="cugraph", __implementation__=graph_call)
    if t == "pagerank":
        rec.update(tol=pr_tol, l1_stop_threshold=pr_tol)   # what cuGraph was given
    if t == "louvain":
        rec.pop("seed", None)            # cuGraph louvain takes no seed
    return Runner(info, fit, outputs, None, sync, record=rec)


# ---- ANN ------------------------------------------------------------------

def _pq_dim(d):
    return d if d <= 16 else d // 4


#: FAIRNESS FIX (2026-10-13 audit): the iterative lanes whose arms stop on their own rule
#: (or run a fixed count): every arm's n_iter_ / n_steps_ is recorded in the quality cell so
#: an opponent that ran more iterations than ours, or an early stop, is visible. Results raced
#: before this change were wrong for these lanes; apply extra scrutiny.
_ITER_LANES = frozenset(("tsne", "mds", "minibatch-kmeans", "bisecting-kmeans", "bayesian-gmm",
                         "affinity-prop"))


def _ours_ivf_train_rows(n, nlist):
    """How many rows OUR IVF builds train the coarse k-means on (IDENTICAL):
    every s-th row, s = ivf_trainset_stride(n, n_lists)
    (ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo, IVF_TRAINSET_STRIDE,
    in ours since 3c76388c9, 2026-10-08), transcribed in integers.

    FAIRNESS FIX (2026-10-13 audit): the opponents' quantizer training sets
    were left at their library defaults (cuVS ivf_sq at most 256 rows per list,
    cuVS ivf_flat / ivf_pq kmeans_trainset_fraction 0.5, FAISS
    max_points_per_centroid 256), so on the ivf-sq race (ours at de2b2b739,
    before the stride, trained on all 400,000 rows) cuVS trained on 262,144:
    less work in its clock. Every opponent is now given this count. Results
    raced before this change were wrong for these lanes; apply extra scrutiny."""
    if n <= 1 or nlist < 1:
        return n
    target = min(n // 2, 256 * nlist)          # IVF_TRAIN_ROWS_PER_LIST
    target = max(target, min(n, 39 * nlist), nlist)   # IVF_TRAIN_MIN_ROWS_PER_LIST
    return n // max(1, n // target)


def _build_ann(lane, arm, D):
    np = _np()
    s = LANES[lane]
    t = s["task"]
    p = dict(s["params"])
    X, Q = D["index"], D["queries"]
    n, d = X.shape
    nlist = min(p.get("n_lists", 1024), max(1, n // 4))
    nprobe = min(p.get("n_probes", 32), nlist)
    k = p["n_neighbors"]
    S = {}
    sync = None
    allowed = (np.arange(n) % 2 == 0)
    n_train = _ours_ivf_train_rows(n, nlist)   # FAIRNESS FIX (2026-10-13 audit): every arm's trainset
    if arm in OURS_ARMS:
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        kw = dict(p)
        if "n_lists" in kw:
            kw.update(n_lists=nlist, n_probes=nprobe)
        if t in ("ivf-pq", "ivf-refine", "ivf-filter"):
            kw["pq_dim"] = _pq_dim(d)
        # the refine step and the sample filter are options of an index that
        # may already exist: until the option does, the race is not built yet
        import inspect
        import mojolearn as ml
        refine_fn = getattr(ml, "refine", None) if t == "ivf-refine" else None
        if refine_fn is not None:          # search k x ratio candidates, then the exact re-rank
            kw.pop("refine_ratio", None)
            kw["n_neighbors"] = k * p["refine_ratio"]
        need = {"ivf-refine": (cls, "refine_ratio"), "ivf-filter": (cls.search, "filter")}.get(t)
        if refine_fn is not None:
            need = None
        if need:
            try:
                params = inspect.signature(need[0]).parameters
            except (TypeError, ValueError):
                params = {}
            if need[1] not in params and not any(v.kind == v.VAR_KEYWORD for v in params.values()):
                raise Skipped("SKIPPED: not built yet (mojolearn.%s has no %s= option)" % (name, need[1]))
        info["config"] = "mojolearn.%s(%s)" % (name, kw)

        def fit():
            S["e"] = cls(**kw).fit(X)

        def infer():
            if t == "ivf-filter":
                S["ind"] = S["e"].search(Q, filter=allowed)[1]
            elif refine_fn is not None:
                cand = S["e"].search(Q)[1]
                S["ind"] = refine_fn(X, Q, cand, k)[1]
            else:
                S["ind"] = S["e"].search(Q)[1]
        rec = dict(kw, __library__="mojolearn")
        if t == "ivf-refine":           # k results after a re-rank of k x refine_ratio candidates
            rec.update(n_neighbors=k, refine_ratio=p["refine_ratio"])
        return _with_upload(Runner(info, fit, lambda: {"ind": _arr(S["ind"], np.int64)}, infer,
                                   record=rec), [("index", X)])
    if arm == "faiss-cpu":
        import faiss
        info = {"library": "faiss", "version": faiss.__version__, "device": "cpu",
                "pre_clock_fit": False, "input_home": "host", "omp_threads": faiss.omp_get_max_threads()}

        def fit():
            quant = faiss.IndexFlatL2(d)
            if t == "cagra":
                idx = faiss.IndexHNSWFlat(d, 32)
                idx.hnsw.efConstruction = 128
            elif t in ("ivf-pq", "ivf-refine", "ivf-filter"):
                idx = faiss.IndexIVFPQ(quant, d, nlist, _pq_dim(d), 8)
            elif t == "ivf-sq":
                idx = faiss.IndexIVFScalarQuantizer(quant, d, nlist, faiss.ScalarQuantizer.QT_8bit)
            elif t == "ivf-rabitq":
                idx = faiss.IndexIVFRaBitQ(quant, d, nlist)
            else:
                idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
            if hasattr(idx, "cp"):
                idx.cp.niter = 20
                idx.cp.seed = SEED
                # FAIRNESS FIX (2026-10-13 audit): FAISS's coarse k-means trained on
                # its default sample (256 rows per list, 262,144 rows); it now trains
                # on ours' count (_ours_ivf_train_rows; FAISS samples n_train // nlist
                # rows per list). Results raced before this change were wrong for
                # this lane; apply extra scrutiny.
                idx.cp.max_points_per_centroid = max(1, n_train // nlist)
            if t == "ivf-refine":
                idx = faiss.IndexRefineFlat(idx)
                idx.k_factor = p["refine_ratio"]
            idx.train(X)
            idx.add(X)
            S["idx"], S["q"] = idx, quant

        def infer():
            idx = S["idx"]
            if t == "cagra":
                idx.hnsw.efSearch = 64
                S["ind"] = idx.search(Q, k)[1]
                return
            base = faiss.downcast_index(idx.base_index) if t == "ivf-refine" else idx
            base.nprobe = nprobe
            if t == "ivf-filter":
                sel = faiss.IDSelectorBatch(np.nonzero(allowed)[0].astype(np.int64))
                S["ind"] = idx.search(Q, k, params=faiss.SearchParametersIVF(sel=sel, nprobe=nprobe))[1]
            else:
                S["ind"] = idx.search(Q, k)[1]
        info["config"] = "faiss %s nlist=%d nprobe=%d k=%d" % (t, nlist, nprobe, k)
        if t == "cagra":
            rec = dict(__library__="faiss", index="IndexHNSWFlat", M=32, efConstruction=128,
                       efSearch=64, n_neighbors=k)
        else:
            rec = dict(__library__="faiss", n_lists=nlist, n_probes=nprobe, n_neighbors=k,
                       kmeans_n_iters=20, seed=SEED,
                       max_points_per_centroid=max(1, n_train // nlist))
            if t in ("ivf-pq", "ivf-refine", "ivf-filter"):
                rec.update(pq_dim=_pq_dim(d), pq_bits=8)
            if t == "ivf-refine":
                rec["refine_ratio"] = p["refine_ratio"]
        return Runner(info, fit, lambda: {"ind": np.asarray(S["ind"], dtype=np.int64)}, infer,
                      record=rec)
    import cuvs
    # FAIRNESS FIX (2026-10-13 audit): hipvs-gpu (AMD) is hipVS, AMD's port of cuVS with the
    # same `cuvs` Python API, on the same code path and pinned settings as cuvs-gpu; the upload
    # helper no longer imports cuML, which AMD does not have. Results raced before this change
    # were wrong for these lanes; apply extra scrutiny.
    dev, info, sync = _cuml_up({"index": X, "queries": Q}, library="cuvs")
    info.update(library="hipvs" if arm == "hipvs-gpu" else "cuvs",
                version=getattr(cuvs, "__version__", "unknown"))
    from cuvs.neighbors import ivf_flat, ivf_pq, cagra

    # FAIRNESS FIX (2026-10-13 audit): cuVS ivf_sq ran with kmeans_n_iters and
    # max_train_points_per_cluster at its defaults (its trainset 262,144 rows, its
    # scalar-quantizer ranges trained on that subset: recall 0.773 against ours'
    # 0.935 on taxi), and ivf_pq / ivf_flat with kmeans_trainset_fraction at its
    # default. Every coarse quantizer now trains on ours' count (n_train,
    # _ours_ivf_train_rows: cuVS's own strided fraction n_train / n for ivf_pq /
    # ivf_flat, n_train // nlist rows per list for ivf_sq), with 20 k-means
    # iterations and 256 PQ training rows per code (ours IDN_PQ_TRAIN_ROWS_PER_CODE)
    # pinned explicitly. Results raced before this change were wrong for these
    # lanes; apply extra scrutiny.
    trainset_fraction = n_train / float(n)
    per_list = max(1, n_train // nlist)

    def fit():
        if t == "cagra":
            S["i"] = cagra.build(cagra.IndexParams(graph_degree=p["graph_degree"],
                                                   intermediate_graph_degree=p["intermediate_graph_degree"]),
                                 dev["index"])
        elif t in ("ivf-pq", "ivf-refine"):
            S["i"] = ivf_pq.build(ivf_pq.IndexParams(n_lists=nlist, metric="sqeuclidean",
                                                     kmeans_n_iters=20, pq_dim=_pq_dim(d), pq_bits=8,
                                                     kmeans_trainset_fraction=trainset_fraction,
                                                     max_train_points_per_pq_code=256),
                                  dev["index"])
        elif t == "ivf-sq":
            from cuvs.neighbors import ivf_sq
            S["i"] = ivf_sq.build(ivf_sq.IndexParams(n_lists=nlist, metric="sqeuclidean",
                                                     kmeans_n_iters=20,
                                                     max_train_points_per_cluster=per_list),
                                  dev["index"])
        else:
            S["i"] = ivf_flat.build(ivf_flat.IndexParams(n_lists=nlist, metric="sqeuclidean",
                                                         kmeans_n_iters=20,
                                                         kmeans_trainset_fraction=trainset_fraction),
                                    dev["index"])

    def infer():
        if t == "cagra":
            S["ind"] = cagra.search(cagra.SearchParams(itopk_size=p["itopk_size"]), S["i"],
                                    dev["queries"], k)[1]
        elif t in ("ivf-pq", "ivf-refine"):
            kk = k * p.get("refine_ratio", 1)
            _dd, ind = ivf_pq.search(ivf_pq.SearchParams(n_probes=nprobe), S["i"], dev["queries"], kk)
            if t == "ivf-refine":
                from cuvs.neighbors import refine
                _dd, ind = refine(dev["index"], dev["queries"], ind, k)
            S["ind"] = ind
        elif t == "ivf-sq":
            from cuvs.neighbors import ivf_sq
            S["ind"] = ivf_sq.search(ivf_sq.SearchParams(n_probes=nprobe), S["i"], dev["queries"], k)[1]
        else:
            S["ind"] = ivf_flat.search(ivf_flat.SearchParams(n_probes=nprobe), S["i"], dev["queries"], k)[1]
    info["config"] = "cuvs %s nlist=%d nprobe=%d k=%d" % (t, nlist, nprobe, k)
    rec = dict(__library__="cuvs", n_neighbors=k)
    if t == "cagra":
        rec.update(graph_degree=p["graph_degree"], intermediate_graph_degree=p["intermediate_graph_degree"],
                   itopk_size=p["itopk_size"])
    else:
        rec.update(n_lists=nlist, n_probes=nprobe, kmeans_n_iters=20)
        if t in ("ivf-pq", "ivf-refine"):
            rec.update(pq_dim=_pq_dim(d), pq_bits=8, kmeans_trainset_fraction=trainset_fraction,
                       max_train_points_per_pq_code=256)
        elif t == "ivf-sq":
            rec.update(max_train_points_per_cluster=per_list)
        info["config"] += " coarse trainset %d of %d rows (ours' count)" % (
            n_train if t != "ivf-sq" else min(n, per_list * nlist), n)
    return Runner(info, fit, lambda: {"ind": _arr(S["ind"], np.int64)}, infer, sync, record=rec)


# ---- torch-side helpers (layers, optimizers) -------------------------------

def _torch_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_metric_array(pred):
    """Export completed predictions as host float64, outside inference timing.

    MPS has no float64 storage. Keep prediction selection in its native device
    dtype and move to the CPU before the conductor's metric dtype conversion.
    Training, forward precision and the prediction values are unchanged.
    """
    return pred.detach().cpu().double().numpy()


def _torch_sync(torch, dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def _layer_inputs(lane, D, torch):
    """(module factory, x, extra forward args) on the CPU in float32, seed 7."""
    np = _np()
    s = LANES[lane]
    t = s["task"]
    p = s["params"]
    g = torch.Generator().manual_seed(SEED)
    nn = torch.nn
    extra = ()
    nb = _tool("neural_fixtures").layer_batch(D)   # a smoke shrinks the batch, never the layer
    if t in ("lstm", "gru", "rnn"):
        raw = torch.from_numpy(np.asarray(D["bytes"][:64 * 256], dtype=np.int64)).view(64, 256)
        emb = torch.randn(256, p["input_size"], generator=g)
        x = emb[raw]
        cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[t]
        make = lambda: cls(**p)  # noqa: E731
    elif t == "embedding":
        x = torch.randint(0, p["num_embeddings"], (nb, 512), generator=g, dtype=torch.int64)
        make = lambda: nn.Embedding(**p)  # noqa: E731
    elif t == "layernorm":
        x = torch.randn(256 * nb, p["normalized_shape"], generator=g)
        make = lambda: nn.LayerNorm(**p)  # noqa: E731
    elif t == "moe":
        x = torch.randn(128 * nb, p["hidden_size"], generator=g)
        make = lambda: _MoE(torch, **p)  # noqa: E731
    elif t == "conv1d":
        x = torch.randn(nb, p["in_channels"], 4096, generator=g)
        make = lambda: nn.Conv1d(**p)  # noqa: E731
    elif t == "conv2d":
        x = torch.randn(nb, p["in_channels"], 56, 56, generator=g)
        make = lambda: nn.Conv2d(**p)  # noqa: E731
    elif t in ("maxpool1d", "avgpool1d"):
        x = torch.randn(nb, 64, 4096, generator=g)
        make = lambda: (nn.MaxPool1d if t == "maxpool1d" else nn.AvgPool1d)(**p)  # noqa: E731
    elif t == "batchnorm1d":
        x = torch.randn(nb, p["num_features"], 1024, generator=g)
        make = lambda: nn.BatchNorm1d(**p)  # noqa: E731
    elif t in ("maxpool2d", "avgpool2d"):
        x = torch.randn(nb, 64, 112, 112, generator=g)
        make = lambda: (nn.MaxPool2d if t == "maxpool2d" else nn.AvgPool2d)(**p)  # noqa: E731
    elif t == "batchnorm2d":
        x = torch.randn(nb, p["num_features"], 56, 56, generator=g)
        make = lambda: nn.BatchNorm2d(**p)  # noqa: E731
    elif t == "dropout2d":
        x = torch.randn(nb, 64, 56, 56, generator=g)
        make = lambda: nn.Dropout2d(**p)  # noqa: E731
    elif t in ("gap", "gmp"):
        x = torch.randn(nb, 512, 7, 7, generator=g)
        make = lambda: (nn.AdaptiveAvgPool2d if t == "gap" else nn.AdaptiveMaxPool2d)(1)  # noqa: E731
    elif t == "resnet":
        x = torch.randn(nb, p["inplanes"], 56, 56, generator=g)
        make = lambda: _basic_block(torch, p["inplanes"], p["planes"])  # noqa: E731
    elif t in ("gcn", "sage"):
        x_np, edges = _graph_layer_inputs(D)
        x = torch.from_numpy(x_np)
        extra = (torch.from_numpy(edges),)
        d = x.shape[1]
        if t == "gcn":
            from torch_geometric.nn import GCNConv
            # FAIRNESS FIX (2026-10-13 audit): ours' GCNConv caches the normalized graph
            # between calls (python/mojolearn/_expansion_cnn.py GCNConv.forward `_gcache`,
            # keyed by the edge_index content, filled in the warm-up) while this arm was
            # GCNConv(cached=False), rebuilding gcn_norm (self loops, degrees, D^-1/2 A D^-1/2)
            # on every call. PyG's own cache (cached=True: the normalized graph is computed on
            # the first call and reused) now does the same per-call work on the same static
            # kNN graph. GCN_TORCH_KW is torch-only (ours has no `cached` argument; it always
            # caches), declared in the lane notes. Results raced before this change were wrong
            # for this lane; apply extra scrutiny.
            make = lambda: GCNConv(d, **dict(p, **GCN_TORCH_KW))  # noqa: E731
        else:
            from torch_geometric.nn import SAGEConv
            make = lambda: SAGEConv(d, **p)  # noqa: E731
    else:
        raise ValueError(t)
    torch.manual_seed(SEED)
    return make, x, extra


def _graph_layer_inputs(D):
    """The graph layer lanes' x and (2, E) edge index, numpy only (the dataset's X and CSR;
    no torch draw, so these are not fixture files: both arms derive them from the data)."""
    np = _np()
    x = np.ascontiguousarray(np.asarray(D["X"], dtype=np.float32))
    ip = np.asarray(D["indptr"])
    src = np.repeat(np.arange(ip.shape[0] - 1), np.diff(ip))
    return x, np.ascontiguousarray(np.stack([np.asarray(D["indices"]), src]).astype(np.int64))


def _load_graph_weights(layer, task, state):
    """FAIRNESS FIX (2026-10-13 audit): torch's seeded PyG weights (the neural fixture's state
    dict) into ours' GCNConv / SAGEConv public weight attributes, so both arms start from the
    same weights and ours' y is comparable with torch's. True when every tensor was present
    with ours' shape; otherwise nothing is changed and False (ours keeps its own init). Results
    raced before this change were wrong for this lane; apply extra scrutiny."""
    np = _np()
    if task == "gcn":
        plan = [(layer, "weight_", "lin.weight"), (layer, "bias_", "bias")]
    elif task == "sage":
        lin = getattr(layer, "lin_l", None)
        plan = [(lin, "weight_", "lin_l.weight"), (lin, "bias_", "lin_l.bias"),
                (layer, "weight_r_", "lin_r.weight")]
        if getattr(layer, "project", False):
            return False                   # the projection's weights are not in this map
    else:
        return False
    vals = []
    for obj, attr, key in plan:
        cur = getattr(obj, attr, None) if obj is not None else None
        if cur is None or key not in state:
            return False
        v = np.ascontiguousarray(np.asarray(state[key]), dtype=np.float32)
        if tuple(v.shape) != tuple(np.shape(cur)):
            return False
        vals.append((obj, attr, v))
    for obj, attr, v in vals:
        setattr(obj, attr, v)
    return True


#: the neural fixture directory this worker reads (set by worker() from --neural-fixtures,
#: default tools/neural_fixtures.default_dir(--data)); see tools/neural_fixtures.py
_NEURAL_FIXTURES = None


def _neural_fixture_drift_check(lane, D, x_cpu, state, torch):
    """Torch arms: assert the tensors torch just built equal the fixture bytes our arm reads
    (x, every state tensor, dy re-drawn from Generator(SEED + 1) at the fixture's shape), so a
    drift in torch's generator or init is caught, not silently compared. No fixture set on
    this box: the torch arm runs as before and says so in its info."""
    np = _np()
    NF = _tool("neural_fixtures")
    s = LANES[lane]
    try:
        fx = NF.load_for_lane(_NEURAL_FIXTURES or os.environ.get(NF.ENV_DIR), lane, s, _DATASET or os.environ.get("MOJOLEARN_BOARD_DATASET"), D)
    except NF.FixturesMissing as exc:
        return {"status": "absent", "detail": str(exc)[:300]}
    bad = []
    if fx["x"] is not None:
        xt = x_cpu.detach().cpu().numpy()
        if xt.dtype != fx["x"].dtype or xt.shape != fx["x"].shape or not np.array_equal(xt, fx["x"]):
            bad.append("x")
    if sorted(state) != sorted(fx["state"]):
        bad.append("state keys %s vs %s" % (sorted(state), sorted(fx["state"])))
    else:
        for k, v in state.items():
            f = fx["state"][k]
            if v.dtype != f.dtype or v.shape != f.shape or not np.array_equal(v, f):
                bad.append("state." + k)
    if fx["dy"] is not None:
        g2 = torch.Generator().manual_seed(SEED + 1)
        dyt = torch.randn(tuple(fx["dy"].shape), generator=g2).numpy()
        if not np.array_equal(dyt, fx["dy"]):
            bad.append("dy")
    if bad:
        raise NF.FixtureMismatch(
            "neural fixture drift: torch %s built tensors that differ from the fixtures our arm "
            "reads (generated with torch %s): %s; regenerate with tools/neural_fixtures.py generate "
            "and re-check before comparing" % (torch.__version__, fx["torch_version"], ", ".join(bad)))
    return dict(NF.provenance(fx), status="match")


def _MoE(torch, hidden_size, intermediate_size, num_experts, top_k, norm_topk_prob):
    ffn_size = intermediate_size
    nn = torch.nn
    F = torch.nn.functional

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(hidden_size, ffn_size, bias=False)
            self.w2 = nn.Linear(ffn_size, hidden_size, bias=False)
            self.w3 = nn.Linear(hidden_size, ffn_size, bias=False)

        def forward(self, h):
            return self.w2(F.silu(self.w1(h)) * self.w3(h))

    class MoE(nn.Module):
        """HF MixtralSparseMoeBlock.forward, transcribed (no jitter)."""

        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden_size, num_experts, bias=False)
            self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])

        def forward(self, h):
            logits = self.gate(h)
            w = F.softmax(logits, dim=1, dtype=torch.float)
            w, sel = torch.topk(w, top_k, dim=-1)
            if norm_topk_prob:
                w = w / w.sum(dim=-1, keepdim=True)
            w = w.to(h.dtype)
            out = torch.zeros_like(h)
            mask = F.one_hot(sel, num_classes=num_experts).permute(2, 1, 0)
            for e in range(num_experts):
                idx, top = torch.where(mask[e])
                out.index_add_(0, top, self.experts[e](h[top]) * w[top, idx, None])
            return out
    return MoE()


def _hf_moe_block(torch, p, state, arm):
    """FAIRNESS FIX (2026-10-13 audit): transformers' own MixtralSparseMoeBlock (softmax router,
    top-k, renormalized weights, SwiGLU experts) in place of the harness's transcription of it
    (_MoE, kept only as the seeded weight source the neural fixtures were generated from). The
    fixture's per-expert weights (gate.weight, experts.<e>.w1/w2/w3.weight) are mapped into the
    installed version's layout (per-expert modules, or the fused gate_up_proj / down_proj
    experts), loaded strictly and read back bit for bit; any layout this cannot map refuses the
    arm by name. Unverified at runtime against transformers 5.17.0. Results raced before this
    change were wrong for this lane; apply extra scrutiny."""
    np = _np()
    try:
        import transformers
        from transformers import MixtralConfig
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("REFUSED: %s: transformers' MixtralSparseMoeBlock does not import (%r); the "
                           "harness's MoE transcription is not an opponent" % (arm, exc))
    E, D, F, K = (int(p["num_experts"]), int(p["hidden_size"]), int(p["intermediate_size"]),
                  int(p["top_k"]))
    if not p.get("norm_topk_prob", True):
        raise RuntimeError("REFUSED: %s: Mixtral renormalizes its top-k weights; this lane says not" % arm)
    cfg = MixtralConfig(hidden_size=D, intermediate_size=F, num_local_experts=E,
                        num_experts_per_tok=K, hidden_act="silu", router_jitter_noise=0.0,
                        num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
                        vocab_size=8)
    torch.manual_seed(SEED)
    blk = MixtralSparseMoeBlock(cfg)
    blk.eval()
    have = {k: tuple(v.shape) for k, v in blk.state_dict().items()}
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    w1 = [state["experts.%d.w1.weight" % e] for e in range(E)]
    w2 = [state["experts.%d.w2.weight" % e] for e in range(E)]
    w3 = [state["experts.%d.w3.weight" % e] for e in range(E)]
    sd = {}
    if "experts.0.w1.weight" in have:                    # the per-expert layout
        sd["gate.weight"] = t(state["gate.weight"])
        for e in range(E):
            for k, w in (("w1", w1), ("w2", w2), ("w3", w3)):
                sd["experts.%d.%s.weight" % (e, k)] = t(w[e])
    else:                                                # the fused experts layout
        router = [k for k, s_ in have.items() if not k.startswith("experts.") and s_ == (E, D)]
        gu = [k for k in have if k.startswith("experts.") and "gate_up" in k]
        dn = [k for k in have if k.startswith("experts.") and "down" in k]
        if len(router) != 1 or len(gu) != 1 or len(dn) != 1 or len(have) != 3:
            raise RuntimeError("REFUSED: %s: transformers %s MixtralSparseMoeBlock state %r is not a "
                               "layout this harness maps" % (arm, transformers.__version__, have))
        gate_up = np.stack([np.concatenate([w1[e], w3[e]]) for e in range(E)])   # (E, 2F, D)
        down = np.stack(w2)                                                      # (E, D, F)
        if have[gu[0]] == (E, D, 2 * F):
            gate_up = np.ascontiguousarray(gate_up.transpose(0, 2, 1))
        if have[dn[0]] == (E, F, D):
            down = np.ascontiguousarray(down.transpose(0, 2, 1))
        sd = {router[0]: t(state["gate.weight"]), gu[0]: t(gate_up), dn[0]: t(down)}
    for k, v in sd.items():
        if have.get(k) != tuple(v.shape):
            raise RuntimeError("REFUSED: %s: %s is %s in transformers, the fixture maps %s"
                               % (arm, k, have.get(k), tuple(v.shape)))
    blk.load_state_dict(sd, strict=True)
    got = blk.state_dict()
    for k, v in sd.items():
        if not torch.equal(got[k].detach().cpu(), v):
            raise RuntimeError("REFUSED: %s: %s did not load the fixture's float32 bits" % (arm, k))
    nn = torch.nn

    class MoE(nn.Module):
        """transformers MixtralSparseMoeBlock over the (tokens, hidden) input."""

        def __init__(self):
            super().__init__()
            self.blk = blk

        def forward(self, h):
            out = self.blk(h.reshape(1, h.shape[0], h.shape[1]))
            out = out[0] if isinstance(out, (tuple, list)) else out
            return out.reshape(h.shape)
    mod = MoE()
    mod.library = "transformers %s MixtralSparseMoeBlock" % transformers.__version__
    return mod


def _library_layer(torch, s, make, state, arm):
    """FAIRNESS FIX (2026-10-13 audit): the module a torch layer arm races: the library's own class
    where the harness used its own transcription (resnet-block: torchvision's BasicBlock; moe:
    transformers' MixtralSparseMoeBlock), else the lane's torch module. Returns (module, loaded):
    loaded True when the weights are already in it. A missing library refuses by name; the
    harness's own module is never raced instead. Results raced before this change were wrong for
    this lane; apply extra scrutiny."""
    t, p = s["task"], s["params"]
    if t == "resnet":
        try:
            from torchvision.models.resnet import BasicBlock
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("REFUSED: %s: torchvision does not import (%r); the harness's BasicBlock "
                               "is not an opponent" % (arm, exc))
        mod = BasicBlock(p["inplanes"], p["planes"])       # stride 1, no downsample: the lane's block
        import torchvision
        mod.library = "torchvision %s BasicBlock" % torchvision.__version__
        return mod, False
    if t == "moe":
        return _hf_moe_block(torch, p, state, arm), True
    return make(), False


def _basic_block(torch, inplanes, planes):
    nn = torch.nn

    class BasicBlock(nn.Module):
        """torchvision.models.resnet.BasicBlock (stride 1, no downsample)."""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + x)
    return BasicBlock()


def _build_layer(lane, arm, D):
    np = _np()
    s = LANES[lane]
    S = {}
    if arm in OURS_ARMS:
        # Our arm imports torch NOWHERE (Andrew 2026-10-09). x, the initial state (torch's
        # seeded default init) and the backward seed dy are the fixture files
        # tools/neural_fixtures.py generated with the very builders the torch arm runs below
        # (_layer_inputs, make(), Generator(SEED + 1)), sha256-verified on load. A missing set
        # refuses (not_ready, infrastructure); there is no torch fallback here
        NF = _tool("neural_fixtures")
        fx = NF.load_for_lane(_NEURAL_FIXTURES or os.environ.get(NF.ENV_DIR), lane, s, _DATASET or os.environ.get("MOJOLEARN_BOARD_DATASET"), D)
        state = fx["state"]
        if s["task"] in ("gcn", "sage"):      # x and edges are the dataset's (numpy, no draw)
            x_np, edges = _graph_layer_inputs(D)
            if int(fx["entry"].get("in_channels", -1)) != int(x_np.shape[1]):
                raise NF.FixtureMismatch("neural fixture %s: in_channels %s, data has %d"
                                         % (lane, fx["entry"].get("in_channels"), x_np.shape[1]))
            x_cpu, extra_cpu = x_np, (edges,)
        else:
            x_cpu, extra_cpu = fx["x"], ()
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        info["neural_fixtures"] = NF.provenance(fx)
        kw = dict(s["params"])
        if name == "layer_norm_forward":      # the functional form: F.layer_norm and its backward
            import mojolearn as ml
            bwd = getattr(ml, "layer_norm_backward", None)
            x = x_cpu
            w_, b_ = state["weight"], state["bias"]
            info["config"] = "mojolearn.layer_norm_forward / layer_norm_backward, eps %g" % kw["eps"]
            info["weights_loaded"] = info["output_comparable"] = True
            dyh = {}
            # lane gap-neural-io: x (and dy, on the first fit) on the device before the clock,
            # y and dx left there, as the torch arm's tensors (MOJOLEARN_SEQ_DEVICE_IO_OFF=1 keeps
            # host arrays: the before arm). y is read back in outputs(), outside the clock.
            from mojolearn._x_sequence_device import to_device as seq_to_dev
            x = seq_to_dev(x)
            on_dev = getattr(x, "_device_tensor", False)
            if on_dev:
                info["input_home"] = "device"

            def fit():
                y = cls(x, None, w_, b_, kw["eps"])
                if "dy" not in dyh:
                    dyh["dy"] = NF.take_dy(fx, tuple(y.shape))
                    if on_dev:
                        dyh["dy"] = seq_to_dev(dyh["dy"])
                if bwd is None:
                    raise RuntimeError("CONTRACT: mojolearn has no layer_norm_backward")
                bwd(dyh["dy"], x, None, w_, b_, kw["eps"])

            def infer():
                S["y"] = cls(x, None, w_, b_, kw["eps"])
            return _with_upload(Runner(info, fit, lambda: {"y": _arr(S["y"], np.float32)}, infer,
                                       record=dict(kw, __library__="mojolearn", __implementation__=cls,
                                                   input_arrays=_input_array_records([("x", x_cpu)] + sorted(state.items())))), [("x", x)])
        if s["task"] == "embedding":          # ours takes the table (weight=), torch's seeded init
            ids = x_cpu                       # (from the fixture file)
            layer = cls(weight=state["weight"], **kw)
            info["config"] = "mojolearn.%s(%s, weight=torch's init via the neural fixture)" % (name, kw)
            info["weights_loaded"] = info["output_comparable"] = True
            dyh = {}
            # lane gap-neural-io: y, dy and the (V, d) gradient stay on the device, as the torch
            # arm's tensors (dy goes up on the first fit, as torch's `.to(dev)`); y is read back in
            # outputs(), outside the clock. A binding without the device entries, or
            # MOJOLEARN_SEQ_DEVICE_IO_OFF=1 (the before arm), keeps host arrays.
            import os as _os
            dev_io = (getattr(layer, "_device_io", False)
                      and _os.environ.get("MOJOLEARN_SEQ_DEVICE_IO_OFF", "") != "1"
                      and _os.environ.get("MOJOLEARN_IDN_ALL_OFF", "") != "1")
            if dev_io:
                try:
                    layer._dev_module()
                except RuntimeError:
                    dev_io = False

            def fit():
                y = layer.forward(ids, device=True) if dev_io else layer.forward(ids)
                if "dy" not in dyh:
                    dyh["dy"] = NF.take_dy(fx, tuple(y.shape) if dev_io else tuple(np.shape(y)))
                    if dev_io:
                        dyh["dy"] = layer.to_device(dyh["dy"])
                layer.backward(ids, dyh["dy"])

            def infer():
                S["y"] = layer.forward(ids, device=True) if dev_io else layer.forward(ids)
            # FAIRNESS FIX (2026-10-13 audit): this early return bypassed the
            # common layer input/implementation receipt, refusing embedding
            # despite both arms loading the same retained table and token IDs.
            return _with_upload(Runner(info, fit, lambda: {"y": _arr(S["y"], np.float32)}, infer,
                                       record=dict(kw, __library__="mojolearn", __implementation__=cls,
                                                   input_arrays=_input_array_records([("x", ids)] + sorted(state.items())))), [("ids", ids)])
        if s["task"] in ("gcn", "sage"):
            kw["in_channels"] = x_cpu.shape[1]
        import inspect
        try:
            if "random_state" in inspect.signature(cls).parameters:
                kw["random_state"] = SEED
        except (TypeError, ValueError):
            pass
        layer = cls(**kw)
        _neural_ab_layer_mode(layer, s)
        info["config"] = "mojolearn.%s(%s)" % (name, kw)
        info["weights_loaded"] = False
        if s["task"] == "moe":                # HF's fused layout from the per-expert transcription
            E = kw["num_experts"]
            layer.load_state_dict({
                "router": state["gate.weight"],
                "gate_up_proj": np.stack([np.concatenate([state["experts.%d.w1.weight" % e],
                                                          state["experts.%d.w3.weight" % e]])
                                          for e in range(E)]),
                "down_proj": np.stack([state["experts.%d.w2.weight" % e] for e in range(E)])})
            info["weights_loaded"] = True
        elif hasattr(layer, "load_state_dict"):
            layer.load_state_dict(state)
            info["weights_loaded"] = True
        elif hasattr(layer, "set_weights") and "weight" in state:
            w, b_ = state["weight"], state.get("bias")
            try:
                layer.set_weights(w, b_)
            except Exception:  # noqa: BLE001  (Conv1d keeps (O, C, 1, k))
                layer.set_weights(w.reshape(w.shape[0], w.shape[1], 1, -1), b_)
            info["weights_loaded"] = True
        elif s["task"] in ("gcn", "sage"):
            # FAIRNESS FIX (2026-10-13 audit): GCNConv/SAGEConv have no load_state_dict, so
            # ours ran on its own random weights and its output was never checked against the
            # opponent's. Torch's seeded initial weights (the fixture's PyG state dict) now go
            # into the layer's public weight attributes (GCN: weight_ = lin.weight, bias_ =
            # bias; SAGE: lin_l.weight_/bias_ = lin_l.*, weight_r_ = lin_r.weight), shapes
            # checked; anything else leaves ours' own init (no comparison, as before). Results
            # raced before this change were wrong for this lane; apply extra scrutiny.
            info["weights_loaded"] = _load_graph_weights(layer, s["task"], state)
        x = x_cpu
        extra = tuple(extra_cpu)
        # lane idn-cnn-resident: a layer that takes device tensors (conv, pool, BasicBlock, GCNConv,
        # SAGEConv) gets x and dy on the device before the clock and keeps y and dx there, as the
        # torch arm does (x_cpu.to(dev), dy.to(dev) on the first fit, y and the gradients left on
        # the device, a synchronize before the clock stops: ours' entries wait before they return).
        # y is read back in outputs(), outside the clock, as torch's .cpu() is.
        # MOJOLEARN_XCNN_DEVICE_IO_OFF=1 keeps host arrays (the before arm).
        # lane layernorm-idn: a layer with its own `to_device` (the sequence LayerNorm, whose
        # resident tensors live on the sequence binding, not the CNN one) uploads through it, so
        # the lane races what a user who puts x on the device gets (the `layer_norm_dev` entry);
        # MOJOLEARN_LN_CLASS_DEVICE_IO_OFF=1 (or MOJOLEARN_SEQ_DEVICE_IO_OFF=1) is its before arm.
        to_dev = None
        if getattr(layer, "_seq_device_io", False):
            # lane neural-io-2: a sequence-binding layer (MoEBlock) takes x as a resident
            # SequenceDeviceTensor and returns y as one (read back in outputs(), outside the
            # clock). MOJOLEARN_SEQ_DEVICE_IO_OFF=1, or a binding built with
            # -D MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF, keeps the host-array entry (the before arm).
            to_dev = layer.to_device
            x = to_dev(x)
            if getattr(x, "_device_tensor", False):
                info["input_home"] = "device"
        elif getattr(layer, "_device_io", False):
            # lane layernorm-idn: a layer with its own to_device (LayerNorm) uploads through it
            to_dev = getattr(layer, "to_device", None)
            if to_dev is None:
                from mojolearn._expansion_cnn import to_device as to_dev
            x = to_dev(x)
            if getattr(x, "_device_tensor", False):
                info["input_home"] = "device"
        fwd = getattr(layer, "forward", None) or layer
        import inspect
        try:
            takes_x = "x" in inspect.signature(layer.backward).parameters
        except (AttributeError, TypeError, ValueError):
            takes_x = False
        dy = None

        def fit():
            nonlocal dy
            y = fwd(x, *extra)
            y = y[0] if isinstance(y, tuple) else y
            if s.get("forward_only"):
                return
            if dy is None:
                dy = NF.take_dy(fx, tuple(y.shape if to_dev else np.shape(y)))
                if to_dev:
                    dy = to_dev(dy)
            if not hasattr(layer, "backward"):
                raise RuntimeError("CONTRACT: mojolearn.%s has no backward(dy); the training "
                                   "column needs it" % name)
            layer.backward(dy, x) if takes_x else layer.backward(dy)

        def infer():
            y = fwd(x, *extra)
            S["y"] = y[0] if isinstance(y, tuple) else y
        # a layer with no trained parameters (pooling, BatchNorm's default affine, LayerNorm)
        # starts equal to torch's; one with random weights is compared only when they loaded
        comparable = info["weights_loaded"] or not any(
            k.endswith("weight") and not k.startswith(("bn", "norm")) and s["task"] not in (
                "batchnorm1d", "batchnorm2d", "layernorm") for k in state)
        info["output_comparable"] = bool(comparable)
        # Lack of shared Torch weights prevents opponent comparison, not an
        # own-host check: both own routes use the same seeded constructor.
        # Keep the distinct key so quality() cannot compare unrelated weights.
        output_key = "y" if comparable else "own_y"
        out = lambda: {output_key: _arr(
            S["y"].numpy() if getattr(S["y"], "_device_tensor", False) else S["y"], np.float32)}
        rec = dict(kw, __library__="mojolearn", __implementation__=cls,
                   input_arrays=_input_array_records([("x", x_cpu)] +
                       [("extra%d" % i, a) for i, a in enumerate(extra_cpu)] + sorted(state.items())))
        # the host x and extra forward arrays (the probe reads input_home=device and stands down
        # when x went to the device before the clock)
        return _with_upload(Runner(info, fit, out, infer, record=rec),
                            [("x", x_cpu)] + [("extra%d" % i, e) for i, e in enumerate(extra_cpu)])
    import torch
    make, x_cpu, extra_cpu = _layer_inputs(lane, D, torch)
    ref = make()
    _neural_ab_layer_mode(ref, s)
    state = {k: v.detach().cpu().numpy() for k, v in ref.state_dict().items()}
    # untimed, before the generator below: torch's tensors must equal our arm's fixture bytes
    drift = _neural_fixture_drift_check(lane, D, x_cpu, state, torch)
    g = torch.Generator().manual_seed(SEED + 1)
    setting = arm[len("torch-"):]
    mode, prec = setting.split("-")
    dev = _torch_device(torch)
    info = {"library": "torch", "version": torch.__version__, "device": "gpu" if dev != "cpu" else "cpu",
            "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
            "pre_clock_fit": False, "input_home": "device", "setting": setting,
            "neural_fixtures": drift}
    if s["task"] in ("gcn", "sage"):
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
    torch.backends.cudnn.allow_tf32 = prec == "tf32"
    # FAIRNESS FIX (2026-10-13 audit): resnet-block and moe raced the harness's own BasicBlock and
    # MoE transcription; _library_layer races torchvision's and transformers' classes (the harness
    # modules stay the seeded weight source of the fixtures, `ref` above). Results raced before
    # this change were wrong for this lane; apply extra scrutiny.
    mod, loaded = _library_layer(torch, s, make, state, arm)
    _neural_ab_layer_mode(mod, s)
    if not loaded:
        mod.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    if getattr(mod, "library", None):
        info["twin"] = mod.library
    mod = mod.to(dev)
    x = x_cpu.to(dev)
    # FAIRNESS FIX (2026-10-13 audit): gcn and sage were excluded here, so torch's backward
    # computed only the weight gradients while ours' GCNConv/SAGEConv.backward also computes
    # grad_x (GCN: one more GEMM; SAGE: a GEMM, two SpMMs and an add), ours doing more work.
    # x now requires grad on gcn and sage too, so both arms compute dx. Results raced before
    # this change were wrong for this lane; apply extra scrutiny.
    if s["task"] != "embedding" and not s.get("forward_only"):
        x.requires_grad_(True)             # backward computes dx, as ours' backward returns it
                                           # (embedding: integer ids, the gradient is the table's)
    extra = tuple(e.to(dev) for e in extra_cpu)
    if s["task"] == "gcn" and GCN_TORCH_KW.get("cached"):
        # FAIRNESS FIX (2026-10-13 audit): GCNConv(cached=True) fills its cache on the first
        # call; filled inside a compiled forward, the next call would fail dynamo's guard on
        # `_cached_edge_index` and recompile inside a timed round. One untimed eager forward
        # here fills it before torch.compile first runs (ours fills its own cache in the
        # untimed warm-up). Results raced before this change were wrong for this lane; apply
        # extra scrutiny.
        with torch.no_grad():
            mod(x, *extra)
        _torch_sync(torch, dev)
        info["gcn_cache"] = "PyG GCNConv(cached=True), filled by one untimed forward before the clock"
    run = torch.compile(mod) if mode == "compile" else mod
    dt = torch.bfloat16 if prec == "bf16" else None
    info["config"] = "%s on %s, %s, %s%s" % (type(mod).__name__, dev, mode, prec,
                                            ", cached=True" if s["task"] == "gcn" else "")
    dyh = {}

    def forward():
        if dt is not None:
            with torch.autocast(device_type=dev if dev != "mps" else "mps", dtype=dt):
                y = run(x, *extra)
        else:
            y = run(x, *extra)
        return y[0] if isinstance(y, tuple) else y

    def fit():
        if s.get("forward_only"):
            with torch.no_grad():
                forward()
            _torch_sync(torch, dev)
            return
        y = forward()
        if "dy" not in dyh:
            dyh["dy"] = torch.randn(tuple(y.shape), generator=g).to(dev)
        y.backward(dyh["dy"].to(y.dtype))
        mod.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        _torch_sync(torch, dev)

    def infer():
        with torch.no_grad():
            S["y"] = forward()
        _torch_sync(torch, dev)

    def outputs():
        # FAIRNESS FIX (2026-10-13 audit): dropout outputs were discarded
        # because RNG masks differ. Retain them for task-quality checks;
        # random masks are never required to match the other library's bits.
        return {"y": S["y"].float().cpu().numpy()}
    # FAIRNESS FIX (2026-10-13 audit): the layer record named torch but lost
    # the actual constructed library class. For MoE the harness only reshapes
    # tensors around transformers' block; witness that block, not its wrapper.
    implementation = type(mod.blk) if s["task"] == "moe" else type(mod)
    rec = dict(s["params"], __library__="torch", __implementation__=implementation,
               seed=SEED, input_arrays=_input_array_records([("x", x_cpu.numpy())] +
                   [("extra%d" % i, a.numpy()) for i, a in enumerate(extra_cpu)] + sorted(state.items())))
    if s["task"] in ("gcn", "sage"):
        rec["in_channels"] = x_cpu.shape[1]
    return Runner(info, fit, outputs, infer, record=rec)


def _opt_state_snapshot(torch, opt, p):
    """The optimizer's per-parameter state right after construction (empty for the lazily
    initialized optimizers, e.g. Adagrad's `sum` and `step` for the eager ones)."""
    import copy
    return {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in opt.state.get(p, {}).items()}


def _input_array_records(arrays):
    # FAIRNESS FIX (2026-10-13 audit): synthetic workers construct their
    # real inputs after dataset preparation. Empty prepared metadata was not
    # a measured shape. Witness the full actual arrays before any clock.
    np = _np()
    return [dict(name=name, shape=list(a.shape), dtype=str(a.dtype),
                 sha256=hashlib.sha256(a.tobytes()).hexdigest())
            for name, value in arrays for a in [np.ascontiguousarray(value)]]


def _opt_state_reset(torch, opt, p, snap):
    """Put the optimizer back to its constructed state IN PLACE (the same optimizer object, the
    same state dict, so a compiled step keeps its guards): lazily initialized state is dropped,
    eagerly initialized tensors are copied back."""
    import copy
    if not snap:
        opt.state.pop(p, None)
        return
    st = opt.state[p]
    for k in list(st):
        if k not in snap:
            del st[k]
    with torch.no_grad():
        for k, v in snap.items():
            if torch.is_tensor(v) and torch.is_tensor(st.get(k)):
                st[k].copy_(v)
            else:
                st[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)


def _build_optim(lane, arm, D):
    np = _np()
    s = LANES[lane]
    P, STEPS = (1 << 16 if D.get("_cap") else 1 << 24), 10
    rng = np.random.default_rng(SEED)
    p0 = rng.standard_normal(P).astype(np.float32)
    grads = [rng.standard_normal(P).astype(np.float32) * 0.01 for _ in range(STEPS)]
    input_arrays = _input_array_records([("p0", p0)] + [("grad%d" % i, g) for i, g in enumerate(grads)])
    S = {}
    hyper = dict(s["params"])
    if arm in OURS_ARMS:
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        info["config"] = "mojolearn.%s([p], %s).step(g) x %d" % (name, hyper, STEPS)

        # lane gap-neural-io: the gradients on the device before the clock and the parameter
        # uploaded inside it and left there, as the torch arm (`p0.to(dev)` in fit, `g_dev`
        # outside it); read back in the outputs, outside the clock. The optimizers without the
        # device step, or MOJOLEARN_SEQ_DEVICE_IO_OFF=1 (the before arm), keep host arrays.
        to_dev = None
        grads_run = grads
        resident = False
        if getattr(cls, "__module__", "").endswith("_x_sequence_optim"):
            from mojolearn._x_sequence_device import to_device as to_dev
            g0 = to_dev(grads[0])
            if getattr(g0, "_device_tensor", False):
                grads_run = [g0] + [to_dev(gr) for gr in grads[1:]]
                info["input_home"] = "device"
            else:
                to_dev = None
        elif getattr(cls, "__module__", "").endswith("_training_impl"):
            # FAIRNESS FIX (2026-10-13 audit): mojolearn.training's SGD / Adam / AdamW ran on host
            # arrays (`opt.step([g])`), so the 16,777,216-float parameter went up AND down every
            # step inside ours' clock while torch's parameter and gradients stayed on the device.
            # Ours' public resident path is used now where the build has it: the gradients go up
            # before the clock (as torch's g_dev), the parameter goes up inside it
            # (`params_to_device()`, as torch's p0.to(dev)), every step is `step_device([g])`, and
            # the parameter comes back in the outputs, outside the clock (`params_to_host()`).
            # A build without it (params_resident_available() False) keeps the host path, named
            # in info. Results raced before this change were wrong for this lane; apply extra
            # scrutiny.
            try:
                ok = bool(cls([p0[:1].copy()], **hyper).params_resident_available())
                if ok:
                    from mojolearn._training_impl import train_to_device
                    grads_run = [train_to_device(gr) for gr in grads]
            except Exception as exc:  # noqa: BLE001  (a CPU host twin: no device arrays)
                ok, grads_run = False, grads
                info["resident_error"] = repr(exc)[:300]
            if ok:
                resident = True
                info["input_home"] = "device"
                info["resident_path"] = ("params_to_device + step_device(device grads) + "
                                         "params_to_host (outside the clock)")
            else:
                info["resident_path"] = "unavailable on this build: host arrays every step"

        def fit():
            p = to_dev(p0) if to_dev else p0.copy()
            opt = cls([p], **hyper)
            if resident:
                opt.params_to_device()
                for gr in grads_run:
                    opt.step_device([gr])
                S["opt"] = opt
            else:
                for gr in grads_run:
                    opt.step([gr])
                S["opt"] = None
            S["p"] = p

        def outputs():
            if S.get("opt") is not None:
                S["opt"].params_to_host()          # outside the clock (outputs are read after it)
            return {"param": np.asarray(S["p"], dtype=np.float32)}
        rec = dict(hyper, __library__="mojolearn", __implementation__=cls, input_arrays=input_arrays)
        return _with_upload(Runner(info, fit, outputs, record=rec),
                            [("p0", p0)] + [("grad%d" % i, gr) for i, gr in enumerate(grads)])
    import torch
    dev = _torch_device(torch)
    mode = arm.split("-")[1]
    info = {"library": "torch", "version": torch.__version__, "device": "gpu" if dev != "cpu" else "cpu",
            "pre_clock_fit": False, "input_home": "device", "setting": arm[len("torch-"):]}
    ocls = getattr(torch.optim, s["torch_opt"], None)
    if ocls is None:
        raise RuntimeError("REFUSED: torch %s has no torch.optim.%s" % (torch.__version__, s["torch_opt"]))
    import inspect
    try:
        sig = inspect.signature(ocls).parameters
    except (TypeError, ValueError):
        sig = {}
    # FAIRNESS FIX (2026-10-13 audit): the implementation switch was left at torch's default; it
    # is pinned per arm now: eager and compile run the foreach implementation (foreach=True,
    # torch's default on GPUs, stated), and torch-fused-fp32 runs torch's fused kernel
    # (fused=True), torch's fastest optimizer path, refused by name where this torch has none for
    # the class or the device. Results raced before this change were wrong for this lane; apply
    # extra scrutiny.
    impl = {}
    if mode == "fused":
        if "fused" not in sig:
            raise RuntimeError("REFUSED: %s: torch.optim.%s takes no fused= in torch %s"
                               % (arm, s["torch_opt"], torch.__version__))
        impl = {"fused": True}
    elif "foreach" in sig:
        impl = {"foreach": True}
    g_dev = [torch.from_numpy(gr).to(dev) for gr in grads]
    p0_host = torch.from_numpy(p0)
    # FAIRNESS FIX (2026-10-13 audit): every fit built a new optimizer and a new
    # torch.compile(opt.step), so the compile arm recompiled (dynamo's guards on the new instance
    # fail) inside every timed round. The parameter, the optimizer and the compiled step are
    # built ONCE here; the untimed warm-up round compiles; each fit puts the parameter back from
    # the host (the same upload as before) and the optimizer back to its constructed state in
    # place (_opt_state_reset), then runs the 10 steps. The eager arm does the same per-round
    # work. Results raced before this change were wrong for this lane; apply extra scrutiny.
    p = torch.nn.Parameter(p0_host.to(dev))
    try:
        opt = ocls([p], **dict(hyper, **impl))
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError("REFUSED: %s: torch.optim.%s(%s) does not construct on %s: %r"
                           % (arm, s["torch_opt"], impl, dev, exc))
    snap = _opt_state_snapshot(torch, opt, p)
    step = torch.compile(opt.step) if mode == "compile" else opt.step
    info["config"] = "torch.optim.%s([p], %s) on %s, %s" % (s["torch_opt"], dict(hyper, **impl), dev, mode)
    info["implementation"] = impl or "torch default (no foreach/fused argument)"
    info["build_once"] = "parameter, optimizer and compiled step built once; reset in place per round"
    # read back from the constructed optimizer (its defaults are what every step uses)
    rec = _BP().arm_record(opt)
    rec["params"]["input_arrays"] = input_arrays

    def fit():
        with torch.no_grad():
            p.copy_(p0_host.to(dev))
        p.grad = None
        _opt_state_reset(torch, opt, p, snap)
        for gr in g_dev:
            p.grad = gr
            step()
        _torch_sync(torch, dev)
        S["p"] = p
    return Runner(info, fit, lambda: {"param": S["p"].detach().cpu().numpy()}, record=rec)


# ---- linalg ---------------------------------------------------------------

def lu_system(n):
    """A = N(0,1) + 2 sqrt(n) I (well conditioned), B N(0,1) (n, 64); seed 7."""
    np = _np()
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((n, n)).astype(np.float32)
    A[np.arange(n), np.arange(n)] += np.float32(2.0 * np.sqrt(n))
    return A, rng.standard_normal((n, 64)).astype(np.float32)


def sym_system(n):
    """A = (M + M^T) / 2 + 2 sqrt(n) I, M N(0,1) (n, n), seed 7: symmetric, and
    positive definite (the semicircle's edge is sqrt(2n) < 2 sqrt(n))."""
    np = _np()
    rng = np.random.default_rng(SEED)
    M = rng.standard_normal((n, n)).astype(np.float32)
    A = (M + M.T) * np.float32(0.5)
    A[np.arange(n), np.arange(n)] += np.float32(2.0 * np.sqrt(n))
    return A


def _sym_n(lane, D):
    full = 4096 if LANES[lane]["task"] == "eigh" else 8192
    return full if not D.get("_cap") else min(full, max(256, int(D["_cap"])))


def _build_linalg(lane, arm, D):
    np = _np()
    s = LANES[lane]
    t = s["task"]
    S = {}
    if t in ("lu", "lufac"):
        n = 8192 if not D.get("_cap") else min(8192, max(256, int(D["_cap"])))
        args = lu_system(n)
    elif t in ("chol", "eigh"):
        args = (sym_system(_sym_n(lane, D)),)
    elif t == "lstsq":
        args = (D["X"], D["y"])
    else:
        args = (np.ascontiguousarray(D["X"], dtype=np.float32),)
    p = s["params"]
    if arm in OURS_ARMS:
        name, fn = _ours_class(lane)
        info = _ours_info(lane)
        info["config"] = "mojolearn.%s(%s)" % (name, ", ".join("%s=%r" % kv for kv in sorted(p.items())))
        if t == "lufac":
            import mojolearn as ml
            solve2 = getattr(ml, "lu_solve", None) or getattr(ml.linalg, "lu_solve", None)
            if solve2 is None:
                raise Skipped("SKIPPED: not built yet (mojolearn has no lu_solve)")

        def fit():
            if t in ("rsvd", "qr", "eigh", "svd"):
                S["r"] = fn(args[0], **p)
            elif t == "chol":
                S["r"] = fn(**p).fit(args[0]).L_
            elif t == "lufac":
                S["r"] = solve2(fn(args[0]), args[1])
            else:
                S["r"] = fn(*args)
    else:
        lib = s["other"][arm]
        if lib in ("numpy", "sklearn", "scipy"):
            info = {"library": lib, "device": "cpu", "pre_clock_fit": False, "input_home": "host"}
            info.update(_tool("classical_two_datasets")._host_info())

            def fit():
                if lib == "scipy" and t in ("lu", "lstsq", "chol", "qr", "eigh", "svd"):
                    # FAIRNESS FIX (2026-10-13 audit): numpy.linalg ran these in float64 on a
                    # float32 input; scipy.linalg keeps float32 (the LAPACK s-routines) with
                    # numpy's drivers, check_finite=False as numpy checks nothing. Results raced
                    # before this change were wrong for these lanes; apply extra scrutiny.
                    import scipy.linalg as sla
                    if t == "lu":
                        S["r"] = sla.solve(args[0], args[1], assume_a="gen", check_finite=False)
                    elif t == "lstsq":
                        S["r"] = sla.lstsq(args[0], args[1], lapack_driver="gelsd",
                                           check_finite=False)[0]
                    elif t == "chol":
                        S["r"] = sla.cholesky(args[0], lower=True, check_finite=False)
                    elif t == "qr":
                        S["r"] = sla.qr(args[0], mode="economic", check_finite=False)
                    elif t == "eigh":
                        S["r"] = sla.eigh(args[0], lower=p["UPLO"] == "L", driver="evd",
                                          check_finite=False)
                    else:
                        S["r"] = sla.svd(args[0], full_matrices=p["full_matrices"],
                                         lapack_driver="gesdd", check_finite=False)
                elif t == "lu":
                    S["r"] = np.linalg.solve(*args)
                elif t == "lufac":
                    import scipy.linalg as sla
                    S["r"] = sla.lu_solve(sla.lu_factor(args[0]), args[1])
                elif t == "lstsq":
                    S["r"] = np.linalg.lstsq(args[0], args[1], rcond=None)[0]
                elif t == "chol":
                    S["r"] = np.linalg.cholesky(args[0])
                elif t == "qr":
                    S["r"] = np.linalg.qr(args[0], **p)
                elif t == "eigh":
                    S["r"] = np.linalg.eigh(args[0], **p)
                elif t == "svd":
                    S["r"] = np.linalg.svd(args[0], **p)
                else:
                    from sklearn.utils.extmath import randomized_svd
                    S["r"] = randomized_svd(args[0], **p)
        elif lib == "torch":
            import torch
            dev = _torch_device(torch)
            info = {"library": "torch", "version": torch.__version__, "device": "gpu",
                    "pre_clock_fit": False, "input_home": "device"}
            targs = [torch.from_numpy(np.ascontiguousarray(a)).to(dev) for a in args]

            def fit():
                if t == "lu":
                    S["r"] = torch.linalg.solve(*targs)
                elif t == "lufac":
                    LU, piv = torch.linalg.lu_factor(targs[0])
                    S["r"] = torch.linalg.lu_solve(LU, piv, targs[1])
                elif t == "lstsq":
                    S["r"] = torch.linalg.lstsq(targs[0], targs[1].reshape(-1, 1)).solution
                elif t == "chol":
                    S["r"] = torch.linalg.cholesky(targs[0])
                elif t == "qr":
                    S["r"] = torch.linalg.qr(targs[0], mode=p["mode"])
                elif t == "eigh":
                    S["r"] = torch.linalg.eigh(targs[0], UPLO=p["UPLO"])
                elif t == "svd":
                    S["r"] = torch.linalg.svd(targs[0], full_matrices=p["full_matrices"])
                else:
                    torch.manual_seed(p["random_state"])      # svd_lowrank has no seed argument
                    q = min(p["n_components"] + p["n_oversamples"], min(targs[0].shape))
                    u, sv, v = torch.svd_lowrank(targs[0], q=q, niter=p["n_iter"])
                    S["r"] = (u, sv, v.T)
                _torch_sync(torch, dev)
        else:
            import cupy as cp
            info = {"library": "cupy", "version": cp.__version__, "device": "gpu",
                    "pre_clock_fit": False, "input_home": "device"}
            # FAIRNESS FIX (2026-10-13 audit): the kernel clock already
            # excluded these copies, but their duration was discarded.
            # Retain the actual synchronized upload separately so new
            # measurements can derive a whole/whole comparison. Never
            # backfill an older cell with this later copy measurement.
            cp.cuda.runtime.deviceSynchronize()
            upload_start = time.perf_counter()
            cargs = [cp.asarray(a) for a in args]
            cp.cuda.runtime.deviceSynchronize()
            info["upload_ms_untimed"] = (time.perf_counter() - upload_start) * 1000.0

            def fit():
                if t == "lu":
                    S["r"] = cp.linalg.solve(*cargs)
                elif t == "lufac":
                    import cupyx.scipy.linalg as csla
                    S["r"] = csla.lu_solve(csla.lu_factor(cargs[0]), cargs[1])
                elif t == "chol":
                    S["r"] = cp.linalg.cholesky(cargs[0])
                elif t == "qr":
                    S["r"] = cp.linalg.qr(cargs[0], **p)
                elif t == "eigh":
                    S["r"] = cp.linalg.eigh(cargs[0], **p)
                elif t == "svd":
                    S["r"] = cp.linalg.svd(cargs[0], **p)
                else:
                    S["r"] = cp.linalg.lstsq(cargs[0], cargs[1], rcond=None)[0]
                cp.cuda.runtime.deviceSynchronize()

    if arm in OURS_ARMS:
        rec = dict(p if t != "lu" else {}, __library__="mojolearn")
    else:
        lib = s["other"][arm]
        rec = {"__library__": lib}
        if t == "rsvd" and lib == "sklearn":
            rec.update(p)
        elif t == "rsvd":                               # torch.svd_lowrank(q, niter), seeded globally
            # FAIRNESS FIX (2026-10-13 audit): torch's range finder normalizes by QR, the value now
            # pinned on ours and scikit-learn. Results raced before this change were wrong for this
            # lane; apply extra scrutiny.
            rec.update(n_components=p["n_components"], n_oversamples=p["n_oversamples"],
                       n_iter=p["n_iter"], random_state=p["random_state"],
                       power_iteration_normalizer="QR")
        elif t in ("qr", "eigh", "svd"):
            rec.update(p)
        elif t == "chol":
            rec.update(jitter=0.0)                     # scipy/torch/CuPy add no ridge

    # FAIRNESS FIX (2026-10-13 audit): retain actual function identity and
    # shared input bytes even for parameterless linear-algebra APIs.
    if arm in OURS_ARMS:
        rec["__implementation__"] = fn
    elif lib == "torch":
        rec["__implementation__"] = torch.svd_lowrank if t == "rsvd" else getattr(torch.linalg,
            {"lu":"solve", "lufac":"lu_factor", "lstsq":"lstsq", "chol":"cholesky",
             "qr":"qr", "eigh":"eigh", "svd":"svd"}[t])
    elif lib == "cupy":
        if t == "lufac":
            import cupyx.scipy.linalg as csla
            rec["__implementation__"] = csla.lu_factor
        else:
            rec["__implementation__"] = getattr(cp.linalg,
                {"lu":"solve", "lstsq":"lstsq", "chol":"cholesky", "qr":"qr",
                 "eigh":"eigh", "svd":"svd"}[t])
    rec.update(seed=SEED, input_arrays=[dict(shape=list(a.shape), dtype=str(a.dtype),
        sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()) for a in args])

    def outputs():
        r = S["r"]
        if t == "lstsq" and isinstance(r, tuple):     # numpy's (x, residuals, rank, s)
            r = r[0]
        if t == "lstsq" and arm not in OURS_ARMS and s["other"].get(arm) == "torch":
            # FAIRNESS FIX (2026-10-13 audit): torch's GPU lstsq is driver 'gels' (full rank
            # assumed); on a rank-deficient X it returned NaN and still counted as an opponent.
            # Results raced before this change were wrong for this lane; apply extra scrutiny.
            if not np.isfinite(_arr(r, np.float64)).all():
                raise RuntimeError("REFUSED: torch.linalg.lstsq (driver 'gels' on CUDA/ROCm: QR "
                                   "without pivoting, full rank assumed) returned a non-finite "
                                   "solution on this rank-deficient X; not a result")
        if t == "rsvd":
            vt = _arr(r[2], np.float64)          # (k, d), scikit-learn's Vt
            # FAIRNESS FIX (2026-10-13 audit): the quality projection alone
            # omitted the other scored factors from retained output bytes.
            return {"components": vt[:p["n_components"]], "U": _arr(r[0], np.float64),
                    "S": _arr(r[1], np.float64), "Vh": vt}
        if t == "chol":
            return {"L": _arr(r, np.float64)}
        if t == "qr":
            # FAIRNESS FIX (2026-10-13 audit): retaining only R lost Q and
            # could accept a correct R paired with an incorrect Q.
            return {"Q": _arr(r[0], np.float64), "R": _arr(r[1], np.float64)}
        if t == "eigh":
            return {"w": _arr(r[0], np.float64), "V": _arr(r[1], np.float64)}
        if t == "svd":
            U = _arr(r[0], np.float64)
            return {"S": _arr(r[1], np.float64), "Vh": _arr(r[2], np.float64),
                    # FAIRNESS FIX (2026-10-13 audit): keep every scored
                    # factor row even when a scorer uses a declared sample.
                    "U": U, "U_rows": _stride(U, 100_000)}
        return {"x": _arr(r, np.float64).reshape(-1)}
    runner = Runner(info, fit, outputs, record=rec)
    if arm in OURS_ARMS:                # the host arrays fit() hands our call (the upload probe)
        _with_upload(runner, [("arg%d" % i, a) for i, a in enumerate(args)])
    return runner


# ---- function calls (resampling, cross-validation) -------------------------

def _scipy_rng_kw(fn):
    """scipy's seed argument: `rng` from 1.15, `random_state` before it."""
    import inspect
    return "rng" if "rng" in inspect.signature(fn).parameters else "random_state"


def _build_fn(lane, arm, D):
    np = _np()
    s = LANES[lane]
    t = s["task"]
    p = s["params"]
    S = {}
    y = np.ascontiguousarray(D["y"], dtype=np.float32)
    yq = np.ascontiguousarray(D["yq"], dtype=np.float32) if "yq" in D else None
    X = D["X"]
    if arm in OURS_ARMS:
        name, fn = _ours_class(lane)
        info = _ours_info(lane)
        kw = {k: _resolve(v, "ours") for k, v in p.items()}
        info["config"] = "mojolearn.%s(%s)" % (name, ", ".join("%s=%r" % kv for kv in sorted(kw.items())))

        def fit():
            if t == "bootstrap":
                r = fn(y, **kw)
                S["o"] = {"ci": np.array([r.confidence_interval.low, r.confidence_interval.high]),
                          "se": np.array([r.standard_error])}
            elif t == "permutation":
                r = fn(y, yq, **kw)
                S["o"] = {"stat": np.array([r.statistic]), "p": np.array([r.pvalue])}
            elif t == "resample":
                Xr, yr = fn(X, y, **kw)
                S["o"] = {"xmean": _arr(Xr, np.float64).mean(0), "ymean": np.array([_arr(yr, np.float64).mean()])}
            else:                           # cross_val_score clones the estimator per fold
                est = _resolve(p["estimator"], "ours")
                S["o"] = {"scores": _arr(fn(est, X, y, cv=p["cv"], scoring=p["scoring"]),
                                         np.float64)}
        rec = {k: ("LinearRegression" if k == "estimator" else v) for k, v in p.items()}
        rec["__library__"] = "mojolearn"
        return Runner(info, fit, lambda: S["o"], record=rec)
    lib = s["other"][arm]
    info = _tool("bench_board_more")._sk_info()
    info.update(library=lib, device="cpu", pre_clock_fit=False, input_home="host")
    if lib == "scipy":
        import scipy
        from scipy import stats
        info["version"] = scipy.__version__
        if t == "bootstrap":
            seed_kw = _scipy_rng_kw(stats.bootstrap)
            kw = dict(n_resamples=p["n_resamples"], confidence_level=p["confidence_level"],
                      method=p["method"], alternative=p["alternative"], vectorized=True, batch=250)
            kw[seed_kw] = p["random_state"]
            info["config"] = "scipy.stats.bootstrap((y,), np.mean, %s)" % kw

            def fit():
                r = stats.bootstrap((y,), np.mean, **kw)
                S["o"] = {"ci": np.array([r.confidence_interval.low, r.confidence_interval.high]),
                          "se": np.array([r.standard_error])}
            rec = dict(statistic="mean", n_resamples=p["n_resamples"],
                       confidence_level=p["confidence_level"], method=p["method"],
                       alternative=p["alternative"], random_state=p["random_state"])
        else:
            seed_kw = _scipy_rng_kw(stats.permutation_test)
            kw = dict(permutation_type=p["permutation_type"], n_resamples=p["n_resamples"],
                      alternative=p["alternative"], vectorized=True, batch=250)
            kw[seed_kw] = p["random_state"]
            info["config"] = "scipy.stats.permutation_test((x, y), mean(x) - mean(y), %s)" % kw

            def diff_means(a, b, axis):
                return np.mean(a, axis=axis) - np.mean(b, axis=axis)

            def fit():
                r = stats.permutation_test((y, yq), diff_means, **kw)
                S["o"] = {"stat": np.array([r.statistic]), "p": np.array([r.pvalue])}
            rec = dict(statistic="diff_means", n_resamples=p["n_resamples"],
                       alternative=p["alternative"], random_state=p["random_state"],
                       permutation_type=p["permutation_type"])
    else:
        if t == "resample":
            from sklearn.utils import resample as skresample
            info["config"] = "sklearn.utils.resample(X, y, %s)" % p

            def fit():
                Xr, yr = skresample(X, y, **p)
                S["o"] = {"xmean": np.asarray(Xr, dtype=np.float64).mean(0),
                          "ymean": np.array([np.asarray(yr, dtype=np.float64).mean()])}
            rec = dict(p)
        else:
            from sklearn.model_selection import cross_val_score as skcv
            info["config"] = ("sklearn.model_selection.cross_val_score(LinearRegression(), X, y, "
                              "cv=%d, scoring=%r, n_jobs=-1)" % (p["cv"], p["scoring"]))

            def fit():
                est = _resolve(p["estimator"], "sklearn")
                S["o"] = {"scores": np.asarray(skcv(est, X, y, cv=p["cv"], scoring=p["scoring"],
                                                    n_jobs=-1), dtype=np.float64)}
            rec = dict(p, estimator="LinearRegression")
    rec["__library__"] = lib
    return Runner(info, fit, lambda: S["o"], record=rec)


# ---- ALS ------------------------------------------------------------------

def _als_split(X):
    """(train csr, held-out item per row or -1): each row's largest-count item
    is held out, ties to the lowest item index."""
    np = _np()
    import scipy.sparse as sp
    Xh = np.asarray(X, dtype=np.float32).copy()
    nz = (Xh > 0).sum(axis=1)
    held = np.where(nz >= 2, np.argmax(Xh, axis=1), -1)
    rows = np.nonzero(held >= 0)[0]
    Xh[rows, held[rows]] = 0.0
    return sp.csr_matrix(Xh), held


def _build_als(lane, arm, D):
    np = _np()
    s = LANES[lane]
    p = s["params"]
    C, held = _als_split(D["X"])
    S = {}
    sync = None
    if arm in OURS_ARMS:
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        info["config"] = "mojolearn.%s(%s).fit(csr)" % (name, p)

        def fit():
            S["e"] = cls(**p).fit(C)

        def outputs():
            return {"U": _arr(S["e"].user_factors_, np.float32), "V": _arr(S["e"].item_factors_, np.float32)}
        return Runner(info, fit, outputs, record=_BP().arm_record(cls(**p)))
    gpu = arm == "implicit-gpu"
    if gpu:
        # BEFORE `import implicit`: its __init__ imports implicit.gpu, which
        # decides HAS_CUDA once, at that import
        _preload_cuda_libs(("cudart", "cublasLt", "cublas", "curand"))
    import implicit
    if gpu:
        import implicit.gpu
        if not implicit.gpu.HAS_CUDA:
            raise RuntimeError("REFUSED: the pinned implicit wheel was built without CUDA "
                               "(implicit.gpu.HAS_CUDA is False)")
    from implicit.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits
    threadpool_limits(1, "blas")          # implicit's documented setting: its own threads, BLAS at 1
    info = {"library": "implicit", "env": dict(ARM_ENV.get(arm, {})), "version": implicit.__version__, "device": "gpu" if gpu else "cpu",
            "pre_clock_fit": False, "input_home": "host",
            "config": "implicit AlternatingLeastSquares(%s, use_gpu=%s)" % (p, gpu)}
    akw = dict(factors=p["factors"], regularization=p["regularization"], alpha=p["alpha"],
               iterations=p["iterations"], random_state=p["random_state"], use_gpu=gpu,
               calculate_training_loss=p["calculate_training_loss"])
    if not gpu:                        # implicit's GPU model has only its CG solver
        # implicit 0.7.3 takes no cg_steps (its CG runs its own fixed steps);
        # with use_cg=False on both sides no CG step runs, so ours' cg_steps
        # has no effect either
        akw.update(use_cg=p["use_cg"])
    rec = dict(akw, __library__="implicit")

    def fit():
        m = AlternatingLeastSquares(**akw)
        m.fit(C, show_progress=False)
        S["m"] = m

    def outputs():
        m = S["m"]
        if gpu:
            m = m.to_cpu()
        return {"U": np.asarray(m.user_factors, dtype=np.float32),
                "V": np.asarray(m.item_factors, dtype=np.float32)}
    return Runner(info, fit, outputs, None, sync, record=rec)


# ---- SHAP -----------------------------------------------------------------

def _shared_ridge_f32(X, y, alpha=1.0):
    """(w32, b32): THE linear model every kernel- and permutation-SHAP arm explains.

    FAIRNESS FIX (2026-10-13 audit): ours explained its own device RidgeCV while the opponents
    explained a NumPy/LAPACK ridge (host-BLAS coefficients, a float64 cupy GEMV on cuML). This
    solves the same objective once, the centered ridge (X_c^T X_c + alpha I) w = X_c^T y_c with
    b = mean(y) - mean(X) . w, in a FIXED ORDER: sequential row folds of elementwise IEEE float64
    operations (no BLAS, no pairwise sum) and a Cholesky solve in Python floats. The result is
    the same bits on every host, so every worker process (each arm builds its own) derives the
    same coefficients, and ours' cross-vendor identity holds. The coefficients are rounded to
    float32, the precision ours' linear predict runs in, and every arm evaluates the model in
    float32. Harness work before every clock. Results raced before this change were wrong for
    these lanes; apply extra scrutiny."""
    np = _np()
    import math
    Xd = np.ascontiguousarray(X, dtype=np.float64)
    yd = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    n, d = Xd.shape
    mu = np.zeros(d, dtype=np.float64)
    for i in range(n):
        mu += Xd[i]
    mu /= float(n)
    ym = 0.0
    for v in yd.tolist():
        ym += v
    ym /= float(n)
    gram = np.zeros((d, d), dtype=np.float64)
    rhs = np.zeros(d, dtype=np.float64)
    xc = np.empty(d, dtype=np.float64)
    outer = np.empty((d, d), dtype=np.float64)
    yl = yd.tolist()
    for i in range(n):
        np.subtract(Xd[i], mu, out=xc)
        np.multiply.outer(xc, xc, out=outer)
        gram += outer
        rhs += xc * (yl[i] - ym)
    A = gram.tolist()
    r = rhs.tolist()
    for j in range(d):
        A[j][j] += float(alpha)
    L = [[0.0] * d for _ in range(d)]
    for j in range(d):
        Lj = L[j]
        s = A[j][j]
        for k in range(j):
            s -= Lj[k] * Lj[k]
        if not s > 0.0:
            raise RuntimeError("the shared ridge's normal matrix is not positive definite")
        Lj[j] = math.sqrt(s)
        for i in range(j + 1, d):
            Li = L[i]
            s = A[i][j]
            for k in range(j):
                s -= Li[k] * Lj[k]
            Li[j] = s / Lj[j]
    z = [0.0] * d
    for i in range(d):
        Li = L[i]
        s = r[i]
        for k in range(i):
            s -= Li[k] * z[k]
        z[i] = s / Li[i]
    wv = [0.0] * d
    for i in range(d - 1, -1, -1):
        s = z[i]
        for k in range(i + 1, d):
            s -= L[k][i] * wv[k]
        wv[i] = s / L[i][i]
    w32 = np.asarray(wv, dtype=np.float32)
    b = ym
    for mj, wj in zip(mu.tolist(), w32.tolist()):
        b -= mj * wj
    return w32, float(np.float32(b))


def _build_shap(lane, arm, D):
    np = _np()
    s = LANES[lane]
    t = s["task"]
    p = s["params"]
    X, y, Xq = D["X"], D["y"], D["Xq"]
    S = {}
    if t == "tree-shap":
        gb = dict(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                  learning_rate=p["learning_rate"])
        if arm not in OURS_ARMS:
            # FAIRNESS FIX (2026-10-13 audit): these arms explained their own GBDT (XGBoost's,
            # LightGBM's) with training covers while ours explains its RandomForestRegressor with
            # 100-row background covers: different models and covers. Refused by name before
            # any fit (the code below is kept for the record). Results raced before this change
            # were wrong for this lane; apply extra scrutiny.
            raise RuntimeError(
                "REFUSED: a different model: ours explains its RandomForestRegressor with node "
                "covers from 100 background rows; %s would explain its own library's GBDT with "
                "training covers. No opponent explains ours' forest with ours' covers through its "
                "public API, so the race has no opponent (fairness audit 2026-10-13)" % arm)
        if arm in OURS_ARMS:
            import mojolearn as ml
            name, cls = _ours_class(lane)
            info = _ours_info(lane)
            model = ml.RandomForestRegressor(n_estimators=gb["n_estimators"], max_depth=gb["max_depth"],
                                             random_state=SEED)
            model.fit(X, y)
            bg = X if _aft_full_tree_workload(lane) else _stride(X, 100)
            info["pre_clock_fit"] = True
            info["config"] = ("mojolearn.%s(RandomForestRegressor(n_estimators=%d, max_depth=%d), "
                              "data=%d %s rows)" % (name, gb["n_estimators"], gb["max_depth"],
                              bg.shape[0], "full" if _aft_full_tree_workload(lane) else "stride"))
            if os.environ.get("MOJOLEARN_AFT_CAPTURE") == "1":
                info["aft_tree_shap"] = {
                    "background_rows": int(bg.shape[0]), "explain_rows": int(Xq.shape[0]),
                    "model_fit_rows": int(X.shape[0]), "model": "RandomForestRegressor",
                    "model_params": {"n_estimators": gb["n_estimators"],
                                     "max_depth": gb["max_depth"], "random_state": SEED},
                    "model_fit_during_build": True,
                }

            def fit():
                ex = cls(model, bg)
                S["phi"] = ex.shap_values(Xq)
                S["base"] = ex.expected_value
                S["margin"] = model.predict(Xq)
        elif arm in ("shap-cpu", "xgboost-cpu", "xgboost-gpu"):
            import xgboost as xgb
            gpu = arm == "xgboost-gpu"
            model = xgb.XGBRegressor(n_estimators=gb["n_estimators"], max_depth=gb["max_depth"],
                                     learning_rate=gb["learning_rate"], tree_method="hist",
                                     device="cuda" if gpu else "cpu", random_state=SEED)
            model.fit(X, y)
            info = {"library": "shap" if arm == "shap-cpu" else "xgboost", "device": "gpu" if gpu else "cpu",
                    "pre_clock_fit": True, "input_home": "host", "version": xgb.__version__}
            booster = model.get_booster()
            dq = xgb.DMatrix(Xq)
            if arm == "shap-cpu":
                import shap
                info["version"] = shap.__version__
                info["config"] = "shap.TreeExplainer(XGBRegressor(%s)).shap_values(Xq)" % gb

                def fit():
                    ex = shap.TreeExplainer(model)
                    S["phi"] = ex.shap_values(Xq)
                    S["base"] = ex.expected_value
                    S["margin"] = booster.predict(dq, output_margin=True)
            else:
                info["config"] = "XGBRegressor(%s, device=%s).get_booster().predict(pred_contribs=True)" % (
                    gb, "cuda" if gpu else "cpu")

                def fit():
                    c = booster.predict(dq, pred_contribs=True)
                    S["phi"], S["base"] = c[:, :-1], c[:, -1]
                    S["margin"] = booster.predict(dq, output_margin=True)
        else:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(n_estimators=gb["n_estimators"], max_depth=gb["max_depth"],
                                      num_leaves=2 ** gb["max_depth"], learning_rate=gb["learning_rate"],
                                      random_state=SEED, verbose=-1)
            model.fit(X, y)
            info = {"library": "lightgbm", "version": lgb.__version__, "device": "cpu",
                    "pre_clock_fit": True, "input_home": "host",
                    "config": "LGBMRegressor(%s).predict(pred_contrib=True)" % gb}

            def fit():
                c = model.predict(Xq, pred_contrib=True)
                S["phi"], S["base"] = c[:, :-1], c[:, -1]
                S["margin"] = model.predict(Xq, raw_score=True)

        def outputs():
            return {"phi": _arr(S["phi"], np.float64),
                    "base": np.broadcast_to(_arr(S["base"], np.float64), (Xq.shape[0],)).copy(),
                    "margin": _arr(S["margin"], np.float64).reshape(-1)}
        rec = dict(__library__="mojolearn" if arm in OURS_ARMS else arm.rsplit("-", 1)[0],
                   n_estimators=gb["n_estimators"], max_depth=gb["max_depth"], seed=SEED,
                   model=("RandomForestRegressor" if arm in OURS_ARMS else
                          "LGBMRegressor" if arm == "lightgbm-cpu" else "XGBRegressor"))
        if arm not in OURS_ARMS:
            rec["learning_rate"] = gb["learning_rate"]
        return Runner(info, fit, outputs, record=rec)
    # kernel / permutation SHAP over a ridge (alpha 1, centered) fit before the clock
    bg = _stride(X, p["n_background"])
    # FAIRNESS FIX (2026-10-13 audit): ours explained its own device RidgeCV (float32 outputs)
    # while the opponents explained the former `numpy_ridge`, a NumPy/LAPACK closed form of the same
    # objective with host-BLAS coefficients, and cuML evaluated it as a float64 cupy GEMV (FP64
    # on the L40S): two models in two precisions. Every arm now explains ONE float32 linear model,
    # w32 and b32 from _shared_ridge_f32 (a fixed-order solve, the same bits on every box and in
    # every worker), each through its own library's device or host evaluation of that model.
    # Results raced before this change were wrong for this lane; apply extra scrutiny.
    w32, b32 = _shared_ridge_f32(X, y)
    w = w32.astype(np.float64)              # the model's coefficients, exactly, for `exact`
    if arm in OURS_ARMS:
        # Lane shap-xvendor-identity (2026-10-08). OUR arm explains this library's own device
        # ridge, mojolearn.RidgeCV(alphas=[1.0]) (the x_linear binding: the ridge-cv lane's fit
        # and predict, the same bits on NVIDIA and AMD on every grid run), fit before the clock.
        # Until then it explained the NumPy closed form of a host ridge: OpenBLAS dgemm/dgesv
        # for w and dgemv for every model call round differently on the two boxes' host CPUs, so
        # the model the explainer was handed and the `exact` reference the digest hashes differed
        # between the vendors (the only two divergent lanes of the board) while every explainer
        # word on the device is an integer soft-float64 unit (xtrees/agnostic.mojo).
        # FAIRNESS FIX (2026-10-13 audit): the fitted RidgeCV now holds the shared coefficients
        # (coef_ = w32, intercept_ = b32; its predict reads both on every call), so ours explains
        # the opponents' model on its own device path. The coefficients are host-independent, so
        # the cross-vendor identity this lane was moved for is kept. Results raced before this
        # change were wrong for this lane; apply extra scrutiny.
        import mojolearn as ml
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        ridge = ml.RidgeCV(alphas=[1.0], fit_intercept=True).fit(X, y)
        ridge.coef_ = w32
        ridge.intercept_ = b32
        info["pre_clock_fit"] = True
        info["model"] = ("mojolearn.RidgeCV(alphas=[1.0]) on the device holding the shared float32 "
                         "coefficients (_shared_ridge_f32)")
        info["config"] = "mojolearn.%s(mojolearn.RidgeCV(alphas=[1.0]).predict, background 100)" % name

        def fit():
            ex = (cls(ridge, bg, link=p["link"], random_state=SEED) if t == "kernel-shap"
                  else cls(ridge, bg, random_state=SEED))
            S["phi"] = (ex.shap_values(Xq, nsamples=p["nsamples"]) if t == "kernel-shap"
                        else ex.shap_values(Xq, npermutations=p["npermutations"]))
    elif arm == "shap-cpu":
        import shap
        b_f32 = np.float32(b32)

        def predict(Z):
            return np.asarray(_host(Z), dtype=np.float32) @ w32 + b_f32
        info = {"library": "shap", "version": shap.__version__, "device": "cpu",
                "pre_clock_fit": True, "input_home": "host",
                "model": "the shared float32 ridge (_shared_ridge_f32) as a float32 NumPy GEMV"}
        if t == "kernel-shap":
            info["config"] = "shap.KernelExplainer(ridge.predict, bg100).shap_values(Xq, nsamples=%d)" % p["nsamples"]

            def fit():
                np.random.seed(SEED)          # shap's KernelExplainer draws from numpy's global RNG
                S["phi"] = shap.KernelExplainer(predict, bg, link=p["link"]).shap_values(
                    Xq, nsamples=p["nsamples"], l1_reg=p["l1_reg"], silent=True)
        else:
            info["config"] = "shap.PermutationExplainer(ridge.predict, bg100)(Xq, max_evals=%d)" % (
                2 * p["npermutations"] * (X.shape[1] + 1))

            def fit():
                ex = shap.PermutationExplainer(predict, bg, seed=SEED)
                S["phi"] = ex(Xq, max_evals=2 * p["npermutations"] * (X.shape[1] + 1)).values
    else:
        from cuml.explainer import KernelExplainer, PermutationExplainer
        # float32 background and rows (the upload is untimed), so cuML generates float32
        # synthetic rows, the precision of the model
        dev, info, sync = _cuml_up({"bg": np.ascontiguousarray(bg, dtype=np.float32),
                                    "Xq": np.ascontiguousarray(Xq, dtype=np.float32)})
        info["pre_clock_fit"] = True
        info["model"] = "the shared float32 ridge (_shared_ridge_f32) as a float32 cupy GEMV"
        import cupy as cp
        wd = cp.asarray(w32)
        bd = cp.float32(b32)

        def cpredict(Z):
            return cp.asarray(Z, dtype=cp.float32) @ wd + bd
        if t == "kernel-shap":
            info["config"] = "cuml.explainer.KernelExplainer(model, bg100, nsamples=%d)" % p["nsamples"]

            def fit():
                S["phi"] = KernelExplainer(model=cpredict, data=dev["bg"], nsamples=p["nsamples"],
                                           random_state=SEED, is_gpu_model=True).shap_values(dev["Xq"])
                cp.cuda.runtime.deviceSynchronize()
        else:
            info["config"] = "cuml.explainer.PermutationExplainer(model, bg100)"

            def fit():
                S["phi"] = PermutationExplainer(model=cpredict, data=dev["bg"], random_state=SEED,
                                                is_gpu_model=True).shap_values(
                    dev["Xq"], npermutations=p["npermutations"])
                cp.cuda.runtime.deviceSynchronize()
    # the linear model's exact SHAP values w_j (x_j - E x_j), E over the background rows: the
    # background mean is one fixed-order fold of its rows (no BLAS, no pairwise sum), so the
    # reference the digest hashes carries no host-CPU bits (harness reference, outside every clock)
    bg64 = bg.astype(np.float64)
    bgsum = np.zeros(bg64.shape[1], dtype=np.float64)
    for row in range(bg64.shape[0]):
        bgsum += bg64[row]
    exact = (Xq.astype(np.float64) - bgsum / float(bg64.shape[0])) * w

    def outputs():
        return {"phi": _arr(S["phi"], np.float64).reshape(exact.shape), "exact": exact}
    rec = dict(__library__="mojolearn" if arm in OURS_ARMS else arm.rsplit("-", 1)[0], seed=SEED,
               n_background=p["n_background"])
    rec.update({k: p[k] for k in ("nsamples", "link", "l1_reg", "npermutations") if k in p})
    return Runner(info, fit, outputs, record=rec)


# ---- SVGP -----------------------------------------------------------------

def _build_svgp(lane, arm, D):
    np = _np()
    s = LANES[lane]
    p = s["params"]
    X, y, Xq = D["X"], D["y"], D["Xq"]
    Z0 = _stride(X, p["n_inducing"])
    S = {}
    hyp = {k: p[k] for k in ("kernel_variance", "lengthscale", "noise_variance", "jitter")}
    if arm in OURS_ARMS:
        name, cls = _ours_class(lane)
        info = _ours_info(lane)
        info["config"] = "mojolearn.%s(inducing_points=%d stride rows, %s)" % (name, Z0.shape[0], hyp)

        def fit():
            S["e"] = cls(inducing_points=Z0, **hyp).fit(X, y)

        def infer():
            S["pred"] = S["e"].predict(Xq)
        rec = dict(hyp, __library__="mojolearn", n_inducing=int(Z0.shape[0]))
        return _with_upload(Runner(info, fit, lambda: {"pred": _arr(S["pred"], np.float64).reshape(-1)},
                                   infer, record=rec),
                            [("inducing_points", Z0), ("X", X), ("y", y)])
    import torch
    import gpytorch
    dev = _torch_device(torch) if arm == "gpytorch-gpu" else "cpu"
    if arm == "gpytorch-gpu" and dev == "cpu":
        raise RuntimeError("REFUSED: no torch GPU device on this box")
    info = {"library": "gpytorch", "version": gpytorch.__version__, "device": "gpu" if dev != "cpu" else "cpu",
            "pre_clock_fit": False, "input_home": "device",
            "config": "gpytorch ExactGP(ZeroMean, InducingPointKernel(ScaleKernel(RBFKernel), Z=%d "
                      "stride rows, fixed)), GaussianLikelihood; outputscale %g, lengthscale %g, noise "
                      "%g, none trained (SGPR, the collapsed bound); sgpr_diagonal_correction(False) "
                      "(the Titsias posterior, no FITC diagonal); the fit clock builds the "
                      "prediction cache" % (Z0.shape[0], hyp["kernel_variance"], hyp["lengthscale"],
                                            hyp["noise_variance"])}
    Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y.astype(np.float32)).to(dev)
    Xqt = torch.from_numpy(Xq).to(dev)
    Zt = torch.from_numpy(Z0).to(dev)

    class GP(gpytorch.models.ExactGP):
        def __init__(self, lik):
            super().__init__(Xt, yt, lik)
            self.mean_module = gpytorch.means.ZeroMean()
            base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            base.outputscale = hyp["kernel_variance"]
            base.base_kernel.lengthscale = hyp["lengthscale"]
            self.covar_module = gpytorch.kernels.InducingPointKernel(base, inducing_points=Zt.clone(),
                                                                    likelihood=lik)
            self.covar_module.inducing_points.requires_grad_(False)

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    # FAIRNESS FIX (2026-10-13 audit): in eval mode GPyTorch's InducingPointKernel adds the
    # diagonal correction diag(K_ff - Q_ff) to the TRAIN covariance as well
    # (gpytorch.settings.sgpr_diagonal_correction, default True; inducing_point_kernel.py
    # _get_covariance), so its mean cache solved a FITC-style posterior, not the Titsias
    # (VFE/DTC) posterior ours computes; the taxi R2 differed (-0.210 vs ours -0.195). The
    # correction is now off in the fit and the predict clock, so both arms solve the same model.
    # Results raced before this change were wrong for this lane; apply extra scrutiny.
    def _ctx():
        return (torch.no_grad(), gpytorch.settings.fast_pred_var(False),
                gpytorch.settings.sgpr_diagonal_correction(False))

    def fit():
        torch.manual_seed(SEED)
        lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
        lik.noise = hyp["noise_variance"]
        model = GP(lik).to(dev)
        model.eval()
        lik.eval()
        a, b, c = _ctx()
        with a, b, c:
            model(Xqt[:1]).mean                # builds the predictive cache (the "fit")
        _torch_sync(torch, dev)
        S["m"] = model

    def infer():
        a, b, c = _ctx()
        with a, b, c:
            S["pred"] = S["m"](Xqt).mean
        _torch_sync(torch, dev)
    rec = dict(hyp, __library__="gpytorch", n_inducing=int(Z0.shape[0]), seed=SEED,
               sgpr_diagonal_correction=False)
    return Runner(info, fit, lambda: {"pred": _torch_metric_array(S["pred"])}, infer, record=rec)


# ---------------------------------------------------------------------------
# worker process
# ---------------------------------------------------------------------------

#: the dataset this process races (set by _load_block; each worker and
#: conductor handles one), read by _derived_params for s["dataset_params"]
_DATASET = None


def _load_block(lane, dataset, data):
    global _DATASET
    _DATASET = dataset
    np = _np()
    name = block_file(lane, dataset)
    if name is None:
        rec = {"block": block_of(lane), "dataset": dataset, "synthetic": True}
        B = {}
        if os.environ.get("MOJOLEARN_ALGOS_SMOKE_ROWS"):
            B["_cap"] = int(os.environ["MOJOLEARN_ALGOS_SMOKE_ROWS"])
        return B, rec
    base = os.path.join(data, name)
    with np.load(base + ".npz") as z:
        B = {k: np.ascontiguousarray(z[k]) for k in z.files}
    with open(base + ".json") as fh:
        rec = json.load(fh)
    if _aft_full_tree_workload(lane):
        actual = {"train": list(B["X"].shape), "test": list(B["Xq"].shape)}
        if (rec.get("full_tree_workload") is not True or rec.get("smoke_max_rows")
                or rec.get("dataset") != dataset or rec.get("source_dimensions") != actual):
            raise ValueError("full tree input must retain uncapped loader shapes and provenance")
    return B, rec


def _aft_capture_worker(args, B, D, rec, runner):
    """Untimed evidence from this actual worker, never estimator runtime."""
    ctd = _tool("classical_two_datasets")
    loaded = {}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not os.path.basename(path).startswith("_mojolearn") or not path.endswith(".so"):
            continue
        path = os.path.realpath(path)
        name = os.path.basename(path)
        if os.path.basename(os.path.dirname(path)) == "identical":
            name = "identical/" + name
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        loaded[name] = {"path": path, "sha256": digest.hexdigest()}
    dimensions = {"train": list(D["X"].shape), "test": list(D["Xq"].shape)}
    full = _aft_full_tree_workload(args.lane)
    cfg = lane_config(args.lane)
    settings = {"lane_config": cfg, "full_tree_workload": full,
                "effective_stride_subsets": {} if full else dict(LANES[args.lane]["sub"])}
    if args.lane == "tree-shap":
        settings["tree_shap"] = dict((runner.info or {}).get("aft_tree_shap", {}))
    input_hashes = {k: ctd.sha256_array(D[k]) for k in ("X", "Xq", "y", "yq") if k in D}
    prepared_hashes = {k: rec.get("arrays", {}).get(k, {}).get("sha256") for k in input_hashes}
    if full and input_hashes != prepared_hashes:
        raise ValueError("prepared full input hashes disagree with actual worker arrays")
    pools = None
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pass
    return {
        "schema": "aft-algos-capture-v1", "dataset": rec.get("dataset"),
        "dimensions": dimensions, "estimator_settings": settings,
        "actual_worker_parameters": getattr(runner, "record", None),
        "actual_worker_config": (runner.info or {}).get("config"),
        "timed_boundary": AFT_ALGOS_BOUNDARY, "loaded_bindings": loaded,
        "mode": (runner.info or {}).get("numeric_mode_used"),
        "vendor": (runner.info or {}).get("vendor_used"),
        "input_array_sha256": input_hashes, "prepared_block": rec,
        "full_dataset_coverage": "all saved loader train/test rows" if full else "pending: historical caps/subsets",
        "intrinsic_caps": {"full_tree_workload": full, "prepared_smoke_max_rows": rec.get("smoke_max_rows"),
                           "historical_FIT_ROWS": FIT_ROWS, "historical_EVAL_ROWS": EVAL_ROWS,
                           "historical_lane_subsets": dict(LANES[args.lane]["sub"]),
                           "applied_lane_subsets": {} if full else dict(LANES[args.lane]["sub"]),
                           "tree_shap_background_rows": settings.get("tree_shap", {}).get("background_rows")},
        "pre_clock_fit": bool((runner.info or {}).get("pre_clock_fit")),
        "whole_operation_requested": os.environ.get("MOJOLEARN_BENCH_WHOLE_OPERATION") == "1",
        "resource_policy": {"cpu_libraries": "unrestricted on dedicated Apple",
                            "cpu_count": os.cpu_count(), "effective_pools": pools,
                            "thread_environment": {key: os.environ.get(key) for key in ctd.THREAD_ENV}},
    }


def worker(args):
    global _NEURAL_FIXTURES, _OURS_INIT
    # FAIRNESS FIX (2026-10-13 audit): the torch twins of the RNN/LSTM/GRU and CNN lanes read
    # ours' exported initial weights and row orders from here (_read_ours_init). Results raced
    # before this change were wrong for this lane; apply extra scrutiny.
    _OURS_INIT = getattr(args, "ours_init", None)
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        proto.flush()

    np = _np()
    try:
        ab_config = _configure_neural_ab(getattr(args, "neural_ab_config", None), args.lane)
        if ab_config is not None and args.arm == "ours-fast":
            raise ValueError("neural A/B configuration requires the ours IDENTICAL arm")
        _NEURAL_FIXTURES = (getattr(args, "neural_fixtures", None)
                            or _tool("neural_fixtures").default_dir(args.data))
        # the untimed own-host reference (bench_board_host_quality.enrich) is a child process that reads the
        # fixtures through the environment; without this every layer lane's quality cell said "neural fixtures
        # missing" on a box that had them (2026-10-11, nv2 v0065 / amd a1647 conv2d + resnet-block)
        if _NEURAL_FIXTURES:
            os.environ[_tool("neural_fixtures").ENV_DIR] = _NEURAL_FIXTURES
        B, rec = _load_block(args.lane, args.dataset, args.data)
        D = lane_arrays(args.lane, B)
        runner = build(args.lane, args.arm, D)
    except Skipped as exc:
        say({"event": "skipped", "reason": str(exc)})
        return 0
    except BaseException as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if getattr(exc, "infrastructure", False):   # e.g. neural fixtures missing on this box
            say({"event": "error", "stage": "ready", "class": "infrastructure",
                 "status": "not_ready", "error": str(exc)})
            return 1
        say({"event": "error", "stage": "ready", "error": repr(exc)})
        return 1
    if isinstance(runner.info, dict):   # the arm's own library version and GPU (the store's key)
        runner.info.update(_tool("bench_board_probe").library_identity(runner.info))
        if ab_config is not None:
            runner.info["neural_ab_config"] = ab_config
            runner.info["neural_ab_lane_config"] = lane_config(args.lane)
            runner.info["neural_ab_binary_admission"] = "requested source flags only; not established by this config"
    say({"event": "ready", "info": runner.info, "pid": os.getpid(),
         "params_record": getattr(runner, "record", None)})
    mem = _tool("bench_board_probe").MemProbe((runner.info or {}).get("device", "gpu"),
        library=(runner.info or {}).get("library") or "?", arm=args.arm)
    last = None
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "round":
            r = int(parts[1])
            try:
                mem.start()
                whole_start = time.perf_counter()
                whole_requested = os.environ.get("MOJOLEARN_BENCH_WHOLE_OPERATION") == "1"
                if whole_requested:
                    # Recreate per operation so constructor-side fit/preparation
                    # (notably kNN/KDE) is included, on every scored arm equally.
                    # Release the previous fitted device model before allocating
                    # its replacement, avoiding a transient double allocation.
                    runner = None
                    runner = build(args.lane, args.arm, D)
                    runner.sync_for_receipt()
                preparation_ms = (time.perf_counter() - whole_start) * 1000.0
                t0 = time.perf_counter()
                runner.fit()
                ms = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                did = runner.infer()
                ims = (time.perf_counter() - t1) * 1000.0 if did else None
                last = runner.outputs()
                runner.sync_for_receipt()
                operation_ms = (time.perf_counter() - whole_start) * 1000.0
                m = mem.stop()
                h = hashlib.sha256()
                for k in sorted(last):
                    h.update(k.encode())
                    h.update(np.ascontiguousarray(last[k]).data)
                digest = h.hexdigest()[:16]
                from bench_board_state import scored_receipt
                state_receipt = scored_receipt(last, runner) if r > 0 else None
                aft_capture = None
                if r > 0 and os.environ.get("MOJOLEARN_AFT_CAPTURE") == "1":
                    aft_capture = _aft_capture_worker(args, B, D, rec, runner)
                # the separate upload clock (OUR arm only), after every clock of this round
                # has closed (ms, infer_ms, operation_ms, mem); see _ours_upload_probe
                upload_sep = None
                if args.arm in OURS_ARMS and (runner.info or {}).get("library") == "mojolearn":
                    upload_sep = _ours_upload_probe(runner)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                say({"event": "error", "stage": "round %d" % r, "error": repr(exc)})
                return 1
            say({"event": "round", "round": r, "ms": ms, "infer_ms": ims, "digest": digest, "mem": m, "state_receipt": state_receipt,
                 "aft_capture": aft_capture, "upload_separate": upload_sep,
                 "operation": {"ms": operation_ms, "preparation_ms": preparation_ms,
                               "scope": "prepare-fit-consume" if whole_requested else "call-consume",
                               "dataset_load_included": False,
                               "fresh_process": False, "startup_probe_preparation_excluded": True}})
        elif parts[0] == "save":
            try:
                path = parts[1]
                tmp = path + ".tmp.npz"
                np.savez(tmp, **last)
                os.replace(tmp, path)
                say({"event": "saved", "path": path})
            except Exception as exc:  # noqa: BLE001
                say({"event": "error", "stage": "save", "error": repr(exc)})
                return 1
        elif parts[0] == "quit":
            say({"event": "bye"})
            return 0
    return 0


# ---------------------------------------------------------------------------
# quality (the conductor, float64 NumPy)
# ---------------------------------------------------------------------------

def _reg(y, p):
    return _tool("bench_board_more")._reg(y, p)


def _relfro(a, b):
    np = _np()
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / nb if nb else float(np.linalg.norm(a - b))


def _subspace(A, B):
    np = _np()
    qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.clip(sv, 0, 1).mean())


def _rbf_embedding_stress(X, Z):
    """Untimed RBF feature-distance loss, independent of opponent outputs.

    Kernel centering preserves pairwise distances. Truncating kernel PCA
    loses some of that distance; report the loss, not an accuracy guarantee.
    A fixed seeded pair sample bounds quality work on large query sets.
    """
    np = _np()
    X, Z = np.asarray(X, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    if (X.ndim != 2 or Z.ndim != 2 or len(X) != len(Z) or len(X) < 2
            or X.shape[1] == 0 or Z.shape[1] == 0
            or not np.isfinite(X).all() or not np.isfinite(Z).all()):
        raise ValueError("kernel PCA quality requires finite aligned query embeddings")
    rng = np.random.default_rng(SEED)
    i = rng.integers(0, len(X), 2048)
    j = (i + rng.integers(1, len(X), 2048)) % len(X)
    distances = ((X[i] - X[j]) ** 2).sum(1)
    target = -2 * np.expm1(-distances / X.shape[1])
    actual = ((Z[i] - Z[j]) ** 2).sum(1)
    denominator = float(np.linalg.norm(target))
    loss = float(np.linalg.norm(actual - target))
    return loss / denominator if denominator else loss


def _kernel_exact(kind, Xc, D):
    np = _np()
    X = Xc.astype(np.float64)
    if kind == "kernel-poly":
        g = 1.0 / D["X"].shape[1]
        return (g * (X @ X.T)) ** 2
    if kind == "kernel-achi2":
        num = 2 * X[:, None, :] * X[None, :, :]
        den = X[:, None, :] + X[None, :, :]
        return np.where(den > 0, num / np.where(den > 0, den, 1), 0).sum(-1)
    c = 1.0
    a = np.sqrt(X + c)
    return np.prod(2 * a[:, None, :] * a[None, :, :] / (X[:, None, :] + X[None, :, :] + 2 * c), axis=-1)


def _stable_topk_indices(d, k):
    """Exact stable argsort prefix without sorting every reference point.

    The cutoff's ties must be filled in original index order: an arbitrary
    argpartition prefix would change recall for duplicate points and masks.
    This is benchmark quality work, outside the measured GPU operation.
    """
    np = _np()
    k = min(max(k, 0), d.shape[1])
    truth = np.empty((d.shape[0], k), dtype=np.intp)
    if not k:
        return truth
    for i, row in enumerate(d):
        cutoff = np.partition(row, k - 1)[k - 1]
        if np.isnan(cutoff):
            truth[i] = np.argsort(row, kind="stable")[:k]
            continue
        below = np.flatnonzero(row < cutoff)
        tied = np.flatnonzero(row == cutoff)[:k - below.size]
        chosen = np.concatenate((below, tied))
        truth[i] = chosen[np.lexsort((chosen, row[chosen]))]
    return truth


def _recall(D, ind, k=KNN_K, allowed=None):
    np = _np()
    X = D["index"].astype(np.float64)
    Q = D["queries"].astype(np.float64)
    sq = (X * X).sum(1)
    hits = 0
    for s in range(0, Q.shape[0], 256):
        q = Q[s:s + 256]
        d = sq[None, :] - 2 * q @ X.T
        if allowed is not None:
            d[:, ~allowed] = np.inf
        truth = _stable_topk_indices(d, k)
        for i in range(q.shape[0]):
            hits += len(set(truth[i].tolist()) & set(ind[s + i][:k].tolist()))
    return hits / float(Q.shape[0] * k)


def _modularity(ip, ix, lab):
    np = _np()
    deg = np.diff(ip).astype(np.float64)
    m2 = deg.sum()
    src = np.repeat(np.arange(ip.shape[0] - 1), np.diff(ip))
    inside = float((lab[src] == lab[ix]).sum())
    tot = np.bincount(lab, weights=deg)
    return inside / m2 - float((tot / m2) ** 2 @ np.ones_like(tot))


def _exact_rfe_support(lane, D):
    """The selection an exactly solved RFE makes: the lane's step and n_features_to_select with
    LogisticRegression(solver='newton-cholesky', tol=1e-10) in float64 at the nested estimator's C.
    The lane's own LogisticRegression(max_iter=200) stops before convergence on wide blocks (istella:
    scikit-learn warns on every fit), so scikit-learn's selection is one inexact answer among others,
    not a reference (rfe-istella-20261012). Untimed scorer work, the same for every arm."""
    np = _np()
    from sklearn.feature_selection import RFE as _RFE
    from sklearn.linear_model import LogisticRegression as _LR
    p = _derived_params(lane, D, LANES[lane]["params"])
    est = p["estimator"]
    r = _RFE(_LR(solver="newton-cholesky", tol=1e-10, max_iter=500, C=float(est["kw"].get("C", 1.0))),
             n_features_to_select=p["n_features_to_select"], step=p["step"])
    r.fit(D["X"].astype(np.float64), D["y"])
    return np.asarray(r.support_, dtype=bool)


def transform_quality_reference(lane, D, outs, directory):
    """Untimed independent scorer for GPU-only transformation comparisons."""
    # FAIRNESS FIX (2026-10-13 audit): GPU-only selections omitted sklearn,
    # so deterministic transforms reported shapes without numerical quality.
    # Execute only the independent test oracle here, after all race workers
    # have stopped. No product work or opponent timing moves out of its clock.
    if quality_kind(lane) not in ("vs-sklearn", "labels", "multilabel") or "sklearn-cpu" in outs:
        return None
    if not outs:
        return None
    from pathlib import Path
    import sklearn
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    runner = build(lane, "sklearn-cpu", D)
    runner.fit()
    runner.infer()
    reference = runner.outputs()
    path = root / "reference.npz"
    _np().savez(path, **reference)
    receipt = dict(role="untimed independent quality oracle; never a timed race arm",
                   library="sklearn", version=sklearn.__version__, parameters=runner.record,
                   output=str(path), output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   inputs={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                                   sha256=hashlib.sha256(v.tobytes()).hexdigest())
                           for k, v in D.items() if isinstance(v, _np().ndarray)})
    (root / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    outs["sklearn-cpu"] = reference
    return receipt


def dropout_quality(x, y, probability):
    """Score a saved NCHW output without requiring identical random masks."""
    np = _np()
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 4 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("dropout quality requires aligned finite NCHW arrays")
    if not 0 <= probability <= 1:
        raise ValueError("invalid dropout probability")
    identifiable = np.any(x != 0, axis=(2, 3))
    count = int(identifiable.sum())
    if not count:
        raise ValueError("all-zero inputs cannot identify the dropout mask")
    dropped = np.all(y == 0, axis=(2, 3))
    fraction = float(dropped[identifiable].mean())
    target = np.zeros_like(x, dtype=np.float64)
    if probability < 1:
        target = x.astype(np.float64) / (1 - probability)
        target[dropped] = 0
    error = float(np.linalg.norm(y.astype(np.float64) - target)
                  / max(float(np.linalg.norm(target)), float(np.linalg.norm(x)), 1e-30))
    variance = probability * (1 - probability) / count
    deviation = (abs(fraction - probability) / variance ** .5 if variance
                 else 0. if fraction == probability else float("inf"))
    return dict(dropout_contract_relative_error=error, dropout_probability_z=deviation,
                dropped_channel_fraction=fraction, expected_dropout_probability=probability,
                identifiable_channels=count)


def quality(lane, D, outs):
    if LANES[lane]["kind"] == "extra":
        return _EXTRA.quality(lane, D, outs, globals())
    np = _np()
    more = _tool("bench_board_more")
    qk = quality_kind(lane)
    s = LANES[lane]
    ref = outs.get("sklearn-cpu")
    ref_arm = "scipy-cpu"                 # the function-call lanes' reference arm
    ref_fn = outs.get(ref_arm)
    q = {}
    exact = None                          # select-exact: the exact RFE selection, computed once
    dropout_input = None
    for arm, o in outs.items():
        e = {}
        try:
            if qk in ("clf", "semi"):
                yq = D["yq"].astype(np.float64)
                e["accuracy"] = float((o["pred"].reshape(-1) == yq).mean())
                if "proba1" in o and o["proba1"].ndim == 1 and set(np.unique(yq)) <= {0.0, 1.0}:
                    pr = np.clip(o["proba1"], 1e-15, 1 - 1e-15)
                    e["logloss"] = float(-np.mean(yq * np.log(pr) + (1 - yq) * np.log(1 - pr)))
            elif qk == "reg":
                e = _reg(D["yq"], o["pred"].reshape(-1))
            elif qk == "multiclf":
                P = o["pred"].reshape(D["Yq"].shape)
                e["accuracy"] = float((P == D["Yq"]).mean())
            elif qk == "multireg":
                P = o["pred"].reshape(D["Yq"].shape)
                e["r2"] = float(np.mean([_reg(D["Yq"][:, j], P[:, j])["r2"] for j in range(P.shape[1])]))
            elif qk == "outlier":
                flag = o["pred"].reshape(-1) < 0
                e["fraction_flagged"] = float(flag.mean())
                if ref is not None and ref["pred"].shape == o["pred"].shape:
                    rf = ref["pred"].reshape(-1) < 0
                    u = float((flag | rf).sum())
                    e["jaccard_vs_sklearn"] = float((flag & rf).sum()) / u if u else 1.0
            elif qk == "cluster":
                lab = o["labels"]
                X = D["X"]
                n = min(10_000, X.shape[0])
                e["n_clusters"] = int(np.unique(lab[lab >= 0]).shape[0])
                e["silhouette"] = more.silhouette(_stride(X, n), _stride(lab, n))
                if "ours" in outs and arm != "ours" and outs["ours"]["labels"].size == lab.size:
                    e["ari_vs_ours"] = more.adjusted_rand(outs["ours"]["labels"], lab)
            elif qk == "gmm":
                e["mean_log_likelihood"] = float(more.gmm_loglik(D["Xq"], o["weights"], o["means"],
                                                                 o["covariances"]).mean())
            elif qk == "embed":
                E = o["out"]
                e["trustworthiness_k15"] = (more.trustworthiness(D["X"], E) if np.all(np.isfinite(E))
                                            else None)
            elif qk == "impute":
                m = np.isnan(D["Xq"])
                e["masked_rmse"] = float(np.sqrt(np.mean((o["pred"][m] - D["Xq_true"][m]) ** 2)))
                if ref is not None and arm != "sklearn-cpu":
                    e["max_abs_diff_vs_sklearn"] = float(np.max(np.abs(o["pred"] - ref["pred"])))
            elif qk in ("select", "select-exact"):
                sup = o["support"].astype(bool)
                e["n_selected"] = int(sup.sum())
                if ref is not None:
                    r = ref["support"].astype(bool)
                    e["jaccard_vs_sklearn"] = float((sup & r).sum()) / max(1, float((sup | r).sum()))
                if qk == "select-exact":
                    if exact is None:
                        exact = _exact_rfe_support(lane, D)
                    r = exact
                    e["jaccard_vs_exact"] = float((sup & r).sum()) / max(1, float((sup | r).sum()))
            elif qk == "covariance":
                e["n_features"] = int(o["covariance"].shape[0])
                if ref is not None and arm != "sklearn-cpu":
                    e["rel_diff_vs_sklearn"] = _relfro(o["covariance"], ref["covariance"])
            elif qk in ("vs-sklearn", "labels", "multilabel"):
                P = o["pred"]
                e["output_shape"] = "x".join(map(str, P.shape))
                if ref is not None:
                    if ref["pred"].shape != P.shape:
                        raise ValueError("transform output shape differs from sklearn reference")
                    if not np.isfinite(P).all() or not np.isfinite(ref["pred"]).all():
                        raise ValueError("nonfinite transform output or sklearn reference")
                    e["max_abs_diff_vs_sklearn"] = float(np.max(np.abs(P - ref["pred"])))
                    e["rel_diff_vs_sklearn"] = _relfro(P, ref["pred"])
            elif qk == "radius":
                P = o["pred"].reshape(-1)
                e["neighbors_total"] = int(P.sum())
                if ref is not None and arm != "sklearn-cpu" and ref["pred"].shape == o["pred"].shape:
                    e["count_agreement_vs_sklearn"] = float((P == ref["pred"].reshape(-1)).mean())
            elif qk == "subspace":
                if lane == "kernel-pca":
                    e["rbf_feature_distance_relative_stress"] = _rbf_embedding_stress(D["Xq"], o["pred"])
                if ref is not None:
                    e["subspace_cos_vs_sklearn"] = _subspace(o["pred"], ref["pred"])
            elif qk == "pca":
                X = D["Xq"].astype(np.float64)
                Xc = X - X.mean(0)
                V, _ = np.linalg.qr(o["components"].T)
                e["explained_variance_fraction"] = float((Xc @ V).var(0).sum() / Xc.var(0).sum())
            elif qk == "distortion":
                X = D["Xq"].astype(np.float64)
                P = o["pred"]
                rng = np.random.default_rng(SEED)
                i, j = rng.integers(0, X.shape[0], (2, 2000))
                ok = i != j
                a = ((X[i] - X[j]) ** 2).sum(1)[ok]
                b = ((P[i] - P[j]) ** 2).sum(1)[ok]
                e["mean_abs_distortion"] = float(np.mean(np.abs(b / np.maximum(a, 1e-30) - 1)))
            elif qk == "nmf":
                X = D["X"].astype(np.float64)
                e["relative_reconstruction_error"] = float(np.linalg.norm(X - o["W"] @ o["components"])
                                                           / np.linalg.norm(X))
            elif qk == "ica":
                S_ = o["pred"]
                z = (S_ - S_.mean(0)) / np.maximum(S_.std(0), 1e-30)
                e["mean_abs_excess_kurtosis"] = float(np.mean(np.abs((z ** 4).mean(0) - 3)))
            elif qk == "fa":
                W, psi, mu = o["components"], o["noise_variance"], o["mean"]
                C = W.T @ W + np.diag(psi)
                X = D["Xq"].astype(np.float64) - mu
                L = np.linalg.cholesky(C)
                z = np.linalg.solve(L, X.T)
                e["mean_log_likelihood"] = float(np.mean(-0.5 * (z * z).sum(0))
                                                 - np.log(np.diag(L)).sum()
                                                 - 0.5 * X.shape[1] * np.log(2 * np.pi))
            elif qk == "cca":
                xs, ys = o["xs"], o["ys"]
                e["mean_canonical_corr"] = float(np.mean([np.corrcoef(xs[:, c], ys[:, c])[0, 1]
                                                          for c in range(xs.shape[1])]))
            elif qk == "recon":
                code, comp = o["pred"], o["components"]
                X = D["Xq"].astype(np.float64)
                mu = o.get("mean", np.zeros(X.shape[1]))
                e["relative_reconstruction_error"] = float(np.linalg.norm(X - mu - code @ comp)
                                                           / np.linalg.norm(X - mu))
                e["component_sparsity"] = float((comp == 0).mean())
            elif qk == "perplexity":
                theta = o["pred"] / np.maximum(o["pred"].sum(1, keepdims=True), 1e-300)
                beta = o["components"] / o["components"].sum(1, keepdims=True)
                Xq = D["Xq"].astype(np.float64)
                ll = 0.0
                for st in range(0, Xq.shape[0], 2048):
                    pr = theta[st:st + 2048] @ beta
                    ll += float((Xq[st:st + 2048] * np.log(np.maximum(pr, 1e-300))).sum())
                e["perplexity"] = float(np.exp(-ll / Xq.sum()))
            elif qk == "shape":
                P = o["pred"]
                e["output_columns"] = int(P.shape[1])
                e["nonzeros_per_row"] = float((P != 0).sum(1).mean())
            elif qk.startswith("kernel-") and qk != "kernel-shap":
                K = _kernel_exact(qk, D["Xq"], D)
                Z = o["pred"]
                e["kernel_rel_error"] = float(np.linalg.norm(Z @ Z.T - K) / np.linalg.norm(K))
            elif qk in ("forecast", "var"):
                hold = D["Yhold"].astype(np.float64)
                fc = o["forecast"]
                hold = hold[:fc.shape[0]]
                e["forecast_rmse"] = float(np.sqrt(np.mean((fc[:, :hold.shape[1]] - hold) ** 2)))
            elif qk == "garch":
                e["mean_llf"] = float(np.mean(o["llf"]))
            elif qk == "decompose":
                sm = outs.get("statsmodels-cpu")
                if sm is not None and arm != "statsmodels-cpu":
                    e["rel_diff_vs_statsmodels"] = _relfro(o["components"], sm["components"])
                resid = D["Yfit"].astype(np.float64) - o["components"]
                e["residual_std"] = float(resid.std())
            elif qk == "pagerank":
                nx_ = outs.get("networkx-cpu")
                e["sum"] = float(o["scores"].sum())
                if nx_ is not None and arm != "networkx-cpu":
                    e["l1_vs_networkx"] = float(np.abs(o["scores"] - nx_["scores"]).sum())
            elif qk == "components":
                e["n_components"] = int(np.unique(o["labels"]).shape[0])
                nx_ = outs.get("networkx-cpu")
                if nx_ is not None and arm != "networkx-cpu":
                    e["ari_vs_networkx"] = more.adjusted_rand(nx_["labels"], o["labels"])
            elif qk == "louvain":
                e["n_communities"] = int(np.unique(o["labels"]).shape[0])
                _, lab = np.unique(o["labels"], return_inverse=True)
                e["modularity"] = _modularity(D["indptr"], D["indices"], lab)
            elif qk == "dropout":
                if dropout_input is None:
                    NF = _tool("neural_fixtures")
                    fx = NF.load_for_lane(_NEURAL_FIXTURES or os.environ.get(NF.ENV_DIR),
                                          lane, s, s["datasets"][0], D)
                    dropout_input = fx["x"]
                e = dropout_quality(dropout_input, o["y"], s["params"]["p"])
            elif qk == "layer":
                r = outs.get("torch-eager-fp32")
                if "y" in o and r is not None and r["y"].shape == o["y"].shape:
                    if not np.isfinite(o["y"]).all() or not np.isfinite(r["y"]).all():
                        raise ValueError("nonfinite layer output or torch reference")
                    den = np.maximum(np.abs(r["y"].astype(np.float64)), 1e-6)
                    e["max_rel_diff_vs_torch_eager_fp32"] = float(np.max(np.abs(o["y"] - r["y"]) / den))
                    e["rel_fro_vs_torch_eager_fp32"] = _relfro(o["y"], r["y"])
            elif qk == "optim":
                r = outs.get("torch-eager-fp32")
                if r is not None:
                    if o["param"].shape != r["param"].shape or not np.isfinite(o["param"]).all() or not np.isfinite(r["param"]).all():
                        raise ValueError("invalid optimizer output or torch reference")
                    e["rel_fro_vs_torch_eager_fp32"] = _relfro(o["param"], r["param"])
                    # Compare the update, not just a large initial parameter
                    # vector that can make a missing optimizer step look close.
                    initial = np.random.default_rng(SEED).standard_normal(o["param"].size).astype(np.float32).reshape(o["param"].shape)
                    e["optimizer_update_error_vs_torch_fp32"] = _relfro(
                        o["param"].astype(np.float64) - initial,
                        r["param"].astype(np.float64) - initial)
            elif qk == "ann":
                allowed = (np.arange(D["index"].shape[0]) % 2 == 0) if s["task"] == "ivf-filter" else None
                e["recall_at_10"] = _recall(D, o["ind"], allowed=allowed)
            elif qk == "lu":
                pass                              # the residual needs A: recomputed below
            elif qk == "lstsq":
                X, y = D["X"].astype(np.float64), D["y"].astype(np.float64)
                b = o["x"][:X.shape[1]]
                e["relative_residual"] = float(np.linalg.norm(X @ b - y) / np.linalg.norm(y))
            elif qk == "rsvd":
                X = D["X"].astype(np.float64)
                V, _ = np.linalg.qr(o["components"].T)
                R = X - (X @ V) @ V.T
                e["relative_reconstruction_error"] = float(np.linalg.norm(R) / np.linalg.norm(X))
            elif qk == "chol":
                Lf = np.tril(o["L"])
                A = sym_system(Lf.shape[0]).astype(np.float64)
                e["relative_residual"] = float(np.linalg.norm(Lf @ Lf.T - A) / np.linalg.norm(A))
            elif qk == "eigh":
                A = sym_system(o["V"].shape[0]).astype(np.float64)
                w, V = o["w"], o["V"]
                e["relative_residual"] = float(np.linalg.norm(A @ V - V * w[None, :])
                                               / np.linalg.norm(A))
                w64 = np.linalg.eigvalsh(A)
                e["max_eigenvalue_error"] = float(np.max(np.abs(np.sort(w) - w64))
                                                  / np.max(np.abs(w64)))
            elif qk == "qr":
                X = D["X"].astype(np.float64)
                G = X.T @ X
                Rm = o["R"]
                e["relative_gram_difference"] = float(np.linalg.norm(Rm.T @ Rm - G) / np.linalg.norm(G))
                # FAIRNESS FIX (2026-10-13 audit): check both full scored
                # factors; R's Gram matrix alone never validated Q.
                Qm = o["Q"]
                e["relative_residual"] = float(np.linalg.norm(Qm @ Rm - X) / np.linalg.norm(X))
                identity = np.eye(Qm.shape[1])
                e["relative_orthogonality_error"] = float(
                    np.linalg.norm(Qm.T @ Qm - identity) / np.linalg.norm(identity))
            elif qk == "svd":
                X = D["X"].astype(np.float64)
                s64 = np.sqrt(np.maximum(np.linalg.eigvalsh(X.T @ X)[::-1], 0))
                k = min(s64.shape[0], o["S"].shape[0])
                e["max_rel_singular_value_error"] = float(
                    np.max(np.abs(o["S"][:k] - s64[:k]) / np.maximum(s64[:k], s64[0] * 1e-12)))
                Xr = _stride(X, o["U_rows"].shape[0])
                Rr = Xr - (o["U_rows"] * o["S"][None, :]) @ o["Vh"]
                e["relative_reconstruction_error_100k_rows"] = float(np.linalg.norm(Rr)
                                                                     / np.linalg.norm(Xr))
            elif qk == "als":
                _C, held = _als_split(D["X"])
                rows = np.nonzero(held >= 0)[0]
                U, V = o["U"].astype(np.float64), o["V"].astype(np.float64)
                hit = 0
                for st in range(0, rows.shape[0], 4096):
                    r = rows[st:st + 4096]
                    sc = U[r] @ V.T
                    sc[_C[r].toarray() > 0] = -np.inf
                    top = np.argpartition(-sc, 10, axis=1)[:, :10]
                    hit += int((top == held[r][:, None]).any(1).sum())
                e["recall_at_10"] = hit / float(max(1, rows.shape[0]))
            elif qk == "tree-shap":
                e["max_additivity_error"] = float(np.max(np.abs(o["phi"].sum(1) + o["base"] - o["margin"])))
            elif qk == "bootstrap":
                e.update(ci_low=float(o["ci"][0]), ci_high=float(o["ci"][1]),
                         standard_error=float(o["se"][0]))
                if ref_fn is not None and arm != ref_arm:
                    w = max(abs(float(ref_fn["ci"][1] - ref_fn["ci"][0])), 1e-30)
                    e["ci_endpoint_diff_over_width_vs_scipy"] = float(
                        np.max(np.abs(o["ci"] - ref_fn["ci"])) / w)
            elif qk == "permutation":
                e.update(statistic=float(o["stat"][0]), pvalue=float(o["p"][0]))
                if ref_fn is not None and arm != ref_arm:
                    e["abs_pvalue_diff_vs_scipy"] = float(abs(o["p"][0] - ref_fn["p"][0]))
            elif qk == "resample":
                X = D["X"].astype(np.float64)
                sd = np.maximum(X.std(0), 1e-12)
                e["max_mean_shift_over_std"] = float(np.max(np.abs(o["xmean"] - X.mean(0)) / sd))
            elif qk == "cv":
                e["mean_r2"] = float(np.mean(o["scores"]))
                if ref is not None and arm != "sklearn-cpu" and ref["scores"].shape == o["scores"].shape:
                    e["max_fold_score_diff_vs_sklearn"] = float(np.max(np.abs(o["scores"] - ref["scores"])))
            elif qk in ("kernel-shap", "permutation-shap"):
                e["rel_error_vs_exact"] = _relfro(o["phi"], o["exact"])
        except Exception as exc:  # noqa: BLE001
            e = {"error": repr(exc)[:300]}
        for key in ("n_iter", "n_steps"):       # FAIRNESS FIX (2026-10-13 audit): _ITER_LANES
            if key in o and isinstance(e, dict) and "error" not in e:
                e[key] = int(o[key].reshape(-1)[0])
        q[arm] = e
    if qk == "lu" and outs:
        n = next(iter(outs.values()))["x"].shape[0] // 64
        A, B = lu_system(n)
        for arm, o in outs.items():
            x = o["x"].reshape(n, 64)
            q[arm] = {"relative_residual": float(np.linalg.norm(A.astype(np.float64) @ x - B)
                                                 / np.linalg.norm(B))}
    return q


# ---------------------------------------------------------------------------
# race: the conductor for one (lane, dataset)
# ---------------------------------------------------------------------------

#: Per-arm environment a library documents as its own setting. implicit asks
#: for a single-threaded BLAS (its CPU solver threads itself); with OpenBLAS
#: on a box with more cores than its build limit its fit otherwise aborts.
ARM_ENV = {"implicit-cpu": {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
           "implicit-gpu": {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}


def _worker_env(arm):
    env = _tool("bench_board_more")._worker_env(arm)
    env.update(ARM_ENV.get(arm, {}))
    return env


def _none_on_both(BP, records, ref="ours"):
    # FAIRNESS FIX (2026-10-13 audit): None in two unrelated libraries is
    # not proof of equal behavior. Only named, source-backed EXCEPTIONS in
    # bench_board_params may explain an intentionally unset setting.
    return []


SCORER_REFUSED = "quality scorer unavailable: "
#: the label arrays each conductor-scored quality kind reads from the lane's block
_SCORER_DATA = {"clf": ("yq",), "semi": ("yq",), "reg": ("yq",), "multiclf": ("Yq",), "multireg": ("Yq",),
                "gmm": ("Xq",)}


class ScorerUnavailable(RuntimeError):
    """This race's quality scorer cannot run: the race is refused before any timed work."""


def scorer_preflight(lane, ds, arms, D, args, fit_calls):
    """THE SCORER PREFLIGHT (lane harness-evidence, 2026-10-14). Before any worker starts, prove the quality
    scorer of this race can run: the reference data it reads is present (the layer lanes' neural fixtures, the
    held-out labels), the scorer and its oracle import, and the scorer runs on known data -- the held-out labels
    scored as a prediction (clf/semi/reg), the untimed scikit-learn oracle scored against itself (vs-sklearn), the
    own-host reference (layer/optim, the scorer of every ours-only race) run in full on the race's inputs and
    compared with itself. The two references depend on the inputs only, never on a timed output, so they are
    computed here, once, and reused after the timed rounds: nothing runs twice and no clock moves.

    Returns (receipt, precomputed): `precomputed` holds {"host": (outputs, receipt path)} and/or
    {"sklearn": (outputs, receipt)}. Raises ScorerUnavailable naming the failed check. Motivating case: AMD
    gcn/graphsage istella+taxi on main-amd-mi325x were timed, then the own-host reference crashed with
    FixturesMissing and the cells still said "ok"."""
    np = _np()
    s = LANES[lane]
    kind, qk = s["kind"], quality_kind(lane)
    receipt = {"lane": lane, "dataset": ds, "quality_kind": qk, "checks": []}
    precomputed = {}

    def check(name, fn):
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001
            receipt["checks"].append({"check": name, "status": "failed", "detail": repr(exc)[:600]})
            receipt["status"] = "refused"
            raise ScorerUnavailable("%s: %s" % (name, str(exc)[:600] or repr(exc)))
        receipt["checks"].append({"check": name, "status": "ok", "detail": detail})
        return detail

    def imports():
        if kind == "extra":
            return {"scorer": "tools/bench_board_extra.py quality", "numpy": np.__version__}
        _tool("bench_board_more")
        return {"scorer": "tools/bench_board_algos.py quality", "numpy": np.__version__}
    check("scorer imports", imports)

    def labels():
        need = _SCORER_DATA.get(qk, ()) if kind != "extra" else ()
        absent = [k for k in need if D.get(k) is None or not np.size(D[k])]
        if absent:
            raise ValueError("the block holds no %s for the %s scorer" % (", ".join(absent), qk))
        return {k: list(np.shape(D[k])) for k in need}
    check("reference data", labels)

    ours_racing = any(a in OURS_ARMS for a in arms)
    if kind == "layer" and (ours_racing or needs_host_reference(lane, arms)):
        def fixtures():
            NF = _tool("neural_fixtures")
            fixdir = (getattr(args, "neural_fixtures", None) or os.environ.get(NF.ENV_DIR)
                      or NF.default_dir(args.data))
            return NF.provenance(NF.load_for_lane(fixdir, lane, s, ds, D))
        check("neural fixtures", fixtures)

    if qk in ("clf", "semi", "reg") and kind != "extra":
        def labels_as_prediction():
            yq = np.asarray(D["yq"])
            got = quality(lane, D, {"preflight": {"pred": yq.astype(np.float64)}}).get("preflight") or {}
            if got.get("error") or not any(k in got for k in ("accuracy", "r2", "rmse")):
                raise ValueError("the %s scorer did not score the held-out labels: %r" % (qk, got))
            return {k: v for k, v in got.items() if isinstance(v, (int, float))}
        check("scorer self-test", labels_as_prediction)

    if qk in ("vs-sklearn", "labels", "multilabel") and "sklearn-cpu" not in arms and s.get("sk"):
        def oracle():
            outs = {"preflight": None}
            got = transform_quality_reference(lane, D, outs, os.path.join(args.work, "%s-%s-sklearn-quality"
                                                                          % (lane, ds)))
            if got is None or outs.get("sklearn-cpu") is None:
                raise ValueError("the scikit-learn oracle produced no reference")
            ref = outs["sklearn-cpu"]
            scored = quality(lane, D, {"sklearn-cpu": ref, "preflight": ref}).get("preflight") or {}
            if scored.get("error"):
                raise ValueError("the scorer failed on the oracle's own output: %s" % scored["error"])
            precomputed["sklearn"] = (ref, got)
            return {"oracle": got.get("library"), "version": got.get("version"),
                    "self_score": {k: v for k, v in scored.items() if isinstance(v, (int, float))}}
        check("scikit-learn oracle", oracle)

    if needs_host_reference(lane, arms):
        def host():
            hq = _tool("bench_board_host_quality")
            outputs, path = hq.reference(lane, D, args.ours_python,
                                         os.path.join(args.work, "%s-%s-host-quality" % (lane, ds)),
                                         args.round_seconds, fit_calls=fit_calls)
            self_score = hq.compare(outputs, outputs)
            precomputed["host"] = (outputs, path)
            return {"receipt": path, "fit_calls": fit_calls, "self_score": self_score}
        check("own host reference", host)
    receipt["status"] = "ok"
    return receipt, precomputed


def race(args):
    ab_config = _configure_neural_ab(getattr(args, "neural_ab_config", None), args.lane)
    if ab_config is not None and args.smoke_rows:
        raise ValueError("neural A/B recipes require full saved workload, without --smoke-rows")
    np = _np()
    ctd = _tool("classical_two_datasets")
    lane, ds = args.lane, args.dataset
    arms = [a for a in args.arms.split(",") if a]
    if ab_config is not None and "ours-fast" in arms:
        raise ValueError("neural A/B configuration cannot relabel an ours-fast worker")
    _tool("bench_board_probe").refuse_our_cpu_arms(arms, "bench_board_algos")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    if not getattr(args, "params_only", False):
        rc = host_reference_preflight(args.lane, args.dataset, arms, args.work)
        if rc is not None:
            return rc
    frozen_config = None
    if ab_config is not None:
        frozen_config = os.path.abspath(os.path.join(args.out, "neural-ab-config.json"))
        with open(frozen_config, "w", encoding="utf-8") as fh:
            json.dump(ab_config, fh, indent=2, sort_keys=True)
    if args.smoke_rows:           # the conductor builds the same seeded arrays as its workers
        os.environ["MOJOLEARN_ALGOS_SMOKE_ROWS"] = str(args.smoke_rows)
    B, rec = _load_block(lane, ds, args.data)
    D = lane_arrays(lane, B)
    shape = "; ".join("%s %s" % (k, "x".join(str(s) for s in np.shape(v)))
                      for k, v in sorted(D.items()) if hasattr(v, "shape"))
    result = {"lane": lane, "dataset": ds, "block": rec, "shape": shape, "arms": {},
              "lane_config": lane_config(lane), "rounds_requested": args.rounds,
              "started": now_utc(), "script": "tools/bench_board_algos.py",
              "commit": os.environ.get("MOJOLEARN_REPO_COMMIT", "unknown")}
    if ab_config is not None:
        result["neural_ab_config"] = ab_config
        result["neural_ab_binary_admission"] = "requested source flags only; identity/quality/timing acceptance separate"
    tag = "%s-%s" % (lane, ds)
    # THE SCORER PREFLIGHT (lane harness-evidence, 2026-10-14): before any worker starts. The layer lanes' fixture
    # dir and the dataset reach the untimed references through the environment, as below for the workers.
    os.environ["MOJOLEARN_BOARD_DATASET"] = ds
    if LANES[lane]["kind"] == "layer":
        os.environ[_tool("neural_fixtures").ENV_DIR] = os.path.abspath(
            getattr(args, "neural_fixtures", None) or _tool("neural_fixtures").default_dir(args.data))
    external_warmup_planned = (os.environ.get("MOJOLEARN_AFT_CAPTURE") == "1"
                               and os.environ.get("MOJOLEARN_AFT_EXTERNAL_WARMUP") == "1"
                               and _aft_full_tree_workload(lane))
    scorer_pre = {}
    if not getattr(args, "params_only", False):
        try:
            result["scorer_preflight"], scorer_pre = scorer_preflight(
                lane, ds, arms, D, args, args.rounds + (0 if external_warmup_planned else 1))
        except ScorerUnavailable as exc:
            why = SCORER_REFUSED + str(exc)
            result["scorer_preflight"] = {"lane": lane, "dataset": ds, "status": "refused", "reason": why}
            for arm in arms:
                result["arms"][arm] = {"command": None, "warmup_ms": None, "ms": [], "infer_ms": [],
                                       "infer_warmup_ms": None, "digests": [], "mem": [],
                                       "status": "REFUSED", "error": why}
                print("ALGOS-REFUSED lane=%s dataset=%s arm=%s stage=scorer-preflight detail=%s"
                      % (lane, ds, arm, json.dumps(why)), flush=True)
            result["quality"] = {"error": why, "scorer": "failed"}
            result["finished"] = now_utc()
            out_json = os.path.join(args.out, "%s.json" % tag)
            with open(out_json + ".tmp", "w") as fh:
                json.dump(result, fh, indent=2, sort_keys=True, default=str)
            os.replace(out_json + ".tmp", out_json)
            return 1
    workers = {}
    env_extra = {}
    if args.smoke_rows:
        env_extra["MOJOLEARN_ALGOS_SMOKE_ROWS"] = str(args.smoke_rows)
    arm_python = dict(a.split("=", 1) for a in (args.arm_python or []))
    # FAIRNESS FIX (2026-10-13 audit): the RNN/LSTM/GRU and CNN torch twins started from numpy
    # weights and row orders that are not ours' (see _read_ours_init). Ours' initial weights
    # and orders are now exported once, untimed, under OUR interpreter, and every torch arm
    # loads them; when the export fails the torch arms are refused by name (no worker starts).
    # Results raced before this change were wrong for this lane; apply extra scrutiny.
    refused_before, ours_init_path = {}, None
    torch_arms = [a for a in arms if a.startswith("torch-")]
    if LANES[lane]["kind"] in OURS_INIT_KINDS and torch_arms:
        ours_init_path, why = _export_ours_init(args, lane, ds, tag, env_extra)
        result["ours_init"] = {"path": ours_init_path, "refused": why}
        if ours_init_path is None:
            refused_before = {a: why for a in torch_arms}
    for arm in arms:
        if arm in refused_before:
            workers[arm] = _tool("bench_board_neural").RefusedArm()
            result["arms"][arm] = {"command": None, "warmup_ms": None, "ms": [], "infer_ms": [],
                                   "infer_warmup_ms": None, "digests": [], "mem": [],
                                   "status": "refused", "error": refused_before[arm]}
            print("ALGOS-REFUSED lane=%s dataset=%s arm=%s stage=plan detail=%s"
                  % (lane, ds, arm, json.dumps(refused_before[arm])), flush=True)
            continue
        py = arm_python.get(arm) or (args.ours_python if arm in OURS_ARMS else args.theirs_python)
        cmd = shlex.split(py) + [os.path.abspath(__file__), "worker", "--arm", arm, "--lane", lane,
                                 "--dataset", ds, "--data", args.data]
        if ours_init_path is not None and arm.startswith("torch-"):
            cmd += ["--ours-init", ours_init_path]
        if frozen_config is not None:
            cmd += ["--neural-ab-config", frozen_config]
        if LANES[lane]["kind"] == "layer":    # the ours arms read x/state/dy from here; torch checks
            fixtures_dir = os.path.abspath(
                getattr(args, "neural_fixtures", None) or _tool("neural_fixtures").default_dir(args.data))
            cmd += ["--neural-fixtures", fixtures_dir]
            # the untimed own-host reference (bench_board_host_quality.enrich) is a child of THIS driver process and
            # reads the fixtures through the environment; the worker's own export (worker(), a1c96a5ba) never reaches
            # it, so every layer lane's quality cell still said "neural fixtures missing" (2026-10-11, nv2 v0082-v0121)
            os.environ[_tool("neural_fixtures").ENV_DIR] = fixtures_dir
        if getattr(args, "full_tree_workload", False):
            raise ValueError("integration benchmark freeze forbids --full-tree-workload; retain the saved race")
            cmd.append("--full-tree-workload")
        env = _worker_env(arm)
        env.update(env_extra)
        workers[arm] = ctd.Worker(arm, cmd, env, os.path.join(args.out, "%s-%s.log" % (tag, arm)), REPO)
        result["arms"][arm] = {"command": cmd, "warmup_ms": None, "ms": [], "infer_ms": [],
                               "infer_warmup_ms": None, "digests": [], "mem": [], "status": "ok"}
    for arm, w in workers.items():
        if arm in refused_before:          # FAIRNESS FIX (2026-10-13 audit): refused above, no worker
            continue
        msg = w.read(args.ready_seconds)
        if msg is not None and msg.get("event") == "skipped":
            w.alive = False
            w.close()
            result["arms"][arm].update(status="skipped", error=msg.get("reason"))
            print("ALGOS-SKIPPED lane=%s dataset=%s arm=%s %s" % (lane, ds, arm, msg.get("reason")),
                  flush=True)
            continue
        if msg is None or msg.get("event") != "ready":
            w.kill("not_ready", msg)
            result["arms"][arm].update(status="not_ready", error=msg)
            print("ALGOS-REFUSED lane=%s dataset=%s arm=%s stage=ready detail=%s"
                  % (lane, ds, arm, json.dumps(msg)), flush=True)
            continue
        w.info = msg["info"]
        result["arms"][arm]["info"] = msg["info"]
        result["arms"][arm]["params_record"] = msg.get("params_record")
    # THE PARAMETER CHECK (tools/bench_board_params.py), before the first timed
    # round: same seed, same tuning parameters, read back from what each worker
    # constructed. A refusal fails the race by name (exit 3, params_refused in
    # the JSON, the BOARD-PARAMS-REFUSED line in this log).
    if getattr(args, "params_only", False):
        return _tool("classical_two_datasets").params_only_exit(
            result, workers, arms, "algos/" + lane, "algos",
            os.path.join(args.out, tag + ".params.json"))
    BP = _BP()
    records = {a: result["arms"][a]["params_record"] for a in arms
               if workers[a].alive and result["arms"][a].get("params_record")}
    lane_id = "algos/" + lane
    extra = _none_on_both(BP, records)
    try:
        result["params_check"] = BP.enforce(lane_id, records, family="algos", extra_exceptions=extra)
    except BP.ParamsRefused as exc:
        result["params_check"] = BP.check(lane_id, records, family="algos", extra_exceptions=extra)
        result["params_refused"] = str(exc)
        for arm in arms:
            if workers[arm].alive:
                workers[arm].kill("params_refused", None)
            if result["arms"][arm]["status"] == "ok":
                result["arms"][arm].update(status="params_refused", error=str(exc)[:2000])
        result["finished"] = now_utc()
        out_json = os.path.join(args.out, "%s.json" % tag)
        with open(out_json + ".tmp", "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, default=str)
        os.replace(out_json + ".tmp", out_json)
        print("ALGOS-PARAMS-REFUSED lane=%s dataset=%s %s" % (lane, ds, str(exc)[:2000]), flush=True)
        return 3
    external_warmup = (
        os.environ.get("MOJOLEARN_AFT_CAPTURE") == "1"
        and os.environ.get("MOJOLEARN_AFT_EXTERNAL_WARMUP") == "1"
        and _aft_full_tree_workload(lane)
    )
    if os.environ.get("MOJOLEARN_AFT_CAPTURE") == "1":
        result["aft_warmup_policy"] = (
            "outer queue supplies excluded warmup in a fresh isolated process; no internal warmup"
            if external_warmup else "one internal excluded warmup"
        )
    for r in range(1 if external_warmup else 0, args.rounds + 1):
        live = [a for a in arms if workers[a].alive]
        if not live:
            break
        shift = r % len(live)
        for arm in live[shift:] + live[:shift]:
            w = workers[arm]
            w.send("round %d" % r)
            # FAIRNESS FIX (2026-10-13 audit): a torch.compile arm compiles in its untimed warm-up
            # and gets its own longer deadline (bench_board_neural.warmup_deadline); its warm-up
            # is recorded as compile_warmup_ms. Results raced before this change were wrong for
            # this lane; apply extra scrutiny.
            deadline = _tool("bench_board_neural").warmup_deadline(arm, args) if r == 0 else args.round_seconds
            msg = w.read(deadline)
            if msg is None or msg.get("event") != "round":
                status = "timeout" if msg is None else "error"
                w.kill(status, msg)
                result["arms"][arm].update(status=status, error=msg, failed_round=r)
                if r == 0 and status == "timeout":
                    result["arms"][arm]["warmup_deadline_s"] = deadline
                print("ALGOS-REFUSED lane=%s dataset=%s arm=%s stage=round%d detail=%s"
                      % (lane, ds, arm, r, json.dumps(msg)), flush=True)
                continue
            a = result["arms"][arm]
            if r == 0:
                a["warmup_ms"], a["infer_warmup_ms"] = msg["ms"], msg.get("infer_ms")
                if "-compile-" in arm:
                    a["compile_warmup_ms"], a["warmup_deadline_s"] = msg["ms"], deadline
            else:
                a["ms"].append(msg["ms"])
                if msg.get("infer_ms") is not None:
                    a["infer_ms"].append(msg["infer_ms"])
            a["digests"].append(msg["digest"])
            if msg.get("operation") is not None:
                a.setdefault("operations", []).append(dict(msg["operation"], round=r, warmup=r == 0))
            if msg.get("state_receipt") is not None:
                a.setdefault("state_receipts", []).append(msg["state_receipt"])
            if msg.get("aft_capture") is not None:
                a.setdefault("aft_captures", []).append(msg["aft_capture"])
            if msg.get("upload_separate") is not None:
                a.setdefault("upload_separate", []).append(dict(msg["upload_separate"], round=r,
                                                                 warmup=r == 0))
            a["mem"].append(msg.get("mem"))
            print("ALGOS-ROUND lane=%s dataset=%s arm=%s round=%d ms=%.3f infer_ms=%s digest=%s"
                  % (lane, ds, arm, r, msg["ms"], msg.get("infer_ms"), msg["digest"]), flush=True)
    outs = {}
    for arm in arms:
        w = workers[arm]
        if w.alive and len(result["arms"][arm]["ms"]) == args.rounds:
            path = os.path.join(args.work, "%s-%s.npz" % (tag, arm))
            w.send("save %s" % path)
            msg = w.read(args.round_seconds)
            if msg is not None and msg.get("event") == "saved":
                with np.load(path) as z:
                    outs[arm] = {k: z[k] for k in z.files}
                result.setdefault("saved_outputs", {})[arm] = path
            else:
                result["arms"][arm].update(status="save_failed", error=msg)
        if w.alive:
            w.close()
    try:
        if "sklearn" in scorer_pre and outs and "sklearn-cpu" not in outs:
            # the scorer preflight's oracle, computed before the timed rounds from the same inputs
            outs["sklearn-cpu"], result["quality_reference_receipt"] = scorer_pre["sklearn"]
        else:
            result["quality_reference_receipt"] = transform_quality_reference(
                lane, D, outs, os.path.join(args.work, tag + "-sklearn-quality"))
        result["quality"] = quality(lane, D, outs)
        hq = _tool("bench_board_host_quality")
        hq_dir = os.path.join(args.work, tag + "-host-quality")
        # the host-quality child rebuilds the lane through board.build() with _DATASET unset; the layer lanes key their
        # fixtures by (lane, dataset, variant), so without this the child asked for "<lane>/None/nb64" (nv2 v0137, amd a1719)
        os.environ["MOJOLEARN_BOARD_DATASET"] = ds
        result["host_quality_receipt"] = hq.enrich(
            lane, D, outs, result["quality"], args.ours_python, hq_dir, args.round_seconds,
            fit_calls=args.rounds + (0 if external_warmup else 1), precomputed=scorer_pre.get("host"))
        # a host twin the reference could not load (one beyond the preflight's, e.g. a second binding the
        # lane's class imports): name it so the box job builds it and races again
        if any("Own host reference failed" in str((result["quality"].get(a) or {}).get("error", ""))
               for a in arms if a in OURS_ARMS):
            for basename in hq.missing_host_bindings(os.path.join(hq_dir, "host.log")):
                _say_needs_host(lane, ds, basename, "after the timed rounds")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        # the scorer itself crashed: every arm's quality is "not scored", never a quality failure of an arm
        result["quality"] = {"error": repr(exc), "scorer": "failed"}
    # FAIRNESS FIX (2026-10-13 audit): worker() saves outputs AFTER the scored
    # runner.infer() and quality() reads those exact bytes. The inference board
    # used to drop this already measured quality and publish an empty dict.
    infer = {"call": infer_call(lane), "batch": "Xq", "rows": None, "arms": {},
             "quality": result["quality"],
             "quality_basis": "same scored post-inference outputs as fit quality; saved_outputs retains bytes"}
    for arm in arms:
        a = result["arms"][arm]
        ok = a["status"] == "ok" and len(a["ms"]) == args.rounds
        a["median_ms"] = statistics.median(a["ms"]) if ok else None
        timed = a["digests"][0 if external_warmup else 1:]
        a["digest_stable"] = (len(set(timed)) == 1) if ok and len(timed) >= 2 else None
        info = a.get("info") or {}
        a["span"] = arm_span(lane, arm, a, ok)
        if has_infer(lane):
            ist = a["status"] if a["status"] != "ok" or a["infer_ms"] else "error"
            infer["arms"][arm] = {"ms": list(a["infer_ms"]), "warmup_ms": a["infer_warmup_ms"],
                                  "status": ist, "error": a.get("error") if ist != "ok" else None,
                                  "digests": a["digests"], "digest_stable": a["digest_stable"],
                                  "info": info}
        print("ALGOS lane=%s dataset=%s arm=%s status=%s median_ms=%s quality=%s"
              % (lane, ds, arm, a["status"], a["median_ms"],
                 json.dumps(result["quality"].get(arm, {}), sort_keys=True, default=str)), flush=True)
    if has_infer(lane):
        result["infer"] = infer
    # FAIRNESS FIX (2026-10-13 audit): on the neural-network lanes (layers, RNN/LSTM/GRU, CNN,
    # GCN, GraphSAGE), when a torch compile arm timed out, the other torch arms stayed ordinary
    # opponents (another setting silently promoted). They now carry `not_opponent` (board mode
    # "context", no ratio). Results raced before this change were wrong for this lane; apply
    # extra scrutiny.
    if LANES[lane]["kind"] in NEURAL_KINDS:
        BBN = _tool("bench_board_neural")
        result["opponent_rule"] = BBN.OPPONENT_RULE
        for arm, why in sorted(BBN.mark_not_opponents(lane, result["arms"]).items()):
            print("ALGOS-NOT-OPPONENT lane=%s dataset=%s arm=%s %s" % (lane, ds, arm, why), flush=True)
    result["finished"] = now_utc()
    out_json = os.path.join(args.out, "%s.json" % tag)
    with open(out_json + ".tmp", "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    os.replace(out_json + ".tmp", out_json)
    ran = [a for a in arms if result["arms"][a]["status"] != "skipped"]
    failed = [a for a in ran if result["arms"][a]["status"] != "ok"]
    return 1 if ran and len(failed) == len(ran) else 0


def arm_span(lane, arm, a, ok):
    """What was inside one arm's clock, from its ready record; OUR arm also carries the
    separate upload clock (`upload_span`). No timing is affected."""
    info = a.get("info") or {}
    # A cuML/cuVS/cuGraph arm set up by ctd._cuml_setup carries no input_home but
    # records the pre-clock upload: its inputs were on the device before its clock
    # (board-two-clocks, 2026-10-08; the old `or "host"` labelled 58 NVIDIA cells host).
    home = info.get("input_home") or ("device" if info.get("upload_ms_untimed") is not None
                                      else "host")
    span = {"input_home": home, "pre_clock_fit": bool(info.get("pre_clock_fit")),
            "upload_ms_untimed": info.get("upload_ms_untimed"),
            "inside_clock": fit_text(lane)}
    if arm in OURS_ARMS:
        span.update(upload_span(a, ok))
    return span


def upload_span(a, ok):
    """THE SEPARATE UPLOAD CLOCK in OUR arm's span (`_ours_upload_probe`): the median of the
    scored rounds' separate uploads beside the scored median, and the DERIVED difference
    (tools/board_clock_audit.py reads `upload_ms_separate` for our kernel clock). The scored
    median is untouched. Opponent spans never carry these fields."""
    recs = [u for u in a.get("upload_separate", []) if not u.get("warmup")]
    ups = [u["ms"] for u in recs if "ms" in u]
    up = statistics.median(ups) if ups else None
    why = None
    if up is None:
        why = next((u["unavailable"] for u in recs if u.get("unavailable")), "no probe record")
    return {"upload_ms_separate": up,
            "upload_bytes_separate": next((u["bytes"] for u in recs if "bytes" in u), None),
            "upload_inputs_separate": next((u["inputs"] for u in recs if "inputs" in u), None),
            "upload_scope_separate": "host_to_device_upload_separate" if up is not None else None,
            "upload_separate_unavailable": why,
            "fit_minus_upload_ms": ((a["median_ms"] - up)
                                    if (ok and up is not None and a.get("median_ms") is not None)
                                    else None),
            "fit_minus_upload_ms_source": "derived: scored median - separate upload median"}


def build_parser():
    p = argparse.ArgumentParser(prog="bench_board_algos", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prep")
    pr.add_argument("--data", required=True)
    pr.add_argument("--lanes", default=",".join(LANE_ORDER))
    pr.add_argument("--datasets", default="taxi,istella")
    pr.add_argument("--max-rows", type=int, default=0)
    pr.add_argument("--full-tree-workload", action="store_true",
                    help="tree-only: save all existing taxi/istella split rows in separate aft-full cls/reg blocks")
    r = sub.add_parser("race")
    r.add_argument("--lane", required=True, choices=LANE_ORDER)
    r.add_argument("--dataset", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--arms", required=True)
    r.add_argument("--neural-ab-config", help="selected neural A/B JSON; applies runtime recipe without compiling or admitting the binary")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--smoke-rows", type=int, default=0)
    r.add_argument("--neural-fixtures", default=None,
                   help="the layer lanes' fixture dir (tools/neural_fixtures.py); default "
                        "$MOJOLEARN_NEURAL_FIXTURES, else <dirname(--data)>/neural-fixtures")
    r.add_argument("--full-tree-workload", action="store_true",
                   help="tree-only: use aft-full inputs, no lane row subsets, full TreeSHAP background")
    r.add_argument("--ours-python", default=sys.executable)
    r.add_argument("--theirs-python", default=sys.executable)
    r.add_argument("--arm-python", action="append", default=[], metavar="ARM=PY",
                   help="this arm's own interpreter (repeatable; ARM_VENVS)")
    r.add_argument("--ready-seconds", type=float, default=1800)
    r.add_argument("--params-only", action="store_true",
                   help="construct every arm, read its parameters back, write <out>/<tag>.params.json "
                        "and stop before the warm-up (the board's opponent-store lookup)")
    r.add_argument("--warmup-seconds", type=float, default=1800)
    # FAIRNESS FIX (2026-10-13 audit): a torch.compile arm's own warm-up deadline. Results raced
    # before this change were wrong for this lane; apply extra scrutiny.
    r.add_argument("--compile-warmup-seconds", type=float, default=None,
                   help="warm-up deadline of a torch.compile arm; default 8 x --round-seconds")
    r.add_argument("--round-seconds", type=float, default=1800)
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True)
    w.add_argument("--lane", required=True, choices=LANE_ORDER)
    w.add_argument("--dataset", required=True)
    w.add_argument("--data", required=True)
    w.add_argument("--neural-ab-config", help="frozen neural A/B JSON from conductor")
    w.add_argument("--neural-fixtures", default=None, help="the layer lanes' fixture dir (from the conductor)")
    # FAIRNESS FIX (2026-10-13 audit): ours' exported initial weights and row orders for the
    # RNN/LSTM/GRU and CNN torch twins (the conductor runs `ours-init`). Results raced before
    # this change were wrong for this lane; apply extra scrutiny.
    w.add_argument("--ours-init", default=None,
                   help="ours' initial weights and row orders (npz from `ours-init`, the conductor)")
    w.add_argument("--full-tree-workload", action="store_true")
    oi = sub.add_parser("ours-init", help="write ours' initial weights and per-epoch row orders for "
                                          "an RNN/LSTM/GRU or CNN race (OUR interpreter, untimed)")
    oi.add_argument("--lane", required=True, choices=LANE_ORDER)
    oi.add_argument("--dataset", required=True)
    oi.add_argument("--data", required=True)
    oi.add_argument("--out", required=True)
    sub.add_parser("table")
    b = sub.add_parser("builds", help="print the BUILDS= list (bindings/build_*.sh, device + host twins) a race "
                                      "of these lanes needs, e.g. for `lq add nv RACE main <lanes> <ds> BUILDS=...`")
    b.add_argument("--lanes", required=True, help="comma list of lanes")
    b.add_argument("--arms", default="ours", help="the race's arms (default ours: the board default)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "full_tree_workload", False):
        raise ValueError("integration benchmark freeze forbids --full-tree-workload; retain the saved race")
        if getattr(args, "smoke_rows", 0) or getattr(args, "max_rows", 0):
            raise ValueError("full-tree-workload is incompatible with row caps")
        if os.environ.get("MOJOLEARN_ALGOS_SMOKE_ROWS"):
            raise ValueError("full-tree-workload refuses MOJOLEARN_ALGOS_SMOKE_ROWS")
        if args.cmd in ("worker", "race") and args.lane not in AFT_FULL_TREE_LANES:
            raise ValueError("full-tree-workload accepts the named tree estimator lanes only")
        os.environ["MOJOLEARN_AFT_FULL_TREE_WORKLOAD"] = "1"
    if args.cmd == "prep":
        return prep(args)
    if args.cmd == "worker":
        return worker(args)
    if args.cmd == "ours-init":            # FAIRNESS FIX (2026-10-13 audit): see _read_ours_init
        return ours_init_main(args)
    if args.cmd == "table":
        print(json.dumps({l: lane_config(l) for l in LANE_ORDER}, indent=1, sort_keys=True))
        return 0
    if args.cmd == "builds":
        out = []
        for lane in [x for x in args.lanes.split(",") if x]:
            if lane not in LANES:
                raise SystemExit("unknown lane %r" % lane)
            for bname in race_builds(lane, [a for a in args.arms.split(",") if a]):
                if bname not in out:
                    out.append(bname)
        print("BUILDS=" + ",".join(out))
        return 0
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    return race(args)


if __name__ == "__main__":
    sys.exit(main())
