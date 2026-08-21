"""Gradient-boosted oblivious trees on the GPU, mirroring CatBoost.

**THE DEFAULTS ARE CatBoost's, NOT scikit-learn's**, and several of them
change results rather than just speed:

    learning_rate   CatBoost 0.03      sklearn 0.1
    n_estimators    CatBoost 100       sklearn 100
    max_depth       CatBoost 6         sklearn 3
    l2_leaf_reg     CatBoost 3.0       sklearn (none)
    border_count    CatBoost 128       sklearn (none)

The tree shape is OBLIVIOUS (symmetric): every node at a level takes the same
split, which is CatBoost's structure and not scikit-learn's. A comparison
against `GradientBoostingRegressor` at matched hyperparameters is still
comparing two different tree families.

X IS ROW-MAJOR HERE AND COLUMN-MAJOR INSIDE. `fit` transposes once. On a
800,000 x 100 matrix that is a 320 MB copy, and it is reported rather than
hidden -- pass `X` already in Fortran order to avoid it.

THE LOSS PICKS THE LEAF ESTIMATOR, and that is CatBoost's decision, not this
wrapper's. `leaf_estimation_method=None` (the default) means "let the loss
decide", which reproduces `SetLeavesEstimationDefault`
(`catboost_options.cpp:273-360`): Newton at one iteration for RMSE, Newton at
ten for Logloss, Newton at TWENTY for Tweedie, Gradient for LogLinQuantile and
for Lq below q=2, and the EXACT weighted-quantile estimator for MAE, MAPE and
Quantile. Overriding it means overriding CatBoost.

PREDICTIONS ARE RAW SCORES FOR EVERY LOSS, exactly as CatBoost's `predict`
without a `prediction_type` is. A `Logloss` model's `predict` returns the
logit; a `Poisson` model's returns the log-rate. This is deliberate: the
number this library benchmarks is the number it returns. `predict_proba`
applies the link where one exists, and `predict_classes` takes its argmax --
separate methods rather than a `predict` that means different things for
different losses.

MULTICLASS IS MULTI-OUTPUT AND ITS LAST CLASS IS NOT STORED. A softmax over
`k` free approxes is over-parameterized -- adding a constant to all of them
changes nothing -- so CatBoost pins the last class's approx at zero and
carries `k - 1`. Therefore:

    MultiClass          predict          (n, n_classes - 1)  free approxes
                        predict_proba    (n, n_classes)      softmax
                        predict_classes  (n,)                class codes

`MultiClassOneVsAll` is implemented and gated in the Mojo layer but is NOT
reachable from Python: its kernels do not fit in the CPython extension under
any basename `bindings/build.sh` has measured (PORTING.md 70). Asking for it
raises with that reason.

Dropping MultiClass's last probability column and renormalising the rest
gives a different and wrong answer: the pinned class is a real class whose
approx happens to be zero. Labels are dense codes `0..k-1` for both, and the
class count is derived from them, as their `TClassificationTargetHelper`
derives it.
"""

import numpy as np

from . import _mojolearn
from ._arrays import _addr, _addr_ro, as_f32_c

#: CatBoost's `ELossFunction` spellings that this port trains. The list is
#: the reachable set of their GPU pointwise target
#: (`pointwise_target_impl.h:259-299`), minus the ones whose leaf estimator
#: or kernel family is not ported.
LOSSES = (
    "MultiClass",
    "RMSE",
    "Logloss",
    "CrossEntropy",
    "Quantile",
    "MAE",
    "LogLinQuantile",
    "MAPE",
    "Poisson",
    "Lq",
    "Expectile",
    "Tweedie",
    "Huber",
)

#: `MultiClass` is the one loss here whose model is MULTI-DIMENSIONAL.
#: Its labels are DENSE CLASS CODES 0..k-1 and the class count is derived
#: from them, as their `TClassificationTargetHelper` derives it -- a count
#: that disagreed with the data would be a wrong model rather than an
#: error, so it is not a parameter.
MULTI_OUTPUT_LOSSES = ("MultiClass",)

#: `MultiClassOneVsAll` IS IMPLEMENTED AND GATED IN THE MOJO LAYER and is
#: NOT REACHABLE FROM PYTHON. Its two kernels push the CPython extension
#: past what any measured basename can carry -- see PORTING.md 70, where
#: the AOT kernel count is a function of the entry file's NAME. Every stem
#: `bindings/build.sh` tries now drops a subsystem, so the build refuses to
#: install rather than shipping an artifact that dies at the first launch.
#:
#: Named here rather than accepted and then failing inside Metal. Use it
#: from Mojo (`gbdt/train.mojo`, `loss="MultiClassOneVsAll"`), where
#: `check-multilogit` and `check-multiclass-train` gate it.
_UNREACHABLE_LOSSES = {
    "MultiClassOneVsAll": (
        "its kernels do not fit in the CPython extension under any "
        "basename bindings/build.sh has measured; see PORTING.md 70. "
        "It works and is gated from Mojo."
    ),
}

#: `gbdt_predict_multi`'s transform, mirroring their `EPredictionType`
#: (`libs/model/eval_processing.h:186-226`).
_PREDICT_RAW = 0
_PREDICT_SOFTMAX = 1   # their `Probability`,      MultiClass
_PREDICT_SIGMOID = 2   # their `MultiProbability`, MultiClassOneVsAll

#: Losses whose parameter CatBoost makes MANDATORY. Passing the loss without
#: it raises here rather than in Mojo, so the message names the Python
#: keyword the caller has to add.
_REQUIRED_PARAM = {
    "Lq": ("q", "loss_q"),
    "Huber": ("delta", "loss_delta"),
    "Tweedie": ("variance_power", "loss_variance_power"),
    "Expectile": ("alpha", "loss_alpha"),
}

#: `EBootstrapType` spellings reachable from their GPU oblivious searcher.
#: MVS is absent because their own searcher asserts it away
#: (`weak_objective_impl.h:30`).
BOOTSTRAP_TYPES = ("Bayesian", "Bernoulli", "Poisson", "No")

#: `ELeavesEstimation`, for the override. `None` means let the loss decide.
LEAF_ESTIMATION_GRADIENT = 0
LEAF_ESTIMATION_NEWTON = 1
LEAF_ESTIMATION_EXACT = 2
LEAF_ESTIMATION_SIMPLE = 3

_LEAF_ESTIMATION_NAMES = {
    "Gradient": LEAF_ESTIMATION_GRADIENT,
    "Newton": LEAF_ESTIMATION_NEWTON,
    "Exact": LEAF_ESTIMATION_EXACT,
    "Simple": LEAF_ESTIMATION_SIMPLE,
}

#: `EScoreFunction`. Cosine is CatBoost's shipped GPU default.
SCORE_FUNCTION_COSINE = 1

_UNSET = -1.0

#: their `EOverfittingDetectorType` spellings. `None` here means the
#: option was not given, and that is NOT the same as `od_type="None"`:
#: unset lets `TOverfittingDetectorOptions::Load`
#: (`overfitting_detector_options.cpp:24-32`) pick the type from whichever
#: of `od_pvalue` / `od_wait` was given, while "None" turns the detector
#: off outright. Wilcoxon is theirs and is not ported.
OD_TYPES = ("None", "IncToDec", "Iter")


def _tri(v):
    """A CatBoost `TOption` tri-state: `None` is `NotSet()`.

    Used for `use_best_model`, whose default is DATA-DEPENDENT on their
    side (`options_helper.cpp:106-108` turns it on when there is a test
    set with a non-constant target), so `False` and "unset" have to stay
    distinguishable across the boundary.
    """
    if v is None:
        return -1
    return 1 if v else 0


class GradientBoosting:
    """Gradient-boosted oblivious trees, mirroring CatBoost's GPU learner.

    Parameters
    ----------
    loss : str, default 'RMSE'
        One of `LOSSES`, CatBoost's own spellings. Four of them require a
        parameter: `Lq` needs `loss_q`, `Huber` needs `loss_delta`,
        `Tweedie` needs `loss_variance_power`, `Expectile` needs
        `loss_alpha`.
    n_estimators : int, default 100
        CatBoost's `iterations`.
    max_depth : int, default 6
        CatBoost's `depth`. The tree is oblivious, so this is exactly
        `2 ** max_depth` leaves.
    learning_rate : float, default 0.03
        CatBoost's default (`boosting_options.cpp:10`), not scikit-learn's
        0.1.
    l2_leaf_reg : float, default 3.0
    border_count : int, default 128
        Quantization bins per numeric feature.
    random_state : int, default 0
    loss_alpha : float, optional
        Quantile level for `Quantile` and `LogLinQuantile` (default 0.5),
        and MANDATORY for `Expectile`. Ignored by `MAE`, whose alpha
        CatBoost fixes at 0.5 (`pointwise_target_impl.h:272-275`).
    loss_q, loss_delta, loss_variance_power : float, optional
        `Lq`'s q, `Huber`'s delta, `Tweedie`'s variance_power.
    loss_border : float, optional
        `Logloss`'s target threshold, default 0.5.
    leaf_estimation_method : {'Newton','Gradient','Exact','Simple'}, optional
        None (default) means the LOSS decides, per CatBoost.
    leaf_estimation_iterations : int, optional
        None (default) means the loss decides.
    bootstrap_type : {'Bayesian','Bernoulli','Poisson','No'}, optional
        None means no row sampling. 'Bernoulli' is the familiar `subsample`
        knob; 'Bayesian' is CatBoost's GPU default and uses
        `bagging_temperature` instead.
    bagging_temperature : float, default 1.0
        Bayesian only. CatBoost refuses `subsample` beside it and so does
        this.
    subsample : float, optional
        Bernoulli and Poisson only. Default 0.66
        (`bootstrap_options.h:15`).
    cat_features : sequence of int, optional
        Column indices holding DENSE CATEGORY CODES 0..k-1. CatBoost's own
        dispatch decides what happens to each
        (`binarizations_manager.cpp:106-115`): one-hot when the cardinality
        is small enough, target statistics (CTRs) otherwise.

    od_type : {'None', 'IncToDec', 'Iter'}, optional
        The overfitting detector. LEAVING IT UNSET IS NOT THE SAME AS
        'None': their `Load` (`overfitting_detector_options.cpp:24-32`)
        picks the type from whichever of `od_pvalue` / `od_wait` was
        given -- a wait alone means 'Iter', a p-value alone means
        'IncToDec', neither means 'None'. Requires `eval_set`; stopping on
        the learn loss would stop on a curve that falls by construction,
        so it raises instead.
    od_pvalue : float, optional
        `stop_pvalue`, IncToDec's threshold. Their default is 0, which
        makes the detector INACTIVE (`IsActive()` is `Threshold > 0`).
    od_wait : int, optional
        `wait_iterations`, default 20. Iterations without a new best
        before stopping.
    use_best_model : bool, optional
        Truncate the returned ensemble to the best held-out iteration,
        their `ShrinkToBestIteration`. UNSET IS NOT FALSE: with an
        `eval_set` whose target is not constant it defaults to True, which
        is their own data-dependent default (`options_helper.cpp:106-108`).
        True without an `eval_set` raises -- they warn and carry on, and a
        warning on a returned model is invisible from here.
    best_model_min_trees : int, default 1
        The shrink may not cut below this many trees
        (`output_file_options.cpp:77`,
        `boosting_progress_tracker.cpp:162`).

    Attributes
    ----------
    model_ : str
        The fitted ensemble, as the text format `check-model-io` gates
        bit-for-bit. It is a plain string: save it, ship it, hand it to
        another process. Floats in it carry their hex bits beside the
        decimal, because `String(Float32)` on this toolchain is one ULP
        wrong for 0.46% of values and a decimal-only round trip would
        change predictions.
    loss_curve_ : ndarray
        The training loss after each iteration. It is CatBoost's
        `functionValue` negated and divided by the row count, so it FALLS.
    test_loss_curve_ : ndarray or None
        The HELD-OUT loss after each iteration, `None` without an
        `eval_set`. This is the curve the detector reads and the one worth
        plotting: `loss_curve_` falls almost by construction.
    best_iteration_ : int
        The index of the lowest `test_loss_curve_` entry, or of the lowest
        learn loss with no eval set. It is the ERROR tracker's best
        (`error_tracker.h:73-75`), so with `best_model_min_trees` above it
        the fitted model holds MORE trees than `best_iteration_ + 1` and
        both numbers are right.
    stopped_early_ : bool
        Whether the detector fired before `n_estimators` was reached.
    """

    def __init__(
        self,
        loss="RMSE",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        border_count=128,
        random_state=0,
        loss_alpha=None,
        loss_q=None,
        loss_delta=None,
        loss_variance_power=None,
        loss_border=None,
        leaf_estimation_method=None,
        leaf_estimation_iterations=None,
        bootstrap_type=None,
        bagging_temperature=1.0,
        subsample=None,
        cat_features=None,
        one_hot_features=None,
        od_type=None,
        od_pvalue=None,
        od_wait=None,
        use_best_model=None,
        best_model_min_trees=1,
    ):
        if loss in _UNREACHABLE_LOSSES:
            raise NotImplementedError(
                f"mojolearn: {loss} is not reachable from Python -- "
                f"{_UNREACHABLE_LOSSES[loss]}"
            )
        if loss not in LOSSES:
            raise ValueError(
                f"mojolearn: loss must be one of {LOSSES}, got {loss!r}"
            )
        if loss in _REQUIRED_PARAM:
            cb_name, py_name = _REQUIRED_PARAM[loss]
            if locals()[py_name] is None:
                raise ValueError(
                    f"mojolearn: {loss} requires {py_name}= "
                    f"(CatBoost's {cb_name!r}, which it makes mandatory)"
                )
        if bootstrap_type is not None and bootstrap_type not in BOOTSTRAP_TYPES:
            raise ValueError(
                f"mojolearn: bootstrap_type must be one of "
                f"{BOOTSTRAP_TYPES}, got {bootstrap_type!r}"
            )
        if bootstrap_type == "Bayesian" and subsample is not None:
            # CatBoost's own validator (`catboost_options.cpp:795`)
            raise ValueError(
                "mojolearn: bootstrap_type='Bayesian' does not support "
                "subsample; it takes bagging_temperature"
            )
        if leaf_estimation_method is not None:
            if leaf_estimation_method not in _LEAF_ESTIMATION_NAMES:
                raise ValueError(
                    f"mojolearn: leaf_estimation_method must be one of "
                    f"{tuple(_LEAF_ESTIMATION_NAMES)}, got "
                    f"{leaf_estimation_method!r}"
                )

        self.loss = loss
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.random_state = random_state
        self.loss_alpha = loss_alpha
        self.loss_q = loss_q
        self.loss_delta = loss_delta
        self.loss_variance_power = loss_variance_power
        self.loss_border = loss_border
        self.leaf_estimation_method = leaf_estimation_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.subsample = subsample
        self.cat_features = cat_features
        self.one_hot_features = one_hot_features
        self.od_type = od_type
        self.od_pvalue = od_pvalue
        self.od_wait = od_wait
        self.use_best_model = use_best_model
        self.best_model_min_trees = int(best_model_min_trees)

        self.model_ = None
        self.loss_curve_ = None
        self.test_loss_curve_ = None
        self.best_iteration_ = None
        self.stopped_early_ = None
        self.n_features_in_ = None
        #: 1 for every single-output loss; `n_classes - 1` for MultiClass,
        #: because the last class's approx is pinned at zero and is not
        #: stored. Read from the fitted model, never assumed.
        self.approx_dim_ = None
        self.n_classes_ = None

    # -- the parameter list, in the ONE order both sides name --------------
    #
    # `bindings/_mojolearn.mojo:gbdt_fit_binding` writes this same order in
    # the same words. A silent reordering here is a wrong answer, not a
    # failure, which is why it is spelled out in both places.
    def _params(self, n_rows, n_features, n_flags, n_weights=0,
                n_eval_rows=0):
        def f(v):
            return _UNSET if v is None else float(v)

        method = self.leaf_estimation_method
        method_code = (
            -1 if method is None else _LEAF_ESTIMATION_NAMES[method]
        )
        iters = self.leaf_estimation_iterations
        return [
            n_rows,                                     # 0
            n_features,                                 # 1
            int(n_weights),                             # 2  n_weights
            n_flags,                                    # 3
            int(self.border_count),                     # 4
            int(self.n_estimators),                     # 5
            int(self.max_depth),                        # 6
            float(self.learning_rate),                  # 7
            float(self.l2_leaf_reg),                    # 8
            int(self.random_state),                     # 9
            SCORE_FUNCTION_COSINE,                      # 10
            f(self.loss_alpha),                         # 11
            f(self.loss_q),                             # 12
            f(self.loss_delta),                         # 13
            f(self.loss_variance_power),                # 14
            f(self.loss_border),                        # 15
            -1 if iters is None else int(iters),        # 16
            method_code,                                # 17
            float(self.bagging_temperature),            # 18
            f(self.subsample),                          # 19
            int(n_eval_rows),                           # 20
            f(self.od_pvalue),                          # 21
            -1 if self.od_wait is None
            else int(self.od_wait),                     # 22
            _tri(self.use_best_model),                  # 23
            int(self.best_model_min_trees),             # 24
        ]

    def _flags(self, n_features):
        """One packed word per feature: bit 0 categorical, bit 1 one-hot."""
        cat = self.cat_features
        one_hot = self.one_hot_features
        if not cat and not one_hot:
            return None
        flags = np.zeros(n_features, dtype=np.uint32)
        for i in cat or ():
            if not 0 <= i < n_features:
                raise ValueError(
                    f"mojolearn: cat_features index {i} out of range for "
                    f"{n_features} features"
                )
            flags[i] |= 1
        for i in one_hot or ():
            if not 0 <= i < n_features:
                raise ValueError(
                    f"mojolearn: one_hot_features index {i} out of range "
                    f"for {n_features} features"
                )
            flags[i] |= 2
        return flags

    def _eval_arrays(self, eval_set, n_features):
        """`(X_eval, y_eval)` -> column-major float32 plus a row count.

        Returns `(None, None, 0)` when there is no eval set, which is the
        "unread" contract the binding documents for the two addresses.
        """
        if eval_set is None:
            return None, None, 0
        if isinstance(eval_set, list):
            if len(eval_set) != 1:
                raise ValueError(
                    "mojolearn: eval_set takes ONE (X, y) pair; the "
                    "boosting carries a single test cursor. CatBoost "
                    f"accepts a list, this takes {len(eval_set)} as an "
                    "error rather than scoring the first and dropping "
                    "the rest"
                )
            eval_set = eval_set[0]
        try:
            Xe, ye = eval_set
        except (TypeError, ValueError):
            raise ValueError(
                "mojolearn: eval_set must be (X_eval, y_eval)"
            ) from None

        Xea, _ = as_f32_c(Xe, "eval_set X")
        n_eval_rows, n_eval_features = Xea.shape
        if n_eval_features != n_features:
            raise ValueError(
                f"mojolearn: eval_set X has {n_eval_features} features "
                f"for a fit on {n_features}"
            )
        if n_eval_rows == 0:
            raise ValueError("mojolearn: eval_set X has no rows")
        yea = np.ascontiguousarray(
            np.asarray(ye).ravel(), dtype=np.float32
        )
        if yea.shape[0] != n_eval_rows:
            raise ValueError(
                f"mojolearn: eval_set y has {yea.shape[0]} values for "
                f"{n_eval_rows} rows"
            )
        # the same transpose `fit` prices for the learn matrix
        Ecol = np.ascontiguousarray(Xea.T).reshape(-1)
        return Ecol, yea, n_eval_rows

    def fit(self, X, y, sample_weight=None, eval_set=None):
        """Fit the ensemble. `X` is (n_samples, n_features), `y` is 1-D.

        `sample_weight` is a per-row weight, `None` meaning all ones. It
        MULTIPLIES with `class_weights` where both are given, which is
        their own combination at pool build
        (`target/data_providers.cpp:168`:
        `rawWeights[i] * rawGroupWeights[i] * classWeights[...]`). Their
        group-weight factor is absent because this port carries no
        `group_id`.

        `eval_set` is `(X_eval, y_eval)`, or a one-element list holding
        that pair. **CatBoost takes a LIST of eval sets and this takes
        one**, because the boosting carries a single test cursor; more
        than one raises rather than silently scoring the first.

        Passing it changes the DEFAULT of `use_best_model` to True, which
        is CatBoost's own data-dependent default
        (`options_helper.cpp:106-108`) and means `model_` holds the trees
        up to the best held-out iteration rather than all of them. Pass
        `use_best_model=False` to keep every tree.

        After the call: `test_loss_curve_` is the held-out loss per
        iteration, `best_iteration_` its argmin, and `stopped_early_` says
        whether the detector fired before `n_estimators`.
        """
        Xa, _ = as_f32_c(X, "X")
        n_rows, n_features = Xa.shape

        ya = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
        if ya.shape[0] != n_rows:
            raise ValueError(
                f"mojolearn: y has {ya.shape[0]} values for {n_rows} rows"
            )

        # COLUMN-MAJOR is what the quantizer walks. This is the transpose
        # the module docstring prices; `ascontiguousarray` on the .T view
        # is what materialises it.
        Xcol = np.ascontiguousarray(Xa.T).reshape(-1)

        flags = self._flags(n_features)
        n_flags = 0 if flags is None else n_features
        flags_holder = flags if flags is not None else np.zeros(1, np.uint32)

        if sample_weight is None:
            wa = Xcol[:1]  # unread while params[2] == 0
            n_weights = 0
        else:
            wa = np.ascontiguousarray(
                np.asarray(sample_weight).ravel(), dtype=np.float32
            )
            if wa.shape[0] != n_rows:
                raise ValueError(
                    f"mojolearn: sample_weight has {wa.shape[0]} entries "
                    f"for {n_rows} rows"
                )
            if (wa < 0).any():
                raise ValueError(
                    "mojolearn: sample_weight has negative entries"
                )
            n_weights = n_rows

        Ecol, ea, n_eval_rows = self._eval_arrays(eval_set, n_features)

        params = self._params(
            n_rows, n_features, n_flags, n_weights, n_eval_rows
        )
        if self.od_type is not None and self.od_type not in OD_TYPES:
            raise ValueError(
                f"mojolearn: od_type must be one of {OD_TYPES}, got "
                f"{self.od_type!r}"
            )
        strs = [self.loss, self.bootstrap_type or "", self.od_type or ""]

        # THE EVAL ADDRESSES ARE UNREAD WHEN params[20] IS 0, and the
        # learn buffer stands in so nothing has to allocate a throwaway --
        # the same stand-in the weights use.
        eval_x_holder = Ecol if Ecol is not None else Xcol
        eval_y_holder = ea if ea is not None else ya

        out = _mojolearn.gbdt_fit(
            _addr_ro(Xcol),
            _addr_ro(ya),
            _addr_ro(wa),
            _addr_ro(flags_holder),
            _addr_ro(eval_x_holder),
            _addr_ro(eval_y_holder),
            params,
            strs,
        )
        self.model_ = out[0]
        self.best_iteration_ = int(out[1])
        self.stopped_early_ = bool(out[2])
        self.loss_curve_ = np.asarray(out[3], dtype=np.float64)
        self.test_loss_curve_ = (
            np.asarray(out[4], dtype=np.float64) if n_eval_rows else None
        )
        self.n_features_in_ = n_features
        self.approx_dim_ = _mojolearn.gbdt_model_dim(self.model_)
        # MULTICLASS DROPS A CLASS AND ONEVSALL DOES NOT. `dim` is
        # `n_classes - 1` for the first (the last class's approx is pinned
        # at zero and not stored) and `n_classes` for the second, whose
        # classes are independent (`multiclass_targets.h:129-134`).
        self.n_classes_ = (
            self.approx_dim_ + 1
            if self.loss == "MultiClass"
            else None
        )
        return self

    def _check_fitted(self, X):
        if self.model_ is None:
            raise RuntimeError("mojolearn: predict() before fit()")
        Xa, _ = as_f32_c(X, "X")
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"mojolearn: model was fitted on {self.n_features_in_} "
                f"features, got {n_features}"
            )
        return np.ascontiguousarray(Xa.T).reshape(-1), n_rows

    def predict(self, X):
        """RAW SCORES, not probabilities. See the module docstring.

        For a single-output loss the result is `(n_samples,)`. For
        `MultiClass` it is `(n_samples, n_classes - 1)` -- the free
        approxes, with the LAST class's pinned at zero and therefore not
        returned. `predict_proba` is what turns those into `n_classes`
        columns; dropping the pinned class and renormalising the rest
        would give a different answer.
        """
        Xcol, n_rows = self._check_fitted(X)

        if self.approx_dim_ > 1:
            out = np.empty(n_rows * self.approx_dim_, dtype=np.float32)
            width = _mojolearn.gbdt_predict_multi(
                self.model_, _addr_ro(Xcol), _addr(out),
                [n_rows, _PREDICT_RAW],
            )
            if width != self.approx_dim_:
                raise RuntimeError(
                    f"mojolearn: predict wrote width {width}, expected "
                    f"{self.approx_dim_}"
                )
            return out.reshape(n_rows, width)

        out = np.empty(n_rows, dtype=np.float32)
        wrote = _mojolearn.gbdt_predict(
            self.model_, _addr_ro(Xcol), _addr(out), [n_rows]
        )
        if wrote != n_rows:
            raise RuntimeError(
                f"mojolearn: predict wrote {wrote} of {n_rows} rows"
            )
        return out

    def predict_proba(self, X):
        """Class probabilities, `(n_samples, n_classes)`.

        Defined for the three losses that have a link: `Logloss` and
        `CrossEntropy` (the sigmoid) and `MultiClass` (the softmax). It
        refuses every other loss rather than returning a number that looks
        like a probability. CatBoost's `prediction_type='Probability'` is
        the same transform over the same raw scores.

        THE TWO MULTI-OUTPUT LOSSES GET DIFFERENT TRANSFORMS, and it is
        not a style choice. `MultiClass` is a softmax -- one shared
        denominator, columns summing to 1, their `Probability`
        (`eval_processing.h:214-221`). `MultiClassOneVsAll` is an
        ELEMENTWISE sigmoid -- their `MultiProbability` (`:222-226`) --
        and **its columns do NOT sum to 1**, because it fitted
        `n_classes` independent "is this row class k" problems and
        renormalising would assert an exclusivity it never learned.

        THE MULTICLASS SOFTMAX IS TAKEN OVER ALL `n_classes`, INCLUDING THE
        PINNED ONE. The model stores `n_classes - 1` free approxes and the
        last class's is zero by construction; the device applies the
        transform, so the max-subtraction matches the one in their
        `MultiLogitValAndFirstDerImpl` rather than being re-derived here.
        The returned columns are in class-code order, `0 .. n_classes - 1`.
        """
        if self.loss in MULTI_OUTPUT_LOSSES:
            Xcol, n_rows = self._check_fitted(X)
            mode = _PREDICT_SOFTMAX
            out = np.empty(n_rows * self.n_classes_, dtype=np.float32)
            width = _mojolearn.gbdt_predict_multi(
                self.model_, _addr_ro(Xcol), _addr(out), [n_rows, mode]
            )
            if width != self.n_classes_:
                raise RuntimeError(
                    f"mojolearn: predict_proba wrote width {width}, "
                    f"expected {self.n_classes_}"
                )
            return out.reshape(n_rows, width)

        if self.loss not in ("Logloss", "CrossEntropy"):
            raise ValueError(
                f"mojolearn: predict_proba is defined for Logloss, "
                f"CrossEntropy and MultiClass; this model was fitted with "
                f"{self.loss!r}. Use predict() and apply the link "
                f"yourself."
            )
        raw = self.predict(X).astype(np.float64)
        p1 = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack((1.0 - p1, p1))

    def predict_classes(self, X):
        """The argmax of `predict_proba`, as dense class codes.

        Named `predict_classes` rather than overloading `predict`, because
        `predict` returns RAW SCORES for every loss in this library and
        making one loss return labels instead would be the kind of silent
        contract change that is worse than an extra method.
        """
        if self.loss not in MULTI_OUTPUT_LOSSES + (
            "Logloss", "CrossEntropy",
        ):
            raise ValueError(
                f"mojolearn: predict_classes needs a classification loss; "
                f"this model was fitted with {self.loss!r}."
            )
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)
