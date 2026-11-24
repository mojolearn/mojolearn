# SPDX-License-Identifier: Apache-2.0
"""The XLinear family as MEMBERS (core/member_abi.mojo, schema v2): the
generalized linear regressors PoissonRegressor, GammaRegressor and
TweedieRegressor, HuberRegressor, BayesianRidge, ARDRegression, Lars, LassoLars, QuantileRegressor, RidgeCV, LassoCV,
ElasticNetCV and IsotonicRegression; and the classifiers RidgeClassifier and LogisticRegressionCV.

One entry, `xlinear_member_v2`, published by each x_linear image
(`xlinear_member_publish`, beside its operations table). It runs the family's
existing typed RUN (x_linear/door_run.mojo `xl_fit_run`, `xl_huber_fit_run`,
`xl_decision_run`, `xl_r2_run`) over the entries of the `DeviceRoute` the
estimator's own doors resolve: `entry` the x_linear operations entry (fit and
decision), `fifth` the metrics image's R^2 entry of the route's tier and
device (filled by the caller's admission when that image is loaded; without
it score is refused in words and the caller predicts and runs its own
metric). No kernel is touched.

Kinds (the second word of each class's `_mojolearn_member`): 0
PoissonRegressor, 1 GammaRegressor, 2 TweedieRegressor, 3 HuberRegressor,
4 BayesianRidge, 5 ARDRegression, 6 Lars, 7 LassoLars, 8 QuantileRegressor, 9 RidgeClassifier, 10 RidgeCV,
11 LassoCV, 12 ElasticNetCV, 13 LogisticRegressionCV, 14 IsotonicRegression.

MEMBER_OP_NEW parameters (Float64 words; word 0 is the kind, the count is
the kind's):
  * kinds 0..2, 7 words: [0] kind, [1] fit_intercept (0 / 1), [2] max_iter,
    [3] link (0 identity, 1 log; Poisson and Gamma: 1), [4] power (Poisson 1,
    Gamma 2, Tweedie its own), [5] alpha, [6] tol.
  * kind 3, 6 words: [0] 3, [1] fit_intercept, [2] max_iter, [3] epsilon,
    [4] alpha, [5] tol.
  * kind 4, 10 words: [0] 4, [1] fit_intercept, [2] max_iter, [3] tol,
    [4] alpha_1, [5] alpha_2, [6] lambda_1, [7] lambda_2, [8] alpha_init,
    [9] lambda_init (-1: None).
  * kind 5, 9 words: [0] 5, [1] fit_intercept, [2] max_iter, [3] tol,
    [4] alpha_1, [5] alpha_2, [6] lambda_1, [7] lambda_2, [8] threshold_lambda.
  * kind 6, 4 words: [0] 6, [1] fit_intercept, [2] n_nonzero_coefs,
    [3] positive (0 / 1).
  * kind 7, 5 words: [0] 7, [1] fit_intercept, [2] max_iter, [3] positive,
    [4] alpha.
  * kind 8, 6 words: [0] 8, [1] fit_intercept, [2] max_iter, [3] quantile,
    [4] alpha, [5] tol.
  * kind 9, 3 words: [0] 9, [1] fit_intercept, [2] alpha; a class_weight
    is the tail from word 3 (lane mf-adm3, core/member_class_weight.mojo:
    'balanced' or a dict's table over the caller's codes).
  * kind 10, 5 + A words: [0] 10, [1] fit_intercept, [2] store_cv_results,
    [3] the cv (0: cv=None, the one fit over the grid; else minus the
    handle of its splitter spec, bindings/msel_split_spec_run.mojo: an
    integer cv is an unshuffled KFold spec marked int_cv, which runs
    `_fit_kfold` unweighted with at least two rows a fold and the split
    path otherwise, as the door; one of our splitters or an index list runs
    the split path; lane mf-split5), [4] A, [5 ..] the A alphas.
  * kinds 11, 12, 10 + L (+ G) words: [0] kind, [1] fit_intercept,
    [2] max_iter, [3] eps, [4] tol, [5] positive (0 / 1), [6] folds (cv=None:
    5; an integer cv: its value; KFold's contiguous folds, made here; minus
    a splitter spec's handle for one of our splitters or an index list,
    lane mf-split5),
    [7] G the alpha grid's length, [8] explicit (0: the path's own grid of G
    alphas; 1: the user's G alphas follow the ratios, sorted descending),
    [9] L, [10 ..] the L l1_ratios (LassoCV: [1.0]), then the G alphas when
    explicit.
  * kind 13, 7 + C words: [0] 13, [1] fit_intercept, [2] max_iter, [3] tol,
    [4] folds (cv=None: 5; the binding's stratified folds; minus a splitter
    spec's handle, lane mf-split5), [5] class_weight
    (0 None, 1 'balanced', 2 a dict: its table is the tail from word 7 + C,
    lane mf-adm3), [6] C, [7 ..] the C values of the Cs grid. A user's own
    splitter object is refused by the admission (never called).
  * kind 14, 7 words: [0] 14, [1] increasing (0 False, 1 True, 2 'auto'),
    [2] y_min given (0 / 1), [3] y_min, [4] y_max given, [5] y_max,
    [6] out_of_bounds (0 'nan', 1 'clip', 2 'raise').
The estimator's own checks (solver, warm_start, alpha >= 0, Tweedie's power
and link words, Huber's epsilon >= 1, BayesianRidge's compute_score, Lars' jitter, LassoLars' alpha >= 0, the quantile in (0, 1), RidgeClassifier's positive and solver words) run at the caller's admission.

Views: X dense row-major host float32; y and sample weights host float32
vectors (one target); kinds 5..7 take no sample weights (a given weight is
refused in words; MEMBER_OP_CAPABILITIES of its state says so). A device-home view, a column index and a nonzero
`fit_output` are refused in words. FOLDS (lane vdf-scope-linear2): on a GPU image whose x_linear entry has
XL_HAS_SLOT_X a row index selects the fold's rows of the whole X on the device and every fit and decision entry
reads THE FOLD X WORD (`_fold_op`, core/member_fold_x.mojo; a RidgeCV fit with a splitter cv takes each split's
rows from the fold slot on the device, lane vdf-scope-linear3); y, THE GIVEN y and the weights are taken bitwise through the index on the host. PREDICT writes the prediction as float32,
rows x 1: link(X @ coef + intercept) for kinds 0..2, X @ coef + intercept
for the others. SCORE is R^2 of that prediction (unweighted; a weighted score
is not published). A GLM's targets outside its loss's range are refused after
the fit, in the estimator's words.

CLASSIFIER (kind 9): y is the caller's int32 class codes 0..classes-1
(`MEMBER_I32`, rows x 1), `call.classes` the class count (>= 2) at FIT and
at IMPORT-done. PREDICT writes int32 codes, rows x 1; DECISION float32
scores, rows x T (T = 1 for two classes, else classes); SCORE is the mean
accuracy (an exact count split over the host's workers). Its blocks: 0
`MEMBER_BLOCK_COEF` float32 T x features, 1 `MEMBER_BLOCK_INTERCEPT` float32
1 x T. A system that does not factor is refused in the estimator's words.
Kind 13 (LogisticRegressionCV) is a classifier of the same form; it also
publishes PREDICT_PROBA (float32 rows x classes, `predict_proba`'s rows) and
holds 2 `MEMBER_BLOCK_SCALARS` float64 [C_ (the Cs entry nearest the fit's
best C), n_iter] and, after a fit, 3 `XL_MEMBER_BLOCK_SCORES`
(`MEMBER_BLOCK_FAMILY` + 6) float32 folds x C (`scores_`, one table every
class label shares). An import needs COEF, INTERCEPT and SCALARS.

Kinds 11, 12 hold 0 COEF, 1 SCALARS float64 [intercept, alpha_, l1_ratio_
(the user's grid entry nearest the fit's best), n_iter] and, after a fit,
2 `XL_MEMBER_BLOCK_ALPHAS` (`MEMBER_BLOCK_FAMILY` + 4) float32 L x G
(`alphas_`) and 3 `XL_MEMBER_BLOCK_MSE_PATH` (`MEMBER_BLOCK_FAMILY` + 5)
float32 (L * G) x folds (`mse_path_`).

Kind 14 (IsotonicRegression) reads X as one float32 column (rows x 1);
PREDICT and TRANSFORM write the interpolated value of each row (rows x 1;
out_of_bounds='raise' refuses a row outside the fitted range in the
estimator's words). It publishes no score and no coefficients. Blocks: 0
SCALARS float64 [increasing_ (0 / 1), X_min_, X_max_]; 1
`XL_MEMBER_BLOCK_THRESHOLDS` (`MEMBER_BLOCK_FAMILY` + 7) float32 2 x knots
(row 0 X_thresholds_, row 1 y_thresholds_); an import needs both.

Regressor fitted-state blocks (EXPORT / IMPORT), in index order:
0 `MEMBER_BLOCK_COEF` float32 1 x features; 1 `MEMBER_BLOCK_SCALARS` float64
in the kind's order: kinds 0..2 [intercept, n_iter]; kind 3 [intercept,
scale, n_iter]; kind 4 [intercept, alpha_, lambda_, n_iter]; kind 5 [intercept, alpha_,
n_iter]; kinds 6, 7 [intercept, n_iter, alpha_]; kind 10 [intercept, alpha_,
best_score_] (with store_cv_results also 2 `XL_MEMBER_BLOCK_CV_RESULTS`,
`MEMBER_BLOCK_FAMILY` + 3, float32 1 x A). Kinds 6, 7 after a fit also holds
2 `XL_MEMBER_BLOCK_ACTIVE` (`MEMBER_BLOCK_FAMILY` + 2) int32 1 x active
features (`active_`, in the order the path took them). Kind 5 also holds 2 `XL_MEMBER_BLOCK_LAMBDA` (`MEMBER_BLOCK_FAMILY` +
1) float32 1 x features (`lambda_`). Kind 3 after a fit also holds 2 `XL_MEMBER_BLOCK_OUTLIERS`
(`MEMBER_BLOCK_FAMILY`) int32 1 x training rows, 1 for an outlier row
(`outliers_`). An import needs COEF and SCALARS; a family block is a training
record, optional at import (an imported state then exports two blocks).
Kinds 4 and 5 after a fit also hold, last, `XL_MEMBER_BLOCK_X_OFFSET`
(`MEMBER_BLOCK_FAMILY` + 9) float32 1 x features (`X_offset_`) and, for
BayesianRidge, `XL_MEMBER_BLOCK_FACTOR` (`MEMBER_BLOCK_FAMILY` + 10) float32
(features + 1) x features (rows 0..features-1 V, eigenvector m in column m;
the last row the clamped eigenvalues ev), for ARDRegression
`XL_MEMBER_BLOCK_SIGMA` (`MEMBER_BLOCK_FAMILY` + 8) float32 dk x dk (`sigma_`
over the kept features), and publish PREDICT_STD: x float32 rows x features
-> dst float32 rows x 1, scikit-learn's predict(return_std=True):
BayesianRidge sqrt(sum_m (V'(x - X_offset_))_m^2 / (alpha_ ev_m + lambda_) +
1 / alpha_) (sigma_ is never formed), ARDRegression sqrt(sum((x -
X_offset_)[:, kept] @ sigma_ * (x - X_offset_)[:, kept], axis=1) + 1 /
alpha_) (kept = lambda_ < threshold_lambda); with `y` given (rows x 1 host
float32) the same rows' mean is written there too. These blocks (and ARD's
LAMBDA, its kept columns) are optional at import; a state imported without
them refuses PREDICT_STD in words.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's form); the lock covers only mint, free and the handle lookup.
Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.entry_table import entry_ref_unresolved
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.linear_entry_table_ext import XL_OP_DECISION, XL_OP_DECISION_CODES, XL_OP_DECISION_ROWS
from core.member_abi import (
    MEMBER_GIL_FREE, MEMBER_READS_Y_GIVEN, MEMBER_HOME_HOST,
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_DTYPE_UNSAID,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_DECISION,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_TRANSFORM, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_TRANSFORM,
    MEMBER_OP_PREDICT_STD, MEMBER_CAN_PREDICT_STD, MEMBER_CAN_NOTICES, MEMBER_BLOCK_NOTICES,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF, MEMBER_CAN_DECISION, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_XLINEAR
from core.device_route import DEVICE_GPU
from core.member_fold_x import (
    MemberFoldX, member_fold_x_open, member_fold_x_close,
    member_fold_take_into, member_fold_store_ready, member_fold_store,
)
from core.member_class_weight import (
    MEMBER_CW_HEAD, MEMBER_CW_NONE, MEMBER_CW_BALANCED, member_cw_mode, member_cw_table_f32,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_linear.door_run import (
    ALGO_GLM, ALGO_HUBER, ALGO_BAYES, ALGO_ARD, ALGO_LARS, ALGO_QUANTILE, LINK_IDENTITY, LINK_EXP,
    ALGO_ENETCV, ALGO_LOGCV, ALGO_ISOTONIC, ALGO_ISOTONIC_PREDICT, ROWS_LRCV_PROBA, XlClassPrepRequest,
    xl_logcv_fit_run, xl_isotonic_fit_run, xl_class_fit_run,
    ALGO_RIDGE, xl_run_block, XlRidgeCvRequest, xl_ridge_out, xl_ridge_singular, xl_ridge_cv_plain_run,
    xl_ridge_cv_kfold_run, XlFitRequest, XlDecisionRequest, xl_fit_run, xl_huber_fit_run, xl_decision_run, xl_r2_run, xl_lbfgs_work,
    XlBayesStdRequest, xl_bayes_std_run, xl_ard_kept_run, xl_slot_x_held,
)
from x_linear.door_run import (
    XlSplitWords, xl_cv_fold_ids_run, xl_cv_roles_limit, xl_cv_roles_run, XlRidgeSplitsOut, xl_ridge_cv_splits_run,
    xl_ridge_cv_choose_run, XL_RIDGE_SPLITS_EMPTY, XL_RIDGE_SPLITS_SINGULAR, XL_RIDGE_SPLITS_ERROR,
)
from core.member_notices import member_notice_add, MEMBER_NOTICE_WORDS, MEMBER_NOTICE_R2_FEW, MEMBER_NOTICE_SPLIT_LEAST
from bindings.msel_split_spec_run import (
    MselSplitSpec, MselSides, MselGroupCodes, msel_spec_copy, msel_spec_sides, msel_spec_sides_targets,
    msel_spec_stratified,
)
from x_linear.cv_fold_ids_run import xl_kfold_ids_f32_run
from x_linear.bayes_words import bayes_out_words, bayes_v_at, bayes_offset_at, ard_out_words, ard_dk_at, ard_offset_at, ard_sigma_at

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime XL_MEMBER_HUBER = 3
comptime XL_MEMBER_BAYES = 4
comptime XL_MEMBER_ARD = 5
comptime XL_MEMBER_LARS = 6
comptime XL_MEMBER_LASSO_LARS = 7
comptime XL_MEMBER_QUANTILE = 8
comptime XL_MEMBER_RIDGE_CLASSIFIER = 9
comptime XL_MEMBER_RIDGE_CV = 10
comptime XL_MEMBER_LASSO_CV = 11
comptime XL_MEMBER_ELASTIC_NET_CV = 12
comptime XL_MEMBER_LOGISTIC_CV = 13
comptime XL_MEMBER_ISOTONIC = 14
comptime XL_MEMBER_BLOCK_CV_RESULTS = MEMBER_BLOCK_FAMILY + 3
comptime XL_MEMBER_BLOCK_ALPHAS = MEMBER_BLOCK_FAMILY + 4
comptime XL_MEMBER_BLOCK_MSE_PATH = MEMBER_BLOCK_FAMILY + 5
comptime XL_MEMBER_BLOCK_SCORES = MEMBER_BLOCK_FAMILY + 6
comptime XL_MEMBER_BLOCK_THRESHOLDS = MEMBER_BLOCK_FAMILY + 7
comptime XL_MEMBER_BLOCK_LAMBDA = MEMBER_BLOCK_FAMILY + 1
comptime XL_MEMBER_BLOCK_ACTIVE = MEMBER_BLOCK_FAMILY + 2
comptime XL_MEMBER_BLOCK_OUTLIERS = MEMBER_BLOCK_FAMILY
comptime XL_MEMBER_BLOCK_SIGMA = MEMBER_BLOCK_FAMILY + 8
comptime XL_MEMBER_BLOCK_X_OFFSET = MEMBER_BLOCK_FAMILY + 9
comptime XL_MEMBER_BLOCK_FACTOR = MEMBER_BLOCK_FAMILY + 10
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


def xl_member_param_count(kind: Int) -> Int:
    """The MEMBER_OP_NEW word count of `kind` (0: unknown kind)."""
    if kind >= 0 and kind <= 2:
        return 7
    if kind == XL_MEMBER_HUBER:
        return 6
    if kind == XL_MEMBER_BAYES:
        return 10
    if kind == XL_MEMBER_ARD:
        return 9
    if kind == XL_MEMBER_LARS:
        return 4
    if kind == XL_MEMBER_LASSO_LARS:
        return 5
    if kind == XL_MEMBER_QUANTILE:
        return 6
    if kind == XL_MEMBER_RIDGE_CLASSIFIER:
        return 3
    if kind == XL_MEMBER_RIDGE_CV:
        return 5  # + the A alphas (word 4)
    if kind == XL_MEMBER_LASSO_CV or kind == XL_MEMBER_ELASTIC_NET_CV:
        return 10  # + the L ratios (word 9) and, when explicit, the G alphas (word 7)
    if kind == XL_MEMBER_LOGISTIC_CV:
        return 7  # + the C values (word 6)
    if kind == XL_MEMBER_ISOTONIC:
        return 7
    return 0


def _enetcv(kind: Int) -> Bool:
    return kind == XL_MEMBER_LASSO_CV or kind == XL_MEMBER_ELASTIC_NET_CV


def _lars(kind: Int) -> Bool:
    return kind == XL_MEMBER_LARS or kind == XL_MEMBER_LASSO_LARS


def xl_member_scalar_count(kind: Int) -> Int:
    """The SCALARS block length of `kind`."""
    if kind == XL_MEMBER_HUBER:
        return 3
    if kind == XL_MEMBER_BAYES:
        return 4
    if kind == XL_MEMBER_ARD or _lars(kind) or kind == XL_MEMBER_RIDGE_CV or kind == XL_MEMBER_ISOTONIC:
        return 3
    if _enetcv(kind):
        return 4
    return 2


struct XLinearMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    # [coef (features) | intercept]: the decision block of one target.
    var block: List[Float32]
    var scalars: List[Float64]
    # HuberRegressor's outlier flag of each training row (kernel bytes), and
    # their int32 export words, made once at the first export.
    var flags: List[UInt8]
    var outliers: List[Int32]
    # A per-feature fitted record of the kind (ARDRegression's lambda_).
    var record: List[Float32]
    # The active feature indices of a LARS path (`active_`).
    var active: List[Int32]
    # True after a fit (the training records are held), False after an import.
    var trained: Bool
    # Classifier kinds: the class count and the decision rows T (`block`
    # then holds T rows of [coef | intercept]; `record` the fit's
    # [coef (T x features) | intercept (T)] words the export views).
    var classes: Int
    var outputs: Int
    # A second fitted record (the CV paths' mse_path_, LogisticRegressionCV's
    # scores_) and the classifiers' intercept words staged by an import.
    var path: List[Float32]
    var extra: List[Float64]
    # BayesianRidge / ARDRegression's posterior (PREDICT_STD; lane
    # mf-iterimp2): BayesianRidge's factor (V features x features, then ev
    # features), ARDRegression's sigma_ (dk x dk) and its dk kept columns
    # (lambda_ < threshold_lambda, the fit's rule), and X_offset_.
    var factor: List[Float32]
    var sigma: List[Float32]
    var offset: List[Float32]
    var kept: List[Int32]
    var dk: Int
    # THE SPLITTER CV (lane mf-split5): the fold count of the last fit (the
    # rows of mse_path_ / scores_; a splitter's is the fit's), and the FIT
    # NOTICES (core/member_notices.mojo: RidgeCV's undefined split R^2).
    var cv_folds: Int
    var notices: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.block = List[Float32]()
        self.scalars = List[Float64]()
        self.flags = List[UInt8]()
        self.outliers = List[Int32]()
        self.record = List[Float32]()
        self.active = List[Int32]()
        self.trained = False
        self.classes = 0
        self.outputs = 1
        self.path = List[Float32]()
        self.extra = List[Float64]()
        self.factor = List[Float32]()
        self.sigma = List[Float32]()
        self.offset = List[Float32]()
        self.kept = List[Int32]()
        self.dk = 0
        self.cv_folds = 0
        self.notices = List[Float64]()

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.block = List[Float32]()
        self.scalars = List[Float64]()
        self.flags = List[UInt8]()
        self.outliers = List[Int32]()
        self.record = List[Float32]()
        self.active = List[Int32]()
        self.trained = False
        self.classes = 0
        self.outputs = 1
        self.path = List[Float32]()
        self.extra = List[Float64]()
        self.factor = List[Float32]()
        self.sigma = List[Float32]()
        self.offset = List[Float32]()
        self.kept = List[Int32]()
        self.dk = 0
        self.cv_folds = 0
        self.notices = List[Float64]()


struct XLinearMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[XLinearMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[XLinearMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[XLinearMemberStore, "MojolearnXLinearMemberStoreV2", 2]
comptime _SlotP = MutPointer[XLinearMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[XLinearMemberSlot]()


def _mint(var params: List[Float64]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live x_linear member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[XLinearMemberSlot](length=_CHUNK, fill=XLinearMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].params = params^
        slot[].forget()
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a stale or foreign handle is refused."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotP(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].forget()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _kind(slot: _SlotP) -> Int:
    return Int(slot[].params[0])


def _name(slot: _SlotP) -> String:
    var kind = _kind(slot)
    if kind == 0:
        return String("PoissonRegressor")
    if kind == 1:
        return String("GammaRegressor")
    if kind == 2:
        return String("TweedieRegressor")
    if kind == XL_MEMBER_HUBER:
        return String("HuberRegressor")
    if kind == XL_MEMBER_BAYES:
        return String("BayesianRidge")
    if kind == XL_MEMBER_ARD:
        return String("ARDRegression")
    if kind == XL_MEMBER_LARS:
        return String("Lars")
    if kind == XL_MEMBER_LASSO_LARS:
        return String("LassoLars")
    if kind == XL_MEMBER_QUANTILE:
        return String("QuantileRegressor")
    if kind == XL_MEMBER_RIDGE_CV:
        return String("RidgeCV")
    if kind == XL_MEMBER_LASSO_CV:
        return String("LassoCV")
    if kind == XL_MEMBER_ELASTIC_NET_CV:
        return String("ElasticNetCV")
    if kind == XL_MEMBER_LOGISTIC_CV:
        return String("LogisticRegressionCV")
    if kind == XL_MEMBER_ISOTONIC:
        return String("IsotonicRegression")
    return String("RidgeClassifier")


def _loss_name(slot: _SlotP) -> String:
    # The class attribute `_loss_name` (Tweedie: None -> "HalfTweedieLoss").
    var kind = _kind(slot)
    return String("HalfPoissonLoss") if kind == 0 else (
        String("HalfGammaLoss") if kind == 1 else String("HalfTweedieLoss"))


def _capabilities(kind: Int) -> Int:
    """The bits `kind` honors; kind -1: the family's union."""
    # MEMBER_GIL_FREE, every kind (THE GIL RULE, core/member_abi.mojo; the
    # trace is in fx-gil-RESULT.md, round 4): the store has its own lock;
    # every operation is calls of the x_linear operations entry (fit, class
    # prep, decisions, and for the expansion kinds the Spearman, Huber and
    # best-alpha words) on per-call buffers, never a device fold
    # (XL_OP_FIT_FOLD is not sent: no model-selection store is read), and the
    # metrics R^2 entry with no weights (no base helper is called). The entry
    # touches no CPython or GIL; the state it reaches (the x_linear context,
    # the pinned download stages) has its own lock. On a GPU route the
    # callers run one worker.
    var bits = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_GIL_FREE
    if kind != XL_MEMBER_ISOTONIC:
        # IsotonicRegression has no score method and no coefficients.
        bits |= MEMBER_CAN_SCORE | MEMBER_CAN_COEF
    if _weighted(kind):
        bits |= MEMBER_CAN_SAMPLE_WEIGHT
    if kind < 0 or _classifier(kind):
        bits |= MEMBER_CAN_DECISION | MEMBER_CAN_CLASSIFIER
    if kind < 0 or kind == XL_MEMBER_LOGISTIC_CV:
        bits |= MEMBER_CAN_PREDICT_PROBA
    if kind < 0 or kind == XL_MEMBER_ISOTONIC:
        bits |= MEMBER_CAN_TRANSFORM
    if kind < 0 or _posterior_kind(kind):
        bits |= MEMBER_CAN_PREDICT_STD
    if kind < 0 or kind == XL_MEMBER_RIDGE_CV or kind == XL_MEMBER_LOGISTIC_CV:
        # Lane mf-split5: RidgeCV's split path (undefined R^2), a splitter's
        # warning (LogisticRegressionCV over StratifiedKFold).
        bits |= MEMBER_CAN_NOTICES
    if kind < 0:
        # THE GIVEN y (lane mf-tail5): some CV kind with a stratified splitter
        # reads it (`_reads_given` answers kind-exactly).
        bits |= MEMBER_READS_Y_GIVEN
    return bits


def _reads_given(kind: Int, words: _F64P, count: Int) -> Int:
    """`MEMBER_READS_Y_GIVEN` (core/member_abi.mojo, lane mf-tail5) of a
    kind and its NEW words: a regressor CV kind whose cv is a STRATIFIED
    splitter spec (RidgeCV word 3, LassoCV / ElasticNetCV word 6: minus the
    spec's handle) stratifies on its targets' values, so it reads the given
    y; every other kind and cv reads none (no caller gathers it for them).
    One spec read under its store's lock, once per member class (a
    capability question)."""
    var at = 3 if kind == XL_MEMBER_RIDGE_CV else (6 if _enetcv(kind) else -1)
    if at < 0 or count <= at or words[at] >= 0.0:
        return 0
    try:
        if msel_spec_stratified(msel_spec_copy(Int(-words[at])).kind):
            return MEMBER_READS_Y_GIVEN
    except:
        return 0
    return 0


def _posterior_kind(kind: Int) -> Bool:
    """The kinds whose fit holds a posterior (PREDICT_STD)."""
    return kind == XL_MEMBER_BAYES or kind == XL_MEMBER_ARD


def _classifier(kind: Int) -> Bool:
    return kind == XL_MEMBER_RIDGE_CLASSIFIER or kind == XL_MEMBER_LOGISTIC_CV


def _weighted(kind: Int) -> Bool:
    """Whether `kind`'s fit takes sample weights (kind -1: some kind does)."""
    return kind != XL_MEMBER_ARD and not _lars(kind)


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _targets(call: _CallP, kind: Int, n: Int, estimator: String, mut joined: List[Float32], mut has_sw: Int) raises -> Int:
    """`_with_weights`: the target words the fit reads (the targets alone,
    or the targets then the weights in `joined`). Returns their address;
    `has_sw` says which."""
    has_sw = 0
    if call[].weights.rows == 0:
        return call[].y.data
    if not _weighted(kind):
        raise Error("TypeError: " + estimator + ".fit() takes no sample_weight")
    _dense(call[].weights, "sample_weight", estimator)
    if call[].weights.rows != n or call[].weights.cols != 1:
        raise Error("ValueError: sample_weight and X have different numbers of rows")
    joined = List[Float32](length=2 * n, fill=0.0)
    unsafe_memcpy(dest=joined.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].y.data), count=n)
    unsafe_memcpy(dest=joined.unsafe_ptr() + n, src=_F32P(unsafe_from_address=call[].weights.data), count=n)
    has_sw = 1
    return Int(joined.unsafe_ptr())


def _keep_block(slot: _SlotP, result: List[Float32], d: Int):
    slot[].block = List[Float32](length=d + 1, fill=0.0)
    unsafe_memcpy(dest=slot[].block.unsafe_ptr(), src=result.unsafe_ptr(), count=d + 1)
    slot[].features = d


def _fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    var kind = _kind(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    var n = call[].x.rows
    var d = call[].x.cols
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member fits one target column")
    if call[].y.rows != n:
        raise Error("ValueError: X and y have different numbers of rows")
    if _enetcv(kind):
        # its weights ride after the fold ids (`_xl_enetcv_unpack`)
        var words = slot[].params.copy()
        slot[].forget()
        _enetcv_fit(slot, call, words, 1 if words[1] != 0.0 else 0, n, d)
        return
    var joined = List[Float32]()
    var has_sw = 0
    var y = _targets(call, kind, n, estimator, joined, has_sw)
    var y_size = n * (1 + has_sw)
    var p = slot[].params.copy()
    var intercept = 1 if p[1] != 0.0 else 0
    slot[].forget()
    if kind == XL_MEMBER_ISOTONIC:
        _isotonic_fit(slot, call, p, n, y, y_size, has_sw)
        _ = joined^
        return
    if kind == XL_MEMBER_HUBER:
        # The request words of bindings/x_linear_regressors_unpack.mojo
        # `xl_fit_huber_unpack`; the outlier flags come from the same session.
        var width = d + 2 if intercept == 1 else d + 1
        var ip: List[Int] = [Int(p[2]), intercept, has_sw]
        var fp: List[Float64] = [p[3], p[4], p[5]]
        var result = List[Float32](length=d + 4 + width, fill=0.0)
        var flags = List[UInt8](length=n, fill=0)
        var request = XlFitRequest(call[].route.entry.call, ALGO_HUBER, call[].x.data, n * d, n, d, y, y_size, ip^,
            fp^, d + 4 + width, xl_lbfgs_work(width) + n, 1, Int(result.unsafe_ptr()), False, 0, 0, 0)
        xl_huber_fit_run(request, call[].y.data, Int(flags.unsafe_ptr()), LINK_IDENTITY, p[3])
        _ = joined^
        _keep_block(slot, result, d)
        slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), Float64(Int(result[d + 2]))]
        slot[].flags = flags^
        slot[].fitted = True
        return
    if kind == XL_MEMBER_RIDGE_CV:
        _ridge_cv_fit(slot, call, p, intercept, n, d, y, has_sw)
        _ = joined^
        return
    if kind >= XL_MEMBER_BAYES:
        _plain_fit(slot, call, kind, p, intercept, n, d, y, y_size, has_sw)
        _ = joined^
        return
    # GLM: the request words of `xl_fit_glm_unpack`.
    var m = d + 1
    var ip: List[Int] = [Int(p[2]), intercept, Int(p[3]), has_sw]
    var fp: List[Float64] = [p[4], p[5], p[6]]
    var result = List[Float32](length=d + 3, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_GLM, call[].x.data, n * d, n, d, y, y_size, ip^, fp^,
        d + 3, 3 * n + m * m + 3 * m, 1, Int(result.unsafe_ptr()), False, 0, 0, 0)
    _ = xl_fit_run(request)
    _ = joined^
    if result[d + 2] < 0:
        raise Error("Some value(s) of y are out of the valid range of the loss '" + _loss_name(slot) + "'.")
    _keep_block(slot, result, d)
    slot[].scalars = [Float64(result[d]), Float64(Int(result[d + 1]))]
    slot[].fitted = True


def _ridge_cv_fit(slot: _SlotP, call: _CallP, p: List[Float64], intercept: Int, n: Int, d: Int, y: Int,
    has_sw: Int) raises:
    """RidgeCV.fit, the door's three paths (bindings/x_linear_cv_unpack.mojo
    `xl_fit_ridge_cv_unpack`): cv=None (`xl_ridge_cv_plain_run`, the grid in
    one fit); an integer cv, unweighted with at least two rows a fold
    (`xl_ridge_cv_kfold_run`); any other cv, a weighted integer cv included
    (lane mf-split5: the split path, `_ridge_cv_splits`). Word 3 is 0 for
    cv=None, else minus the handle of the cv's splitter spec (an integer cv
    is an unshuffled KFold spec marked `int_cv`)."""
    var grid = Int(p[4])
    var alphas = List[Float64](length=grid, fill=0.0)
    for i in range(grid):  # driver-loop: over the alpha grid
        alphas[i] = p[5 + i]
    var store = p[2] != 0.0
    var none = entry_ref_unresolved()
    if p[3] == 0.0:
        var result = List[Float32](length=xl_ridge_out(d, grid), fill=0.0)
        var plain = XlRidgeCvRequest(call[].route.entry.call, call[].x.data, n, d, y, has_sw == 1, intercept == 1,
            alphas^, Int(result.unsafe_ptr()), 0, 0, 0, 0, none.call, none)
        xl_ridge_cv_plain_run(plain)
        if xl_ridge_singular(Int(result.unsafe_ptr()), d, grid):
            raise Error("mojolearn RidgeCV: X'X + alpha I is singular even in float-float (alpha="
                + String(Float64(result[d + 1])) + "); rescale X or use a larger alpha")
        _keep_block(slot, result, d)
        slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), Float64(result[d + 2])]
        if store:
            slot[].record = List[Float32](length=grid, fill=0.0)
            unsafe_memcpy(dest=slot[].record.unsafe_ptr(), src=result.unsafe_ptr() + d + 3, count=grid)
        slot[].trained = True
        slot[].fitted = True
        return
    var spec = msel_spec_copy(Int(-p[3]))
    var folds = spec.k
    if spec.int_cv and has_sw == 0 and folds >= 2 and n // folds >= 2:
        # `_fit_kfold`: the k-fold scores on the binding, the first best
        # finite score chosen in Mojo (the door's choice of path, by the same
        # rule: an unweighted integer cv with at least two rows a fold)
        if not (2 <= folds and folds <= n):
            raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
        var scores = List[Float32](length=grid, fill=0.0)
        var result = List[Float32](length=xl_ridge_out(d, 1), fill=0.0)
        var chosen = alphas.copy()
        var kfold = XlRidgeCvRequest(call[].route.entry.call, call[].x.data, n, d, y, False, intercept == 1, alphas^,
            Int(result.unsafe_ptr()), folds, Int(scores.unsafe_ptr()), 0, 0, none.call, none)
        var best = xl_ridge_cv_kfold_run(kfold)
        if xl_ridge_singular(Int(result.unsafe_ptr()), d, 1):
            raise Error("mojolearn RidgeCV: X'X + alpha I is singular even in float-float (alpha="
                + String(Float64(result[d + 1])) + "); rescale X or use a larger alpha")
        _keep_block(slot, result, d)
        # alpha_ is the user's own value; best_score_ that alpha's k-fold score
        slot[].scalars = [Float64(result[d]), chosen[best], Float64(scores[best])]
        slot[].trained = True
        slot[].fitted = True
        return
    if spec.int_cv and not (2 <= folds and folds <= n):
        raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
    _ridge_cv_splits(slot, call, spec, alphas^, intercept, n, d, y, has_sw)


def _ridge_cv_splits(slot: _SlotP, call: _CallP, spec: MselSplitSpec, var alphas: List[Float64], intercept: Int,
    n: Int, d: Int, y: Int, has_sw: Int) raises:
    """RidgeCV's split path (`_fit_splits`, lane mf-split5): the splits of the
    cv over this fit's rows (`msel_spec_sides`, the splitter door's plans; an
    integer cv: KFold's unshuffled folds, which the door makes by the same
    plan, bindings/x_linear_cv_unpack.mojo `_xl_default_folds`), then the
    door's ONE RUN `xl_ridge_cv_splits_run` (a fit, a prediction and an R^2
    per split and alpha), its choice and refit `xl_ridge_cv_choose_run`. A
    split with fewer than two test rows is scikit-learn's undefined R^2: one
    fit notice each (MEMBER_NOTICE_R2_FEW, raised by the caller's PACK in the
    door's words), every mean NaN, the first alpha. A stop is refused in the
    door's words. NUMERICALLY this is the door's split path, not its k-fold
    path: per split R^2 means (the k-fold path scores every fold on the
    binding in one call); the door takes the split path for any cv but an
    unweighted integer one with two rows a fold, and so does this member."""
    var estimator = _name(slot)
    if not call[].route.has_fifth:
        raise Error("ImportError: mojolearn RidgeCV: the split path scores each split with the metrics image's "
            + "R^2 entry, which is not loaded")
    # a regressor's targets: a stratified splitter stratifies on their values, as the door's split (lane mf-del5)
    # (lane mf-tail5: with THE GIVEN y, its values; core/member_abi.mojo `MemberCall.y_given`)
    var sides = msel_spec_sides_targets(spec, n, call[].y.data, _given_at(call), call[].y_given.dtype, MselGroupCodes())
    var k = sides.splits()
    if k == 0:
        raise Error("mojolearn: cv produced no (train, test) split")
    var words = _split_words_of(sides)
    var grid = len(alphas)
    var table = List[Float64](length=max(grid * k, 1), fill=0.0)
    var means = List[Float64](length=max(grid, 1), fill=0.0)
    var result = List[Float32](length=xl_ridge_out(d, 1), fill=0.0)
    var fit_words = List[Float32](length=xl_ridge_out(d, 1), fill=0.0)
    var status_word = List[Int32](length=1, fill=Int32(0))
    var chosen = alphas.copy()
    var request = XlRidgeCvRequest(call[].route.entry.call, call[].x.data, n, d, y, has_sw == 1, intercept == 1,
        alphas^, Int(result.unsafe_ptr()), 0, Int(table.unsafe_ptr()), Int(means.unsafe_ptr()), k,
        call[].route.fifth.call, spec.rows_helpers)
    if call[].x.data < 0:
        # THE FOLD X WORD (lane vdf-scope-linear3): the split path takes each
        # split's rows from the fold slot on the device (x_linear/door_run.mojo
        # `_xl_split_take`) on the same x_prep store.
        request.fold_store = member_fold_store(call)
    var ran = xl_ridge_cv_splits_run(request, words, Int(fit_words.unsafe_ptr()), Int(status_word.unsafe_ptr()),
        estimator)
    var undefined = False
    for s in range(len(ran.undefined)):  # driver-loop: over the cv splits reached (the undefined flag)
        if ran.undefined[s]:
            undefined = True
    if ran.stop_kind == XL_RIDGE_SPLITS_EMPTY:
        raise Error("mojolearn RidgeCV: every cv split needs training and test rows")
    if ran.stop_kind == XL_RIDGE_SPLITS_SINGULAR:
        raise Error("mojolearn RidgeCV: X'X + alpha I is singular even in float-float (alpha="
            + String(Float64(fit_words[d + 1])) + "); rescale X or use a larger alpha")
    if ran.stop_kind == XL_RIDGE_SPLITS_ERROR:
        raise Error(ran.error)
    var best = xl_ridge_cv_choose_run(request, undefined)
    _ = len(sides.train)  # the RUN read every side through its address
    _ = len(table)
    _ = len(status_word)
    if xl_ridge_singular(Int(result.unsafe_ptr()), d, 1):
        raise Error("mojolearn RidgeCV: X'X + alpha I is singular even in float-float (alpha="
            + String(Float64(result[d + 1])) + "); rescale X or use a larger alpha")
    _keep_block(slot, result, d)
    var nan = Float64(0.0) / Float64(0.0)
    slot[].scalars = [Float64(result[d]), chosen[best], nan if undefined else means[best]]
    for s in range(len(ran.undefined)):  # driver-loop: over the cv splits reached (one notice per undefined R^2)
        if ran.undefined[s]:
            member_notice_add(slot[].notices, MEMBER_NOTICE_R2_FEW)
    slot[].cv_folds = k
    slot[].trained = True
    slot[].fitted = True


def _split_words_of(sides: MselSides) -> XlSplitWords:
    """The splits' int64 sides as the x_linear RUNs read them: each side's
    address (0 when empty) and count. `sides` stays alive in the caller."""
    var words = XlSplitWords(List[Int](), List[Int](), List[Int](), List[Int]())
    for s in range(sides.splits()):  # driver-loop: over the cv splits (their addresses)
        var tr = len(sides.train[s])
        var te = len(sides.test[s])
        words.train.append(Int(sides.train[s].unsafe_ptr()) if tr > 0 else 0)
        words.train_sizes.append(tr)
        words.test.append(Int(sides.test[s].unsafe_ptr()) if te > 0 else 0)
        words.test_sizes.append(te)
    return words^


def _cv_fold_words(spec: MselSplitSpec, sides: MselSides, n: Int, estimator: String, ids_dst: Int,
    mut roles: List[Float32]) raises:
    """`_cv_fold_ids_f32(splits, n, name)` (bindings/x_linear_cv_unpack.mojo
    `_xl_cv_fold_ids_f32`, the same two RUNs on the same base helper table):
    a partition's fold of every row as float32 at `ids_dst` (roles empty);
    otherwise -(n + 1) there and each split's train (1) / test (2) row marks
    in `roles` (splits x n float32)."""
    var words = _split_words_of(sides)
    var ids = List[Int32](length=max(n, 1), fill=Int32(-1))
    var partition = xl_cv_fold_ids_run(spec.rows_helpers, words, n, Int(ids.unsafe_ptr()))
    if partition:
        _codes_f32(Int(ids.unsafe_ptr()), ids_dst, n)
        _ = len(ids)
        _ = len(sides.train)
        return
    xl_cv_roles_limit(n, estimator)
    roles = List[Float32](length=max(sides.splits() * n, 1), fill=0.0)
    xl_cv_roles_run(spec.rows_helpers, words, n, Int(roles.unsafe_ptr()), estimator)
    _ = len(sides.train)  # the RUNs read every side through its address
    roles.resize(sides.splits() * n, 0.0)
    _fill_f32(ids_dst, n, Float32(-(n + 1)))


def _given_at(call: _CallP) -> Int:
    """THE GIVEN y's address (core/member_abi.mojo `MemberCall.y_given`,
    lane mf-tail5): a dense host column as long as the fit's y, else 0 (the
    float32 targets are the given y)."""
    var g = call[].y_given
    if g.given() and g.home == MEMBER_HOME_HOST and g.rows == call[].y.rows and g.cols == 1 and g.stride == 1 and g.offset == 0:
        return g.data
    return 0


def _cv_sides(word: Float64, n: Int, codes: Int, classes: Int, targets: Int, given: Int, given_dtype: Int,
              mut spec: MselSplitSpec) raises -> MselSides:
    """THE SPLITTER CV of an x_linear CV member (lane mf-split5): its spec
    (word: minus its handle) and the splits of this fit's rows, by the
    splitter door's plans (`msel_spec_sides`; `codes`: the classifier's int32
    class codes over `classes`, 0 for a regressor, whose float32 `targets`
    a stratified splitter stratifies on, as the door's split does:
    `msel_spec_sides_targets`, lane mf-del5; `given`: THE GIVEN y, lane
    mf-tail5, 0: none). No split: the door's
    `_xl_cv_splits` words."""
    spec = msel_spec_copy(Int(-word))
    var sides: MselSides
    if codes == 0:
        sides = msel_spec_sides_targets(spec, n, targets, given, given_dtype, MselGroupCodes())
    else:
        sides = msel_spec_sides(spec, n, codes, classes, MselGroupCodes())
    if sides.splits() == 0:
        raise Error("mojolearn: cv produced no (train, test) split")
    return sides^


def _codes_f32(src: Int, dst: Int, n: Int):
    """The n int32 class codes at `src` as float32 words at `dst` (the
    codes' `labels_cast`), rows split across the host's workers."""
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var source = _I32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(worker: Int) {imm chunk, imm source, imm target, imm n}:
        var begin = worker * chunk
        var end = min(begin + chunk, n)
        for i in range(begin, end):
            target[i] = source[i].cast[DType.float32]()

    host_parallelize(task, tasks)


def _fill_f32(dst: Int, count: Int, value: Float32):
    """`count` float32 words of `value` at `dst`, split across the host's
    workers (`labels_full_f32`)."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var target = _F32P(unsafe_from_address=dst)

    def task(worker: Int) {imm chunk, imm target, imm count, imm value}:
        var begin = worker * chunk
        var end = min(begin + chunk, count)
        for i in range(begin, end):
            target[i] = value

    host_parallelize(task, tasks)


def _nearest(p: List[Float64], first: Int, count: Int, value: Float64) -> Int:
    """The first of the `count` grid words from `first` nearest `value`
    (the packs' user-value pick)."""
    var pick = 0
    var gap = abs(p[first] - value)
    for i in range(1, count):  # driver-loop: over the grid (model structure)
        var g = abs(p[first + i] - value)
        if g < gap:
            gap = g
            pick = i
    return pick


def _enetcv_fit(slot: _SlotP, call: _CallP, p: List[Float64], intercept: Int, n: Int, d: Int) raises:
    """LassoCV / ElasticNetCV.fit (`_enetcv_fit`): the request words of
    bindings/x_linear_cv_unpack.mojo `_xl_enetcv_unpack` over the target
    words [y | fold ids | weights (| per-split roles)], one `xl_fit_run`
    call, then the result words [coef | intercept | alpha_ | l1_ratio |
    n_iter | alphas_ | mse_path_]. Word 6: cv=None or an integer cv's fold
    count (KFold's contiguous folds, made here), or minus the handle of the
    cv's splitter spec (lane mf-split5: one of our splitters or an index
    list; its splits over this fit's rows, a partition's fold ids or, when
    the splits are not a partition, the per-split roles riding the weighted
    arm, exactly as the door's `_xl_cv_fold_ids_f32`)."""
    var estimator = _name(slot)
    var folds = Int(p[6])
    var grid = Int(p[7])
    var explicit = p[8] != 0.0
    var ratios = Int(p[9])
    var given = call[].weights.rows > 0
    if given:
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != n or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
    var ids = List[Float32](length=max(n, 1), fill=0.0)
    var roles = List[Float32]()
    var spec = MselSplitSpec()
    if p[6] < 0.0:
        var sides = _cv_sides(p[6], n, 0, 0, call[].y.data, _given_at(call), call[].y_given.dtype, spec)
        folds = sides.splits()
        _cv_fold_words(spec, sides, n, estimator, Int(ids.unsafe_ptr()), roles)
    else:
        if folds < 2:
            raise Error("mojolearn: cv must be None or an int in [2, n_samples]")
        if folds > n:
            raise Error("Cannot have number of splits n_splits=" + String(folds) + " greater than the "
                + "number of samples: n_samples=" + String(n) + ".")
        xl_kfold_ids_f32_run(Int(ids.unsafe_ptr()), n, folds)  # the door's body (x_linear/cv_fold_ids_run.mojo)
    var listed_roles = len(roles) > 0
    var weighted = 1 if (given or listed_roles) else 0
    var y_size = (2 + weighted) * n + len(roles)
    var parts = List[Float32](length=y_size, fill=0.0)
    unsafe_memcpy(dest=parts.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].y.data), count=n)
    unsafe_memcpy(dest=parts.unsafe_ptr() + n, src=ids.unsafe_ptr(), count=n)
    if weighted == 1:
        if given:
            unsafe_memcpy(dest=parts.unsafe_ptr() + 2 * n, src=_F32P(unsafe_from_address=call[].weights.data), count=n)
        else:
            _fill_f32(Int(parts.unsafe_ptr()) + 4 * 2 * n, n, 1.0)  # per-split roles ride the weighted arm
    if listed_roles:
        unsafe_memcpy(dest=parts.unsafe_ptr() + 3 * n, src=roles.unsafe_ptr(), count=len(roles))
    _ = len(ids)
    _ = len(roles)
    var ip: List[Int] = [Int(p[2]), intercept, grid, folds, ratios, 1 if explicit else 0, 1 if p[5] != 0.0 else 0,
        weighted]
    var fp: List[Float64] = [p[3], p[4]]
    var listed = ratios + (grid if explicit else 0)
    for i in range(listed):  # driver-loop: over the l1_ratio and alpha grids
        fp.append(p[10 + i])
    var paths = ratios * grid
    var n_out = d + 4 + paths + paths * folds
    var result = List[Float32](length=n_out, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_ENETCV, call[].x.data, n * d, n, d,
        Int(parts.unsafe_ptr()), y_size, ip^, fp^, n_out, d * d + 4 * d + 3 + grid * (d + 2), 1,
        Int(result.unsafe_ptr()), False, 0, 0, 0)
    _ = xl_fit_run(request)
    _ = parts^
    _keep_block(slot, result, d)
    # l1_ratio_: the user's own grid entry nearest the returned float32 word
    var pick = _nearest(p, 10, ratios, Float64(result[d + 2]))
    slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), p[10 + pick], Float64(Int(result[d + 3]))]
    slot[].record = List[Float32](length=paths, fill=0.0)
    unsafe_memcpy(dest=slot[].record.unsafe_ptr(), src=result.unsafe_ptr() + d + 4, count=paths)
    slot[].path = List[Float32](length=paths * folds, fill=0.0)
    unsafe_memcpy(dest=slot[].path.unsafe_ptr(), src=result.unsafe_ptr() + d + 4 + paths, count=paths * folds)
    slot[].cv_folds = folds
    slot[].trained = True
    slot[].fitted = True


def _isotonic_fit(slot: _SlotP, call: _CallP, p: List[Float64], n: Int, y: Int, y_size: Int, has_sw: Int) raises:
    """IsotonicRegression.fit: the request words of
    bindings/x_linear_isotonic_unpack.mojo `xl_isotonic_fit_unpack`, the
    direction and the one fit (`xl_isotonic_fit_run`), then the result
    words [k | X_min_ | X_max_ | X_thresholds_ (n slots) | y_thresholds_]."""
    if call[].x.cols != 1:
        raise Error("ValueError: Isotonic regression input X should be a 1d array or 2d array with 1 feature")
    var increasing = Int(p[1])
    var ip: List[Int] = [1 if increasing == 1 else 0, Int(p[2]), Int(p[4]), has_sw]
    var fp: List[Float64] = [p[3], p[5]]
    var result = List[Float32](length=3 + 2 * n, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_ISOTONIC, call[].x.data, n, n, 1, y, y_size, ip^,
        fp^, 3 + 2 * n, 6 * n, 3 * n, Int(result.unsafe_ptr()), False, 0, 0, 0)
    var rising = xl_isotonic_fit_run(request, increasing == 2, call[].y.data)
    var k = Int(result[0])
    slot[].scalars = [1.0 if rising else 0.0, Float64(result[1]), Float64(result[2])]
    # the joined knots [X_thresholds_ | y_thresholds_] the predict reads
    slot[].record = List[Float32](length=2 * k, fill=0.0)
    unsafe_memcpy(dest=slot[].record.unsafe_ptr(), src=result.unsafe_ptr() + 3, count=k)
    unsafe_memcpy(dest=slot[].record.unsafe_ptr() + k, src=result.unsafe_ptr() + 3 + n, count=k)
    slot[].features = 1
    slot[].trained = True
    slot[].fitted = True


def _isotonic_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    """IsotonicRegression.predict / transform over the held knots
    (`_xl_isotonic_transform_build`): one ALGO_ISOTONIC_PREDICT call; with
    out_of_bounds='raise' the flag word after the predictions refuses."""
    var n = call[].x.rows
    var k = len(slot[].record) // 2
    var oob = Int(slot[].params[6])
    var raise_oob = oob == 2
    # 'raise': the predictions and the flag word (zero) in one buffer, as
    # the estimator's own door; otherwise straight into dst
    var spill = List[Float32](length=n + 1 if raise_oob else 0, fill=0.0)
    var target = Int(spill.unsafe_ptr()) if raise_oob else dst
    var ip: List[Int] = [k, oob]
    var fp: List[Float64] = [slot[].scalars[1], slot[].scalars[2]]
    var request = XlFitRequest(call[].route.entry.call, ALGO_ISOTONIC_PREDICT, call[].x.data, n, n, 1,
        Int(slot[].record.unsafe_ptr()), 2 * k, ip^, fp^, n + 1 if raise_oob else n, 1, 1, target, False, 0, 0, 0)
    _ = xl_fit_run(request)
    if raise_oob:
        if spill[n] != 0.0:
            raise Error("A value in x_new is below/above the interpolation range.")
        unsafe_memcpy(dest=_F32P(unsafe_from_address=dst), src=spill.unsafe_ptr(), count=n)
    _ = spill^


def _plain_fit(slot: _SlotP, call: _CallP, kind: Int, p: List[Float64], intercept: Int, n: Int, d: Int, y: Int,
    y_size: Int, has_sw: Int) raises:
    """Kinds 4..: one `xl_fit_run` call with the request words of the kind's
    `xl_fit_<kind>_unpack` (bindings/x_linear_regressors_unpack.mojo), then
    the kind's result words into the state."""
    var algo = ALGO_BAYES
    var ip = List[Int]()
    var fp = List[Float64]()
    var n_out = 0
    var n_fw = 0
    var n_iw = 1
    if kind == XL_MEMBER_BAYES:
        ip = [Int(p[2]), intercept, has_sw]
        fp = [p[3], p[4], p[5], p[6], p[7], p[8], p[9]]
        n_out = bayes_out_words(d)
        n_fw = 3 * d * d + 5 * d + n
    elif kind == XL_MEMBER_ARD:
        if n < 2:
            raise Error("mojolearn ARDRegression: at least 2 samples are required")
        algo = ALGO_ARD
        ip = [Int(p[2]), intercept]
        fp = [p[3], p[4], p[5], p[6], p[7], p[8]]
        n_out = ard_out_words(d)
        n_fw = 3 * d * d + 4 * d + n
        n_iw = 2 * d
    elif kind == XL_MEMBER_QUANTILE:
        # `xl_fit_quantile_unpack`: fp [quantile, alpha, tol / 100, tol, 1 (unread)]
        algo = ALGO_QUANTILE
        var m = d + 1
        ip = [Int(p[2]), intercept, has_sw]
        fp = [p[3], p[4], p[5] / 100.0, p[5], 1.0]
        n_out = d + 3
        n_fw = m * m + 3 * m + 4 * n + 3 * d
    elif _lars(kind):
        # `xl_fit_lars_unpack`: [n_nonzero_coefs, fit_intercept, lasso 0, positive], [alpha_min 0]
        algo = ALGO_LARS
        ip = [Int(p[2]), intercept, 0, Int(p[3])]
        fp = [0.0]
        n_out = 2 * d + 4
        if kind == XL_MEMBER_LASSO_LARS:
            # [max_iter, fit_intercept, lasso 1, positive], [alpha]
            ip = [Int(p[2]), intercept, 1, Int(p[3])]
            fp = [p[4]]
        n_fw = 2 * d * d + 8 * d
        n_iw = 2 * d
    var result = List[Float32](length=n_out, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, algo, call[].x.data, n * d, n, d, y, y_size, ip^, fp^,
        n_out, n_fw, n_iw, Int(result.unsafe_ptr()), False, 0, 0, 0)
    _ = xl_fit_run(request)
    _keep_block(slot, result, d)
    if kind == XL_MEMBER_BAYES:
        # [coef | intercept | alpha_ | lambda_ | n_iter]
        slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), Float64(result[d + 2]),
            Float64(Int(result[d + 3]))]
        # the posterior (x_linear/bayes_words.mojo): the factor V | ev (one
        # O(d^2) copy; sigma_ is never formed here), X_offset_
        slot[].factor = List[Float32](length=d * d + d, fill=0.0)
        if d > 0:
            unsafe_memcpy(dest=slot[].factor.unsafe_ptr(), src=result.unsafe_ptr() + bayes_v_at(d), count=d * d + d)
        _keep_posterior(slot, result, 0, 0, bayes_offset_at(d), d)
        slot[].dk = d
    elif kind == XL_MEMBER_ARD:
        # [coef | intercept | alpha_ | lambda_ (d) | n_iter | dk | X_offset_ | sigma_ dk x dk]
        slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), Float64(Int(result[2 * d + 2]))]
        slot[].record = List[Float32](length=d, fill=0.0)
        unsafe_memcpy(dest=slot[].record.unsafe_ptr(), src=result.unsafe_ptr() + d + 2, count=d)
        _keep_posterior(slot, result, ard_sigma_at(d), Int(result[ard_dk_at(d)]), ard_offset_at(d), d)
    elif kind == XL_MEMBER_QUANTILE:
        # [coef | intercept | n_iter]
        slot[].scalars = [Float64(result[d]), Float64(Int(result[d + 1]))]
    else:
        # LARS: [coef | intercept | n_iter | alpha_ | k | k active indices]
        slot[].scalars = [Float64(result[d]), Float64(Int(result[d + 1])), Float64(result[d + 2])]
        var k = max(Int(result[d + 3]), 0)
        slot[].active = List[Int32](length=k, fill=0)
        for i in range(k):  # driver-loop: over the active features (model structure)
            slot[].active[i] = Int32(Int(result[d + 4 + i]))
    slot[].trained = True
    slot[].fitted = True


def _keep_posterior(slot: _SlotP, result: List[Float32], sigma_at: Int, dk: Int, offset_at: Int, d: Int):
    """The fit's posterior words into the state (one copy each, at fit),
    and for ARD the kept columns (`_ard_kept`)."""
    slot[].sigma = List[Float32](length=dk * dk, fill=0.0)
    if dk > 0:
        unsafe_memcpy(dest=slot[].sigma.unsafe_ptr(), src=result.unsafe_ptr() + sigma_at, count=dk * dk)
    slot[].offset = List[Float32](length=d, fill=0.0)
    if d > 0:
        unsafe_memcpy(dest=slot[].offset.unsafe_ptr(), src=result.unsafe_ptr() + offset_at, count=d)
    slot[].dk = dk
    _ard_kept(slot)


def _ard_kept(slot: _SlotP):
    """ARDRegression's kept columns from its lambda_ record (the fit's rule,
    x_linear/door_run.mojo `xl_ard_kept_run`); every column otherwise."""
    slot[].kept = List[Int32]()
    if _kind(slot) != XL_MEMBER_ARD or len(slot[].record) == 0:
        return
    var d = len(slot[].record)
    var room = List[Int32](length=d, fill=0)
    var count = xl_ard_kept_run(Int(slot[].record.unsafe_ptr()), d, slot[].params[8], Int(room.unsafe_ptr()))
    slot[].kept = List[Int32](length=count, fill=0)
    if count > 0:
        unsafe_memcpy(dest=slot[].kept.unsafe_ptr(), src=room.unsafe_ptr(), count=count)


def _predict_std(slot: _SlotP, call: _CallP) raises:
    """PREDICT_STD (BayesianRidge, ARDRegression): dst rows x 1 float32 the
    predictive standard deviation (XL_OP_BAYES_STD); with `y` given (rows x
    1 host float32) the same rows' mean too, as predict(return_std=True)
    returns both (IterativeImputer's sample_posterior reads the pair)."""
    var estimator = _name(slot)
    if not _posterior_kind(_kind(slot)):
        member_unsupported(estimator, MEMBER_OP_PREDICT_STD)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _dense(call[].dst, "the standard deviation buffer", estimator)
    var rows = call[].x.rows
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the standard deviation buffer holds one value per row of X")
    var d = slot[].features
    if not _has_posterior(slot):
        raise Error("NotFittedError: this " + estimator + " member state holds no posterior (X_offset_ and "
            + ("sigma_" if _kind(slot) == XL_MEMBER_ARD else "the posterior factor") + ")")
    var ard = _kind(slot) == XL_MEMBER_ARD
    if ard and len(slot[].kept) != slot[].dk:
        raise Error("ValueError: this " + estimator + " member state's sigma_ does not match its kept features")
    if call[].y.rows > 0:
        _dense(call[].y, "the mean buffer", estimator)
        if call[].y.rows != rows or call[].y.cols != 1:
            raise Error("ValueError: the mean buffer holds one value per row of X")
        _predict_into(slot, call, call[].y.data)
    else:
        # `_predict_into`'s view and width checks, without its decision
        _whole(call, estimator)
        _dense(call[].x, "X", estimator)
        if call[].x.cols != d:
            raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
                + " is expecting " + String(d) + " features as input.")
    if rows == 0:
        return
    # ARD: sigma_ over its kept columns; BayesianRidge: the factor V | ev,
    # alpha_ and lambda_ (sigma_ is never formed)
    var s = Int(slot[].sigma.unsafe_ptr()) if ard else Int(slot[].factor.unsafe_ptr())
    var ev = 0 if ard else Int(slot[].factor.unsafe_ptr()) + 4 * d * d
    var lam = 0.0 if ard else slot[].scalars[2]
    xl_bayes_std_run(XlBayesStdRequest(call[].route.entry.call, call[].x.data, s, ev,
        Int(slot[].offset.unsafe_ptr()), Int(slot[].kept.unsafe_ptr()) if ard else 0, call[].dst.data, rows, d,
        slot[].dk if ard else d, not ard, slot[].scalars[1], lam))


def _predict_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    if call[].x.rows == 0:
        return
    if _kind(slot) == XL_MEMBER_ISOTONIC:
        _isotonic_into(slot, call, dst)
        return
    var link = LINK_EXP if _kind(slot) <= 2 and slot[].params[3] == 1.0 else LINK_IDENTITY
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION, call[].x.data,
        Int(slot[].block.unsafe_ptr()), dst, call[].x.rows, call[].x.cols, 1, link, True, 0, 0, 0, 0))


def _predict(slot: _SlotP, call: _CallP) raises:
    _dense(call[].dst, "the prediction buffer", _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    _predict_into(slot, call, call[].dst.data)


def _score(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    if _kind(slot) == XL_MEMBER_ISOTONIC:
        member_unsupported(estimator, MEMBER_OP_SCORE)
    var route = call[].route
    _dense(call[].y, "y", estimator)
    if call[].y.cols != 1 or call[].y.rows != call[].x.rows:
        raise Error("ValueError: y must be 1-D with one entry per row of X")
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
            + " member is not published; the caller predicts and runs its own weighted metric")
    if not route.has_fifth:
        raise Error("ImportError: mojolearn: the " + estimator + " member's score needs the metrics R^2 entry on its route")
    var prediction = List[Float32](length=call[].x.rows, fill=0.0)
    _predict_into(slot, call, Int(prediction.unsafe_ptr()))
    call[].value = xl_r2_run(route.fifth.call, entry_ref_unresolved(), call[].y.data, Int(prediction.unsafe_ptr()),
        0, call[].x.rows, True)
    _ = prediction^


def _outlier_words(slot: _SlotP):
    """The int32 words of the outlier flags, made once, rows split across
    the host's workers."""
    var rows = len(slot[].flags)
    if len(slot[].outliers) == rows:
        return
    slot[].outliers = List[Int32](length=rows, fill=0)
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _U8P(unsafe_from_address=Int(slot[].flags.unsafe_ptr()))
    var target = _I32P(unsafe_from_address=Int(slot[].outliers.unsafe_ptr()))

    def task(worker: Int) {imm chunk, imm source, imm target, imm rows}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        for r in range(begin, end):
            target[r] = Int32(1) if source[r] != 0 else Int32(0)

    host_parallelize(task, tasks)


def _family_block(slot: _SlotP) -> Int:
    """The name of the one family block this state holds after the two
    shared ones, 0 for none (an imported state holds no training record)."""
    if _kind(slot) == XL_MEMBER_HUBER and len(slot[].flags) > 0:
        return XL_MEMBER_BLOCK_OUTLIERS
    if _kind(slot) == XL_MEMBER_ARD and len(slot[].record) > 0:
        return XL_MEMBER_BLOCK_LAMBDA
    if _lars(_kind(slot)) and slot[].trained:
        return XL_MEMBER_BLOCK_ACTIVE
    if _kind(slot) == XL_MEMBER_RIDGE_CV and len(slot[].record) > 0:
        return XL_MEMBER_BLOCK_CV_RESULTS
    return 0


def _held_blocks(slot: _SlotP) -> List[Int]:
    """The names of the blocks this regressor state holds, in index order."""
    var kind = _kind(slot)
    var names = List[Int]()
    if kind == XL_MEMBER_ISOTONIC:
        names.append(MEMBER_BLOCK_SCALARS)
        names.append(XL_MEMBER_BLOCK_THRESHOLDS)
        return names^
    names.append(MEMBER_BLOCK_COEF)
    names.append(MEMBER_BLOCK_SCALARS)
    if _enetcv(kind):
        if slot[].trained:
            names.append(XL_MEMBER_BLOCK_ALPHAS)
            names.append(XL_MEMBER_BLOCK_MSE_PATH)
        return names^
    var family = _family_block(slot)
    if family != 0:
        names.append(family)
    if _has_posterior(slot):
        names.append(XL_MEMBER_BLOCK_X_OFFSET)
        names.append(XL_MEMBER_BLOCK_SIGMA if _kind(slot) == XL_MEMBER_ARD else XL_MEMBER_BLOCK_FACTOR)
    return names^


def _has_posterior(slot: _SlotP) -> Bool:
    """A BayesianRidge / ARDRegression state holding its posterior (after a
    fit, or an import that brought it): X_offset_ and BayesianRidge's
    factor or ARDRegression's sigma_."""
    var d = slot[].features
    if not _posterior_kind(_kind(slot)) or d == 0 or len(slot[].offset) != d:
        return False
    if _kind(slot) == XL_MEMBER_ARD:
        return len(slot[].sigma) == slot[].dk * slot[].dk
    return len(slot[].factor) == d * d + d


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block, owned by the slot until its next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var names = _held_blocks(slot)
    var count = len(names)
    var family = _family_block(slot)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= count:
            raise Error("member ABI: this " + _name(slot) + " member state has " + String(count) + " fitted blocks")
        block = names[block]
    var held = False
    for i in range(count):  # driver-loop: over the state's few block names
        held = held or names[i] == block
    if not held:
        raise Error("member ABI: this " + _name(slot) + " member state holds no fitted block " + String(block))
    if block == XL_MEMBER_BLOCK_ALPHAS:
        call[].dst = MemberMatrix(Int(slot[].record.unsafe_ptr()), Int(slot[].params[9]), Int(slot[].params[7]))
    elif block == XL_MEMBER_BLOCK_MSE_PATH:
        call[].dst = MemberMatrix(Int(slot[].path.unsafe_ptr()), Int(slot[].params[9]) * Int(slot[].params[7]),
            slot[].cv_folds)
    elif block == XL_MEMBER_BLOCK_THRESHOLDS:
        call[].dst = MemberMatrix(Int(slot[].record.unsafe_ptr()), 2, len(slot[].record) // 2)
    elif block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(Int(slot[].block.unsafe_ptr()), 1, slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block != 0 and block == family and family == XL_MEMBER_BLOCK_OUTLIERS:
        _outlier_words(slot)
        call[].dst = MemberMatrix(Int(slot[].outliers.unsafe_ptr()), 1, len(slot[].outliers), MEMBER_I32)
    elif block != 0 and block == family and (family == XL_MEMBER_BLOCK_LAMBDA or family == XL_MEMBER_BLOCK_CV_RESULTS):
        call[].dst = MemberMatrix(Int(slot[].record.unsafe_ptr()), 1, len(slot[].record))
    elif block != 0 and block == family and family == XL_MEMBER_BLOCK_ACTIVE:
        call[].dst = MemberMatrix(Int(slot[].active.unsafe_ptr()), 1, len(slot[].active), MEMBER_I32)
    elif block == XL_MEMBER_BLOCK_X_OFFSET:
        call[].dst = MemberMatrix(Int(slot[].offset.unsafe_ptr()), 1, len(slot[].offset))
    elif block == XL_MEMBER_BLOCK_SIGMA:
        call[].dst = MemberMatrix(Int(slot[].sigma.unsafe_ptr()), slot[].dk, slot[].dk)
    elif block == XL_MEMBER_BLOCK_FACTOR:
        call[].dst = MemberMatrix(Int(slot[].factor.unsafe_ptr()), slot[].features + 1, slot[].features)
    else:
        raise Error("member ABI: this " + _name(slot) + " member state holds no fitted block " + String(block))
    call[].block = block
    call[].out_cols = count


def _isotonic_import(slot: _SlotP, call: _CallP) raises:
    """IsotonicRegression: SCALARS and THRESHOLDS (2 x knots), then DONE."""
    var estimator = _name(slot)
    if call[].block == XL_MEMBER_BLOCK_THRESHOLDS:
        _dense(call[].x, "a fitted block", estimator)
        if call[].x.rows != 2:
            raise Error("member ABI: the IsotonicRegression thresholds block is 2 x knots float32 words")
        var count = 2 * call[].x.cols
        slot[].fitted = False
        slot[].record = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=slot[].record.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 3:
            raise Error("member ABI: the IsotonicRegression scalars block is 3 host float64 words")
        slot[].fitted = False
        slot[].scalars = List[Float64](length=3, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=3)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an IsotonicRegression member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != 3 or len(slot[].record) == 0:
        raise Error("member ABI: a fitted IsotonicRegression state needs its scalars and thresholds blocks")
    slot[].features = 1
    slot[].fitted = True


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    if _kind(slot) == XL_MEMBER_ISOTONIC:
        _isotonic_import(slot, call)
        return
    var scalar_count = xl_member_scalar_count(_kind(slot))
    if call[].block == MEMBER_BLOCK_COEF:
        _dense(call[].x, "a fitted block", _name(slot))
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count + 1, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        slot[].features = count
        slot[].block = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != scalar_count:
            raise Error("member ABI: the " + _name(slot) + " scalars block is " + String(scalar_count)
                + " host float64 words")
        var source = _F64P(unsafe_from_address=call[].x.data)
        slot[].fitted = False
        slot[].scalars = List[Float64](length=scalar_count, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=source, count=scalar_count)
    elif _posterior_kind(_kind(slot)) and (call[].block == XL_MEMBER_BLOCK_X_OFFSET
            or (_kind(slot) == XL_MEMBER_ARD and (call[].block == XL_MEMBER_BLOCK_SIGMA
                or call[].block == XL_MEMBER_BLOCK_LAMBDA))
            or (_kind(slot) == XL_MEMBER_BAYES and call[].block == XL_MEMBER_BLOCK_FACTOR)):
        # the posterior a predict_std reads (lane mf-iterimp2), and ARD's lambda_
        # (its kept columns); optional: a state without them predicts only
        _dense(call[].x, "a fitted block", _name(slot))
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        if call[].block == XL_MEMBER_BLOCK_SIGMA:
            if call[].x.rows != call[].x.cols:
                raise Error("member ABI: the " + _name(slot) + " sigma block is dk x dk float32 words")
            slot[].sigma = words^
            slot[].dk = call[].x.rows
        elif call[].block == XL_MEMBER_BLOCK_X_OFFSET:
            slot[].offset = words^
        elif call[].block == XL_MEMBER_BLOCK_FACTOR:
            if call[].x.rows != call[].x.cols + 1:
                raise Error("member ABI: the " + _name(slot) + " factor block is (d + 1) x d float32 words")
            slot[].factor = words^
            slot[].dk = call[].x.cols
        else:
            slot[].record = words^
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + _name(slot) + " member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].block) == 0 or len(slot[].scalars) != scalar_count:
        raise Error("member ABI: a fitted " + _name(slot) + " state needs its coefficient and scalars blocks")
    slot[].block[slot[].features] = Float32(slot[].scalars[0])
    _ard_kept(slot)
    slot[].fitted = True


# ------------------------------------------------------------- classifiers


def _codes_view(view: MemberMatrix, rows: Int, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_I32) or view.cols != 1 or view.rows != rows:
        raise Error("TypeError: a " + estimator + " member reads " + what
            + " as int32 class codes, one per row of X (a dense host view)")


def _keep_classes(slot: _SlotP, var record: List[Float32], outputs: Int, d: Int, classes: Int):
    """Hold the fit's [coef (T x d) | intercept (T)] words and build the
    decision block (T rows of [coef | intercept]) from them once."""
    slot[].block = List[Float32](length=outputs * (d + 1), fill=0.0)
    xl_run_block(Int(slot[].block.unsafe_ptr()), Int(record.unsafe_ptr()), Int(record.unsafe_ptr()) + 4 * outputs * d,
        outputs, d)
    slot[].record = record^
    slot[].outputs = outputs
    slot[].features = d
    slot[].classes = classes


def _cls_fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    var n = call[].x.rows
    var d = call[].x.cols
    _codes_view(call[].y, n, "y", estimator)
    var k = call[].classes
    if k < 2:
        raise Error("mojolearn " + estimator + ": y has one class")
    var outputs = 1 if k == 2 else k
    if _kind(slot) == XL_MEMBER_LOGISTIC_CV:
        var words = slot[].params.copy()
        slot[].forget()
        _logcv_fit(slot, call, words, n, d, k, outputs)
        return
    var joined = List[Float32]()
    var has_sw = 0
    var p = slot[].params.copy()
    var mode = member_cw_mode(p, 3)
    var class_weight = List[Float32]()
    var prep = XlClassPrepRequest(False, 0, 0, 0, 0, n, k, False, False)
    var y = 0
    if mode == MEMBER_CW_NONE:
        # `_ridge_run(..., codes_mode=True)`: the codes' 32-bit words, then the weights
        y = _targets(call, _kind(slot), n, estimator, joined, has_sw)
    else:
        # class_weight (lane mf-adm3), as `xl_fit_ridge_classifier_unpack`:
        # the codes' 32-bit words, then the n row weights the class step
        # writes in RUN (sample_weight x the class weights; 'balanced' from
        # the unweighted counts), checked as the weight admission checks
        # them (`xl_class_fit_run`).
        var sw = 0
        if call[].weights.rows > 0:
            _dense(call[].weights, "sample_weight", estimator)
            if call[].weights.rows != n or call[].weights.cols != 1:
                raise Error("ValueError: sample_weight and X have different numbers of rows")
            sw = call[].weights.data
        joined = List[Float32](length=2 * n, fill=0.0)
        unsafe_memcpy(dest=joined.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].y.data), count=n)
        var balanced = mode == MEMBER_CW_BALANCED
        class_weight = List[Float32](length=k, fill=1.0)
        if not balanced:
            class_weight = member_cw_table_f32(p, 3, k, "class_weight contains labels absent from y")
        prep = XlClassPrepRequest(True, call[].y.data, sw, Int(class_weight.unsafe_ptr()),
            Int(joined.unsafe_ptr()) + 4 * n, n, k, balanced, False)
        y = Int(joined.unsafe_ptr())
        has_sw = 1
    slot[].forget()
    var ip: List[Int] = [outputs, 1 if p[1] != 0.0 else 0, 1, has_sw, 1]
    var fp: List[Float64] = [p[2]]
    var n_out = outputs * d + outputs + 4
    var result = List[Float32](length=n_out, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_RIDGE, call[].x.data, n * d, n, d, y, n * (1 + has_sw),
        ip^, fp^, n_out, 3 * d * d + 3 * d + outputs + d * outputs + 2 * n, 1, Int(result.unsafe_ptr()), False, 0, 0,
        0)
    _ = xl_class_fit_run(request, prep)
    _ = joined^
    _ = class_weight^
    # status 2: X'X + alpha I does not factor even in float-float
    if result[outputs * d + outputs + 3] == 2.0:
        raise Error("mojolearn " + estimator + ": X'X + alpha I is singular even in float-float (alpha="
            + String(Float64(result[outputs * d + outputs])) + "); rescale X or use a larger alpha")
    var record = List[Float32](length=outputs * d + outputs, fill=0.0)
    unsafe_memcpy(dest=record.unsafe_ptr(), src=result.unsafe_ptr(), count=outputs * d + outputs)
    _keep_classes(slot, record^, outputs, d, k)
    slot[].trained = True
    slot[].fitted = True


def _logcv_fit(slot: _SlotP, call: _CallP, p: List[Float64], n: Int, d: Int, k: Int, kp: Int) raises:
    """LogisticRegressionCV.fit: the request words of
    bindings/x_linear_cv_unpack.mojo `xl_fit_logistic_cv_unpack` over the
    target words [codes | fold ids | fit weights | raw weights (| per-split
    roles)]; the class-weight step, the stratified-fold check and the fit are
    one `xl_logcv_fit_run`. Word 4: cv=None or an integer cv's fold count
    (the binding's stratified folds; the fold-id part is zeros), or minus
    the handle of the cv's splitter spec (lane mf-split5: our splitters and
    index lists, the door's EXPLICIT path: the splits of this fit's rows over
    its class codes, a partition's fold ids or the per-split roles, the
    explicit word, no stratified assignment; the weights part is present
    whenever roles are). Result words: [coef (kp x d) | intercept (kp) |
    best C | n_iter | scores (folds x C)]."""
    var estimator = _name(slot)
    var folds = Int(p[4])
    var explicit = p[4] < 0.0
    var ids = List[Float32](length=max(n, 1), fill=0.0)
    var roles = List[Float32]()
    var spec = MselSplitSpec()
    if explicit:
        var sides = _cv_sides(p[4], n, call[].y.data, k, 0, 0, 0, spec)
        folds = sides.splits()
        _cv_fold_words(spec, sides, n, estimator, Int(ids.unsafe_ptr()), roles)
        for w in range(len(sides.least)):  # small-loop(one notice per splitter warning: StratifiedKFold's)
            member_notice_add(slot[].notices, MEMBER_NOTICE_SPLIT_LEAST, Float64(sides.least[w]), Float64(spec.k))
    # word 5: 1 'balanced', 2 a dict (its table in the tail; lane mf-adm3)
    var balanced = p[5] == 1.0
    var keyed = p[5] == 2.0
    var nc = Int(p[6])
    var given = call[].weights.rows > 0
    if given:
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != n or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
    var weigh = given or balanced or keyed or len(roles) > 0
    var y_size = (4 if weigh else 2) * n + len(roles)
    var parts = List[Float32](length=y_size, fill=0.0)
    var base = Int(parts.unsafe_ptr())
    _codes_f32(call[].y.data, base, n)
    if explicit:
        unsafe_memcpy(dest=parts.unsafe_ptr() + n, src=ids.unsafe_ptr(), count=n)
    if len(roles) > 0:
        unsafe_memcpy(dest=parts.unsafe_ptr() + 4 * n, src=roles.unsafe_ptr(), count=len(roles))
    _ = len(ids)
    _ = len(roles)
    if weigh:
        # The fit weights, then the raw weights; with 'balanced' the
        # class-weight step writes the fit weights over the third n words.
        if given:
            unsafe_memcpy(dest=parts.unsafe_ptr() + 2 * n, src=_F32P(unsafe_from_address=call[].weights.data), count=n)
            unsafe_memcpy(dest=parts.unsafe_ptr() + 3 * n, src=_F32P(unsafe_from_address=call[].weights.data), count=n)
        else:
            _fill_f32(base + 4 * 2 * n, 2 * n, 1.0)
    var class_weight = List[Float32](length=k, fill=1.0)
    if keyed:
        # the door's `xl_class_weights_unpack` words, read at UNPACK
        class_weight = member_cw_table_f32(p, 7 + nc, k, "class_weight contains labels absent from y")
    var prep = XlClassPrepRequest(True, call[].y.data, call[].weights.data if given else 0,
        Int(class_weight.unsafe_ptr()), base + 4 * 2 * n if (balanced or keyed) else 0, n, k, balanced, True)
    var width = kp * (d + 1)
    var ip: List[Int] = [Int(p[2]), 1 if p[1] != 0.0 else 0, kp, nc, folds, 1 if weigh else 0]
    if explicit:
        ip.append(1)
    var fp: List[Float64] = [p[3]]
    for i in range(nc):  # driver-loop: over the Cs grid
        fp.append(p[7 + i])
    var n_out = kp * d + kp + 2 + folds * nc
    var result = List[Float32](length=n_out, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_LOGCV, call[].x.data, n * d, n, d, base, y_size, ip^,
        fp^, n_out, width + 1 + n * (kp + 1) + xl_lbfgs_work(width), 5, Int(result.unsafe_ptr()), False, 0, 0, 0)
    xl_logcv_fit_run(request, prep, not explicit, folds)
    _ = parts^
    _ = class_weight^
    var record = List[Float32](length=kp * d + kp, fill=0.0)
    unsafe_memcpy(dest=record.unsafe_ptr(), src=result.unsafe_ptr(), count=kp * d + kp)
    _keep_classes(slot, record^, kp, d, k)
    # C_: the grid entry nearest the returned best C (the first on a tie)
    var pick = _nearest(p, 7, nc, Float64(result[kp * d + kp]))
    slot[].scalars = [p[7 + pick], Float64(Int(result[kp * d + kp + 1]))]
    slot[].path = List[Float32](length=folds * nc, fill=0.0)
    unsafe_memcpy(dest=slot[].path.unsafe_ptr(), src=result.unsafe_ptr() + kp * d + kp + 2, count=folds * nc)
    slot[].cv_folds = folds
    slot[].trained = True
    slot[].fitted = True


def _cls_query(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")


def _cls_codes(slot: _SlotP, call: _CallP, dst: Int) raises:
    """Each row's int32 class code (`_LinearClassifierMixin.predict`: a
    binary score picks class 1 only when > 0, `strict`, as the own door)."""
    if call[].x.rows == 0:
        return
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION_CODES, call[].x.data,
        Int(slot[].block.unsafe_ptr()), dst, call[].x.rows, call[].x.cols, slot[].outputs, LINK_IDENTITY, True, 0, 1,
        0, 0))


def _cls_predict(slot: _SlotP, call: _CallP) raises:
    _cls_query(slot, call)
    _codes_view(call[].dst, call[].x.rows, "the prediction buffer", _name(slot))
    _cls_codes(slot, call, call[].dst.data)


def _cls_decision(slot: _SlotP, call: _CallP) raises:
    _cls_query(slot, call)
    _dense(call[].dst, "the decision buffer", _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != slot[].outputs:
        raise Error("ValueError: the decision buffer holds " + String(slot[].outputs) + " scores per row of X")
    if call[].x.rows == 0:
        return
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION, call[].x.data,
        Int(slot[].block.unsafe_ptr()), call[].dst.data, call[].x.rows, call[].x.cols, slot[].outputs, LINK_IDENTITY,
        True, 0, 0, 0, 0))


def _cls_proba(slot: _SlotP, call: _CallP) raises:
    """LogisticRegressionCV.predict_proba: the decision and its row link
    in one call (XL_OP_DECISION_ROWS, ROWS_LRCV_PROBA), rows x classes."""
    if _kind(slot) != XL_MEMBER_LOGISTIC_CV:
        member_unsupported(_name(slot), MEMBER_OP_PREDICT_PROBA)
    _cls_query(slot, call)
    var width = 2 if slot[].outputs == 1 else slot[].outputs
    _dense(call[].dst, "the probability buffer", _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != width:
        raise Error("ValueError: the probability buffer holds " + String(width) + " values per row of X")
    if call[].x.rows == 0:
        return
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION_ROWS, call[].x.data,
        Int(slot[].block.unsafe_ptr()), call[].dst.data, call[].x.rows, call[].x.cols, slot[].outputs, LINK_IDENTITY,
        False, 0, 0, ROWS_LRCV_PROBA, 0))


def _cls_score(slot: _SlotP, call: _CallP) raises:
    _cls_query(slot, call)
    var rows = call[].x.rows
    var estimator = _name(slot)
    _codes_view(call[].y, rows, "y", estimator)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
            + " member is not published; the caller predicts and runs its own weighted metric")
    if rows == 0:
        raise Error("ValueError: Found array with 0 sample(s) while a minimum of 1 is required.")
    var predicted = List[Int32](length=rows, fill=0)
    _cls_codes(slot, call, Int(predicted.unsafe_ptr()))
    # The exact count of equal codes (an integer, so the split cannot change it).
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var counts = List[Int](length=tasks, fill=0)
    var hits = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))
    var codes = _I32P(unsafe_from_address=Int(predicted.unsafe_ptr()))
    var truth = _I32P(unsafe_from_address=call[].y.data)

    def task(worker: Int) {imm hits, imm codes, imm truth, imm chunk, imm rows}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        var count = 0
        for r in range(begin, end):
            if truth[r] == codes[r]:
                count += 1
        hits[worker] = count

    host_parallelize(task, tasks)
    var total = 0
    for worker in range(tasks):  # driver-loop: over the workers' partial counts
        total += counts[worker]
    _ = predicted^
    call[].value = Float64(total) / Float64(rows)


def _cls_export(slot: _SlotP, call: _CallP) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var logcv = _kind(slot) == XL_MEMBER_LOGISTIC_CV
    # COEF, INTERCEPT; LogisticRegressionCV also SCALARS and, after a fit, SCORES
    var count = 2 if not logcv else (4 if len(slot[].path) > 0 else 3)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= count:
            raise Error("member ABI: a " + _name(slot) + " member state has " + String(count) + " fitted blocks")
        block = MEMBER_BLOCK_COEF if block == 0 else (MEMBER_BLOCK_INTERCEPT if block == 1 else (
            MEMBER_BLOCK_SCALARS if block == 2 else XL_MEMBER_BLOCK_SCORES))
    var words = Int(slot[].record.unsafe_ptr())
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(words, slot[].outputs, slot[].features)
    elif block == MEMBER_BLOCK_INTERCEPT:
        call[].dst = MemberMatrix(words + 4 * slot[].outputs * slot[].features, 1, slot[].outputs)
    elif logcv and block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif logcv and block == XL_MEMBER_BLOCK_SCORES and len(slot[].path) > 0:
        call[].dst = MemberMatrix(Int(slot[].path.unsafe_ptr()), slot[].cv_folds, Int(slot[].params[6]))
    else:
        raise Error("member ABI: a " + _name(slot) + " member state holds no fitted block " + String(block))
    call[].block = block
    call[].out_cols = count
    call[].classes = slot[].classes


def _cls_import(slot: _SlotP, call: _CallP) raises:
    """COEF (T x features) and INTERCEPT (1 x T), then DONE with
    `call.classes`."""
    if call[].block == MEMBER_BLOCK_COEF or call[].block == MEMBER_BLOCK_INTERCEPT:
        _dense(call[].x, "a fitted block", _name(slot))
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        if call[].block == MEMBER_BLOCK_COEF:
            slot[].outputs = call[].x.rows
            slot[].features = call[].x.cols
            slot[].block = words^
        else:
            slot[].extra = List[Float64](length=count, fill=0.0)
            for j in range(count):  # driver-loop: over the fitted classes' intercepts (model structure)
                slot[].extra[j] = Float64(words[j])
    elif call[].block == MEMBER_BLOCK_SCALARS and _kind(slot) == XL_MEMBER_LOGISTIC_CV:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 2:
            raise Error("member ABI: the LogisticRegressionCV scalars block is 2 host float64 words")
        slot[].fitted = False
        slot[].scalars = List[Float64](length=2, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=2)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + _name(slot) + " member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var outputs = slot[].outputs
    var d = slot[].features
    if len(slot[].block) != outputs * d or len(slot[].extra) != outputs or outputs < 1:
        raise Error("member ABI: a fitted " + _name(slot) + " state needs its coefficient (T x features) and "
            + "intercept (1 x T) blocks")
    var k = call[].classes
    if k < 2 or (1 if k == 2 else k) != outputs:
        raise Error("member ABI: a " + _name(slot) + " state of " + String(outputs) + " decision rows needs its "
            + "class count at import")
    var record = List[Float32](length=outputs * d + outputs, fill=0.0)
    unsafe_memcpy(dest=record.unsafe_ptr(), src=slot[].block.unsafe_ptr(), count=outputs * d)
    if _kind(slot) == XL_MEMBER_LOGISTIC_CV and len(slot[].scalars) != 2:
        raise Error("member ABI: a fitted LogisticRegressionCV state needs its scalars block")
    for j in range(outputs):  # driver-loop: over the fitted classes' intercepts (model structure)
        record[outputs * d + j] = Float32(slot[].extra[j])
    slot[].extra = List[Float64]()
    _keep_classes(slot, record^, outputs, d, k)
    slot[].fitted = True


def _xlinear_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # A live state answers for its kind and settings; without one the
        # NEW words do when given (kind-exact, item 17), else the union.
        if call[].state == 0:
            if not call[].params or call[].param_count < 1:
                return _capabilities(-1)
            var words = _F64P(unsafe_from_address=call[].params)
            var named = Int(words[0])
            # RidgeCV takes weights on every cv (lane mf-split5: a weighted
            # integer cv runs the split path over KFold's folds, as the door).
            return _capabilities(named) | _reads_given(named, words, call[].param_count)
        var asked = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        var bits = _capabilities(_kind(asked))
        bits |= _reads_given(_kind(asked), _F64P(unsafe_from_address=Int(asked[].params.unsafe_ptr())),
                             len(asked[].params))
        if not _classifier(_kind(asked)) and not call[].route.has_fifth:
            # A regressor's score is the metrics image's R^2 entry: without it
            # on the route the score would be refused at RUN, so it is not
            # advertised (the caller predicts and runs its own metric).
            bits &= ~MEMBER_CAN_SCORE
        # MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE (lane vdf-scope-linear2):
        # every x_linear fit and decision of an image with XL_HAS_SLOT_X reads
        # THE FOLD X WORD; `_fold_op`.
        if _route_slot_x(call):
            bits |= MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
        return bits
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: an x_linear member takes its kind as parameter word 0")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        var count = xl_member_param_count(kind)
        if kind == XL_MEMBER_RIDGE_CV and call[].param_count >= 5:
            count = 5 + Int(words[4])
        if _enetcv(kind) and call[].param_count >= 10:
            count = 10 + Int(words[9]) + (Int(words[7]) if words[8] != 0.0 else 0)
        if kind == XL_MEMBER_LOGISTIC_CV and call[].param_count >= 7:
            count = 7 + Int(words[6])
        if Float64(kind) != words[0] or count == 0:
            raise Error("member ABI: unknown x_linear member kind")
        # RidgeClassifier, and LogisticRegressionCV with a dict (word 5 = 2):
        # and a class-weight tail (lane mf-adm3)
        var tailed = kind == XL_MEMBER_RIDGE_CLASSIFIER or (kind == XL_MEMBER_LOGISTIC_CV and count > 7
            and words[5] == 2.0)
        if tailed and call[].param_count >= count + MEMBER_CW_HEAD:
            count = call[].param_count
        if call[].param_count != count:
            raise Error("member ABI: an x_linear member of kind " + String(kind) + " takes " + String(count)
                + " parameters")
        if kind <= 2 and words[3] != 0.0 and words[3] != 1.0:
            raise Error("member ABI: an x_linear GLM link word is 0 (identity) or 1 (log)")
        var params = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=params.unsafe_ptr(), src=words, count=count)
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # With a state the state answers; without one the NEW words (the
        # kind) and the caller's class count do (MEMBER-ABI-V2 item 10).
        var asked_kind = -1
        var classes = 0
        if call[].state != 0:
            var held = _SlotP(unsafe_from_address=_lookup(call[].state, False))
            asked_kind = _kind(held)
            classes = held[].classes
        elif call[].params and call[].param_count >= 1:
            asked_kind = Int(_F64P(unsafe_from_address=call[].params)[0])
            classes = call[].classes
        var op = call[].flags
        if asked_kind >= 0:
            # An operation the kind lacks is refused, never sized.
            var can = _capabilities(asked_kind)
            var served = ((op == MEMBER_OP_PREDICT and (can & MEMBER_CAN_PREDICT) != 0)
                or (op == MEMBER_OP_DECISION and (can & MEMBER_CAN_DECISION) != 0)
                or (op == MEMBER_OP_PREDICT_PROBA and (can & MEMBER_CAN_PREDICT_PROBA) != 0)
                or (op == MEMBER_OP_TRANSFORM and (can & MEMBER_CAN_TRANSFORM) != 0)
                or (op == MEMBER_OP_PREDICT_STD and (can & MEMBER_CAN_PREDICT_STD) != 0))
            if not served:
                member_unsupported("an x_linear member of kind " + String(asked_kind), op)
        call[].out_cols = 1
        call[].classes = 0
        # Regressors and IsotonicRegression's transform: float32; a
        # classifier's predict: int32 codes; its decision and proba: float32.
        call[].out_dtype = MEMBER_F32
        if asked_kind >= 0 and _classifier(asked_kind):
            call[].classes = classes
            if op == MEMBER_OP_PREDICT:
                call[].out_dtype = MEMBER_I32
            elif op == MEMBER_OP_DECISION:
                call[].out_cols = max(1, 1 if classes == 2 else classes)
            elif op == MEMBER_OP_PREDICT_PROBA:
                call[].out_cols = max(1, classes)
        elif asked_kind < 0 and op == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_DTYPE_UNSAID  # the kind decides (codes or values)
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if (call[].row_index.rows > 0 and call[].column_index.rows == 0 and call[].route.device == DEVICE_GPU
            and operation != MEMBER_OP_EXPORT and operation != MEMBER_OP_IMPORT):
        _fold_op(slot, call, operation)
        return 0
    _op(slot, call, operation)
    return 0


def _route_slot_x(call: _CallP) -> Bool:
    """THE FOLD X WORD on this route: a GPU image whose x_linear entry reads
    it (XL_HAS_SLOT_X) and ships the x_prep family store. Asked with the
    capabilities only."""
    if call[].route.device != DEVICE_GPU:
        return False
    if not xl_slot_x_held(call[].route.entry.call):
        return False
    return member_fold_store_ready(call)


def _fold_op(slot: _SlotP, call: _CallP, operation: Int) raises:
    """An operation over the rows `call.row_index` of the WHOLE host X
    (core/member_fold_x.mojo): X resident in the caller's scope, the fold's
    rows taken on the device and read by the x_linear fit and decision
    entries (THE FOLD X WORD); y (values or int32 class codes), THE GIVEN y
    and the weights are the base's columns taken bitwise through the index on
    the host. Every fit and query reads the fold slot (lane vdf-scope-linear3:
    RidgeCV's split path takes each split's rows from it on the device).
    Views restored before this returns or raises."""
    var estimator = _name(slot)
    var x = call[].x
    var y = call[].y
    var given = call[].y_given
    var weights = call[].weights
    var index = call[].row_index
    var fold = MemberFoldX()
    var host_y = List[UInt64]()
    var host_g = List[UInt64]()
    var host_w = List[UInt64]()
    var failure = String("")
    try:
        member_fold_x_open(call, fold, estimator)
        call[].x = MemberMatrix(fold.word(), fold.rows, fold.cols)
        if y.rows > 0 and (operation == MEMBER_OP_FIT or operation == MEMBER_OP_SCORE):
            if not y.dense_host(y.dtype) or y.rows != x.rows:
                raise Error("ValueError: X and y have different numbers of rows")
            call[].y = member_fold_take_into(y, fold, host_y)
        if given.given() and given.rows == x.rows and given.dense_host(given.dtype):
            call[].y_given = member_fold_take_into(given, fold, host_g)
        if weights.rows > 0:
            _dense(weights, "sample_weight", estimator)
            if weights.rows != x.rows or weights.cols != 1:
                raise Error("ValueError: sample_weight and X have different numbers of rows")
            call[].weights = member_fold_take_into(weights, fold, host_w)
        call[].row_index = MemberMatrix()
        _op(slot, call, operation)
    except error:
        failure = String(error)
    call[].x = x
    call[].y = y
    call[].y_given = given
    call[].weights = weights
    call[].row_index = index
    try:
        member_fold_x_close(fold)
    except close_error:
        if failure.byte_length() == 0:
            failure = "RuntimeError: a " + estimator + " member's fold slot was not freed: " + String(close_error)
    _ = host_y^
    _ = host_g^
    _ = host_w^
    if failure.byte_length() > 0:
        raise Error(failure)


def _op(slot: _SlotP, call: _CallP, operation: Int) raises:
    if _classifier(_kind(slot)):
        if operation == MEMBER_OP_FIT:
            _cls_fit(slot, call)
        elif operation == MEMBER_OP_EXPORT and call[].block == MEMBER_BLOCK_NOTICES and (
                call[].flags & MEMBER_BLOCK_BY_INDEX) == 0:
            call[].dst = MemberMatrix(Int(slot[].notices.unsafe_ptr()), len(slot[].notices) // MEMBER_NOTICE_WORDS,
                                      MEMBER_NOTICE_WORDS, MEMBER_F64)
        elif operation == MEMBER_OP_PREDICT:
            _cls_predict(slot, call)
        elif operation == MEMBER_OP_DECISION:
            _cls_decision(slot, call)
        elif operation == MEMBER_OP_PREDICT_PROBA:
            _cls_proba(slot, call)
        elif operation == MEMBER_OP_SCORE:
            _cls_score(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _cls_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _cls_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_EXPORT and call[].block == MEMBER_BLOCK_NOTICES and (
            call[].flags & MEMBER_BLOCK_BY_INDEX) == 0:
        call[].dst = MemberMatrix(Int(slot[].notices.unsafe_ptr()), len(slot[].notices) // MEMBER_NOTICE_WORDS,
                                  MEMBER_NOTICE_WORDS, MEMBER_F64)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_TRANSFORM and _kind(slot) == XL_MEMBER_ISOTONIC:
        _predict(slot, call)
    elif operation == MEMBER_OP_PREDICT_STD:
        _predict_std(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)


def xlinear_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the x_linear regressors (schema `MEMBER_ABI`)."""
    try:
        var word = _xlinear_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def xlinear_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's x_linear member entry. One line in the image's
    module initialization, after its operations table is published."""
    member_publish[MEMBER_FAMILY_XLINEAR](mode, vendor, publisher, xlinear_member_v2)
