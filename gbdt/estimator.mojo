# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The callable surface over the GBDT fit.

**Why this file exists.** `gbdt/train.mojo` already has `train` and
`predict_floats`, and neither is callable from outside this repository:
they take and return `List[Float32]` and a `TrainedModel`, so every existing
caller is a check or a benchmark that built its own lists in Mojo.

Nothing here is a port. `gbdt/` mirrors CatBoost and is governed by COPY, DO
NOT IMPROVE; this file is host-side policy in the same category as
`checks/`, following `neighbors/estimator.mojo` and `cluster/estimator.mojo`
-- including their convention that data crosses as raw pointers plus lengths
so a CPython extension can pass buffer addresses straight through.

THE POLICY CHOICES
------------------

1. **THE MODEL CROSSES AS ITS OWN TEXT, NOT AS A HANDLE.** `fit` returns the
   string `gbdt/models/model_text.mojo` writes and `predict` takes that string
   back. The alternative -- an integer handle into a table of live
   `TrainedModel`s on this side -- would put object lifetime across the
   CPython boundary, where nothing in this repository can check it, and would
   make a leak invisible. The text format is already gated: `check-model-io`
   trains, saves, loads, and requires BIT-IDENTICAL predictions from both the
   host apply and the device evaluator, then proves five sabotages turn it
   red. Reusing it means the boundary inherits that gate instead of needing a
   new one.

   IT COSTS A SERIALISE AND A PARSE PER `predict` CALL. That is real and it is
   not hidden: a caller predicting in a loop pays it every time. The fix, when
   someone measures it and wants it gone, is a cached parse keyed on the text,
   not a handle table slipped in quietly.

   Floats in that text are written as `<decimal>/<hex bits>` and read back
   from the HEX, because `String(Float32)` on this toolchain returns a
   one-ULP-wrong value for 0.46% of float32 values. A model that round-tripped
   through decimal alone would predict differently after a save.

2. **THE ARRAYS ARE COPIED INTO `List`s.** `train` takes `List[Float32]`, so
   `x`, `y` and the weights are read out of the caller's buffers into owned
   lists here -- as one flat memcpy per array since 2026-08-26; the
   per-element append loops that stood here were ~70 ms per 1M x 28 rows,
   all inside the timed fit. At 800k x 100 it is still 80 million float
   reads before any device work starts, a real cost on the fixed-cost side
   of `ms/tree = a + b*rows`, and it is stated rather than absorbed;
   removing the copy itself means teaching `train` to take pointers, which
   is a change to a ported file's signature and belongs in its own session.

3. **PREDICTIONS ARE RAW APPROXES FOR EVERY LOSS**, as `train`'s docstring
   says and as their `predict` without a `prediction_type` does. A Logloss
   caller applies the sigmoid, a Poisson caller applies `exp`. This file does
   not guess which the caller wanted, because guessing would make the number
   this repository benchmarks a different number from the one it returns.
"""

from max.gpu.host import DeviceContext
from std.memory import memcpy

from gbdt.models.model_text import load_model_text, model_text
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    TCatFeatureParams,
)
from gbdt.methods.doc_parallel_boosting import model_approx_dim
from gbdt.options.overfitting_detector_options import (
    load_overfitting_detector_options,
)
from gbdt.overfitting_detector.overfitting_detector import od_type_name
from gbdt.models.model_text import load_model_text
from gbdt.train import (
    model_input_features,
    multiclass_probabilities,
    one_vs_all_probabilities,
    predict_floats,
    predict_multi_floats,
    train,
)

#: `gbdt_predict_multi`'s transform, mirroring their `EPredictionType`
#: (`libs/model/eval_processing.h:186-226`). `RAW` is their `RawFormulaVal`,
#: `SOFTMAX` their `Probability` for a multi-output model, `SIGMOID` their
#: `MultiProbability`.
comptime PREDICT_RAW = 0
comptime PREDICT_SOFTMAX = 1
comptime PREDICT_SIGMOID = 2


@fieldwise_init
struct GbdtFitParams(Copyable, Movable):
    """Everything `train` takes that is not an array.

    Their spellings where they have one; this struct exists so the CPython
    binding's flat parameter list is unpacked ONCE, here, rather than at
    every call site.

    NOT `ImplicitlyCopyable`, and `class_weights` is why: a `List[Float32]`
    field cannot synthesize an implicit copy constructor, so the struct
    cannot either. Copies are explicit at the call sites instead, which is
    the right way round for a field that owns a heap allocation.
    """

    var border_count: Int
    var n_estimators: Int
    var max_depth: Int
    var learning_rate: Float32
    var l2_leaf_reg: Float32
    var random_seed: UInt64
    var score_function: Int
    var loss: String
    var loss_alpha: Float32
    var loss_q: Float32
    var loss_delta: Float32
    var loss_variance_power: Float32
    var loss_border: Float32
    var leaf_estimation_iterations: Int
    var leaf_estimation_method: Int
    var bootstrap_type: String
    var bagging_temperature: Float32
    var subsample: Float32
    #: their `TOverfittingDetectorOptions`, UNRESOLVED. An empty
    #: `od_type` and a negative `od_pvalue` / `od_wait` are their
    #: `options.Has(...)` returning false, and
    #: `load_overfitting_detector_options` is what turns the three into a
    #: detector type -- `od_wait` alone means Iter, `od_pvalue` alone
    #: means IncToDec, neither means None.
    var od_type: String
    var od_pvalue: Float64
    var od_wait: Int
    #: their `use_best_model`, TRI-STATE: -1 unset (on when there is an
    #: eval set with a non-constant target), 0 off, 1 on.
    var use_best_model: Int
    #: their `best_model_min_trees`, default 1 (`output_file_options.cpp:77`)
    var best_model_min_trees: Int
    #: their `ENanMode` spelling -- "Min", "Max" or "Forbidden".
    #: `data_processing_options.cpp:26` defaults it to Min, and so does
    #: `train`. Forbidden RAISES on a NaN rather than binning it, which is
    #: their behavior and not a validation nicety.
    var nan_mode: String
    #: their `random_strength` (`oblivious_tree_options.cpp:17`), CatBoost
    #: default 1.0 and OURS 0.0 -- `train`'s own default, kept here so the
    #: binding does not quietly disagree with the function it calls. On the
    #: GREEDY searcher the noise cancels in the gain
    #: (`compute_scores.cu:84-134`) so a non-zero value moves only float
    #: rounding; it is a real knob on the pointwise searcher.
    var random_strength: Float32
    #: `TDocParallelObliviousTreeSearcher` instead of the greedy subsets
    #: searcher. Both are CatBoost's.
    var use_pointwise_searcher: Bool
    #: their `border_build_max_samples`, the subsample the BORDER SEARCH
    #: runs on, default 200000 (`data_processing_options.cpp:37`). 0 means
    #: "every row", which is `train`'s documented restore.
    var border_build_max_samples: Int
    #: their `permutation_count` (`boosting_options.cpp:16`), -1 UNSET so
    #: `UpdateGpuSpecificDefaults` resolves it. Only the CTR path reads it.
    var permutation_count: Int
    #: their estimation permutation (`doc_parallel_boosting.h:101-103`),
    #: -1 UNSET meaning `permutation_count - 1`. Only the CTR path reads it.
    var ctr_estimation_permutation_id: Int
    #: their `boost_from_average`, tri-state exactly as `train` takes it:
    #: -1 is their unset option, resolved by the port of
    #: `AdjustBoostFromAverageDefaultValue` inside `train`; 0/1 explicit.
    var boost_from_average: Int
    #: their `class_weights` (`data_processing_options.cpp:52`), EMPTY for
    #: none. Multiplied into the row weight, not substituted for it
    #: (`target/data_providers.cpp:168`). One entry per class slot; `train`
    #: raises if the length disagrees with the label set.
    var class_weights: List[Float32]
    #: their `grow_policy` spelling -- "SymmetricTree", "Depthwise" or
    #: "Lossguide" (`oblivious_tree_options.cpp:23`); empty is the default
    #: SymmetricTree. DEVIATION 259.
    var grow_policy: String
    #: their `max_leaves`, -1 UNSET (`IsDefault()`): `1 << depth` for every
    #: policy but Lossguide, 31 under Lossguide (`catboost_options.cpp:
    #: 993-1001`, `oblivious_tree_options.cpp:24`). Read under Lossguide
    #: only; any other policy refuses a value that is not `1 << depth`.
    var max_leaves: Int
    #: their `min_data_in_leaf`, default 1 (`oblivious_tree_options.cpp:
    #: 25`); live under Depthwise and Lossguide, refused at any other value
    #: under SymmetricTree, where CatBoost discards it.
    var min_data_in_leaf: Int


def default_gbdt_fit_params() -> GbdtFitParams:
    """CatBoost's own defaults, and the ones `train` already carries.

    `learning_rate` is 0.03 (`boosting_options.cpp:10`), `l2_leaf_reg` 3.0,
    `depth` 6, `border_count` 128, `iterations` 100. The three `-1`s mean
    UNSET, which is their `TOption::NotSet()`: the loss decides.
    """
    return GbdtFitParams(
        128, 100, 6,
        Float32(0.03), Float32(3.0), UInt64(0), SCORE_FUNCTION_COSINE,
        String("RMSE"),
        Float32(-1.0), Float32(-1.0), Float32(-1.0), Float32(-1.0),
        Float32(-1.0),
        -1, -1,
        String(""), Float32(1.0), Float32(-1.0),
        String(""), Float64(-1.0), -1,
        -1, 1,
        # nan_mode, random_strength, use_pointwise_searcher,
        # border_build_max_samples, permutation_count,
        # ctr_estimation_permutation_id, class_weights.
        #
        # EVERY ONE OF THESE IS `train`'s OWN DEFAULT, not CatBoost's where
        # the two differ, because this struct's job is to reproduce a bare
        # `train(...)` call and a second opinion about a default is a second
        # answer. `random_strength` is the one that differs: CatBoost ships
        # 1.0 (`oblivious_tree_options.cpp:17`) and `train` ships 0.0.
        String("Min"), Float32(0.0), False,
        200_000, -1, -1,
        # boost_from_average -1: their unset option, resolved inside
        # `train` by the AdjustBoostFromAverageDefaultValue port
        -1,
        List[Float32](),
        # grow_policy SymmetricTree, max_leaves unset, min_data_in_leaf 1
        String("SymmetricTree"), -1, 1,
    )


@fieldwise_init
struct GbdtFitResult(Movable):
    """What a fit produces besides the model.

    THE LOSS CURVES ARE NOT DECORATION. `test_losses` is the series the
    overfitting detector stopped on, and without it a caller cannot tell a
    model that stopped early from one that ran out of iterations, nor see
    the shape that made it stop. Their `TMetricsAndTimeLeftHistory` carries
    the same two curves out of `TrainModel` for the same reason.

    `best_iteration` is the ERROR tracker's best (`error_tracker.h:73-75`),
    not the min-trees tracker's, so with `best_model_min_trees` above it
    the returned ensemble holds MORE trees than `best_iteration + 1`.
    """

    var text: String
    var best_iteration: Int
    var stopped_early: Bool
    var learn_losses: List[Float64]
    var test_losses: List[Float64]


def gbdt_fit(
    ctx: DeviceContext,
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    y: MutPointer[Float32, MutUntrackedOrigin],
    weights: MutPointer[Float32, MutUntrackedOrigin],
    n_weights: Int,
    cat_flags: MutPointer[UInt32, MutUntrackedOrigin],
    n_flags: Int,
    eval_x: MutPointer[Float32, MutUntrackedOrigin],
    eval_y: MutPointer[Float32, MutUntrackedOrigin],
    n_eval_rows: Int,
    params: GbdtFitParams,
) raises -> GbdtFitResult:
    """Fit and return the model as `model_text`, plus both loss curves.

    `x` is COLUMN-MAJOR, `[feature * n_rows + row]`, which is the layout
    `train` documents and the layout the quantizer walks.

    `cat_flags` holds one word per feature, or is unread when `n_flags` is
    zero:

        bit 0   this column is CATEGORICAL (dense codes 0..k-1)
        bit 1   this column is ONE-HOT (equality splits, no border search)

    Those are the two `List[Bool]`s `train` takes. They arrive packed
    because the binding's parameter budget is arity-limited and a second
    address is cheaper than a second list.

    `n_weights` of 0 means unit weights and `weights` is never read, the
    same contract `kmeans_fit` has for its weight pointer.

    `eval_x` is COLUMN-MAJOR over `n_eval_rows` rows and the SAME
    `n_features` columns, and `eval_y` its target. `n_eval_rows == 0`
    means no held-out set and neither pointer is read -- the same contract
    the weights have. The held-out rows are quantized inside `train`,
    against the borders that fit built, which is the only place they can
    be.
    """
    if n_rows <= 0:
        raise Error("gbdt_fit: n_rows must be positive")
    if n_features <= 0:
        raise Error("gbdt_fit: n_features must be positive")
    if n_weights != 0 and n_weights != n_rows:
        raise Error(
            "gbdt_fit: n_weights must be 0 or n_rows, got "
            + String(n_weights)
        )
    if n_flags != 0 and n_flags != n_features:
        raise Error(
            "gbdt_fit: n_flags must be 0 or n_features, got "
            + String(n_flags)
        )
    if n_eval_rows < 0:
        raise Error(
            "gbdt_fit: n_eval_rows must not be negative, got "
            + String(n_eval_rows)
        )

    # one flat resize + memcpy per array; the unreserved append loops this
    # replaces were ~70 ms per 1M x 28 rows of bounds-checked
    # doubling-and-copying appends (~2.5 ns each), all inside the timed
    # fit -- the same swap `train`'s column build made
    # (`gbdt/train.mojo:885`). Same bytes in the same order: the caller's
    # buffer is already the column-major layout `train` takes.
    var n_x = n_rows * n_features
    var xs = List[Float32]()
    xs.resize(n_x, Float32(0.0))
    memcpy(dest=xs.unsafe_ptr(), src=x, count=n_x)
    var ys = List[Float32]()
    ys.resize(n_rows, Float32(0.0))
    memcpy(dest=ys.unsafe_ptr(), src=y, count=n_rows)

    var cats = List[Bool]()
    var one_hot = List[Bool]()
    if n_flags != 0:
        for f in range(n_features):
            var w = cat_flags.unsafe_load(f)
            cats.append((w & UInt32(1)) != UInt32(0))
            one_hot.append((w & UInt32(2)) != UInt32(0))

    # THEIR PRODUCT (`target/data_providers.cpp:168`): the row weight and
    # the class weight multiply. `n_weights == 0` means unit weights and
    # `weights` is never read, the same contract `kmeans_fit` has.
    var ws = List[Float32]()
    if n_weights != 0:
        ws.resize(n_rows, Float32(0.0))
        memcpy(dest=ws.unsafe_ptr(), src=weights, count=n_rows)

    var eval_xs = List[Float32]()
    var eval_ys = List[Float32]()
    if n_eval_rows != 0:
        var n_ex = n_eval_rows * n_features
        eval_xs.resize(n_ex, Float32(0.0))
        memcpy(dest=eval_xs.unsafe_ptr(), src=eval_x, count=n_ex)
        eval_ys.resize(n_eval_rows, Float32(0.0))
        memcpy(dest=eval_ys.unsafe_ptr(), src=eval_y, count=n_eval_rows)

    # `TOverfittingDetectorOptions::Load` (`:24-40`), which is where the
    # detector TYPE comes from when the caller named only a wait or only a
    # p-value.
    var od = load_overfitting_detector_options(
        params.od_type, params.od_pvalue, params.od_wait
    )

    var tm = train(
        ctx, xs, ys, n_rows, n_features,
        border_count=params.border_count,
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.l2_leaf_reg,
        one_hot=one_hot,
        random_seed=params.random_seed,
        score_function=params.score_function,
        cat_features=cats,
        sample_weight=ws,
        loss=params.loss,
        loss_alpha=params.loss_alpha,
        loss_q=params.loss_q,
        loss_delta=params.loss_delta,
        loss_variance_power=params.loss_variance_power,
        loss_border=params.loss_border,
        leaf_estimation_iterations=params.leaf_estimation_iterations,
        leaf_estimation_method=params.leaf_estimation_method,
        bootstrap_type=params.bootstrap_type,
        bagging_temperature=params.bagging_temperature,
        subsample=params.subsample,
        eval_x_colmajor=eval_xs,
        eval_y=eval_ys,
        od_type=od_type_name(od.od_type),
        od_pvalue=od.auto_stop_p_value,
        od_wait=od.iterations_wait,
        use_best_model=params.use_best_model,
        best_model_min_trees=params.best_model_min_trees,
        nan_mode=params.nan_mode,
        random_strength=params.random_strength,
        use_pointwise_searcher=params.use_pointwise_searcher,
        border_build_max_samples=params.border_build_max_samples,
        permutation_count=params.permutation_count,
        ctr_estimation_permutation_id=params.ctr_estimation_permutation_id,
        boost_from_average=params.boost_from_average,
        class_weights=params.class_weights.copy(),
        grow_policy=params.grow_policy,
        max_leaves=params.max_leaves,
        min_data_in_leaf=params.min_data_in_leaf,
    )
    var learn_losses = tm.losses.copy()
    var test_losses = tm.test_losses.copy()
    return GbdtFitResult(
        model_text(tm),
        tm.best_iteration,
        tm.stopped_early,
        learn_losses^,
        test_losses^,
    )


def gbdt_model_dim(text: String) raises -> Int:
    """The model's approx dimension, so a caller can size its output.

    `1` for every single-dimensional loss; `numClasses - 1` for MultiClass,
    because the last class's approx is pinned at zero and is not stored.
    The Python wrapper reads this to decide the shape of `predict` and to
    recover `n_classes` as `dim + 1`.

    It costs a parse of the model text. That is the price of the
    handle-free boundary this file's header argues for, and it is paid once
    per `fit` rather than once per row.
    """
    return model_approx_dim(load_model_text(text).model)


def gbdt_predict_multi(
    ctx: DeviceContext,
    text: String,
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    out_preds: MutPointer[Float32, MutUntrackedOrigin],
    mode: Int,
) raises -> Int:
    """Apply a MULTI-DIMENSIONAL model. Returns the width written.

    `out_preds` is ROW-MAJOR, `[row * width + k]`.

    `mode` picks the transform, mirroring their `EPredictionType`:

      PREDICT_RAW      `dim` columns, the raw approxes -- their
                       `RawFormulaVal`, the contract every other predict
                       here has.
      PREDICT_SOFTMAX  `dim + 1` columns -- their `Probability` for a
                       multi-output model (`eval_processing.h:214-221`),
                       over ALL `numClasses` INCLUDING the pinned one
                       whose approx is zero. **MultiClass only.** That
                       last column is not padding; it is a real class, and
                       dropping it would renormalise over the wrong set.
      PREDICT_SIGMOID  `dim` columns -- their `MultiProbability`
                       (`:222-226`), an ELEMENTWISE sigmoid. **OneVsAll
                       only**, whose classes are independent and whose
                       columns therefore do NOT sum to one.

    THE MODEL TEXT DOES NOT RECORD THE LOSS, so this cannot choose between
    the last two itself -- the caller knows which loss it fitted and says
    so. That is deliberate rather than an omission: a model file that
    carried the loss would let a caller's `predict_proba` disagree with
    the loss they trained under, and the wrapper already has to remember
    it for other reasons.

    A ONE-DIMENSIONAL MODEL IS ACCEPTED at `PREDICT_RAW` and writes one
    column, so a caller that always routes through this entry point does
    not need to branch. The two probability modes raise on one dimension,
    because a two-class link is the plain sigmoid and belongs to Logloss,
    whose own `predict_proba` already does it.
    """
    if n_rows <= 0:
        raise Error("gbdt_predict_multi: n_rows must be positive")
    var tm = load_model_text(text)
    var dim = model_approx_dim(tm.model)
    var n_features = model_input_features(tm)

    # resize + memcpy, the same swap gbdt_fit made; the append loop this
    # replaces re-read the whole matrix element-wise on every predict
    var n_x = n_rows * n_features
    var xs = List[Float32]()
    xs.resize(n_x, Float32(0.0))
    memcpy(dest=xs.unsafe_ptr(), src=x, count=n_x)

    var ap = predict_multi_floats(ctx, tm, xs, n_rows)
    if mode == PREDICT_RAW:
        for i in range(n_rows * dim):
            out_preds.unsafe_store(i, ap[i])
        return dim

    if dim < 2:
        raise Error(
            "gbdt_predict_multi: a probability mode needs a"
            " multi-dimensional model; this one has dim " + String(dim)
            + ". A two-class problem's link is the sigmoid, which"
            " Logloss's own predict_proba applies."
        )
    if mode == PREDICT_SOFTMAX:
        var pr = multiclass_probabilities(ap, n_rows, dim + 1)
        for i in range(n_rows * (dim + 1)):
            out_preds.unsafe_store(i, pr[i])
        return dim + 1
    if mode == PREDICT_SIGMOID:
        var ps = one_vs_all_probabilities(ap, n_rows, dim)
        for i in range(n_rows * dim):
            out_preds.unsafe_store(i, ps[i])
        return dim
    raise Error(
        "gbdt_predict_multi: unknown mode " + String(mode)
    )


def gbdt_predict(
    ctx: DeviceContext,
    text: String,
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    out_preds: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    """Apply a `model_text` model to RAW column-major floats.

    Returns the number of rows written, which is `n_rows`; the return is
    there so a caller can assert the boundary agreed about the shape rather
    than trusting it.

    RAW APPROXES, per policy choice 3 above.
    """
    if n_rows <= 0:
        raise Error("gbdt_predict: n_rows must be positive")
    var tm = load_model_text(text)
    var n_features = model_input_features(tm)

    # resize + memcpy, the same swap gbdt_fit made
    var n_x = n_rows * n_features
    var xs = List[Float32]()
    xs.resize(n_x, Float32(0.0))
    memcpy(dest=xs.unsafe_ptr(), src=x, count=n_x)

    var p = predict_floats(ctx, tm, xs, n_rows)
    for i in range(n_rows):
        out_preds.unsafe_store(i, p[i])
    return n_rows
