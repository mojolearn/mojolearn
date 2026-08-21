"""The callable surface over the GBDT fit.

**Why this file exists.** `gbdt/train.mojo` already has `train` and
`predict_floats`, and neither is callable from outside this repository:
they take and return `List[Float32]` and a `TrainedModel`, so every existing
caller is a check or a benchmark that built its own lists in Mojo.

Nothing here is a port. `gbdt/` mirrors CatBoost and is governed by COPY, DO
NOT IMPROVE; this file is host-side policy in the same category as
`mojo_only/`, following `neighbors/estimator.mojo` and `cluster/estimator.mojo`
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
   lists here. At 800k x 100 that is 80 million float reads before any device
   work starts. It is a real cost on the fixed-cost side of
   `ms/tree = a + b*rows` and it is stated rather than absorbed; removing it
   means teaching `train` to take pointers, which is a change to a ported
   file's signature and belongs in its own session.

3. **PREDICTIONS ARE RAW APPROXES FOR EVERY LOSS**, as `train`'s docstring
   says and as their `predict` without a `prediction_type` does. A Logloss
   caller applies the sigmoid, a Poisson caller applies `exp`. This file does
   not guess which the caller wanted, because guessing would make the number
   this repository benchmarks a different number from the one it returns.
"""

from max.gpu.host import DeviceContext

from gbdt.models.model_text import load_model_text, model_text
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    TCatFeatureParams,
)
from gbdt.methods.doc_parallel_boosting import model_approx_dim
from gbdt.models.model_text import load_model_text
from gbdt.train import (
    model_input_features,
    multiclass_probabilities,
    predict_floats,
    predict_multi_floats,
    train,
)


@fieldwise_init
struct GbdtFitParams(Copyable, ImplicitlyCopyable, Movable):
    """Everything `train` takes that is not an array.

    Their spellings where they have one; this struct exists so the CPython
    binding's flat parameter list is unpacked ONCE, here, rather than at
    every call site.
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
    )


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
    params: GbdtFitParams,
) raises -> String:
    """Fit and return the model as `model_text`.

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

    var xs = List[Float32]()
    for i in range(n_rows * n_features):
        xs.append(x.unsafe_load(i))
    var ys = List[Float32]()
    for i in range(n_rows):
        ys.append(y.unsafe_load(i))

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
        for i in range(n_rows):
            ws.append(weights.unsafe_load(i))

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
    )
    return model_text(tm)


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
    as_probabilities: Bool,
) raises -> Int:
    """Apply a MULTI-DIMENSIONAL model. Returns the width written.

    `out_preds` is ROW-MAJOR, `[row * width + k]`.

    `as_probabilities = False` writes the raw approxes, `dim` of them per
    row -- the same raw-score contract every other predict here has, and
    their `predict` without a `prediction_type`.

    `as_probabilities = True` writes `dim + 1` per row: the softmax their
    `prediction_type='Probability'` applies, over ALL `numClasses`
    including the pinned one whose approx is zero. That last column is not
    padding -- it is a real class, and a caller that dropped it would
    renormalise over the wrong set.

    A ONE-DIMENSIONAL MODEL IS ACCEPTED HERE and writes one column, so a
    caller that always routes through this entry point does not need to
    branch. `as_probabilities` on a one-dimensional model raises, because
    a two-class softmax over a single free approx is the SIGMOID and
    belongs to Logloss, whose own `predict_proba` already does it.
    """
    if n_rows <= 0:
        raise Error("gbdt_predict_multi: n_rows must be positive")
    var tm = load_model_text(text)
    var dim = model_approx_dim(tm.model)
    var n_features = model_input_features(tm)

    var xs = List[Float32]()
    for i in range(n_rows * n_features):
        xs.append(x.unsafe_load(i))

    var ap = predict_multi_floats(ctx, tm, xs, n_rows)
    if not as_probabilities:
        for i in range(n_rows * dim):
            out_preds.unsafe_store(i, ap[i])
        return dim

    if dim < 2:
        raise Error(
            "gbdt_predict_multi: as_probabilities needs a"
            " multi-dimensional model; this one has dim " + String(dim)
            + ". A two-class problem's link is the sigmoid, which"
            " Logloss's own predict_proba applies."
        )
    var pr = multiclass_probabilities(ap, n_rows, dim + 1)
    for i in range(n_rows * (dim + 1)):
        out_preds.unsafe_store(i, pr[i])
    return dim + 1


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

    var xs = List[Float32]()
    for i in range(n_rows * n_features):
        xs.append(x.unsafe_load(i))

    var p = predict_floats(ctx, tm, xs, n_rows)
    for i in range(n_rows):
        out_preds.unsafe_store(i, p[i])
    return n_rows
