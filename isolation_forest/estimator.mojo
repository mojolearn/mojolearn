# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python-facing surface: `IsolationForest(n_estimators, max_samples,
max_depth, max_features, bootstrap, random_state, contamination)` with
`fit`, `score_samples`, `decision_function`, `predict`, `fit_predict`.

Reference: `python/cuml/cuml/ensemble/isolation_forest.pyx` at rapidsai/cuml
v26.08.00: the constructor defaults (`:474-480`), `fit`'s resolution of
`max_features` (`:616-641`), `contamination` (`:643-660`), `max_samples`
(`:663-702`), `max_depth` (`:704-709`), the seed (`:712`,
`check_random_seed`: an int in [0, 2^32-1] is passed through), the
`IF_params` fill (`:715-721`), the contamination quantile `offset_ =
cp.percentile(score_samples(X), 100 * contamination)` else `-0.5`
(`:778-785`), `score_samples = -paper_score` (`:959`),
`decision_function = score_samples - offset_` (`:978`), `predict =
-C++predict(threshold = -offset_)` (`:1023-1042`). `warm_start` and
`sample_weight` raise `UnsupportedOnGPU` there (`:592-595`) and raise by
name here. Treelite / nvForest export (`as_treelite`, `as_nvforest`,
`_score_samples_nvforest`) is NOT implemented (NOT_IMPLEMENTED.tsv).

Host only; the device work is `impl/isolation_forest/`. The percentile
is numpy/cupy's default `linear` interpolation computed in Float64 over
the sorted float32 scores (`percentile_linear`); the threshold handed to
the device is `Float32(-offset_)` as their `<float>threshold` cast.
`random_state=None` is NOT a fresh random seed here: it is 0, stated,
because a fit whose seed nobody recorded cannot be reproduced and this
repository's product is the reproducible fit.

`iforest_run_host` at the foot is the ONE-SHOT form the CPython binding
calls (`bindings/_mojolearn_svm.mojo`, `python/mojolearn/_iforest_impl.py`).
The estimator struct above is the one the gates use, and it keeps its model
between calls; the binding may not, and DEVIATION 874 at that entry says
what that costs.
"""

from std.ffi import _Global
from std.math import fma
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoSvmContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoSvmContextOther"


from core.identity_trace import IdentityTrace
from isolation_forest.impl.isolation_forest import (
    IDN_IF_EPILOGUE_DEVICE,
    IDN_IF_QUERY_DEVICE,
    IDN_IF_RESIDENT,
    IF_params,
    IFLaunchKnobs,
    IsolationForestModel,
    fit as if_fit,
    predict as if_predict,
    predict_into as if_predict_into,
    score_samples as if_score_samples,
    score_samples_into as if_score_samples_into,
)


def percentile_linear(values: List[Float32], q: Float64) -> Float64:
    """`np.percentile(values, q)` with the default `linear` method:
    sorted, `index = q/100 * (n-1)`, `lo + (hi - lo) * frac`."""
    var s = values.copy()
    # heap-free sort: insertion is O(n^2) but n is the training size at a
    # single host call; a simple shell sort keeps it tolerable.
    var n = len(s)
    var gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            var v = s[i]
            var j = i
            while j >= gap and s[j - gap] > v:
                s[j] = s[j - gap]
                j -= gap
            s[j] = v
        gap //= 2
    if n == 0:
        return 0.0
    var index = q / 100.0 * Float64(n - 1)
    var lo = Int(index)
    if lo < 0:
        lo = 0
    if lo > n - 1:
        lo = n - 1
    var hi = lo + 1 if lo + 1 < n else lo
    var frac = index - Float64(lo)
    var a = Float64(s[lo])
    var b = Float64(s[hi])
    return fma(b - a, frac, a)  # the default build's fused op (lane/pinned-mul-contract-free)


struct IsolationForestEstimator(Movable):
    """`cuml.ensemble.IsolationForest`. `max_samples_mode`: 0 = "auto"
    (`min(256, n_samples)`), 1 = int, 2 = float fraction;
    `max_features_mode`: 0 = float fraction (default 1.0), 1 = int;
    `contamination_auto` or a fraction in (0, 0.5]."""

    var n_estimators: Int
    var max_samples_mode: Int
    var max_samples_int: Int
    var max_samples_frac: Float64
    var max_depth: Int
    """-1 = None (auto)."""
    var max_features_mode: Int
    var max_features_int: Int
    var max_features_frac: Float64
    var bootstrap: Bool
    var random_state: Int
    var contamination_auto: Bool
    var contamination: Float64
    var warm_start: Bool
    var offset_: Float64
    var max_samples_: Int
    var n_features_in_: Int
    var model: IsolationForestModel
    var fitted: Bool
    var knobs: IFLaunchKnobs

    def __init__(out self, ctx: DeviceContext) raises:
        """DEVIATION 1944: the empty model is built on the CALLER'S context.
        Until 2026-08-29 this read `IsolationForestModel(DeviceContext())`,
        a second DeviceContext created while `iforest_run_host`'s own was
        alive, and `fit` then replaced that model's eight buffers with ones
        on the caller's context, so the second context's buffers were freed
        while the first was mid-fit. On an RTX 4090 (driver 580, CUDA 13)
        that never returned: GPU idle, every host thread in futex wait, in
        every numeric tier, on four hosts, while the same fit through ONE
        context (`checks/if_hang_probe.mojo`) returned the M4's bits.
        H100, M4 and MI325X never minded. One context per call is the rule
        `bindings/_mojolearn_estimators.mojo` already states; this is the
        estimator obeying it."""
        self.n_estimators = 100
        self.max_samples_mode = 0
        self.max_samples_int = 256
        self.max_samples_frac = 1.0
        self.max_depth = -1
        self.max_features_mode = 0
        self.max_features_int = 0
        self.max_features_frac = 1.0
        self.bootstrap = False
        self.random_state = 0
        self.contamination_auto = True
        self.contamination = 0.0
        self.warm_start = False
        self.offset_ = -0.5
        self.max_samples_ = 0
        self.n_features_in_ = 0
        self.model = IsolationForestModel(ctx)
        self.fitted = False
        self.knobs = IFLaunchKnobs.default()

    def fit(
        mut self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int,
        src_addr: Int = 0,
    ) raises:
        """`IsolationForest.fit(X)` (`:572-787`). DEVIATION 2638: a nonzero
        `src_addr` lends the ROW-major X by address (the CPython binding's
        path, `x_rowmajor` then empty) and the column-major copy below is
        written straight into the pinned upload stage instead. `sample_weight` has no
        argument here; it would raise by name as `warm_start` does."""
        if self.warm_start:
            raise Error("`warm_start=True` is not supported")
        # max_features (:616-641)
        var actual_max_features: Int
        if self.max_features_mode == 1:
            if self.max_features_int < 1 or self.max_features_int > n_cols:
                raise Error(
                    "max_features must be an int in [1, n_features] or a float in (0.0, 1.0]."
                )
            actual_max_features = self.max_features_int
        else:
            if self.max_features_frac <= 0.0 or self.max_features_frac > 1.0:
                raise Error(
                    "max_features must be an int in [1, n_features] or a float in (0.0, 1.0]."
                )
            actual_max_features = Int(self.max_features_frac * Float64(n_cols))
            if actual_max_features < 1:
                actual_max_features = 1
        # contamination (:643-660)
        var use_quantile = False
        if not self.contamination_auto:
            if self.contamination <= 0.0 or self.contamination > 0.5:
                raise Error(
                    "contamination must be 'auto' or a float in the range (0.0, 0.5]."
                )
            use_quantile = True
        # max_samples (:663-702)
        var actual_max_samples: Int
        if self.max_samples_mode == 0:
            actual_max_samples = 256 if n_rows > 256 else n_rows
        elif self.max_samples_mode == 1:
            if self.max_samples_int <= 0:
                raise Error("max_samples must be a positive integer.")
            actual_max_samples = self.max_samples_int if self.max_samples_int < n_rows else n_rows
        else:
            if self.max_samples_frac <= 0.0 or self.max_samples_frac > 1.0:
                raise Error("float max_samples must be in (0.0, 1.0].")
            actual_max_samples = Int(self.max_samples_frac * Float64(n_rows))
            if actual_max_samples < 1:
                raise Error(
                    "max_samples resolves to 0 samples; increase max_samples or the number of rows."
                )
        self.max_samples_ = actual_max_samples
        # seed (:712): check_random_seed passes an int in [0, 2^32-1] through
        if self.random_state < 0 or self.random_state >= 4294967296:
            raise Error(
                "Expected `0 <= random_state <= 2**32 - 1`, got " + String(self.random_state)
            )
        var params = IF_params.default()
        params.n_estimators = self.n_estimators
        params.max_samples = actual_max_samples
        params.max_depth = self.max_depth if self.max_depth > 0 else -1
        params.max_features = actual_max_features
        params.bootstrap = self.bootstrap
        params.seed = UInt64(self.random_state)
        self.n_features_in_ = n_cols

        var trace = IdentityTrace()
        if src_addr != 0:
            # DEVIATION 2638: order="F" (:599-605) happens inside the upload.
            if_fit(ctx, self.model, List[Float32](), n_rows, n_cols, params, trace,
                   self.knobs, src_addr)
        else:
            # order="F" for fit (:599-605): a copy, no arithmetic
            var x_col = List[Float32]()
            for k in range(n_cols):
                for i in range(n_rows):
                    x_col.append(x_rowmajor[i * n_cols + k])
            if_fit(ctx, self.model, x_col, n_rows, n_cols, params, trace, self.knobs)
        self.fitted = True

        if use_quantile:
            var training_scores: List[Float32]
            if src_addr != 0:
                comptime if IDN_IF_QUERY_DEVICE:
                    # lane fam-forests: the borrowed block is scored from its
                    # address (no List copy of n x d cells).
                    training_scores = self.score_samples(
                        ctx, List[Float32](), n_rows, n_cols, src_addr
                    )
                else:
                    var sp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=src_addr)
                    var x_rows = List[Float32](capacity=n_rows * n_cols)
                    for i in range(n_rows * n_cols):
                        x_rows.append(sp.unsafe_load(i))
                    training_scores = self.score_samples(ctx, x_rows, n_rows, n_cols)
            else:
                training_scores = self.score_samples(ctx, x_rowmajor, n_rows, n_cols)
            self.offset_ = percentile_linear(training_scores, 100.0 * self.contamination)
        else:
            self.offset_ = -0.5

    def score_samples(
        self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int,
        src_addr: Int = 0,
    ) raises -> List[Float32]:
        """`score_samples(X)` (`:894-959`): `-paper_score`. A nonzero
        `src_addr` lends the ROW-major X instead of `x_rowmajor`
        (`IDN_IF_QUERY_DEVICE`)."""
        if not self.fitted:
            raise Error("Model has not been fitted. Call fit() first.")
        var trace = IdentityTrace.disabled()
        var paper = if_score_samples(
            ctx, self.model, x_rowmajor, n_rows, n_cols, trace, self.knobs, src_addr
        )
        var out = List[Float32]()
        for i in range(n_rows):
            out.append(-paper[i])
        return out^

    def decision_function(
        self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int,
        src_addr: Int = 0,
    ) raises -> List[Float32]:
        """`decision_function(X) = score_samples(X) - offset_` (`:978`).
        The subtraction is the Python layer's (float32 array minus a
        Python float: numpy/cupy compute it in float32)."""
        var s = self.score_samples(ctx, x_rowmajor, n_rows, n_cols, src_addr)
        var off = Float32(self.offset_)
        var out = List[Float32]()
        for i in range(n_rows):
            out.append(s[i] - off)
        return out^

    def predict(
        self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int,
        src_addr: Int = 0,
    ) raises -> List[Int32]:
        """`predict(X)` (`:981-1042`): `-C++predict(X, threshold =
        -offset_)`; sklearn convention, -1 = anomaly, 1 = inlier."""
        if not self.fitted:
            raise Error("Model has not been fitted. Call fit() first.")
        var threshold = Float32(-self.offset_)
        var raw = if_predict(
            ctx, self.model, x_rowmajor, n_rows, n_cols, threshold, self.knobs, src_addr
        )
        var out = List[Int32]()
        for i in range(n_rows):
            out.append(-raw[i])
        return out^

    def fit_predict(
        mut self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int
    ) raises -> List[Int32]:
        self.fit(ctx, x_rowmajor, n_rows, n_cols)
        return self.predict(ctx, x_rowmajor, n_rows, n_cols)


# ===========================================================================
# THE ONE-SHOT HOST ENTRY: what `bindings/_mojolearn_svm.mojo` calls.
#
# `IsolationForestEstimator` above holds an `IsolationForestModel`, and that
# model is eight `DeviceBuffer`s. `bindings/_mojolearn_estimators.mojo`'s
# header states the rule this lane inherits -- "all device buffers and
# contexts live for one call and no pointer is retained" -- so the estimator
# CANNOT live between two Python calls.
#
# DEVIATION 874: therefore `iforest_run_host` fits and scores in ONE call,
# and `python/mojolearn/_iforest_impl.py` keeps the training matrix and
# calls it again for every `score_samples`, `decision_function` and
# `predict`. Each of those REFITS the forest.
#
# That is honest rather than free. It is CORRECT because the forest is a
# pure function of `(random_state, tree_id, X bits)` -- `isolation_forest/
# README.md` "Where the identity actually lives", and `random_state=None`
# is 0 here rather than a fresh draw, stated in this file's own docstring
# -- so the refit is the same forest bit for bit and `check_if_launch_
# invariance` is the gate on that. It COSTS a full fit per scoring call.
#
# The alternative, handing the four node arrays and the tree offsets back
# to Python and uploading them again at predict, was not taken: it needs a
# model-reconstruction path that no gate in this lane covers, and a mistake
# in it returns wrong scores silently rather than raising.
# ===========================================================================


comptime IF_WANT_SCORE_SAMPLES = 0
comptime IF_WANT_DECISION_FUNCTION = 1
comptime IF_WANT_PREDICT = 2


struct IFRunOutputs(Copyable, Movable):
    """One fit-and-score. `values` is filled for `score_samples` and
    `decision_function`, `labels` for `predict`; the other is empty. The
    three fitted attributes come back on every call because the Python
    layer publishes them as `offset_`, `max_samples_` and
    `n_features_in_`."""

    var values: List[Float32]
    var labels: List[Int32]
    var offset_: Float64
    var max_samples_: Int
    var n_features_in_: Int
    var token: Int
    """`IDN_IF_RESIDENT`: the resident forest's token (0 on the one-shot
    entry, which keeps nothing)."""

    def __init__(out self):
        self.values = List[Float32]()
        self.labels = List[Int32]()
        self.offset_ = -0.5
        self.max_samples_ = 0
        self.n_features_in_ = 0
        self.token = 0


def iforest_run_host(
    train: List[Float32],
    n_train: Int,
    n_features: Int,
    query: List[Float32],
    n_query: Int,
    n_estimators: Int,
    max_samples_mode: Int,
    max_samples_int: Int,
    max_samples_frac: Float64,
    max_depth: Int,
    max_features_mode: Int,
    max_features_int: Int,
    max_features_frac: Float64,
    bootstrap: Bool,
    random_state: Int,
    contamination_auto: Bool,
    contamination: Float64,
    want: Int,
    train_addr: Int = 0,
    query_addr: Int = 0,
) raises -> IFRunOutputs:
    """`IsolationForest(...).fit(train)` then one of `score_samples`,
    `decision_function` or `predict` on `query`, in one call.

    Both matrices are ROW-MAJOR `n x n_features`; `fit` transposes to
    column-major itself, as cuML's `fit` does (`order="F"`, `:599-605`).

    `max_samples_mode`: 0 = "auto" (`min(256, n_samples)`), 1 = an int,
    2 = a float fraction. `max_features_mode`: 0 = a float fraction
    (their default 1.0), 1 = an int. `max_depth` -1 is their `None`.
    `contamination_auto` true is their `"auto"` (`offset_ = -0.5`), false
    takes the quantile and needs `contamination` in (0, 0.5].

    Every refusal here is `IsolationForestEstimator.fit`'s, which matches cuML's
    `fit`, plus DEVIATION 680's finiteness scan inside the
    implemented `fit`. `warm_start` and `sample_weight` have no argument on this
    entry at all; the Python layer refuses them by name before it gets here,
    which is where their `UnsupportedOnGPU` sits too (`:592-595`).
    """
    if n_train <= 0:
        raise Error("iforest_run_host: n_rows must be at least one")
    if n_features <= 0:
        raise Error("iforest_run_host: n_features must be at least one")
    if train_addr == 0 and len(train) != n_train * n_features:
        raise Error(
            "iforest_run_host: X has " + String(len(train))
            + " values, n_rows x n_features is " + String(n_train * n_features)
        )
    if n_query <= 0:
        raise Error("iforest_run_host: the query matrix must have at least one row")
    if query_addr == 0 and len(query) != n_query * n_features:
        raise Error(
            "iforest_run_host: the query X has " + String(len(query))
            + " values, n_rows x n_features is " + String(n_query * n_features)
        )
    if want < IF_WANT_SCORE_SAMPLES or want > IF_WANT_PREDICT:
        raise Error(
            "iforest_run_host: want=" + String(want) + " is not one of 0"
            " (score_samples), 1 (decision_function), 2 (predict)"
        )

    var ctx = process_ctx[_DEVCTX_SLOT]()
    var est = IsolationForestEstimator(ctx)
    est.n_estimators = n_estimators
    est.max_samples_mode = max_samples_mode
    est.max_samples_int = max_samples_int
    est.max_samples_frac = max_samples_frac
    est.max_depth = max_depth
    est.max_features_mode = max_features_mode
    est.max_features_int = max_features_int
    est.max_features_frac = max_features_frac
    est.bootstrap = bootstrap
    est.random_state = random_state
    est.contamination_auto = contamination_auto
    est.contamination = contamination
    est.warm_start = False
    # DEVIATION 2638: `train_addr` lends the ROW-major training block.
    # Lane fam-forests: `query_addr` (nonzero only under
    # `IDN_IF_QUERY_DEVICE`) lends the ROW-major query block the same way.
    est.fit(ctx, train, n_train, n_features, src_addr=train_addr)

    var out = IFRunOutputs()
    out.offset_ = est.offset_
    out.max_samples_ = est.max_samples_
    out.n_features_in_ = est.n_features_in_
    if want == IF_WANT_PREDICT:
        out.labels = est.predict(ctx, query, n_query, n_features, query_addr)
    elif want == IF_WANT_DECISION_FUNCTION:
        out.values = est.decision_function(ctx, query, n_query, n_features, query_addr)
    else:
        out.values = est.score_samples(ctx, query, n_query, n_features, query_addr)
    _ = est^
    # DEVIATION 1946: THE CONTEXT DIES LAST. `est.model` holds EIGHT
    # `DeviceBuffer`s (`isolation_forest.mojo:155-162`). Mojo destroys a value
    # at its LAST USE, so without this line `ctx` was destroyed at the
    # `score_samples`/`predict` call above and those eight buffers were then
    # freed against a context that had already gone -- the same class as
    # DEVIATION 1944, one call later. On an RTX 4090 (driver 580, CUDA 13)
    # that left the process wedged: the FIRST binding call returned, and the
    # NEXT GPU call in the process never did, GPU idle, every host thread in
    # futex wait. H100, M4 and MI325X never minded either shape.
    _ = ctx^
    return out^


# ===========================================================================
# THE RESIDENT ENTRY (lane fam2-forests, `IDN_IF_RESIDENT`).
#
# DEVIATION 874 above refits the forest inside every scoring call because
# the one-shot entry keeps nothing. This entry keeps the fitted
# `IsolationForestEstimator` (its model's device buffers, `offset_`, the
# knobs) in a process-lifetime store on the binding's ONE process context
# (`process_ctx`, which outlives every call, so DEVIATION 1944/1946's
# "context dies first" cannot happen to a resident buffer: each entry also
# holds its own copy of the context and destroys the model before it).
#
# Nothing is reconstructed: the resident model IS the struct `fit` filled,
# the same one the gates score through, so the route DEVIATION 874 declined
# (node arrays out to Python and back) is not taken either.
#
# Protocol. The Python layer holds a TOKEN per fitted estimator:
#   * token 0 (`fit`): build a fresh forest, keep it, hand back a new token;
#   * token t (scoring): the entry with token t AND an equal key (training
#     address, shape and every parameter the fit depends on) scores without
#     fitting. Anything else (evicted, another process, a parameter changed
#     after fit) refits from the training matrix exactly as DEVIATION 874
#     did, keeps that forest and hands back its new token.
# The entry is TAKEN OUT of the store while a call uses it (the binding
# does the take and the put under the GIL and scores with the GIL
# released), so two threads never share one entry and an eviction never
# frees buffers under a running launch; the second thread misses and refits,
# which is the old behavior.
#
# What differs from the refit form, stated: the training matrix is read at
# `fit` and not again. A caller who mutates the fitted array in place and
# then scores WITHOUT calling `fit` got a forest of the mutated data before
# and gets the fitted forest now (scikit-learn's behavior).
# ===========================================================================


#: Resident forests kept at once; the oldest is dropped past it (its owner
#: refits on its next scoring call).
comptime IF_RESIDENT_CAP = 8

comptime _IF_RESIDENT_SLOT = "MojoIForestResidentIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoIForestResidentOther"


@fieldwise_init
struct IFResidentKey(Copyable, Movable):
    """Everything the fit depends on that the caller can change between
    calls. `train_addr` is the borrowed ROW-major training block."""

    var train_addr: Int
    var n_train: Int
    var n_features: Int
    var n_estimators: Int
    var max_samples_mode: Int
    var max_samples_int: Int
    var max_samples_frac: Float64
    var max_depth: Int
    var max_features_mode: Int
    var max_features_int: Int
    var max_features_frac: Float64
    var bootstrap: Bool
    var random_state: Int
    var contamination_auto: Bool
    var contamination: Float64

    def same(self, other: Self) -> Bool:
        return (
            self.train_addr == other.train_addr
            and self.n_train == other.n_train
            and self.n_features == other.n_features
            and self.n_estimators == other.n_estimators
            and self.max_samples_mode == other.max_samples_mode
            and self.max_samples_int == other.max_samples_int
            and self.max_samples_frac == other.max_samples_frac
            and self.max_depth == other.max_depth
            and self.max_features_mode == other.max_features_mode
            and self.max_features_int == other.max_features_int
            and self.max_features_frac == other.max_features_frac
            and self.bootstrap == other.bootstrap
            and self.random_state == other.random_state
            and self.contamination_auto == other.contamination_auto
            and self.contamination == other.contamination
        )


struct IFResidentEntry(Movable):
    """One resident fitted forest. The estimator (and its model's device
    buffers) is destroyed BEFORE this entry's copy of the context
    (`IFModelShard`'s rule)."""

    var token: Int
    var key: IFResidentKey
    var ctx: DeviceContext
    var est: IsolationForestEstimator

    def __init__(
        out self, token: Int, key: IFResidentKey, ctx: DeviceContext,
        var est: IsolationForestEstimator,
    ):
        self.token = token
        self.key = key.copy()
        self.ctx = ctx.copy()
        self.est = est^

    def __deinit__(deinit self):
        _ = self.est^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


struct _IFResidentStore(Defaultable, Movable):
    var entries: List[IFResidentEntry]
    var next: Int
    var nonce: Int

    def __init__(out self):
        self.entries = List[IFResidentEntry]()
        self.next = 0
        self.nonce = 0


comptime IF_RESIDENT_STORE = _Global[
    StorageType=_IFResidentStore, name=_IF_RESIDENT_SLOT, init_fn=_IFResidentStore.__init__
]


def if_resident_new_token() raises -> Int:
    """A token no other entry of this process carries and (through the
    per-process nonce) one a token pickled in another process will not
    equal. Never 0; below 2^51, so it is exact in the float64 `info` slot.
    Call under the GIL."""
    var p = IF_RESIDENT_STORE.get_or_create_ptr()
    if p[].nonce == 0:
        p[].nonce = ((Int(perf_counter_ns()) & 0x3FFFFFF) + 1) << 24
    p[].next = (p[].next % 0xFFFFFF) + 1
    return p[].nonce + p[].next


def if_resident_take(token: Int, key: IFResidentKey) raises -> List[IFResidentEntry]:
    """The resident entry `token` names, REMOVED from the store, as a list
    of one; an empty list when there is none or its key differs (a stale
    entry under that token is dropped). Call under the GIL."""
    var held = List[IFResidentEntry]()
    if token == 0:
        return held^
    var p = IF_RESIDENT_STORE.get_or_create_ptr()
    var found = -1
    for i in range(len(p[].entries)):
        if p[].entries[i].token == token:
            found = i
            break
    if found >= 0:
        var entry = p[].entries.pop(found)
        if entry.key.same(key):
            held.append(entry^)
        else:
            _ = entry^
    return held^


def if_resident_put(mut held: List[IFResidentEntry]) raises:
    """Return a taken (or freshly fitted) entry to the store. Call under the
    GIL."""
    if len(held) == 0:
        return
    var p = IF_RESIDENT_STORE.get_or_create_ptr()
    while len(p[].entries) >= IF_RESIDENT_CAP:
        var oldest = p[].entries.pop(0)
        _ = oldest^
    p[].entries.append(held.pop())


def if_resident_release(token: Int) raises:
    """Drop the resident forest `token` names, if any (the Python estimator
    was refitted or collected). Call under the GIL."""
    if token == 0:
        return
    var p = IF_RESIDENT_STORE.get_or_create_ptr()
    var found = -1
    for i in range(len(p[].entries)):
        if p[].entries[i].token == token:
            found = i
            break
    if found >= 0:
        var entry = p[].entries.pop(found)
        _ = entry^


def iforest_run_resident(
    mut held: List[IFResidentEntry],
    key: IFResidentKey,
    new_token: Int,
    query: List[Float32],
    query_addr: Int,
    n_query: Int,
    want: Int,
    out_f32_addr: Int,
    out_i32_addr: Int,
) raises -> IFRunOutputs:
    """`iforest_run_host` with the forest resident. `held` is what
    `if_resident_take` returned: one entry scores without fitting; empty
    fits from `key.train_addr` (the refusals and the contamination quantile
    are `IsolationForestEstimator.fit`'s, unchanged) and leaves the new
    entry, under `new_token`, in `held` for `if_resident_put`.

    The result is written at `out_f32_addr` (`want` 0 and 1, `n_query`
    float32) or `out_i32_addr` (`want` 2, `n_query` int32); the returned
    lists are empty. `query_addr` nonzero lends the ROW-major query block
    (`IDN_IF_QUERY_DEVICE`), else `query` holds it."""
    if key.n_train <= 0:
        raise Error("iforest_run_host: n_rows must be at least one")
    if key.n_features <= 0:
        raise Error("iforest_run_host: n_features must be at least one")
    if key.train_addr == 0:
        raise Error("iforest_run_resident: the training matrix address is null")
    if n_query <= 0:
        raise Error("iforest_run_host: the query matrix must have at least one row")
    if query_addr == 0 and len(query) != n_query * key.n_features:
        raise Error(
            "iforest_run_host: the query X has " + String(len(query))
            + " values, n_rows x n_features is " + String(n_query * key.n_features)
        )
    if want < IF_WANT_SCORE_SAMPLES or want > IF_WANT_PREDICT:
        raise Error(
            "iforest_run_host: want=" + String(want) + " is not one of 0"
            " (score_samples), 1 (decision_function), 2 (predict)"
        )
    if want == IF_WANT_PREDICT:
        if out_i32_addr == 0:
            raise Error("iforest_run_resident: the label output address is null")
    elif out_f32_addr == 0:
        raise Error("iforest_run_resident: the score output address is null")

    var ctx = process_ctx[_DEVCTX_SLOT]()
    if len(held) == 0:
        var est = IsolationForestEstimator(ctx)
        est.n_estimators = key.n_estimators
        est.max_samples_mode = key.max_samples_mode
        est.max_samples_int = key.max_samples_int
        est.max_samples_frac = key.max_samples_frac
        est.max_depth = key.max_depth
        est.max_features_mode = key.max_features_mode
        est.max_features_int = key.max_features_int
        est.max_features_frac = key.max_features_frac
        est.bootstrap = key.bootstrap
        est.random_state = key.random_state
        est.contamination_auto = key.contamination_auto
        est.contamination = key.contamination
        est.warm_start = False
        est.fit(ctx, List[Float32](), key.n_train, key.n_features, src_addr=key.train_addr)
        held.append(IFResidentEntry(new_token, key, ctx, est^))

    var out = IFRunOutputs()
    out.offset_ = held[0].est.offset_
    out.max_samples_ = held[0].est.max_samples_
    out.n_features_in_ = held[0].est.n_features_in_
    out.token = held[0].token
    var n_features = key.n_features
    comptime if IDN_IF_EPILOGUE_DEVICE:
        if not held[0].est.fitted:
            raise Error("Model has not been fitted. Call fit() first.")
        if want == IF_WANT_PREDICT:
            if_predict_into(
                ctx, held[0].est.model, query, n_query, n_features,
                Float32(-held[0].est.offset_), held[0].est.knobs, query_addr, out_i32_addr,
            )
        else:
            if_score_samples_into(
                ctx, held[0].est.model, query, n_query, n_features, held[0].est.knobs,
                query_addr, out_f32_addr, want == IF_WANT_DECISION_FUNCTION,
                Float32(held[0].est.offset_),
            )
    else:
        if want == IF_WANT_PREDICT:
            var labels = held[0].est.predict(ctx, query, n_query, n_features, query_addr)
            var oi = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=out_i32_addr)
            for i in range(n_query):
                oi.unsafe_store(i, labels[i])
        else:
            var values: List[Float32]
            if want == IF_WANT_DECISION_FUNCTION:
                values = held[0].est.decision_function(
                    ctx, query, n_query, n_features, query_addr
                )
            else:
                values = held[0].est.score_samples(ctx, query, n_query, n_features, query_addr)
            var of = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_f32_addr)
            for i in range(n_query):
                of.unsafe_store(i, values[i])
    # The context dies after everything this call launched (DEVIATION 1946);
    # the resident buffers hold their own copy of it.
    _ = ctx^
    return out^
