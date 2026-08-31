# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python-facing surface: `IsolationForest(n_estimators, max_samples,
max_depth, max_features, bootstrap, random_state, contamination)` with
`fit`, `score_samples`, `decision_function`, `predict`, `fit_predict`.

MIRRORS `python/cuml/cuml/ensemble/isolation_forest.pyx` at rapidsai/cuml
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
`_score_samples_nvforest`) is NOT ported (NOT_IMPLEMENTED.tsv).

Host only; the device work is `derived/isolation_forest/`. The percentile
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

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from isolation_forest.derived.isolation_forest.isolation_forest import (
    IF_params,
    IFLaunchKnobs,
    IsolationForestModel,
    fit as if_fit,
    predict as if_predict,
    score_samples as if_score_samples,
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
    return a + (b - a) * frac


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
        context (`original/if_hang_probe.mojo`) returned the M4's bits.
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
        mut self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int
    ) raises:
        """`IsolationForest.fit(X)` (`:572-787`). `sample_weight` has no
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

        # order="F" for fit (:599-605): a copy, no arithmetic
        var x_col = List[Float32]()
        for k in range(n_cols):
            for i in range(n_rows):
                x_col.append(x_rowmajor[i * n_cols + k])
        var trace = IdentityTrace()
        if_fit(ctx, self.model, x_col, n_rows, n_cols, params, trace, self.knobs)
        self.fitted = True

        if use_quantile:
            var training_scores = self.score_samples(ctx, x_rowmajor, n_rows, n_cols)
            self.offset_ = percentile_linear(training_scores, 100.0 * self.contamination)
        else:
            self.offset_ = -0.5

    def score_samples(
        self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int
    ) raises -> List[Float32]:
        """`score_samples(X)` (`:894-959`): `-paper_score`."""
        if not self.fitted:
            raise Error("Model has not been fitted. Call fit() first.")
        var trace = IdentityTrace.disabled()
        var paper = if_score_samples(ctx, self.model, x_rowmajor, n_rows, n_cols, trace, self.knobs)
        var out = List[Float32]()
        for i in range(n_rows):
            out.append(-paper[i])
        return out^

    def decision_function(
        self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int
    ) raises -> List[Float32]:
        """`decision_function(X) = score_samples(X) - offset_` (`:978`).
        The subtraction is the Python layer's (float32 array minus a
        Python float: numpy/cupy compute it in float32)."""
        var s = self.score_samples(ctx, x_rowmajor, n_rows, n_cols)
        var off = Float32(self.offset_)
        var out = List[Float32]()
        for i in range(n_rows):
            out.append(s[i] - off)
        return out^

    def predict(
        self, ctx: DeviceContext, x_rowmajor: List[Float32], n_rows: Int, n_cols: Int
    ) raises -> List[Int32]:
        """`predict(X)` (`:981-1042`): `-C++predict(X, threshold =
        -offset_)`; sklearn convention, -1 = anomaly, 1 = inlier."""
        if not self.fitted:
            raise Error("Model has not been fitted. Call fit() first.")
        var threshold = Float32(-self.offset_)
        var raw = if_predict(ctx, self.model, x_rowmajor, n_rows, n_cols, threshold, self.knobs)
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

    def __init__(out self):
        self.values = List[Float32]()
        self.labels = List[Int32]()
        self.offset_ = -0.5
        self.max_samples_ = 0
        self.n_features_in_ = 0


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

    Every refusal here is `IsolationForestEstimator.fit`'s, which is cuML's
    `fit` transcribed, plus DEVIATION 680's finiteness scan inside the
    ported `fit`. `warm_start` and `sample_weight` have no argument on this
    entry at all; the Python layer refuses them by name before it gets here,
    which is where their `UnsupportedOnGPU` sits too (`:592-595`).
    """
    if n_train <= 0:
        raise Error("iforest_run_host: n_rows must be at least one")
    if n_features <= 0:
        raise Error("iforest_run_host: n_features must be at least one")
    if len(train) != n_train * n_features:
        raise Error(
            "iforest_run_host: X has " + String(len(train))
            + " values, n_rows x n_features is " + String(n_train * n_features)
        )
    if n_query <= 0:
        raise Error("iforest_run_host: the query matrix must have at least one row")
    if len(query) != n_query * n_features:
        raise Error(
            "iforest_run_host: the query X has " + String(len(query))
            + " values, n_rows x n_features is " + String(n_query * n_features)
        )
    if want < IF_WANT_SCORE_SAMPLES or want > IF_WANT_PREDICT:
        raise Error(
            "iforest_run_host: want=" + String(want) + " is not one of 0"
            " (score_samples), 1 (decision_function), 2 (predict)"
        )

    var ctx = DeviceContext()
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
    est.fit(ctx, train, n_train, n_features)

    var out = IFRunOutputs()
    out.offset_ = est.offset_
    out.max_samples_ = est.max_samples_
    out.n_features_in_ = est.n_features_in_
    if want == IF_WANT_PREDICT:
        out.labels = est.predict(ctx, query, n_query, n_features)
    elif want == IF_WANT_DECISION_FUNCTION:
        out.values = est.decision_function(ctx, query, n_query, n_features)
    else:
        out.values = est.score_samples(ctx, query, n_query, n_features)
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
