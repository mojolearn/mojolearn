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
`_score_samples_nvforest`) is NOT ported (UNPORTED.tsv).

Host only; the device work is `ported/isolation_forest/`. The percentile
is numpy/cupy's default `linear` interpolation computed in Float64 over
the sorted float32 scores (`percentile_linear`); the threshold handed to
the device is `Float32(-offset_)` as their `<float>threshold` cast.
`random_state=None` is NOT a fresh random seed here: it is 0, stated,
because a fit whose seed nobody recorded cannot be reproduced and this
repository's product is the reproducible fit.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from isolation_forest.ported.isolation_forest.isolation_forest import (
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

    def __init__(out self) raises:
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
        self.model = IsolationForestModel(DeviceContext())
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
