# SPDX-License-Identifier: Apache-2.0
"""RUN of MultiOutputRegressor's one-program ridge (the batched path of
bindings/trees_public_meta_door.mojo `_multioutput_ridge_fit`; no Python
import, no member call).

`RidgeMultiFit` is the fit as plain values: the admitted X and Y addresses
and dimensions, the member's alpha and intercept choice (read once by the
door's UNPACK), and the addresses of the buffers the door allocated before
RUN and publishes in PACK. `ridge_multi_fit_run` is the four programs of the
estimators image's ridge multi table (core/linear_multi_calls.mojo): the
column sums, the means, the centering and the ridge program over every
target.

Each program is one synchronous launch of the image; the parallelism is
inside it. Nothing here loops over rows or targets.
"""
from core.entry_table import EntryRef
from core.linear_multi_calls import rm_col_sums, rm_means_finish, rm_center, rm_fit, rm_fit_slot


struct RidgeMultiFit(Movable):
    """One batched ridge fit (see the module docstring). `sums`, `means64`
    and `work` are read only when `intercept` is set; `x_mean` and `icpt`
    are then written, else left as allocated."""
    var x: Int
    var y: Int
    var n: Int
    var d: Int
    var m: Int
    var alpha: Float64
    var intercept: Bool
    #: float64[d] scratch: the column sums and the float64 means
    var sums: Int
    var means64: Int
    #: float32[d]: the column means the members store
    var x_mean: Int
    #: float32[n x d]: the centered X
    var work: Int
    #: float32[m x d], float32[m], float32[m]
    var coef: Int
    var y_mean: Int
    var icpt: Int
    #: THE FOLD FROM THE STORE (lane vdf-meta5): the estimators image's
    #: resident store slot holding X (n x d, a model-selection fold taken on
    #: the device), or -1: X is the host words at `x`
    var slot: Int

    def __init__(out self, x: Int, y: Int, n: Int, d: Int, m: Int, alpha: Float64, intercept: Bool):
        self.slot = -1
        self.x = x
        self.y = y
        self.n = n
        self.d = d
        self.m = m
        self.alpha = alpha
        self.intercept = intercept
        self.sums = 0
        self.means64 = 0
        self.x_mean = 0
        self.work = 0
        self.coef = 0
        self.y_mean = 0
        self.icpt = 0


def ridge_multi_fit_run(key: EntryRef, fit: RidgeMultiFit) raises:
    """Every target's ridge in one program: with an intercept, the column
    means of X (`_column_means`: the sums, then the finish) and the centered
    X first; then the ridge program, which writes the coefficients, the
    target means and (with an intercept) the intercepts. A fit over a
    store slot (`fit.slot`) runs the same four programs inside the image in
    one call (`rm_fit_slot`): X is never uploaded and the centered X never
    leaves the device (`work` is not read)."""
    if fit.slot >= 0:
        rm_fit_slot(key, fit.slot, fit.y, fit.coef, fit.y_mean, fit.n, fit.d, fit.m, fit.alpha, fit.intercept,
                    fit.sums, fit.means64, fit.x_mean, fit.icpt)
        return
    if fit.intercept:
        rm_col_sums(key, fit.x, fit.sums, fit.n, fit.d)
        rm_means_finish(key, fit.sums, fit.means64, fit.x_mean, fit.d, fit.n, False, 0.0)
        rm_center(key, fit.x, fit.x_mean, fit.work, fit.n, fit.d)
        rm_fit(key, fit.work, fit.y, fit.coef, fit.y_mean, fit.n, fit.d, fit.m, fit.alpha, True, fit.x_mean,
               fit.icpt)
        return
    rm_fit(key, fit.x, fit.y, fit.coef, fit.y_mean, fit.n, fit.d, fit.m, fit.alpha, False, 0, 0)
