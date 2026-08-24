"""Host-pointer surfaces for the coordinate-descent section: Lasso, ElasticNet.

**THIS IS THE ENTRY THE PYTHON PACKAGE USES.**
`bindings/_mojolearn_solver.mojo` calls `cd_fit_host` and `cd_predict_host`,
and `python/mojolearn/_solver_impl.py` calls those, so everything below is
what a `mojolearn.Lasso().fit(X, y)` actually runs. The shape is
`dbscan/estimator.mojo`'s and `glm/estimator.mojo`'s: host pointers in,
device buffers owned here for exactly one call, results read back, nothing
retained.

The ported solver is `solver/ported/solver/cd.mojo` (cuML
`cpp/src/solver/cd.cuh`, pinned at `v26.08.00` = `265b9da`); the lane's
README, `PORTED_MAP.tsv` and `UNPORTED.tsv` are the record of what is and is
not in it. Nothing here re-decides any of that: every guard cdFit carries is
reached through `cd_fit_traced`, and this file adds no guard of its own
beyond the two shape checks below.

THE DESIGN MATRIX IS COLUMN-MAJOR
---------------------------------
`cdFit` takes `input` in cuML's `F` order: element `(i, j)` of an
`n_rows x n_cols` design lives at `j * n_rows + i`. `x_ptr` here is that
layout, NOT the row-major layout every other host surface in this repository
takes. The Python layer does the conversion with `np.asfortranarray` and
names the copy, because doing it here would be a scalar host loop over
`n_rows * n_cols` elements in Mojo where numpy has a blocked one.
`cd_predict_host`'s `x_ptr` is column-major for the same reason
(`linearRegH` reads the same layout, `solver/ported/functions/linear_reg.mojo`).

==========================================================================
DEVIATION 880 -- THE CALLER'S DESIGN MATRIX IS NOT MUTATED
==========================================================================
WHAT THEIRS DOES: `cdFit` centers `input` and `labels` IN PLACE under
`fit_intercept` (`preProcessData`) and un-centers them at the end
(`postProcessData`), and the un-centering does not restore the original
bits -- `x + mu` after `x - mu` is not `x` in float32. cuML's Python hands
the user's own device array to that function, so a cuML `ElasticNet.fit`
leaves the caller's `X` and `y` subtly changed.

WHAT OURS DOES: `cd_fit_host` copies `x_ptr` and `y_ptr` into device
buffers it owns and the centering happens there, so the caller's arrays
come back byte for byte as they went in. This is a DEPARTURE from theirs
and it is recorded here rather than being silently better, because the
difference is observable: a caller who fits twice on the same array gets
the same answer here and does not there.

The cost is one device-resident copy of `X`, which the host-pointer
boundary requires anyway (there is no way to hand a numpy buffer to a
kernel without it), so the deviation is free.

WHAT IS REFUSED, AND WHERE
--------------------------
Every refusal below is `cd.mojo`'s, reached from here, not re-implemented:
`loss != SQRD_LOSS`, `sample_weight`, `shuffle=true` (`selection='random'`;
DEVIATION 611 reserved, not spent), `n_cols <= 0`, `n_rows <= 1`,
`alpha < 0`, `l1_ratio` outside `[0, 1]`, and DEVIATION 613's non-finite
`alpha` / NaN `l1_ratio` / NaN `tol`. `has_sample_weight` and `shuffle`
are plumbed through as parameters SO THAT those two refusals are reachable
from this surface -- a refusal only the Python layer can raise is a
refusal the Mojo entry never proves it still has.

THE IDENTITY CARD
-----------------
`cd_fit_host` calls `cd_fit_traced` with a live `IdentityTrace()`, so
`MOJOLEARN_IDENTITY_TRACE=<path>` set around a Python fit writes the same
20-record `cd.*` card `solver/cd_main.mojo` writes for the same problem.
That is DEVIATION 517's lesson applied here before it could be repeated:
the one path with a Python user on the other end must not be the one path
that cannot leave a card. The trace is off unless the variable is set.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from solver.ported.solver.cd import CdLaunch, cd_fit_traced, cd_predict
from solver.ported.solvers.params import LOSS_SQRD_LOSS


def cd_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    fit_intercept: Bool,
    epochs: Int,
    alpha: Float32,
    l1_ratio: Float32,
    tol: Float32,
    shuffle: Bool,
    has_sample_weight: Bool,
) raises -> Int:
    """`cdFit` over host pointers. Returns `n_iter` (the epochs actually run).

    `x_ptr` is COLUMN-MAJOR `n_rows x n_cols` float32 (see the module
    docstring); `y_ptr` is `n_rows`; `coef_ptr` receives `n_cols` floats;
    `info_ptr[0]` receives the intercept (0.0 when `fit_intercept` is False,
    which is the literal `cd_fit_traced` returns, not a computed zero).

    THE COEFFICIENT VECTOR IS ZEROED HERE, AND THAT IS A DECISION.
    `cdFit` READS `coef` as its starting point and never zeroes it
    (`cd.cuh` has no memset); cuML's Python passes `cp.zeros` on every fit
    because `warm_start=True` is refused upstream (`_params_from_cpu`).
    This surface does the same, so a fit is a pure function of `(X, y,
    params)` and not of whatever was in the caller's output array. Warm
    starting is therefore NOT reachable from Python, which matches cuML.
    """
    if n_rows < 1 or n_cols < 1:
        raise Error(
            "cd_fit_host needs n_rows and n_cols >= 1: got "
            + String(n_rows) + ", " + String(n_cols)
        )
    if epochs < 0:
        raise Error(
            "cd_fit_host: max_iter cannot be negative, got " + String(epochs)
        )

    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var coef = ctx.enqueue_create_buffer[DType.float32](n_cols)
    # `want_residual` is False below, so this never receives a copy; it
    # exists because `cd_fit_traced` takes the destination by `mut`.
    var residual_out = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y, src_ptr=y_ptr)
    ctx.enqueue_memset(coef, Float32(0.0))
    ctx.synchronize()

    # The trace reads MOJOLEARN_IDENTITY_TRACE and is disabled unless it is
    # set. `cd_fit_traced` writes its own header line, so none is written
    # here (`glm/estimator.mojo` writes one because `ols_fit_traced` does
    # not; the two are not the same shape and copying that line would put
    # two headers on one card).
    var trace = IdentityTrace()
    var out = cd_fit_traced(
        ctx, x, n_rows, n_cols, y, coef,
        fit_intercept, epochs, LOSS_SQRD_LOSS, alpha, l1_ratio, shuffle,
        tol, has_sample_weight, trace, "cd", CdLaunch.default(),
        residual_out, False,
    )
    var n_iter = out[0]
    var intercept = out[1]

    var hc = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=coef)
    ctx.synchronize()
    for i in range(n_cols):
        coef_ptr.unsafe_store(i, hc.unsafe_ptr().unsafe_load(i))
    info_ptr.unsafe_store(0, intercept)

    # [[mojo-buffer-freed-at-last-use]]: every buffer outlives the queue.
    _ = hc^
    _ = x^
    _ = y^
    _ = coef^
    _ = residual_out^
    return n_iter


def cd_predict_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    intercept: Float32,
) raises:
    """`cdPredict` over host pointers (`cd.cuh:331-346` in the
    `upstream/cuml-v26.08.00` checkout, 265b9da). `x_ptr` is COLUMN-MAJOR `n_rows x n_cols`; `out_ptr` receives
    `n_rows` floats.

    `cdPredict` REFUSES `n_rows <= 1` exactly as `cdFit` does (their
    `ASSERT(n_rows > 1, ...)`), so predicting on a single row raises by
    name rather than returning one number. That is theirs, carried, and the
    Python layer says so on the class.
    """
    if n_rows < 1 or n_cols < 1:
        raise Error(
            "cd_predict_host needs n_rows and n_cols >= 1: got "
            + String(n_rows) + ", " + String(n_cols)
        )

    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var coef = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var preds = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=coef, src_ptr=coef_ptr)
    ctx.synchronize()

    cd_predict(ctx, x, n_rows, n_cols, coef, intercept, preds, LOSS_SQRD_LOSS)

    var hp = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hp.unsafe_ptr(), src_buf=preds)
    ctx.synchronize()
    for i in range(n_rows):
        out_ptr.unsafe_store(i, hp.unsafe_ptr().unsafe_load(i))

    # [[mojo-buffer-freed-at-last-use]]
    _ = hp^
    _ = x^
    _ = coef^
    _ = preds^
