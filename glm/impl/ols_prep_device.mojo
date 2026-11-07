# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LinearRegression / Ridge general-route preparation in one native call
(the device column; `glm/host/ols_prep_host.mojo` is the host column).

ne2-g-linear (N403): `bindings/classical_ols.mojo` and
`bindings/classical_ridge.mojo` made one binding crossing per step (weight
sum, weighted row scale, column sums, means finish, the y twins, two
centerings, the weight roots and two row scalings). This function runs the
same typed steps (`glm/impl/center_device.mojo`, `glm/impl/lm_finish.mojo`,
`core/py2mojo_rows.mojo` ROWS_SQRT_F32) in the same order on the same words,
so the prepared X and y, the means and the y mean keep their bits.

means_mode: OLS_PREP_SEPARATE (column sums of X, then of y),
OLS_PREP_PAIRED (`col_sums_pair_device`, the lm_classical_stats route),
OLS_PREP_WEIGHTED (sample-weighted means; w required).
"""
from max.gpu.host import DeviceContext

from core.py2mojo_rows import ROWS_SQRT_F32, rows_call, rows_device
from glm.impl.center_device import center_device, col_sums_device, col_sums_pair_device, scale_rows_device
from glm.impl.lm_finish import lm_means_finish

comptime OLS_PREP_SEPARATE = 0
comptime OLS_PREP_PAIRED = 1
comptime OLS_PREP_WEIGHTED = 2


def _f64p(address: Int) -> MutPointer[Float64, MutUntrackedOrigin]:
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)


def _f32p(address: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def ols_prep_device(
    ctx: DeviceContext, x: Int, y: Int, w: Int, work_x: Int, work_y: Int,
    xm64: Int, xm32: Int, ym64: Int, ym32: Int,
    rows: Int, cols: Int, intercept: Bool, means_mode: Int,
) raises:
    """work_x (rows x cols) and work_y (rows) float32: X and y centered by
    their (weighted) means when `intercept`, then each row scaled by
    sqrt(w) when w != 0. xm64/xm32 (cols) and ym64/ym32 (1) receive the
    binary64 means and their float32 roundings when `intercept`."""
    if rows <= 0 or cols <= 0:
        raise Error("ols_prep: rows and cols must be positive")
    var weighted = w != 0
    if means_mode == OLS_PREP_WEIGHTED and not weighted:
        raise Error("ols_prep: weighted means need sample weights")
    # Host scratch: the binary64 sums (as words), one row-scaled copy for the
    # weighted means and the centered copies before the root scaling.
    var sx = List[Float64](length=cols, fill=0.0)
    var sy = List[Float64](length=1, fill=0.0)
    var tx = List[Float32](length=rows * cols if weighted else 1, fill=0.0)
    var ty = List[Float32](length=rows if weighted else 1, fill=0.0)
    var sxa = Int(sx.unsafe_ptr())
    var sya = Int(sy.unsafe_ptr())
    var cx = x
    var cy = y
    if intercept:
        if means_mode == OLS_PREP_WEIGHTED:
            # classical_weighted_means(x, w) then (y, w): sum(w), refusal,
            # x * w, its column sums, the weighted finish.
            var wsum = List[Float64](length=1, fill=0.0)
            col_sums_device(ctx, w, Int(wsum.unsafe_ptr()), rows, 1)
            var total = wsum[0]
            if total <= 0:
                raise Error("sample_weight sums to zero")
            scale_rows_device(ctx, x, w, Int(tx.unsafe_ptr()), rows, cols)
            col_sums_device(ctx, Int(tx.unsafe_ptr()), sxa, rows, cols)
            lm_means_finish(_f64p(sxa), _f64p(xm64), _f32p(xm32), cols, rows, True, total)
            col_sums_device(ctx, w, Int(wsum.unsafe_ptr()), rows, 1)
            total = wsum[0]
            if total <= 0:
                raise Error("sample_weight sums to zero")
            scale_rows_device(ctx, y, w, Int(ty.unsafe_ptr()), rows, 1)
            col_sums_device(ctx, Int(ty.unsafe_ptr()), sya, rows, 1)
            lm_means_finish(_f64p(sya), _f64p(ym64), _f32p(ym32), 1, rows, True, total)
            _ = wsum^
        elif means_mode == OLS_PREP_PAIRED:
            col_sums_pair_device(ctx, x, y, sxa, sya, rows, cols)
            lm_means_finish(_f64p(sxa), _f64p(xm64), _f32p(xm32), cols, rows, False, 0.0)
            lm_means_finish(_f64p(sya), _f64p(ym64), _f32p(ym32), 1, rows, False, 0.0)
        else:
            col_sums_device(ctx, x, sxa, rows, cols)
            lm_means_finish(_f64p(sxa), _f64p(xm64), _f32p(xm32), cols, rows, False, 0.0)
            col_sums_device(ctx, y, sya, rows, 1)
            lm_means_finish(_f64p(sya), _f64p(ym64), _f32p(ym32), 1, rows, False, 0.0)
        # classical_center: into the scratch when the root scaling follows,
        # else straight into the outputs.
        var dx = Int(tx.unsafe_ptr()) if weighted else work_x
        var dy = Int(ty.unsafe_ptr()) if weighted else work_y
        center_device(ctx, x, xm32, dx, rows, cols)
        center_device(ctx, y, ym32, dy, rows, 1)
        cx = dx
        cy = dy
    if weighted:
        # py2mojo_rows ROWS_SQRT_F32 over the weights, then the row scaling.
        var root = List[Float32](length=rows, fill=0.0)
        var call = rows_call(ROWS_SQRT_F32, rows, 1, w, Int(root.unsafe_ptr()), 0, 0.0)
        rows_device(ctx, call)
        scale_rows_device(ctx, cx, Int(root.unsafe_ptr()), work_x, rows, cols)
        scale_rows_device(ctx, cy, Int(root.unsafe_ptr()), work_y, rows, 1)
        _ = root^
    _ = sx^
    _ = sy^
    _ = tx^
    _ = ty^
