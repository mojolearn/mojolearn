# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of `glm/impl/ols_prep_device.mojo` (ne2-g-linear N403):
the same steps in the same order through the host bindings' own typed
routines (`glm/host/center_host.mojo`, `glm/impl/lm_finish.mojo`,
`core/py2mojo_rows.mojo` ROWS_SQRT_F32's `rows_host`), so the same words."""
from core.py2mojo_rows import ROWS_SQRT_F32, rows_call, rows_host
from glm.host.center_host import center_on_cpu, col_sums_on_cpu, scale_rows_on_cpu
from glm.impl.lm_finish import lm_means_finish

comptime OLS_PREP_HOST_SEPARATE = 0
comptime OLS_PREP_HOST_PAIRED = 1
comptime OLS_PREP_HOST_WEIGHTED = 2


def _f64p(address: Int) -> MutPointer[Float64, MutUntrackedOrigin]:
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)


def _f32p(address: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def ols_prep_host(
    x: Int, y: Int, w: Int, work_x: Int, work_y: Int,
    xm64: Int, xm32: Int, ym64: Int, ym32: Int,
    rows: Int, cols: Int, intercept: Bool, means_mode: Int,
) raises:
    """`ols_prep_device`'s contract on the host."""
    if rows <= 0 or cols <= 0:
        raise Error("ols_prep: rows and cols must be positive")
    var weighted = w != 0
    if not intercept and not weighted:
        raise Error("ols_prep: nothing to prepare (no intercept, no sample weights)")
    if means_mode == OLS_PREP_HOST_WEIGHTED and not weighted:
        raise Error("ols_prep: weighted means need sample weights")
    var sx = List[Float64](length=cols, fill=0.0)
    var sy = List[Float64](length=1, fill=0.0)
    var tx = List[Float32](length=rows * cols if weighted else 1, fill=0.0)
    var ty = List[Float32](length=rows if weighted else 1, fill=0.0)
    var sxa = Int(sx.unsafe_ptr())
    var sya = Int(sy.unsafe_ptr())
    var cx = x
    var cy = y
    if intercept:
        if means_mode == OLS_PREP_HOST_WEIGHTED:
            var wsum = List[Float64](length=1, fill=0.0)
            col_sums_on_cpu(w, Int(wsum.unsafe_ptr()), rows, 1)
            var total = wsum[0]
            if total <= 0:
                raise Error("sample_weight sums to zero")
            scale_rows_on_cpu(x, w, Int(tx.unsafe_ptr()), rows, cols)
            col_sums_on_cpu(Int(tx.unsafe_ptr()), sxa, rows, cols)
            lm_means_finish(_f64p(sxa), _f64p(xm64), _f32p(xm32), cols, rows, True, total)
            col_sums_on_cpu(w, Int(wsum.unsafe_ptr()), rows, 1)
            total = wsum[0]
            if total <= 0:
                raise Error("sample_weight sums to zero")
            scale_rows_on_cpu(y, w, Int(ty.unsafe_ptr()), rows, 1)
            col_sums_on_cpu(Int(ty.unsafe_ptr()), sya, rows, 1)
            lm_means_finish(_f64p(sya), _f64p(ym64), _f32p(ym32), 1, rows, True, total)
            _ = wsum^
        else:
            # The host binding's lm_col_sums_pair is the two column-sum
            # passes, so the paired and separate modes are the same words.
            col_sums_on_cpu(x, sxa, rows, cols)
            lm_means_finish(_f64p(sxa), _f64p(xm64), _f32p(xm32), cols, rows, False, 0.0)
            col_sums_on_cpu(y, sya, rows, 1)
            lm_means_finish(_f64p(sya), _f64p(ym64), _f32p(ym32), 1, rows, False, 0.0)
        var dx = Int(tx.unsafe_ptr()) if weighted else work_x
        var dy = Int(ty.unsafe_ptr()) if weighted else work_y
        center_on_cpu(x, xm32, dx, rows, cols)
        center_on_cpu(y, ym32, dy, rows, 1)
        cx = dx
        cy = dy
    if weighted:
        var root = List[Float32](length=rows, fill=0.0)
        var call = rows_call(ROWS_SQRT_F32, rows, 1, w, Int(root.unsafe_ptr()), 0, 0.0)
        rows_host(call)
        scale_rows_on_cpu(cx, Int(root.unsafe_ptr()), work_x, rows, cols)
        scale_rows_on_cpu(cy, Int(root.unsafe_ptr()), work_y, rows, 1)
        _ = root^
    _ = sx^
    _ = sy^
    _ = tx^
    _ = ty^
