# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""`cuml/cpp/src/glm/preprocess.cuh::{preProcessData, postProcessData}`, the
UNWEIGHTED arms -- cuML 26.08, pinned in `solver/DERIVATION_MAP.tsv`.

This file was written under `solver/derived/glm/` by the cd lane (the glm
section had `preprocess.cuh` as NOT PORTED; OLS's Python surface centers
on the host instead) and MOVED here 2026-08-23 by the identity lane, per
the hand-off in `solver/README.md`: it is the same upstream file and this
is where it belongs. `solver/derived/solver/cd.mojo` is its one caller
today; `glm/derived/glm/ols.mojo` still refuses `fit_intercept` on the
device and could take it next (glm/NOT_IMPLEMENTED.tsv). Nothing here is
solver-specific.

`preProcessData(handle, input, n_rows, n_cols, labels, intercept, mu_input,
mu_labels, fit_intercept, sample_weight = nullptr)`, `preprocess.cuh:36-73`:

    ASSERT(n_cols > 0); ASSERT(n_rows > 1)
    if fit_intercept:
        mean<false>(mu_input, input, n_cols, n_rows)          :58
        meanCenter<false, true>(input, input, mu_input, n_cols, n_rows)   :60
        mean<false>(mu_labels, labels, 1, n_rows)             :65
        meanCenter<false, true>(labels, labels, mu_labels, 1, n_rows)     :67

`postProcessData(...)`, `preprocess.cuh:75-110`:

    gemm(mu_input [1 x n_cols], coef [n_cols x 1]) -> d_intercept   :88-98
    subtract(d_intercept, mu_labels, d_intercept)                   :100
    *intercept = d_intercept.value(stream)                          :101
    meanAdd<false, true>(input, input, mu_input, n_cols, n_rows)    :103
    meanAdd<false, true>(labels, labels, mu_labels, 1, n_rows)      :104

BOTH MUTATE THE CALLER'S `input` AND `labels` IN PLACE, and `postProcess`
puts back `x - mu + mu`, which is NOT bitwise `x`: a caller's matrix comes
back rounded in its last bit wherever the centering lost one. That is
their behavior and it is mirrored, not fixed; `solver/README.md` says so
under the objective mapping. (The 26.08 file has no `normalize` arm -- the
`meanvar`/`colNorm` scaling older cuML had was removed -- so there is no
`normalize` parameter to honor or refuse here; the Python estimators do not
expose one either.)

`sample_weight` (the `weightedMean` arms, `:55-56,:62-63`) is REFUSED BY
NAME at `cd_fit`; it never reaches this file.

THE `gemm` ON THE MEANS IS cuBLAS (CLOSED). `1 x n_cols` times `n_cols x
1`: under FAST the mirror is `core/gemm.mojo::gemv_n` at `m = 1`
(`linalg.gemv.gemv_gpu`; `linalg.matmul` does not write at `n = 1`, VENDOR
_LIBS.md), under IDENTICAL it is the profile dot over `k = n_cols`
(`profile_dot.mojo`), so the intercept is a function of the inputs alone.
The `subtract` is a one-thread kernel on the device, as theirs, and the
host reads the result once.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.gemm import gemv_n
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from solver.original.profile_dot import profile_dot_into
from solver.derived.stats.mean import column_mean, mean_add, mean_center


def pre_process_data(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut labels: DeviceBuffer[DType.float32],
    mut mu_input: DeviceBuffer[DType.float32],
    mut mu_labels: DeviceBuffer[DType.float32],
    fit_intercept: Bool,
    mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    plan: Int = -1,
) raises:
    """`preProcessData`, unweighted. `ones`/`ws`/`plan` feed the IDENTICAL
    arm of the two means (`solver/derived/stats/mean.mojo`)."""
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if fit_intercept:
        column_mean(ctx, mu_input, x, n_cols, n_rows, ones, ws, plan)
        mean_center(ctx, x, mu_input, n_cols, n_rows)
        column_mean(ctx, mu_labels, labels, 1, n_rows, ones, ws, plan)
        mean_center(ctx, labels, mu_labels, 1, n_rows)


def subtract_one_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    in1: MutPointer[Float32, MutAnyOrigin],
    in2: MutPointer[Float32, MutAnyOrigin],
):
    """`raft::linalg::subtract(out, in1, in2, 1)`: `out[0] = in1[0] - in2[0]`,
    one thread. The intercept is a seam the host reads: `ftz`."""
    out_v.unsafe_store(0, ftz(in1.unsafe_load(0) - in2.unsafe_load(0)))


def post_process_data(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut labels: DeviceBuffer[DType.float32],
    mut coef: DeviceBuffer[DType.float32],
    mut mu_input: DeviceBuffer[DType.float32],
    mut mu_labels: DeviceBuffer[DType.float32],
    mut d_intercept: DeviceBuffer[DType.float32],
    mut ws_cols: DeviceBuffer[DType.float32],
    plan: Int = -1,
) raises -> Float32:
    """`postProcessData` with `fit_intercept = true`. Returns the intercept
    read from the device (`d_intercept.value(stream)`). `d_intercept` is a
    1-float device scalar the caller owns; `ws_cols` is a profile workspace
    sized for `k = n_cols` (IDENTICAL arm)."""
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    # gemm(mu_input, 1, n_cols, coef, d_intercept, 1, 1, N, N)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        profile_dot_into(ctx, d_intercept, mu_input, coef, ws_cols, n_cols, plan)
    else:
        gemv_n(ctx, d_intercept, mu_input, coef, 1, n_cols)
    # subtract(d_intercept, mu_labels, d_intercept, 1): in place, and Mojo
    # refuses one buffer as two launch arguments, so `out` is a second
    # 1-float scratch: the same arithmetic, one more address.
    var tmp = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[subtract_one_kernel](
        tmp.unsafe_ptr(),
        mu_labels.unsafe_ptr(),
        d_intercept.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=d_intercept, src_buf=tmp)
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_intercept)
    ctx.synchronize()
    var intercept = h.unsafe_ptr().unsafe_load(0)
    mean_add(ctx, x, mu_input, n_cols, n_rows)
    mean_add(ctx, labels, mu_labels, 1, n_rows)
    _ = tmp^
    _ = h^
    return intercept
