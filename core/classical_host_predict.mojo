# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Classical inference on the host, for a box with no GPU (the classical host
inference lane, 2026-09-13; brief
docs/lanes/BRIEF_forest_host_inference_2026-09-13.md, "Classical lanes").

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and the GPU bindings do not import this file. It exists because the five
inference entries of `bindings/_mojolearn_estimators.mojo` that
LinearRegression, Ridge, TruncatedSVD, LogisticRegression and PCA call
(`ols_predict`, `tsvd_transform`, `qn_decision_function`, `qn_sigmoid`,
`pca_transform`) all reach kernels in files that import `max.gpu.host` at
module level (`core/gemm.mojo:8`, `core/column_stats.mojo`,
`glm/impl/qn/glm_base.mojo`, `glm/estimator.mojo:37-38`), while the
arithmetic each kernel performs is a handful of `checks/numerics.mojo`
calls, which is GPU-free.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS. Every function below names the
kernel it mirrors and keeps its statements in its order:

  `host_pinned_cell`       `pinned_gemm_nt_kernel`, `core/gemm.mojo:33-62`,
                           one output cell; `pinned_gemv_n_kernel`,
                           `core/gemm.mojo:93-113`, is the same loop over one
                           row, so `gemm_nt` (`:137`, which routes `n == 1`
                           to `gemv_n`, `:322`) and `gemv_n` are one cell
                           function here. Under IDENTICAL, `gemm_nt` launches
                           the pinned kernel unless the build defines
                           MOJOLEARN_537_GEMM_IDENT_SWAP, which no build script
                           does (`git grep 537_GEMM_IDENT_SWAP` finds only
                           bench/speed/gemm_speed_main.mojo's opt-in).
  `host_gemm_nt`           the kernel's grid, every cell in row-major order.
  `host_ols_predict`       `ols_predict_host`, `glm/estimator.mojo:192-231`:
                           the gemv, then `_add_scalar_kernel` (`:53-74`)
                           only when the intercept is not exactly 0.0.
  `host_qn_decision`       `linear_fwd` at C == 1, `glm/impl/qn/glm_base.mojo:
                           198-244`: the gemv over the first D entries of W,
                           then `add_bias_kernel` (`:121-134`) when
                           fit_intercept, whose bias read is NOT flushed.
  `host_qn_sigmoid`        `qn_sigmoid_host`, `glm/estimator.mojo:384-410`,
                           which already runs on the host and is relocated
                           here because its file imports `std.gpu`.
  `host_qn_decision_multi` `linear_fwd` at C > 1 (lane/logistic-multiclass,
                           2026-09-14), `glm/impl/qn/glm_base.mojo`:
                           `transpose_w_kernel` (a copy, `w_rm[c*D + j] =
                           w[c + C*j]`), `gemm_nt(z, x, w_rm, n, C, D)` (the
                           pinned cell, row-major `z[i*C + c]`), then
                           `add_bias_multi_kernel` when fit_intercept:
                           `z[cell] = ftz(z[cell] + w[C*D + c])`, `c = cell
                           % C`, the bias read NOT flushed.
  `host_qn_softmax`        `qn_softmax_host`, `glm/estimator.mojo`, the
                           multinomial predict_proba link in Float64 on the
                           host: first maximum under a strict `>`, serial
                           ascending sum of `identical_exp64(z - m)`, one
                           division per cell.
  `host_pca_transform`     `pca_transform`, `decomposition/impl/linalg/detail/
                           pca.mojo:319-355`: `shift_columns_kernel` with
                           sign -1.0 (`core/column_stats.mojo:153-204`), then
                           `gemm_nt`. The restoring shift (+1.0) is applied to
                           the device copy of X, which `pca_transform_host`
                           never reads back, so it changes no output and is
                           not restated.
  `host_tsvd_transform`    `tsvd_transform_host`, `decomposition/estimator.mojo:
                           265-281`: one `gemm_nt`.
  `host_whiten_scalar`     `whiten_scalar`, `decomposition/impl/linalg/detail/
                           pca.mojo:366-375`: `Float32(sqrt(Float64(n_fit_rows
                           - 1)))` forward, its Float64 reciprocal inverse; the
                           Float64 sqrt is correctly rounded on every host.
  `host_whiten_components` `whiten_scale_kernel` through `whiten_components`
                           (`pca.mojo:377-430`), one cell per statement: `v =
                           ftz(identical_mul(src, scalar))`, kept when the
                           component's singular value is below
                           WHITEN_SKIP_ZERO, else `ftz(identical_div(v, s))`
                           forward and `ftz(identical_mul(v, s))` inverse.
  `host_pca_whiten_transform`
                           `pca_whiten_transform_host`, `decomposition/
                           estimator.mojo:339-396` (the kde svc host lane,
                           2026-09-14): whiten the components forward, then
                           `pca_transform` over the whitened copy.
  `host_pca_whiten_inverse_transform`
                           `pca_whiten_inverse_transform_host`, `:399-455`:
                           whiten inverse, `transpose_kernel` (data movement,
                           restated as an index swap), `gemm_nt(out, scores,
                           components_t, n_rows, n_features, n_components)`,
                           then `shift_columns_kernel` at sign +1.0.

The restatement is a prediction until measured. tools/classical_host_gate.py
is the measurement, and the brief records what it has shown.

THE POINTER ENTRIES AND THE ROW SPLIT (lane/infer-speed-classical,
2026-09-17, DEVIATION 2920). Every entry above has an `_into` twin that
reads X through the caller's pointer and writes the caller's output
directly, so the host bindings copy nothing (a 500,000 x 220 X was one
440 MB `read_f32` copy per predict), and `host_gemm_nt_into` splits its
output ROWS into contiguous tasks (`core/host_predict_threads.mojo`, the
count from MOJOLEARN_CPU_THREADS or one per physical core). One cell is
still `host_pinned_cell_ptr`, the same statements in the same order, and
one row's cells are still written in ascending column order by one task;
the intercept and bias epilogues, which are one statement per output, run
on the calling thread after the join exactly as before. The List entries
below are kept for the checks and route through the same twins, so there
is one spelling of every arithmetic and the sabotage arm reaches it from
either door. `host_pca_transform_into` centers each row into a task-local
row buffer with `host_center_cell` (the statement `shift_columns_kernel`
spells) instead of a whole centered copy of X; the cell reads the same
flushed values.
"""
from std.sys.compile import is_defined

from std.math import sqrt

from max.algorithm import sync_parallelize

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp64,
    identical_mul,
    identical_mul_add,
)
from core.host_predict_threads import (
    HostF32Ptr,
    HostF64Ptr,
    host_list_ptr,
    host_predict_chunk,
    host_predict_task_count,
)


#: The gate's negative control, the phase 1 host bindings' define
#: (`gemm/host/gemm_oracle.mojo:90`, `kde/host/kde_oracle.mojo:86`), so
#: one -D MOJOLEARN_HOST_SABOTAGE=1 build sabotages every host arithmetic in
#: `_mojolearn_estimators_host`. A build with it walks every dot product's
#: k loop DESCENDING, which is wrong on purpose (a serial float32 fold in
#: the other order is a different bit pattern on almost every row), and
#: the gate must say so. Read back by `estimators_host_sabotage`.
comptime CLASSICAL_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


@always_inline
def host_pinned_cell_ptr(
    x: HostF32Ptr, x_off: Int, y: HostF32Ptr, y_off: Int, k: Int,
) -> Float32:
    """One cell of `pinned_gemm_nt_kernel` (`core/gemm.mojo:52-62`):

        var acc = Float32(0.0)
        for p in range(k):
            acc = ftz(identical_mul_add(ftz(x[i*k+p]), ftz(y[j*k+p]), acc))
        z[cell] = ftz(Float32(0.0) + ftz(acc))

    `x_off` is `i * k`, `y_off` is `j * k`. The final `0.0 + acc` turns a
    `-0.0` accumulator into `+0.0`, and is kept for that reason. THE ONE
    SPELLING: `host_pinned_cell` and every `_into` entry call this."""
    var acc = Float32(0.0)
    comptime if CLASSICAL_HOST_SABOTAGE:
        # THE SABOTAGE ARM: the same fold, walked DESCENDING. Wrong on
        # purpose; see CLASSICAL_HOST_SABOTAGE.
        for q in range(k):
            var p = k - 1 - q
            acc = ftz(identical_mul_add(
                ftz(x.unsafe_load(x_off + p)), ftz(y.unsafe_load(y_off + p)), acc
            ))
        return ftz(Float32(0.0) + ftz(acc))
    for p in range(k):
        acc = ftz(identical_mul_add(
            ftz(x.unsafe_load(x_off + p)), ftz(y.unsafe_load(y_off + p)), acc
        ))
    return ftz(Float32(0.0) + ftz(acc))


def host_pinned_cell(
    x: List[Float32], x_off: Int, y: List[Float32], y_off: Int, k: Int,
) -> Float32:
    """`host_pinned_cell_ptr` over two Lists, the checks' door."""
    return host_pinned_cell_ptr(host_list_ptr(x), x_off, host_list_ptr(y), y_off, k)


def host_gemm_nt_into(
    x: HostF32Ptr, y: HostF32Ptr, z: HostF32Ptr, m: Int, n: Int, k: Int,
    tasks: Int,
):
    """`z[m x n] = x[m x k] . y[n x k]^T`, `gemm_nt` under IDENTICAL
    (`core/gemm.mojo:137-160`): `pinned_gemm_nt_kernel` over `m * n` cells,
    `i = cell // n`, `j = cell % n`; `n == 1` is `pinned_gemv_n_kernel`, the
    same fold over the one column. DEVIATION 2920: the rows are split into
    at most `tasks` contiguous ranges (`core/host_predict_threads.mojo`);
    a task writes its own rows' cells in ascending cell order, reads `x`
    and `y` only, and every cell is `host_pinned_cell_ptr`, so the bits are
    the serial walk's. One task runs on the calling thread."""
    if m <= 0 or n <= 0:
        return
    var t = tasks
    if t < 1:
        t = 1
    if t > m:
        t = m
    var chunk = host_predict_chunk(m, t)

    def _rows(c: Int) {imm x, imm y, imm z, imm chunk, imm m, imm n, imm k}:
        var lo = c * chunk
        var hi = min(lo + chunk, m)
        for i in range(lo, hi):
            for j in range(n):
                z.unsafe_store(i * n + j, host_pinned_cell_ptr(x, i * k, y, j * k, k))

    if t == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, t)


def host_gemm_nt(
    x: List[Float32], y: List[Float32], m: Int, n: Int, k: Int,
) -> List[Float32]:
    """`host_gemm_nt_into` into a fresh List, at the policy's task count."""
    var z = List[Float32](length=m * n, fill=Float32(0.0))
    host_gemm_nt_into(
        host_list_ptr(x), host_list_ptr(y), host_list_ptr(z), m, n, k,
        host_predict_task_count(m),
    )
    return z^


def host_ols_predict(
    x: List[Float32], coef: List[Float32], n_rows: Int, n_features: Int,
    intercept: Float32,
) -> List[Float32]:
    """`ols_predict_host` (`glm/estimator.mojo:192-231`): `gemv_n`, then
    `_add_scalar_kernel` (`:53-74`, `dst[i] = ftz(ftz(dst[i]) + ftz(value))`)
    only `if intercept != Float32(0.0)`, the host compare that file audits
    under DEVIATION 527."""
    var out = List[Float32](length=n_rows, fill=Float32(0.0))
    host_ols_predict_into(
        host_list_ptr(x), host_list_ptr(coef), host_list_ptr(out), n_rows,
        n_features, intercept, host_predict_task_count(n_rows),
    )
    return out^


def host_ols_predict_into(
    x: HostF32Ptr, coef: HostF32Ptr, dst: HostF32Ptr, n_rows: Int,
    n_features: Int, intercept: Float32, tasks: Int,
):
    """`host_ols_predict` over the caller's memory (DEVIATION 2920): the
    gemv rows split across `tasks`, then the intercept epilogue, one
    statement per row on the calling thread, only `if intercept !=
    Float32(0.0)`."""
    host_gemm_nt_into(x, coef, dst, n_rows, 1, n_features, tasks)
    if intercept != Float32(0.0):
        for i in range(n_rows):
            var v = ftz(dst.unsafe_load(i))
            dst.unsafe_store(i, ftz(v + ftz(intercept)))


def host_qn_decision(
    x: List[Float32], w: List[Float32], n_rows: Int, n_features: Int,
    fit_intercept: Bool,
) -> List[Float32]:
    """`qn_decision_function_host` (`glm/estimator.mojo:353-381`) at the
    binary logistic shape, `n_targets = 1`: `linear_fwd`'s C == 1 arm
    (`glm/impl/qn/glm_base.mojo:230-244`), `gemv_n` over `w[0:D]` (the
    `w_head` copy), then `add_bias_kernel` (`:121-134`) when
    `fit_intercept`: `z[i] = ftz(z[i] + b)` with `b = w[D]` read as it is,
    NOT flushed (that kernel flushes the sum only)."""
    var z = List[Float32](length=n_rows, fill=Float32(0.0))
    host_qn_decision_into(
        host_list_ptr(x), host_list_ptr(w), host_list_ptr(z), n_rows,
        n_features, fit_intercept, host_predict_task_count(n_rows),
    )
    return z^


def host_qn_decision_into(
    x: HostF32Ptr, w: HostF32Ptr, z: HostF32Ptr, n_rows: Int,
    n_features: Int, fit_intercept: Bool, tasks: Int,
):
    """`host_qn_decision` over the caller's memory (DEVIATION 2920): the
    `w_head` copy, the gemv rows split across `tasks`, then the bias
    epilogue on the calling thread, `b = w[D]` read as it is."""
    var w_head = List[Float32](length=n_features, fill=Float32(0.0))
    for j in range(n_features):
        w_head[j] = w.unsafe_load(j)
    host_gemm_nt_into(x, host_list_ptr(w_head), z, n_rows, 1, n_features, tasks)
    if fit_intercept:
        var b = w.unsafe_load(n_features)
        for i in range(n_rows):
            z.unsafe_store(i, ftz(z.unsafe_load(i) + b))
    _ = w_head^


def host_qn_sigmoid(
    scores: List[Float32], n_rows: Int,
) -> List[Float64]:
    """`qn_sigmoid_host` (`glm/estimator.mojo:384-410`), DEVIATION 549:
    `p = 1 / (1 + identical_exp64(-Float64(z)))` in float64, `1 - p` for
    class 0, `p` for class 1, `(n_rows, 2)` row-major."""
    var out = List[Float64](length=2 * n_rows, fill=Float64(0.0))
    host_qn_sigmoid_into(
        host_list_ptr(scores), rebind[HostF64Ptr](out.unsafe_ptr()), n_rows,
        1,
    )
    return out^


def host_qn_sigmoid_into(
    scores: HostF32Ptr, dst: HostF64Ptr, n_rows: Int, tasks: Int,
):
    """`host_qn_sigmoid` over caller-owned buffers, split by rows.

    A row owns both adjacent output cells and performs the same Float64
    conversion, exponential, division and subtraction as the serial entry.
    No reduction crosses rows, so task scheduling cannot move a bit.
    """
    if n_rows <= 0:
        return
    var t = max(1, min(tasks, n_rows))
    var chunk = host_predict_chunk(n_rows, t)

    def _rows(c: Int) {imm scores, imm dst, imm chunk, imm n_rows}:
        var lo = c * chunk
        var hi = min(lo + chunk, n_rows)
        for i in range(lo, hi):
            var z = Float64(scores.unsafe_load(i))
            var p = 1.0 / (1.0 + identical_exp64(-z))
            dst.unsafe_store(2 * i, 1.0 - p)
            dst.unsafe_store(2 * i + 1, p)

    if t == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, t)


def host_qn_decision_multi(
    x: List[Float32], w: List[Float32], n_rows: Int, n_features: Int,
    n_classes: Int, fit_intercept: Bool,
) -> List[Float32]:
    """`qn_decision_function_host` at the softmax shape (`n_classes > 2`,
    lane/logistic-multiclass, 2026-09-14): `linear_fwd`'s C > 1 arm
    (`glm/impl/qn/glm_base.mojo`). `transpose_w_kernel` copies the
    column-major weight block to row-major `w_rm[c*D + j] = w[c + C*j]`
    (no arithmetic); `gemm_nt(z, x, w_rm, n_rows, C, D)` is the pinned
    cell over `n_rows * C` cells, `z[i*C + c]`; `add_bias_multi_kernel`,
    when `fit_intercept`, stores `ftz(z[cell] + b)` with `b = w[C*D + c]`
    read as it is, `c = cell % C`."""
    var z = List[Float32](length=n_rows * n_classes, fill=Float32(0.0))
    host_qn_decision_multi_into(
        host_list_ptr(x), host_list_ptr(w), host_list_ptr(z), n_rows,
        n_features, n_classes, fit_intercept, host_predict_task_count(n_rows),
    )
    return z^


def host_qn_decision_multi_into(
    x: HostF32Ptr, w: HostF32Ptr, z: HostF32Ptr, n_rows: Int,
    n_features: Int, n_classes: Int, fit_intercept: Bool, tasks: Int,
):
    """`host_qn_decision_multi` over the caller's memory (DEVIATION 2920):
    the row-major weight copy, the gemm rows split across `tasks`, then
    the bias epilogue on the calling thread in cell order."""
    var d = n_features
    var w_rm = List[Float32](length=n_classes * d, fill=Float32(0.0))
    for cell in range(n_classes * d):
        var c = cell // d
        var j = cell % d
        w_rm[cell] = w.unsafe_load(c + n_classes * j)
    host_gemm_nt_into(x, host_list_ptr(w_rm), z, n_rows, n_classes, d, tasks)
    if fit_intercept:
        for cell in range(n_classes * n_rows):
            var c = cell % n_classes
            var b = w.unsafe_load(n_classes * d + c)
            z.unsafe_store(cell, ftz(z.unsafe_load(cell) + b))
    _ = w_rm^


def host_qn_softmax(
    scores: List[Float32], n_rows: Int, n_classes: Int,
) -> List[Float64]:
    """`qn_softmax_host` (`glm/estimator.mojo`, lane/logistic-multiclass,
    2026-09-14) statement for statement: per row of the `(n_rows, C)`
    float32 scores, `m` the first maximum under a strict `>` from the
    first entry, `s` the serial ascending sum of `identical_exp64(z - m)`,
    `p_c = identical_exp64(z_c - m) / s`; `(n_rows, C)` float64
    row-major."""
    var out = List[Float64](length=n_rows * n_classes, fill=Float64(0.0))
    host_qn_softmax_into(
        host_list_ptr(scores), rebind[HostF64Ptr](out.unsafe_ptr()), n_rows,
        n_classes, 1,
    )
    return out^


def host_qn_softmax_into(
    scores: HostF32Ptr, dst: HostF64Ptr, n_rows: Int, n_classes: Int,
    tasks: Int,
):
    """`host_qn_softmax` over caller-owned buffers, split by rows.

    Each task retains the reference's strict ascending maximum, sum and
    output loops within every row. Rows share no state, so parallelism and
    the removal of the two boundary Lists are bitwise inert.
    """
    if n_rows <= 0:
        return
    var t = max(1, min(tasks, n_rows))
    var chunk = host_predict_chunk(n_rows, t)

    def _rows(task: Int) {imm scores, imm dst, imm chunk, imm n_rows, imm n_classes}:
        var lo = task * chunk
        var hi = min(lo + chunk, n_rows)
        for i in range(lo, hi):
            var base = i * n_classes
            var m = Float64(scores.unsafe_load(base))
            for c in range(1, n_classes):
                var v = Float64(scores.unsafe_load(base + c))
                if v > m:
                    m = v
            var s = 0.0
            for c in range(n_classes):
                var z = Float64(scores.unsafe_load(base + c))
                s = s + identical_exp64(z - m)
            for c in range(n_classes):
                var z = Float64(scores.unsafe_load(base + c))
                dst.unsafe_store(base + c, identical_exp64(z - m) / s)

    if t == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, t)


def host_center(
    x: List[Float32], mu: List[Float32], n_rows: Int, n_cols: Int,
    sign: Float32,
) -> List[Float32]:
    """`shift_columns_kernel` (`core/column_stats.mojo:153-204`), one local
    per flush as that kernel spells it:

        var xv = ftz(x[idx]); var mv = ftz(mu[col])
        x[idx] = ftz(xv + sign * mv)

    Returns the shifted copy; the kernel shifts the device copy in place."""
    var out = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for idx in range(n_rows * n_cols):
        var col = idx % n_cols
        out[idx] = host_center_cell(x[idx], mu[col], sign)
    return out^


@always_inline
def host_center_cell(x: Float32, m: Float32, sign: Float32) -> Float32:
    """One cell of `shift_columns_kernel`, the two locals as it spells
    them: `xv = ftz(x)`, `mv = ftz(mu)`, `ftz(xv + sign * mv)`."""
    var xv = ftz(x)
    var mv = ftz(m)
    return ftz(xv + sign * mv)


def host_pca_transform(
    x: List[Float32], mu: List[Float32], components: List[Float32],
    n_rows: Int, n_cols: Int, n_components: Int,
) -> List[Float32]:
    """`pca_transform` (`decomposition/impl/linalg/detail/pca.mojo:319-355`):
    center with sign `-1.0`, then `gemm_nt(out, x, components, n_rows,
    n_components, n_cols)`; `host_pca_transform_into` into a fresh List."""
    var out = List[Float32](length=n_rows * n_components, fill=Float32(0.0))
    host_pca_transform_into(
        host_list_ptr(x), host_list_ptr(mu), host_list_ptr(components),
        host_list_ptr(out), n_rows, n_cols, n_components,
        host_predict_task_count(n_rows),
    )
    return out^


def host_pca_transform_into(
    x: HostF32Ptr, mu: HostF32Ptr, components: HostF32Ptr, dst: HostF32Ptr,
    n_rows: Int, n_cols: Int, n_components: Int, tasks: Int,
):
    """`host_pca_transform` over the caller's memory (DEVIATION 2920). A
    task centers each of its rows into its own `n_cols` row buffer with
    `host_center_cell` (the value `shift_columns_kernel` leaves in the
    device copy of that cell), then writes the row's `n_components` cells
    through `host_pinned_cell_ptr` over that buffer, ascending. The same
    flushed values reach the same fold in the same order as the whole
    centered copy did; no float crosses a row."""
    if n_rows <= 0 or n_components <= 0:
        return
    var t = tasks
    if t < 1:
        t = 1
    if t > n_rows:
        t = n_rows
    var chunk = host_predict_chunk(n_rows, t)

    def _rows(c: Int) {imm x, imm mu, imm components, imm dst, imm chunk, imm n_rows, imm n_cols, imm n_components}:
        var centered = List[Float32](length=n_cols, fill=Float32(0.0))
        var cp = host_list_ptr(centered)
        var lo = c * chunk
        var hi = min(lo + chunk, n_rows)
        for i in range(lo, hi):
            for col in range(n_cols):
                centered[col] = host_center_cell(
                    x.unsafe_load(i * n_cols + col), mu.unsafe_load(col), Float32(-1.0)
                )
            for j in range(n_components):
                dst.unsafe_store(
                    i * n_components + j,
                    host_pinned_cell_ptr(cp, 0, components, j * n_cols, n_cols),
                )
        _ = centered^

    if t == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, t)


def host_tsvd_transform(
    x: List[Float32], components: List[Float32],
    n_rows: Int, n_cols: Int, n_components: Int,
) -> List[Float32]:
    """`tsvd_transform_host` (`decomposition/estimator.mojo:265-281`): one
    `gemm_nt(out, x, components, n_rows, n_components, n_features)`."""
    return host_gemm_nt(x, components, n_rows, n_components, n_cols)


def host_tsvd_transform_into(
    x: HostF32Ptr, components: HostF32Ptr, dst: HostF32Ptr,
    n_rows: Int, n_cols: Int, n_components: Int, tasks: Int,
):
    """`host_tsvd_transform` over the caller's memory (DEVIATION 2920)."""
    host_gemm_nt_into(x, components, dst, n_rows, n_components, n_cols, tasks)


#: `decomposition/impl/linalg/detail/pca.mojo::WHITEN_SKIP_ZERO`, cuML's
#: skip-zero threshold, spelled a second time so this file imports nothing
#: from a module that imports `max.gpu`.
comptime HOST_WHITEN_SKIP_ZERO = 1.0e-10


def host_whiten_scalar(n_fit_rows: Int, inverse: Bool) -> Float32:
    """`whiten_scalar` (`pca.mojo:366-375`): `sqrt(n_fit_rows - 1)` forward,
    `1 / sqrt(n_fit_rows - 1)` inverse, both in Float64 and rounded once to
    Float32; 0.0 when the guard the reference wrote fires."""
    var d = Float64(n_fit_rows - 1)
    if d <= 0.0:
        return Float32(0.0)
    var r = sqrt(d)
    if inverse:
        return Float32(1.0 / r)
    return Float32(r)


def host_whiten_components(
    components: List[Float32], singular: List[Float32],
    n_components: Int, n_cols: Int, n_fit_rows: Int, inverse: Bool,
) -> List[Float32]:
    """`whiten_components` (`pca.mojo:400-430`) launching `whiten_scale_kernel`
    (`:377-397`) over `n_components * n_cols` cells, `c = i // n_cols`:

        var s = singular[c]
        var v = ftz(identical_mul(src[i], scalar))
        if abs(s) < WHITEN_SKIP_ZERO:  dst[i] = v
        elif divide:                   dst[i] = ftz(identical_div(v, s))
        else:                          dst[i] = ftz(identical_mul(v, s))

    `divide` is 1 forward and 0 inverse, as `whiten_components` sets it."""
    var scalar = host_whiten_scalar(n_fit_rows, inverse)
    var divide = not inverse
    var cells = n_components * n_cols
    var dst = List[Float32](length=cells, fill=Float32(0.0))
    for i in range(cells):
        var c = i // n_cols
        var s = singular[c]
        var v = ftz(identical_mul(components[i], scalar))
        if abs(s) < Float32(HOST_WHITEN_SKIP_ZERO):
            dst[i] = v
        elif divide:
            dst[i] = ftz(identical_div(v, s))
        else:
            dst[i] = ftz(identical_mul(v, s))
    return dst^


def host_pca_whiten_transform(
    x: List[Float32], mu: List[Float32], components: List[Float32],
    singular: List[Float32], n_rows: Int, n_cols: Int, n_components: Int,
    n_fit_rows: Int,
) -> List[Float32]:
    """`pca_whiten_transform_host` (`decomposition/estimator.mojo:339-396`):
    `whiten_components(..., inverse=False)` into a copy, then `pca_transform`
    over that copy, the caller's components untouched."""
    var components_w = host_whiten_components(
        components, singular, n_components, n_cols, n_fit_rows, False
    )
    return host_pca_transform(x, mu, components_w, n_rows, n_cols, n_components)


def host_pca_whiten_transform_into(
    x: HostF32Ptr, mu: List[Float32], components: List[Float32],
    singular: List[Float32], dst: HostF32Ptr, n_rows: Int, n_cols: Int,
    n_components: Int, n_fit_rows: Int, tasks: Int,
):
    """The whitened forward transform over caller-owned buffers.

    Only the small component matrix is materialized because whitening changes
    it.  Input rows and output cells stay in caller memory, exactly as in the
    non-whitened ``host_pca_transform_into`` path.
    """
    var components_w = host_whiten_components(
        components, singular, n_components, n_cols, n_fit_rows, False
    )
    host_pca_transform_into(
        x, host_list_ptr(mu), host_list_ptr(components_w), dst, n_rows, n_cols,
        n_components, tasks,
    )
    _ = components_w^


def host_pca_whiten_inverse_transform(
    scores: List[Float32], components: List[Float32], singular: List[Float32],
    mu: List[Float32], n_rows: Int, n_cols: Int, n_components: Int,
    n_fit_rows: Int,
) -> List[Float32]:
    """`pca_whiten_inverse_transform_host` (`decomposition/estimator.mojo:
    399-455`): `whiten_components(..., inverse=True)`, `transpose_kernel`
    (`components_t[f * n_components + c] = components_w[c * n_cols + f]`, a
    copy and no arithmetic), `gemm_nt(out, scores, components_t, n_rows,
    n_features, n_components)`, then `shift_columns_kernel` with sign +1.0
    over the product, which `host_center` spells."""
    var components_w = host_whiten_components(
        components, singular, n_components, n_cols, n_fit_rows, True
    )
    var components_t = List[Float32](length=n_cols * n_components, fill=Float32(0.0))
    for c in range(n_components):
        for f in range(n_cols):
            components_t[f * n_components + c] = components_w[c * n_cols + f]
    var product = host_gemm_nt(scores, components_t, n_rows, n_cols, n_components)
    return host_center(product, mu, n_rows, n_cols, Float32(1.0))


def host_pca_whiten_inverse_transform_into(
    scores: HostF32Ptr, components: List[Float32], singular: List[Float32],
    mu: List[Float32], dst: HostF32Ptr, n_rows: Int, n_cols: Int,
    n_components: Int, n_fit_rows: Int, tasks: Int,
):
    """The whitened inverse transform over caller-owned buffers.

    The transpose remains pure data movement.  The GEMM writes directly into
    ``dst`` and the final mean shift visits those same cells in place; both
    statements are identical to the List-returning reference above.
    """
    var components_w = host_whiten_components(
        components, singular, n_components, n_cols, n_fit_rows, True
    )
    var components_t = List[Float32](
        length=n_cols * n_components, fill=Float32(0.0)
    )
    for c in range(n_components):
        for f in range(n_cols):
            components_t[f * n_components + c] = components_w[c * n_cols + f]
    host_gemm_nt_into(
        scores, host_list_ptr(components_t), dst, n_rows, n_cols,
        n_components, tasks,
    )
    for i in range(n_rows):
        for f in range(n_cols):
            var idx = i * n_cols + f
            dst.unsafe_store(
                idx,
                host_center_cell(
                    dst.unsafe_load(idx), mu[f], Float32(1.0)
                ),
            )
    _ = components_t^
    _ = components_w^
