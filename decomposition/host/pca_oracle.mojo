# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""PCA and truncated SVD TRAINING on the host, for a box with no GPU
(workstream E of docs/lanes/TEMP_claim_surface_plan_2026-09-14.md, the
lanes pca, pca-whiten and tsvd of docs/lanes/BRIEF_cpu_training_2026-09-13.md
section 1.1 "pca, tsvd", 2026-09-14).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and no GPU binding imports this file. Every arithmetic statement is a
`checks/numerics.mojo` leaf (`ftz`, `identical_mul_add`, `identical_sqrt`)
or a plain IEEE float32 operation the device kernel also performs
unpinned, spelled a SECOND time from the device kernels named below, in
their order. The brief's "stage only" verdict on this lane was exact: the
existing host Jacobi (`decomposition/checks/jacobi_eigh.mojo`) is Float64
at 1e-12 and 60 sweeps and disclaims bit agreement, and the split-K Gram
was gated against a Float64 tolerance, so the FOLDS are restated here at
the device's settings rather than imported.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS. Every function names the
kernel it mirrors and keeps its statements in its order:

  `host_column_mean`       `column_mean_kernel`, `core/column_stats.mojo:104`:
                           one block of STATS_TPB lanes per column, lane `t`
                           adds rows `t, t + STATS_TPB, ...` with a PLAIN
                           float32 `+=` (no flush inside the chain), the
                           partials fold through `pinned_block_sum`'s
                           halving tree (`core/pinned_reduce.mojo:102`,
                           `two_phase_halving_sum`, whose two phases are
                           the tree `red[t] += red[t + step]` for `step =
                           TPB/2 .. 1` in that order), the total is
                           flushed, then `mu = ftz(s0 / Float32(n_rows))`.
  `host_gram_splitk`       `_gram_splitk_partial_body` and
                           `gram_splitk_reduce_kernel`,
                           `core/gram_splitk.mojo:488, 822`, the arm
                           `gemm_tn` takes under IDENTICAL for every Gram
                           of at most GRAM_MAX_COLS columns on EVERY column
                           (DEVIATION 521). The k axis is cut into
                           `PINNED_GRAM_SPLITK_CHUNKS = 128` chunks of
                           `ceil(k / 128)` rows (at least 1); chunk `c`
                           owns rows `c * kc .. min((c + 1) * kc, k)`, and
                           a chunk past `k` writes an all-zero partial. Per
                           chunk and per cell `(i, j)` the products
                           `tile[r, i] * tile[r, j]` (the i operand FIRST)
                           accumulate through `identical_mul_add` with NO
                           flush between steps, r ascending; the partial is
                           flushed at its store. The reduce is the serial
                           ascending chunk fold `acc = ftz(acc +
                           ftz(partial))`, then `ftz(acc)`. Which thread
                           owns which cell (the register-tile, uniform-jj
                           and strided arms) is scheduling and is not
                           restated. The CENTERED read is the fused
                           epilogue `ftz(ftz(x) - ftz(mu))` (DEVIATION 42,
                           522), the one `compute_covariance` takes for PCA.
  `host_gemm_tn`           `gemm_tn`, `core/gemm.mojo:205`: the split-K
                           arm where `gram_splitk_applies` (`m <= 128` and
                           `m * m <= GRAM_TPB * GRAM_MAX_CELLS_PER_THREAD`),
                           else `gemm_tn_identical_v1`, profile
                           mojolearn.identical.gemm.fp32.v1 at OP_TN, whose
                           host definition is `gemm/host/gemm_oracle.mojo::
                           gemm_oracle`. No identity_break fixture is wider
                           than 17 columns, so the v1 arm is UNMEASURED on
                           this lane and is named here rather than refused.
  `host_shift_columns`     `shift_columns_kernel`, `core/column_stats.mojo:
                           153`: `ftz(ftz(x) + sign * ftz(mu))`. PCA's
                           unfused arm only (past GRAM_MAX_COLS); the
                           restore pass writes a copy no output reads and
                           is not restated.
  `host_scale_in_place`    `scale_in_place_kernel`, `:324`: `ftz(a * scale)`
                           with `scale = Float32(1.0) / Float32(n_rows - 1)`
                           computed on the host as `compute_covariance`
                           computes it.
  `host_jacobi_eigh`       `jacobi_eigh_kernel`, `decomposition/checks/
                           jacobi_eigh_device.mojo:243`, one block, cyclic
                           by row. The two folds (`||A||_F^2` once, the
                           off-diagonal sum per sweep) stride the flattened
                           matrix by JACOBI_TPB lanes with `acc =
                           ftz(identical_mul_add(v, v, acc))` per lane and
                           fold through the same halving tree
                           (`_fold_lead_lanes_and_broadcast`, whose wide arm
                           is `two_phase_halving_sum[JACOBI_TPB]` with the
                           lanes past JACOBI_TPB contributing nothing). The
                           launch width JACOBI_ROT_TPB is scheduling
                           (DEVIATION 2680) and is not restated. `limit =
                           ftz(ftz(tol * tol) * fro2)`, the test `2 * off <=
                           limit`, the rotation `(c, s)` of
                           `jacobi_rotation_cs` (its flushes and its
                           `identical_sqrt` in its order), and the merged
                           phase of DEVIATION 2671: for every `k` outside
                           `{p, q}` the column pair `(k, p), (k, q)` then
                           the row pair `(p, k), (q, k)`, the lane owning
                           `k == p` doing the 2 x 2 block in
                           `_rotate_pair_block`'s column-then-row order,
                           every lane its basis pair. No cell is written by
                           two lanes and no lane reads a cell another lane
                           of the same phase writes, so a serial walk over
                           `k` in any order stores the same values; this
                           one walks `k` ascending. The rotation formulas
                           are `_rot_sub` and `_rot_add` verbatim: fuse the
                           first product, round the second, flush both.
                           `info` is (converged, rel, executed) as the
                           kernel writes it.
  `host_sign_flip`         `sign_flip_kernel`, `decomposition/impl/linalg/
                           detail/pca.mojo:143`, per column: the largest
                           |v| under a strict `>` from `+0.0` (so a NaN
                           never enters), the smallest row index whose |v|
                           equals it (sentinel `n`), negate the column when
                           that row's value is negative. A max and a min
                           are selections and are order free; the serial
                           scan is the same value the halving selections
                           return.
  `host_order_truncate_spectrum`
                           `order_truncate_spectrum`, `pca.mojo:203`, the
                           HOST Float64 tail of the fit, copied: the
                           descending selection sort of indices under a
                           strict `>`, the total, the components as columns
                           of the basis, `sqrt(lam * singular_scale)` with
                           the stdlib Float64 `sqrt` (correctly rounded on
                           every host, not an `identical_*` seam, and it
                           must stay that way), the noise variance.
  `host_pca_fit`           `pca_fit_host`, `decomposition/estimator.mojo:53`:
                           `pca_validate`, `compute_covariance` (mean,
                           centered split-K Gram or shift plus `gemm_tn`,
                           scale by `1 / (n_rows - 1)`), `eig_and_truncate`
                           (Jacobi, sign flip, the convergence refusal in
                           the same words, the Float64 tail at
                           `singular_scale = n_rows - 1`), the mean.
  `host_tsvd_fit`          `tsvd_fit_host`, `:219`: `pca_validate`, the
                           PLAIN split-K Gram of X as stored through
                           `gemm_tn`, `eig_and_truncate` at
                           `singular_scale = 1`.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` (the routed families'
one define, `python/mojolearn/host_surface.py`) makes `host_gram_splitk`'s
reduce walk the 128 chunk partials DESCENDING, a different summation
order for every Gram cell with more than one live chunk, so the
covariance the Jacobi receives differs in its last bits and every PCA and
tSVD output moves. Read back by `estimators_host_sabotage`. The v1 arm
past 128 columns carries `gemm_oracle`'s own arm under the same define.

The restatement is a prediction until measured. The CPU identity gate
(`tools/identity_break.py --diff <3 GPU columns> <cpu json> --lanes
pca,pca-whiten,tsvd --require-columns 4`) is the measurement, and the
brief records what it has shown.
"""
from std.math import sqrt
from std.sys.compile import is_defined

from max.algorithm import sync_parallelize

from checks.kernel_matrix import (
    K_LIB_COLUMN_STATS,
    K_LIB_JACOBI_EIGH,
    TARGET_COLUMN,
    lib_block_size_for,
)
from checks.numerics import ftz, identical_mul_add, identical_sqrt
from core.host_predict_threads import (
    host_list_ptr,
    host_predict_chunk,
    host_predict_task_count,
)
from gemm.host.identical_gemm import OP_TN, gemm_oracle


#: The gate's negative control (the CPU training lane, brief section 3.4):
#: the same define every routed host arithmetic reads. Read back by
#: `estimators_host_sabotage`; refused outside the gate by
#: `python/mojolearn/_backend.py::load_host_module`.
comptime PCA_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: The fold widths, READ FROM THE MATRIX as the kernels read them
#: (`core/column_stats.mojo:101`, `jacobi_eigh_device.mojo:28`). Both are
#: numeric rows (`lib_block_bounds_a_float_fold`), resolved to
#: COLUMN_BIT_IDENTICAL under IDENTICAL whatever the target column, so the
#: CPU column reads the value every GPU column folds at.
comptime STATS_TPB = lib_block_size_for[K_LIB_COLUMN_STATS, TARGET_COLUMN]()
comptime JACOBI_TPB = lib_block_size_for[K_LIB_JACOBI_EIGH, TARGET_COLUMN]()

#: `PINNED_GRAM_SPLITK_CHUNKS` (`core/gram_splitk.mojo:274`, DEVIATION 520),
#: `GRAM_MAX_COLS` (`:230`) and `GRAM_TPB * GRAM_MAX_CELLS_PER_THREAD`
#: (`:178`, `:290`; GRAM_TPB is `lib_block_size_for[K_LIB_GRAM_SPLITK]`,
#: 256 on every column), restated because that file imports the GPU. The
#: chunk count IS the summation split; the other two are the capacity bound
#: `gram_splitk_applies` tests.
comptime GRAM_SPLITK_CHUNKS = 128
comptime GRAM_MAX_COLS = 128
comptime GRAM_CELL_CAP = 256 * 64

#: `JACOBI_TOL` and `JACOBI_SWEEPS` (`jacobi_eigh_device.mojo:83-84`),
#: RAFT's `eigJacobi` defaults; the kernel receives `Float32(JACOBI_TOL)`.
comptime JACOBI_TOL = 1.0e-7
comptime JACOBI_SWEEPS = 15


def host_halving_sum(partials: List[Float32]) -> Float32:
    """`two_phase_halving_sum[width]` over `width = len(partials)` lane
    partials: the tree `red[t] = red[t] + red[t + step]`, `step = width/2
    .. 1`, plain float32 adds, no flush inside (the callers flush the
    total as their kernels do). The two phases of the device spelling
    combine exactly these operands in exactly this order."""
    var red = List[Float32]()
    for t in range(len(partials)):
        red.append(partials[t])
    var step = len(partials) // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return red[0]


def host_column_mean(x: List[Float32], n_rows: Int, n_cols: Int) -> List[Float32]:
    """`column_mean_kernel`, one STATS_TPB block per column."""
    var mu = List[Float32](length=n_cols, fill=Float32(0.0))
    for col in range(n_cols):
        var partials = List[Float32](length=STATS_TPB, fill=Float32(0.0))
        for t in range(STATS_TPB):
            var acc = Float32(0.0)
            var r = t
            while r < n_rows:
                acc += x[r * n_cols + col]
                r += STATS_TPB
            partials[t] = acc
        var s0 = ftz(host_halving_sum(partials))
        mu[col] = ftz(s0 / Float32(n_rows))
    return mu^


def host_gram_applies(m: Int) -> Bool:
    """`gram_splitk_applies(m, m, k)` under IDENTICAL: the capacity test
    alone (the starvation test is not consulted, DEVIATION 521)."""
    if m > GRAM_MAX_COLS:
        return False
    if m * m > GRAM_CELL_CAP:
        return False
    return True


def host_gram_splitk(
    x: List[Float32], mu: List[Float32], centered: Bool, m: Int, k: Int,
) -> List[Float32]:
    """`z[m x m] = T^T . T` over `T = X` (plain) or `T = X - 1 mu^T`
    (centered), `X` row-major `k x m`, on the split-K pair: 128 chunk
    partials, each cell one `identical_mul_add` chain r ascending, the
    partial flushed at its store, then the serial ascending chunk fold."""
    var mn = m * m
    var kc = (k + GRAM_SPLITK_CHUNKS - 1) // GRAM_SPLITK_CHUNKS
    if kc < 1:
        kc = 1
    var partials = List[Float32](
        length=GRAM_SPLITK_CHUNKS * mn, fill=Float32(0.0)
    )
    var xp = host_list_ptr(x)
    var mup = host_list_ptr(mu)
    var pp = host_list_ptr(partials)
    # Thread launch dominates tiny fits.  The threshold counts the Gram's
    # multiply-add cells, not bytes, and does not affect arithmetic order.
    var tasks = 1
    if k * mn >= 131072:
        tasks = host_predict_task_count(GRAM_SPLITK_CHUNKS)
    var chunks_per_task = host_predict_chunk(GRAM_SPLITK_CHUNKS, tasks)

    def _chunks(task: Int) {imm xp, imm mup, imm pp, imm chunks_per_task,
                            imm kc, imm k, imm m, imm mn, imm centered}:
        var c0 = task * chunks_per_task
        var c1 = c0 + chunks_per_task
        if c1 > GRAM_SPLITK_CHUNKS:
            c1 = GRAM_SPLITK_CHUNKS
        for chunk in range(c0, c1):
            var t0 = chunk * kc
            var t1 = t0 + kc
            if t1 > k:
                t1 = k
            if t0 >= t1:
                # The zero-filled partial already spells ftz(0.0).
                continue
            # The staging tile of this chunk: the centered read is the fused
            # epilogue, `ftz(ftz(x) - ftz(mu))`, DEVIATION 522.
            var rows = t1 - t0
            var tile = List[Float32](length=rows * m, fill=Float32(0.0))
            for r in range(rows):
                for j in range(m):
                    var v = xp.unsafe_load((t0 + r) * m + j)
                    if centered:
                        tile[r * m + j] = ftz(ftz(v) - ftz(mup.unsafe_load(j)))
                    else:
                        tile[r * m + j] = v
            for cell in range(mn):
                var i = cell // m
                var j = cell - i * m
                var acc = Float32(0.0)
                for r in range(rows):
                    var base = r * m
                    acc = identical_mul_add(tile[base + i], tile[base + j], acc)
                pp.unsafe_store(chunk * mn + cell, ftz(acc))
    if tasks == 1:
        _chunks(0)
    else:
        sync_parallelize(_chunks, tasks)
    var z = List[Float32](length=mn, fill=Float32(0.0))
    for cell in range(mn):
        var acc = Float32(0.0)
        comptime if PCA_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM: the same fold, chunks walked DESCENDING.
            # Wrong on purpose; see the module docstring.
            for cc in range(GRAM_SPLITK_CHUNKS):
                var c = GRAM_SPLITK_CHUNKS - 1 - cc
                acc = ftz(acc + ftz(partials[c * mn + cell]))
        else:
            for c in range(GRAM_SPLITK_CHUNKS):
                acc = ftz(acc + ftz(partials[c * mn + cell]))
        z[cell] = ftz(acc)
    return z^


def host_gemm_tn(x: List[Float32], m: Int, k: Int) -> List[Float32]:
    """`gemm_tn(z, x, ..., m, m, k)` under IDENTICAL: the plain split-K
    Gram where it applies, else the v1 profile at OP_TN (`gemm_oracle`)."""
    if host_gram_applies(m):
        var no_mu = List[Float32]()
        return host_gram_splitk(x, no_mu, False, m, k)
    return gemm_oracle(x, x, OP_TN, m, m, k)


def host_shift_columns(
    x: List[Float32], mu: List[Float32], n_rows: Int, n_cols: Int, sign: Float32,
) -> List[Float32]:
    """`shift_columns_kernel`: `ftz(ftz(x) + sign * ftz(mu))`, a copy."""
    var out = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for idx in range(n_rows * n_cols):
        var col = idx % n_cols
        var xv = ftz(x[idx])
        var mv = ftz(mu[col])
        out[idx] = ftz(xv + sign * mv)
    return out^


def host_scale_in_place(mut a: List[Float32], scale: Float32):
    """`scale_in_place_kernel`: `ftz(a * scale)` per cell."""
    for i in range(len(a)):
        a[i] = ftz(a[i] * scale)


@always_inline
def _rot_sub(c: Float32, x: Float32, s: Float32, y: Float32) -> Float32:
    """`c*x - s*y`, `jacobi_eigh_device.mojo::_rot_sub` under IDENTICAL.

    THE SABOTAGE ARM SPLITS THE FMA (added 2026-09-19, lane/linalg-public).
    `identical_mul_add(c, x, -ftz(s*y))` rounds ONCE; `ftz(c*x) - ftz(s*y)`
    rounds twice and differs in the last bits. This is where the Jacobi
    rotation's arithmetic actually lives, and it had NO negative control:
    the only arm in this file was `host_gram_splitk`'s, which the
    eigensolver reaches only when something upstream built its matrix with
    a gram product. `linalg.eigh` does not, so the `linalg-eigh` lane ran
    against a sabotage build and returned a BYTE-IDENTICAL digest -- reached
    and INERT, which reads in a column exactly like coverage. Every consumer
    of this rotation (PCA, TruncatedSVD, Nystroem, SpectralClustering, the
    GP, and the one-sided Jacobi SVD, which imports this pair) was equally
    uncovered HERE and moved only because a fold above it moved.
    """
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        return ftz(ftz(c * x) - ftz(s * y))
    return ftz(identical_mul_add(c, x, -ftz(s * y)))


@always_inline
def _rot_add(s: Float32, x: Float32, c: Float32, y: Float32) -> Float32:
    """`s*x + c*y`, `jacobi_eigh_device.mojo::_rot_add` under IDENTICAL.
    The sabotage arm splits the FMA, for `_rot_sub`'s reason."""
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        return ftz(ftz(s * x) + ftz(c * y))
    return ftz(identical_mul_add(s, x, ftz(c * y)))


def host_jacobi_rotation_cs(
    app_in: Float32, aqq_in: Float32, apq_in: Float32
) -> SIMD[DType.float32, 2]:
    """`jacobi_rotation_cs`, statement for statement."""
    var apq = ftz(apq_in)
    if apq == Float32(0.0):
        return SIMD[DType.float32, 2](Float32(1.0), Float32(0.0))
    var aqq = ftz(aqq_in)
    var app = ftz(app_in)
    var theta = ftz(ftz(aqq - app) / ftz(Float32(2.0) * apq))
    var root = ftz(
        identical_sqrt(ftz(identical_mul_add(theta, theta, Float32(1.0))))
    )
    var t: Float32
    if theta >= Float32(0.0):
        t = ftz(Float32(1.0) / ftz(theta + root))
    else:
        t = ftz(Float32(-1.0) / ftz(root - theta))
    var croot = ftz(
        identical_sqrt(ftz(identical_mul_add(t, t, Float32(1.0))))
    )
    var c = ftz(Float32(1.0) / croot)
    return SIMD[DType.float32, 2](c, ftz(t * c))


def _rotate_pair_block(
    mut a: List[Float32], n: Int, p: Int, q: Int, c: Float32, s: Float32,
):
    """`_rotate_pair_block`: the 2 x 2 block in the column-then-row order."""
    var app = ftz(a[p * n + p])
    var apq = ftz(a[p * n + q])
    a[p * n + p] = _rot_sub(c, app, s, apq)
    a[p * n + q] = _rot_add(s, app, c, apq)
    var aqp = ftz(a[q * n + p])
    var aqq = ftz(a[q * n + q])
    a[q * n + p] = _rot_sub(c, aqp, s, aqq)
    a[q * n + q] = _rot_add(s, aqp, c, aqq)
    var rpp = ftz(a[p * n + p])
    var rqp = ftz(a[q * n + p])
    a[p * n + p] = _rot_sub(c, rpp, s, rqp)
    a[q * n + p] = _rot_add(s, rpp, c, rqp)
    var rpq = ftz(a[p * n + q])
    var rqq = ftz(a[q * n + q])
    a[p * n + q] = _rot_sub(c, rpq, s, rqq)
    a[q * n + q] = _rot_add(s, rpq, c, rqq)


def _host_jacobi_fold(a: List[Float32], n: Int, off_diagonal_only: Bool) -> Float32:
    """The kernel's two folds: JACOBI_TPB lane partials over the flattened
    matrix, `acc = ftz(identical_mul_add(v, v, acc))`, then the halving
    tree. `off_diagonal_only` is the per-sweep `j > i` fold."""
    var partials = List[Float32](length=JACOBI_TPB, fill=Float32(0.0))
    for t in range(JACOBI_TPB):
        var acc = Float32(0.0)
        var e = t
        while e < n * n:
            var take = True
            if off_diagonal_only:
                var i = e // n
                var j = e - i * n
                take = j > i
            if take:
                var v = ftz(a[e])
                acc = ftz(identical_mul_add(v, v, acc))
            e += JACOBI_TPB
        partials[t] = acc
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: the same fold, lanes added SERIALLY DESCENDING
        # instead of through the halving tree. Wrong on purpose.
        #
        # ADDED 2026-09-19 (lane/linalg-public) BECAUSE THIS FOLD HAD NONE.
        # WHAT THIS ARM DOES NOT DO, measured the day it was written: it does
        # NOT move `linalg-eigh` on its fixture. This fold's only consumer is
        # `limit`, the convergence THRESHOLD, and perturbing a threshold that
        # a well-separated spectrum clears in the same sweep changes nothing
        # observable. It is kept because a MARGINAL matrix -- one that stops
        # within a rounding of the tolerance -- does change sweep count here,
        # and that path deserves a control. The arm that actually moves the
        # eigendecomposition is in `_rot_sub` / `_rot_add` below.
        var serial = Float32(0.0)
        for tt in range(JACOBI_TPB):
            serial = serial + partials[JACOBI_TPB - 1 - tt]
        return serial
    return host_halving_sum(partials)


@fieldwise_init
struct JacobiHostResult(Movable):
    """What `jacobi_eigh_kernel` leaves: the basis (eigenvector `i` in
    COLUMN `i`) and the three info slots."""

    var vectors: List[Float32]
    var converged: Bool
    var rel: Float32
    var executed: Int


def host_jacobi_eigh(
    mut a: List[Float32], n: Int, max_sweeps: Int, tol: Float32,
) -> JacobiHostResult:
    """`jacobi_eigh_kernel` replayed serially; `a` is consumed in place and
    ends with the eigenvalues on its diagonal."""
    var v = List[Float32](length=n * n, fill=Float32(0.0))
    for i in range(n):
        v[i * n + i] = Float32(1.0)

    var fro2 = _host_jacobi_fold(a, n, False)
    var limit = ftz(ftz(tol * tol) * fro2)

    var executed = 0
    var converged = False
    var last_off = Float32(0.0)

    for _sweep in range(max_sweeps):
        var off = _host_jacobi_fold(a, n, True)
        last_off = off
        if Float32(2.0) * off <= limit:
            converged = True
            break
        executed += 1

        for p in range(n):
            for q in range(p + 1, n):
                var cs = host_jacobi_rotation_cs(
                    a[p * n + p], a[q * n + q], a[p * n + q]
                )
                var c = cs[0]
                var s = cs[1]
                for k in range(n):
                    if k != p and k != q:
                        var akp = ftz(a[k * n + p])
                        var akq = ftz(a[k * n + q])
                        a[k * n + p] = _rot_sub(c, akp, s, akq)
                        a[k * n + q] = _rot_add(s, akp, c, akq)
                        var apk = ftz(a[p * n + k])
                        var aqk = ftz(a[q * n + k])
                        a[p * n + k] = _rot_sub(c, apk, s, aqk)
                        a[q * n + k] = _rot_add(s, apk, c, aqk)
                    elif k == p:
                        _rotate_pair_block(a, n, p, q, c, s)
                    var vkp = ftz(v[k * n + p])
                    var vkq = ftz(v[k * n + q])
                    v[k * n + p] = _rot_sub(c, vkp, s, vkq)
                    v[k * n + q] = _rot_add(s, vkp, c, vkq)

    var rel = Float32(0.0)
    if fro2 > Float32(0.0):
        rel = ftz(identical_sqrt(ftz(ftz(Float32(2.0) * last_off) / fro2)))
    return JacobiHostResult(v^, converged, rel, executed)


def host_sign_flip(mut v: List[Float32], n: Int):
    """`sign_flip_kernel` per column of the `n x n` basis."""
    for col in range(n):
        var biggest = Float32(0.0)
        for f in range(n):
            var m = abs(v[f * n + col])
            if m > biggest:
                biggest = m
        var first = Float32(n)
        for f in range(n):
            var fv = Float32(f)
            if abs(v[f * n + col]) == biggest:
                if fv < first:
                    first = fv
        var sign = Float32(1.0)
        for f in range(n):
            if Float32(f) == first:
                if v[f * n + col] < Float32(0.0):
                    sign = Float32(-1.0)
                else:
                    sign = Float32(1.0)
        if sign < Float32(0.0):
            for f in range(n):
                v[f * n + col] = -v[f * n + col]


@fieldwise_init
struct PCAHostResult(Movable):
    """`PCAResult`: what the fit writes back, in Float64 as the device tail
    holds it (every value is a widened float32 and narrows back exactly,
    except the singular values and the ratios, which are Float64
    arithmetic narrowed once at the store)."""

    var components: List[Float64]
    var explained_var: List[Float64]
    var explained_var_ratio: List[Float64]
    var singular_vals: List[Float64]
    var noise_var: Float64


def host_order_truncate_spectrum(
    diag: List[Float64],
    vecs: List[Float64],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
) raises -> PCAHostResult:
    """`order_truncate_spectrum` (`pca.mojo:203`) at `spectrum_count = 0`."""
    var count = n_cols
    var order = List[Int]()
    for i in range(count):
        order.append(i)
    for i in range(count):
        for j in range(i + 1, count):
            if diag[order[j]] > diag[order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t

    var total = 0.0
    for i in range(count):
        total += diag[i]

    var components = List[Float64]()
    var explained_var = List[Float64]()
    var explained_var_ratio = List[Float64]()
    var singular_vals = List[Float64]()

    for c in range(n_components):
        var src = order[c]
        var lam = diag[src]
        for f in range(n_cols):
            components.append(vecs[f * n_cols + src])
        explained_var.append(lam)
        explained_var_ratio.append(lam / total if total != 0.0 else 0.0)
        singular_vals.append(sqrt(lam * Float64(singular_scale)))

    var noise = 0.0
    if n_components < count and n_components <= singular_scale:
        for c in range(n_components, count):
            noise += diag[order[c]]
        noise /= Float64(count - n_components)

    return PCAHostResult(
        components^, explained_var^, explained_var_ratio^, singular_vals^, noise
    )


def host_pca_validate(n_rows: Int, n_cols: Int, n_components: Int) raises:
    """`pca_validate`, the four refusals in their order and words."""
    if n_cols <= 1:
        raise Error("Parameter n_cols: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if n_components <= 0:
        raise Error(
            "Parameter n_components: number of components cannot be less than one"
        )
    if n_components > n_cols:
        raise Error("n_components cannot exceed n_cols")


def host_eig_and_truncate(
    mut cov: List[Float32], n_cols: Int, n_components: Int, singular_scale: Int,
) raises -> PCAHostResult:
    """`eig_and_truncate`: the Jacobi at the device's settings, the sign
    flip, the convergence refusal in its words, the Float64 tail."""
    var jac = host_jacobi_eigh(cov, n_cols, JACOBI_SWEEPS, Float32(JACOBI_TOL))
    var vecs32 = jac.vectors.copy()
    host_sign_flip(vecs32, n_cols)
    if not jac.converged:
        raise Error(
            "the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ": ||offdiag(A)||_F / ||A||_F is still "
            + String(jac.rel)
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". cuSOLVER's syevj has the same failure mode and the same"
            " remedy, which is more sweeps. A non-symmetric covariance"
            " produces this too; see check_covariance_is_symmetric."
        )
    var diag = List[Float64]()
    for i in range(n_cols):
        diag.append(Float64(cov[i * n_cols + i]))
    var vecs = List[Float64]()
    for i in range(n_cols * n_cols):
        vecs.append(Float64(vecs32[i]))
    return host_order_truncate_spectrum(
        diag, vecs, n_cols, n_components, singular_scale
    )


@fieldwise_init
struct PCAHostFit(Movable):
    """`host_pca_fit`'s answer: the spectrum and the column means."""

    var result: PCAHostResult
    var mean: List[Float32]


def host_pca_fit(
    x: List[Float32], n_rows: Int, n_cols: Int, n_components: Int,
) raises -> PCAHostFit:
    """`pca_fit_host` without the DeviceContext."""
    host_pca_validate(n_rows, n_cols, n_components)
    var mu = host_column_mean(x, n_rows, n_cols)
    var cov: List[Float32]
    if host_gram_applies(n_cols):
        cov = host_gram_splitk(x, mu, True, n_cols, n_rows)
    else:
        var centered = host_shift_columns(x, mu, n_rows, n_cols, Float32(-1.0))
        cov = host_gemm_tn(centered, n_cols, n_rows)
    host_scale_in_place(cov, Float32(1.0) / Float32(n_rows - 1))
    var result = host_eig_and_truncate(cov, n_cols, n_components, n_rows - 1)
    return PCAHostFit(result^, mu^)


def host_tsvd_fit(
    x: List[Float32], n_rows: Int, n_cols: Int, n_components: Int,
) raises -> PCAHostResult:
    """`tsvd_fit_host` without the DeviceContext."""
    host_pca_validate(n_rows, n_cols, n_components)
    var gram = host_gemm_tn(x, n_cols, n_rows)
    return host_eig_and_truncate(gram, n_cols, n_components, 1)
