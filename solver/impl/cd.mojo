# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cuml/cpp/src/solver/cd.cuh` -- `cdFit` and `cdPredict`, coordinate descent
for Lasso / ElasticNet (cuML 26.08.00).

The control plane below is `cdFit` line for line
(`cd.cuh:115-274`); every RAFT primitive it calls is under
`solver/impl/{linalg,stats,glm,functions}/` with its own header, and the
only thing cuML never needed -- a reduction whose shape is the same on
three vendors -- is `solver/checks/profile_dot.mojo`.

THE OBJECTIVE, AND THE DOCSTRING THAT UNDERSTATES IT. `cd.cuh:77-84`
documents

    f(coef) = 1/2 ||labels - input coef||^2
            + 1/2 alpha (1 - l1_ratio) ||coef||^2 + alpha l1_ratio ||coef||_1

but the CODE (`cd.cuh:168-169`) scales both penalties by `n_rows`:

    l2_alpha = (1 - l1_ratio) * alpha * n_rows
    l1_alpha =      l1_ratio  * alpha * n_rows

so what is minimized is `n_rows` times scikit-learn's ElasticNet objective

    (1/(2n)) ||y - Xw||^2 + alpha l1_ratio ||w||_1 + (alpha/2)(1 - l1_ratio) ||w||^2

and the two libraries agree on `alpha` and `l1_ratio` exactly (scikit-learn
`_cd_fast.pyx` uses the same `l1_reg = alpha l1_ratio n`, `l2_reg = alpha
(1 - l1_ratio) n` in its coordinate update). Their docstring is off by the
factor `n`; the code is what is implemented. Where they DIFFER is the stopping
rule, the soft-threshold guard and `tol`'s default -- `solver/README.md`
has the table.

THE PER-COORDINATE STEP (`cd.cuh:189-229`), five device operations and NO
host read:

    conv.coef = coef[ci]                                       raft::copy
    residual += coef[ci] * X[:, ci]                            axpy, DEVICE alpha
    coef[ci]  = dot(X[:, ci], residual)                        gemv (cuBLAS)
    cdUpdateCoefKernel<<<1, 1>>>(coef + ci, squared + ci, conv, l1_alpha)
    residual += conv.coef * X[:, ci]      (conv.coef == -new)  axpy, DEVICE alpha

The first axpy ADDS the old coefficient's contribution back into the
residual (so the dot sees the residual WITHOUT coordinate `ci`) and the
second removes the new one; `cdUpdateCoefKernel` stores `-r` into
`conv.coef` precisely so the second axpy can read its alpha from device
memory. The host reads `ConvState` ONCE per epoch (`cd.cuh:231-232`) and
stops on

    coefMax < tol  ||  diffMax / coefMax < tol                 cd.cuh:236

`cdUpdateCoefKernel` (`cd.cuh:51-68`):

    coef = *coefLoc
    r = coef > l1_alpha ? coef - l1_alpha : (coef < -l1_alpha ? coef + l1_alpha : 0)
    squared = *squaredLoc
    r = squared > 1e-5 ? r / squared : 0
    diff = |convState.coef - r|;  diffMax = max(diffMax, diff)
    absv = |r|;                   coefMax = max(coefMax, absv)
    convState.coef = -r;  *coefLoc = r

`squared` is the column's sum of squares PLUS `l2_alpha` (`cd.cuh:173`,
`addScalar`), and the `1e-5` guard is ABSOLUTE on a quantity that scales
with `n_rows` and with the square of the data -- the `OLS_NONZERO_THRESH`
class `glm/README.md` records for `lstsqEig`. Carried as theirs, named in
`solver/NOT_IMPLEMENTED.tsv`, and the card records `cd.squared` so a column the
guard zeroes is visible.

WHAT IS REFUSED BY NAME (every one raises with the parameter's name):
`loss != SQRD_LOSS` (`cd.cuh:130`, theirs asserts too), `sample_weight`
(`cd.cuh:136-163,:240-251`, the weighted arms), `shuffle = true`
(`solver/impl/shuffle.mojo`: `std::shuffle` is not specified by the
standard, so the permutation is not a pure function of the seed), `n_cols
<= 0`, `n_rows <= 1` (theirs, `cd.cuh:128-129`), `alpha < 0` and `l1_ratio`
outside `[0, 1]` (cuML's Python layer, `elastic_net.py:199-206`), and --
DEVIATION 613, ours -- a non-finite `alpha`, a NaN `l1_ratio`, a NaN `tol`
(their guards let every NaN through).

THE CARD AND NaN (DEVIATION 612, `solver/checks/record_canon.mojo`):
every float stage is hashed through a copy whose NaNs are rewritten to the
one payload `0x7FC00000`, because a computed NaN's payload is the vendor's
(IDENTITY_PATHS row 39) and a non-finite label or an overflowing dot can
put one in `residual`, `coef` and `ConvState` (the soft threshold maps a
NaN dot to a `0` coefficient; cuML does the same and never says which NaN).

IDENTITY (DEVIATION 610). cuML's `cdFit` runs its four row-length
reductions on three fold shapes (the colNorm and the means on a RAFT
kernel chosen by SM count, the dot on cuBLAS), and the per-coordinate
branch `coef > l1_alpha` and the per-epoch stopping test both branch on
those bits, so the ITERATION COUNT is a function of the card. Under
IDENTICAL every one of them is the `gemm.fp32.v1` dot (`profile_dot.mojo`),
the two axpys are `identical_mul_add` with `ftz` on the residual, and the
update kernel flushes its quotient, its `diff` and its `|r|` -- so `coef`,
`residual`, `ConvState`, the epoch count and the intercept are a function of
the inputs alone, and the card (`cd_fit_traced`) carries each of them per
epoch: `cd.input.x/y`, `cd.l1_alpha/l2_alpha`, `cd.mu_input/mu_labels`,
`cd.colnorm`, `cd.squared`, `cd.sweepNNN.coef/resid/conv`, `cd.final.coef`,
`cd.intercept`, `cd.n_iter`.

`CdLaunch` is the SCHEDULING surface the gates turn: the axpy block size
and grid shape and the gemm lane's execution plan for the dot. None of them
can reach a bit, and `check_cd_is_launch_invariant` is what keeps that
sentence true.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined

from core.gemm import gemm_nt, gemv_n
from checks.numerics import NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from solver.checks.profile_dot import (
    profile_dot_into,
    profile_dot_workspace_floats,
)
from solver.checks.record_canon import (
    record_device_canon,
    record_scalar_f32_canon,
)
from solver.impl.functions.linear_reg import linear_reg_h
from glm.impl.preprocess import post_process_data, pre_process_data
from solver.impl.linalg.axpy import AXPY_TPB, axpy_device_alpha
from std.os import getenv
from checks.numerics import identical_mul_add
from gemm.checks.gemm_identical import (
    APPLE_LEAF_PREFETCH,
    CONTRACT_MAX_LEAVES,
    PLAN_SPLITK,
    SPLITK_FOLD_TPB,
    SPLITK_LEAF_LAUNCH_TPB,
    choose_gemm_plan,
    contract_partition,
    identical_gemm_leaf_kernel,
)
from solver.impl.linalg.norm import col_norm_l2_squared
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_NT
from solver.impl.cd_gram_rule import CD_IDN_GRAM_ON, cd_idn_gram_shape
from checks.rtf_seam import rtf_mul_add
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_NVIDIA, COLUMN_AMD
from solver.impl.shuffle import init_shuffle
from solver.impl.solvers.params import LOSS_SQRD_LOSS, loss_funct_name

#: SABOTAGE (a no-op in every build that does not name it): the soft
#: threshold's subtraction is spelled with its operands swapped and negated,
#: `-(l1_alpha - coef)` for `coef - l1_alpha` and `l1_alpha + coef` for
#: `coef + l1_alpha`. IEEE subtraction is exactly anticommutative and
#: addition exactly commutative in round-to-nearest, so this MUST move no
#: bit; `check_cd_soft_threshold_operand_order` builds with it and REPORTS.
comptime SAB_SOFT_SWAP = is_defined["MOJOLEARN_CD_SABOTAGE_SOFT_SWAP"]()

#: SABOTAGES for IDENTITY_PATHS row 39 (each a no-op unless named): the
#: `coefMax` fold is respelled as a HARDWARE `max` whose candidate is the
#: SIGNED `r` when `r` is a zero (so a -0.0 becomes a candidate, which the
#: `abs()` spelling never lets happen) and `|r|` otherwise -- bit-inert on
#: every fixture without a zero coefficient. `max(conv, cand)` (ZERO_FOLD
#: _MAX) returns the SECOND operand on Apple, so a trailing -0.0 coordinate
#: leaves -0.0 in `ConvState.coefMax`; NVIDIA/AMD's IEEE-2019 maximum
#: returns +0.0 and cannot see it. The swapped `max(cand, conv)` (ZERO_FOLD
#: _MAX_SWAPPED) returns the +0.0 seed on Apple and +0.0 on the others: a
#: spelling that is inert everywhere by accident of operand order. The
#: `check_cd_signed_zero_coefficients` gate runs both; the README has the
#: lines.
comptime SAB_ZERO_FOLD_MAX = is_defined["MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX"]()
comptime SAB_ZERO_FOLD_MAX_SWAPPED = is_defined[
    "MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX_SWAPPED"
]()

#: `cd.cuh:62`'s guard, `math_t(1e-5)`.
comptime CD_SQUARED_GUARD = Float32(1.0e-5)

#: FAST on Apple: coordinate descent in GRAM form when the data is tall
#: (n_rows >= 4 * n_cols, n_cols <= CD_GRAM_MAX_COLS). One product
#: [X ; y] [X ; y]^T gives G = X^T X and c = X^T y; every sweep then runs on
#: the host in float64 over p-sized vectors -- rho = c_j + G_jj w_j, the
#: same soft-threshold / (G_jj + l2) update and guard, c -= G[:, j] dw --
#: instead of two axpys and a dot over all n_rows per coordinate. The same
#: cyclic algorithm and convergence test on the same objective; only the
#: rounding differs. `-D MOJOLEARN_CD_FAST_GRAM_OFF` keeps the row sweeps.
comptime CD_FAST_GRAM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CD_FAST_GRAM_OFF"]()
)
comptime CD_GRAM_MAX_COLS = 256
comptime CD_GRAM_ROWS = 256
comptime CD_GRAM_CELLS = 1024


def cd_gram_partial_kernel(
    b: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    p1_in: Int32,
    n_in: Int32,
):
    """CD_FAST_GRAM: block k's float32 partial of [X ; y] [X ; y]^T over
    rows [k * CD_GRAM_ROWS, ...), one thread per cell; the host sums the
    partials in float64 so the Gram is not one 1M-long float32 chain."""
    var p1 = Int(p1_in)
    var n = Int(n_in)
    var t = Int(thread_idx.x)
    var r0 = Int(block_idx.x) * CD_GRAM_ROWS
    var r1 = r0 + CD_GRAM_ROWS
    if r1 > n:
        r1 = n
    var sh = stack_allocation[
        CD_GRAM_ROWS * 32, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var e = t
    while e < p1 * CD_GRAM_ROWS:
        var a = e // CD_GRAM_ROWS
        var i = e % CD_GRAM_ROWS
        sh[e] = b[a * n + r0 + i] if r0 + i < r1 else Float32(0)
        e += CD_GRAM_CELLS
    barrier()
    if t < p1 * p1:
        var a = t // p1
        var c = t % p1
        var acc = Float32(0)
        for i in range(r1 - r0):
            acc += sh[a * CD_GRAM_ROWS + i] * sh[c * CD_GRAM_ROWS + i]
        part[Int(block_idx.x) * p1 * p1 + t] = acc


#: lane/linear-apple3 (WIP, opt-in `-D MOJOLEARN_CD_GRAM_BLOCKS=1`): the
#: CD_FAST_GRAM product from blocks of CD_GB_ROWS rows and CD_GB_TPB
#: threads, one thread per cell of the UPPER triangle of [X ; y]^T [X ; y],
#: read straight from x (column-major) and the labels: no (p + 1) x n copy,
#: no staging in threadgroup memory, a quarter of the partials to read
#: back, and no block of 1024 threads (the M2 Pro drops a dispatch above its
#: pipeline limit with no error). The partition is fixed by n_rows alone.
comptime CD_GRAM_BLOCKS = CD_FAST_GRAM and is_defined["MOJOLEARN_CD_GRAM_BLOCKS"]()
comptime CD_GB_ROWS = 1024
comptime CD_GB_TPB = 256


def cd_gram_blocks_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    p_in: Int32,
    n_in: Int32,
):
    """Block k's float32 partial of every upper-triangle cell (a, b), b >= a,
    row by row, of [X ; y]^T [X ; y] over rows [k * CD_GB_ROWS, ...)."""
    var p = Int(p_in)
    var n = Int(n_in)
    var p1 = p + 1
    var cells = p1 * (p1 + 1) // 2
    var blk = Int(block_idx.x)
    var r0 = blk * CD_GB_ROWS
    var r1 = r0 + CD_GB_ROWS
    if r1 > n:
        r1 = n
    var c = Int(thread_idx.x)
    while c < cells:
        var a = 0
        var q = c
        while q >= p1 - a:
            q -= p1 - a
            a += 1
        var bb = a + q
        var acc = Float32(0)
        if bb < p:
            for i in range(r0, r1):
                acc += x[a * n + i] * x[bb * n + i]
        elif a < p:
            for i in range(r0, r1):
                acc += x[a * n + i] * y[i]
        else:
            for i in range(r0, r1):
                acc += y[i] * y[i]
        part[blk * cells + c] = acc
        c += CD_GB_TPB



# ===========================================================================
# lane/apple-fast-linear (2026-10-02): THE CD_FAST_GRAM PRODUCT ON THE GRID
# ===========================================================================
#
# Two build defines, FAST + Apple only (inside CD_FAST_GRAM), default on:
#
#   -D MOJOLEARN_CD_FAST_GRID_GRAM=1   [X ; y]^T [X ; y] as 32 x 32 tiles over
#       8192-row chunks (both operands staged in threadgroup memory, 4 cells
#       a thread, x_linear/enetcv_fast.mojo ef_gram_kernel's shape), the
#       chunks folded on the device (`cd_grid_gram_red_kernel`), ONE readback
#       of (p + 1)^2 floats. Cause (cd_fit_traced below, the `bmat` arm): the
#       FAST Gram first copies X and y into a (p + 1) x n buffer (884 MB on
#       Istella) and then runs `gemm_nt` -- MAX's matmul at M = N = 221,
#       K = 1,000,000, which has no split-K on Apple, so 16 output tiles'
#       worth of threadgroups carry a million-deep reduction -- or, at
#       p < 32, `cd_gram_partial_kernel`'s 1024-thread blocks over 256 rows.
#       Expected: the Gram at memory speed and no copy of X.
#
#   -D MOJOLEARN_CD_FAST_ROWMAJOR=1    (with solver/estimator.mojo)
#       X arrives ROW-MAJOR as the caller holds it (main's `cd_fit_host`
#       takes a C-order design and transposes it on the device,
#       lane/gap-nv-classical2; here that transpose is skipped and the grid
#       Gram reads row-major tiles), no centering and
#       un-centering passes over X (`pre_process_data` / `post_process_data`,
#       two read-write passes, the second only to restore a copy nobody
#       reads), no `colNorm` pass. The column means come from the same
#       chunks (`cd_grid_sums_kernel`), the Gram is centered at them while
#       the tiles are staged, and the intercept is `y_mean - mu . w` on the
#       host from the means read back beside the Gram. The sweeps are the
#       Gram sweeps already here (the same words). Refused by name where
#       CD_FAST_GRAM does not hold (IDENTICAL builds compile the old code).
#
# Default on (FAST + Apple) since the M3 A/B (istella n=1: lasso 264 -> 149
# ms with both, enet 262 -> 143 ms, r2 .2616 -> .2609). `-D
# MOJOLEARN_CD_FAST_GRID_GRAM_OFF` turns off both (the row-major arm reads
# the grid Gram's tiles); `-D MOJOLEARN_CD_FAST_ROWMAJOR_OFF` the row-major
# arm alone. The old `-D MOJOLEARN_CD_FAST_GRID_GRAM` / `_ROWMAJOR` names are
# harmless.
comptime CD_FAST_GRID_GRAM = CD_FAST_GRAM and not is_defined["MOJOLEARN_CD_FAST_GRID_GRAM_OFF"]()
comptime CD_FAST_ROWMAJOR = CD_FAST_GRID_GRAM and not is_defined["MOJOLEARN_CD_FAST_ROWMAJOR_OFF"]()
comptime CD_GG_TPB = 256
comptime CD_GG_CH = 8192
comptime CD_GG_TS = 32
comptime CD_GG_RB = 32


def cd_fast_rowmajor_serves(n_rows: Int, n_cols: Int) -> Bool:
    """CD_FAST_ROWMAJOR: whether `cd_fit_traced` takes this shape row-major
    (its guard below); False in every other build."""
    comptime if CD_FAST_ROWMAJOR:
        return n_cols <= CD_GRAM_MAX_COLS and n_rows >= 4 * n_cols
    return False


@always_inline
def _cd_gg_pair(pr: Int, nt: Int) -> Tuple[Int, Int]:
    """The pr-th upper tile pair (tj <= tk), row-major."""
    var q = pr
    var tj = 0
    while q >= nt - tj:
        q -= nt - tj
        tj += 1
    return (tj, tj + q)


@always_inline
def _cd_gg_cell(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    p: Int,
    i: Int,
    c: Int,
    rowmajor: Bool,
) -> Float32:
    """[X ; y] at (row i, column c): X column-major (c * n + i) or row-major
    (i * p + c); column p is y."""
    if c < p:
        if rowmajor:
            return x[i * p + c]
        return x[c * n + i]
    return y[i]


def cd_grid_sums_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    part_s: MutPointer[Float32, MutAnyOrigin],
    p_in: Int32,
    n_in: Int32,
    rowmajor_in: Int32,
):
    """Chunk block_idx.x: the column sums of [X ; y] over its rows, a thread
    per column (MOJOLEARN_CD_FAST_ROWMAJOR's means)."""
    var p = Int(p_in)
    var m = p + 1
    var n = Int(n_in)
    var rowmajor = rowmajor_in != 0
    var ch = Int(block_idx.x)
    var lo = ch * CD_GG_CH
    var cnt = n - lo
    if cnt > CD_GG_CH:
        cnt = CD_GG_CH
    var c = Int(thread_idx.x)
    while c < m:
        var acc = Float32(0)
        for r in range(cnt):
            acc += _cd_gg_cell(x, y, n, p, lo + r, c, rowmajor)
        part_s[ch * m + c] = acc
        c += CD_GG_TPB


def cd_grid_means_kernel(
    part_s: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    nch_in: Int32,
    n_in: Int32,
):
    """Thread c: the mean of column c of [X ; y] from the chunk sums."""
    var m = Int(m_in)
    var c = Int(block_idx.x) * CD_GG_TPB + Int(thread_idx.x)
    if c < m:
        var s = Float32(0)
        for ch in range(Int(nch_in)):
            s += part_s[ch * m + c]
        mu[c] = s / Float32(Int(n_in))


def cd_grid_gram_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    p_in: Int32,
    n_in: Int32,
    rowmajor_in: Int32,
    npairs_in: Int32,
    nt_in: Int32,
):
    """Block b: tile pair b % npairs of chunk b // npairs of
    ([X ; y] - mu)^T ([X ; y] - mu) over the chunk's rows; 4 cells a thread.
    `mu` is zero where the columns are centered already."""
    var p = Int(p_in)
    var m = p + 1
    var n = Int(n_in)
    var rowmajor = rowmajor_in != 0
    var npairs = Int(npairs_in)
    var b = Int(block_idx.x)
    var pr = b % npairs
    var ch = b // npairs
    var tjk = _cd_gg_pair(pr, Int(nt_in))
    var j0 = tjk[0] * CD_GG_TS
    var k0 = tjk[1] * CD_GG_TS
    var lo = ch * CD_GG_CH
    var cnt = n - lo
    if cnt > CD_GG_CH:
        cnt = CD_GG_CH
    var tid = Int(thread_idx.x)
    var sa = stack_allocation[
        CD_GG_RB * CD_GG_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sb = stack_allocation[
        CD_GG_RB * CD_GG_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var r = tid // 8
    var c0 = (tid % 8) * 4
    var acc = SIMD[DType.float32, 4](0)
    var rb = 0
    while rb < cnt:
        comptime for u in range((CD_GG_RB * CD_GG_TS) // CD_GG_TPB):
            var e = tid + u * CD_GG_TPB
            # row-major X: neighbouring threads read neighbouring columns of
            # one row; column-major X: neighbouring rows of one column
            var rr = e // CD_GG_TS if rowmajor else e % CD_GG_RB
            var cc = e % CD_GG_TS if rowmajor else e // CD_GG_RB
            var row = rb + rr
            var va = Float32(0)
            var vb = Float32(0)
            if row < cnt:
                var i = lo + row
                var ja = j0 + cc
                var kb = k0 + cc
                if ja < m:
                    va = _cd_gg_cell(x, y, n, p, i, ja, rowmajor) - mu[ja]
                if kb < m:
                    vb = _cd_gg_cell(x, y, n, p, i, kb, rowmajor) - mu[kb]
            sa[rr * CD_GG_TS + cc] = va
            sb[rr * CD_GG_TS + cc] = vb
        barrier()
        comptime for rr in range(CD_GG_RB):
            var a = sa[rr * CD_GG_TS + r]
            var bv = (sb + rr * CD_GG_TS + c0).load[width=4]()
            acc += a * bv
        barrier()
        rb += CD_GG_RB
    var o = b * (CD_GG_TS * CD_GG_TS) + r * CD_GG_TS + c0
    comptime for e in range(4):
        part[o + e] = acc[e]


def cd_grid_gram_red_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    gram: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    nch_in: Int32,
    npairs_in: Int32,
    nt_in: Int32,
):
    """Thread (pair, cell): the Gram cell summed over the chunks, both
    triangles of the m x m result."""
    var m = Int(m_in)
    var npairs = Int(npairs_in)
    comptime TT = CD_GG_TS * CD_GG_TS
    var t = Int(block_idx.x) * CD_GG_TPB + Int(thread_idx.x)
    if t < npairs * TT:
        var pr = t // TT
        var cell = t - pr * TT
        var tjk = _cd_gg_pair(pr, Int(nt_in))
        var j = tjk[0] * CD_GG_TS + cell // CD_GG_TS
        var k = tjk[1] * CD_GG_TS + cell % CD_GG_TS
        if j < m and k < m:
            var s = Float32(0)
            for ch in range(Int(nch_in)):
                s += part[(ch * npairs + pr) * TT + cell]
            gram[j * m + k] = s
            gram[k * m + j] = s


# ===========================================================================
# lane/linear-apple2: THE IDENTICAL SWEEP IN THREE LAUNCHES PER COORDINATE
# ===========================================================================
#
# Apple only (APPLE_LEAF_PREFETCH, IDENTICAL). A coordinate of the device
# sweep was six launches: remember, axpy, the profile dot's PLAN_SPLITK leaf
# and fold, update, axpy. They regroup into three, with every stored word
# computed by the same expression from the same words:
#
#   cd_axpy_pair_kernel   one thread per residual row: the PREVIOUS
#                         coordinate's closing axpy (alpha conv[0]) and this
#                         coordinate's opening axpy (alpha coef[ci]), each
#                         `axpy_device_alpha_kernel`'s
#                         `ftz(identical_mul_add(ftz(a), ftz(x), ftz(y)))`,
#                         in that order, one store;
#   identical_gemm_leaf_kernel   the profile dot's leaves, unchanged
#                         (the same launch `identical_gemm_with_plan` makes);
#   cd_fold_update_kernel one block: thread 0 remembers coef[ci] into
#                         conv[0] (after the pair kernel read the previous
#                         delta there), the block folds the P partials with
#                         `identical_gemm_fold_kernel`'s tree (cell 0,
#                         stride P) into coef[ci], and thread 0 runs
#                         `cd_update_coef`.
#
# Round 1's two-launch form (06ef7f558, reverted) put the axpys inside the
# leaf chains and lengthened them; here the axpys stay a coalesced
# elementwise pass. The last coordinate's closing axpy stays its own launch.
# -D MOJOLEARN_CD_THREE_LAUNCH_OFF=1 restores the six-launch sweep.

comptime CD_THREE_LAUNCH = (
    APPLE_LEAF_PREFETCH
    and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_CD_THREE_LAUNCH_OFF"]()
)


def cd_axpy_pair_kernel(
    residual: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    coef: MutPointer[Float32, MutAnyOrigin],
    conv: MutPointer[Float32, MutAnyOrigin],
    ci_in: Int32,
    prev_ci_in: Int32,
    n_in: Int32,
):
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var prev = Int(prev_ci_in)
    var r = residual.unsafe_load(i)
    if prev >= 0:
        var a0 = ftz(conv.unsafe_load(0))
        r = ftz(identical_mul_add(a0, ftz(x.unsafe_load(prev * n + i)), ftz(r)))
    var a1 = ftz(coef.unsafe_load(Int(ci_in)))
    r = ftz(identical_mul_add(a1, ftz(x.unsafe_load(Int(ci_in) * n + i)), ftz(r)))
    residual.unsafe_store(i, r)


comptime CD_STEP_TPB = 256
comptime CD_STEP_MAX_BLOCKS = 64
comptime CD_TWO_STEP = (
    CD_THREE_LAUNCH
    and not SAB_SOFT_SWAP
    and not SAB_ZERO_FOLD_MAX
    and not SAB_ZERO_FOLD_MAX_SWAPPED
    and not is_defined["MOJOLEARN_CD_TWO_STEP_OFF"]()
)
"""lane/linear-apple2: TWO launches per coordinate (Apple, IDENTICAL, on by
default; -D MOJOLEARN_CD_TWO_STEP_OFF=1 returns to three). The fold and the
update of coordinate `prev` move into the NEXT coordinate's axpy launch,
`cd_step_kernel`: every block folds the P partials itself (the same tree)
and runs the update body on its own copy, so every block holds the same -r
for the closing axpy without waiting for another; block 0 alone stores it.
Two buffers keep every read race-free: the coefficients read this epoch
(coef_in) are not the ones written (coef_out), and the running maxima
alternate between two conv buffers per update. The epoch ends with a step
that only updates the last coordinate and applies its closing axpy."""


def cd_step_kernel(
    residual: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Float32, MutAnyOrigin],
    coef_in: MutPointer[Float32, MutAnyOrigin],
    coef_out: MutPointer[Float32, MutAnyOrigin],
    squared: MutPointer[Float32, MutAnyOrigin],
    conv_in: MutPointer[Float32, MutAnyOrigin],
    conv_out: MutPointer[Float32, MutAnyOrigin],
    prev_in: Int32,
    ci_in: Int32,
    n_in: Int32,
    p_in: Int32,
    l1_alpha: Float32,
):
    var tid = Int(thread_idx.x)
    var nth = Int(block_dim.x)
    var bid = Int(block_idx.x)
    var n = Int(n_in)
    var prev = Int(prev_in)
    var ci = Int(ci_in)
    var p_count = Int(p_in)
    var buf = stack_allocation[
        2 * CONTRACT_MAX_LEAVES + 1,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var a0 = Float32(0.0)
    if prev >= 0:
        # identical_gemm_fold_kernel's tree for cell 0, stride P
        var cur = 0
        var nxt = CONTRACT_MAX_LEAVES
        var q0 = tid
        while q0 < p_count:
            buf[unsafe_offset = cur + q0] = ws.unsafe_load(q0)
            q0 += nth
        barrier()
        var w = p_count
        while w > 1:
            var pairs = w // 2
            var q = tid
            while q < pairs:
                buf[unsafe_offset = nxt + q] = ftz(
                    ftz(buf[unsafe_offset = cur + 2 * q])
                    + ftz(buf[unsafe_offset = cur + 2 * q + 1])
                )
                q += nth
            var w_next = pairs
            if w % 2 == 1:
                if tid == 0:
                    buf[unsafe_offset = nxt + pairs] = buf[unsafe_offset = cur + w - 1]
                w_next = pairs + 1
            barrier()
            var swap = cur
            cur = nxt
            nxt = swap
            w = w_next
        if tid == 0:
            var root = Float32(0.0)
            if p_count > 0:
                root = buf[unsafe_offset=cur]
            # cd_update_coef's body on the stored dot ftz(root), the
            # remembered coefficient coef_in[prev] and the maxima in conv_in
            var c = ftz(ftz(root))
            var r: Float32
            if c > l1_alpha:
                r = c - l1_alpha
            elif c < -l1_alpha:
                r = c + l1_alpha
            else:
                r = Float32(0.0)
            var sq = ftz(squared.unsafe_load(prev))
            if sq > CD_SQUARED_GUARD:
                r = r / sq
            else:
                r = Float32(0.0)
            r = ftz(r)
            var diff = ftz(abs(ftz(coef_in.unsafe_load(prev)) - r))
            var dmax = conv_in.unsafe_load(2)
            if dmax < diff:
                dmax = diff
            var absv = abs(r)
            var cmax = conv_in.unsafe_load(1)
            if cmax < absv:
                cmax = absv
            buf[unsafe_offset = 2 * CONTRACT_MAX_LEAVES] = -r
            if bid == 0:
                coef_out.unsafe_store(prev, r)
                conv_out.unsafe_store(0, -r)
                conv_out.unsafe_store(1, cmax)
                conv_out.unsafe_store(2, dmax)
        barrier()
        a0 = ftz(buf[unsafe_offset = 2 * CONTRACT_MAX_LEAVES])
    var a1 = Float32(0.0)
    if ci >= 0:
        a1 = ftz(coef_in.unsafe_load(ci))
    var i = bid * nth + tid
    var stride = Int(grid_dim.x) * nth
    while i < n:
        var r = residual.unsafe_load(i)
        if prev >= 0:
            r = ftz(identical_mul_add(a0, ftz(x.unsafe_load(prev * n + i)), ftz(r)))
        if ci >= 0:
            r = ftz(identical_mul_add(a1, ftz(x.unsafe_load(ci * n + i)), ftz(r)))
        residual.unsafe_store(i, r)
        i += stride


# ===========================================================================
# lane/gap-nv-classical2: ONE LAUNCH PER COORDINATE ON NVIDIA AND AMD
# ===========================================================================
#
# Off Apple the sweep ran six launches per coordinate (remember, axpy, the
# profile dot's split leaf and fold, update, axpy), two of them one thread,
# and read the 4 MB residual and the column three times. `cd_fused_step_kernel`
# is `cd_step_kernel` (the fold and update of `prev`, the closing axpy of
# `prev` and the opening axpy of `ci`) with the dot's LEAVES folded into the
# same pass: each block owns CD_FUSED_LEAVES consecutive leaves of the
# contract partition and walks them in windows of CD_FUSED_STEPS rows. The
# window's rows are updated coalesced (the same two
# `ftz(identical_mul_add(...))` per row, one store), the updated residual
# and the flushed column are staged in threadgroup memory, and thread `t`
# then runs leaf `t`'s chain over the window: `rtf_mul_add(ftz(x), ftz(r),
# acc)` in ascending row order, exactly `identical_gemm_leaf_kernel`'s chain
# on the same words. The leaf partials go to the other half of a two-buffer
# workspace, which the NEXT coordinate's launch folds with the same tree.
# Every stored word is the same expression of the same words, so no bit
# moves. -D MOJOLEARN_CD_FUSED_OFF=1 restores the six-launch sweep.
comptime CD_FUSED_LEAVES = 4
comptime CD_FUSED_STEPS = 128
comptime CD_FUSED_TPB = 128
comptime CD_FUSED_PER_THREAD = CD_FUSED_LEAVES * CD_FUSED_STEPS // CD_FUSED_TPB
comptime CD_FUSED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and not SAB_SOFT_SWAP
    and not SAB_ZERO_FOLD_MAX
    and not SAB_ZERO_FOLD_MAX_SWAPPED
    and not is_defined["MOJOLEARN_CD_FUSED_OFF"]()
)


def cd_fused_step_kernel(
    residual: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    ws_in: MutPointer[Float32, MutAnyOrigin],
    ws_out: MutPointer[Float32, MutAnyOrigin],
    coef_in: MutPointer[Float32, MutAnyOrigin],
    coef_out: MutPointer[Float32, MutAnyOrigin],
    squared: MutPointer[Float32, MutAnyOrigin],
    conv_in: MutPointer[Float32, MutAnyOrigin],
    conv_out: MutPointer[Float32, MutAnyOrigin],
    prev_in: Int32,
    ci_in: Int32,
    n_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    l1_alpha: Float32,
):
    var tid = Int(thread_idx.x)
    var nth = Int(block_dim.x)
    var bid = Int(block_idx.x)
    var n = Int(n_in)
    var prev = Int(prev_in)
    var ci = Int(ci_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var buf = stack_allocation[
        2 * CONTRACT_MAX_LEAVES + 1,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var a0 = Float32(0.0)
    if prev >= 0:
        # cd_step_kernel's fold of `prev`'s P partials and its update body
        var cur = 0
        var nxt = CONTRACT_MAX_LEAVES
        var q0 = tid
        while q0 < p_count:
            buf[unsafe_offset = cur + q0] = ws_in.unsafe_load(q0)
            q0 += nth
        barrier()
        var w = p_count
        while w > 1:
            var pairs = w // 2
            var q = tid
            while q < pairs:
                buf[unsafe_offset = nxt + q] = ftz(
                    ftz(buf[unsafe_offset = cur + 2 * q])
                    + ftz(buf[unsafe_offset = cur + 2 * q + 1])
                )
                q += nth
            var w_next = pairs
            if w % 2 == 1:
                if tid == 0:
                    buf[unsafe_offset = nxt + pairs] = buf[unsafe_offset = cur + w - 1]
                w_next = pairs + 1
            barrier()
            var swap = cur
            cur = nxt
            nxt = swap
            w = w_next
        if tid == 0:
            var root = Float32(0.0)
            if p_count > 0:
                root = buf[unsafe_offset=cur]
            var c = ftz(ftz(root))
            var r: Float32
            if c > l1_alpha:
                r = c - l1_alpha
            elif c < -l1_alpha:
                r = c + l1_alpha
            else:
                r = Float32(0.0)
            var sq = ftz(squared.unsafe_load(prev))
            if sq > CD_SQUARED_GUARD:
                r = r / sq
            else:
                r = Float32(0.0)
            r = ftz(r)
            var diff = ftz(abs(ftz(coef_in.unsafe_load(prev)) - r))
            var dmax = conv_in.unsafe_load(2)
            if dmax < diff:
                dmax = diff
            var absv = abs(r)
            var cmax = conv_in.unsafe_load(1)
            if cmax < absv:
                cmax = absv
            buf[unsafe_offset = 2 * CONTRACT_MAX_LEAVES] = -r
            if bid == 0:
                coef_out.unsafe_store(prev, r)
                conv_out.unsafe_store(0, -r)
                conv_out.unsafe_store(1, cmax)
                conv_out.unsafe_store(2, dmax)
        barrier()
        a0 = ftz(buf[unsafe_offset = 2 * CONTRACT_MAX_LEAVES])
    if ci < 0:
        # the epoch's last step: `prev`'s closing axpy alone
        var i = bid * nth + tid
        var stride = Int(grid_dim.x) * nth
        while i < n:
            var r = residual.unsafe_load(i)
            r = ftz(identical_mul_add(a0, ftz(x.unsafe_load(prev * n + i)), ftz(r)))
            residual.unsafe_store(i, r)
            i += stride
        return
    var a1 = ftz(coef_in.unsafe_load(ci))
    comptime SP = CD_FUSED_STEPS + 1
    var xs = stack_allocation[
        CD_FUSED_LEAVES * SP, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var rs = stack_allocation[
        CD_FUSED_LEAVES * SP, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var g0 = bid * CD_FUSED_LEAVES
    var my_leaf = g0 + tid
    var mine = tid < CD_FUSED_LEAVES and my_leaf < p_count
    var my_begin = my_leaf * leaf
    var my_end = my_begin + leaf
    if my_end > n:
        my_end = n
    var acc = Float32(0.0)
    # the window's operands, loaded one window ahead of their use
    var rv = InlineArray[Float32, CD_FUSED_PER_THREAD](fill=Float32(0.0))
    var pv = InlineArray[Float32, CD_FUSED_PER_THREAD](fill=Float32(0.0))
    var cv = InlineArray[Float32, CD_FUSED_PER_THREAD](fill=Float32(0.0))
    var w = 0
    comptime for u in range(CD_FUSED_PER_THREAD):
        var e = tid + u * CD_FUSED_TPB
        var lj = e // CD_FUSED_STEPS
        var s = e - lj * CD_FUSED_STEPS
        var lf = g0 + lj
        var i = lf * leaf + s
        if lf < p_count and s < leaf and i < n:
            rv[u] = residual.unsafe_load(i)
            if prev >= 0:
                pv[u] = x.unsafe_load(prev * n + i)
            cv[u] = x.unsafe_load(ci * n + i)
    while w < leaf:
        comptime for u in range(CD_FUSED_PER_THREAD):
            var e = tid + u * CD_FUSED_TPB
            var lj = e // CD_FUSED_STEPS
            var s = e - lj * CD_FUSED_STEPS
            var lf = g0 + lj
            var i = lf * leaf + w + s
            if lf < p_count and w + s < leaf and i < n:
                var r = rv[u]
                if prev >= 0:
                    r = ftz(identical_mul_add(a0, ftz(pv[u]), ftz(r)))
                var xv = ftz(cv[u])
                r = ftz(identical_mul_add(a1, xv, ftz(r)))
                residual.unsafe_store(i, r)
                xs[unsafe_offset = lj * SP + s] = xv
                rs[unsafe_offset = lj * SP + s] = r
        barrier()
        var wn = w + CD_FUSED_STEPS
        if wn < leaf:
            comptime for u in range(CD_FUSED_PER_THREAD):
                var e = tid + u * CD_FUSED_TPB
                var lj = e // CD_FUSED_STEPS
                var s = e - lj * CD_FUSED_STEPS
                var lf = g0 + lj
                var i = lf * leaf + wn + s
                if lf < p_count and wn + s < leaf and i < n:
                    rv[u] = residual.unsafe_load(i)
                    if prev >= 0:
                        pv[u] = x.unsafe_load(prev * n + i)
                    cv[u] = x.unsafe_load(ci * n + i)
        if mine:
            var s_end = my_end - (my_begin + w)
            if s_end > CD_FUSED_STEPS:
                s_end = CD_FUSED_STEPS
            var base = tid * SP
            for s in range(s_end):
                acc = rtf_mul_add(
                    ftz(xs[unsafe_offset = base + s]),
                    ftz(rs[unsafe_offset = base + s]),
                    acc,
                )
        barrier()
        w = wn
    if mine:
        ws_out.unsafe_store(my_leaf, ftz(acc))


def cd_fold_update_kernel(
    coef: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Float32, MutAnyOrigin],
    squared: MutPointer[Float32, MutAnyOrigin],
    conv: MutPointer[Float32, MutAnyOrigin],
    ci_in: Int32,
    p_in: Int32,
    l1_alpha: Float32,
):
    """Remember, `identical_gemm_fold_kernel`'s non-sabotage body for cell 0
    at stride P (P partials into threadgroup memory; level by level node q =
    ftz(ftz(child 2q) + ftz(child 2q + 1)); an odd tail carried bit for bit;
    the root stored ftz'd by thread 0), then thread 0 runs the update."""
    var ci = Int(ci_in)
    var p_count = Int(p_in)
    var tid = Int(thread_idx.x)
    var nth = Int(block_dim.x)
    if tid == 0:
        conv.unsafe_store(0, coef.unsafe_load(ci))
    var buf = stack_allocation[
        2 * CONTRACT_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var cur = 0
    var nxt = CONTRACT_MAX_LEAVES
    var q0 = tid
    while q0 < p_count:
        buf[unsafe_offset = cur + q0] = ws.unsafe_load(q0)
        q0 += nth
    barrier()
    var w = p_count
    while w > 1:
        var pairs = w // 2
        var q = tid
        while q < pairs:
            buf[unsafe_offset = nxt + q] = ftz(
                ftz(buf[unsafe_offset = cur + 2 * q])
                + ftz(buf[unsafe_offset = cur + 2 * q + 1])
            )
            q += nth
        var w_next = pairs
        if w % 2 == 1:
            if tid == 0:
                buf[unsafe_offset = nxt + pairs] = buf[unsafe_offset = cur + w - 1]
            w_next = pairs + 1
        barrier()
        var swap = cur
        cur = nxt
        nxt = swap
        w = w_next
    if tid == 0:
        var root = Float32(0.0)
        if p_count > 0:
            root = buf[unsafe_offset=cur]
        coef.unsafe_store(ci, ftz(root))
        cd_update_coef(coef, ci_in, squared, conv, l1_alpha)


# ===========================================================================
# lane/fam-linear (2026-10-04): IDENTICAL GRAM SWEEPS, EVERY VENDOR
# ===========================================================================
#
# The row sweeps read the residual and a column of X once per coordinate:
# n_cols launches over n_rows rows per epoch, and one host read per epoch.
# CD_IDN_GRAM forms the Gram once, G = X^T X and q = X^T y through the
# profile GEMM (`identical_gemm_into`, OP_NT over the column-major design:
# the same contract partition and fold tree on every vendor), and then runs
# the SAME cyclic algorithm on the n_cols x n_cols Gram, where q stays
# X^T residual:
#
#     old   = ftz(coef[j])
#     c     = ftz(fma(ftz(G[j, j]), old, q[j]))      x_j . (residual + old x_j)
#     r     = SoftThreshold(c, l1_alpha) / squared[j]   (`cd_update_coef`'s
#             statements, guard included), r = ftz(r)
#     delta = ftz(old - r)
#     q[k]  = ftz(fma(delta, ftz(G[k, j]), q[k]))    every k, one thread each
#
# with `cd_update_coef`'s strict-`<` maxima and the same stopping test
# (`coef_max < tol or diff_max / coef_max < tol`), decided ON THE DEVICE:
# one launch runs up to CD_IDN_GRAM_EPOCHS epochs and freezes at the epoch
# that converged; the host reads four words per launch. The sweep is one
# block because the algorithm is sequential in j; inside a coordinate every
# q[k] moves in parallel (thread k owns q[k] in a register, the shared page
# carries delta and the two maxima between barriers).
#
# BITS CHANGE (the rounding of each coordinate's dot differs from the row
# form) on NVIDIA, AMD, Apple and the host column together:
# `solver/host/cd_oracle.mojo::cd_oracle_fit(gram=True)` is the same
# arithmetic, and `solver/impl/cd_gram_rule.mojo` is the one shape rule both
# read. A traced fit (MOJOLEARN_IDENTITY_TRACE) and a fit that wants the
# residual keep the row sweeps on both columns, so the stage card and its
# checks are unchanged. `-D MOJOLEARN_CD_IDN_GRAM_OFF` restores the row
# sweeps (pass it to the host build too).
comptime CD_IDN_GRAM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and CD_IDN_GRAM_ON
    and not SAB_SOFT_SWAP
    and not SAB_ZERO_FOLD_MAX
    and not SAB_ZERO_FOLD_MAX_SWAPPED
)
comptime CD_IDN_GRAM_TPB = 256
comptime CD_IDN_GRAM_EPOCHS = 16


def cd_idn_gram_sweep_kernel(
    g: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    coef: MutPointer[Float32, MutAnyOrigin],
    squared: MutPointer[Float32, MutAnyOrigin],
    state: MutPointer[Float32, MutAnyOrigin],
    p_in: Int32,
    epochs_in: Int32,
    l1_alpha: Float32,
    tol: Float32,
):
    """Up to `epochs_in` Gram sweeps in one block of CD_IDN_GRAM_TPB threads
    (n_cols <= CD_IDN_GRAM_TPB). `state`: [0] 1 once converged, [1] the
    epochs run so far (UInt32 bits), [2] coef_max and [3] diff_max of the last epoch run.
    After convergence the remaining epochs do no work (the barriers stay
    unconditional so every thread takes the same path)."""
    var tid = Int(thread_idx.x)
    var p = Int(p_in)
    # sh: 0 delta, 1 coef_max, 2 diff_max, 3 done
    var sh = stack_allocation[
        4, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var qk = Float32(0.0)
    if tid < p:
        qk = ftz(q.unsafe_load(tid))
    if tid == 0:
        sh[unsafe_offset = 0] = Float32(0.0)
        sh[unsafe_offset = 1] = Float32(0.0)
        sh[unsafe_offset = 2] = Float32(0.0)
        sh[unsafe_offset = 3] = state.unsafe_load(0)
    barrier()
    var e = 0
    while e < Int(epochs_in):
        var live = sh[unsafe_offset = 3] == Float32(0.0)
        barrier()
        if live and tid == 0:
            sh[unsafe_offset = 1] = Float32(0.0)
            sh[unsafe_offset = 2] = Float32(0.0)
        for j in range(p):
            if live and tid == j:
                var old = ftz(coef.unsafe_load(j))
                var c = ftz(identical_mul_add(ftz(g.unsafe_load(j * p + j)), old, qk))
                var r: Float32
                if c > l1_alpha:
                    r = c - l1_alpha
                elif c < -l1_alpha:
                    r = c + l1_alpha
                else:
                    r = Float32(0.0)
                var sq = ftz(squared.unsafe_load(j))
                if sq > CD_SQUARED_GUARD:
                    r = r / sq
                else:
                    r = Float32(0.0)
                r = ftz(r)
                var diff = ftz(abs(old - r))
                if sh[unsafe_offset = 2] < diff:
                    sh[unsafe_offset = 2] = diff
                var absv = abs(r)
                if sh[unsafe_offset = 1] < absv:
                    sh[unsafe_offset = 1] = absv
                coef.unsafe_store(j, r)
                sh[unsafe_offset = 0] = ftz(old - r)
            barrier()
            if live and tid < p:
                qk = ftz(
                    identical_mul_add(
                        sh[unsafe_offset = 0], ftz(g.unsafe_load(tid * p + j)), qk
                    )
                )
            barrier()
        if live and tid == 0:
            var cmax = sh[unsafe_offset = 1]
            var dmax = sh[unsafe_offset = 2]
            # lane/review-fixes: the epoch count is an integer (UInt32 bits
            # in the float32 slot); a Float32 counter stops at 2^24.
            state.unsafe_store(
                1, bitcast[DType.float32](bitcast[DType.uint32](state.unsafe_load(1)) + UInt32(1))
            )
            state.unsafe_store(2, cmax)
            state.unsafe_store(3, dmax)
            if cmax < tol or (dmax / cmax) < tol:
                sh[unsafe_offset = 3] = Float32(1.0)
                state.unsafe_store(0, Float32(1.0))
        barrier()
        e += 1
    if tid < p:
        q.unsafe_store(tid, qk)


@fieldwise_init
struct CdLaunch(Copyable, Movable, ImplicitlyCopyable):
    """SCHEDULING knobs. `dot_plan < 0` lets the gemm lane's dispatcher pick
    (production); the gates name plans. `axpy_tpb` must be a multiple of the
    lane width a backend needs for a full block; 256 and 64 are the gates'."""

    var axpy_tpb: Int
    var axpy_two_d_grid: Bool
    var dot_plan: Int

    @staticmethod
    def default() -> Self:
        return Self(AXPY_TPB, False, -1)


def cd_remember_coef_kernel(
    conv: MutPointer[Float32, MutAnyOrigin],
    coef: MutPointer[Float32, MutAnyOrigin],
    ci_in: Int32,
):
    """`raft::copy(&(convStateLoc->coef), coef_loc, 1, stream)`, `cd.cuh:198`:
    one float moved on the device, no arithmetic."""
    conv.unsafe_store(0, coef.unsafe_load(Int(ci_in)))


def cd_update_coef_kernel(
    coef: MutPointer[Float32, MutAnyOrigin],
    ci_in: Int32,
    squared: MutPointer[Float32, MutAnyOrigin],
    conv: MutPointer[Float32, MutAnyOrigin],
    l1_alpha: Float32,
):
    """The launch; the body is `cd_update_coef` (lane/linear-apple2: the
    three-launch coordinate calls the body from its fold kernel; a kernel
    called from another kernel crashes the Metal AIR pass)."""
    cd_update_coef(coef, ci_in, squared, conv, l1_alpha)


@always_inline
def cd_update_coef(
    coef: MutPointer[Float32, MutAnyOrigin],
    ci_in: Int32,
    squared: MutPointer[Float32, MutAnyOrigin],
    conv: MutPointer[Float32, MutAnyOrigin],
    l1_alpha: Float32,
):
    """`cdUpdateCoefKernel`, `cd.cuh:51-68`, launched `<<<1, 1>>>`.
    `conv` is `ConvState{coef, coefMax, diffMax}` as three floats."""
    var ci = Int(ci_in)
    # Row 10: operands flushed on load (bit-inert on an FTZ backend, and the
    # caller's warm-start `coef` is the one input this kernel reads raw).
    var c = ftz(coef.unsafe_load(ci))
    var r: Float32
    comptime if SAB_SOFT_SWAP:
        if c > l1_alpha:
            r = -(l1_alpha - c)
        elif c < -l1_alpha:
            r = l1_alpha + c
        else:
            r = Float32(0.0)
    else:
        if c > l1_alpha:
            r = c - l1_alpha
        elif c < -l1_alpha:
            r = c + l1_alpha
        else:
            r = Float32(0.0)
    var sq = ftz(squared.unsafe_load(ci))
    if sq > CD_SQUARED_GUARD:
        r = r / sq
    else:
        r = Float32(0.0)
    # Row 10: the quotient is a seam (the next axpy's alpha, the card, the
    # host's |coef| test), so it gets its own flushed local.
    r = ftz(r)
    # IDENTITY_PATHS row 39 (signed zero, NaN). `r` CAN be -0.0 here: a
    # negative quotient below the normal floor flushes to a SIGNED zero
    # (Apple's hardware and `ftz` agree, row 10), and it is stored to
    # `coef` and negated into `conv[0]` as the IEEE bits it is -- negation
    # and the flush are vendor-invariant, `check_cd_signed_zero_
    # coefficients` plants both zeros and compares the card bit for bit.
    # The two folds below NEVER see a -0.0 or a NaN as a candidate that
    # could win: `abs()` clears the sign bit, so both candidates are +0.0
    # or positive; the seed is the +0.0 `enqueue_memset` wrote; the compare
    # is a STRICT `<`, so on a +0.0/+0.0 tie the SEED (the earlier value)
    # survives by position, not by a vendor's `max`, and a NaN candidate
    # (`x < NaN` is false) never enters. No hardware max/min is spelled.
    var diff = ftz(abs(ftz(conv.unsafe_load(0)) - r))
    if conv.unsafe_load(2) < diff:
        conv.unsafe_store(2, diff)
    var absv = abs(r)
    comptime if SAB_ZERO_FOLD_MAX or SAB_ZERO_FOLD_MAX_SWAPPED:
        var cand = absv
        if r == Float32(0.0):
            cand = r  # the SIGNED zero becomes a candidate
        comptime if SAB_ZERO_FOLD_MAX:
            conv.unsafe_store(1, max(conv.unsafe_load(1), cand))
        else:
            conv.unsafe_store(1, max(cand, conv.unsafe_load(1)))
    else:
        if conv.unsafe_load(1) < absv:
            conv.unsafe_store(1, absv)
    conv.unsafe_store(0, -r)
    coef.unsafe_store(ci, r)


def add_scalar_cols_kernel(
    v: MutPointer[Float32, MutAnyOrigin], n_in: Int32, s: Float32
):
    """`raft::linalg::addScalar(squared, squared, l2_alpha, n_cols)`,
    `cd.cuh:173`."""
    var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if i < Int(n_in):
        v.unsafe_store(i, ftz(v.unsafe_load(i) + s))


def cd_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut labels: DeviceBuffer[DType.float32],
    mut coef: DeviceBuffer[DType.float32],
    fit_intercept: Bool,
    epochs: Int,
    loss: Int,
    alpha: Float32,
    l1_ratio: Float32,
    shuffle: Bool,
    tol: Float32,
    has_sample_weight: Bool = False,
) raises -> Tuple[Int, Float32]:
    """`cdFit(handle, input, n_rows, n_cols, labels, coef, intercept,
    fit_intercept, epochs, loss, alpha, l1_ratio, shuffle, tol,
    sample_weight)`. Returns `(n_iter, intercept)`.

    `x` is column-major `n_rows x n_cols`; `coef` holds `n_cols` floats and
    is READ AS THE STARTING POINT (theirs does not zero it; cuML's Python
    passes `cp.zeros`, and a caller here must do the same). `x` and `labels`
    are MUTATED IN PLACE under `fit_intercept` (centered, then un-centered
    by `postProcessData`, which does not restore the bits exactly -- see
    `solver/impl/glm/preprocess.mojo`).
    """
    var tr = IdentityTrace.disabled()
    var res = ctx.enqueue_create_buffer[DType.float32](1)
    var out = cd_fit_traced(
        ctx, x, n_rows, n_cols, labels, coef, fit_intercept, epochs, loss,
        alpha, l1_ratio, shuffle, tol, has_sample_weight, tr, "cd",
        CdLaunch.default(), res, False,
    )
    _ = res^
    return out


def cd_fit_traced(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut labels: DeviceBuffer[DType.float32],
    mut coef: DeviceBuffer[DType.float32],
    fit_intercept: Bool,
    epochs: Int,
    loss: Int,
    alpha: Float32,
    l1_ratio: Float32,
    shuffle: Bool,
    tol: Float32,
    has_sample_weight: Bool,
    mut trace: IdentityTrace,
    prefix: String,
    launch: CdLaunch,
    mut residual_out: DeviceBuffer[DType.float32],
    want_residual: Bool,
    row_major: Bool = False,
) raises -> Tuple[Int, Float32]:
    """`cdFit` with a stage card, a scheduling surface and the residual
    handed out. The dispatch guards first, in their order.

    `row_major` (lane/apple-fast-linear, -D MOJOLEARN_CD_FAST_ROWMAJOR): `x`
    is the caller's ROW-MAJOR design; only the CD_FAST_GRAM grid path reads
    it (the kernels above), so it is refused by name everywhere else."""
    if row_major:
        comptime if not CD_FAST_GRAM:
            raise Error(
                "cd_fit: a row-major design is served only by the FAST Apple"
                " Gram path (CD_FAST_GRAM); hand the column-major design"
            )
        if (
            trace.enabled
            or want_residual
            or n_cols > CD_GRAM_MAX_COLS
            or n_rows < 4 * n_cols
        ):
            raise Error(
                "cd_fit: a row-major design needs the Gram path: no trace, no"
                " residual, n_cols <= 256 and n_rows >= 4 n_cols"
            )
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if loss != LOSS_SQRD_LOSS:
        raise Error(
            "Parameter loss: Only SQRT_LOSS function is supported for now"
            " (got " + loss_funct_name(loss) + ")"
        )
    if has_sample_weight:
        raise Error(
            "Parameter sample_weight: REFUSED BY NAME. cd.cuh:136-163 and"
            " :240-251 (the weighted preprocess, the sqrt-weight scaling of"
            " input and labels, and their undo) are not implemented"
        )
    if shuffle:
        raise Error(
            "Parameter shuffle: REFUSED BY NAME (cuML selection='random')."
            " std::shuffle's algorithm is unspecified by the C++ standard,"
            " so cuML's permutation is not a pure function of its seed; only"
            " the cyclic order (shuffle=false, selection='cyclic') is implemented."
            " See solver/impl/shuffle.mojo"
        )
    if alpha < Float32(0.0):
        raise Error("Expected alpha >= 0, got " + String(alpha))
    if l1_ratio < Float32(0.0) or l1_ratio > Float32(1.0):
        raise Error(
            "Expected 0.0 <= l1_ratio <= 1.0, got " + String(l1_ratio)
        )
    # ======================================================================
    # DEVIATION 613 -- non-finite `alpha`, NaN `l1_ratio`, NaN `tol`
    # ======================================================================
    # WHAT THEIRS DOES: cuML's Python guards (`elastic_net.py:199-206`,
    # mirrored just above) are `alpha < 0` and `l1_ratio < 0 or > 1`, both
    # FALSE for a NaN, and nothing bounds `alpha` above; a NaN `alpha` or
    # `l1_ratio` makes `l1_alpha`/`l2_alpha` NaN (`cd.cuh:168-169`), every
    # `coef > l1_alpha` false and every coefficient 0, and an `alpha` of
    # +inf at `l1_ratio = 1` makes `l2_alpha = 0 * inf = NaN`. A NaN `tol`
    # never stops (`cd.cuh:236`, both tests false). The fit runs and
    # returns zeros or runs out the epochs.
    # WHAT OURS DOES: refuses each BY NAME. The guard's intent is "a
    # nonnegative number", and a NaN is not one; with these three refused,
    # `cd.l1_alpha`/`cd.l2_alpha` are products of finite operands (finite or
    # +inf, never NaN) and no PARAMETER can put a NaN on the card -- only
    # data can, and DEVIATION 612 (`solver/checks/record_canon.mojo`)
    # canonicalizes that at the record.
    # MEASURED: `check_cd_refuses_by_name` (alpha=NaN, alpha=inf,
    # l1_ratio=NaN, tol=NaN each raise naming the parameter).
    # ======================================================================
    if alpha != alpha or alpha - alpha != Float32(0.0):
        raise Error(
            "Parameter alpha: must be a finite number, got " + String(alpha)
        )
    if l1_ratio != l1_ratio:
        raise Error("Parameter l1_ratio: must not be NaN")
    if tol != tol:
        raise Error("Parameter tol: must not be NaN")

    trace.header(
        String("cdFit n_rows=") + String(n_rows) + " n_cols=" + String(n_cols)
        + " fit_intercept=" + String(fit_intercept) + " epochs="
        + String(epochs) + " alpha=" + String(alpha) + " l1_ratio="
        + String(l1_ratio) + " tol=" + String(tol) + " shuffle=false"
    )
    # DEVIATION 612: every float stage is hashed through a NaN-canonicalized
    # COPY (`solver/checks/record_canon.mojo`); the scratch is sized for
    # the largest stage (`x`) and is one float when the trace is off.
    var canon_n = n_rows * n_cols
    if canon_n < 3:
        canon_n = 3
    if not trace.enabled:
        canon_n = 1
    var canon_ws = ctx.enqueue_create_buffer[DType.float32](canon_n)
    record_device_canon(ctx, trace, prefix + ".input.x", x, n_rows * n_cols, canon_ws)
    record_device_canon(ctx, trace, prefix + ".input.y", labels, n_rows, canon_ws)

    var residual = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var squared = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var mu_input = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var mu_labels = ctx.enqueue_create_buffer[DType.float32](1)
    # The profile dot's scratch (IDENTICAL arm; 1 float and unused under
    # FAST) and the ones vector the means are dotted against.
    var ws_rows = ctx.enqueue_create_buffer[DType.float32](
        profile_dot_workspace_floats(n_rows)
    )
    var ws_cols = ctx.enqueue_create_buffer[DType.float32](
        profile_dot_workspace_floats(n_cols)
    )
    var ones = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.enqueue_memset(ones, Float32(1.0))
    ctx.enqueue_memset(ws_rows, Float32(0.0))
    ctx.enqueue_memset(ws_cols, Float32(0.0))

    if fit_intercept and not row_major:
        pre_process_data(
            ctx, x, n_rows, n_cols, labels, mu_input, mu_labels,
            fit_intercept, ones, ws_rows, launch.dot_plan,
        )
        record_device_canon(ctx, trace, prefix + ".mu_input", mu_input, n_cols, canon_ws)
        record_device_canon(ctx, trace, prefix + ".mu_labels", mu_labels, 1, canon_ws)

    var ri = init_shuffle(n_cols)

    # cd.cuh:168-169, in math_t = float, left to right. Each factor in its
    # own local so no codegen may contract `(1 - l1_ratio) * alpha` into an
    # fma across the statement; the values are on the card regardless.
    var one_minus = Float32(1.0) - l1_ratio
    var l2_a = one_minus * alpha
    var l2_alpha = l2_a * Float32(n_rows)
    var l1_a = l1_ratio * alpha
    var l1_alpha = l1_a * Float32(n_rows)
    # Row 39 / DEVIATION 613: finite * [0,1] * n_rows -- finite or +inf,
    # never NaN, so these two host scalars are recorded raw.
    trace.record_scalar_f32(prefix + ".l1_alpha", l1_alpha)
    trace.record_scalar_f32(prefix + ".l2_alpha", l2_alpha)

    # Precompute: colNorm, + l2_alpha, residual = labels.
    # (row_major: the Gram sweeps take the diagonal of the Gram instead)
    if not row_major:
        col_norm_l2_squared(ctx, squared, x, n_cols, n_rows, ws_rows, launch.dot_plan)
        record_device_canon(ctx, trace, prefix + ".colnorm", squared, n_cols, canon_ws)
        ctx.enqueue_function[add_scalar_cols_kernel](
            squared.unsafe_ptr(), Int32(n_cols), l2_alpha,
            grid_dim=((n_cols + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        record_device_canon(ctx, trace, prefix + ".squared", squared, n_cols, canon_ws)
    ctx.enqueue_copy(dst_buf=residual, src_buf=labels)

    var conv = ctx.enqueue_create_buffer[DType.float32](3)
    var h_conv = ctx.enqueue_create_host_buffer[DType.float32](3)

    var n_iter = 0
    var device_sweeps = True
    # lane/apple-fast-linear: the intercept of the row-major form, from the
    # means read back with the Gram (set inside the Gram arm below)
    var rm_intercept = 0.0
    comptime if CD_FAST_GRAM:
        if (
            not trace.enabled
            and not want_residual
            and n_cols <= CD_GRAM_MAX_COLS
            and n_rows >= 4 * n_cols
        ):
            device_sweeps = False
            var p = n_cols
            var p1 = p + 1
            var gsum = List[Float64](capacity=p1 * p1)
            for _ in range(p1 * p1):
                gsum.append(0.0)
            var blocks_done = False
            comptime if CD_GRAM_BLOCKS:
                if p1 * p1 <= CD_GRAM_CELLS:
                    # lane/linear-apple3: the upper triangle straight from x
                    # and the labels, CD_GB_ROWS rows a block (no staged copy)
                    blocks_done = True
                    var gcells = p1 * (p1 + 1) // 2
                    var gnb = (n_rows + CD_GB_ROWS - 1) // CD_GB_ROWS
                    var gp = ctx.enqueue_create_buffer[DType.float32](gnb * gcells)
                    var hgp = ctx.enqueue_create_host_buffer[DType.float32](gnb * gcells)
                    ctx.enqueue_function[cd_gram_blocks_kernel](
                        x.unsafe_ptr(), labels.unsafe_ptr(), gp.unsafe_ptr(), Int32(p),
                        Int32(n_rows), grid_dim=(gnb, 1, 1),
                        block_dim=(CD_GB_TPB, 1, 1),
                    )
                    ctx.enqueue_copy(dst_buf=hgp, src_buf=gp)
                    ctx.synchronize()
                    var gc = 0
                    for a in range(p1):
                        for bb in range(a, p1):
                            var acc = 0.0
                            for k in range(gnb):
                                acc += Float64(hgp.unsafe_ptr().unsafe_load(k * gcells + gc))
                            gsum[a * p1 + bb] = acc
                            gsum[bb * p1 + a] = acc
                            gc += 1
                    _ = gp^
                    _ = hgp^
            var gmu = List[Float64](capacity=p1)
            if not blocks_done and (row_major or CD_FAST_GRID_GRAM):
                # lane/apple-fast-linear: the product on the grid (see
                # cd_grid_gram_kernel's banner); row_major also takes the
                # means from the chunks and centers the tiles at them
                blocks_done = True
                var nch = (n_rows + CD_GG_CH - 1) // CD_GG_CH
                var nt = (p1 + CD_GG_TS - 1) // CD_GG_TS
                var npairs = nt * (nt + 1) // 2
                var rm = Int32(1) if row_major else Int32(0)
                var dps = ctx.enqueue_create_buffer[DType.float32](nch * p1)
                var dmu = ctx.enqueue_create_buffer[DType.float32](p1)
                var dpg = ctx.enqueue_create_buffer[DType.float32](
                    nch * npairs * CD_GG_TS * CD_GG_TS
                )
                var dgram = ctx.enqueue_create_buffer[DType.float32](p1 * p1)
                var hgram = ctx.enqueue_create_host_buffer[DType.float32](p1 * p1)
                var hmu = ctx.enqueue_create_host_buffer[DType.float32](p1)
                if row_major and fit_intercept:
                    ctx.enqueue_function[cd_grid_sums_kernel](
                        x.unsafe_ptr(), labels.unsafe_ptr(), dps.unsafe_ptr(),
                        Int32(p), Int32(n_rows), rm,
                        grid_dim=(nch, 1, 1), block_dim=(CD_GG_TPB, 1, 1),
                    )
                    ctx.enqueue_function[cd_grid_means_kernel](
                        dps.unsafe_ptr(), dmu.unsafe_ptr(), Int32(p1), Int32(nch),
                        Int32(n_rows),
                        grid_dim=((p1 + CD_GG_TPB - 1) // CD_GG_TPB, 1, 1),
                        block_dim=(CD_GG_TPB, 1, 1),
                    )
                else:
                    # the columns are centered already (pre_process_data),
                    # or there is no intercept to center for
                    ctx.enqueue_memset(dmu, Float32(0.0))
                ctx.enqueue_function[cd_grid_gram_kernel](
                    x.unsafe_ptr(), labels.unsafe_ptr(), dmu.unsafe_ptr(),
                    dpg.unsafe_ptr(), Int32(p), Int32(n_rows), rm, Int32(npairs),
                    Int32(nt),
                    grid_dim=(nch * npairs, 1, 1), block_dim=(CD_GG_TPB, 1, 1),
                )
                ctx.enqueue_function[cd_grid_gram_red_kernel](
                    dpg.unsafe_ptr(), dgram.unsafe_ptr(), Int32(p1), Int32(nch),
                    Int32(npairs), Int32(nt),
                    grid_dim=(
                        (npairs * CD_GG_TS * CD_GG_TS + CD_GG_TPB - 1) // CD_GG_TPB, 1, 1
                    ),
                    block_dim=(CD_GG_TPB, 1, 1),
                )
                ctx.enqueue_copy(dst_buf=hgram, src_buf=dgram)
                ctx.enqueue_copy(dst_buf=hmu, src_buf=dmu)
                ctx.synchronize()
                for q in range(p1 * p1):
                    gsum[q] = Float64(hgram.unsafe_ptr().unsafe_load(q))
                for q in range(p1):
                    gmu.append(Float64(hmu.unsafe_ptr().unsafe_load(q)))
                _ = dps^
                _ = dmu^
                _ = dpg^
                _ = dgram^
                _ = hgram^
                _ = hmu^
            if not blocks_done:
                # B = [X^T rows | y] as (p + 1) x n_rows row-major: X is column-
                # major, so its columns are already the first p rows.
                var bmat = ctx.enqueue_create_buffer[DType.float32](p1 * n_rows)
                ctx.enqueue_copy(
                    dst_buf=bmat.create_sub_buffer[DType.float32](0, p * n_rows),
                    src_buf=x.create_sub_buffer[DType.float32](0, p * n_rows),
                )
                ctx.enqueue_copy(
                    dst_buf=bmat.create_sub_buffer[DType.float32](p * n_rows, n_rows),
                    src_buf=labels.create_sub_buffer[DType.float32](0, n_rows),
                )
                var n_parts = (n_rows + CD_GRAM_ROWS - 1) // CD_GRAM_ROWS
                var gext = ctx.enqueue_create_buffer[DType.float32](
                    p1 * p1 if p1 * p1 > CD_GRAM_CELLS else n_parts * p1 * p1
                )
                var bview = bmat.create_sub_buffer[DType.float32](0, p1 * n_rows)
                var hg = ctx.enqueue_create_host_buffer[DType.float32](len(gext))
                if p1 * p1 > CD_GRAM_CELLS:
                    gemm_nt(ctx, gext, bmat, bview, p1, p1, n_rows)
                    n_parts = 1
                else:
                    ctx.enqueue_function[cd_gram_partial_kernel](
                        bmat.unsafe_ptr(), gext.unsafe_ptr(), Int32(p1),
                        Int32(n_rows), grid_dim=(n_parts, 1, 1),
                        block_dim=(CD_GRAM_CELLS, 1, 1),
                    )
                ctx.enqueue_copy(dst_buf=hg, src_buf=gext)
                ctx.synchronize()
                for k in range(n_parts):
                    for q in range(p1 * p1):
                        gsum[q] += Float64(hg.unsafe_ptr().unsafe_load(k * p1 * p1 + q))
                _ = bview^
                _ = bmat^
                _ = gext^
                _ = hg^
            var G = List[Float64](capacity=p * p)
            var c = List[Float64](capacity=p)
            for a in range(p):
                for b in range(p):
                    G.append(gsum[a * p1 + b])
                c.append(gsum[a * p1 + p])
            var w = List[Float64](capacity=p)
            var sq = List[Float64](capacity=p)
            for j in range(p):
                w.append(0.0)
                sq.append(G[j * p + j] + Float64(l2_alpha))
            var l1 = Float64(l1_alpha)
            var tol64 = Float64(tol)
            while n_iter < epochs:
                var coef_max = 0.0
                var diff_max = 0.0
                for jj in range(p):
                    var j = ri[jj]
                    var old = w[j]
                    var rho = c[j] + G[j * p + j] * old
                    var r = 0.0
                    if rho > l1:
                        r = rho - l1
                    elif rho < -l1:
                        r = rho + l1
                    if sq[j] > Float64(CD_SQUARED_GUARD):
                        r = r / sq[j]
                    else:
                        r = 0.0
                    var dw = r - old
                    if dw != 0.0:
                        for k in range(p):
                            c[k] -= G[k * p + j] * dw
                    w[j] = r
                    if abs(dw) > diff_max:
                        diff_max = abs(dw)
                    if abs(r) > coef_max:
                        coef_max = abs(r)
                n_iter += 1
                if coef_max < tol64 or (diff_max / coef_max) < tol64:
                    break
            var hw = ctx.enqueue_create_host_buffer[DType.float32](p)
            for j in range(p):
                hw.unsafe_ptr().unsafe_store(j, Float32(w[j]))
            ctx.enqueue_copy(dst_buf=coef, src_buf=hw)
            ctx.synchronize()
            _ = hw^
            if row_major and fit_intercept and len(gmu) == p1:
                # postProcessData's intercept, mu_labels - mu_input . coef,
                # from the means the Gram pass read back
                var bb = gmu[p]
                for j in range(p):
                    bb -= gmu[j] * w[j]
                rm_intercept = bb
            _ = gmu^
    # lane/fam-linear: the IDENTICAL Gram sweeps (see CD_IDN_GRAM above); a
    # traced fit and a fit that wants the residual keep the row sweeps.
    comptime if CD_IDN_GRAM:
        if (
            not trace.enabled
            and not want_residual
            and not row_major
            and cd_idn_gram_shape(n_rows, n_cols)
        ):
            device_sweeps = False
            var gp = n_cols
            var gram = ctx.enqueue_create_buffer[DType.float32](gp * gp)
            var gq = ctx.enqueue_create_buffer[DType.float32](gp)
            var gst = ctx.enqueue_create_buffer[DType.float32](4)
            var h_gst = ctx.enqueue_create_host_buffer[DType.float32](4)
            var gws_a = identical_gemm_workspace_max_floats(gp, gp, n_rows)
            var gws_b = identical_gemm_workspace_max_floats(gp, 1, n_rows)
            var gws_n = gws_a if gws_a > gws_b else gws_b
            if gws_n < 1:
                gws_n = 1
            var gws = ctx.enqueue_create_buffer[DType.float32](gws_n)
            var x_b = x.create_sub_buffer[DType.float32](0, gp * n_rows)
            # X is column-major n_rows x n_cols: its columns are the rows of
            # an n_cols x n_rows NT operand. G = X^T X, q = X^T y.
            identical_gemm_into(ctx, gram, x, x_b, gws, gp, gp, n_rows, OP_NT)
            identical_gemm_into(ctx, gq, x, labels, gws, gp, 1, n_rows, OP_NT)
            ctx.enqueue_memset(gst, Float32(0.0))
            while n_iter < epochs:
                var e_here = epochs - n_iter
                if e_here > CD_IDN_GRAM_EPOCHS:
                    e_here = CD_IDN_GRAM_EPOCHS
                ctx.enqueue_function[cd_idn_gram_sweep_kernel](
                    gram.unsafe_ptr(), gq.unsafe_ptr(), coef.unsafe_ptr(),
                    squared.unsafe_ptr(), gst.unsafe_ptr(), Int32(gp),
                    Int32(e_here), l1_alpha, tol,
                    grid_dim=(1, 1, 1), block_dim=(CD_IDN_GRAM_TPB, 1, 1),
                )
                ctx.enqueue_copy(dst_ptr=h_gst.unsafe_ptr(), src_buf=gst)
                ctx.synchronize()
                n_iter = Int(bitcast[DType.uint32](h_gst.unsafe_ptr().unsafe_load(1)))
                if h_gst.unsafe_ptr().unsafe_load(0) != Float32(0.0):
                    break
            _ = x_b^
            _ = gws^
            _ = gram^
            _ = gq^
            _ = gst^
            _ = h_gst^
    # lane/linear-apple2: the three-launch coordinate (see
    # cd_axpy_pair_kernel) where the profile dot is PLAN_SPLITK on one device.
    var three = False
    var part = contract_partition(n_rows)
    comptime if CD_THREE_LAUNCH:
        var dc = String(getenv("MOJOLEARN_SOLVER_DEVICE_COUNT"))
        three = (
            launch.dot_plan < 0
            and (dc == "" or dc == "1")
            and part[1] > 0
            and choose_gemm_plan(1, 1, n_rows) == PLAN_SPLITK
        )
    var two = False
    comptime if CD_TWO_STEP:
        two = three
    # lane/gap-nv-classical2: the one-launch coordinate off Apple
    var fused = False
    comptime if CD_FUSED:
        var dcf = String(getenv("MOJOLEARN_SOLVER_DEVICE_COUNT"))
        fused = (
            launch.dot_plan < 0
            and (dcf == "" or dcf == "1")
            and part[1] > 0
            and part[1] <= CONTRACT_MAX_LEAVES
        )
        if fused:
            two = True
    var ws_b = ctx.enqueue_create_buffer[DType.float32](part[1] if fused and part[1] > 0 else 1)
    var ws_in_a = True  # the next fold reads ws_rows (else ws_b)
    var fused_blocks = (part[1] + CD_FUSED_LEAVES - 1) // CD_FUSED_LEAVES
    var coef_b = ctx.enqueue_create_buffer[DType.float32](n_cols if two else 1)
    var conv_b = ctx.enqueue_create_buffer[DType.float32](3)
    var in_a = True  # coef_in is `coef` (else coef_b)
    var step_blocks = min(CD_STEP_MAX_BLOCKS, (n_rows + CD_STEP_TPB - 1) // CD_STEP_TPB)
    if two:
        ctx.enqueue_copy(dst_buf=coef_b, src_buf=coef)
    while device_sweeps and two and n_iter < epochs:
        var pa = coef.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pb = coef_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var cin = pa if in_a else pb
        var cout = pb if in_a else pa
        var vin = conv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var vout = conv_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ctx.enqueue_memset(conv, Float32(0.0))
        for j in range(n_cols + 1):
            var ci = ri[j] if j < n_cols else -1
            var prev = ri[j - 1] if j > 0 else -1
            if fused:
                var wa = ws_rows.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                var wb = ws_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                var w_in = wa if ws_in_a else wb
                var w_out = wb if ws_in_a else wa
                ctx.enqueue_function[cd_fused_step_kernel](
                    residual.unsafe_ptr(), x.unsafe_ptr(), w_in, w_out,
                    cin, cout, squared.unsafe_ptr(), vin, vout,
                    Int32(prev), Int32(ci), Int32(n_rows), Int32(part[0]),
                    Int32(part[1]), l1_alpha,
                    grid_dim=(fused_blocks, 1, 1), block_dim=(CD_FUSED_TPB, 1, 1),
                )
                if ci >= 0:
                    ws_in_a = not ws_in_a
            else:
                ctx.enqueue_function[cd_step_kernel](
                    residual.unsafe_ptr(), x.unsafe_ptr(), ws_rows.unsafe_ptr(),
                    cin, cout, squared.unsafe_ptr(), vin, vout,
                    Int32(prev), Int32(ci), Int32(n_rows), Int32(part[1]), l1_alpha,
                    grid_dim=(step_blocks, 1, 1), block_dim=(CD_STEP_TPB, 1, 1),
                )
            if prev >= 0:
                var t = vin
                vin = vout
                vout = t
            if ci >= 0 and not fused:
                ctx.enqueue_function[identical_gemm_leaf_kernel](
                    ws_rows.unsafe_ptr(), x.unsafe_ptr() + ci * n_rows,
                    residual.unsafe_ptr(),
                    Int32(1), Int32(1), Int32(n_rows), Int32(part[0]), Int32(part[1]),
                    Int32(n_rows), Int32(1), Int32(1), Int32(n_rows), Int32(part[1]),
                    grid_dim=((part[1] + SPLITK_LEAF_LAUNCH_TPB - 1) // SPLITK_LEAF_LAUNCH_TPB, 1, 1),
                    block_dim=(SPLITK_LEAF_LAUNCH_TPB, 1, 1),
                )
        # the last update's maxima are in `vin`; keep them in `conv`
        if vin != conv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]():
            ctx.enqueue_copy(dst_buf=conv, src_buf=conv_b)
        in_a = not in_a
        ctx.enqueue_copy(dst_ptr=h_conv.unsafe_ptr(), src_buf=conv)
        ctx.synchronize()
        var coef_max = h_conv.unsafe_ptr().unsafe_load(1)
        var diff_max = h_conv.unsafe_ptr().unsafe_load(2)
        n_iter += 1
        var tag = prefix + ".sweep" + _pad3(n_iter - 1)
        if in_a:
            record_device_canon(ctx, trace, tag + ".coef", coef, n_cols, canon_ws)
        else:
            record_device_canon(ctx, trace, tag + ".coef", coef_b, n_cols, canon_ws)
        record_device_canon(ctx, trace, tag + ".resid", residual, n_rows, canon_ws)
        record_device_canon(ctx, trace, tag + ".conv", conv, 3, canon_ws)
        if coef_max < tol or (diff_max / coef_max) < tol:
            break
    if two and not in_a:
        ctx.enqueue_copy(dst_buf=coef, src_buf=coef_b)
    while device_sweeps and three and not two and n_iter < epochs:
        ctx.enqueue_memset(conv, Float32(0.0))
        for j in range(n_cols):
            var ci = ri[j]
            var prev = ri[j - 1] if j > 0 else -1
            ctx.enqueue_function[cd_axpy_pair_kernel](
                residual.unsafe_ptr(), x.unsafe_ptr(), coef.unsafe_ptr(),
                conv.unsafe_ptr(), Int32(ci), Int32(prev), Int32(n_rows),
                grid_dim=((n_rows + AXPY_TPB - 1) // AXPY_TPB, 1, 1),
                block_dim=(AXPY_TPB, 1, 1),
            )
            ctx.enqueue_function[identical_gemm_leaf_kernel](
                ws_rows.unsafe_ptr(), x.unsafe_ptr() + ci * n_rows,
                residual.unsafe_ptr(),
                Int32(1), Int32(1), Int32(n_rows), Int32(part[0]), Int32(part[1]),
                Int32(n_rows), Int32(1), Int32(1), Int32(n_rows), Int32(part[1]),
                grid_dim=((part[1] + SPLITK_LEAF_LAUNCH_TPB - 1) // SPLITK_LEAF_LAUNCH_TPB, 1, 1),
                block_dim=(SPLITK_LEAF_LAUNCH_TPB, 1, 1),
            )
            ctx.enqueue_function[cd_fold_update_kernel](
                coef.unsafe_ptr(), ws_rows.unsafe_ptr(), squared.unsafe_ptr(),
                conv.unsafe_ptr(), Int32(ci), Int32(part[1]), l1_alpha,
                grid_dim=(1, 1, 1), block_dim=(SPLITK_FOLD_TPB, 1, 1),
            )
        # the last coordinate's closing axpy: residual += conv.coef * X[:, ci]
        axpy_device_alpha(
            ctx, residual, x, ri[n_cols - 1] * n_rows, conv, 0, n_rows,
            launch.axpy_tpb, launch.axpy_two_d_grid,
        )
        ctx.enqueue_copy(dst_ptr=h_conv.unsafe_ptr(), src_buf=conv)
        ctx.synchronize()
        var coef_max = h_conv.unsafe_ptr().unsafe_load(1)
        var diff_max = h_conv.unsafe_ptr().unsafe_load(2)
        n_iter += 1
        var tag = prefix + ".sweep" + _pad3(n_iter - 1)
        record_device_canon(ctx, trace, tag + ".coef", coef, n_cols, canon_ws)
        record_device_canon(ctx, trace, tag + ".resid", residual, n_rows, canon_ws)
        record_device_canon(ctx, trace, tag + ".conv", conv, 3, canon_ws)
        if coef_max < tol or (diff_max / coef_max) < tol:
            break
    while device_sweeps and not three and not two and n_iter < epochs:
        # shuffle=true refused above; ri stays the identity.
        ctx.enqueue_memset(conv, Float32(0.0))
        for j in range(n_cols):
            var ci = ri[j]
            # remember current coef
            ctx.enqueue_function[cd_remember_coef_kernel](
                conv.unsafe_ptr(), coef.unsafe_ptr(), Int32(ci),
                grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
            )
            # residual[:] += coef[ci] * X[:, ci]
            axpy_device_alpha(
                ctx, residual, x, ci * n_rows, coef, ci, n_rows,
                launch.axpy_tpb, launch.axpy_two_d_grid,
            )
            # coef[ci] = dot(X[:, ci], residual[:])
            var x_col = x.create_sub_buffer[DType.float32](ci * n_rows, n_rows)
            var coef_ci = coef.create_sub_buffer[DType.float32](ci, 1)
            comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                profile_dot_into(
                    ctx, coef_ci, x_col, residual, ws_rows, n_rows, launch.dot_plan
                )
            else:
                # raft::linalg::gemv<math_t, true>(false, 1, n_rows, ...):
                # cuBLAS, CLOSED; MAX's gemv is the mirror.
                gemv_n(ctx, coef_ci, x_col, residual, 1, n_rows)
            # SoftThreshold(dot, l1_alpha) / squared, and the criteria.
            ctx.enqueue_function[cd_update_coef_kernel](
                coef.unsafe_ptr(), Int32(ci), squared.unsafe_ptr(),
                conv.unsafe_ptr(), l1_alpha,
                grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
            )
            # residual[:] += conv.coef * X[:, ci]   (conv.coef == -r)
            axpy_device_alpha(
                ctx, residual, x, ci * n_rows, conv, 0, n_rows,
                launch.axpy_tpb, launch.axpy_two_d_grid,
            )
            _ = x_col^
            _ = coef_ci^
        # update_host(&h_convState, convStateLoc, 1); sync
        ctx.enqueue_copy(dst_ptr=h_conv.unsafe_ptr(), src_buf=conv)
        ctx.synchronize()
        var coef_max = h_conv.unsafe_ptr().unsafe_load(1)
        var diff_max = h_conv.unsafe_ptr().unsafe_load(2)
        n_iter += 1
        var tag = prefix + ".sweep" + _pad3(n_iter - 1)
        record_device_canon(ctx, trace, tag + ".coef", coef, n_cols, canon_ws)
        record_device_canon(ctx, trace, tag + ".resid", residual, n_rows, canon_ws)
        record_device_canon(ctx, trace, tag + ".conv", conv, 3, canon_ws)
        if coef_max < tol or (diff_max / coef_max) < tol:
            break

    var intercept = Float32(0.0)
    if fit_intercept and row_major:
        intercept = Float32(rm_intercept)
    elif fit_intercept:
        var d_intercept = ctx.enqueue_create_buffer[DType.float32](1)
        intercept = post_process_data(
            ctx, x, n_rows, n_cols, labels, coef, mu_input, mu_labels,
            d_intercept, ws_cols, launch.dot_plan,
        )
        _ = d_intercept^
    ctx.synchronize()
    record_device_canon(ctx, trace, prefix + ".final.coef", coef, n_cols, canon_ws)
    record_scalar_f32_canon(trace, prefix + ".intercept", intercept)
    var iters = List[Int32]()
    iters.append(Int32(n_iter))
    trace.record_list_i32(prefix + ".n_iter", iters)

    if want_residual:
        ctx.enqueue_copy(dst_buf=residual_out, src_buf=residual)
        ctx.synchronize()

    # [[mojo-buffer-freed-at-last-use]]: every scratch outlives the queue.
    _ = residual^
    _ = squared^
    _ = mu_input^
    _ = mu_labels^
    _ = ws_rows^
    _ = ws_cols^
    _ = ones^
    _ = conv^
    _ = conv_b^
    _ = coef_b^
    _ = ws_b^
    _ = h_conv^
    _ = canon_ws^
    return (n_iter, intercept)


def _pad3(i: Int) -> String:
    var s = String(i)
    while s.byte_length() < 3:
        s = String("0") + s
    return s


def cd_predict(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut coef: DeviceBuffer[DType.float32],
    intercept: Float32,
    mut preds: DeviceBuffer[DType.float32],
    loss: Int,
) raises:
    """`cdPredict`, `cd.cuh:293-307`: the guards, then `linearRegH`."""
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if loss != LOSS_SQRD_LOSS:
        raise Error(
            "Parameter loss: Only SQRT_LOSS function is supported for now"
            " (got " + loss_funct_name(loss) + ")"
        )
    linear_reg_h(ctx, x, n_rows, n_cols, coef, preds, intercept)
