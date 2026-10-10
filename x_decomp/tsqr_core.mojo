# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BLOCKED TSQR (lane neural-pass140, 2026-10-02): the shape rules and the
scalar steps shared by the device kernels (x_decomp/tsqr_device.mojo) and
their host replay (x_decomp/tsqr_host.mojo). No GPU import: the CPU binding
compiles this file.

WHAT IT COMPUTES. The Householder QR of a tall m x n matrix (m >= n,
n <= TS_MAX_N), as R (n x n) and, on request, Q C for a small n x k C (the
implicit Q applied to C, never Q itself formed first): `linalg.qr` takes
C = diag(+-1) (Q with R's diagonal made non-negative), `linalg.svd` takes
C = U_R (the left vectors of R's own SVD, so U = Q U_R), `lstsq` and
`LinearRegression` factor [A | B] and read Q^T B and the residual block out
of R. NO GRAM MATRIX IS EVER FORMED: every inner product is of two
Householder vectors or of a vector and a data column.

THE ORDER, WHICH IS THE WHOLE IDENTITY CLAIM. Every choice below is a pure
function of the shape (m, n, k) and of nothing on the machine:

  * rows are cut into nb = max(1, m // TS_ROWS) blocks of TS_ROWS rows, the
    last block taking the remainder (so every block holds >= n rows);
  * each block is factored by Householder reflectors in column PANELS of
    TS_NB: inside a panel one reflector at a time (the panel's own columns
    updated reflector by reflector), then the panel's compact WY factor T
    (LAPACK larft, forward, columnwise) and ONE blocked update of the
    trailing columns, A <- (I - Y T^T Y^T) A;
  * every inner product over a block's rows [lo, hi) is TS_P chains, chain g
    the rows i == g (mod TS_P) ascending (ftz(fma(x_i, y_i, acc)) from 0),
    folded by `ts_fold` (a halving tree with a flush per addition). The
    device runs chain g on one thread; the host runs the same chains in one
    ascending pass with TS_P accumulators;
  * the nb block R factors are combined in a FIXED BINARY TREE: at level L
    (stride s = 2^L) tile a = 2 s t absorbs tile a + s (when it exists) by the
    structured Householder QR of the two stacked upper triangles; the
    reflectors' lower parts stay in the absorbed tile and their scalars in
    `tau_tree[(a + s) n + j]`. The root is tile 0;
  * Q C runs the same tree top down (H_{n-1} first at every node), then
    every block's panels last to first, x <- (I - Y T Y^T) x.

The reflector is `core/householder_qr.mojo`'s (DEVIATION 586: s =
-sign(a_jj) with sign(+-0) = +1, r = s ||x||, u1 = a_jj - r, tau = -s u1 /
||x||), restated in `ts_reflector` because that file imports the GPU, the
way decomposition/host/pca_full_oracle.mojo restates it. A zero column
(||x|| flushed to 0) is a zero diagonal and tau = 0 (DEVIATION 588): no
refusal, its reflector's stored tail is zeroed so no later product reads it.
"""
from experiments.classical_identical_ideas.linear_controls import C24_PANEL8, C24_ROWS2048, C24_TREE4
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined

#: rows per leaf block (the last block takes the remainder)
comptime TS_TREE_ARITY = 4 if C24_TREE4 else 2
#: (2048 by default since lane/grid-flips-1; `-D MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF` = 4096)
comptime TS_ROWS = 2048 if C24_ROWS2048 else 4096
#: chains per inner product over a block's rows (the fold width)
comptime TS_P = 16
#: columns per panel (also the device's column lanes)
comptime TS_NB = 8 if C24_PANEL8 else 16
#: device threads per block: TS_P row groups x TS_NB column lanes
comptime TS_TPB = TS_P * TS_NB
#: the widest matrix the route takes (a leaf block holds >= TS_ROWS rows, a
#: tree combine is O(n^3) cells in one threadgroup)
comptime TS_MAX_N = 512

# lane gap-tsqr (2026-10-08, docs/plans/gaps-2026-10-08.md section 7), IDENTICAL
# only; a FAST build keeps the per-panel forms. Both switches change bits, on
# every vendor and in the host replay together.
#
# TS_WY_PAIR (`-D MOJOLEARN_IDN_TSQR_WY_PAIR_OFF` restores the per-panel
# update): two consecutive panels (2 TS_NB reflectors, TS_W2 wide) share ONE
# blocked update of the trailing columns and ONE pass of Q C. Panel a is
# factored, its block reflector is applied to panel b's columns only (the
# words panel b saw before), panel b is factored, and the pair's compact WY
# factor T2 = [[Ta, -Ta (Ya^T Yb) Tb], [0, Tb]] (larft of the concatenated
# reflectors) drives x <- (I - Y T2^T Y^T) x over the columns right of the
# pair. The trailing matrix crosses the device twice per PAIR instead of
# twice per panel: half the traffic of the update passes, which are
# bandwidth-bound (TS_W2 multiply-adds per loaded word). The size rule: the
# pair width is what one thread's register file holds as accumulators
# (TS_W2 = 32 chains per thread) and what one TS_PART page folds in two
# rounds; a wider group (64) is the next step after this A/B, never a shape
# rule. Bits change in every column right of a pair (one 32-wide fold
# instead of two 16-wide ones); the panel columns themselves and R's first
# TS_NB columns keep their words.
#
# TS_TREE_PAR (`-D MOJOLEARN_IDN_TSQR_TREE_PAR_OFF` restores the serial
# forms): the tree combine's reflector norm is a lane-parallel fixed fold
# (lane t of TS_TPB holds alpha^2 (t == 0) then rows i == t (mod TS_TPB)
# ascending, a halving tree over the TS_TPB partials, `ts_tree_fold`) in
# place of one serial chain recomputed by every thread; the reflector
# column is staged in threadgroup memory for the column threads; the Q C
# tree apply tiles C's columns over threadgroups (TS_TPB a tile) with the
# reflector staged the same way. Bits change in the tree's norms only; the
# column updates keep their chains.
comptime TS_WY_PAIR = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_TSQR_WY_PAIR_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime TS_TREE_PAR = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_TSQR_TREE_PAR_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: reflectors of a panel pair (the pair's T2 is TS_W2 x TS_W2)
comptime TS_W2 = 2 * TS_NB
comptime TS_T2 = TS_W2 * TS_W2


def ts_pairs(n: Int) -> Int:
    """Panel pairs of an n-column factorization: pair q holds panels 2 q and
    2 q + 1 (the last pair is one panel when ts_panels(n) is odd)."""
    return (ts_panels(n) + 1) // 2


def ts_pair_width(n: Int, q: Int) -> Int:
    """Reflectors of pair q: TS_W2, or what is left of n."""
    return min(TS_W2, n - 2 * q * TS_NB)


def ts_tree_fold(mut s: List[Float32]):
    """The combine norm's fold of TS_TPB lane partials, in place: s[t] +=
    s[t + w] for w = TS_TPB / 2, ..., 1, each sum flushed; s[0] is the sum.
    The device runs the same tree in threadgroup memory."""
    var w = TS_TPB // 2
    while w > 0:
        for t in range(w):
            s[t] = ftz(s[t] + s[t + w])
        w = w // 2

# lane/classical-structural (2026-10-07), IDENTICAL only (default on since
# 2026-10-10, below; `-D MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO_OFF` turns it off). randomized_svd orthonormalizes its
# tall sketch (m x l, l small) nine times per fit through `orth`: two passes
# of the sliced Householder `qr_factor` (64 slices x 32 threads, 18 serial
# reflector steps each), a one-thread rank guard and a per-row trsm, about
# 154 launches, 6 syncs and 4 allocations per orth. Under this define `orth`
# is ONE blocked TSQR pass (`ts_factor_device`, the OLS kernels: 4096-row
# leaves in parallel, a fixed combine tree) with the explicit Q formed by
# applying the kept reflectors to a selection matrix (`ts_apply_device`,
# C = diag(R[j, j] != 0 after the rank guard)), about 25 launches and 1 sync
# per orth. Cost reasoning: the sketch is read three times at full width
# instead of being walked by 2,048 threads with strided loads; the gain is
# latency, not flops, so it holds for any tall m x l with l small. Bits
# change (one Householder pass with the TSQR's reflector order instead of
# two sliced passes): the host column (`HostExec.orth`) takes the same route
# through `ts_factor_host` / `ts_apply_host`, the TSQR's host replay, so the
# two columns move together. A dependent column (rank guard zero) maps to a
# zero column of C and so to an exactly zero Q column, which keeps
# randomized_svd's `nlive` compaction contract. Shapes the TSQR does not
# take (l > TS_MAX_N, m < l, an Int32 overflow) keep the two-pass route on
# both columns.
# PROMOTED to the IDENTICAL default 2026-10-10 (lane/grid-act-6, grid freeze
# 20261010 @50ebe26a5, runs g50ebe26a5/h: NVIDIA L40S sm_89 nv2 v1177 + AMD
# MI325X gfx942 amd2 b0038, full board data, one run per arm, incumbent once
# + stored floors). randomized-svd NV / AMD ms, two-pass orth -> TSQR orth:
#   istella 194.5 -> 160.2 (0.824) / 231.6 -> 141.6 (0.611), 0.710x combined;
#   taxi     68.9 ->  52.2 (0.757) / 114.3 ->  61.5 (0.538), 0.638x combined.
# Bits change vs the old default (one TSQR Householder pass) on the host
# column and both GPU vendors together. Absent = TSQR orth in IDENTICAL;
# -D MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO_OFF restores the two-pass route
# (the old on-define is refused by core/six_lane_experiment_guards.mojo).
comptime RSVD_IDN_TSQR_ORTHO = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO_OFF"]()
)


def orth_tsqr_shape_ok(m: Int, l: Int) -> Bool:
    """The shapes `tsqr_r_py` accepts (x_decomp/api.mojo): the TSQR orth is
    taken exactly for these on both columns."""
    return l >= 1 and m >= l and l <= TS_MAX_N and m * l <= 2147483647
#: cells (multiply-adds) per device launch: macOS silently cuts a long Metal
#: command buffer, so every phase is sliced to about this much work. 2^34
#: (lane apple-fast-tsqr; was 2^27, which left ONE to three leaf blocks per
#: launch at 221 columns and ~2,300 synchronized launches on istella, a
#: serial route on the Apple column): a few hundred leaf blocks per launch,
#: tens of ms of device work, far below the seconds macOS allows.
comptime TS_LAUNCH_CELLS = 1 << 34
#: the fewest blocks (or tree pairs) a sliced launch carries, whatever the
#: cell estimate says (TS_MAX_N columns: 256 x 3 x 8191 x 512 x 16 cells, a
#: few hundred ms at worst on an M2 Pro); slicing never changes a bit
comptime TS_LAUNCH_MIN_BLOCKS = 256


def ts_blocks(m: Int) -> Int:
    var nb = m // TS_ROWS
    return nb if nb > 1 else 1


def ts_block_lo(b: Int) -> Int:
    return b * TS_ROWS


def ts_block_hi(b: Int, nb: Int, m: Int) -> Int:
    return m if b == nb - 1 else (b + 1) * TS_ROWS


def ts_panels(n: Int) -> Int:
    return (n + TS_NB - 1) // TS_NB


def ts_first_row(lo: Int, g: Int) -> Int:
    """The first row i >= lo with i == g (mod TS_P): chain g's start."""
    var r = lo % TS_P
    var d = g - r
    if d < 0:
        d += TS_P
    return lo + d


def ts_fold(p: InlineArray[Float32, TS_P]) -> Float32:
    """The fold of the TS_P chains: p[t] += p[t + h] for h = 8, 4, 2, 1, each
    sum flushed; p[0]."""
    var w = p.copy()
    comptime for k in range(4):
        comptime h = TS_P >> (k + 1)
        comptime for t in range(h):
            w[t] = ftz(w[t] + w[t + h])
    return w[0]


@always_inline
def ts_reflector(ajj: Float32, normx: Float32) -> SIMD[DType.float32, 4]:
    """(r, u1, tau, 0) of `qr_reflector_r`, `qr_reflector_u1` and
    `qr_reflector_tau` (core/householder_qr.mojo, DEVIATION 586), spelled
    statement for statement. Call with normx != 0."""
    var s = Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)
    var r = ftz(s * normx)
    var u1 = ftz(ajj - r)
    var tau = ftz(identical_div(ftz(ftz(-s) * u1), normx))
    return SIMD[DType.float32, 4](r, u1, tau, Float32(0.0))


@always_inline
def ts_scale(tau: Float32, total: Float32) -> Float32:
    """td = tau * total, the pinned product (no contraction into the subtraction
    that follows)."""
    return ftz(identical_mul(tau, total))


@always_inline
def ts_fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return ftz(identical_mul_add(a, b, c))
