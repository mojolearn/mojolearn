# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-identical-neural (2026-09-27): the IDENTICAL expanded-L2
distance tile on Apple's simdgroup matrix unit, with the register tile's
bits.

`pinned_distance_register_tile_kernel` computes, per cell, the dot product
`q_i . y_j` as ONE chain over the features, ascending, from +0.0, each step
`_rt_step` (round-then-flush, the contract; Apple's zero repair), then the
epilogue `max(ftz(fma(-2, dot, ftz(|q_i|^2 + |y_j|^2))), 0)` and the
optional root. This kernel writes the same `rows x cols` matrix:

  * the dot on the matrix unit, features staged in 16-deep windows (flushed
    on load, the zeros past `d` at the END of the chain only). The unit
    returns exactly the chain of Apple FMAs, ascending, from its seed
    (`gemm/checks/apple_simdgroup_probe.mojo`), i.e. flush-before-round; it
    differs from the contract's round-then-flush only where an exact step
    result lies in [2^-126 - 2^-150, 2^-126) (`checks/rtf_seam.mojo`);
  * a cell whose row's minimum nonzero exponent field plus its column's is
    at least 151 cannot meet that window (every product's lowest bit at or
    above 2^-149, every float a multiple of it: `APPLE_MMA_ADMIT_EXP_SUM`),
    so its matrix value IS the chain's; every other cell is recomputed with
    `_rt_step` over the features in order;
  * trailing zero products can only turn a -0.0 dot into +0.0, and the
    epilogue's add of the (non-negative) norm sum and its clamp return the
    same word for either;
  * the epilogue is the register tile's, verbatim.

`-D MOJOLEARN_KNN_APPLE_MMA_OFF` keeps the register tile.
"""

from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_mul_add, identical_sqrt
from gemm.checks.gemm_identical import (
    APPLE_MMA_ADMIT_EXP_SUM,
    _AMMA_M64,
    _amma_load_t,
    _amma_mma,
)
from neighbors.checks.pinned_distance_tile import _rt_step

comptime AMD_NT = 128
comptime AMD_BM = 64
comptime AMD_BN = 64
comptime AMD_KB = 16


@always_inline
def _exp_nz(x: Float32) -> UInt32:
    """Biased exponent field of a flushed word, 255 for a zero (a zero
    product is exact and constrains nothing)."""
    var e = (bitcast[DType.uint32](x) >> UInt32(23)) & UInt32(0xFF)
    return UInt32(0xFF) if e == UInt32(0) else e


def apple_mma_distance_tile_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    yt: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    y_stride_in: Int32,
    n_features_in: Int32,
    is_sqrt_in: Int32,
):
    """`z[i][j]` for rows [64 by, +64) x columns [64 bx, +64); `q` row-major
    `n_rows x d`, `yt` feature-major with row stride `y_stride`."""
    comptime NT = AMD_NT
    comptime BM = AMD_BM
    comptime BN = AMD_BN
    comptime KB = AMD_KB
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime FM = 4
    comptime FN = 4
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var y_stride = Int(y_stride_in)
    var d = Int(n_features_in)
    var row0 = Int(block_idx.y) * BM
    var col0 = Int(block_idx.x) * BN
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // 2
    var sgn = sg % 2
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rmin = stack_allocation[BM, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var cmin = stack_allocation[BN, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_AMMA_M64, FM * FN](fill=_AMMA_M64(0))
    # Thread t < 64 owns row t's running minimum, t >= 64 column t - 64's.
    var mine = UInt32(0xFF)
    var k0 = 0
    while k0 < d:
        # Stage A^T (feature-major) and B (column-major), flushed; zeros
        # outside the matrix and past d.
        comptime for s in range((BM * KB) // NT):
            var idx = s * NT + tid
            var r = idx // KB
            var f = idx % KB
            var gr = row0 + r
            var gf = k0 + f
            var v = Float32(0.0)
            if gr < n_rows and gf < d:
                v = ftz(q[gr * d + gf])
            at[f * AST + r] = v
        comptime for s in range((BN * KB) // NT):
            var idx = s * NT + tid
            var f = idx // BN
            var c = idx % BN
            var gc = col0 + c
            var gf = k0 + f
            var v = Float32(0.0)
            if gc < n_cols and gf < d:
                v = ftz(yt[gf * y_stride + gc])
            bt[c * BST + f] = v
        barrier()
        if tid < BM:
            for f in range(KB):
                mine = min(mine, _exp_nz(at[f * AST + tid]))
        else:
            var c = tid - BM
            for f in range(KB):
                mine = min(mine, _exp_nz(bt[c * BST + f]))
        comptime for p8 in range(KB // 8):
            comptime for fm in range(FM):
                var af = _amma_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
                comptime for fq in range(FN):
                    var bf = _amma_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                    acc[fm * FN + fq] = _amma_mma(af, bf, acc[fm * FN + fq])
        barrier()
        k0 += KB
    if tid < BM:
        rmin[tid] = mine
    else:
        cmin[tid - BM] = mine
    barrier()
    comptime for fm in range(FM):
        var lr = (sgm * FM + fm) * 8 + frow
        var row = row0 + lr
        if row < n_rows:
            var qn = ftz(q_norm.unsafe_load(row))
            comptime for fq in range(FN):
                comptime for e in range(2):
                    var lc = (sgn * FN + fq) * 8 + fcol + e
                    var col = col0 + lc
                    if col < n_cols:
                        var dot = acc[fm * FN + fq][e]
                        comptime if is_defined["MOJOLEARN_KNN_APPLE_MMA_SABOTAGE"]():
                            pass
                        else:
                            if rmin[lr] + cmin[lc] < UInt32(APPLE_MMA_ADMIT_EXP_SUM):
                                var t = Float32(0.0)
                                for f in range(d):
                                    t = _rt_step(ftz(q[row * d + f]), ftz(yt[f * y_stride + col]), t)
                                dot = t
                        var dist = ftz(
                            identical_mul_add(
                                Float32(-2.0),
                                dot,
                                ftz(qn + ftz(y_norm.unsafe_load(col))),
                            )
                        )
                        if dist <= Float32(0.0):
                            dist = Float32(0.0)
                        if is_sqrt_in != 0:
                            dist = ftz(identical_sqrt(dist))
                        z.unsafe_store(row * n_cols + col, dist)
