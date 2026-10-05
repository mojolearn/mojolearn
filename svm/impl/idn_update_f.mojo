# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL SMO gradient update without the kernel tile (lane/fam-linear,
2026-10-04).

After each block solve the default path writes the `nnz x batch` kernel tile
(`KernelCache.get_next_batch_kernel`: the identical GEMM, then the RBF
epilogue launch over the same cells) and `update_f_kernel` reads it back
once. `idn_fused_update_f_kernel` computes each kernel value where it is
used: one thread per training row keeps its row in registers, the moved
points stream through shared memory IN DEVIATION 634's FOLD ORDER, and the
thread folds `K(xw_j, x_i) * da_j` into `f_i` (and, for SVR, into the second
half at `i + n_rows`, which shares the kernel row).

BITS: unchanged, by construction. Every expression is the one the tile path
evaluates, in the same order, per cell:

    dot  = ftz(chain), chain: acc = ftz(fma(ftz(xw[j, p]), ftz(x[i, p]), acc)),
           p ascending from +0.0: profile `identical.gemm.fp32.v1` at
           k <= CONTRACT_K_LEAF_MIN (128) is ONE leaf and no fold addition
           (`gemm/contract.mojo::contract_leaf_size`,
           `gemm/host/gemm_oracle.mojo::gemm_oracle_cell`); this kernel
           serves k <= 64 only
    RBF  = `rbf_kernel_expanded_kernel`'s cell: s = ftz(ftz(ftz(norm_w[j]) +
           ftz(norm_x[i])) - ftz(2 * ftz(dot))), e = ftz((-gamma) * s),
           K = ftz(identical_exp(e))
    f    = `update_f_kernel`: acc = ftz(fma(K, da_j, acc)) over j in
           `fold_order`, then f_i = ftz(f_i + acc)

so the host column (`svm/host/smo_oracle.mojo`) is untouched. Polynomial,
tanh and precomputed kernels, k > 64 and the multi-device kernel rows keep
the tile path.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_exp, identical_mul_add

comptime IUF_TPB = 256
#: the staged working-set rows: the FAST kernel's page (`fast_update_f.mojo`
#: FUF_TILE_FLOATS), 12 KB plus two `ROWS`-float side pages (at most 6 KB at
#: KPAD = 4), inside every vendor's block page and Apple's 32 KB threadgroup
#: limit
comptime IUF_TILE_FLOATS = 3072
comptime IUF_MAX_K = 64


def idn_fused_update_f_kernel[KPAD: Int, RBF: Bool](
    f: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    norm_x: MutPointer[Float32, MutAnyOrigin],
    xw: MutPointer[Float32, MutAnyOrigin],
    norm_w: MutPointer[Float32, MutAnyOrigin],
    da: MutPointer[Float32, MutAnyOrigin],
    fold_order: MutPointer[Int32, MutAnyOrigin],
    nnz_in: Int32,
    n_rows_in: Int32,
    k_in: Int32,
    gain: Float32,
    svr: Int32,
):
    """`f[i] = ftz(f[i] + fold_j K(xw_j, x_i) * da_j)`, j through
    `fold_order`. `KPAD` is k rounded up to a multiple of 4; the feature
    chain runs over exactly k terms (pad lanes are skipped, not added)."""
    comptime ROWS = IUF_TILE_FLOATS // KPAD
    var nnz = Int(nnz_in)
    var n = Int(n_rows_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * IUF_TPB + tid
    var active = i < n
    var tile = stack_allocation[
        IUF_TILE_FLOATS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var tnorm = stack_allocation[
        ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var tda = stack_allocation[
        ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var xr = stack_allocation[KPAD, Scalar[DType.float32]]()
    var ni = Float32(0.0)
    comptime for c in range(KPAD):
        var v = Float32(0.0)
        if active and c < k:
            v = ftz(x.unsafe_load(i * k + c))
        xr.unsafe_store(c, v)
    comptime if RBF:
        if active:
            ni = ftz(norm_x.unsafe_load(i))
    var acc = Float32(0.0)
    var j0 = 0
    while j0 < nnz:
        var rows_here = nnz - j0
        if rows_here > ROWS:
            rows_here = ROWS
        barrier()
        var total = rows_here * KPAD
        var idx = tid
        while idx < total:
            var r = idx // KPAD
            var c = idx - r * KPAD
            var v = Float32(0.0)
            if c < k:
                var j = Int(fold_order.unsafe_load(j0 + r))
                v = ftz(xw.unsafe_load(j * k + c))
            tile.unsafe_store(idx, v)
            idx += IUF_TPB
        idx = tid
        while idx < rows_here:
            var j = Int(fold_order.unsafe_load(j0 + idx))
            tda.unsafe_store(idx, da.unsafe_load(j))
            comptime if RBF:
                tnorm.unsafe_store(idx, ftz(norm_w.unsafe_load(j)))
            idx += IUF_TPB
        barrier()
        if active:
            for r in range(rows_here):
                var tb = r * KPAD
                var dot = Float32(0.0)
                comptime for c in range(KPAD):
                    if c < k:
                        dot = ftz(
                            identical_mul_add(
                                tile.unsafe_load(tb + c), xr.unsafe_load(c), dot
                            )
                        )
                # the GEMM's output seam, then the epilogue's own read seam
                dot = ftz(dot)
                var kv: Float32
                comptime if RBF:
                    var s = ftz(
                        ftz(tnorm.unsafe_load(r) + ni)
                        - ftz(Float32(2.0) * dot)
                    )
                    var e = ftz((-gain) * s)
                    kv = ftz(identical_exp(e))
                else:
                    kv = dot
                acc = ftz(identical_mul_add(kv, tda.unsafe_load(r), acc))
        j0 += ROWS
    if active:
        f.unsafe_store(i, ftz(f.unsafe_load(i) + acc))
        if svr != 0:
            f.unsafe_store(i + n, ftz(f.unsafe_load(i + n) + acc))


def idn_update_f(
    ctx: DeviceContext,
    mut f: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut norm_x: DeviceBuffer[DType.float32],
    mut xw: DeviceBuffer[DType.float32],
    mut norm_w: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut fold_order: DeviceBuffer[DType.int32],
    nnz: Int,
    n_rows: Int,
    k: Int,
    gain: Float32,
    rbf: Bool,
    svr: Bool,
) raises -> Bool:
    """`f[i] += fold_j K(xw_j, x_i) * da_j` over all `n_rows` rows (and the
    SVR second half), one launch, no tile. False when `k` is too wide for
    this kernel; the caller then takes the tile path."""
    if k > IUF_MAX_K or k < 1 or nnz <= 0:
        return False
    var kpad = ((k + 3) // 4) * 4
    var grid = (n_rows + IUF_TPB - 1) // IUF_TPB
    var s = Int32(1) if svr else Int32(0)
    comptime for KP in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]:
        if kpad == KP:
            if rbf:
                ctx.enqueue_function[idn_fused_update_f_kernel[KP, True]](
                    f.unsafe_ptr(), x.unsafe_ptr(), norm_x.unsafe_ptr(),
                    xw.unsafe_ptr(), norm_w.unsafe_ptr(), da.unsafe_ptr(),
                    fold_order.unsafe_ptr(),
                    Int32(nnz), Int32(n_rows), Int32(k), gain, s,
                    grid_dim=grid, block_dim=IUF_TPB,
                )
            else:
                ctx.enqueue_function[idn_fused_update_f_kernel[KP, False]](
                    f.unsafe_ptr(), x.unsafe_ptr(), norm_x.unsafe_ptr(),
                    xw.unsafe_ptr(), norm_w.unsafe_ptr(), da.unsafe_ptr(),
                    fold_order.unsafe_ptr(),
                    Int32(nnz), Int32(n_rows), Int32(k), gain, s,
                    grid_dim=grid, block_dim=IUF_TPB,
                )
    return True
