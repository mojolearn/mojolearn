# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The centered moments of [X | Y] on the whole device (lane/neural-pass120).

A team fit computed its column means, centered Gram and X'Y on ONE block,
one thread per value folding every row from device memory (M4, RidgeClassifier
taxi 1M: 0.37 s against sklearn's 0.11; MI325X, istella: 69 s against 4.6).
Here the same values come from launches over the device, the rows staged in
threadgroup memory (x_linear/cd_grid.mojo's staging, one prep):
  * the means: a block per 16 columns of [X | Y], thread c folds column c
    (`fold_fa`'s statements, then the quotient by n; zeros without an
    intercept);
  * the cross products: a block per pair of 16-column tiles of [X | Y],
    thread (a, b) the cell (j, k), j <= k: X'X (mirrored) and X'Y
    (`chain_cfmad`'s statements, x first), the Y'Y cells skipped.
Each value is one thread's chain over the rows ascending, so the words are
the team fit's. Outputs land at the caller's offsets of `fw`.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from x_linear.ops import IP
from x_linear.witness import witness_end
from x_linear.ops import FP, fs, fd, ld, st, i2f
from x_linear.tops import upper_cell, _acc_fa, _acc_fmad

comptime MG_TR = 128
comptime MG_TC = 16
comptime MG_NT = 256
comptime MG_RU = 16
comptime MG_LM = MG_TR * MG_TC // MG_NT
comptime MG_LG = MG_TR * 2 * MG_TC // MG_NT
comptime MG_BYTES = MG_TR * 2 * MG_TC * 4
comptime MOMENTS_GRID = lib_smem_page_fits_for[TARGET_COLUMN, MG_BYTES]()


@always_inline
def _augv(x: FP, y: FP, n: Int, d: Int, t_n: Int, row: Int, col: Int) -> Float32:
    """[X | Y] at (row, col), 0 past the edges; both loads from clamped
    in-range addresses, the word selected (no branch per load)."""
    var rr = min(row, n - 1)
    var xv = ld(x, rr * d + min(col, d - 1))
    var yv = ld(y, rr * t_n + min(max(col - d, 0), max(t_n - 1, 0)))  # t_n == 0: a load in range, never selected
    var v = xv if col < d else yv
    return v if (row < n and col < d + t_n) else Float32(0)


def mg_means_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, fi: Int32, fw: FP, xm: Int32, ym: Int32, wf: IP, woff: Int32, nonce: Int32):
    var nn = Int(n)
    var dd = Int(d)
    var tn = Int(t_n)
    var tid = Int(thread_idx.x)
    var c0 = Int(block_idx.x) * MG_TC
    var col = c0 + tid
    var live = tid < MG_TC and col < dd + tn
    var xs = stack_allocation[MG_TR * MG_TC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var reg = SIMD[DType.float32, MG_LM]()
    comptime for v in range(MG_LM):
        var u = v * MG_NT + tid
        reg[v] = _augv(x, y, nn, dd, tn, u // MG_TC, c0 + u % MG_TC)
    var acc = Float32(0)
    var r0 = 0
    while r0 < nn:
        barrier()
        comptime for v in range(MG_LM):
            xs[v * MG_NT + tid] = reg[v]
        barrier()
        var r1 = r0 + MG_TR
        if r1 < nn:
            comptime for v in range(MG_LM):
                var u = v * MG_NT + tid
                reg[v] = _augv(x, y, nn, dd, tn, r1 + u // MG_TC, c0 + u % MG_TC)
        if live:
            var cnt = Int32(min(MG_TR, nn - r0))
            var px = xs + tid
            var r = Int32(0)
            while r + MG_RU <= cnt:
                var bv = SIMD[DType.float32, MG_RU]()
                comptime for u in range(MG_RU):
                    bv[u] = px[u * MG_TC]
                comptime for u in range(MG_RU):
                    acc = _acc_fa(acc, bv[u])
                px += MG_RU * MG_TC
                r += MG_RU
            while r < cnt:
                acc = _acc_fa(acc, px[0])
                px += MG_TC
                r += 1
        r0 = r1
    if live:
        var v = fd(acc, i2f(nn)) if fi != 0 else Float32(0)
        if col < dd:
            st(fw, Int(xm) + col, v)
        else:
            st(fw, Int(ym) + col - dd, v)
    witness_end(wf, woff, nonce)

def mg_cross_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, fw: FP, xm: Int32, ym: Int32,
                    gg: Int32, xty: Int32, wf: IP, woff: Int32, nonce: Int32):
    var nn = Int(n)
    var dd = Int(d)
    var tn = Int(t_n)
    var aug = dd + tn
    var tiles = (aug + MG_TC - 1) // MG_TC
    var jk = upper_cell(Int(block_idx.x), tiles)
    var tid = Int(thread_idx.x)
    var j = jk[0] * MG_TC + tid // MG_TC
    var k = jk[1] * MG_TC + tid % MG_TC
    var live = j <= k and k < aug and j < dd
    var xs = stack_allocation[MG_TR * 2 * MG_TC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var mj = Float32(0)
    var mk = Float32(0)
    if live:
        mj = ld(fw, Int(xm) + j)
        mk = ld(fw, Int(xm) + k) if k < dd else ld(fw, Int(ym) + k - dd)
    var a = tid // MG_TC
    var b = MG_TC + tid % MG_TC
    var cj = jk[0] * MG_TC
    var ck = jk[1] * MG_TC
    var reg = SIMD[DType.float32, MG_LG]()
    comptime for v in range(MG_LG):
        var u = v * MG_NT + tid
        var cc = u % (2 * MG_TC)
        reg[v] = _augv(x, y, nn, dd, tn, u // (2 * MG_TC), cj + cc if cc < MG_TC else ck + cc - MG_TC)
    var acc = Float32(0)
    var r0 = 0
    while r0 < nn:
        barrier()
        comptime for v in range(MG_LG):
            xs[v * MG_NT + tid] = reg[v]
        barrier()
        var r1 = r0 + MG_TR
        if r1 < nn:
            comptime for v in range(MG_LG):
                var u = v * MG_NT + tid
                var cc = u % (2 * MG_TC)
                reg[v] = _augv(x, y, nn, dd, tn, r1 + u // (2 * MG_TC), cj + cc if cc < MG_TC else ck + cc - MG_TC)
        if live:
            var cnt = Int32(min(MG_TR, nn - r0))
            var pj = xs + a
            var pk = xs + b
            var r = Int32(0)
            while r + MG_RU <= cnt:
                var bj = SIMD[DType.float32, MG_RU]()
                var bk = SIMD[DType.float32, MG_RU]()
                comptime for u in range(MG_RU):
                    bj[u] = pj[u * 2 * MG_TC]
                    bk[u] = pk[u * 2 * MG_TC]
                comptime for u in range(MG_RU):
                    acc = _acc_fmad(fs(bj[u], mj), fs(bk[u], mk), acc)
                pj += MG_RU * 2 * MG_TC
                pk += MG_RU * 2 * MG_TC
                r += MG_RU
            while r < cnt:
                acc = _acc_fmad(fs(pj[0], mj), fs(pk[0], mk), acc)
                pj += 2 * MG_TC
                pk += 2 * MG_TC
                r += 1
        r0 = r1
    if live:
        if k < dd:
            st(fw, Int(gg) + j * dd + k, acc)
            st(fw, Int(gg) + k * dd + j, acc)
        else:
            st(fw, Int(xty) + (k - dd) * dd + j, acc)
    witness_end(wf, woff, nonce)

def mg_tiles(d: Int, t_n: Int) -> Int:
    return (d + t_n + MG_TC - 1) // MG_TC
