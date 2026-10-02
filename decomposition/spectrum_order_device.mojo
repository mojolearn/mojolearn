# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The spectrum's order on the device (cpu-gpu-cleanup c-decomp, 2026-10-02):
`decomposition/host/linalg_public.mojo::eigh_ascending` and
`svdvals_descending` ran on the host after every device eigh / svdvals,
the values and vectors downloaded, permuted and copied out on the host.

THE ORDER is `_argsort_desc`'s (the host twin keeps the same definition):
descending by `spectrum_key` (the float bits made monotone, -0.0 keyed as
+0.0), ties to the LOWER index, a strict total order, so each value's
position is its rank: the number of values before it in that order. One
thread per value forms its rank (comparisons only), then one thread per
cell scatters: eigh's ascending position is n - 1 - rank (descending,
walked backwards), svdvals' is the rank. No arithmetic: the same words."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]
comptime _TPB = 128


@always_inline
def spectrum_key(v: Float32) -> UInt32:
    """Monotone in the float order (-0.0 keyed as +0.0)."""
    var b = bitcast[DType.uint32](v)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    return b ^ UInt32(0xFFFFFFFF) if (b >> 31) == 1 else b | UInt32(0x80000000)


@always_inline
def spectrum_rank_desc(key: _F, n: Int, i: Int) -> Int:
    """Values ahead of i in the descending order (ties: lower index first)."""
    var ki = spectrum_key(key.unsafe_load(i))
    var r = 0
    for j in range(n):
        var kj = spectrum_key(key.unsafe_load(j))
        if kj > ki or (kj == ki and j < i):
            r += 1
    return r


def _rank_kernel(key: _F, n: Int32, pos: _I):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        pos.unsafe_store(i, Int32(spectrum_rank_desc(key, Int(n), i)))


def _eigh_scatter_kernel(key: _F, kstride: Int32, vecs: _F, pos: _I, n: Int32, w_out: _F, v_out: _F):
    """Cell t = r * n + i: column i of `vecs` to column n - 1 - pos[i];
    row 0's thread also moves the value."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    if t < nn * nn:
        var r = t // nn
        var i = t - r * nn
        var c = nn - 1 - Int(pos.unsafe_load(i))
        v_out.unsafe_store(r * nn + c, vecs.unsafe_load(r * nn + i))
        if r == 0:
            w_out.unsafe_store(c, key.unsafe_load(i * Int(kstride)))


def _desc_scatter_kernel(key: _F, pos: _I, n: Int32, out: _F):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        out.unsafe_store(Int(pos.unsafe_load(i)), key.unsafe_load(i))


def _gather_diag_kernel(a: _F, n: Int32, diag: _F):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        diag.unsafe_store(i, a.unsafe_load(i * Int(n) + i))


def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


def enqueue_eigh_ascending(
    ctx: DeviceContext, diag: _F, diag_stride: Int, vecs: _F, n: Int, mut pos: DeviceBuffer[DType.int32],
    w_out: _F, v_out: _F,
) raises:
    """`eigh_ascending` on device pointers, enqueued: the eigenvalues at
    diag[i * diag_stride] (diag_stride n + 1 reads a matrix's diagonal), the
    sign-flipped basis `vecs` (eigenvector i in column i), into device
    `w_out` (n) and `v_out` (n x n, not `vecs`). `pos` holds n int32."""
    if n <= 0:
        return
    var dk = ctx.enqueue_create_buffer[DType.float32](n)
    var kp = dk.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if diag_stride == 1:
        ctx.enqueue_function[_desc_copy_kernel](diag, Int32(n), kp, grid_dim=_blocks(n), block_dim=_TPB)
    else:
        ctx.enqueue_function[_gather_diag_kernel](diag, Int32(n), kp, grid_dim=_blocks(n), block_dim=_TPB)
    var pp = pos.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[_rank_kernel](kp, Int32(n), pp, grid_dim=_blocks(n), block_dim=_TPB)
    ctx.enqueue_function[_eigh_scatter_kernel](
        kp, Int32(1), vecs, pp, Int32(n), w_out, v_out, grid_dim=_blocks(n * n), block_dim=_TPB
    )
    ctx.synchronize()
    _ = dk^


def _desc_copy_kernel(src: _F, n: Int32, dst: _F):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst.unsafe_store(i, src.unsafe_load(i))


def enqueue_svdvals_descending(
    ctx: DeviceContext, s: _F, n: Int, mut pos: DeviceBuffer[DType.int32], out: _F
) raises:
    """`svdvals_descending` on device pointers, enqueued: `s` (n) into device
    `out` (n, not `s`), descending."""
    if n <= 0:
        return
    var pp = pos.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[_rank_kernel](s, Int32(n), pp, grid_dim=_blocks(n), block_dim=_TPB)
    ctx.enqueue_function[_desc_scatter_kernel](s, pp, Int32(n), out, grid_dim=_blocks(n), block_dim=_TPB)
