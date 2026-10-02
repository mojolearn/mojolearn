# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The kernels of x_decomp/lle_local.mojo's cells (cgr-decomp, 2026-10-03):
one thread per output, or per sample for the k-sized serial steps. The
driver is `DevExec.lle_local` (x_decomp/device.mojo)."""
from std.gpu import block_dim, block_idx, thread_idx

from x_decomp.cells import F32Ptr
from x_decomp.lle_local import (
    hessian_cell,
    hessian_comp_cell,
    hessian_q_cell,
    lle_gram_cell,
    lle_mean_cell,
    ltsa_cell,
    mlle_eta,
    mlle_key,
    mlle_rows_cell,
    mlle_unkey,
    mlle_weights_cell,
)

comptime U32Ptr = MutPointer[UInt32, MutAnyOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def lle_mean_kernel(x: F32Ptr, idx: F32Ptr, mu: F32Ptr, n: Int32, d: Int32, nn: Int32):
    var t = _tid()
    if t < Int(n) * Int(d):
        var i = t // Int(d)
        lle_mean_cell(x, idx, mu, i, t - i * Int(d), Int(d), Int(nn))


def lle_gram_kernel(x: F32Ptr, idx: F32Ptr, c: F32Ptr, g: F32Ptr, n: Int32, d: Int32, nn: Int32):
    var k = Int(nn)
    var t = _tid()
    if t < Int(n) * k * k:
        var i = t // (k * k)
        var r = t - i * k * k
        var a = r // k
        lle_gram_cell(x, idx, c, g, i, a, r - a * k, Int(d), k)


def ltsa_kernel(v: F32Ptr, idx: F32Ptr, bmat: F32Ptr, n: Int32, nn: Int32, nc: Int32):
    var k = Int(nn)
    var t = _tid()
    if t < Int(n) * k * k:
        var i = t // (k * k)
        var r = t - i * k * k
        var a = r // k
        ltsa_cell(v, idx, bmat, i, a, r - a * k, Int(n), k, Int(nc))


def hessian_q_kernel(v: F32Ptr, q: F32Ptr, n: Int32, nn: Int32, nc: Int32):
    var i = _tid()
    if i < Int(n):
        hessian_q_cell(v, q, i, Int(nn), Int(nc))


def hessian_comp_kernel(q: F32Ptr, cmat: F32Ptr, n: Int32, nn: Int32, nc: Int32):
    var k = Int(nn)
    var t = _tid()
    if t < Int(n) * k * k:
        var i = t // (k * k)
        var r = t - i * k * k
        var a = r // k
        hessian_comp_cell(q, cmat, i, a, r - a * k, k, Int(nc))


def hessian_kernel(q: F32Ptr, vc: F32Ptr, idx: F32Ptr, bmat: F32Ptr, n: Int32, nn: Int32, nc: Int32, tol: Float32):
    var ncol = Int(nn) - 1 - Int(nc)
    var t = _tid()
    if t < Int(n) * ncol:
        var i = t // ncol
        hessian_cell(q, vc, idx, bmat, i, t - i * ncol, Int(n), Int(nn), Int(nc), tol)


def mlle_weights_kernel(
    w: F32Ptr, v: F32Ptr, wreg: F32Ptr, rho: F32Ptr, scr: F32Ptr, n: Int32, nn: Int32, nev: Int32, nc: Int32
):
    var i = _tid()
    if i < Int(n):
        mlle_weights_cell(w, v, wreg, rho, scr, i, Int(nn), Int(nev), Int(nc))


def mlle_key_kernel(rho: F32Ptr, keys: U32Ptr, vals: U32Ptr, n: Int32):
    """rho's order-preserving uint32 keys (the radix sort's input)."""
    var i = _tid()
    if i < Int(n):
        keys.unsafe_store(i, mlle_key(rho.unsafe_load(i)))
        vals.unsafe_store(i, UInt32(i))


def mlle_unkey_kernel(keys: U32Ptr, srt: F32Ptr, n: Int32):
    var i = _tid()
    if i < Int(n):
        srt.unsafe_store(i, mlle_unkey(keys.unsafe_load(i)))


def mlle_rows_kernel(
    w: F32Ptr, v: F32Ptr, wreg: F32Ptr, idx: F32Ptr, bmat: F32Ptr, scr: F32Ptr, srt: F32Ptr, n: Int32,
    nn: Int32, nev: Int32, tol: Float32,
):
    var i = _tid()
    if i < Int(n):
        var eta = mlle_eta(srt, Int(n))
        mlle_rows_cell(w, v, wreg, idx, bmat, scr, eta, i, Int(n), Int(nn), Int(nev), tol)
