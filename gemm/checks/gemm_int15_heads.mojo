# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int15i64.v1` over ATTENTION HEADS in one launch:
the score product S11 of every (batch, head) at once.

Lane lane/lowbit-default, 2026-09-29 (the maintainer's lever 1: the
resident decode step under numeric_profile="fixed15_v1" spent its time in
one gather, two quantizer launches, one product and one scatter PER HEAD).

    C[bb, h] (l x s) = A[bb, h] (l x k) . B[bb, h // n_rep] (s x k)^T

A is `b * nh * l` rows of `k` codes as planes, head-major: row
`(bb * nh + h) * l + t`, one exponent per row. B is `b * nkv * s` rows,
row `(bb * nkv + kvh) * s + j`. C is `[b, nh, l, s]`, row-major. `repeat_kv`
is the index map `kvh = h // n_rep`, never a copy.

NO NEW ARITHMETIC. Each cell's three sums (`HH`, `HL + LH`, `LL`) are the
exact Int32 sums of the planes' products over `k`, in any order (exact
integers: the order cannot change a bit, clause W-5), and the cell is
stored by `int15_store_cell`, the function every plan of the profile
stores through, with the row's and the key's exponents (W-7). So a cell is
the reference plan's cell of the same (batch, head) product, bit for bit,
and every sabotage arm of the epilogue reaches it.

THE SIZE IT IS FOR: one thread per cell and a plain `k` loop, which is
right for the decode rows (`l <= INT15_HEADS_MAX_L`, `k` = head_dim) and not
for a prefill's `l x s` blocks, where the per-head matrix-unit plans stay.
The entry refuses a larger `l` by name.

The gate is `gemm/checks/gemm_int15_heads_check.mojo` (`pixi run
check-gemm-int15-heads`, with the value, epilogue and exponent arms).
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gemm.checks.gemm_int15 import INT15_TPB
from gemm.checks.gemm_int15_epilogue import int15_store_cell
from gemm.host.gemm_int15_oracle import INT15_MAX_K

#: The most query rows per head the one-launch form takes (the decode rows).
comptime INT15_HEADS_MAX_L = 16


def int15_heads_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    l_in: Int32,
    s_in: Int32,
    k_in: Int32,
):
    """One thread per cell `(bb, h, t, j)` of `C`."""
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var l = Int(l_in)
    var s = Int(s_in)
    var k = Int(k_in)
    var cells = Int(b_in) * nh * l * s
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= cells:
        return
    var j = cell % s
    var rest = cell // s
    var t = rest % l
    var bh_idx = rest // l  # bb * nh + h
    var h = bh_idx % nh
    var bb = bh_idx // nh
    var kvh = h // (nh // nkv)
    var arow = bh_idx * l + t
    var brow = (bb * nkv + kvh) * s + j
    var hh = Int32(0)
    var mid = Int32(0)
    var ll = Int32(0)
    var pa = arow * k
    var pb = brow * k
    for p in range(k):
        var a_h = Int32(ah.unsafe_load(pa + p))
        var a_l = Int32(al.unsafe_load(pa + p))
        var b_h = Int32(bh.unsafe_load(pb + p))
        var b_l = Int32(bl.unsafe_load(pb + p))
        hh += a_h * b_h
        mid += a_h * b_l + a_l * b_h
        ll += a_l * b_l
    # the (bb, h) product's own l x s block, its rows' and its keys' exponents
    int15_store_cell(
        c + bh_idx * l * s, ea + bh_idx * l, eb + (bb * nkv + kvh) * s,
        hh, mid, ll, t, j, l, s,
    )


def identical_gemm_int15_heads_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    b: Int,
    nh: Int,
    nkv: Int,
    l: Int,
    s: Int,
    k: Int,
) raises:
    """THE ENTRY POINT: every (batch, head) score product in one launch.
    Asynchronous. Shapes are refused by name before anything is enqueued."""
    if b < 1 or nh < 1 or nkv < 1 or l < 1 or s < 1 or k < 1:
        raise Error(
            "identical_gemm_int15_heads: b, n_heads, n_kv_heads, l, s and k must be positive, got b="
            + String(b) + " nh=" + String(nh) + " nkv=" + String(nkv) + " l=" + String(l)
            + " s=" + String(s) + " k=" + String(k)
        )
    if nh % nkv != 0:
        raise Error(
            "identical_gemm_int15_heads: n_heads " + String(nh) + " is not a multiple of n_kv_heads "
            + String(nkv)
        )
    if l > INT15_HEADS_MAX_L:
        raise Error(
            "identical_gemm_int15_heads: l=" + String(l) + " is above " + String(INT15_HEADS_MAX_L)
            + ", the rows this one-launch form is for; the per-head plans take it"
        )
    if k > INT15_MAX_K:
        raise Error(
            "identical_gemm_int15_heads: k must be at most " + String(INT15_MAX_K)
            + " (contract W-4), got " + String(k)
        )
    var cells = b * nh * l * s
    if (
        len(c) < cells or len(ah) < b * nh * l * k or len(al) < b * nh * l * k or len(ea) < b * nh * l
        or len(bh) < b * nkv * s * k or len(bl) < b * nkv * s * k or len(eb) < b * nkv * s
    ):
        raise Error("identical_gemm_int15_heads: a buffer is shorter than the shape")
    ctx.enqueue_function[int15_heads_kernel](
        c.unsafe_ptr(), ah.unsafe_ptr(), al.unsafe_ptr(), ea.unsafe_ptr(),
        bh.unsafe_ptr(), bl.unsafe_ptr(), eb.unsafe_ptr(),
        Int32(b), Int32(nh), Int32(nkv), Int32(l), Int32(s), Int32(k),
        grid_dim=((cells + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )
