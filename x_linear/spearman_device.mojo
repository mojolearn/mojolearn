# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`x_linear_spearman_sign` on the device (lane cpu2-l10-linear, 2026-10-04).

The exact sign of x_linear/spearman.mojo as grid launches, one
witness-guarded unit (every output is rebuilt from the uploaded x and y):

  per column (x, then y):
    keys     one thread per row: `sp_key`, the identity permutation
    sort     the isotonic block radix sort (x_linear/device.mojo
             rs_count / rs_scan / rs_scatter), eight 4-bit passes, stable
    starts   one thread per SP_CH sorted positions counts the tie-group
             starts; one block scans the counts (`iso_scan1_kernel`); the
             chunk threads write each group's first position and every
             position's group id
    ranks    one thread per sorted position: rank2[row] = first + last
  then
    blocks   one thread per FOLD_BLOCK rows: `sp_block_sum` (Int64)
    sign     one thread: `sp_fold_sign` over the block sums

One int32 word comes back: the sign.
"""

from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from x_linear.ops import FP, IP, ld, ldi, sti
from x_linear.tops import fold_blocks
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.scan import SC_NT
from x_linear.spearman import I64P, sp_key, sp_block_sum, sp_fold_sign
from x_linear.device import (
    rs_count_kernel, rs_scan_kernel, rs_scatter_kernel, iso_scan1_kernel, RS_BITS, RS_D, RS_NT, RS_TILE, RS_BYTES,
)

comptime SP_TPB = 256
#: sorted positions one chunk thread walks when it finds the group starts
comptime SP_CH = 256


def _sp_blocks(count: Int) -> Int:
    return max((count + SP_TPB - 1) // SP_TPB, 1)


@always_inline
def _sp_start(keys: IP, perm: IP, p: Int) -> Bool:
    """Sorted position p opens a tie group."""
    if p == 0:
        return True
    return ldi(keys, ldi(perm, p)) != ldi(keys, ldi(perm, p - 1))


def sp_keys_kernel(v: FP, n: Int32, keys: IP, perm: IP, wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * SP_TPB + Int(thread_idx.x)
    if i < Int(n):
        keys.unsafe_store(i, sp_key(ld(v, i)))
        sti(perm, i, i)
    witness_end(wf, woff, nonce)


def sp_start_count_kernel(keys: IP, perm: IP, n: Int32, nch: Int32, bc: IP, wf: IP, woff: Int32, nonce: Int32):
    var c = Int(block_idx.x) * SP_TPB + Int(thread_idx.x)
    if c < Int(nch):
        var lo = c * SP_CH
        var hi = min(lo + SP_CH, Int(n))
        var cnt = 0
        for p in range(lo, hi):
            if _sp_start(keys, perm, p):
                cnt += 1
        sti(bc, c, cnt)
    witness_end(wf, woff, nonce)


def sp_start_write_kernel(keys: IP, perm: IP, n: Int32, nch: Int32, bc: IP, gs: IP, gid: IP,
                          wf: IP, woff: Int32, nonce: Int32):
    """bc: the exclusive prefix of the chunk counts. gs[g] = group g's first
    sorted position (gs[G] = n), gid[p] = position p's group."""
    var c = Int(block_idx.x) * SP_TPB + Int(thread_idx.x)
    if c < Int(nch):
        var lo = c * SP_CH
        var hi = min(lo + SP_CH, Int(n))
        var off = ldi(bc, c)
        for p in range(lo, hi):
            if _sp_start(keys, perm, p):
                sti(gs, off, p)
                off += 1
            sti(gid, p, off - 1)
        if c == Int(nch) - 1:
            sti(gs, off, Int(n))
    witness_end(wf, woff, nonce)


def sp_rank_kernel(perm: IP, n: Int32, gs: IP, gid: IP, rank2: IP, wf: IP, woff: Int32, nonce: Int32):
    var p = Int(block_idx.x) * SP_TPB + Int(thread_idx.x)
    if p < Int(n):
        var g = ldi(gid, p)
        sti(rank2, ldi(perm, p), ldi(gs, g) + ldi(gs, g + 1) - 1)
    witness_end(wf, woff, nonce)


def sp_block_kernel(rx: IP, ry: IP, n: Int32, nb: Int32, parts: I64P, wf: IP, woff: Int32, nonce: Int32):
    var b = Int(block_idx.x) * SP_TPB + Int(thread_idx.x)
    if b < Int(nb):
        parts.unsafe_store(b, sp_block_sum(rx, ry, Int(n), b))
    witness_end(wf, woff, nonce)


def sp_sign_kernel(parts: I64P, nb: Int32, out: IP, wf: IP, woff: Int32, nonce: Int32):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        sti(out, 0, sp_fold_sign(parts, Int(nb)))
    witness_end(wf, woff, nonce)


def spearman_sign_device(var ctx: DeviceContext, x: FP, y: FP, n: Int) raises -> Int:
    """The GPU column of `spearman_sign_host` (x_linear/spearman.mojo): x, y
    host addresses of n float32 each. Returns -1, 0 or 1."""
    comptime assert lib_smem_page_fits_for[TARGET_COLUMN, RS_BYTES](), "the block radix page must fit"
    # one buffer per column: a kernel argument is a buffer's own pointer,
    # never an offset into one (Metal)
    var dvx = ctx.enqueue_create_buffer[DType.float32](n)
    var dvy = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=dvx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dvy, src_ptr=y)
    var dkey = ctx.enqueue_create_buffer[DType.int32](n)
    var dpa = ctx.enqueue_create_buffer[DType.int32](n)
    var dpb = ctx.enqueue_create_buffer[DType.int32](n)
    var nb_rs = max((n + RS_TILE - 1) // RS_TILE, 1)
    var dbt = ctx.enqueue_create_buffer[DType.int32](RS_D * nb_rs)
    var dbo = ctx.enqueue_create_buffer[DType.int32](RS_D * nb_rs)
    var dfl = ctx.enqueue_create_buffer[DType.int32](RS_D)
    var nch = (n + SP_CH - 1) // SP_CH
    var dbc = ctx.enqueue_create_buffer[DType.int32](nch + 1)
    var dtot = ctx.enqueue_create_buffer[DType.int32](1)
    var dgs = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var dgid = ctx.enqueue_create_buffer[DType.int32](n)
    var drx = ctx.enqueue_create_buffer[DType.int32](n)
    var dry = ctx.enqueue_create_buffer[DType.int32](n)
    var nb = fold_blocks(n)
    var dparts = ctx.enqueue_create_buffer[DType.int64](max(nb, 1))
    var dout = ctx.enqueue_create_buffer[DType.int32](1)
    var g_rows = _sp_blocks(n)
    var g_ch = _sp_blocks(nch)
    var npass = 32 // RS_BITS
    var per_col = g_rows + npass * (2 * nb_rs + RS_D) + g_ch + 1 + g_ch + g_rows
    var wit = Witness(ctx, 2 * per_col + _sp_blocks(nb) + 1 + 1)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        for col in range(2):
            var vp = FP(unsafe_from_address=Int(dvx.unsafe_ptr()) if col == 0 else Int(dvy.unsafe_ptr()))
            var rk = IP(unsafe_from_address=Int(drx.unsafe_ptr()) if col == 0 else Int(dry.unsafe_ptr()))
            ctx.enqueue_function[sp_keys_kernel](vp, Int32(n), dkey.unsafe_ptr(), dpa.unsafe_ptr(),
                                                 wit.p(), Int32(wo), nonce, grid_dim=g_rows, block_dim=SP_TPB)
            wo += g_rows
            var cur_a = True
            for pas in range(npass):
                var shift = Int32(RS_BITS * pas)
                var src = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
                var dst = IP(unsafe_from_address=Int(dpb.unsafe_ptr()) if cur_a else Int(dpa.unsafe_ptr()))
                ctx.enqueue_function[rs_count_kernel](dkey.unsafe_ptr(), src, Int32(n), shift, dbt.unsafe_ptr(), Int32(nb_rs),
                                                      wit.p(), Int32(wo), nonce, grid_dim=nb_rs, block_dim=RS_NT)
                wo += nb_rs
                ctx.enqueue_function[rs_scan_kernel](dbt.unsafe_ptr(), Int32(nb_rs), Int32(n), dbo.unsafe_ptr(), dfl.unsafe_ptr(),
                                                     wit.p(), Int32(wo), nonce, grid_dim=RS_D, block_dim=SC_NT)
                wo += RS_D
                ctx.enqueue_function[rs_scatter_kernel](dkey.unsafe_ptr(), src, dst, Int32(n), shift, dbo.unsafe_ptr(),
                                                        Int32(nb_rs), dfl.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                        grid_dim=nb_rs, block_dim=RS_NT)
                wo += nb_rs
                cur_a = not cur_a
            var pp = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
            ctx.enqueue_function[sp_start_count_kernel](dkey.unsafe_ptr(), pp, Int32(n), Int32(nch), dbc.unsafe_ptr(),
                                                        wit.p(), Int32(wo), nonce, grid_dim=g_ch, block_dim=SP_TPB)
            wo += g_ch
            ctx.enqueue_function[iso_scan1_kernel](dbc.unsafe_ptr(), Int32(nch), dtot.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                   grid_dim=1, block_dim=SC_NT)
            wo += 1
            ctx.enqueue_function[sp_start_write_kernel](dkey.unsafe_ptr(), pp, Int32(n), Int32(nch), dbc.unsafe_ptr(),
                                                        dgs.unsafe_ptr(), dgid.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                        grid_dim=g_ch, block_dim=SP_TPB)
            wo += g_ch
            ctx.enqueue_function[sp_rank_kernel](pp, Int32(n), dgs.unsafe_ptr(), dgid.unsafe_ptr(), rk,
                                                 wit.p(), Int32(wo), nonce, grid_dim=g_rows, block_dim=SP_TPB)
            wo += g_rows
        ctx.enqueue_function[sp_block_kernel](drx.unsafe_ptr(), dry.unsafe_ptr(), Int32(n), Int32(nb), dparts.unsafe_ptr(),
                                              wit.p(), Int32(wo), nonce, grid_dim=_sp_blocks(nb), block_dim=SP_TPB)
        wo += _sp_blocks(nb)
        ctx.enqueue_function[sp_sign_kernel](dparts.unsafe_ptr(), Int32(nb), dout.unsafe_ptr(),
                                             wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=SP_TPB)
        wo += 1
        if wit.ok(ctx, wo, "spearman sign"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    var h = List[Int32](length=1, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()
    var sign = Int(h[0])
    _ = h^
    _ = dvx^
    _ = dvy^
    _ = dkey^
    _ = dpa^
    _ = dpb^
    _ = dbt^
    _ = dbo^
    _ = dfl^
    _ = dbc^
    _ = dtot^
    _ = dgs^
    _ = dgid^
    _ = drx^
    _ = dry^
    _ = dparts^
    _ = dout^
    _ = wit^
    return sign
