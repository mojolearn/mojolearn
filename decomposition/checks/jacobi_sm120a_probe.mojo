# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2711 (2026-09-14): the per-sweep probe for the sm_120a Jacobi refusals.

THE FINDING. The first identity leg on an RTX 5090 (Blackwell consumer,
`sm_120a`, commit 6796ceff9) carried the same bits as the Apple M4, the H100
and the MI300X on 410 of 414 training cells and on every inference and model
cell, and refused four: pca, tsvd, ols and ridge on the `odd` fixture
(12345 x 17 standard normals, `tools/identity_break.py`), every one of them
through `jacobi_eigh_kernel` on the 17 x 17 covariance or Gram. The reported
off-diagonal ratios after 15 sweeps were 0.0027 (pca), 0.0012 (tsvd), 0.505
(ols, equilibrated Gram) and 0.0027 (ridge). Every one of those matrices
STARTS at 0.0343 (numpy, Float64), so 0.0027 is a 13x reduction where the
three recorded vendors reach 1e-7 in a handful of sweeps, and 0.505 is 15x
WORSE than the input. That is a gross failure, not a last-bit one.

WHAT THIS PROBE SEPARATES. Two stages feed the refusal and the leg's message
cannot tell them apart:

  1. THE GRAM. `odd` is the only fixture whose width is not a multiple of 4
     and not divisible by 2, so it is the only fixture that takes the
     split-K Gram kernel's SCALAR staging arm and STRIDED-SINGLES ownership
     arm (`core/gram_splitk.mojo`, `gram_splitk_stage_vectorized(17)` and
     `gram_splitk_reg_tiled[4](17)` are both False). No other cell on the
     5090 exercised those arms. A Gram that is wrong or non-symmetric makes
     the two-sided Jacobi plateau exactly as reported.
  2. THE JACOBI. At n = 17 the kernel does nothing it does not do at 16
     except that lane 16 carries a row; the fold strides by 32 either way.

So this probe builds the matrices exactly as the four fits do (the same
`compute_covariance`, the same `gemm_tn`, the same equilibration kernels),
prints each one's hash, its BITWISE asymmetry count and its error against a
Float64 host product, then runs the SHIPPED kernel and a copy of it that
dumps the working matrix and basis at the top of every sweep, and prints per
sweep the device fold's `off`, the host-recomputed off-diagonal ratio and a
hash of `a` and of `v`. Two controls run in the same process: the Float64
host covariance rounded to fp32 (Jacobi with a device Gram out of the
picture) and the 16 x 16 leading block of the device covariance (the width
the 5090 converged at). The probe kernel is held equal to the shipped kernel
bit for bit at the end of every case, so its per-sweep lines ARE the shipped
trajectory.

RUN. `mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . decomposition/checks/jacobi_sm120a_probe.mojo <odd_x.f32>`
where `<odd_x.f32>` is `tools/identity_break.py`'s `fixture("odd")[0]`
written with `X.tofile(...)` (12345 x 17 float32 row-major, 839,460 bytes,
sha256 595dda3a45cf8a3e...).
"""

from max.gpu import thread_idx
from std.math import sqrt
from std.memory import bitcast, unsafe_memcpy
from std.sys import argv
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import (
    ftz,
    identical_mul_add,
    identical_sqrt,
    numeric_mode_name,
)
from core.gemm import gemm_tn
from core.gram_splitk import (
    GRAM_STRIDED_ARM,
    GRAM_TPB,
    _splitk_chunk_rows,
    gram_splitk_chunk_count,
    gram_splitk_partial_kernel_arm,
    gram_splitk_reduce_kernel,
)
from core.column_stats import diagonal_to_vector_kernel
from decomposition.impl.linalg.detail.pca import compute_covariance
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_ROT_TPB,
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
    _fold_lead_lanes_and_broadcast,
    _rot_add,
    _rot_sub,
    _rotate_pair_block,
    jacobi_eigh_kernel,
    jacobi_rotation_cs,
)
from glm.impl.linalg.detail.lstsq import ols_equilibration_scale
from glm.impl.matrix.math import (
    MATRIX_ELEM_TPB,
    matrix_vector_binary_mult_kernel,
    row_vector_binary_mult_kernel,
)


comptime ODD_ROWS = 12345
comptime ODD_COLS = 17


# ---------------------------------------------------------------------------
# The probe kernel: `jacobi_eigh_kernel` with a dump at the top of every sweep
# ---------------------------------------------------------------------------


def jacobi_probe_kernel[rot_tpb: Int](
    a_io: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    dump: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """`jacobi_eigh_kernel[rot_tpb]`, statement for statement, plus a dump.

    Slot `s` of `dump` (stride `2 n^2 + 1`) holds, at the top of sweep `s`
    after the off-diagonal fold: the fold's `off`, then `a`, then `v`. The
    slot is written between the fold's trailing barrier and the first
    rotation's barrier, where nothing writes `a` or `v`, so it is a pure
    read and the arithmetic below is the shipped kernel's. After the loop
    one more fold and dump land in slot `executed` so the final state is
    visible when the sweep cap is hit; that fold does not touch `info`.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var nn = n * n
    var stride = 2 * nn + 1

    var a = a_io
    var v = v_out
    var rot = stack_allocation[
        2,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var idx = tid
    while idx < nn:
        var r = idx // n
        var cc = idx % n
        v.unsafe_store(idx, Float32(1.0) if r == cc else Float32(0.0))
        idx += rot_tpb
    barrier()

    var local_f = Float32(0.0)
    if tid < JACOBI_TPB:
        var fe = tid
        while fe < nn:
            var fv = ftz(a.unsafe_load(fe))
            local_f = ftz(identical_mul_add(fv, fv, local_f))
            fe += JACOBI_TPB
    var fro2 = _fold_lead_lanes_and_broadcast[JACOBI_TPB, rot_tpb](local_f)
    var limit = ftz(ftz(tol_in * tol_in) * fro2)

    var executed = 0
    var converged = False
    var last_off = Float32(0.0)

    for _sweep in range(Int(max_sweeps_in)):
        var local_off = Float32(0.0)
        if tid < JACOBI_TPB:
            var e = tid
            while e < nn:
                var i = e // n
                var j = e - i * n
                if j > i:
                    var av = ftz(a.unsafe_load(e))
                    local_off = ftz(identical_mul_add(av, av, local_off))
                e += JACOBI_TPB
        var off = _fold_lead_lanes_and_broadcast[JACOBI_TPB, rot_tpb](
            local_off
        )
        last_off = off

        # THE DUMP. Reads only; the fold above ended in a barrier, so every
        # write of the previous sweep is visible, and the first rotation's
        # writes sit behind a barrier every lane reaches after this loop.
        var d = _sweep * stride
        if tid == 0:
            dump.unsafe_store(d, off)
        var ci = tid
        while ci < nn:
            dump.unsafe_store(d + 1 + ci, a.unsafe_load(ci))
            dump.unsafe_store(d + 1 + nn + ci, v.unsafe_load(ci))
            ci += rot_tpb
        barrier()

        if Float32(2.0) * off <= limit:
            converged = True
            break
        executed += 1

        for p in range(n):
            for q in range(p + 1, n):
                if tid == 0:
                    var cs = jacobi_rotation_cs(
                        a.unsafe_load(p * n + p),
                        a.unsafe_load(q * n + q),
                        a.unsafe_load(p * n + q),
                    )
                    rot[0] = cs[0]
                    rot[1] = cs[1]
                barrier()

                var c = rot[0]
                var s = rot[1]

                var k = tid
                while k < n:
                    if k != p and k != q:
                        var akp = ftz(a.unsafe_load(k * n + p))
                        var akq = ftz(a.unsafe_load(k * n + q))
                        a.unsafe_store(k * n + p, _rot_sub(c, akp, s, akq))
                        a.unsafe_store(k * n + q, _rot_add(s, akp, c, akq))
                        var apk = ftz(a.unsafe_load(p * n + k))
                        var aqk = ftz(a.unsafe_load(q * n + k))
                        a.unsafe_store(p * n + k, _rot_sub(c, apk, s, aqk))
                        a.unsafe_store(q * n + k, _rot_add(s, apk, c, aqk))
                    elif k == p:
                        _rotate_pair_block(a, n, p, q, c, s)
                    var vkp = ftz(v.unsafe_load(k * n + p))
                    var vkq = ftz(v.unsafe_load(k * n + q))
                    v.unsafe_store(k * n + p, _rot_sub(c, vkp, s, vkq))
                    v.unsafe_store(k * n + q, _rot_add(s, vkp, c, vkq))
                    k += rot_tpb
                barrier()

    if tid == 0:
        info_out.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        var rel = Float32(0.0)
        if fro2 > Float32(0.0):
            rel = ftz(
                identical_sqrt(ftz(ftz(Float32(2.0) * last_off) / fro2))
            )
        info_out.unsafe_store(1, rel)
        info_out.unsafe_store(2, Float32(executed))
    barrier()

    # The final state, for the sweep-cap exit: the same fold, into slot
    # `executed` (a converged exit already wrote that slot with the same
    # bytes at the top of the last sweep, and rewrites it identically).
    var tail_off = Float32(0.0)
    if tid < JACOBI_TPB:
        var e2 = tid
        while e2 < nn:
            var i2 = e2 // n
            var j2 = e2 - i2 * n
            if j2 > i2:
                var av2 = ftz(a.unsafe_load(e2))
                tail_off = ftz(identical_mul_add(av2, av2, tail_off))
            e2 += JACOBI_TPB
    var off_tail = _fold_lead_lanes_and_broadcast[JACOBI_TPB, rot_tpb](
        tail_off
    )
    var dt = executed * stride
    if tid == 0:
        dump.unsafe_store(dt, off_tail)
    var ct = tid
    while ct < nn:
        dump.unsafe_store(dt + 1 + ct, a.unsafe_load(ct))
        dump.unsafe_store(dt + 1 + nn + ct, v.unsafe_load(ct))
        ct += rot_tpb


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _fnv1a64(x: List[Float32], start: Int, count: Int) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(start, start + count):
        var u = bitcast[DType.uint32](x[i])
        for k in range(4):
            h = (
                h ^ UInt64((u >> UInt32(8 * k)) & UInt32(0xFF))
            ) * UInt64(0x100000001B3)
    return h


def _hex64(h: UInt64) -> String:
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(16):
        var nib = Int((h >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _hex32(x: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](x)
    var out = String("")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _read_f32(path: String, n: Int) raises -> List[Float32]:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) != n * 4:
        raise Error(
            path
            + " holds "
            + String(len(bytes))
            + " bytes, expected "
            + String(n * 4)
            + " (12345 x 17 float32, tools/identity_break.py fixture('odd'))"
        )
    var out = List[Float32](length=n, fill=Float32(0.0))
    unsafe_memcpy(
        dest=out.unsafe_ptr().bitcast[UInt8](),
        src=bytes.unsafe_ptr(),
        count=n * 4,
    )
    return out^


def _host_rel_offdiag(a: List[Float32], start: Int, n: Int) -> Float64:
    """`||offdiag(A)||_F / ||A||_F` recomputed in Float64 from the dumped bits."""
    var off = Float64(0.0)
    var all = Float64(0.0)
    for i in range(n):
        for j in range(n):
            var x = Float64(a[start + i * n + j])
            all += x * x
            if j > i:
                off += x * x
    if all == 0.0:
        return 0.0
    return sqrt(2.0 * off / all)


def _cells_line(tag: String, a: List[Float32], start: Int, n: Int):
    var s = String("CELLS ") + tag + " n=" + String(n) + " bits="
    for i in range(n * n):
        if i > 0:
            s += ","
        s += _hex32(a[start + i])
    print(s)


def _to_device(
    ctx: DeviceContext, a: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](len(a))
    var h = ctx.enqueue_create_host_buffer[DType.float32](len(a))
    ctx.synchronize()
    for i in range(len(a)):
        h.unsafe_ptr().unsafe_store(i, a[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return buf^


def _to_host(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], count: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(count):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def _matrix_report(
    tag: String, a: List[Float32], n: Int, ref64: List[Float64]
) raises:
    """Hash, BITWISE asymmetry count and the error against a Float64 host product."""
    var asym = 0
    var worst_asym = Float64(0.0)
    var worst_err = Float64(0.0)
    var ref_max = Float64(0.0)
    for i in range(n):
        for j in range(n):
            var x = a[i * n + j]
            var y = a[j * n + i]
            if bitcast[DType.uint32](x) != bitcast[DType.uint32](y):
                asym += 1
                var dxy = abs(Float64(x) - Float64(y))
                if dxy > worst_asym:
                    worst_asym = dxy
            var r = ref64[i * n + j]
            if abs(r) > ref_max:
                ref_max = abs(r)
            var e = abs(Float64(x) - r)
            if e > worst_err:
                worst_err = e
    print(
        "MATRIX",
        tag,
        "n=" + String(n),
        "hash=" + _hex64(_fnv1a64(a, 0, n * n)),
        "asymmetric_cells=" + String(asym),
        "max_asymmetry=" + String(worst_asym),
        "max_abs_err_vs_host64=" + String(worst_err),
        "ref_max_abs=" + String(ref_max),
        "rel_offdiag_host64=" + String(_host_rel_offdiag(a, 0, n)),
    )


def _run_case(ctx: DeviceContext, tag: String, a: List[Float32], n: Int) raises:
    """The shipped kernel and the probe kernel on the same bytes; per-sweep lines."""
    var nn = n * n
    var stride = 2 * nn + 1
    var slots = JACOBI_SWEEPS + 1

    # The shipped kernel, exactly as `eig_and_truncate` / `lstsq_eig` /
    # `svd_eig` launch it.
    var a_s = _to_device(ctx, a)
    var v_s = ctx.enqueue_create_buffer[DType.float32](nn)
    var i_s = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    ctx.enqueue_function[jacobi_eigh_kernel[JACOBI_ROT_TPB]](
        a_s.unsafe_ptr(),
        v_s.unsafe_ptr(),
        i_s.unsafe_ptr(),
        Int32(n),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_ROT_TPB, 1, 1),
    )
    ctx.synchronize()
    var sa = _to_host(ctx, a_s, nn)
    var sv = _to_host(ctx, v_s, nn)
    var si = _to_host(ctx, i_s, 3)

    # The shipped kernel AGAIN on a fresh copy. A deterministic contract
    # violation (a different FMA, a different fold) repeats bit for bit; a
    # data race or an uninitialized read does not. The 5090 leg's ols and
    # tsvd cells are the same Gram up to one power-of-two scale (which the
    # Jacobi is exactly invariant to; see the Mac reference), yet they
    # reported 0.505 and 0.0012, so this line is the one that names the
    # failure's kind.
    var a_r = _to_device(ctx, a)
    var v_r = ctx.enqueue_create_buffer[DType.float32](nn)
    var i_r = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    ctx.enqueue_function[jacobi_eigh_kernel[JACOBI_ROT_TPB]](
        a_r.unsafe_ptr(),
        v_r.unsafe_ptr(),
        i_r.unsafe_ptr(),
        Int32(n),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_ROT_TPB, 1, 1),
    )
    ctx.synchronize()
    var ra = _to_host(ctx, a_r, nn)
    var rv = _to_host(ctx, v_r, nn)
    var ri = _to_host(ctx, i_r, 3)
    var repeat_moved = 0
    for i in range(nn):
        if bitcast[DType.uint32](sa[i]) != bitcast[DType.uint32](ra[i]):
            repeat_moved += 1
        if bitcast[DType.uint32](sv[i]) != bitcast[DType.uint32](rv[i]):
            repeat_moved += 1
    for i in range(3):
        if bitcast[DType.uint32](si[i]) != bitcast[DType.uint32](ri[i]):
            repeat_moved += 1

    # The probe kernel on a fresh copy of the same bytes.
    var a_p = _to_device(ctx, a)
    var v_p = ctx.enqueue_create_buffer[DType.float32](nn)
    var i_p = ctx.enqueue_create_buffer[DType.float32](3)
    var dump = ctx.enqueue_create_buffer[DType.float32](slots * stride)
    ctx.synchronize()
    dump.enqueue_fill(Float32(0.0))
    ctx.synchronize()
    ctx.enqueue_function[jacobi_probe_kernel[JACOBI_ROT_TPB]](
        a_p.unsafe_ptr(),
        v_p.unsafe_ptr(),
        i_p.unsafe_ptr(),
        dump.unsafe_ptr(),
        Int32(n),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_ROT_TPB, 1, 1),
    )
    ctx.synchronize()
    var pa = _to_host(ctx, a_p, nn)
    var pv = _to_host(ctx, v_p, nn)
    var pi = _to_host(ctx, i_p, 3)
    var hd = _to_host(ctx, dump, slots * stride)

    var moved = 0
    for i in range(nn):
        if bitcast[DType.uint32](sa[i]) != bitcast[DType.uint32](pa[i]):
            moved += 1
        if bitcast[DType.uint32](sv[i]) != bitcast[DType.uint32](pv[i]):
            moved += 1
    for i in range(3):
        if bitcast[DType.uint32](si[i]) != bitcast[DType.uint32](pi[i]):
            moved += 1

    var executed = Int(si[2])
    print(
        "PROBE case=" + tag,
        "n=" + String(n),
        "shipped_converged=" + String(Int(si[0])),
        "shipped_executed=" + String(executed),
        "shipped_rel=" + String(si[1]),
        "shipped_a_hash=" + _hex64(_fnv1a64(sa, 0, nn)),
        "shipped_v_hash=" + _hex64(_fnv1a64(sv, 0, nn)),
        "probe_equals_shipped=" + ("yes" if moved == 0 else "NO:" + String(moved)),
        "shipped_repeat_identical="
        + ("yes" if repeat_moved == 0 else "NO:" + String(repeat_moved)),
        "repeat_rel=" + String(ri[1]),
        "repeat_executed=" + String(Int(ri[2])),
    )
    var last = executed
    if last > JACOBI_SWEEPS:
        last = JACOBI_SWEEPS
    for s in range(last + 1):
        var d = s * stride
        print(
            "PROBE case=" + tag,
            "sweep=" + String(s),
            "fold_off=" + String(hd[d]),
            "fold_off_bits=" + _hex32(hd[d]),
            "rel_host64=" + String(_host_rel_offdiag(hd, d + 1, n)),
            "a_hash=" + _hex64(_fnv1a64(hd, d + 1, nn)),
            "v_hash=" + _hex64(_fnv1a64(hd, d + 1 + nn, nn)),
        )
    _cells_line(tag + " sweep=0 a", hd, 1, n)
    _cells_line(tag + " sweep=" + String(last) + " a", hd, last * stride + 1, n)
    _cells_line(tag + " sweep=" + String(last) + " v", hd, last * stride + 1 + nn, n)
    if moved != 0:
        raise Error(
            "jacobi_sm120a_probe: the probe kernel and the shipped kernel"
            " disagree on " + String(moved) + " cells of case " + tag
            + "; the per-sweep lines above are NOT the shipped trajectory"
        )


def _stage_report[ARM: Int](
    ctx: DeviceContext,
    x: List[Float32],
    mut xd: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    gemm_out: List[Float32],
) raises:
    """The split-K Gram BY STAGE, one strided arm: every chunk's partial
    against a Float64 host partial over the same rows, then the reduce.

    DEVIATION 2711 (the 5090 leg of 2026-09-14 10:50Z): the AOT sm_120a
    build wrote cells 0..32 of the 17 x 17 Gram wrong, exactly the `c = 0`
    cells of the 33 threads whose `c = 1` cell (256..288) is also live,
    and those threads' `c = 1` cells right. This prints, per arm, how many
    chunks carry a wrong partial and which cells of the first wrong chunk,
    so the leg that confirms the fix also says where the old arm breaks.
    """
    var m = n_cols
    var mn = m * m
    var n_chunks = gram_splitk_chunk_count()
    var kc = _splitk_chunk_rows(n_rows, n_chunks)
    var partials = ctx.enqueue_create_buffer[DType.float32](n_chunks * mn)
    var z = ctx.enqueue_create_buffer[DType.float32](mn)
    ctx.synchronize()
    partials.enqueue_fill(Float32(0.0))
    ctx.synchronize()
    comptime kern = gram_splitk_partial_kernel_arm[4, ARM]
    ctx.enqueue_function[kern](
        partials.unsafe_ptr(),
        xd.unsafe_ptr(),
        Int32(m),
        Int32(n_rows),
        Int32(kc),
        grid_dim=(n_chunks, 1, 1),
        block_dim=(GRAM_TPB, 1, 1),
    )
    ctx.synchronize()
    var hp = _to_host(ctx, partials, n_chunks * mn)
    var bad_chunks = 0
    var first_bad = -1
    var bad_cells_first = String("")
    var worst = Float64(0.0)
    for c in range(n_chunks):
        var r0 = c * kc
        var r1 = r0 + kc
        if r1 > n_rows:
            r1 = n_rows
        var bad_here = 0
        for i in range(m):
            for j in range(m):
                var h = Float64(0.0)
                var r = r0
                while r < r1:
                    h += Float64(x[r * m + i]) * Float64(x[r * m + j])
                    r += 1
                var d = Float64(hp[c * mn + i * m + j])
                var tol = 1.0e-3 * (abs(h) if abs(h) > 1.0 else 1.0)
                var e = abs(d - h)
                if e > worst:
                    worst = e
                if e > tol:
                    bad_here += 1
                    if first_bad < 0 or first_bad == c:
                        if bad_cells_first.byte_length() > 0:
                            bad_cells_first += ","
                        bad_cells_first += String(i * m + j)
        if bad_here > 0:
            bad_chunks += 1
            if first_bad < 0:
                first_bad = c
    ctx.enqueue_function[gram_splitk_reduce_kernel](
        z.unsafe_ptr(),
        partials.unsafe_ptr(),
        Int32(mn),
        Int32(n_chunks),
        grid_dim=((mn + GRAM_TPB - 1) // GRAM_TPB, 1, 1),
        block_dim=(GRAM_TPB, 1, 1),
    )
    ctx.synchronize()
    var hz = _to_host(ctx, z, mn)
    var moved = 0
    for i in range(mn):
        if bitcast[DType.uint32](hz[i]) != bitcast[DType.uint32](gemm_out[i]):
            moved += 1
    print(
        "STAGE arm=" + String(ARM),
        "shipped_arm=" + String(GRAM_STRIDED_ARM),
        "chunks=" + String(n_chunks),
        "chunk_rows=" + String(kc),
        "partials_hash=" + _hex64(_fnv1a64(hp, 0, n_chunks * mn)),
        "bad_chunks=" + String(bad_chunks),
        "first_bad_chunk=" + String(first_bad),
        "worst_partial_abs_err=" + String(worst),
        "bad_cells_of_first_bad_chunk=" + bad_cells_first,
        "reduce_hash=" + _hex64(_fnv1a64(hz, 0, mn)),
        "reduce_equals_gemm_tn=" + ("yes" if moved == 0 else "NO:" + String(moved)),
    )


def _host_cov64(x: List[Float32], n_rows: Int, n_cols: Int, centered: Bool, scale: Float64) -> List[Float64]:
    """Float64 host `X^T X` (or the centered covariance) for the error columns."""
    var mu = List[Float64](length=n_cols, fill=Float64(0.0))
    if centered:
        for r in range(n_rows):
            for c in range(n_cols):
                mu[c] += Float64(x[r * n_cols + c])
        for c in range(n_cols):
            mu[c] /= Float64(n_rows)
    var out = List[Float64](length=n_cols * n_cols, fill=Float64(0.0))
    for r in range(n_rows):
        for i in range(n_cols):
            var xi = Float64(x[r * n_cols + i]) - mu[i]
            for j in range(n_cols):
                var xj = Float64(x[r * n_cols + j]) - mu[j]
                out[i * n_cols + j] += xi * xj
    for i in range(n_cols * n_cols):
        out[i] *= scale
    return out^


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error(
            "usage: jacobi_sm120a_probe <odd_x.f32> (12345 x 17 float32"
            " row-major, tools/identity_break.py fixture('odd')[0].tofile)"
        )
    var path = String(args[1])
    var n_rows = ODD_ROWS
    var n_cols = ODD_COLS
    var x = _read_f32(path, n_rows * n_cols)

    var ctx = DeviceContext()
    print(
        "PROBE_DEVICE name=" + ctx.name(),
        "compute_capability=" + String(ctx.compute_capability()),
        "column=" + column_name(TARGET_COLUMN),
        "mode=" + numeric_mode_name(),
        "rot_tpb=" + String(JACOBI_ROT_TPB),
        "fold_tpb=" + String(JACOBI_TPB),
        "sweeps=" + String(JACOBI_SWEEPS),
        "tol=" + String(JACOBI_TOL),
        "x_hash=" + _hex64(_fnv1a64(x, 0, n_rows * n_cols)),
    )

    var ref_cov = _host_cov64(x, n_rows, n_cols, True, 1.0 / Float64(n_rows - 1))
    var ref_gram = _host_cov64(x, n_rows, n_cols, False, 1.0)

    # CASE 1: pca. `compute_covariance` as `pca_fit_host` calls it, then the
    # shipped Jacobi on the covariance it wrote.
    var xd = _to_device(ctx, x)
    var xa = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var cov = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    ctx.synchronize()
    compute_covariance(ctx, xd, xa, xa2, mu, cov, n_rows, n_cols, True)
    var cov_h = _to_host(ctx, cov, n_cols * n_cols)
    _matrix_report("pca.cov", cov_h, n_cols, ref_cov)
    var x_after = _to_host(ctx, xd, n_rows * n_cols)
    var restored = 0
    for i in range(n_rows * n_cols):
        if bitcast[DType.uint32](x_after[i]) != bitcast[DType.uint32](x[i]):
            restored += 1
    print("MATRIX pca.input_cells_changed_by_compute_covariance=" + String(restored))
    # The same product again: a race in the split-K Gram's scalar staging
    # or strided ownership arm (the arms only this width takes) shows here
    # as two different hashes from one input.
    var cov_again = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    ctx.synchronize()
    compute_covariance(ctx, xd, xa, xa2, mu, cov_again, n_rows, n_cols, True)
    var cov_again_h = _to_host(ctx, cov_again, n_cols * n_cols)
    var cov_moved = 0
    for i in range(n_cols * n_cols):
        if bitcast[DType.uint32](cov_h[i]) != bitcast[DType.uint32](cov_again_h[i]):
            cov_moved += 1
    print(
        "MATRIX pca.cov.repeat hash=" + _hex64(_fnv1a64(cov_again_h, 0, n_cols * n_cols)),
        "repeat_identical=" + ("yes" if cov_moved == 0 else "NO:" + String(cov_moved)),
    )
    _run_case(ctx, "pca.cov", cov_h, n_cols)

    # CASE 2: tsvd and ridge's svdEig. `gemm_tn` on the raw design, as
    # `tsvd_fit_host` and `svd_eig_traced` call it (ridge hands it the
    # host-centered design; the Gram of the raw design is tsvd's matrix and
    # the same kernel arms).
    var gram = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    ctx.synchronize()
    gemm_tn(ctx, gram, xd, xa, xa2, n_cols, n_cols, n_rows)
    ctx.synchronize()
    var gram_h = _to_host(ctx, gram, n_cols * n_cols)
    _matrix_report("tsvd.gram", gram_h, n_cols, ref_gram)
    _run_case(ctx, "tsvd.gram", gram_h, n_cols)

    # THE GRAM BY STAGE, both strided arms (DEVIATION 2711): the old
    # SIMD-lane accumulation and the per-cell scalar loop, each chunk's
    # partial against a Float64 host partial, then the reduce.
    _stage_report[0](ctx, x, xd, n_rows, n_cols, gram_h)
    _stage_report[1](ctx, x, xd, n_rows, n_cols, gram_h)

    # CASE 3: ols. The same Gram, equilibrated on the device exactly as
    # `lstsq_eig` does it (host power-of-two scales from the diagonal bits,
    # then the row and column scaling kernels).
    var gram2 = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var scale = ctx.enqueue_create_buffer[DType.float32](n_cols)
    ctx.synchronize()
    gemm_tn(ctx, gram2, xd, xa, xa2, n_cols, n_cols, n_rows)
    ctx.synchronize()
    var gram2_h = _to_host(ctx, gram2, n_cols * n_cols)
    var gram_moved = 0
    for i in range(n_cols * n_cols):
        if bitcast[DType.uint32](gram_h[i]) != bitcast[DType.uint32](gram2_h[i]):
            gram_moved += 1
    print(
        "MATRIX tsvd.gram.repeat hash=" + _hex64(_fnv1a64(gram2_h, 0, n_cols * n_cols)),
        "repeat_identical=" + ("yes" if gram_moved == 0 else "NO:" + String(gram_moved)),
    )
    var elem_tpb = MATRIX_ELEM_TPB
    ctx.enqueue_function[diagonal_to_vector_kernel](
        s_vec.unsafe_ptr(),
        gram2.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()
    var diag_h = _to_host(ctx, s_vec, n_cols)
    var scale_h = List[Float32]()
    var ref_eq = List[Float64](length=n_cols * n_cols, fill=Float64(0.0))
    for i in range(n_cols):
        scale_h.append(ols_equilibration_scale(diag_h[i]))
    for i in range(n_cols):
        for j in range(n_cols):
            ref_eq[i * n_cols + j] = (
                Float64(scale_h[i]) * ref_gram[i * n_cols + j] * Float64(scale_h[j])
            )
    var scale_d = _to_device(ctx, scale_h)
    var cells = n_cols * n_cols
    ctx.enqueue_function[row_vector_binary_mult_kernel](
        gram2.unsafe_ptr(),
        scale_d.unsafe_ptr(),
        Int32(n_cols),
        Int32(n_cols),
        grid_dim=((cells + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.enqueue_function[matrix_vector_binary_mult_kernel](
        gram2.unsafe_ptr(),
        scale_d.unsafe_ptr(),
        Int32(n_cols),
        Int32(n_cols),
        grid_dim=((cells + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()
    var eq_h = _to_host(ctx, gram2, n_cols * n_cols)
    var sc = String("MATRIX ols.scales bits=")
    for i in range(n_cols):
        if i > 0:
            sc += ","
        sc += _hex32(scale_h[i])
    print(sc)
    _matrix_report("ols.equilibrated", eq_h, n_cols, ref_eq)
    _run_case(ctx, "ols.equilibrated", eq_h, n_cols)

    # CONTROL A: the Float64 host covariance rounded to fp32. Jacobi alone,
    # no device Gram in the chain. If this converges on a device whose
    # pca.cov case does not, the Gram is the stage; if it does not, the
    # Jacobi is.
    var host32 = List[Float32]()
    for i in range(n_cols * n_cols):
        host32.append(Float32(ref_cov[i]))
    _matrix_report("host64.cov.f32", host32, n_cols, ref_cov)
    _run_case(ctx, "host64.cov.f32", host32, n_cols)

    # CONTROL B: the 16 x 16 leading block of the device covariance, the
    # width every other fixture converged at on the 5090.
    var sub = List[Float32]()
    var sub_ref = List[Float64]()
    for i in range(16):
        for j in range(16):
            sub.append(cov_h[i * n_cols + j])
            sub_ref.append(ref_cov[i * n_cols + j])
    _matrix_report("pca.cov.16", sub, 16, sub_ref)
    _run_case(ctx, "pca.cov.16", sub, 16)

    print("PROBE_DONE")
