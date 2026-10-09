# SPDX-License-Identifier: Apache-2.0
"""Implemented output-gate and D-skip tail of Mamba-3 backward."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.os import getenv
from std.time import perf_counter_ns
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from checks.kernel_matrix import COLUMN_AMD, COLUMN_APPLE, TARGET_COLUMN, lib_smem_page_fits_for

from checks.numerics import ftz, identical_div, identical_exp, identical_mul_add, identical_rsqrt, identical_sigmoid, identical_silu, identical_tanh, portable_cosf, portable_sinf, identical_mul
from mamba.checks.mamba3_fixture import M3_A_FLOOR, M3_D_STATE, M3_HEADDIM, M3_NUM_ROPE_ANGLES, M3_PI, M3_RMS_EPS, Mamba3Dims
comptime M3_BWD_TPB = 128
comptime M3_S16_V_SHARED = not is_defined["MOJOLEARN_MAMBA3_S16_V_NAIVE"]()
"""lane/neural-apple2 (2026-09-28): the S16 d_v half runs one block per
(row, head) with the q . k dot of each later row computed ONCE into shared
memory (`mamba3_s16_v_shared_kernel`) instead of once per value column p (64
times), and the d_q / d_k half alone in `mamba3_s16_qk_backward_kernel`. Every
dot keeps its operands and its n-ascending order, every d_v chain its
operands and its i-ascending order: the naive kernel's bits.
`-D MOJOLEARN_MAMBA3_S16_V_NAIVE` runs the naive kernel (the host restatement
always does)."""
comptime M3_S17_TAIL_SHARED = not is_defined["MOJOLEARN_MAMBA3_S17_TAIL_NAIVE"]()
"""lane/neural-apple2 (2026-09-28): the S17 operands kernel's chunk-end
`add` chain (qs x 64 x 128 + 64 x 128 serial steps on ONE thread per chunk
and head, every operand a global load) runs in `mamba3_s17_tail_shared_kernel`:
the block stages the operands, thread 0 folds them in the same order.
`-D MOJOLEARN_MAMBA3_S17_TAIL_NAIVE` reverts (the host restatement always
does)."""


def _grid(n: Int) -> Int:
    var g = (n + M3_BWD_TPB - 1) // M3_BWD_TPB
    if g < 1:
        return 1
    return g


def mamba3_gate_skip_backward_kernel(
    d_skip: MutPointer[Float32, MutAnyOrigin],
    d_z: MutPointer[Float32, MutAnyOrigin],
    d_v: MutPointer[Float32, MutAnyOrigin],
    d_qkdot: MutPointer[Float32, MutAnyOrigin],
    d_d_product: MutPointer[Float32, MutAnyOrigin],
    d_gate: MutPointer[Float32, MutAnyOrigin],
    skip: MutPointer[Float32, MutAnyOrigin],
    qkdot: MutPointer[Float32, MutAnyOrigin],
    in_proj: MutPointer[Float32, MutAnyOrigin],
    d_weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    z_col_in: Int32,
    x_col_in: Int32,
):
    """Backward S19 then S18; one thread owns a token/head and folds P."""
    var m = Int(m_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var z_col = Int(z_col_in)
    var x_col = Int(x_col_in)
    var th = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if th >= m * nh:
        return
    var token = th // nh
    var head = th % nh
    var tq = ftz(
        ftz(d_weight.unsafe_load(head))
        + ftz(qkdot.unsafe_load(token * nh + head))
    )
    var dq = Float32(0.0)
    var dd = Float32(0.0)
    for p in range(M3_HEADDIM):
        var cell = (token * nh + head) * M3_HEADDIM + p
        var z = ftz(in_proj.unsafe_load(token * dip + z_col + head * M3_HEADDIM + p))
        var v = ftz(in_proj.unsafe_load(token * dip + x_col + head * M3_HEADDIM + p))
        var dg = ftz(d_gate.unsafe_load(cell))
        var sk = ftz(skip.unsafe_load(cell))
        var ds = ftz(identical_mul(dg, ftz(identical_silu(z))))
        d_skip.unsafe_store(cell, ds)
        var sig = ftz(identical_sigmoid(z))
        var middle = ftz(identical_mul_add(z, ftz(Float32(1.0) - sig), Float32(1.0)))
        var prime = ftz(identical_mul(sig, middle))
        d_z.unsafe_store(cell, ftz(identical_mul(ftz(identical_mul(dg, sk)), prime)))
        d_v.unsafe_store(cell, ftz(identical_mul(ds, tq)))
        dq = ftz(identical_mul_add(ds, v, dq))
        dd = ftz(identical_mul_add(ds, v, dd))
    d_qkdot.unsafe_store(th, dq)
    d_d_product.unsafe_store(th, dd)


def mamba3_backward_gate_skip_into(
    ctx: DeviceContext,
    mut d_skip: DeviceBuffer[DType.float32],
    mut d_z: DeviceBuffer[DType.float32],
    mut d_v: DeviceBuffer[DType.float32],
    mut d_qkdot: DeviceBuffer[DType.float32],
    mut d_d_product: DeviceBuffer[DType.float32],
    mut d_gate: DeviceBuffer[DType.float32],
    mut skip: DeviceBuffer[DType.float32],
    mut qkdot: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    mut d_weight: DeviceBuffer[DType.float32],
    dims: Mamba3Dims,
    m: Int,
) raises:
    var cells = m * dims.nheads
    ctx.enqueue_function[mamba3_gate_skip_backward_kernel](
        d_skip.unsafe_ptr(), d_z.unsafe_ptr(), d_v.unsafe_ptr(),
        d_qkdot.unsafe_ptr(), d_d_product.unsafe_ptr(), d_gate.unsafe_ptr(),
        skip.unsafe_ptr(), qkdot.unsafe_ptr(), in_proj.unsafe_ptr(),
        d_weight.unsafe_ptr(), Int32(m), Int32(dims.nheads),
        Int32(dims.d_in_proj()), Int32(0), Int32(dims.d_inner),
        grid_dim=(_grid(cells), 1, 1), block_dim=(M3_BWD_TPB, 1, 1),
    )


def mamba3_qkdot_backward_kernel(
    d_b: MutPointer[Float32, MutAnyOrigin],
    d_c: MutPointer[Float32, MutAnyOrigin],
    d_b_bias: MutPointer[Float32, MutAnyOrigin],
    d_c_bias: MutPointer[Float32, MutAnyOrigin],
    d_gamma: MutPointer[Float32, MutAnyOrigin],
    d_dt: MutPointer[Float32, MutAnyOrigin],
    d_trap: MutPointer[Float32, MutAnyOrigin],
    d_qkdot: MutPointer[Float32, MutAnyOrigin],
    bcb: MutPointer[Float32, MutAnyOrigin],
    bcc: MutPointer[Float32, MutAnyOrigin],
    b_bias: MutPointer[Float32, MutAnyOrigin],
    c_bias: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin],
    sigma: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    nh_in: Int32,
):
    """Backward S14's pre-rotation dot; outputs head-private B/C partials."""
    var m = Int(m_in)
    var nh = Int(nh_in)
    var th = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if th >= m * nh:
        return
    var token = th // nh
    var head = th % nh
    var incoming = ftz(d_qkdot.unsafe_load(th))
    var gam = ftz(gamma.unsafe_load(th))
    var dot = Float32(0.0)
    for n in range(M3_D_STATE):
        var q = ftz(ftz(bcc.unsafe_load(token * M3_D_STATE + n)) + ftz(c_bias.unsafe_load(head * M3_D_STATE + n)))
        var k = ftz(ftz(bcb.unsafe_load(token * M3_D_STATE + n)) + ftz(b_bias.unsafe_load(head * M3_D_STATE + n)))
        dot = ftz(identical_mul_add(q, k, dot))
        var common = ftz(identical_mul(incoming, gam))
        var db = ftz(identical_mul(common, q))
        var dc = ftz(identical_mul(common, k))
        var cell = (token * nh + head) * M3_D_STATE + n
        d_b.unsafe_store(cell, db)
        d_c.unsafe_store(cell, dc)
        d_b_bias.unsafe_store(cell, db)
        d_c_bias.unsafe_store(cell, dc)
    var dgam = ftz(identical_mul(incoming, dot))
    d_gamma.unsafe_store(th, dgam)
    var sig = ftz(sigma.unsafe_load(th))
    d_dt.unsafe_store(th, ftz(identical_mul(dgam, sig)))
    var dsig = ftz(identical_mul(dgam, ftz(dt.unsafe_load(th))))
    d_trap.unsafe_store(
        th,
        ftz(identical_mul(ftz(identical_mul(dsig, sig)), ftz(Float32(1.0) - sig))),
    )


def mamba3_backward_qkdot_into(
    ctx: DeviceContext,
    mut d_b: DeviceBuffer[DType.float32],
    mut d_c: DeviceBuffer[DType.float32],
    mut d_b_bias: DeviceBuffer[DType.float32],
    mut d_c_bias: DeviceBuffer[DType.float32],
    mut d_gamma: DeviceBuffer[DType.float32],
    mut d_dt: DeviceBuffer[DType.float32],
    mut d_trap: DeviceBuffer[DType.float32],
    mut d_qkdot: DeviceBuffer[DType.float32],
    mut bcb: DeviceBuffer[DType.float32],
    mut bcc: DeviceBuffer[DType.float32],
    mut b_bias: DeviceBuffer[DType.float32],
    mut c_bias: DeviceBuffer[DType.float32],
    mut gamma: DeviceBuffer[DType.float32],
    mut dt: DeviceBuffer[DType.float32],
    mut sigma: DeviceBuffer[DType.float32],
    dims: Mamba3Dims,
    m: Int,
) raises:
    var cells = m * dims.nheads
    ctx.enqueue_function[mamba3_qkdot_backward_kernel](
        d_b.unsafe_ptr(), d_c.unsafe_ptr(), d_b_bias.unsafe_ptr(),
        d_c_bias.unsafe_ptr(), d_gamma.unsafe_ptr(), d_dt.unsafe_ptr(),
        d_trap.unsafe_ptr(), d_qkdot.unsafe_ptr(),
        bcb.unsafe_ptr(), bcc.unsafe_ptr(), b_bias.unsafe_ptr(),
        c_bias.unsafe_ptr(), gamma.unsafe_ptr(), dt.unsafe_ptr(),
        sigma.unsafe_ptr(), Int32(m), Int32(dims.nheads),
        grid_dim=(_grid(cells), 1, 1), block_dim=(M3_BWD_TPB, 1, 1),
    )


def mamba3_s16_qkv_backward_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_v: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """Naive, ownership-safe S16 backward over real rows within each chunk."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var q_cells = b * l * nh * M3_D_STATE
    var v_cells = b * l * nh * M3_HEADDIM
    if cell < q_cells:
        var n = cell % M3_D_STATE
        var rowh = cell // M3_D_STATE
        var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
        var chunk = token // qs; var inner = token % qs
        var dq = Float32(0.0); var dk = Float32(0.0)
        for p in range(M3_HEADDIM):
            var dy_i = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
            for j in range(inner):
                var tj = chunk * qs + j
                if tj < l:
                    var lv = ftz(seg_l.unsafe_load((((bb * ((l + qs - 1) // qs) + chunk) * nh + h) * qs + inner) * qs + j))
                    var kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
                    var vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
                    dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(ftz(identical_mul(kv, lv)), vv)), dq))
            for i in range(inner + 1, qs):
                var ti = chunk * qs + i
                if ti < l:
                    var dy = ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p))
                    var lv2 = ftz(seg_l.unsafe_load((((bb * ((l + qs - 1) // qs) + chunk) * nh + h) * qs + i) * qs + inner))
                    var qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
                    var vv2 = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
                    dk = ftz(identical_mul_add(dy, ftz(identical_mul(ftz(identical_mul(qv, lv2)), vv2)), dk))
        d_q.unsafe_store(cell, dq); d_k.unsafe_store(cell, dk)
    if cell < v_cells:
        var p = cell % M3_HEADDIM
        var rowh = cell // M3_HEADDIM
        var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
        var chunk = token // qs; var inner = token % qs; var dv = Float32(0.0)
        for i in range(inner + 1, qs):
            var ti = chunk * qs + i
            if ti < l:
                var dot = Float32(0.0)
                for n in range(M3_D_STATE):
                    dot = ftz(identical_mul_add(ftz(q.unsafe_load(((bb*l+ti)*nh+h)*M3_D_STATE+n)), ftz(k.unsafe_load(((bb*l+token)*nh+h)*M3_D_STATE+n)), dot))
                var lv = ftz(seg_l.unsafe_load((((bb*((l+qs-1)//qs)+chunk)*nh+h)*qs+i)*qs+inner))
                dv = ftz(identical_mul_add(ftz(d_y.unsafe_load(((bb*l+ti)*nh+h)*M3_HEADDIM+p)), ftz(identical_mul(dot, lv)), dv))
        d_v.unsafe_store(cell, dv)


#: lane/neural-net-experiment (2026-09-30): the S16 d_q / d_k half with
#: its operands STAGED. The naive kernel is one thread per (row, n) cell
#: whose chain of 64 x qs fmas loads four device words per fma, three of
#: which do not depend on the chain's own p index (k[tj, n], the segment
#: factor, and v[tj, p] shared by the whole row): the L40S priced the
#: s16_s15 stage at 23.7 ms of a 37.6 ms Mamba-3 backward (63%). Here one
#: block of M3_D_STATE threads owns one (batch, token, head) row (thread
#: n): the chunk's v rows and d_y rows, the row's two segment vectors and
#: its own v / d_y rows are staged once in threadgroup memory; thread n
#: forms `ftz(mul(k[tj, n], l[inner, j]))` once per j and
#: `ftz(mul(q[ti, n], l[i, inner]))` once per i (the naive kernel formed
#: each of them 64 times, once per p, from the same two operands: the
#: same float); the two chains then run in the naive order (p outer, j /
#: i inner) over staged values. SAME BITS: every fma has the same three
#: operands in the same order. MOJOLEARN_MAMBA3_S16_QK_NAIVE=1 (a build
#: define) keeps the naive kernel for the A/B and the digest gate.
comptime M3_S16_QK_SHARED = not is_defined["MOJOLEARN_MAMBA3_S16_QK_NAIVE"]()
comptime M3_S16_QK_MAXQ = 64


def mamba3_s16_qk_shared_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qk_backward_kernel`'s chains over staged operands, one
    block of M3_D_STATE threads per (batch, token, head) row, thread n
    the cell (row, n). Rows of a chunk beyond `l` are staged as zero and
    never entered (the naive kernel's `tj < l` / `ti < l` guards)."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    # the chunk's v rows [qs][HEADDIM] and d_y rows [qs][HEADDIM], flushed
    var vs = stack_allocation[M3_S16_QK_MAXQ * M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[M3_S16_QK_MAXQ * M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # l[inner, j] over j, l[i, inner] over i, this row's v and d_y, flushed
    var lrow = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lcol = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var vtok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dytok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rowh = Int(block_idx.x)
    var n = Int(thread_idx.x)
    if rowh >= b * l * nh:
        return
    var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
    var chunk = token // qs; var inner = token % qs
    var nc = (l + qs - 1) // qs
    var lbase = ((bb * nc + chunk) * nh + h) * qs * qs
    # staging: qs * HEADDIM cells of v and d_y over M3_D_STATE threads
    var e = n
    while e < qs * M3_HEADDIM:
        var j = e // M3_HEADDIM
        var p = e - j * M3_HEADDIM
        var tj = chunk * qs + j
        var vv = Float32(0.0)
        var dv = Float32(0.0)
        if tj < l:
            vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
            dv = ftz(d_y.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
        vs[e] = vv
        dys[e] = dv
        e += M3_D_STATE
    if n < qs:
        lrow[n] = ftz(seg_l.unsafe_load(lbase + inner * qs + n))
        lcol[n] = ftz(seg_l.unsafe_load(lbase + n * qs + inner))
    if n < M3_HEADDIM:
        vtok[n] = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
        dytok[n] = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
    barrier()
    # thread n's own per-j and per-i products, formed once
    var kl = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32]]()
    var ql = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32]]()
    for j in range(inner):
        var tj = chunk * qs + j
        var kv = Float32(0.0)
        if tj < l:
            kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
        kl[j] = ftz(identical_mul(kv, lrow[j]))
    for i in range(inner + 1, qs):
        var ti = chunk * qs + i
        var qv = Float32(0.0)
        if ti < l:
            qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
        ql[i] = ftz(identical_mul(qv, lcol[i]))
    var dq = Float32(0.0); var dk = Float32(0.0)
    for p in range(M3_HEADDIM):
        var dy_i = dytok[p]
        for j in range(inner):
            var tj = chunk * qs + j
            if tj < l:
                dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(kl[j], vs[j * M3_HEADDIM + p])), dq))
        for i in range(inner + 1, qs):
            var ti = chunk * qs + i
            if ti < l:
                dk = ftz(identical_mul_add(dys[i * M3_HEADDIM + p], ftz(identical_mul(ql[i], vtok[p])), dk))
    d_q.unsafe_store(rowh * M3_D_STATE + n, dq)
    d_k.unsafe_store(rowh * M3_D_STATE + n, dk)


#: lane/neural-net-experiment (2026-09-30, the S16 pass): the L40S priced
#: the staged kernel above at 21.9 ms against the naive 23.8 (8%), where
#: the instruction count says a few ms. The staged kernel's per-thread
#: `kl` / `ql` arrays are LOCAL memory indexed at run time (64 words a
#: thread; 384 threads a multiprocessor next to 99 KB of staged pages
#: leave no L1 for them), so every chain step still pays a cached device
#: load, as the naive kernel's did. Two more arms, the same chains:
#:   regs   `mamba3_s16_qk_regs_kernel`: the j and i loops are unrolled at
#:          compile time over M3_S16_QK_MAXQ, so `kl` / `ql` are registers
#:          and the guards are predicates; the p loop stays a loop.
#:   smem48 `mamba3_s16_qk_smem48_kernel`: `kl` for every n of the block's
#:          row in threadgroup memory ([j][n], 32 KB) beside the staged
#:          v / d_y page (16 KB), the dq chain and then the dk chain each
#:          over its own page (their accumulators are separate, so each
#:          chain's order is untouched). Exactly the column's 48 KB; only
#:          where `lib_smem_page_fits_for` says the page fits.
#: MOJOLEARN_MAMBA3_S16_QK_ARM = naive | shared | regs (default; regs2h on AMD and Apple) | regs2 | smem48 | regs2h | regsh
#: picks the launch at run time, one build for the whole sweep; under
#: MOJOLEARN_MAMBA_TIMING the S16 driver prints a wall per kernel.
comptime M3_S16_SMEM48_BYTES = (M3_S16_QK_MAXQ * M3_D_STATE + M3_S16_QK_MAXQ * M3_HEADDIM) * 4
comptime M3_S16_SMEM48_FITS = lib_smem_page_fits_for[TARGET_COLUMN, M3_S16_SMEM48_BYTES]()


def m3_s16_qk_arm() -> Int:
    """The S16 q/k arm: `regs` by default (lane/neural-pass4, 2026-09-30:
    on the L4 `regs` read 9.2 ms against `regs2`'s 10.8 at the board shape
    and 83.5 against 86.0 at the default shape, the same bits on every arm;
    `regs2` was the S16 pass's default and stays one env value away).
    On the AMD column the default is `regs2h` (lane/mamba3-amd-regs2h,
    2026-10-01, MI325X at B=2 L=512 d_model=384: the q/k stage read 4.5 ms
    on `regs2h` against `regs2`'s 11.5 and `regs`'s 18.6, the backward's
    stage sum 56.9 against 71.0, same bits on every arm;
    bench/results/mamba3-apple-s16-halfpage-20261001/amd). On the Apple
    column the default is `regs2h` (lane/neural-apple3, 2026-10-01): the
    full-page arms are over Metal's 32 KB and fell through to the naive
    kernel; the half-page arms fit (M3 Ultra: 2.5x the naive run;
    `regsh` is the A/B)."""
    var a = String(getenv("MOJOLEARN_MAMBA3_S16_QK_ARM"))
    if a == "naive":
        return 0
    if a == "shared":
        return 1
    if a == "regs2":
        return 4
    if a == "smem48":
        return 3
    if a == "regs":
        return 2
    if a == "regs2h":
        return 5
    if a == "regsh":
        return 6
    comptime if TARGET_COLUMN == COLUMN_AMD or TARGET_COLUMN == COLUMN_APPLE:
        return 5
    return 2


#: The staged q/k arms (shared, regs, regs2) claim 33 to 36 KB of threadgroup
#: memory a block, over the Apple column's 32 KB; where the page does not
#: fit the driver runs the naive kernel whatever the arm asks.
comptime M3_S16_QK_PAGE_BYTES = (2 * M3_S16_QK_MAXQ * M3_HEADDIM + 2 * M3_S16_QK_MAXQ + 2 * M3_HEADDIM) * 4
comptime M3_S16_QK_PAGE_FITS = lib_smem_page_fits_for[TARGET_COLUMN, M3_S16_QK_PAGE_BYTES]()
#: regs2's transposed pages: row p of v (and of d_y) is M3_S16_QK_STRIDE
#: words, 64 of them used, 4 of padding so a row starts 16-byte aligned
#: (a 4-wide shared load a step) and consecutive p rows spread over the
#: banks at the fill.
comptime M3_S16_QK_STRIDE = M3_S16_QK_MAXQ + 4
comptime M3_S16_QK_REGS2_BYTES = (2 * M3_HEADDIM * M3_S16_QK_STRIDE + 2 * M3_S16_QK_MAXQ + 2 * M3_HEADDIM) * 4
comptime M3_S16_QK_REGS2_FITS = lib_smem_page_fits_for[TARGET_COLUMN, M3_S16_QK_REGS2_BYTES]()


def mamba3_s16_qk_regs2_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qk_regs_kernel` leaner (lane/neural-net-experiment,
    2026-09-30, the S16 pass, second batch), the same chains in the same
    order:
      - ONE register array `kq`: `kl[j]` for j < inner and `ql[j]` for
        j > inner never overlap, so they share the slots (64 registers
        where the regs kernel holds 128; occupancy, not bits);
      - the v and d_y pages transposed, `vs[p * STRIDE + j]`, so the j
        steps of a p row read consecutive words: a 4-wide shared load per
        four steps instead of a scalar load per step (the same words);
      - the dq chain's `tj < l` test dropped: j < inner = token mod qs and
        token < l give it (the dk chain keeps its own, the last chunk is
        short)."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var vs = stack_allocation[M3_HEADDIM * M3_S16_QK_STRIDE, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[M3_HEADDIM * M3_S16_QK_STRIDE, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var lrow = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lcol = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var vtok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dytok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rowh = Int(block_idx.x)
    var n = Int(thread_idx.x)
    if rowh >= b * l * nh:
        return
    var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
    var chunk = token // qs; var inner = token % qs
    var nc = (l + qs - 1) // qs
    var lbase = ((bb * nc + chunk) * nh + h) * qs * qs
    var e = n
    while e < qs * M3_HEADDIM:
        var j = e // M3_HEADDIM
        var p = e - j * M3_HEADDIM
        var tj = chunk * qs + j
        var vv = Float32(0.0)
        var dv = Float32(0.0)
        if tj < l:
            vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
            dv = ftz(d_y.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
        vs[p * M3_S16_QK_STRIDE + j] = vv
        dys[p * M3_S16_QK_STRIDE + j] = dv
        e += M3_D_STATE
    if n < qs:
        lrow[n] = ftz(seg_l.unsafe_load(lbase + inner * qs + n))
        lcol[n] = ftz(seg_l.unsafe_load(lbase + n * qs + inner))
    if n < M3_HEADDIM:
        vtok[n] = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
        dytok[n] = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
    barrier()
    var kq = InlineArray[Float32, M3_S16_QK_MAXQ](fill=Float32(0))
    comptime for j in range(M3_S16_QK_MAXQ):
        if j < inner:
            var kv = ftz(k.unsafe_load(((bb * l + chunk * qs + j) * nh + h) * M3_D_STATE + n))
            kq[j] = ftz(identical_mul(kv, lrow[j]))
        elif j > inner and j < qs:
            var qv = Float32(0.0)
            if chunk * qs + j < l:
                qv = ftz(q.unsafe_load(((bb * l + chunk * qs + j) * nh + h) * M3_D_STATE + n))
            kq[j] = ftz(identical_mul(qv, lcol[j]))
    var dq = Float32(0.0); var dk = Float32(0.0)
    for p in range(M3_HEADDIM):
        var dy_i = dytok[p]
        var vt = vtok[p]
        var prow = p * M3_S16_QK_STRIDE
        comptime for j4 in range(0, M3_S16_QK_MAXQ, 4):
            if j4 < inner:
                var v4 = vs.unsafe_load[width=4, alignment=16](prow + j4)
                comptime for jj in range(4):
                    if j4 + jj < inner:
                        dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(kq[j4 + jj], v4[jj])), dq))
        comptime for i4 in range(0, M3_S16_QK_MAXQ, 4):
            if i4 + 3 > inner and i4 < qs:
                var d4 = dys.unsafe_load[width=4, alignment=16](prow + i4)
                comptime for ii in range(4):
                    if i4 + ii > inner and i4 + ii < qs:
                        if chunk * qs + i4 + ii < l:
                            dk = ftz(identical_mul_add(d4[ii], ftz(identical_mul(kq[i4 + ii], vt)), dk))
    d_q.unsafe_store(rowh * M3_D_STATE + n, dq)
    d_k.unsafe_store(rowh * M3_D_STATE + n, dk)


def mamba3_s16_qk_regs_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qk_shared_kernel` with the per-thread products in
    registers: the j and i loops unrolled over M3_S16_QK_MAXQ at compile
    time, each step predicated on the naive kernel's own bounds."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var vs = stack_allocation[M3_S16_QK_MAXQ * M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[M3_S16_QK_MAXQ * M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lrow = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lcol = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var vtok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dytok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rowh = Int(block_idx.x)
    var n = Int(thread_idx.x)
    if rowh >= b * l * nh:
        return
    var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
    var chunk = token // qs; var inner = token % qs
    var nc = (l + qs - 1) // qs
    var lbase = ((bb * nc + chunk) * nh + h) * qs * qs
    var e = n
    while e < qs * M3_HEADDIM:
        var j = e // M3_HEADDIM
        var p = e - j * M3_HEADDIM
        var tj = chunk * qs + j
        var vv = Float32(0.0)
        var dv = Float32(0.0)
        if tj < l:
            vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
            dv = ftz(d_y.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
        vs[e] = vv
        dys[e] = dv
        e += M3_D_STATE
    if n < qs:
        lrow[n] = ftz(seg_l.unsafe_load(lbase + inner * qs + n))
        lcol[n] = ftz(seg_l.unsafe_load(lbase + n * qs + inner))
    if n < M3_HEADDIM:
        vtok[n] = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
        dytok[n] = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
    barrier()
    var kl = InlineArray[Float32, M3_S16_QK_MAXQ](fill=Float32(0))
    var ql = InlineArray[Float32, M3_S16_QK_MAXQ](fill=Float32(0))
    comptime for j in range(M3_S16_QK_MAXQ):
        if j < inner:
            var tj = chunk * qs + j
            var kv = Float32(0.0)
            if tj < l:
                kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
            kl[j] = ftz(identical_mul(kv, lrow[j]))
        elif j > inner and j < qs:
            var ti = chunk * qs + j
            var qv = Float32(0.0)
            if ti < l:
                qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
            ql[j] = ftz(identical_mul(qv, lcol[j]))
    var dq = Float32(0.0); var dk = Float32(0.0)
    for p in range(M3_HEADDIM):
        var dy_i = dytok[p]
        var vt = vtok[p]
        comptime for j in range(M3_S16_QK_MAXQ):
            if j < inner:
                if chunk * qs + j < l:
                    dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(kl[j], vs[j * M3_HEADDIM + p])), dq))
        comptime for i in range(M3_S16_QK_MAXQ):
            if i > inner and i < qs:
                if chunk * qs + i < l:
                    dk = ftz(identical_mul_add(dys[i * M3_HEADDIM + p], ftz(identical_mul(ql[i], vt)), dk))
    d_q.unsafe_store(rowh * M3_D_STATE + n, dq)
    d_k.unsafe_store(rowh * M3_D_STATE + n, dk)


#: lane/neural-apple3 (2026-10-01): the staged q/k arms above claim 33 to
#: 36 KB of threadgroup memory a block and the Apple column allows 32, so
#: Metal fell through to the naive kernel for every Mamba-3 backward (on the
#: L4 the naive kernel read 65.1 ms against regs's 9.2 at the board shape,
#: bench/results/s16-pass-20260930). Two half-page arms, the same chains in
#: the same order: the v and d_y pages hold M3_S16_QK_PH = 32 value columns
#: at a time and are staged twice (p 0..31, a barrier, then p 32..63). The
#: p loop is the chains' OUTER loop and the per-thread products stay in
#: registers across the halves, so each thread's fma sequence is the
#: full-page kernel's: p ascending, j ascending below `inner`, i ascending
#: above it. SAME BITS.
#:   regs2h `mamba3_s16_qk_regs2h_kernel`: regs2 over half pages (18,432
#:          bytes); the Apple column's default (unmeasured on a Mac when
#:          written: regs2's one register array and its 4-wide page loads
#:          are the lighter budget).
#:   regsh  `mamba3_s16_qk_regsh_kernel`: regs over half pages (17,408 bytes).
#: Both compile on every column for the A/B (opt-in off Apple).
comptime M3_S16_QK_PH = M3_HEADDIM // 2
comptime M3_S16_QK_REGS2H_BYTES = (2 * M3_S16_QK_PH * M3_S16_QK_STRIDE + 2 * M3_S16_QK_MAXQ + 2 * M3_HEADDIM) * 4
comptime M3_S16_QK_REGS2H_FITS = lib_smem_page_fits_for[TARGET_COLUMN, M3_S16_QK_REGS2H_BYTES]()
comptime M3_S16_QK_REGSH_BYTES = (2 * M3_S16_QK_MAXQ * M3_S16_QK_PH + 2 * M3_S16_QK_MAXQ + 2 * M3_HEADDIM) * 4
comptime M3_S16_QK_REGSH_FITS = lib_smem_page_fits_for[TARGET_COLUMN, M3_S16_QK_REGSH_BYTES]()


def mamba3_s16_qk_regs2h_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qk_regs2_kernel` over half pages: the transposed v and
    d_y pages hold M3_S16_QK_PH value columns, staged twice; the chains
    walk p 0..31 off the first staging and 32..63 off the second, the
    same steps in the same order."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var vs = stack_allocation[M3_S16_QK_PH * M3_S16_QK_STRIDE, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[M3_S16_QK_PH * M3_S16_QK_STRIDE, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var lrow = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lcol = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var vtok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dytok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rowh = Int(block_idx.x)
    var n = Int(thread_idx.x)
    if rowh >= b * l * nh:
        return
    var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
    var chunk = token // qs; var inner = token % qs
    var nc = (l + qs - 1) // qs
    var lbase = ((bb * nc + chunk) * nh + h) * qs * qs
    if n < qs:
        lrow[n] = ftz(seg_l.unsafe_load(lbase + inner * qs + n))
        lcol[n] = ftz(seg_l.unsafe_load(lbase + n * qs + inner))
    if n < M3_HEADDIM:
        vtok[n] = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
        dytok[n] = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
    var kq = InlineArray[Float32, M3_S16_QK_MAXQ](fill=Float32(0))
    var dq = Float32(0.0); var dk = Float32(0.0)
    for half in range(2):
        var p0 = half * M3_S16_QK_PH
        if half > 0:
            barrier()  # every thread is off the first half's page
        var e = n
        while e < qs * M3_S16_QK_PH:
            var j = e // M3_S16_QK_PH
            var pp = e - j * M3_S16_QK_PH
            var tj = chunk * qs + j
            var vv = Float32(0.0)
            var dv = Float32(0.0)
            if tj < l:
                vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p0 + pp))
                dv = ftz(d_y.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p0 + pp))
            vs[pp * M3_S16_QK_STRIDE + j] = vv
            dys[pp * M3_S16_QK_STRIDE + j] = dv
            e += M3_D_STATE
        barrier()
        if half == 0:
            comptime for j in range(M3_S16_QK_MAXQ):
                if j < inner:
                    var kv = ftz(k.unsafe_load(((bb * l + chunk * qs + j) * nh + h) * M3_D_STATE + n))
                    kq[j] = ftz(identical_mul(kv, lrow[j]))
                elif j > inner and j < qs:
                    var qv = Float32(0.0)
                    if chunk * qs + j < l:
                        qv = ftz(q.unsafe_load(((bb * l + chunk * qs + j) * nh + h) * M3_D_STATE + n))
                    kq[j] = ftz(identical_mul(qv, lcol[j]))
        for pp in range(M3_S16_QK_PH):
            var dy_i = dytok[p0 + pp]
            var vt = vtok[p0 + pp]
            var prow = pp * M3_S16_QK_STRIDE
            comptime for j4 in range(0, M3_S16_QK_MAXQ, 4):
                if j4 < inner:
                    var v4 = vs.unsafe_load[width=4, alignment=16](prow + j4)
                    comptime for jj in range(4):
                        if j4 + jj < inner:
                            dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(kq[j4 + jj], v4[jj])), dq))
            comptime for i4 in range(0, M3_S16_QK_MAXQ, 4):
                if i4 + 3 > inner and i4 < qs:
                    var d4 = dys.unsafe_load[width=4, alignment=16](prow + i4)
                    comptime for ii in range(4):
                        if i4 + ii > inner and i4 + ii < qs:
                            if chunk * qs + i4 + ii < l:
                                dk = ftz(identical_mul_add(d4[ii], ftz(identical_mul(kq[i4 + ii], vt)), dk))
    d_q.unsafe_store(rowh * M3_D_STATE + n, dq)
    d_k.unsafe_store(rowh * M3_D_STATE + n, dk)


def mamba3_s16_qk_regsh_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qk_regs_kernel` over half pages: the v and d_y pages
    hold M3_S16_QK_PH value columns ([j][pp]), staged twice; the same
    steps in the same order."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var vs = stack_allocation[M3_S16_QK_MAXQ * M3_S16_QK_PH, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[M3_S16_QK_MAXQ * M3_S16_QK_PH, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lrow = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lcol = stack_allocation[M3_S16_QK_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var vtok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dytok = stack_allocation[M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rowh = Int(block_idx.x)
    var n = Int(thread_idx.x)
    if rowh >= b * l * nh:
        return
    var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
    var chunk = token // qs; var inner = token % qs
    var nc = (l + qs - 1) // qs
    var lbase = ((bb * nc + chunk) * nh + h) * qs * qs
    if n < qs:
        lrow[n] = ftz(seg_l.unsafe_load(lbase + inner * qs + n))
        lcol[n] = ftz(seg_l.unsafe_load(lbase + n * qs + inner))
    if n < M3_HEADDIM:
        vtok[n] = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
        dytok[n] = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + n))
    var kl = InlineArray[Float32, M3_S16_QK_MAXQ](fill=Float32(0))
    var ql = InlineArray[Float32, M3_S16_QK_MAXQ](fill=Float32(0))
    var dq = Float32(0.0); var dk = Float32(0.0)
    for half in range(2):
        var p0 = half * M3_S16_QK_PH
        if half > 0:
            barrier()  # every thread is off the first half's page
        var e = n
        while e < qs * M3_S16_QK_PH:
            var j = e // M3_S16_QK_PH
            var pp = e - j * M3_S16_QK_PH
            var tj = chunk * qs + j
            var vv = Float32(0.0)
            var dv = Float32(0.0)
            if tj < l:
                vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p0 + pp))
                dv = ftz(d_y.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p0 + pp))
            vs[e] = vv
            dys[e] = dv
            e += M3_D_STATE
        barrier()
        if half == 0:
            comptime for j in range(M3_S16_QK_MAXQ):
                if j < inner:
                    var tj = chunk * qs + j
                    var kv = Float32(0.0)
                    if tj < l:
                        kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
                    kl[j] = ftz(identical_mul(kv, lrow[j]))
                elif j > inner and j < qs:
                    var ti = chunk * qs + j
                    var qv = Float32(0.0)
                    if ti < l:
                        qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
                    ql[j] = ftz(identical_mul(qv, lcol[j]))
        for pp in range(M3_S16_QK_PH):
            var dy_i = dytok[p0 + pp]
            var vt = vtok[p0 + pp]
            comptime for j in range(M3_S16_QK_MAXQ):
                if j < inner:
                    if chunk * qs + j < l:
                        dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(kl[j], vs[j * M3_S16_QK_PH + pp])), dq))
            comptime for i in range(M3_S16_QK_MAXQ):
                if i > inner and i < qs:
                    if chunk * qs + i < l:
                        dk = ftz(identical_mul_add(dys[i * M3_S16_QK_PH + pp], ftz(identical_mul(ql[i], vt)), dk))
    d_q.unsafe_store(rowh * M3_D_STATE + n, dq)
    d_k.unsafe_store(rowh * M3_D_STATE + n, dk)


def mamba3_s16_qk_smem48_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qk_shared_kernel` with the block's `kl` / `ql` products
    in threadgroup memory ([j][n]), the dq chain over the k page and the
    v page, then the dk chain over the q page and the d_y page (the same
    two arrays, refilled; the two chains have separate accumulators, so
    each keeps the naive order). 48 KB of threadgroup memory."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var prod = stack_allocation[M3_S16_QK_MAXQ * M3_D_STATE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var page = stack_allocation[M3_S16_QK_MAXQ * M3_HEADDIM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rowh = Int(block_idx.x)
    var n = Int(thread_idx.x)
    if rowh >= b * l * nh:
        return
    var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
    var chunk = token // qs; var inner = token % qs
    var nc = (l + qs - 1) // qs
    var lbase = ((bb * nc + chunk) * nh + h) * qs * qs
    var tokbase = ((bb * l + token) * nh + h) * M3_HEADDIM
    # phase 1: the v page and thread n's kl[j] for j < inner
    var e = n
    while e < qs * M3_HEADDIM:
        var j = e // M3_HEADDIM
        var p = e - j * M3_HEADDIM
        var tj = chunk * qs + j
        var vv = Float32(0.0)
        if tj < l:
            vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
        page[e] = vv
        e += M3_D_STATE
    for j in range(inner):
        var tj = chunk * qs + j
        var kv = Float32(0.0)
        if tj < l:
            kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
        prod[j * M3_D_STATE + n] = ftz(identical_mul(kv, ftz(seg_l.unsafe_load(lbase + inner * qs + j))))
    barrier()
    var dq = Float32(0.0)
    for p in range(M3_HEADDIM):
        var dy_i = ftz(d_y.unsafe_load(tokbase + p))
        for j in range(inner):
            if chunk * qs + j < l:
                dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(prod[j * M3_D_STATE + n], page[j * M3_HEADDIM + p])), dq))
    d_q.unsafe_store(rowh * M3_D_STATE + n, dq)
    barrier()
    # phase 2: the d_y page and thread n's ql[i] for i > inner
    e = n
    while e < qs * M3_HEADDIM:
        var i = e // M3_HEADDIM
        var p = e - i * M3_HEADDIM
        var ti = chunk * qs + i
        var dv = Float32(0.0)
        if ti < l:
            dv = ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p))
        page[e] = dv
        e += M3_D_STATE
    for i in range(inner + 1, qs):
        var ti = chunk * qs + i
        var qv = Float32(0.0)
        if ti < l:
            qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
        prod[i * M3_D_STATE + n] = ftz(identical_mul(qv, ftz(seg_l.unsafe_load(lbase + i * qs + inner))))
    barrier()
    var dk = Float32(0.0)
    for p in range(M3_HEADDIM):
        var vt = ftz(v.unsafe_load(tokbase + p))
        for i in range(inner + 1, qs):
            if chunk * qs + i < l:
                dk = ftz(identical_mul_add(page[i * M3_HEADDIM + p], ftz(identical_mul(prod[i * M3_D_STATE + n], vt)), dk))
    d_k.unsafe_store(rowh * M3_D_STATE + n, dk)


def _s16_tick(ctx: DeviceContext, on: Bool, mut t: Int, name: String) raises:
    """MOJOLEARN_MAMBA_TIMING: a wall per S16 kernel, `timing m3bwd.s16.<name>`."""
    if not on:
        return
    ctx.synchronize()
    var now = Int(perf_counter_ns())
    print("timing m3bwd.s16." + name + " " + String(Float64(now - t) / 1000000.0) + " ms")
    t = now


def mamba3_s16_qk_backward_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_v: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qkv_backward_kernel`'s d_q / d_k half, line for line
    (M3_S16_V_SHARED; `d_v` is unused)."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var q_cells = b * l * nh * M3_D_STATE
    if cell < q_cells:
        var n = cell % M3_D_STATE
        var rowh = cell // M3_D_STATE
        var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
        var chunk = token // qs; var inner = token % qs
        var dq = Float32(0.0); var dk = Float32(0.0)
        for p in range(M3_HEADDIM):
            var dy_i = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
            for j in range(inner):
                var tj = chunk * qs + j
                if tj < l:
                    var lv = ftz(seg_l.unsafe_load((((bb * ((l + qs - 1) // qs) + chunk) * nh + h) * qs + inner) * qs + j))
                    var kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
                    var vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
                    dq = ftz(identical_mul_add(dy_i, ftz(identical_mul(ftz(identical_mul(kv, lv)), vv)), dq))
            for i in range(inner + 1, qs):
                var ti = chunk * qs + i
                if ti < l:
                    var dy = ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p))
                    var lv2 = ftz(seg_l.unsafe_load((((bb * ((l + qs - 1) // qs) + chunk) * nh + h) * qs + i) * qs + inner))
                    var qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
                    var vv2 = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
                    dk = ftz(identical_mul_add(dy, ftz(identical_mul(ftz(identical_mul(qv, lv2)), vv2)), dk))
        d_q.unsafe_store(cell, dq); d_k.unsafe_store(cell, dk)


comptime M3_S16_V_MAXQ = 256


def mamba3_s16_v_shared_kernel(
    d_v: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """`mamba3_s16_qkv_backward_kernel`'s d_v half, one block of
    M3_HEADDIM threads per (batch, token, head) row: thread r computes the
    later rows' `ftz(identical_mul(dot, lv))` for i = inner + 1 + r,
    inner + 1 + r + M3_HEADDIM, ... (each dot over n ascending, as the naive
    kernel), then thread p folds its d_v chain over i ascending."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var dl = stack_allocation[M3_S16_V_MAXQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rowh = Int(block_idx.x)
    var p = Int(thread_idx.x)
    if rowh >= b * l * nh:
        return
    var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
    var chunk = token // qs; var inner = token % qs
    var i = inner + 1 + p
    while i < qs:
        var ti = chunk * qs + i
        if ti < l:
            var dot = Float32(0.0)
            for n in range(M3_D_STATE):
                dot = ftz(identical_mul_add(ftz(q.unsafe_load(((bb*l+ti)*nh+h)*M3_D_STATE+n)), ftz(k.unsafe_load(((bb*l+token)*nh+h)*M3_D_STATE+n)), dot))
            var lv = ftz(seg_l.unsafe_load((((bb*((l+qs-1)//qs)+chunk)*nh+h)*qs+i)*qs+inner))
            dl[i] = ftz(identical_mul(dot, lv))
        i += M3_HEADDIM
    barrier()
    var dv = Float32(0.0)
    for i2 in range(inner + 1, qs):
        var ti2 = chunk * qs + i2
        if ti2 < l:
            dv = ftz(identical_mul_add(ftz(d_y.unsafe_load(((bb*l+ti2)*nh+h)*M3_HEADDIM+p)), dl[i2], dv))
    d_v.unsafe_store(rowh * M3_HEADDIM + p, dv)


def mamba3_s15_backward_kernel(
    d_krot: MutPointer[Float32, MutAnyOrigin], d_scale: MutPointer[Float32, MutAnyOrigin],
    d_kscaled: MutPointer[Float32, MutAnyOrigin], krot: MutPointer[Float32, MutAnyOrigin],
    scale: MutPointer[Float32, MutAnyOrigin], rows_in: Int32,
):
    var rows = Int(rows_in); var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= rows: return
    var acc = Float32(0.0); var sc = ftz(scale.unsafe_load(row))
    for n in range(M3_D_STATE):
        var cell = row * M3_D_STATE + n; var dk = ftz(d_kscaled.unsafe_load(cell)); var kr = ftz(krot.unsafe_load(cell))
        d_krot.unsafe_store(cell, ftz(identical_mul(dk, sc)))
        acc = ftz(identical_mul_add(dk, kr, acc))
    d_scale.unsafe_store(row, acc)


def mamba3_backward_s16_s15_into(
    ctx: DeviceContext, mut d_q: DeviceBuffer[DType.float32], mut d_ks: DeviceBuffer[DType.float32],
    mut d_v: DeviceBuffer[DType.float32], mut d_krot: DeviceBuffer[DType.float32], mut d_scale: DeviceBuffer[DType.float32],
    mut d_y: DeviceBuffer[DType.float32], mut q: DeviceBuffer[DType.float32], mut ks: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32], mut seg_l: DeviceBuffer[DType.float32], mut krot: DeviceBuffer[DType.float32],
    mut scale: DeviceBuffer[DType.float32], b: Int, l: Int, dims: Mamba3Dims, qsize: Int,
) raises:
    var qcells = b*l*dims.nheads*M3_D_STATE; var vcells = b*l*dims.nheads*M3_HEADDIM
    var cells = qcells if qcells > vcells else vcells
    comptime if M3_S16_V_SHARED:
        if qsize > M3_S16_V_MAXQ:
            raise Error("mamba3 backward S16: chunk size " + String(qsize) + " above the shared d_v page (" + String(M3_S16_V_MAXQ) + ")")
        comptime if M3_S16_QK_SHARED:
            var ton = String(getenv("MOJOLEARN_MAMBA_TIMING")) != ""
            var tk = Int(perf_counter_ns())
            var arm = m3_s16_qk_arm()
            if arm == 0:
                ctx.enqueue_function[mamba3_s16_qk_backward_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_v.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(_grid(cells),1,1), block_dim=(M3_BWD_TPB,1,1))
                _s16_tick(ctx, ton, tk, "qk_naive")
            else:
                if qsize > M3_S16_QK_MAXQ:
                    raise Error("mamba3 backward S16: chunk size " + String(qsize) + " above the shared q/k page (" + String(M3_S16_QK_MAXQ) + ")")
                var took = False
                comptime if M3_S16_SMEM48_FITS:
                    if arm == 3:
                        ctx.enqueue_function[mamba3_s16_qk_smem48_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(b*l*dims.nheads,1,1), block_dim=(M3_D_STATE,1,1))
                        _s16_tick(ctx, ton, tk, "qk_smem48")
                        took = True
                comptime if M3_S16_QK_REGS2_FITS:
                    if not took and arm == 4:
                        ctx.enqueue_function[mamba3_s16_qk_regs2_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(b*l*dims.nheads,1,1), block_dim=(M3_D_STATE,1,1))
                        _s16_tick(ctx, ton, tk, "qk_regs2")
                        took = True
                comptime if M3_S16_QK_REGS2H_FITS:
                    if not took and arm == 5:
                        ctx.enqueue_function[mamba3_s16_qk_regs2h_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(b*l*dims.nheads,1,1), block_dim=(M3_D_STATE,1,1))
                        _s16_tick(ctx, ton, tk, "qk_regs2h")
                        took = True
                comptime if M3_S16_QK_REGSH_FITS:
                    if not took and arm == 6:
                        ctx.enqueue_function[mamba3_s16_qk_regsh_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(b*l*dims.nheads,1,1), block_dim=(M3_D_STATE,1,1))
                        _s16_tick(ctx, ton, tk, "qk_regsh")
                        took = True
                comptime if M3_S16_QK_PAGE_FITS:
                    if not took:
                        if arm == 1:
                            ctx.enqueue_function[mamba3_s16_qk_shared_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(b*l*dims.nheads,1,1), block_dim=(M3_D_STATE,1,1))
                            _s16_tick(ctx, ton, tk, "qk_shared")
                        else:
                            ctx.enqueue_function[mamba3_s16_qk_regs_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(b*l*dims.nheads,1,1), block_dim=(M3_D_STATE,1,1))
                            _s16_tick(ctx, ton, tk, "qk_regs")
                        took = True
                if not took:
                    ctx.enqueue_function[mamba3_s16_qk_backward_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_v.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(_grid(cells),1,1), block_dim=(M3_BWD_TPB,1,1))
                    _s16_tick(ctx, ton, tk, "qk_naive_nofit")
        else:
            ctx.enqueue_function[mamba3_s16_qk_backward_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_v.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(_grid(qcells),1,1), block_dim=(M3_BWD_TPB,1,1))
        ctx.enqueue_function[mamba3_s16_v_shared_kernel](d_v.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(b*l*dims.nheads,1,1), block_dim=(M3_HEADDIM,1,1))
        comptime if M3_S16_QK_SHARED:
            var tv = Int(perf_counter_ns())
            _s16_tick(ctx, String(getenv("MOJOLEARN_MAMBA_TIMING")) != "", tv, "v_shared")
    else:
        ctx.enqueue_function[mamba3_s16_qkv_backward_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_v.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(_grid(cells),1,1), block_dim=(M3_BWD_TPB,1,1))
    ctx.enqueue_function[mamba3_s15_backward_kernel](d_krot.unsafe_ptr(), d_scale.unsafe_ptr(), d_ks.unsafe_ptr(), krot.unsafe_ptr(), scale.unsafe_ptr(), Int32(b*l*dims.nheads), grid_dim=(_grid(b*l*dims.nheads),1,1), block_dim=(M3_BWD_TPB,1,1))


def mamba3_join_value_scale_kernel(
    d_value: MutPointer[Float32, MutAnyOrigin],
    d_gamma: MutPointer[Float32, MutAnyOrigin],
    d_beta: MutPointer[Float32, MutAnyOrigin],
    d_value_skip: MutPointer[Float32, MutAnyOrigin],
    d_value_s16: MutPointer[Float32, MutAnyOrigin],
    d_scale: MutPointer[Float32, MutAnyOrigin],
    value_cells_in: Int32,
    scale_cells_in: Int32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < Int(value_cells_in):
        d_value.unsafe_store(cell, ftz(ftz(d_value_skip.unsafe_load(cell)) + ftz(d_value_s16.unsafe_load(cell))))
    if cell < Int(scale_cells_in):
        var ds = ftz(d_scale.unsafe_load(cell))
        d_gamma.unsafe_store(cell, ds)
        d_beta.unsafe_store(cell, ds)


def mamba3_rotary_backward_kernel(
    d_qraw: MutPointer[Float32, MutAnyOrigin], d_kraw: MutPointer[Float32, MutAnyOrigin],
    d_theta: MutPointer[Float32, MutAnyOrigin], d_qrot: MutPointer[Float32, MutAnyOrigin],
    d_krot: MutPointer[Float32, MutAnyOrigin], bcb: MutPointer[Float32, MutAnyOrigin],
    bcc: MutPointer[Float32, MutAnyOrigin], b_bias: MutPointer[Float32, MutAnyOrigin],
    c_bias: MutPointer[Float32, MutAnyOrigin], theta: MutPointer[Float32, MutAnyOrigin],
    pairs_in: Int32, nh_in: Int32,
):
    var pairs = Int(pairs_in); var nh = Int(nh_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= pairs: return
    var pair = cell % (M3_D_STATE // 2); var rowh = cell // (M3_D_STATE // 2)
    var h = rowh % nh; var token = rowh // nh; var e0 = 2*pair; var e1 = e0+1
    var base = rowh*M3_D_STATE
    var dq0 = ftz(d_qrot.unsafe_load(base+e0)); var dq1 = ftz(d_qrot.unsafe_load(base+e1))
    var dk0 = ftz(d_krot.unsafe_load(base+e0)); var dk1 = ftz(d_krot.unsafe_load(base+e1))
    if pair >= M3_NUM_ROPE_ANGLES:
        d_qraw.unsafe_store(base+e0,dq0); d_qraw.unsafe_store(base+e1,dq1)
        d_kraw.unsafe_store(base+e0,dk0); d_kraw.unsafe_store(base+e1,dk1); return
    var th = ftz(theta.unsafe_load(rowh*M3_NUM_ROPE_ANGLES+pair))
    var c = ftz(portable_cosf(th)); var s = ftz(portable_sinf(th))
    d_qraw.unsafe_store(base+e0,ftz(ftz(identical_mul(dq0,c))+ftz(identical_mul(dq1,s))))
    d_qraw.unsafe_store(base+e1,ftz(ftz(identical_mul(dq1,c))-ftz(identical_mul(dq0,s))))
    d_kraw.unsafe_store(base+e0,ftz(ftz(identical_mul(dk0,c))+ftz(identical_mul(dk1,s))))
    d_kraw.unsafe_store(base+e1,ftz(ftz(identical_mul(dk1,c))-ftz(identical_mul(dk0,s))))
    var q0 = ftz(ftz(bcc.unsafe_load(token*M3_D_STATE+e0))+ftz(c_bias.unsafe_load(h*M3_D_STATE+e0)))
    var q1 = ftz(ftz(bcc.unsafe_load(token*M3_D_STATE+e1))+ftz(c_bias.unsafe_load(h*M3_D_STATE+e1)))
    var k0 = ftz(ftz(bcb.unsafe_load(token*M3_D_STATE+e0))+ftz(b_bias.unsafe_load(h*M3_D_STATE+e0)))
    var k1 = ftz(ftz(bcb.unsafe_load(token*M3_D_STATE+e1))+ftz(b_bias.unsafe_load(h*M3_D_STATE+e1)))
    var dt = Float32(0.0)
    dt = ftz(identical_mul_add(dq0,ftz(-ftz(identical_mul(q0,s))-ftz(identical_mul(q1,c))),dt))
    dt = ftz(identical_mul_add(dq1,ftz(ftz(identical_mul(q0,c))-ftz(identical_mul(q1,s))),dt))
    dt = ftz(identical_mul_add(dk0,ftz(-ftz(identical_mul(k0,s))-ftz(identical_mul(k1,c))),dt))
    dt = ftz(identical_mul_add(dk1,ftz(ftz(identical_mul(k0,c))-ftz(identical_mul(k1,s))),dt))
    d_theta.unsafe_store(rowh*M3_NUM_ROPE_ANGLES+pair,dt)


def mamba3_backward_join_rotary_into(
    ctx: DeviceContext, mut d_value: DeviceBuffer[DType.float32], mut d_gamma: DeviceBuffer[DType.float32], mut d_beta: DeviceBuffer[DType.float32],
    mut d_qraw: DeviceBuffer[DType.float32], mut d_kraw: DeviceBuffer[DType.float32], mut d_theta: DeviceBuffer[DType.float32],
    mut d_value_skip: DeviceBuffer[DType.float32], mut d_value_s16: DeviceBuffer[DType.float32], mut d_scale: DeviceBuffer[DType.float32],
    mut d_qrot: DeviceBuffer[DType.float32], mut d_krot: DeviceBuffer[DType.float32], mut bcb: DeviceBuffer[DType.float32], mut bcc: DeviceBuffer[DType.float32],
    mut b_bias: DeviceBuffer[DType.float32], mut c_bias: DeviceBuffer[DType.float32], mut theta: DeviceBuffer[DType.float32], m: Int, dims: Mamba3Dims,
) raises:
    var vc=m*dims.d_inner; var sc=m*dims.nheads; var cells=vc if vc>sc else sc
    ctx.enqueue_function[mamba3_join_value_scale_kernel](d_value.unsafe_ptr(),d_gamma.unsafe_ptr(),d_beta.unsafe_ptr(),d_value_skip.unsafe_ptr(),d_value_s16.unsafe_ptr(),d_scale.unsafe_ptr(),Int32(vc),Int32(sc),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))
    var pairs=m*dims.nheads*(M3_D_STATE//2)
    ctx.enqueue_function[mamba3_rotary_backward_kernel](d_qraw.unsafe_ptr(),d_kraw.unsafe_ptr(),d_theta.unsafe_ptr(),d_qrot.unsafe_ptr(),d_krot.unsafe_ptr(),bcb.unsafe_ptr(),bcc.unsafe_ptr(),b_bias.unsafe_ptr(),c_bias.unsafe_ptr(),theta.unsafe_ptr(),Int32(pairs),Int32(dims.nheads),grid_dim=(_grid(pairs),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_join_bc_kernel(
    out_b: MutPointer[Float32, MutAnyOrigin], out_c: MutPointer[Float32, MutAnyOrigin],
    qk_b: MutPointer[Float32, MutAnyOrigin], qk_c: MutPointer[Float32, MutAnyOrigin],
    rot_b: MutPointer[Float32, MutAnyOrigin], rot_c: MutPointer[Float32, MutAnyOrigin], n_in: Int32,
):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i>=Int(n_in): return
    out_b.unsafe_store(i,ftz(ftz(qk_b.unsafe_load(i))+ftz(rot_b.unsafe_load(i))))
    out_c.unsafe_store(i,ftz(ftz(qk_c.unsafe_load(i))+ftz(rot_c.unsafe_load(i))))


def mamba3_beta_join_kernel(
    out_gamma: MutPointer[Float32, MutAnyOrigin], out_dt: MutPointer[Float32, MutAnyOrigin],
    out_trap: MutPointer[Float32, MutAnyOrigin], qk_gamma: MutPointer[Float32, MutAnyOrigin],
    scale_gamma: MutPointer[Float32, MutAnyOrigin], qk_dt: MutPointer[Float32, MutAnyOrigin],
    qk_trap: MutPointer[Float32, MutAnyOrigin], d_beta: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin], sigma: MutPointer[Float32, MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,
):
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in)
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i>=b*l*nh:return
    var li=(i//nh)%l
    # scale = gamma + beta, gamma = dt * sigma. Both the diagonal qk
    # branch and the off-diagonal scale branch reach dt and trap_raw.
    var sig=ftz(sigma.unsafe_load(i));var dtv=ftz(dt.unsafe_load(i))
    var ds_gamma=ftz(scale_gamma.unsafe_load(i))
    var ddt=ftz(ftz(qk_dt.unsafe_load(i))+ftz(identical_mul(ds_gamma,sig)))
    var scale_trap=ftz(identical_mul(ftz(identical_mul(ftz(identical_mul(ds_gamma,dtv)),sig)),ftz(Float32(1.0)-sig)))
    var dtr=ftz(ftz(qk_trap.unsafe_load(i))+scale_trap)
    if li>0:
        var db=ftz(d_beta.unsafe_load(i-nh))
        ddt=ftz(ddt+ftz(identical_mul(db,ftz(Float32(1.0)-sig))))
        var ds=ftz(-ftz(identical_mul(db,dtv)))
        dtr=ftz(dtr+ftz(identical_mul(ftz(identical_mul(ds,sig)),ftz(Float32(1.0)-sig))))
    out_gamma.unsafe_store(i,ftz(ftz(qk_gamma.unsafe_load(i))+ftz(scale_gamma.unsafe_load(i))))
    out_dt.unsafe_store(i,ddt);out_trap.unsafe_store(i,dtr)


def mamba3_backward_join_current_into(
    ctx:DeviceContext,mut out_b:DeviceBuffer[DType.float32],mut out_c:DeviceBuffer[DType.float32],
    mut out_gamma:DeviceBuffer[DType.float32],mut out_dt:DeviceBuffer[DType.float32],mut out_trap:DeviceBuffer[DType.float32],
    mut qk_b:DeviceBuffer[DType.float32],mut qk_c:DeviceBuffer[DType.float32],mut rot_b:DeviceBuffer[DType.float32],mut rot_c:DeviceBuffer[DType.float32],
    mut qk_gamma:DeviceBuffer[DType.float32],mut scale_gamma:DeviceBuffer[DType.float32],mut qk_dt:DeviceBuffer[DType.float32],mut qk_trap:DeviceBuffer[DType.float32],
    mut d_beta:DeviceBuffer[DType.float32],mut dt:DeviceBuffer[DType.float32],mut sigma:DeviceBuffer[DType.float32],b:Int,l:Int,dims:Mamba3Dims,
) raises:
    var bc=b*l*dims.nheads*M3_D_STATE;var hs=b*l*dims.nheads
    ctx.enqueue_function[mamba3_join_bc_kernel](out_b.unsafe_ptr(),out_c.unsafe_ptr(),qk_b.unsafe_ptr(),qk_c.unsafe_ptr(),rot_b.unsafe_ptr(),rot_c.unsafe_ptr(),Int32(bc),grid_dim=(_grid(bc),1,1),block_dim=(M3_BWD_TPB,1,1))
    ctx.enqueue_function[mamba3_beta_join_kernel](out_gamma.unsafe_ptr(),out_dt.unsafe_ptr(),out_trap.unsafe_ptr(),qk_gamma.unsafe_ptr(),scale_gamma.unsafe_ptr(),qk_dt.unsafe_ptr(),qk_trap.unsafe_ptr(),d_beta.unsafe_ptr(),dt.unsafe_ptr(),sigma.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),grid_dim=(_grid(hs),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_theta_reverse_kernel(
    d_rate: MutPointer[Float32, MutAnyOrigin], d_theta: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin], b_in:Int32,l_in:Int32,nh_in:Int32,
):
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in)
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=b*nh*M3_NUM_ROPE_ANGLES:return
    var r=cell%M3_NUM_ROPE_ANGLES;var bh=cell//M3_NUM_ROPE_ANGLES
    var h=bh%nh;var bb=bh//nh;var carry=Float32(0.0)
    for rev in range(l):
        var t=l-1-rev;var rowh=(bb*l+t)*nh+h
        carry=ftz(carry+ftz(d_theta.unsafe_load(rowh*M3_NUM_ROPE_ANGLES+r)))
        d_rate.unsafe_store(rowh*M3_NUM_ROPE_ANGLES+r,ftz(identical_mul(carry,ftz(dt.unsafe_load(rowh)))))


#: lane/neural-net-experiment (2026-09-30, the S16 pass): the angle stage's
#: d_dt half. `mamba3_angle_reduce_kernel`'s second half is one thread per
#: (token, head) folding, for each of the 32 angles, the suffix of d_theta
#: from its token to the end of the sequence -- 8k dependent adds a thread
#: at L = 512, each a device load 768 bytes from the last (a new line each),
#: on 6,144 threads: the L40S priced the stage at 2.0 ms (5% of the pass).
#: The folds are pinned (each token's suffix ascending from its own index,
#: so no prefix is shared between tokens) and stay exactly as they are;
#: what moves is WHERE a value waits: one block per (batch, head) stages
#: each angle's whole column of d_theta once (L words), and each thread
#: folds its token's suffixes from threadgroup memory. SAME BITS.
#: Columns up to M3_ANGLE_DT_MAXL words take it; longer sequences keep the
#: old kernel. MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE=1 (a build define) keeps
#: the old kernel for the A/B.
comptime M3_ANGLE_DT_SHARED = not is_defined["MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE"]()
comptime M3_ANGLE_DT_MAXL = 4096
comptime M3_ANGLE_DT_TPB = 256


def mamba3_angle_dt_shared_kernel(
    d_dt: MutPointer[Float32, MutAnyOrigin], d_theta: MutPointer[Float32, MutAnyOrigin],
    angle_raw: MutPointer[Float32, MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,dip_in:Int32,col_angle_in:Int32,
):
    """`mamba3_angle_reduce_kernel`'s d_dt half over a staged column: block
    (batch, head), thread `tid` the tokens `tid, tid + TPB, ...`; for each
    angle r the column d_theta[:, h, r] is staged, then each token folds
    its suffix ascending from itself, as the naive kernel does."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var dip=Int(dip_in);var ca=Int(col_angle_in)
    var col=stack_allocation[M3_ANGLE_DT_MAXL,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var blk=Int(block_idx.x);var tid=Int(thread_idx.x)
    if blk>=b*nh:return
    var h=blk%nh;var bb=blk//nh
    # every token's running accdt, over the angles in order
    var acc=stack_allocation[16,Scalar[DType.float32]]()
    var slots=(l+M3_ANGLE_DT_TPB-1)//M3_ANGLE_DT_TPB
    for sl in range(16):
        acc[sl]=Float32(0.0)
    for r in range(M3_NUM_ROPE_ANGLES):
        var u=tid
        while u<l:
            col[u]=ftz(d_theta.unsafe_load(((bb*l+u)*nh+h)*M3_NUM_ROPE_ANGLES+r))
            u+=M3_ANGLE_DT_TPB
        barrier()
        for sl in range(slots):
            var li=tid+sl*M3_ANGLE_DT_TPB
            if li<l:
                var token=bb*l+li
                var raw=ftz(angle_raw.unsafe_load(token*dip+ca+r));var rate=ftz(identical_mul(ftz(identical_tanh(raw)),M3_PI))
                var carry=Float32(0.0)
                for uu in range(li,l):
                    carry=ftz(carry+col[uu])
                acc[sl]=ftz(identical_mul_add(carry,rate,acc[sl]))
        barrier()
    for sl in range(slots):
        var li=tid+sl*M3_ANGLE_DT_TPB
        if li<l:
            d_dt.unsafe_store((bb*l+li)*nh+h,acc[sl])


def mamba3_angle_reduce_kernel(
    d_angle: MutPointer[Float32, MutAnyOrigin], d_dt: MutPointer[Float32, MutAnyOrigin],
    d_rate: MutPointer[Float32, MutAnyOrigin], d_theta: MutPointer[Float32, MutAnyOrigin],
    angle_raw: MutPointer[Float32, MutAnyOrigin], dt: MutPointer[Float32, MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,dip_in:Int32,col_angle_in:Int32,do_dt_in:Int32,
):
    var b=Int(b_in);var l=Int(l_in);var m=b*l;var nh=Int(nh_in);var dip=Int(dip_in);var ca=Int(col_angle_in)
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell<m*M3_NUM_ROPE_ANGLES:
        var r=cell%M3_NUM_ROPE_ANGLES;var token=cell//M3_NUM_ROPE_ANGLES;var acc=Float32(0.0)
        for h in range(nh):
            var dr=ftz(d_rate.unsafe_load((token*nh+h)*M3_NUM_ROPE_ANGLES+r))
            acc=ftz(acc+dr)
        var raw=ftz(angle_raw.unsafe_load(token*dip+ca+r));var tv=ftz(identical_tanh(raw))
        var prime=ftz(identical_mul(M3_PI,ftz(Float32(1.0)-ftz(identical_mul(tv,tv)))))
        d_angle.unsafe_store(cell,ftz(identical_mul(acc,prime)))
    if cell<m*nh and do_dt_in!=Int32(0):
        var token=cell//nh;var h=cell%nh;var bb=token//l;var li=token%l;var accdt=Float32(0.0)
        for r in range(M3_NUM_ROPE_ANGLES):
            var raw=ftz(angle_raw.unsafe_load(token*dip+ca+r));var rate=ftz(identical_mul(ftz(identical_tanh(raw)),M3_PI))
            var carry=Float32(0.0)
            for u in range(li,l):
                carry=ftz(carry+ftz(d_theta.unsafe_load(((bb*l+u)*nh+h)*M3_NUM_ROPE_ANGLES+r)))
            accdt=ftz(identical_mul_add(carry,rate,accdt))
        d_dt.unsafe_store(cell,accdt)


# ---------------------------------------------------------------------------
# lane nr-mamba (2026-10-04, roadmap B3 as the review corrected it)
# IDN_M3_ANGLE_DT_SUFFIX (default ON; `-D MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF`
# or `-D MOJOLEARN_IDN_ALL_OFF` restores main). Every numeric mode on every
# column but the Apple FAST chunk arm (AFN_M3_BWD_CHUNK keeps its own).
#
# Main folded, for each (token, head, angle), the suffix d_theta[li..L-1]
# ascending from +0.0 in one thread: O(L^2) adds per (head, angle). Here
# each (batch, head, angle) chain's reverse cumulative sum is computed ONCE
# and every token reads its own entry:
#   carry(t) = sum over u >= t of d_theta(u), as fixed chunks of
#   M3_ANGLE_SUFFIX_CHUNK tokens pinned to absolute position t (chunk k is
#   tokens [64k, 64k + 64)): each chunk's total folds descending from +0.0;
#   a token's carry is (the totals of the later chunks, folded descending
#   from +0.0) seeded into its own chunk's descending walk.
# d_rate = carry * dt (main's theta_reverse product) and
# d_dt = fold over the 32 angles ascending of fma(carry, rate, acc) (main's
# chain over r, from the new carry). The chunk is a constant, not a shape
# rule, and no fold depends on the launch.
# BITS CHANGE: d_rate and d_dt, so d_angle, d_dt_bias, d_dt_raw, the in_proj
# dt and angle columns, d_W_in and through d_x every upstream gradient and
# the optimizer state. One source serves the device columns (NVIDIA, AMD,
# Metal; IDENTICAL and FAST on NVIDIA/AMD), the generated host column and the
# backward checks generated from it; both prefill-backward call sites and the
# tail dump pass the chunk-sum scratch.
# ---------------------------------------------------------------------------
comptime IDN_M3_ANGLE_DT_SUFFIX = (
    not is_defined["MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime M3_ANGLE_SUFFIX_CHUNK = 64
# NI43: linear-work scheduling of the existing versioned suffix.
# Each chain folds chunk totals descending exactly once, storing the exclusive
# carry before adding this chunk. In-place scratch is safe: one owner per chain,
# one launch boundary before consumers. No token/angle fold or gradient changes.
# Promoted 2026-10-08 (lane grid-act-4, IDENTICAL grid ge123e6f9, one run per
# arm, incumbent -> carry cache ms): samba-train-step NV 96.0 -> 76.9, AMD
# 147.2 -> 143.5 (0.883x combined); output hashes equal to the incumbent on
# both vendors, so no bit moves. Each chain does linear work in its chunk count
# instead of re-summing every later chunk, so the win grows with sequence
# length. Measured alone; it now combines with the promoted resident Samba step
# and act_retain=2, and the post-merge race measures the combination. Default
# on in IDENTICAL; -D MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF restores the
# per-chunk suffix re-walk.
comptime IDN_M3_ANGLE_CARRY_CACHE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_M3_ANGLE_DT_SUFFIX
    and not is_defined["MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def mamba3_angle_suffix_sums_cells(b: Int, l: Int, nh: Int) -> Int:
    """Floats of the chunk-sum scratch `mamba3_backward_angle_into` takes."""
    var k = (l + M3_ANGLE_SUFFIX_CHUNK - 1) // M3_ANGLE_SUFFIX_CHUNK
    if k < 1:
        k = 1
    return b * nh * M3_NUM_ROPE_ANGLES * k


def mamba3_angle_chunk_sum_kernel(
    sums: MutPointer[Float32, MutAnyOrigin], d_theta: MutPointer[Float32, MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,
):
    """One thread per (chain, chunk): the chunk's d_theta total, descending
    from +0.0. sums[chain * K + k], chain = (b * nh + h) * R + r."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in)
    var nk=(l+M3_ANGLE_SUFFIX_CHUNK-1)//M3_ANGLE_SUFFIX_CHUNK
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=b*nh*M3_NUM_ROPE_ANGLES*nk:return
    var k=cell%nk;var chain=cell//nk
    var r=chain%M3_NUM_ROPE_ANGLES;var bh=chain//M3_NUM_ROPE_ANGLES
    var h=bh%nh;var bb=bh//nh
    var t0=k*M3_ANGLE_SUFFIX_CHUNK;var t1=t0+M3_ANGLE_SUFFIX_CHUNK
    if t1>l:t1=l
    var s=Float32(0.0)
    var t=t1-1
    while t>=t0:
        s=ftz(s+ftz(d_theta.unsafe_load(((bb*l+t)*nh+h)*M3_NUM_ROPE_ANGLES+r)))
        t-=1
    sums.unsafe_store(cell,s)


# NN38 (retired define MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS) was the same
# scheduling arm as NI43: both enabled exactly the seeds kernel below. One
# switch, MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE (L11 dedupe, 2026-10-07).


def nn38_angle_chunk_seeds_kernel(sums: MutPointer[Float32, MutAnyOrigin], chains_in: Int32, nk_in: Int32):
    """One owner per chain replaces each chunk total by its exclusive
    descending suffix. Read original total before overwriting its cell.
    The accumulator visits nk-1..k+1 exactly as each incumbent task did."""
    var chain = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if chain >= Int(chains_in):
        return
    var nk = Int(nk_in)
    var carry = Float32(0.0)
    var k = nk - 1
    while k >= 0:
        var v = ftz(sums.unsafe_load(chain * nk + k))
        sums.unsafe_store(chain * nk + k, carry)
        carry = ftz(carry + v)
        k -= 1


def mamba3_angle_suffix_kernel(
    carry_out: MutPointer[Float32, MutAnyOrigin], d_theta: MutPointer[Float32, MutAnyOrigin],
    sums: MutPointer[Float32, MutAnyOrigin], b_in:Int32,l_in:Int32,nh_in:Int32,
):
    """One thread per (chain, chunk): seed = the later chunks' totals
    folded descending from +0.0, then the chunk walked descending; each
    token's carry lands in carry_out (the d_rate buffer, [B, L, H, R])."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in)
    var nk=(l+M3_ANGLE_SUFFIX_CHUNK-1)//M3_ANGLE_SUFFIX_CHUNK
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=b*nh*M3_NUM_ROPE_ANGLES*nk:return
    var k=cell%nk;var chain=cell//nk
    var r=chain%M3_NUM_ROPE_ANGLES;var bh=chain//M3_NUM_ROPE_ANGLES
    var h=bh%nh;var bb=bh//nh
    var carry=Float32(0.0)
    comptime if IDN_M3_ANGLE_CARRY_CACHE:
        carry=ftz(sums.unsafe_load(chain*nk+k))
    else:
        var kk=nk-1
        while kk>k:
            carry=ftz(carry+ftz(sums.unsafe_load(chain*nk+kk)))
            kk-=1
    var t0=k*M3_ANGLE_SUFFIX_CHUNK;var t1=t0+M3_ANGLE_SUFFIX_CHUNK
    if t1>l:t1=l
    var t=t1-1
    while t>=t0:
        var idx=((bb*l+t)*nh+h)*M3_NUM_ROPE_ANGLES+r
        carry=ftz(carry+ftz(d_theta.unsafe_load(idx)))
        carry_out.unsafe_store(idx,carry)
        t-=1


def mamba3_angle_suffix_dt_kernel(
    d_dt: MutPointer[Float32, MutAnyOrigin], d_rate: MutPointer[Float32, MutAnyOrigin],
    angle_raw: MutPointer[Float32, MutAnyOrigin], dt: MutPointer[Float32, MutAnyOrigin],
    m_in:Int32,nh_in:Int32,dip_in:Int32,col_angle_in:Int32,
):
    """One thread per (token, head): d_dt = fold over r ascending of
    fma(carry, rate, acc) from +0.0, then the row's carries become
    d_rate = carry * dt in place (this thread alone reads and writes them)."""
    var m=Int(m_in);var nh=Int(nh_in);var dip=Int(dip_in);var ca=Int(col_angle_in)
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=m*nh:return
    var token=cell//nh
    var dtv=ftz(dt.unsafe_load(cell))
    var accdt=Float32(0.0)
    for r in range(M3_NUM_ROPE_ANGLES):
        var raw=ftz(angle_raw.unsafe_load(token*dip+ca+r));var rate=ftz(identical_mul(ftz(identical_tanh(raw)),M3_PI))
        var carry=ftz(d_rate.unsafe_load(cell*M3_NUM_ROPE_ANGLES+r))
        accdt=ftz(identical_mul_add(carry,rate,accdt))
        d_rate.unsafe_store(cell*M3_NUM_ROPE_ANGLES+r,ftz(identical_mul(carry,dtv)))
    d_dt.unsafe_store(cell,accdt)


def mamba3_backward_angle_into(
    ctx:DeviceContext,mut d_rate:DeviceBuffer[DType.float32],mut d_angle:DeviceBuffer[DType.float32],mut d_dt:DeviceBuffer[DType.float32],
    mut d_theta:DeviceBuffer[DType.float32],mut dt:DeviceBuffer[DType.float32],mut in_proj:DeviceBuffer[DType.float32],b:Int,l:Int,dims:Mamba3Dims,
    mut sums:DeviceBuffer[DType.float32],
) raises:
    """`sums` holds `mamba3_angle_suffix_sums_cells(b, l, nh)` floats (read
    only under IDN_M3_ANGLE_DT_SUFFIX)."""
    comptime if IDN_M3_ANGLE_DT_SUFFIX:
        var nk=(l+M3_ANGLE_SUFFIX_CHUNK-1)//M3_ANGLE_SUFFIX_CHUNK
        var work=b*dims.nheads*M3_NUM_ROPE_ANGLES*nk
        ctx.enqueue_function[mamba3_angle_chunk_sum_kernel](sums.unsafe_ptr(),d_theta.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),grid_dim=(_grid(work),1,1),block_dim=(M3_BWD_TPB,1,1))
        comptime if IDN_M3_ANGLE_CARRY_CACHE:
            var chains=b*dims.nheads*M3_NUM_ROPE_ANGLES
            ctx.enqueue_function[nn38_angle_chunk_seeds_kernel](sums.unsafe_ptr(),Int32(chains),Int32(nk),grid_dim=(_grid(chains),1,1),block_dim=(M3_BWD_TPB,1,1))
        ctx.enqueue_function[mamba3_angle_suffix_kernel](d_rate.unsafe_ptr(),d_theta.unsafe_ptr(),sums.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),grid_dim=(_grid(work),1,1),block_dim=(M3_BWD_TPB,1,1))
        var m=b*l
        ctx.enqueue_function[mamba3_angle_suffix_dt_kernel](d_dt.unsafe_ptr(),d_rate.unsafe_ptr(),in_proj.unsafe_ptr(),dt.unsafe_ptr(),Int32(m),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(dims.col_angle()),grid_dim=(_grid(m*dims.nheads),1,1),block_dim=(M3_BWD_TPB,1,1))
        # d_angle half only (do_dt = 0): it reads the finished d_rate.
        ctx.enqueue_function[mamba3_angle_reduce_kernel](d_angle.unsafe_ptr(),d_dt.unsafe_ptr(),d_rate.unsafe_ptr(),d_theta.unsafe_ptr(),in_proj.unsafe_ptr(),dt.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(dims.col_angle()),Int32(0),grid_dim=(_grid(m*M3_NUM_ROPE_ANGLES),1,1),block_dim=(M3_BWD_TPB,1,1))
        return
    var chains=b*dims.nheads*M3_NUM_ROPE_ANGLES
    ctx.enqueue_function[mamba3_theta_reverse_kernel](d_rate.unsafe_ptr(),d_theta.unsafe_ptr(),dt.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),grid_dim=(_grid(chains),1,1),block_dim=(M3_BWD_TPB,1,1))
    var m=b*l;var cells=m*M3_NUM_ROPE_ANGLES if m*M3_NUM_ROPE_ANGLES>m*dims.nheads else m*dims.nheads
    var dt_shared=False
    comptime if M3_ANGLE_DT_SHARED:
        dt_shared=l<=M3_ANGLE_DT_MAXL and (l+M3_ANGLE_DT_TPB-1)//M3_ANGLE_DT_TPB<=16
    ctx.enqueue_function[mamba3_angle_reduce_kernel](d_angle.unsafe_ptr(),d_dt.unsafe_ptr(),d_rate.unsafe_ptr(),d_theta.unsafe_ptr(),in_proj.unsafe_ptr(),dt.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(dims.col_angle()),Int32(0 if dt_shared else 1),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))
    if dt_shared:
        ctx.enqueue_function[mamba3_angle_dt_shared_kernel](d_dt.unsafe_ptr(),d_theta.unsafe_ptr(),in_proj.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(dims.col_angle()),grid_dim=(b*dims.nheads,1,1),block_dim=(M3_ANGLE_DT_TPB,1,1))


# ===========================================================================
# lane afn-samba (2026-10-03): MOJOLEARN_AFN_MAMBA3_BWD_CHUNK, Apple FAST only.
# The angle stage's two reverse-time chains, chunk-parallel with a free fold
# order (f32 throughout, no approximation):
#   * `mamba3_theta_reverse_kernel` walks each (batch, head, angle) chain's
#     whole sequence in one thread (768 chains of L dependent load-add-store
#     steps at the board shape). Here two launches over (chain, chunk of
#     AFN_M3_SCAN_CHUNK tokens): the chunk sums, then each chunk's own
#     reverse walk seeded with the sum of the later chunks (at most
#     L / chunk adds).
#   * `mamba3_angle_dt_shared_kernel` folds, for every token and angle, the
#     suffix of the staged column from its own index (L^2 / 2 dependent
#     shared-memory adds per angle per block). Here each thread owns a
#     contiguous token segment, scans it in place, publishes the segment
#     total, and every token's suffix is its in-segment suffix plus the
#     later segments' totals: O(L) per angle per block, two barriers.
# Every line is inside the AFN_M3_BWD_CHUNK guard; IDENTICAL and the other
# vendors compile the kernels above unchanged.
# ===========================================================================
comptime AFN_M3_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)
comptime AFN_M3_BWD_CHUNK = AFN_M3_APPLE_FAST and (
    is_defined["MOJOLEARN_AFN_MAMBA3_BWD_CHUNK"]()
    or is_defined["MOJOLEARN_AFN_SAMBA_ALL"]()
)
comptime AFN_M3_SCAN_CHUNK = 64


def afn_m3_theta_chunks(l: Int) -> Int:
    return (l + AFN_M3_SCAN_CHUNK - 1) // AFN_M3_SCAN_CHUNK


def afn_m3_theta_chunk_sum_kernel(
    sums: MutPointer[Float32, MutAnyOrigin],
    d_theta: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
):
    """`sums[chain, c]`: the sum of `d_theta` over chunk `c` of chain
    (batch, head, angle); one thread per (chain, chunk)."""
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nchunks = afn_m3_theta_chunks(l)
    var chains = Int(b_in) * nh * M3_NUM_ROPE_ANGLES
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= chains * nchunks:
        return
    var c = cell % nchunks
    var chain = cell // nchunks
    var r = chain % M3_NUM_ROPE_ANGLES
    var bh = chain // M3_NUM_ROPE_ANGLES
    var h = bh % nh
    var bb = bh // nh
    var t0 = c * AFN_M3_SCAN_CHUNK
    var t1 = t0 + AFN_M3_SCAN_CHUNK
    if t1 > l:
        t1 = l
    var s = Float32(0.0)
    for t in range(t0, t1):
        s += d_theta.unsafe_load(((bb * l + t) * nh + h) * M3_NUM_ROPE_ANGLES + r)
    sums.unsafe_store(cell, s)


def afn_m3_theta_chunk_apply_kernel(
    d_rate: MutPointer[Float32, MutAnyOrigin],
    d_theta: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin],
    sums: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
):
    """Chunk `c`'s reverse walk of its chain, seeded with the later chunks'
    sums: `d_rate[t] = (sum_{u >= t} d_theta[u]) * dt[t]`."""
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nchunks = afn_m3_theta_chunks(l)
    var chains = Int(b_in) * nh * M3_NUM_ROPE_ANGLES
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= chains * nchunks:
        return
    var c = cell % nchunks
    var chain = cell // nchunks
    var r = chain % M3_NUM_ROPE_ANGLES
    var bh = chain // M3_NUM_ROPE_ANGLES
    var h = bh % nh
    var bb = bh // nh
    var carry = Float32(0.0)
    for cc in range(c + 1, nchunks):
        carry += sums.unsafe_load(chain * nchunks + cc)
    var t0 = c * AFN_M3_SCAN_CHUNK
    var t = t0 + AFN_M3_SCAN_CHUNK - 1
    if t > l - 1:
        t = l - 1
    while t >= t0:
        var rowh = (bb * l + t) * nh + h
        var at = rowh * M3_NUM_ROPE_ANGLES + r
        carry += d_theta.unsafe_load(at)
        d_rate.unsafe_store(at, carry * dt.unsafe_load(rowh))
        t -= 1


comptime AFN_M3_ANGLE_SEG = M3_ANGLE_DT_MAXL // M3_ANGLE_DT_TPB  # 16 tokens a thread


def afn_m3_angle_dt_scan_kernel(
    d_dt: MutPointer[Float32, MutAnyOrigin],
    d_theta: MutPointer[Float32, MutAnyOrigin],
    angle_raw: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    col_angle_in: Int32,
):
    """`mamba3_angle_dt_shared_kernel`'s result by a segment scan: block
    (batch, head); thread `tid` owns tokens `[tid * seglen, (tid + 1) *
    seglen)`; per angle the column is staged, each thread scans its segment
    in place (reverse), publishes its total, and token `t`'s suffix is its
    in-segment suffix plus the totals of the later segments. `l` is at most
    M3_ANGLE_DT_MAXL (the launcher's gate)."""
    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var ca = Int(col_angle_in)
    var col = stack_allocation[
        M3_ANGLE_DT_MAXL, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var segt = stack_allocation[
        M3_ANGLE_DT_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if blk >= b * nh:
        return
    var h = blk % nh
    var bb = blk // nh
    var seglen = (l + M3_ANGLE_DT_TPB - 1) // M3_ANGLE_DT_TPB
    var s0 = tid * seglen
    var s1 = s0 + seglen
    if s1 > l:
        s1 = l
    var acc = stack_allocation[AFN_M3_ANGLE_SEG, Scalar[DType.float32]]()
    for i in range(AFN_M3_ANGLE_SEG):
        acc[i] = Float32(0.0)
    for r in range(M3_NUM_ROPE_ANGLES):
        var u = tid
        while u < l:
            col[u] = d_theta.unsafe_load(((bb * l + u) * nh + h) * M3_NUM_ROPE_ANGLES + r)
            u += M3_ANGLE_DT_TPB
        barrier()
        var carry = Float32(0.0)
        var i = s1 - 1
        while i >= s0:
            carry += col[i]
            col[i] = carry
            i -= 1
        segt[tid] = carry
        barrier()
        var cin = Float32(0.0)
        for j in range(tid + 1, M3_ANGLE_DT_TPB):
            cin += segt[j]
        for t in range(s0, s1):
            var token = bb * l + t
            var raw = angle_raw.unsafe_load(token * dip + ca + r)
            var rate = identical_tanh(raw) * M3_PI
            acc[t - s0] += (col[t] + cin) * rate
        barrier()
    for t in range(s0, s1):
        d_dt.unsafe_store((bb * l + t) * nh + h, acc[t - s0])


def mamba3_afn_backward_angle_into(
    ctx: DeviceContext,
    mut d_rate: DeviceBuffer[DType.float32],
    mut d_angle: DeviceBuffer[DType.float32],
    mut d_dt: DeviceBuffer[DType.float32],
    mut d_theta: DeviceBuffer[DType.float32],
    mut dt: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    mut sums: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    dims: Mamba3Dims,
) raises:
    """`mamba3_backward_angle_into` with the chunked chains. `sums` holds
    `b * nheads * M3_NUM_ROPE_ANGLES * afn_m3_theta_chunks(l)` floats and
    is the caller's (alive past its wait)."""
    var chains = b * dims.nheads * M3_NUM_ROPE_ANGLES
    var cells1 = chains * afn_m3_theta_chunks(l)
    ctx.enqueue_function[afn_m3_theta_chunk_sum_kernel](
        sums.unsafe_ptr(), d_theta.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads),
        grid_dim=(_grid(cells1), 1, 1),
        block_dim=(M3_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[afn_m3_theta_chunk_apply_kernel](
        d_rate.unsafe_ptr(), d_theta.unsafe_ptr(), dt.unsafe_ptr(), sums.unsafe_ptr(),
        Int32(b), Int32(l), Int32(dims.nheads),
        grid_dim=(_grid(cells1), 1, 1),
        block_dim=(M3_BWD_TPB, 1, 1),
    )
    var m = b * l
    var cells = m * M3_NUM_ROPE_ANGLES
    if m * dims.nheads > cells:
        cells = m * dims.nheads
    var dt_scan = l <= M3_ANGLE_DT_MAXL
    ctx.enqueue_function[mamba3_angle_reduce_kernel](
        d_angle.unsafe_ptr(), d_dt.unsafe_ptr(), d_rate.unsafe_ptr(), d_theta.unsafe_ptr(),
        in_proj.unsafe_ptr(), dt.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads),
        Int32(dims.d_in_proj()), Int32(dims.col_angle()), Int32(0 if dt_scan else 1),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(M3_BWD_TPB, 1, 1),
    )
    if dt_scan:
        ctx.enqueue_function[afn_m3_angle_dt_scan_kernel](
            d_dt.unsafe_ptr(), d_theta.unsafe_ptr(), in_proj.unsafe_ptr(),
            Int32(b), Int32(l), Int32(dims.nheads), Int32(dims.d_in_proj()),
            Int32(dims.col_angle()),
            grid_dim=(b * dims.nheads, 1, 1),
            block_dim=(M3_ANGLE_DT_TPB, 1, 1),
        )


def mamba3_dt_softplus_partial_kernel(
    d_dt_total: MutPointer[Float32, MutAnyOrigin],
    d_dt_raw: MutPointer[Float32, MutAnyOrigin],
    d_dt_bias_rows: MutPointer[Float32, MutAnyOrigin],
    d_dt_current: MutPointer[Float32, MutAnyOrigin],
    d_dt_angle: MutPointer[Float32, MutAnyOrigin],
    in_proj: MutPointer[Float32, MutAnyOrigin],
    dt_bias: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    col_dt_in: Int32,
):
    """Join available dt legs and apply S6 softplus' derivative."""
    var cells=Int(cells_in);var nh=Int(nh_in);var dip=Int(dip_in);var col=Int(col_dt_in)
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=cells:return
    var token=cell//nh;var h=cell%nh
    var total=ftz(ftz(d_dt_current.unsafe_load(cell))+ftz(d_dt_angle.unsafe_load(cell)))
    d_dt_total.unsafe_store(cell,total)
    var pre=ftz(ftz(in_proj.unsafe_load(token*dip+col+h))+ftz(dt_bias.unsafe_load(h)))
    var prime=Float32(1.0)
    if pre<=Float32(20.0):
        prime=ftz(identical_sigmoid(pre))
    var draw=ftz(identical_mul(total,prime))
    d_dt_raw.unsafe_store(cell,draw)
    d_dt_bias_rows.unsafe_store(cell,draw)


def mamba3_backward_dt_partial_into(
    ctx:DeviceContext,
    mut d_dt_total:DeviceBuffer[DType.float32],mut d_dt_raw:DeviceBuffer[DType.float32],mut d_dt_bias_rows:DeviceBuffer[DType.float32],
    mut d_dt_current:DeviceBuffer[DType.float32],mut d_dt_angle:DeviceBuffer[DType.float32],mut in_proj:DeviceBuffer[DType.float32],mut dt_bias:DeviceBuffer[DType.float32],
    m:Int,dims:Mamba3Dims,
) raises:
    var cells=m*dims.nheads
    ctx.enqueue_function[mamba3_dt_softplus_partial_kernel](d_dt_total.unsafe_ptr(),d_dt_raw.unsafe_ptr(),d_dt_bias_rows.unsafe_ptr(),d_dt_current.unsafe_ptr(),d_dt_angle.unsafe_ptr(),in_proj.unsafe_ptr(),dt_bias.unsafe_ptr(),Int32(cells),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(dims.col_dt()),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_s16_dseg_kernel(
    d_seg:MutPointer[Float32,MutAnyOrigin],d_y:MutPointer[Float32,MutAnyOrigin],
    q:MutPointer[Float32,MutAnyOrigin],k:MutPointer[Float32,MutAnyOrigin],v:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=b*nc*nh*qs*qs:return
    var j=cell%qs;var z=cell//qs;var i=z%qs;z=z//qs;var h=z%nh;z=z//nh;var c=z%nc;var bb=z//nc
    var ti=c*qs+i;var tj=c*qs+j
    if j>=i or ti>=l or tj>=l:
        d_seg.unsafe_store(cell,Float32(0.0));return
    var dot=Float32(0.0)
    for n in range(M3_D_STATE):
        dot=ftz(identical_mul_add(ftz(q.unsafe_load(((bb*l+ti)*nh+h)*M3_D_STATE+n)),ftz(k.unsafe_load(((bb*l+tj)*nh+h)*M3_D_STATE+n)),dot))
    var acc=Float32(0.0)
    for p in range(M3_HEADDIM):
        var dy=ftz(d_y.unsafe_load(((bb*l+ti)*nh+h)*M3_HEADDIM+p));var vv=ftz(v.unsafe_load(((bb*l+tj)*nh+h)*M3_HEADDIM+p))
        acc=ftz(identical_mul_add(dy,ftz(identical_mul(dot,vv)),acc))
    d_seg.unsafe_store(cell,acc)


def mamba3_seg_to_adt_kernel(
    d_adt:MutPointer[Float32,MutAnyOrigin],d_seg:MutPointer[Float32,MutAnyOrigin],seg:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=b*l*nh:return
    var h=cell%nh;var token=(cell//nh)%l;var bb=cell//(nh*l);var c=token//qs;var s=token%qs;var acc=Float32(0.0)
    for i in range(s,qs):
        if c*qs+i<l:
            for j in range(s):
                var idx=((((bb*nc+c)*nh+h)*qs+i)*qs+j)
                acc=ftz(identical_mul_add(ftz(d_seg.unsafe_load(idx)),ftz(seg.unsafe_load(idx)),acc))
    d_adt.unsafe_store(cell,acc)


def mamba3_backward_seg_adt_into(
    ctx:DeviceContext,mut d_seg:DeviceBuffer[DType.float32],mut d_adt:DeviceBuffer[DType.float32],mut d_y:DeviceBuffer[DType.float32],
    mut q:DeviceBuffer[DType.float32],mut k:DeviceBuffer[DType.float32],mut v:DeviceBuffer[DType.float32],mut seg:DeviceBuffer[DType.float32],
    b:Int,l:Int,dims:Mamba3Dims,qs:Int,
) raises:
    var nc=(l+qs-1)//qs;var segcells=b*nc*dims.nheads*qs*qs;var hcells=b*l*dims.nheads
    ctx.enqueue_function[mamba3_s16_dseg_kernel](d_seg.unsafe_ptr(),d_y.unsafe_ptr(),q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(_grid(segcells),1,1),block_dim=(M3_BWD_TPB,1,1))
    ctx.enqueue_function[mamba3_seg_to_adt_kernel](d_adt.unsafe_ptr(),d_seg.unsafe_ptr(),seg.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(_grid(hcells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_adt_product_backward_kernel(
    d_a:MutPointer[Float32,MutAnyOrigin],d_dt_from_adt:MutPointer[Float32,MutAnyOrigin],
    d_dt_with_seg:MutPointer[Float32,MutAnyOrigin],d_adt:MutPointer[Float32,MutAnyOrigin],
    a:MutPointer[Float32,MutAnyOrigin],dt:MutPointer[Float32,MutAnyOrigin],
    d_dt_available:MutPointer[Float32,MutAnyOrigin],cells_in:Int32,
):
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=Int(cells_in):return
    var da_dt=ftz(d_adt.unsafe_load(cell));var av=ftz(a.unsafe_load(cell));var dtv=ftz(dt.unsafe_load(cell))
    var da=ftz(identical_mul(da_dt,dtv));var ddt=ftz(identical_mul(da_dt,av))
    d_a.unsafe_store(cell,da);d_dt_from_adt.unsafe_store(cell,ddt)
    d_dt_with_seg.unsafe_store(cell,ftz(ftz(d_dt_available.unsafe_load(cell))+ddt))


def mamba3_backward_adt_product_into(
    ctx:DeviceContext,mut d_a:DeviceBuffer[DType.float32],mut d_dt_from_adt:DeviceBuffer[DType.float32],mut d_dt_with_seg:DeviceBuffer[DType.float32],
    mut d_adt:DeviceBuffer[DType.float32],mut a:DeviceBuffer[DType.float32],mut dt:DeviceBuffer[DType.float32],mut d_dt_available:DeviceBuffer[DType.float32],cells:Int,
) raises:
    ctx.enqueue_function[mamba3_adt_product_backward_kernel](d_a.unsafe_ptr(),d_dt_from_adt.unsafe_ptr(),d_dt_with_seg.unsafe_ptr(),d_adt.unsafe_ptr(),a.unsafe_ptr(),dt.unsafe_ptr(),d_dt_available.unsafe_ptr(),Int32(cells),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_a_heavy_tail_backward_kernel(d_raw:MutPointer[Float32,MutAnyOrigin],d_a:MutPointer[Float32,MutAnyOrigin],in_proj:MutPointer[Float32,MutAnyOrigin],cells_in:Int32,nh_in:Int32,dip_in:Int32,col_a_in:Int32):
    """Reverse S5 heavy-tail and its upper clamp; one owner per packed A cell."""
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=Int(cells_in):return
    var nh=Int(nh_in);var token=cell//nh;var h=cell%nh;var raw=ftz(in_proj.unsafe_load(token*Int(dip_in)+Int(col_a_in)+h));var prime=Float32(1.0);var ht:Float32
    if raw>=Float32(0.0):
        ht=ftz(Float32(1.0)+raw)
    else:
        var den=ftz(Float32(1.0)-raw);ht=ftz(identical_div(Float32(1.0),den));prime=ftz(identical_mul(ht,ht))
    if -ht> -M3_A_FLOOR:
        d_raw.unsafe_store(cell,Float32(0.0))
    else:
        d_raw.unsafe_store(cell,ftz(-ftz(identical_mul(ftz(d_a.unsafe_load(cell)),prime))))


def mamba3_backward_a_heavy_tail_into(ctx:DeviceContext,mut d_raw:DeviceBuffer[DType.float32],mut d_a:DeviceBuffer[DType.float32],mut in_proj:DeviceBuffer[DType.float32],m:Int,dims:Mamba3Dims) raises:
    var cells=m*dims.nheads
    ctx.enqueue_function[mamba3_a_heavy_tail_backward_kernel](d_raw.unsafe_ptr(),d_a.unsafe_ptr(),in_proj.unsafe_ptr(),Int32(cells),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(dims.col_a()),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_bcnorm_backward_kernel(dx:MutPointer[Float32,MutAnyOrigin],dwrow:MutPointer[Float32,MutAnyOrigin],dy:MutPointer[Float32,MutAnyOrigin],raw:MutPointer[Float32,MutAnyOrigin],weight:MutPointer[Float32,MutAnyOrigin],m_in:Int32,nh_in:Int32,dip_in:Int32,col_in:Int32):
    var t=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if t>=Int(m_in):return
    var ss=Float32(0.0)
    for n in range(M3_D_STATE):
        var x=ftz(raw.unsafe_load(t*Int(dip_in)+Int(col_in)+n));ss=ftz(identical_mul_add(x,x,ss))
    var r=ftz(identical_rsqrt(ftz(ftz(identical_div(ss,Float32(M3_D_STATE)))+M3_RMS_EPS)))
    var dot=Float32(0.0)
    for n in range(M3_D_STATE):
        var g=Float32(0.0)
        for h in range(Int(nh_in)):g=ftz(g+ftz(dy.unsafe_load((t*Int(nh_in)+h)*M3_D_STATE+n)))
        var x=ftz(raw.unsafe_load(t*Int(dip_in)+Int(col_in)+n));var gw=ftz(identical_mul(g,ftz(weight.unsafe_load(n))))
        dwrow.unsafe_store(t*M3_D_STATE+n,ftz(identical_mul(g,ftz(identical_mul(x,r)))));dot=ftz(identical_mul_add(gw,x,dot))
    var corr=ftz(identical_mul(ftz(identical_mul(r,r)),ftz(identical_div(dot,Float32(M3_D_STATE)))))
    for n in range(M3_D_STATE):
        var g=Float32(0.0)
        for h in range(Int(nh_in)):g=ftz(g+ftz(dy.unsafe_load((t*Int(nh_in)+h)*M3_D_STATE+n)))
        var x=ftz(raw.unsafe_load(t*Int(dip_in)+Int(col_in)+n));dx.unsafe_store(t*M3_D_STATE+n,ftz(identical_mul(r,ftz(ftz(identical_mul(g,ftz(weight.unsafe_load(n))))-ftz(identical_mul(x,corr))))))


def mamba3_backward_bcnorm_into(ctx:DeviceContext,mut dx:DeviceBuffer[DType.float32],mut dwrow:DeviceBuffer[DType.float32],mut dy:DeviceBuffer[DType.float32],mut raw:DeviceBuffer[DType.float32],mut weight:DeviceBuffer[DType.float32],m:Int,dims:Mamba3Dims,col:Int) raises:
    ctx.enqueue_function[mamba3_bcnorm_backward_kernel](dx.unsafe_ptr(),dwrow.unsafe_ptr(),dy.unsafe_ptr(),raw.unsafe_ptr(),weight.unsafe_ptr(),Int32(m),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(col),grid_dim=(_grid(m),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_pack_in_proj_backward_kernel(
    packed: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    dx: MutPointer[Float32, MutAnyOrigin],
    db: MutPointer[Float32, MutAnyOrigin],
    dc: MutPointer[Float32, MutAnyOrigin],
    ddt: MutPointer[Float32, MutAnyOrigin],
    da: MutPointer[Float32, MutAnyOrigin],
    dtrap: MutPointer[Float32, MutAnyOrigin],
    dangle: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    di_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
):
    """Pack the eight split adjoints in the normative projection order."""
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(cells_in):
        return
    var dip = Int(dip_in)
    var di = Int(di_in)
    var nh = Int(nh_in)
    var token = cell // dip
    var col = cell % dip
    var value: Float32
    if col < di:
        value = dz.unsafe_load(token * di + col)
    elif col < 2 * di:
        value = dx.unsafe_load(token * di + col - di)
    elif col < 2 * di + M3_D_STATE:
        value = db.unsafe_load(token * M3_D_STATE + col - 2 * di)
    elif col < 2 * di + 2 * M3_D_STATE:
        value = dc.unsafe_load(token * M3_D_STATE + col - 2 * di - M3_D_STATE)
    elif col < 2 * di + 2 * M3_D_STATE + nh:
        value = ddt.unsafe_load(token * nh + col - 2 * di - 2 * M3_D_STATE)
    elif col < 2 * di + 2 * M3_D_STATE + 2 * nh:
        value = da.unsafe_load(token * nh + col - 2 * di - 2 * M3_D_STATE - nh)
    elif col < 2 * di + 2 * M3_D_STATE + 3 * nh:
        value = dtrap.unsafe_load(token * nh + col - 2 * di - 2 * M3_D_STATE - 2 * nh)
    else:
        value = dangle.unsafe_load(
            token * M3_NUM_ROPE_ANGLES
            + col - 2 * di - 2 * M3_D_STATE - 3 * nh
        )
    packed.unsafe_store(cell, ftz(value))


def mamba3_backward_pack_in_proj_into(
    ctx: DeviceContext,
    mut packed: DeviceBuffer[DType.float32],
    mut dz: DeviceBuffer[DType.float32],
    mut dx: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut ddt: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut dtrap: DeviceBuffer[DType.float32],
    mut dangle: DeviceBuffer[DType.float32],
    m: Int,
    dims: Mamba3Dims,
) raises:
    var cells = m * dims.d_in_proj()
    ctx.enqueue_function[mamba3_pack_in_proj_backward_kernel](
        packed.unsafe_ptr(), dz.unsafe_ptr(), dx.unsafe_ptr(), db.unsafe_ptr(),
        dc.unsafe_ptr(), ddt.unsafe_ptr(), da.unsafe_ptr(), dtrap.unsafe_ptr(),
        dangle.unsafe_ptr(), Int32(cells), Int32(dims.d_inner),
        Int32(dims.nheads), Int32(dims.d_in_proj()),
        grid_dim=(_grid(cells), 1, 1), block_dim=(M3_BWD_TPB, 1, 1),
    )


def mamba3_block_norm_backward_kernel(
    dx: MutPointer[Float32, MutAnyOrigin],
    dw_rows: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    width_in: Int32,
):
    """Reverse the block RMSNorm; one owner serially folds each row."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var m = Int(m_in)
    var width = Int(width_in)
    if row >= m:
        return
    var rstd = ftz(identical_rsqrt(ftz(
        ftz(identical_div(sumsq.unsafe_load(row), Float32(width)))
        + M3_RMS_EPS
    )))
    var dot = Float32(0.0)
    for col in range(width):
        var cell = row * width + col
        var weighted = ftz(identical_mul(
            ftz(dy.unsafe_load(cell)), ftz(weight.unsafe_load(col))
        ))
        dot = ftz(identical_mul_add(
            weighted, ftz(x.unsafe_load(cell)), dot
        ))
    var correction = ftz(identical_mul(
        ftz(identical_mul(rstd, rstd)),
        ftz(identical_div(dot, Float32(width))),
    ))
    for col in range(width):
        var cell = row * width + col
        var xv = ftz(x.unsafe_load(cell))
        var dyv = ftz(dy.unsafe_load(cell))
        var weighted = ftz(identical_mul(dyv, ftz(weight.unsafe_load(col))))
        dx.unsafe_store(cell, ftz(identical_mul(
            rstd, ftz(weighted - ftz(identical_mul(xv, correction)))
        )))
        dw_rows.unsafe_store(cell, ftz(identical_mul(
            dyv, ftz(identical_mul(xv, rstd))
        )))


def mamba3_residual_join_kernel(
    dx: MutPointer[Float32, MutAnyOrigin],
    residual_cotangent: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < Int(cells_in):
        dx.unsafe_store(cell, ftz(
            ftz(dx.unsafe_load(cell))
            + ftz(residual_cotangent.unsafe_load(cell))
        ))


def mamba3_backward_block_norm_into(
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dw_rows: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut residual_cotangent: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut sumsq: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    m: Int,
    dims: Mamba3Dims,
) raises:
    ctx.enqueue_function[mamba3_block_norm_backward_kernel](
        dx.unsafe_ptr(), dw_rows.unsafe_ptr(), dy.unsafe_ptr(), x.unsafe_ptr(),
        sumsq.unsafe_ptr(), weight.unsafe_ptr(), Int32(m),
        Int32(dims.d_model), grid_dim=(_grid(m), 1, 1),
        block_dim=(M3_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[mamba3_residual_join_kernel](
        dx.unsafe_ptr(), residual_cotangent.unsafe_ptr(),
        Int32(m * dims.d_model),
        grid_dim=(_grid(m * dims.d_model), 1, 1),
        block_dim=(M3_BWD_TPB, 1, 1),
    )


def mamba3_s17_reverse_state_kernel(
    d_state_direct:MutPointer[Float32,MutAnyOrigin],d_state_total:MutPointer[Float32,MutAnyOrigin],d_initial:MutPointer[Float32,MutAnyOrigin],
    d_y:MutPointer[Float32,MutAnyOrigin],q:MutPointer[Float32,MutAnyOrigin],dacs:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    """Direct S17 readout adjoint plus descending chunk-state carry."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=b*nh*M3_HEADDIM*M3_D_STATE:return
    var n=cell%M3_D_STATE;var z=cell//M3_D_STATE;var p=z%M3_HEADDIM;z=z//M3_HEADDIM;var h=z%nh;var bb=z//nh
    var carry=Float32(0.0)
    for rev in range(nc):
        var c=nc-1-rev;var direct=Float32(0.0)
        for i in range(qs):
            var t=c*qs+i
            if t<l:
                var dy=ftz(d_y.unsafe_load(((bb*l+t)*nh+h)*M3_HEADDIM+p))
                var qv=ftz(q.unsafe_load(((bb*l+t)*nh+h)*M3_D_STATE+n))
                var ev=ftz(identical_exp(ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+i))))
                direct=ftz(identical_mul_add(dy,ftz(identical_mul(qv,ev)),direct))
        var idx=((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n)
        d_state_direct.unsafe_store(idx,direct)
        var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)))
        var total=ftz(direct+ftz(identical_mul(carry,ftz(identical_exp(last)))))
        d_state_total.unsafe_store(idx,total);carry=total
    d_initial.unsafe_store(((bb*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n,carry)


def mamba3_backward_s17_state_into(
    ctx:DeviceContext,mut d_direct:DeviceBuffer[DType.float32],mut d_total:DeviceBuffer[DType.float32],mut d_initial:DeviceBuffer[DType.float32],
    mut d_y:DeviceBuffer[DType.float32],mut q:DeviceBuffer[DType.float32],mut dacs:DeviceBuffer[DType.float32],b:Int,l:Int,dims:Mamba3Dims,qs:Int,
) raises:
    var cells=b*dims.nheads*M3_HEADDIM*M3_D_STATE
    ctx.enqueue_function[mamba3_s17_reverse_state_kernel](d_direct.unsafe_ptr(),d_total.unsafe_ptr(),d_initial.unsafe_ptr(),d_y.unsafe_ptr(),q.unsafe_ptr(),dacs.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_s17_operands_kernel(
    d_q_read:MutPointer[Float32,MutAnyOrigin],d_dacs_read:MutPointer[Float32,MutAnyOrigin],
    d_k_rec:MutPointer[Float32,MutAnyOrigin],d_v_rec:MutPointer[Float32,MutAnyOrigin],d_dacs_rec:MutPointer[Float32,MutAnyOrigin],
    d_y:MutPointer[Float32,MutAnyOrigin],q:MutPointer[Float32,MutAnyOrigin],k:MutPointer[Float32,MutAnyOrigin],v:MutPointer[Float32,MutAnyOrigin],
    dacs:MutPointer[Float32,MutAnyOrigin],states:MutPointer[Float32,MutAnyOrigin],d_states:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var th=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if th>=b*l*nh:return
    var h=th%nh;var token=(th//nh)%l;var bb=th//(nh*l);var c=token//qs;var inner=token%qs
    var dcidx=((bb*nh+h)*nc+c)*qs+inner;var ev=ftz(identical_exp(ftz(dacs.unsafe_load(dcidx))))
    var read_scalar=Float32(0.0)
    for n in range(M3_D_STATE):
        var dq=Float32(0.0)
        for p in range(M3_HEADDIM):
            var dy=ftz(d_y.unsafe_load(th*M3_HEADDIM+p));var hs=ftz(states.unsafe_load((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            dq=ftz(identical_mul_add(dy,hs,dq))
        d_q_read.unsafe_store(th*M3_D_STATE+n,ftz(identical_mul(dq,ev)))
        read_scalar=ftz(identical_mul_add(ftz(q.unsafe_load(th*M3_D_STATE+n)),dq,read_scalar))
    d_dacs_read.unsafe_store(th,ftz(identical_mul(read_scalar,ev)))
    var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)));var dec=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(dcidx)))))
    var rec_scalar=Float32(0.0)
    for n in range(M3_D_STATE):
        var dk=Float32(0.0)
        for p in range(M3_HEADDIM):
            var carry=Float32(0.0)
            if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            dk=ftz(identical_mul_add(carry,ftz(v.unsafe_load(th*M3_HEADDIM+p)),dk))
        d_k_rec.unsafe_store(th*M3_D_STATE+n,ftz(identical_mul(dk,dec)))
    for p in range(M3_HEADDIM):
        var dv=Float32(0.0)
        for n in range(M3_D_STATE):
            var carry=Float32(0.0)
            if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            dv=ftz(identical_mul_add(carry,ftz(k.unsafe_load(th*M3_D_STATE+n)),dv))
        var outv=ftz(identical_mul(dv,dec));d_v_rec.unsafe_store(th*M3_HEADDIM+p,outv)
        rec_scalar=ftz(identical_mul_add(outv,ftz(v.unsafe_load(th*M3_HEADDIM+p)),rec_scalar))
    var dr=ftz(-rec_scalar)
    if (inner==qs-1 or token==l-1) and not M3_S17_TAIL_SHARED:
        var add=Float32(0.0)
        for j in range(qs):
            var tj=c*qs+j
            if tj<l:
                var idx=((bb*nh+h)*nc+c)*qs+j;var de=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(idx)))))
                for p in range(M3_HEADDIM):
                    for n in range(M3_D_STATE):
                        var carry=Float32(0.0)
                        if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                        add=ftz(identical_mul_add(carry,ftz(identical_mul(ftz(identical_mul(ftz(v.unsafe_load(((bb*l+tj)*nh+h)*M3_HEADDIM+p)),ftz(k.unsafe_load(((bb*l+tj)*nh+h)*M3_D_STATE+n)))),de)),add))
        for p in range(M3_HEADDIM):
            for n in range(M3_D_STATE):
                var carry=Float32(0.0)
                if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                var hs=ftz(states.unsafe_load((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                add=ftz(identical_mul_add(carry,ftz(identical_mul(hs,ftz(identical_exp(last)))),add))
        dr=ftz(dr+add)
    d_dacs_rec.unsafe_store(th,dr)


#: lane/neural-net-experiment (2026-09-30): the S17 operands kernel with
#: its chains spread over a block. The naive kernel is ONE thread per
#: (batch, token, head) row -- 6,144 threads at the board's Samba shape,
#: a third of an L40S -- each walking 128 x 64 fmas three times over with
#: a device load per fma: 9.0 ms of a 37.6 ms Mamba-3 backward (24%).
#: Here a block of M3_D_STATE threads owns one row: thread n runs the
#: `d_q_read[n]` chain over p and the `d_k_rec[n]` chain over p, thread p
#: runs the `d_v_rec[p]` chain over n, and thread 0 folds the two row
#: scalars over n and over p in the naive order from the chains' results
#: in threadgroup memory. The row's dy, v, q, k and the next chunk's
#: carried state (its [p][n] tile, once) are staged; the chunk state is
#: read straight from device memory (coalesced across n). SAME BITS:
#: every chain is the naive kernel's chain with the same operands in the
#: same order; the row scalars are the same folds in the same order. The
#: chunk-end `add` chain stays in `mamba3_s17_tail_shared_kernel` (the
#: default); a build with MOJOLEARN_MAMBA3_S17_TAIL_NAIVE keeps the naive
#: operands kernel too. MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE=1 (a build
#: define) keeps the naive kernel for the A/B and the digest gate.
comptime M3_S17_CARRY_STRIDE = M3_D_STATE + 1
#: The operands kernel's threadgroup page: the carried [p][n] tile at its
#: padded stride, the row's dy, v and d_v over p, and q, k and d_q over n
#: (lane neural-pass24, 2026-10-01: 35,328 bytes, OVER Apple's 32 KB, and the
#: arm had no fits gate, so the 0.8.32 macOS wheel failed to create the
#: pipeline for every Mamba-3 backward; where the page does not fit the
#: naive operands kernel runs, the same chains in the same order).
comptime M3_S17_OPERANDS_BYTES = (
    M3_HEADDIM * M3_S17_CARRY_STRIDE + 3 * M3_HEADDIM + 3 * M3_D_STATE
) * 4
comptime M3_S17_OPERANDS_SHARED = (
    not is_defined["MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE"]()
    and M3_S17_TAIL_SHARED
    and lib_smem_page_fits_for[TARGET_COLUMN, M3_S17_OPERANDS_BYTES]()
)


def mamba3_s17_operands_shared_kernel(
    d_q_read:MutPointer[Float32,MutAnyOrigin],d_dacs_read:MutPointer[Float32,MutAnyOrigin],
    d_k_rec:MutPointer[Float32,MutAnyOrigin],d_v_rec:MutPointer[Float32,MutAnyOrigin],d_dacs_rec:MutPointer[Float32,MutAnyOrigin],
    d_y:MutPointer[Float32,MutAnyOrigin],q:MutPointer[Float32,MutAnyOrigin],k:MutPointer[Float32,MutAnyOrigin],v:MutPointer[Float32,MutAnyOrigin],
    dacs:MutPointer[Float32,MutAnyOrigin],states:MutPointer[Float32,MutAnyOrigin],d_states:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    """`mamba3_s17_operands_kernel`'s chains over a block per row (see the
    section comment); the chunk-end `add` is the tail kernel's."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var carry_s=stack_allocation[M3_HEADDIM*M3_S17_CARRY_STRIDE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var dys=stack_allocation[M3_HEADDIM,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var vs=stack_allocation[M3_HEADDIM,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var qs_=stack_allocation[M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var ks=stack_allocation[M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var dqs=stack_allocation[M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var dvs=stack_allocation[M3_HEADDIM,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var th=Int(block_idx.x)
    var n=Int(thread_idx.x)
    if th>=b*l*nh:return
    var h=th%nh;var token=(th//nh)%l;var bb=th//(nh*l);var c=token//qs;var inner=token%qs
    var dcidx=((bb*nh+h)*nc+c)*qs+inner;var ev=ftz(identical_exp(ftz(dacs.unsafe_load(dcidx))))
    var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)));var dec=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(dcidx)))))
    # staging: the row's four vectors and the carried state tile [p][n]
    qs_[n]=ftz(q.unsafe_load(th*M3_D_STATE+n))
    ks[n]=ftz(k.unsafe_load(th*M3_D_STATE+n))
    if n<M3_HEADDIM:
        dys[n]=ftz(d_y.unsafe_load(th*M3_HEADDIM+n))
        vs[n]=ftz(v.unsafe_load(th*M3_HEADDIM+n))
    var cbase=(((bb*nc+c+1)*nh+h)*M3_HEADDIM)*M3_D_STATE
    for p in range(M3_HEADDIM):
        var carry=Float32(0.0)
        if c+1<nc:carry=ftz(d_states.unsafe_load(cbase+p*M3_D_STATE+n))
        carry_s[p*M3_S17_CARRY_STRIDE+n]=carry
    barrier()
    # pass A: d_q_read[n] over p, then thread 0's read_scalar over n
    var sbase=(((bb*nc+c)*nh+h)*M3_HEADDIM)*M3_D_STATE
    var dq=Float32(0.0)
    for p in range(M3_HEADDIM):
        var hs=ftz(states.unsafe_load(sbase+p*M3_D_STATE+n))
        dq=ftz(identical_mul_add(dys[p],hs,dq))
    d_q_read.unsafe_store(th*M3_D_STATE+n,ftz(identical_mul(dq,ev)))
    dqs[n]=dq
    # pass B: d_k_rec[n] over p
    var dk=Float32(0.0)
    for p in range(M3_HEADDIM):
        dk=ftz(identical_mul_add(carry_s[p*M3_S17_CARRY_STRIDE+n],vs[p],dk))
    d_k_rec.unsafe_store(th*M3_D_STATE+n,ftz(identical_mul(dk,dec)))
    # pass C: d_v_rec[p] over n (threads p < HEADDIM)
    if n<M3_HEADDIM:
        var p=n
        var dv=Float32(0.0)
        for nn in range(M3_D_STATE):
            dv=ftz(identical_mul_add(carry_s[p*M3_S17_CARRY_STRIDE+nn],ks[nn],dv))
        var outv=ftz(identical_mul(dv,dec))
        d_v_rec.unsafe_store(th*M3_HEADDIM+p,outv)
        dvs[p]=outv
    barrier()
    if n==0:
        var read_scalar=Float32(0.0)
        for nn in range(M3_D_STATE):
            read_scalar=ftz(identical_mul_add(qs_[nn],dqs[nn],read_scalar))
        d_dacs_read.unsafe_store(th,ftz(identical_mul(read_scalar,ev)))
        var rec_scalar=Float32(0.0)
        for p in range(M3_HEADDIM):
            rec_scalar=ftz(identical_mul_add(dvs[p],vs[p],rec_scalar))
        d_dacs_rec.unsafe_store(th,ftz(-rec_scalar))


#: lane/neural-apple3 (2026-10-01): the shared operands kernel's page is
#: 35,328 bytes, over the Apple column's 32 KB, so Metal ran the naive
#: kernel (one thread a row, a device load an fma: 9.0 of a 37.6 ms Mamba-3
#: backward on the L40S against the staged kernel's 1.4). This kernel stages
#: the carried [p][n] tile in two halves of M3_S17_PH rows: pass B's chain
#: over p walks rows 0..31 off the first staging and 32..63 off the second
#: (p ascending, the one chain), and pass C's thread p walks its own row,
#: which sits whole in one half; pass A and the row scalars are the shared
#: kernel's. SAME BITS. Taken at comptime where the full page does not fit
#: and the half page (18,816 bytes) does; the same env define as the shared
#: kernel keeps the naive kernel for the A/B.
comptime M3_S17_PH = M3_HEADDIM // 2
comptime M3_S17_OPERANDS_HALF_BYTES = (
    M3_S17_PH * M3_S17_CARRY_STRIDE + 3 * M3_HEADDIM + 3 * M3_D_STATE
) * 4
comptime M3_S17_OPERANDS_HALF = (
    not is_defined["MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE"]()
    and M3_S17_TAIL_SHARED
    and not M3_S17_OPERANDS_SHARED
    and lib_smem_page_fits_for[TARGET_COLUMN, M3_S17_OPERANDS_HALF_BYTES]()
)


def mamba3_s17_operands_half_kernel(
    d_q_read:MutPointer[Float32,MutAnyOrigin],d_dacs_read:MutPointer[Float32,MutAnyOrigin],
    d_k_rec:MutPointer[Float32,MutAnyOrigin],d_v_rec:MutPointer[Float32,MutAnyOrigin],d_dacs_rec:MutPointer[Float32,MutAnyOrigin],
    d_y:MutPointer[Float32,MutAnyOrigin],q:MutPointer[Float32,MutAnyOrigin],k:MutPointer[Float32,MutAnyOrigin],v:MutPointer[Float32,MutAnyOrigin],
    dacs:MutPointer[Float32,MutAnyOrigin],states:MutPointer[Float32,MutAnyOrigin],d_states:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    """`mamba3_s17_operands_shared_kernel` with the carried tile staged in
    two halves of M3_S17_PH rows (see the section comment); the chunk-end
    `add` is the tail kernel's."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var carry_s=stack_allocation[M3_S17_PH*M3_S17_CARRY_STRIDE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var dys=stack_allocation[M3_HEADDIM,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var vs=stack_allocation[M3_HEADDIM,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var qs_=stack_allocation[M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var ks=stack_allocation[M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var dqs=stack_allocation[M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var dvs=stack_allocation[M3_HEADDIM,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var th=Int(block_idx.x)
    var n=Int(thread_idx.x)
    if th>=b*l*nh:return
    var h=th%nh;var token=(th//nh)%l;var bb=th//(nh*l);var c=token//qs;var inner=token%qs
    var dcidx=((bb*nh+h)*nc+c)*qs+inner;var ev=ftz(identical_exp(ftz(dacs.unsafe_load(dcidx))))
    var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)));var dec=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(dcidx)))))
    # staging: the row's four vectors; the carried tile comes in two halves
    qs_[n]=ftz(q.unsafe_load(th*M3_D_STATE+n))
    ks[n]=ftz(k.unsafe_load(th*M3_D_STATE+n))
    if n<M3_HEADDIM:
        dys[n]=ftz(d_y.unsafe_load(th*M3_HEADDIM+n))
        vs[n]=ftz(v.unsafe_load(th*M3_HEADDIM+n))
    var cbase=(((bb*nc+c+1)*nh+h)*M3_HEADDIM)*M3_D_STATE
    var sbase=(((bb*nc+c)*nh+h)*M3_HEADDIM)*M3_D_STATE
    var dk=Float32(0.0)
    for half in range(2):
        var p0=half*M3_S17_PH
        if half>0:barrier()  # every thread is off the first half's tile
        for pp in range(M3_S17_PH):
            var carry=Float32(0.0)
            if c+1<nc:carry=ftz(d_states.unsafe_load(cbase+(p0+pp)*M3_D_STATE+n))
            carry_s[pp*M3_S17_CARRY_STRIDE+n]=carry
        barrier()
        if half==0:
            # pass A: d_q_read[n] over p (the chunk state from device memory)
            var dq=Float32(0.0)
            for p in range(M3_HEADDIM):
                var hs=ftz(states.unsafe_load(sbase+p*M3_D_STATE+n))
                dq=ftz(identical_mul_add(dys[p],hs,dq))
            d_q_read.unsafe_store(th*M3_D_STATE+n,ftz(identical_mul(dq,ev)))
            dqs[n]=dq
        # pass B: d_k_rec[n] over this half's p rows, the one chain continued
        for pp in range(M3_S17_PH):
            dk=ftz(identical_mul_add(carry_s[pp*M3_S17_CARRY_STRIDE+n],vs[p0+pp],dk))
        # pass C: d_v_rec[p] over n for the p rows this half holds
        if n>=p0 and n<p0+M3_S17_PH:
            var p=n
            var dv=Float32(0.0)
            for nn in range(M3_D_STATE):
                dv=ftz(identical_mul_add(carry_s[(p-p0)*M3_S17_CARRY_STRIDE+nn],ks[nn],dv))
            var outv=ftz(identical_mul(dv,dec))
            d_v_rec.unsafe_store(th*M3_HEADDIM+p,outv)
            dvs[p]=outv
    d_k_rec.unsafe_store(th*M3_D_STATE+n,ftz(identical_mul(dk,dec)))
    barrier()
    if n==0:
        var read_scalar=Float32(0.0)
        for nn in range(M3_D_STATE):
            read_scalar=ftz(identical_mul_add(qs_[nn],dqs[nn],read_scalar))
        d_dacs_read.unsafe_store(th,ftz(identical_mul(read_scalar,ev)))
        var rec_scalar=Float32(0.0)
        for p in range(M3_HEADDIM):
            rec_scalar=ftz(identical_mul_add(dvs[p],vs[p],rec_scalar))
        d_dacs_rec.unsafe_store(th,ftz(-rec_scalar))


#: lane/neural-net-experiment (2026-09-30, the S16 pass): the tail's tile
#: was 8 value columns (8 KB of threadgroup memory, 512 barriers a chunk at
#: L = 512); 32 columns where the column's page allows it (32 KB, 128
#: barriers), 16 (16 KB, 256 barriers) on a 32 KB column such as Apple
#: (lane/neural-apple3). The chain thread 0 walks is the same chain in the
#: same order; only the staging cadence moves.
comptime M3_S17_TP = 32 if lib_smem_page_fits_for[TARGET_COLUMN, 49152]() else (16 if lib_smem_page_fits_for[TARGET_COLUMN, 24576]() else 8)


def mamba3_s17_tail_shared_kernel(
    d_dacs_rec:MutPointer[Float32,MutAnyOrigin],k:MutPointer[Float32,MutAnyOrigin],v:MutPointer[Float32,MutAnyOrigin],
    dacs:MutPointer[Float32,MutAnyOrigin],states:MutPointer[Float32,MutAnyOrigin],d_states:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    """`mamba3_s17_operands_kernel`'s chunk-end `add` chain (M3_S17_TAIL_SHARED),
    one block of M3_D_STATE threads per (batch, chunk, head): per tile of
    M3_S17_TP value columns thread n stages each step's two operands, exactly
    the naive kernel's expressions, into shared memory, and thread 0 runs the
    chain over them in the naive order (j, then p, then n; then p, n for the
    state term). Then `d_dacs_rec[th] = ftz(d_dacs_rec[th] + add)`, the naive
    `dr = ftz(dr + add)` over the dr the operands kernel stored."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var cs=stack_allocation[M3_S17_TP*M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var ts=stack_allocation[M3_S17_TP*M3_D_STATE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var blk=Int(block_idx.x);var n=Int(thread_idx.x)
    if blk>=b*nc*nh:return
    var h=blk%nh;var c=(blk//nh)%nc;var bb=blk//(nh*nc)
    var token=c*qs+qs-1
    if token>l-1:token=l-1
    var th=(bb*l+token)*nh+h
    var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)))
    var add=Float32(0.0)
    for j in range(qs):
        var tj=c*qs+j
        if tj<l:
            var idx=((bb*nh+h)*nc+c)*qs+j;var de=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(idx)))))
            var p0=0
            while p0<M3_HEADDIM:
                for pp in range(M3_S17_TP):
                    var p=p0+pp
                    var carry=Float32(0.0)
                    if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                    cs[pp*M3_D_STATE+n]=carry
                    ts[pp*M3_D_STATE+n]=ftz(identical_mul(ftz(identical_mul(ftz(v.unsafe_load(((bb*l+tj)*nh+h)*M3_HEADDIM+p)),ftz(k.unsafe_load(((bb*l+tj)*nh+h)*M3_D_STATE+n)))),de))
                barrier()
                if n==0:
                    for e in range(M3_S17_TP*M3_D_STATE):
                        add=ftz(identical_mul_add(cs[e],ts[e],add))
                barrier()
                p0+=M3_S17_TP
    var el=ftz(identical_exp(last))
    var p1=0
    while p1<M3_HEADDIM:
        for pp in range(M3_S17_TP):
            var p=p1+pp
            var carry=Float32(0.0)
            if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            var hs=ftz(states.unsafe_load((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            cs[pp*M3_D_STATE+n]=carry
            ts[pp*M3_D_STATE+n]=ftz(identical_mul(hs,el))
        barrier()
        if n==0:
            for e in range(M3_S17_TP*M3_D_STATE):
                add=ftz(identical_mul_add(cs[e],ts[e],add))
        barrier()
        p1+=M3_S17_TP
    if n==0:
        d_dacs_rec.unsafe_store(th,ftz(d_dacs_rec.unsafe_load(th)+add))


#: Default-OFF tail candidates, ported by hand from lane/neural-pass5
#: (0dcfceabd, fefa1db0d) onto the current tile (M3_S17_TP per column).
#: Both walk the shared kernel's chain over the same pairs in the same
#: order (j, then p, then n; then p, n for the state term): same bits.
#:   -D MOJOLEARN_IDN_M3_S17_TAIL_DBUF: two half tiles resident, every
#:      thread stages tile t while thread 0 folds tile t - 1, one barrier a
#:      tile.
#:   -D MOJOLEARN_IDN_M3_S17_TAIL_PIPE: each step's operands stored as an
#:      adjacent pair and prefetched a group of M3_S17_PIPE_GROUP ahead into
#:      thread 0's registers. MEASURED SLOWER on the L4 (8.46 against 7.27
#:      ms at the board shape, the same bits).
#: Both defined: DBUF runs (PIPE is the measured loser). Neither runs under
#: MOJOLEARN_IDN_ALL_OFF, under MOJOLEARN_MAMBA3_S17_TAIL_NAIVE (the shared
#: tail must be the active arm), or where the page does not fit the column;
#: then the shared kernel runs. The page is the shared kernel's (two
#: M3_S17_TP x M3_D_STATE float tiles) on both arms.
comptime M3_S17_TAIL_PAGE_BYTES = 2 * M3_S17_TP * M3_D_STATE * 4
comptime M3_S17_TAIL_ALT_FITS = lib_smem_page_fits_for[TARGET_COLUMN, M3_S17_TAIL_PAGE_BYTES]()
comptime M3_S17_TAIL_DBUF = (
    M3_S17_TAIL_SHARED
    and M3_S17_TAIL_ALT_FITS
    and is_defined["MOJOLEARN_IDN_M3_S17_TAIL_DBUF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime M3_S17_TAIL_PIPE = (
    M3_S17_TAIL_SHARED
    and M3_S17_TAIL_ALT_FITS
    and not M3_S17_TAIL_DBUF
    and is_defined["MOJOLEARN_IDN_M3_S17_TAIL_PIPE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime M3_S17_PIPE_GROUP = 16
#: dbuf: two tiles of M3_S17_DB_TP value columns are the shared kernel's
#: one tile of M3_S17_TP (the same page).
comptime M3_S17_DB_TP = M3_S17_TP // 2


def mamba3_s17_tail_dbuf_kernel(
    d_dacs_rec:MutPointer[Float32,MutAnyOrigin],k:MutPointer[Float32,MutAnyOrigin],v:MutPointer[Float32,MutAnyOrigin],
    dacs:MutPointer[Float32,MutAnyOrigin],states:MutPointer[Float32,MutAnyOrigin],d_states:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    """`mamba3_s17_tail_shared_kernel` double buffered (M3_S17_TAIL_DBUF).
    Tiles t = 0 .. ntiles - 1: the chain tiles first, `TPJ` a token in token
    order (tile t is token j = t // TPJ, columns (t % TPJ) * TPD ..), then the
    `TPJ` state tiles. At step t every thread stages tile t into page t % 2
    and thread 0 folds tile t - 1 from page (t - 1) % 2; one barrier a step.
    Thread 0 walks the same pairs in the same order as the shared kernel."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    comptime TPD=M3_S17_DB_TP
    comptime TILE=TPD*M3_D_STATE
    comptime TPJ=M3_HEADDIM//TPD
    var cs=stack_allocation[2*TILE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var ts=stack_allocation[2*TILE,Scalar[DType.float32],address_space=AddressSpace.SHARED]()
    var blk=Int(block_idx.x);var n=Int(thread_idx.x)
    if blk>=b*nc*nh:return
    var h=blk%nh;var c=(blk//nh)%nc;var bb=blk//(nh*nc)
    var token=c*qs+qs-1
    if token>l-1:token=l-1
    var th=(bb*l+token)*nh+h
    var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)))
    var el=ftz(identical_exp(last))
    var jn=qs
    if c*qs+qs>l:jn=l-c*qs
    var ntiles=jn*TPJ+TPJ
    var add=Float32(0.0)
    for t in range(ntiles+1):
        if t<ntiles:
            var page=(t%2)*TILE
            if t<jn*TPJ:
                var j=t//TPJ;var pp0=(t%TPJ)*TPD;var tj=c*qs+j
                var idx=((bb*nh+h)*nc+c)*qs+j;var de=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(idx)))))
                for pp in range(TPD):
                    var p=pp0+pp
                    var carry=Float32(0.0)
                    if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                    cs[page+pp*M3_D_STATE+n]=carry
                    ts[page+pp*M3_D_STATE+n]=ftz(identical_mul(ftz(identical_mul(ftz(v.unsafe_load(((bb*l+tj)*nh+h)*M3_HEADDIM+p)),ftz(k.unsafe_load(((bb*l+tj)*nh+h)*M3_D_STATE+n)))),de))
            else:
                var pp0=(t-jn*TPJ)*TPD
                for pp in range(TPD):
                    var p=pp0+pp
                    var carry=Float32(0.0)
                    if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                    var hs=ftz(states.unsafe_load((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                    cs[page+pp*M3_D_STATE+n]=carry
                    ts[page+pp*M3_D_STATE+n]=ftz(identical_mul(hs,el))
        if n==0 and t>=1:
            var page=((t-1)%2)*TILE
            for e in range(TILE):
                add=ftz(identical_mul_add(cs[page+e],ts[page+e],add))
        barrier()
    if n==0:
        d_dacs_rec.unsafe_store(th,ftz(d_dacs_rec.unsafe_load(th)+add))


def mamba3_s17_tail_pipe_kernel(
    d_dacs_rec:MutPointer[Float32,MutAnyOrigin],k:MutPointer[Float32,MutAnyOrigin],v:MutPointer[Float32,MutAnyOrigin],
    dacs:MutPointer[Float32,MutAnyOrigin],states:MutPointer[Float32,MutAnyOrigin],d_states:MutPointer[Float32,MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32,
):
    """`mamba3_s17_tail_shared_kernel` with the chain's operand pairs adjacent
    in threadgroup memory and prefetched a group ahead (M3_S17_TAIL_PIPE).
    Thread n stages the shared kernel's expressions into `cts[2 e]`,
    `cts[2 e + 1]` for its step index `e = pp * M3_D_STATE + n`; thread 0
    folds the tile in `e` order."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    comptime STEPS = M3_S17_TP * M3_D_STATE
    comptime G = M3_S17_PIPE_GROUP
    comptime GROUPS = STEPS // G
    var cts=stack_allocation[2*STEPS,Scalar[DType.float32],alignment=16,address_space=AddressSpace.SHARED]()
    var blk=Int(block_idx.x);var n=Int(thread_idx.x)
    if blk>=b*nc*nh:return
    var h=blk%nh;var c=(blk//nh)%nc;var bb=blk//(nh*nc)
    var token=c*qs+qs-1
    if token>l-1:token=l-1
    var th=(bb*l+token)*nh+h
    var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)))
    var add=Float32(0.0)
    for j in range(qs):
        var tj=c*qs+j
        if tj<l:
            var idx=((bb*nh+h)*nc+c)*qs+j;var de=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(idx)))))
            var p0=0
            while p0<M3_HEADDIM:
                for pp in range(M3_S17_TP):
                    var p=p0+pp
                    var carry=Float32(0.0)
                    if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                    var e=pp*M3_D_STATE+n
                    cts[2*e]=carry
                    cts[2*e+1]=ftz(identical_mul(ftz(identical_mul(ftz(v.unsafe_load(((bb*l+tj)*nh+h)*M3_HEADDIM+p)),ftz(k.unsafe_load(((bb*l+tj)*nh+h)*M3_D_STATE+n)))),de))
                barrier()
                if n==0:
                    var cur=InlineArray[Float32,2*G](fill=Float32(0))
                    comptime for i in range(G):
                        var pr=cts.unsafe_load[width=2,alignment=8](2*i)
                        cur[2*i]=pr[0];cur[2*i+1]=pr[1]
                    for g in range(GROUPS):
                        var nxt=InlineArray[Float32,2*G](fill=Float32(0))
                        if g+1<GROUPS:
                            comptime for i in range(G):
                                var pr=cts.unsafe_load[width=2,alignment=8](2*((g+1)*G+i))
                                nxt[2*i]=pr[0];nxt[2*i+1]=pr[1]
                        comptime for i in range(G):
                            add=ftz(identical_mul_add(cur[2*i],cur[2*i+1],add))
                        cur=nxt^
                barrier()
                p0+=M3_S17_TP
    var el=ftz(identical_exp(last))
    var p1=0
    while p1<M3_HEADDIM:
        for pp in range(M3_S17_TP):
            var p=p1+pp
            var carry=Float32(0.0)
            if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            var hs=ftz(states.unsafe_load((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            var e=pp*M3_D_STATE+n
            cts[2*e]=carry
            cts[2*e+1]=ftz(identical_mul(hs,el))
        barrier()
        if n==0:
            var cur=InlineArray[Float32,2*G](fill=Float32(0))
            comptime for i in range(G):
                var pr=cts.unsafe_load[width=2,alignment=8](2*i)
                cur[2*i]=pr[0];cur[2*i+1]=pr[1]
            for g in range(GROUPS):
                var nxt=InlineArray[Float32,2*G](fill=Float32(0))
                if g+1<GROUPS:
                    comptime for i in range(G):
                        var pr=cts.unsafe_load[width=2,alignment=8](2*((g+1)*G+i))
                        nxt[2*i]=pr[0];nxt[2*i+1]=pr[1]
                comptime for i in range(G):
                    add=ftz(identical_mul_add(cur[2*i],cur[2*i+1],add))
                cur=nxt^
        barrier()
        p1+=M3_S17_TP
    if n==0:
        d_dacs_rec.unsafe_store(th,ftz(d_dacs_rec.unsafe_load(th)+add))


def mamba3_backward_s17_operands_into(ctx:DeviceContext,mut dq:DeviceBuffer[DType.float32],mut ddr:DeviceBuffer[DType.float32],mut dk:DeviceBuffer[DType.float32],mut dv:DeviceBuffer[DType.float32],mut ddc:DeviceBuffer[DType.float32],mut dy:DeviceBuffer[DType.float32],mut q:DeviceBuffer[DType.float32],mut k:DeviceBuffer[DType.float32],mut v:DeviceBuffer[DType.float32],mut dacs:DeviceBuffer[DType.float32],mut states:DeviceBuffer[DType.float32],mut dstates:DeviceBuffer[DType.float32],b:Int,l:Int,dims:Mamba3Dims,qs:Int) raises:
    var cells=b*l*dims.nheads
    comptime if M3_S17_OPERANDS_SHARED:
        ctx.enqueue_function[mamba3_s17_operands_shared_kernel](dq.unsafe_ptr(),ddr.unsafe_ptr(),dk.unsafe_ptr(),dv.unsafe_ptr(),ddc.unsafe_ptr(),dy.unsafe_ptr(),q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),dacs.unsafe_ptr(),states.unsafe_ptr(),dstates.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(cells,1,1),block_dim=(M3_D_STATE,1,1))
    else:
        comptime if M3_S17_OPERANDS_HALF:
            ctx.enqueue_function[mamba3_s17_operands_half_kernel](dq.unsafe_ptr(),ddr.unsafe_ptr(),dk.unsafe_ptr(),dv.unsafe_ptr(),ddc.unsafe_ptr(),dy.unsafe_ptr(),q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),dacs.unsafe_ptr(),states.unsafe_ptr(),dstates.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(cells,1,1),block_dim=(M3_D_STATE,1,1))
        else:
            ctx.enqueue_function[mamba3_s17_operands_kernel](dq.unsafe_ptr(),ddr.unsafe_ptr(),dk.unsafe_ptr(),dv.unsafe_ptr(),ddc.unsafe_ptr(),dy.unsafe_ptr(),q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),dacs.unsafe_ptr(),states.unsafe_ptr(),dstates.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))
    comptime if M3_S17_TAIL_SHARED:
        var nc=(l+qs-1)//qs
        comptime if M3_S17_TAIL_DBUF:
            ctx.enqueue_function[mamba3_s17_tail_dbuf_kernel](ddc.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),dacs.unsafe_ptr(),states.unsafe_ptr(),dstates.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(b*nc*dims.nheads,1,1),block_dim=(M3_D_STATE,1,1))
        elif M3_S17_TAIL_PIPE:
            ctx.enqueue_function[mamba3_s17_tail_pipe_kernel](ddc.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),dacs.unsafe_ptr(),states.unsafe_ptr(),dstates.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(b*nc*dims.nheads,1,1),block_dim=(M3_D_STATE,1,1))
        else:
            ctx.enqueue_function[mamba3_s17_tail_shared_kernel](ddc.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),dacs.unsafe_ptr(),states.unsafe_ptr(),dstates.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(b*nc*dims.nheads,1,1),block_dim=(M3_D_STATE,1,1))


def mamba3_join_two_kernel(dst:MutPointer[Float32,MutAnyOrigin],a:MutPointer[Float32,MutAnyOrigin],b:MutPointer[Float32,MutAnyOrigin],n_in:Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n_in):dst.unsafe_store(i,ftz(ftz(a.unsafe_load(i))+ftz(b.unsafe_load(i))))


def mamba3_dacs_reverse_kernel(d_adt:MutPointer[Float32,MutAnyOrigin],d_dacs:MutPointer[Float32,MutAnyOrigin],b_in:Int32,l_in:Int32,nh_in:Int32,qs_in:Int32):
    """Transpose of the chunk-local inclusive cumsum that forms dACS."""
    var b=Int(b_in);var l=Int(l_in);var nh=Int(nh_in);var qs=Int(qs_in);var nc=(l+qs-1)//qs
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=b*nc*nh:return
    var h=cell%nh;var c=(cell//nh)%nc;var bb=cell//(nh*nc);var carry=Float32(0.0)
    for rev in range(qs):
        var token=c*qs+qs-1-rev
        if token<l:
            var idx=(bb*l+token)*nh+h
            carry=ftz(carry+ftz(d_dacs.unsafe_load(idx)));d_adt.unsafe_store(idx,carry)


def mamba3_backward_dacs_to_adt_into(ctx:DeviceContext,mut d_adt:DeviceBuffer[DType.float32],mut d_dacs:DeviceBuffer[DType.float32],b:Int,l:Int,nh:Int,qs:Int) raises:
    var cells=b*((l+qs-1)//qs)*nh
    ctx.enqueue_function[mamba3_dacs_reverse_kernel](d_adt.unsafe_ptr(),d_dacs.unsafe_ptr(),Int32(b),Int32(l),Int32(nh),Int32(qs),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_backward_join_two_into(ctx:DeviceContext,mut dst:DeviceBuffer[DType.float32],mut a:DeviceBuffer[DType.float32],mut b:DeviceBuffer[DType.float32],cells:Int) raises:
    ctx.enqueue_function[mamba3_join_two_kernel](dst.unsafe_ptr(),a.unsafe_ptr(),b.unsafe_ptr(),Int32(cells),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_backward_join_s16_s17_into(ctx:DeviceContext,mut qout:DeviceBuffer[DType.float32],mut kout:DeviceBuffer[DType.float32],mut vout:DeviceBuffer[DType.float32],mut dout:DeviceBuffer[DType.float32],mut q16:DeviceBuffer[DType.float32],mut q17:DeviceBuffer[DType.float32],mut k16:DeviceBuffer[DType.float32],mut k17:DeviceBuffer[DType.float32],mut vold:DeviceBuffer[DType.float32],mut v17:DeviceBuffer[DType.float32],mut dr:DeviceBuffer[DType.float32],mut dc:DeviceBuffer[DType.float32],state_cells:Int,value_cells:Int,head_cells:Int) raises:
    ctx.enqueue_function[mamba3_join_two_kernel](qout.unsafe_ptr(),q16.unsafe_ptr(),q17.unsafe_ptr(),Int32(state_cells),grid_dim=(_grid(state_cells),1,1),block_dim=(M3_BWD_TPB,1,1))
    ctx.enqueue_function[mamba3_join_two_kernel](kout.unsafe_ptr(),k16.unsafe_ptr(),k17.unsafe_ptr(),Int32(state_cells),grid_dim=(_grid(state_cells),1,1),block_dim=(M3_BWD_TPB,1,1))
    ctx.enqueue_function[mamba3_join_two_kernel](vout.unsafe_ptr(),vold.unsafe_ptr(),v17.unsafe_ptr(),Int32(value_cells),grid_dim=(_grid(value_cells),1,1),block_dim=(M3_BWD_TPB,1,1))
    ctx.enqueue_function[mamba3_join_two_kernel](dout.unsafe_ptr(),dr.unsafe_ptr(),dc.unsafe_ptr(),Int32(head_cells),grid_dim=(_grid(head_cells),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_backward_s15_only_into(ctx:DeviceContext,mut d_krot:DeviceBuffer[DType.float32],mut d_scale:DeviceBuffer[DType.float32],mut d_kscaled:DeviceBuffer[DType.float32],mut krot:DeviceBuffer[DType.float32],mut scale:DeviceBuffer[DType.float32],rows:Int) raises:
    ctx.enqueue_function[mamba3_s15_backward_kernel](d_krot.unsafe_ptr(),d_scale.unsafe_ptr(),d_kscaled.unsafe_ptr(),krot.unsafe_ptr(),scale.unsafe_ptr(),Int32(rows),grid_dim=(_grid(rows),1,1),block_dim=(M3_BWD_TPB,1,1))


def mamba3_backward_rotary_only_into(ctx:DeviceContext,mut dqraw:DeviceBuffer[DType.float32],mut dkraw:DeviceBuffer[DType.float32],mut dtheta:DeviceBuffer[DType.float32],mut dqrot:DeviceBuffer[DType.float32],mut dkrot:DeviceBuffer[DType.float32],mut bcb:DeviceBuffer[DType.float32],mut bcc:DeviceBuffer[DType.float32],mut bbias:DeviceBuffer[DType.float32],mut cbias:DeviceBuffer[DType.float32],mut theta:DeviceBuffer[DType.float32],pairs:Int,nh:Int) raises:
    ctx.enqueue_function[mamba3_rotary_backward_kernel](dqraw.unsafe_ptr(),dkraw.unsafe_ptr(),dtheta.unsafe_ptr(),dqrot.unsafe_ptr(),dkrot.unsafe_ptr(),bcb.unsafe_ptr(),bcc.unsafe_ptr(),bbias.unsafe_ptr(),cbias.unsafe_ptr(),theta.unsafe_ptr(),Int32(pairs),Int32(nh),grid_dim=(_grid(pairs),1,1),block_dim=(M3_BWD_TPB,1,1))
