# SPDX-License-Identifier: Apache-2.0
"""Implemented output-gate and D-skip tail of Mamba-3 backward."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_div, identical_exp, identical_mul_add, identical_rsqrt, identical_sigmoid, identical_silu, identical_tanh, portable_cosf, portable_sinf
from mamba.checks.mamba3_fixture import M3_A_FLOOR, M3_D_STATE, M3_HEADDIM, M3_NUM_ROPE_ANGLES, M3_PI, M3_RMS_EPS, Mamba3Dims
from mamba.impl.transformers.models.mamba.modeling_mamba import pinned_mul

comptime M3_BWD_TPB = 128


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
        var ds = ftz(pinned_mul(dg, ftz(identical_silu(z))))
        d_skip.unsafe_store(cell, ds)
        var sig = ftz(identical_sigmoid(z))
        var middle = ftz(identical_mul_add(z, ftz(Float32(1.0) - sig), Float32(1.0)))
        var prime = ftz(pinned_mul(sig, middle))
        d_z.unsafe_store(cell, ftz(pinned_mul(ftz(pinned_mul(dg, sk)), prime)))
        d_v.unsafe_store(cell, ftz(pinned_mul(ds, tq)))
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
        var common = ftz(pinned_mul(incoming, gam))
        var db = ftz(pinned_mul(common, q))
        var dc = ftz(pinned_mul(common, k))
        var cell = (token * nh + head) * M3_D_STATE + n
        d_b.unsafe_store(cell, db)
        d_c.unsafe_store(cell, dc)
        d_b_bias.unsafe_store(cell, db)
        d_c_bias.unsafe_store(cell, dc)
    var dgam = ftz(pinned_mul(incoming, dot))
    d_gamma.unsafe_store(th, dgam)
    var sig = ftz(sigma.unsafe_load(th))
    d_dt.unsafe_store(th, ftz(pinned_mul(dgam, sig)))
    var dsig = ftz(pinned_mul(dgam, ftz(dt.unsafe_load(th))))
    d_trap.unsafe_store(
        th,
        ftz(pinned_mul(ftz(pinned_mul(dsig, sig)), ftz(Float32(1.0) - sig))),
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
                    dq = ftz(identical_mul_add(dy_i, ftz(pinned_mul(ftz(pinned_mul(kv, lv)), vv)), dq))
            for i in range(inner + 1, qs):
                var ti = chunk * qs + i
                if ti < l:
                    var dy = ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p))
                    var lv2 = ftz(seg_l.unsafe_load((((bb * ((l + qs - 1) // qs) + chunk) * nh + h) * qs + i) * qs + inner))
                    var qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
                    var vv2 = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
                    dk = ftz(identical_mul_add(dy, ftz(pinned_mul(ftz(pinned_mul(qv, lv2)), vv2)), dk))
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
                dv = ftz(identical_mul_add(ftz(d_y.unsafe_load(((bb*l+ti)*nh+h)*M3_HEADDIM+p)), ftz(pinned_mul(dot, lv)), dv))
        d_v.unsafe_store(cell, dv)


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
        d_krot.unsafe_store(cell, ftz(pinned_mul(dk, sc)))
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
    d_qraw.unsafe_store(base+e0,ftz(ftz(pinned_mul(dq0,c))+ftz(pinned_mul(dq1,s))))
    d_qraw.unsafe_store(base+e1,ftz(ftz(pinned_mul(dq1,c))-ftz(pinned_mul(dq0,s))))
    d_kraw.unsafe_store(base+e0,ftz(ftz(pinned_mul(dk0,c))+ftz(pinned_mul(dk1,s))))
    d_kraw.unsafe_store(base+e1,ftz(ftz(pinned_mul(dk1,c))-ftz(pinned_mul(dk0,s))))
    var q0 = ftz(ftz(bcc.unsafe_load(token*M3_D_STATE+e0))+ftz(c_bias.unsafe_load(h*M3_D_STATE+e0)))
    var q1 = ftz(ftz(bcc.unsafe_load(token*M3_D_STATE+e1))+ftz(c_bias.unsafe_load(h*M3_D_STATE+e1)))
    var k0 = ftz(ftz(bcb.unsafe_load(token*M3_D_STATE+e0))+ftz(b_bias.unsafe_load(h*M3_D_STATE+e0)))
    var k1 = ftz(ftz(bcb.unsafe_load(token*M3_D_STATE+e1))+ftz(b_bias.unsafe_load(h*M3_D_STATE+e1)))
    var dt = Float32(0.0)
    dt = ftz(identical_mul_add(dq0,ftz(-ftz(pinned_mul(q0,s))-ftz(pinned_mul(q1,c))),dt))
    dt = ftz(identical_mul_add(dq1,ftz(ftz(pinned_mul(q0,c))-ftz(pinned_mul(q1,s))),dt))
    dt = ftz(identical_mul_add(dk0,ftz(-ftz(pinned_mul(k0,s))-ftz(pinned_mul(k1,c))),dt))
    dt = ftz(identical_mul_add(dk1,ftz(ftz(pinned_mul(k0,c))-ftz(pinned_mul(k1,s))),dt))
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
    var ddt=ftz(qk_dt.unsafe_load(i));var dtr=ftz(qk_trap.unsafe_load(i))
    if li>0:
        var db=ftz(d_beta.unsafe_load(i-nh));var sig=ftz(sigma.unsafe_load(i));var dtv=ftz(dt.unsafe_load(i))
        ddt=ftz(ddt+ftz(pinned_mul(db,ftz(Float32(1.0)-sig))))
        var ds=ftz(-ftz(pinned_mul(db,dtv)))
        dtr=ftz(dtr+ftz(pinned_mul(ftz(pinned_mul(ds,sig)),ftz(Float32(1.0)-sig))))
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
        d_rate.unsafe_store(rowh*M3_NUM_ROPE_ANGLES+r,ftz(pinned_mul(carry,ftz(dt.unsafe_load(rowh)))))


def mamba3_angle_reduce_kernel(
    d_angle: MutPointer[Float32, MutAnyOrigin], d_dt: MutPointer[Float32, MutAnyOrigin],
    d_rate: MutPointer[Float32, MutAnyOrigin], d_theta: MutPointer[Float32, MutAnyOrigin],
    angle_raw: MutPointer[Float32, MutAnyOrigin], dt: MutPointer[Float32, MutAnyOrigin],
    b_in:Int32,l_in:Int32,nh_in:Int32,dip_in:Int32,col_angle_in:Int32,
):
    var b=Int(b_in);var l=Int(l_in);var m=b*l;var nh=Int(nh_in);var dip=Int(dip_in);var ca=Int(col_angle_in)
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell<m*M3_NUM_ROPE_ANGLES:
        var r=cell%M3_NUM_ROPE_ANGLES;var token=cell//M3_NUM_ROPE_ANGLES;var acc=Float32(0.0)
        for h in range(nh):
            var dr=ftz(d_rate.unsafe_load((token*nh+h)*M3_NUM_ROPE_ANGLES+r))
            acc=ftz(acc+dr)
        var raw=ftz(angle_raw.unsafe_load(token*dip+ca+r));var tv=ftz(identical_tanh(raw))
        var prime=ftz(pinned_mul(M3_PI,ftz(Float32(1.0)-ftz(pinned_mul(tv,tv)))))
        d_angle.unsafe_store(cell,ftz(pinned_mul(acc,prime)))
    if cell<m*nh:
        var token=cell//nh;var h=cell%nh;var bb=token//l;var li=token%l;var accdt=Float32(0.0)
        for r in range(M3_NUM_ROPE_ANGLES):
            var raw=ftz(angle_raw.unsafe_load(token*dip+ca+r));var rate=ftz(pinned_mul(ftz(identical_tanh(raw)),M3_PI))
            var carry=Float32(0.0)
            for u in range(li,l):
                carry=ftz(carry+ftz(d_theta.unsafe_load(((bb*l+u)*nh+h)*M3_NUM_ROPE_ANGLES+r)))
            accdt=ftz(identical_mul_add(carry,rate,accdt))
        d_dt.unsafe_store(cell,accdt)


def mamba3_backward_angle_into(
    ctx:DeviceContext,mut d_rate:DeviceBuffer[DType.float32],mut d_angle:DeviceBuffer[DType.float32],mut d_dt:DeviceBuffer[DType.float32],
    mut d_theta:DeviceBuffer[DType.float32],mut dt:DeviceBuffer[DType.float32],mut in_proj:DeviceBuffer[DType.float32],b:Int,l:Int,dims:Mamba3Dims,
) raises:
    var chains=b*dims.nheads*M3_NUM_ROPE_ANGLES
    ctx.enqueue_function[mamba3_theta_reverse_kernel](d_rate.unsafe_ptr(),d_theta.unsafe_ptr(),dt.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),grid_dim=(_grid(chains),1,1),block_dim=(M3_BWD_TPB,1,1))
    var m=b*l;var cells=m*M3_NUM_ROPE_ANGLES if m*M3_NUM_ROPE_ANGLES>m*dims.nheads else m*dims.nheads
    ctx.enqueue_function[mamba3_angle_reduce_kernel](d_angle.unsafe_ptr(),d_dt.unsafe_ptr(),d_rate.unsafe_ptr(),d_theta.unsafe_ptr(),in_proj.unsafe_ptr(),dt.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(dims.d_in_proj()),Int32(dims.col_angle()),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


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
    var draw=ftz(pinned_mul(total,prime))
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
        acc=ftz(identical_mul_add(dy,ftz(pinned_mul(dot,vv)),acc))
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
    var da=ftz(pinned_mul(da_dt,dtv));var ddt=ftz(pinned_mul(da_dt,av))
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
        var den=ftz(Float32(1.0)-raw);ht=ftz(identical_div(Float32(1.0),den));prime=ftz(pinned_mul(ht,ht))
    if -ht> -M3_A_FLOOR:
        d_raw.unsafe_store(cell,Float32(0.0))
    else:
        d_raw.unsafe_store(cell,ftz(-ftz(pinned_mul(ftz(d_a.unsafe_load(cell)),prime))))


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
        var x=ftz(raw.unsafe_load(t*Int(dip_in)+Int(col_in)+n));var gw=ftz(pinned_mul(g,ftz(weight.unsafe_load(n))))
        dwrow.unsafe_store(t*M3_D_STATE+n,ftz(pinned_mul(g,ftz(pinned_mul(x,r)))));dot=ftz(identical_mul_add(gw,x,dot))
    var corr=ftz(pinned_mul(ftz(pinned_mul(r,r)),ftz(identical_div(dot,Float32(M3_D_STATE)))))
    for n in range(M3_D_STATE):
        var g=Float32(0.0)
        for h in range(Int(nh_in)):g=ftz(g+ftz(dy.unsafe_load((t*Int(nh_in)+h)*M3_D_STATE+n)))
        var x=ftz(raw.unsafe_load(t*Int(dip_in)+Int(col_in)+n));dx.unsafe_store(t*M3_D_STATE+n,ftz(pinned_mul(r,ftz(ftz(pinned_mul(g,ftz(weight.unsafe_load(n))))-ftz(pinned_mul(x,corr))))))


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
                direct=ftz(identical_mul_add(dy,ftz(pinned_mul(qv,ev)),direct))
        var idx=((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n)
        d_state_direct.unsafe_store(idx,direct)
        var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)))
        var total=ftz(direct+ftz(pinned_mul(carry,ftz(identical_exp(last)))))
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
        d_q_read.unsafe_store(th*M3_D_STATE+n,ftz(pinned_mul(dq,ev)))
        read_scalar=ftz(identical_mul_add(ftz(q.unsafe_load(th*M3_D_STATE+n)),dq,read_scalar))
    d_dacs_read.unsafe_store(th,ftz(pinned_mul(read_scalar,ev)))
    var last=ftz(dacs.unsafe_load(((bb*nh+h)*nc+c)*qs+(qs-1)));var dec=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(dcidx)))))
    var rec_scalar=Float32(0.0)
    for n in range(M3_D_STATE):
        var dk=Float32(0.0)
        for p in range(M3_HEADDIM):
            var carry=Float32(0.0)
            if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            dk=ftz(identical_mul_add(carry,ftz(v.unsafe_load(th*M3_HEADDIM+p)),dk))
        d_k_rec.unsafe_store(th*M3_D_STATE+n,ftz(pinned_mul(dk,dec)))
    for p in range(M3_HEADDIM):
        var dv=Float32(0.0)
        for n in range(M3_D_STATE):
            var carry=Float32(0.0)
            if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
            dv=ftz(identical_mul_add(carry,ftz(k.unsafe_load(th*M3_D_STATE+n)),dv))
        var outv=ftz(pinned_mul(dv,dec));d_v_rec.unsafe_store(th*M3_HEADDIM+p,outv)
        rec_scalar=ftz(identical_mul_add(outv,ftz(v.unsafe_load(th*M3_HEADDIM+p)),rec_scalar))
    var dr=ftz(-rec_scalar)
    if inner==qs-1 or token==l-1:
        var add=Float32(0.0)
        for j in range(qs):
            var tj=c*qs+j
            if tj<l:
                var idx=((bb*nh+h)*nc+c)*qs+j;var de=ftz(identical_exp(ftz(last-ftz(dacs.unsafe_load(idx)))))
                for p in range(M3_HEADDIM):
                    for n in range(M3_D_STATE):
                        var carry=Float32(0.0)
                        if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                        add=ftz(identical_mul_add(carry,ftz(pinned_mul(ftz(pinned_mul(ftz(v.unsafe_load(((bb*l+tj)*nh+h)*M3_HEADDIM+p)),ftz(k.unsafe_load(((bb*l+tj)*nh+h)*M3_D_STATE+n)))),de)),add))
        for p in range(M3_HEADDIM):
            for n in range(M3_D_STATE):
                var carry=Float32(0.0)
                if c+1<nc:carry=ftz(d_states.unsafe_load((((bb*nc+c+1)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                var hs=ftz(states.unsafe_load((((bb*nc+c)*nh+h)*M3_HEADDIM+p)*M3_D_STATE+n))
                add=ftz(identical_mul_add(carry,ftz(pinned_mul(hs,ftz(identical_exp(last)))),add))
        dr=ftz(dr+add)
    d_dacs_rec.unsafe_store(th,dr)


def mamba3_backward_s17_operands_into(ctx:DeviceContext,mut dq:DeviceBuffer[DType.float32],mut ddr:DeviceBuffer[DType.float32],mut dk:DeviceBuffer[DType.float32],mut dv:DeviceBuffer[DType.float32],mut ddc:DeviceBuffer[DType.float32],mut dy:DeviceBuffer[DType.float32],mut q:DeviceBuffer[DType.float32],mut k:DeviceBuffer[DType.float32],mut v:DeviceBuffer[DType.float32],mut dacs:DeviceBuffer[DType.float32],mut states:DeviceBuffer[DType.float32],mut dstates:DeviceBuffer[DType.float32],b:Int,l:Int,dims:Mamba3Dims,qs:Int) raises:
    var cells=b*l*dims.nheads
    ctx.enqueue_function[mamba3_s17_operands_kernel](dq.unsafe_ptr(),ddr.unsafe_ptr(),dk.unsafe_ptr(),dv.unsafe_ptr(),ddc.unsafe_ptr(),dy.unsafe_ptr(),q.unsafe_ptr(),k.unsafe_ptr(),v.unsafe_ptr(),dacs.unsafe_ptr(),states.unsafe_ptr(),dstates.unsafe_ptr(),Int32(b),Int32(l),Int32(dims.nheads),Int32(qs),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))


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
