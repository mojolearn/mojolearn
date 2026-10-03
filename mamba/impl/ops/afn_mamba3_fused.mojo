# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Mamba-3 SISO forward: the per-token elementwise stages fused (lane
afn-mamba, 2026-10-03; `-D MOJOLEARN_AFN_MAMBA3_SISO_FUSED`, FAST + Apple
only; `mamba3_block_forward` and `m3_siso_forward` dispatch here).

Main's forward launches nine small elementwise kernels around the SISO
core (a_dt, assemble_vnew, pre, scale, angle_increment, rot, kscale,
dacs, state_decay, reports) and waits five times. On the M3 the small
stages are launch and buffer bound, not kernel bound, so this file keeps
every seam's arithmetic and cuts the launches:

  afn_m3_prep_kernel         S5 + S6 (A, dt) + the working-row copies
                             (dt, ADT, sigma) + the new rows' raw v:
                             a_dt + pre + assemble_vnew in ONE launch, one
                             thread per (b, li, h).
  afn_m3_scale_angle_kernel  S9 (gamma, beta', scale) over every working
                             row + the S10 angle increments of the new
                             rows: scale + angle_increment in ONE launch.
  afn_m3_rot_kscale_kernel   S12/S11/S13 rotation of the new rows + S15
                             k scaling of EVERY working row + the k_last /
                             v_last reports: rot + kscale + reports in ONE
                             launch, one thread per (b, t, h, pair).
  afn_m3_dacs_decay_kernel   the per-chunk ADT cumsum (mamba2 S11) + the
                             state-pass decays exp(last - cs): dacs +
                             state_decay in ONE launch.

The serial S10 angle chain (`m3_angle_kernel`) stays as it is: 384
threads x L steps of a few flops is not where the time goes, and its
mod-2pi-per-step spelling is a seam. The sabotage check arms of the
replaced kernels (A_FLOOR_UNCLAMPED, TRAP_LEFT_ONLY, SEGSUM_DESCENDING,
ROTATE_HALF_SPLIT) are IDENTICAL falsifiers and are not spelled here.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    ftz,
    identical_clamp,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sigmoid,
    identical_softplus,
    identical_tanh,
    portable_cosf,
    portable_sinf,
)
from mamba.checks.mamba3_fixture import (
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    M3_PI,
)

comptime AFN_M3_TPB = 128


def _afn_grid(n: Int) -> Int:
    var g = (n + AFN_M3_TPB - 1) // AFN_M3_TPB
    if g < 1:
        return 1
    return g


def afn_m3_prep_kernel(
    a_out: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    dt_out: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    adt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    sig_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    dt_bias: MutPointer[Float32, MutAnyOrigin],  # [H]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_dt_in: Int32,
    c_a_in: Int32,
    c_trap_in: Int32,
    c_x_in: Int32,
    neg_inf: Float32,
    neg_floor: Float32,
):
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_dt = Int(c_dt_in)
    var c_a = Int(c_a_in)
    var c_trap = Int(c_trap_in)
    var c_x = Int(c_x_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh:
        return
    var bb = cell // (l * nh)
    var rem = cell - bb * l * nh
    var li = rem // nh
    var hh = rem - li * nh
    var mm = bb * l + li
    var widx = (bb * t_work + q0 + li) * nh + hh

    # S5: piecewise heavy-tail, negate exact, clamp (m3_a_dt_kernel).
    var xa = ftz(in_proj.unsafe_load(mm * dip + c_a + hh))
    var ht: Float32
    if xa >= Float32(0.0):
        ht = ftz(Float32(1.0) + xa)
    else:
        ht = ftz(identical_div(Float32(1.0), ftz(Float32(1.0) - xa)))
    var av = ftz(identical_clamp(-ht, neg_inf, neg_floor))
    a_out.unsafe_store(mm * nh + hh, av)
    # S6: dt = softplus(dd_dt + dt_bias).
    var biased = ftz(
        ftz(in_proj.unsafe_load(mm * dip + c_dt + hh)) + ftz(dt_bias.unsafe_load(hh))
    )
    var dtv = ftz(identical_softplus(biased))
    dt_out.unsafe_store(mm * nh + hh, dtv)
    # m3_pre_kernel: the working-row copies and products.
    dt_work.unsafe_store(widx, dtv)
    adt_work.unsafe_store(widx, ftz(identical_mul(ftz(av), ftz(dtv))))
    sig_work.unsafe_store(
        widx, ftz(identical_sigmoid(ftz(in_proj.unsafe_load(mm * dip + c_trap + hh))))
    )
    # m3_assemble_vnew_kernel: the new row's raw v (copies).
    for p in range(p_dim):
        v_work.unsafe_store(
            widx * p_dim + p, in_proj.unsafe_load(mm * dip + c_x + hh * p_dim + p)
        )


def afn_m3_scale_angle_kernel(
    gamma_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    betap_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    scale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    theta_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, R] (increments)
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    sig_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_ang_in: Int32,
):
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_ang = Int(c_ang_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t_work * nh:
        return
    var bb = cell // (t_work * nh)
    var rem = cell - bb * t_work * nh
    var t = rem // nh
    var hh = rem - t * nh
    # S9 (m3_scale_kernel).
    var dtv = ftz(dt_work.unsafe_load(cell))
    var g = ftz(identical_mul(dtv, ftz(sig_work.unsafe_load(cell))))
    var bp = Float32(0.0)
    if t + 1 < t_work:
        var nxt = (bb * t_work + t + 1) * nh + hh
        bp = ftz(
            identical_mul(
                ftz(dt_work.unsafe_load(nxt)),
                ftz(Float32(1.0) - ftz(sig_work.unsafe_load(nxt))),
            )
        )
    gamma_work.unsafe_store(cell, g)
    betap_work.unsafe_store(cell, bp)
    scale_work.unsafe_store(cell, ftz(g + bp))
    # S10's increments for the new rows (m3_angle_increment_kernel).
    if t >= q0:
        var mm = bb * l + (t - q0)
        for r in range(r_ang):
            var a = ftz(
                identical_mul(
                    identical_tanh(ftz(in_proj.unsafe_load(mm * dip + c_ang + r))), M3_PI
                )
            )
            theta_out.unsafe_store(
                (mm * nh + hh) * r_ang + r, ftz(identical_mul(a, dtv))
            )


def afn_m3_rot_kscale_kernel(
    rotq_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    rotk_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    kscale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    k_last: MutPointer[Float32, MutAnyOrigin],  # [B, H, N]
    v_last: MutPointer[Float32, MutAnyOrigin],  # [B, H, P]
    theta_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, R] (angles)
    bcb: MutPointer[Float32, MutAnyOrigin],  # [M, N]
    bcc: MutPointer[Float32, MutAnyOrigin],  # [M, N]
    b_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    c_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    scale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    comptime r_ang = M3_NUM_ROPE_ANGLES
    comptime n_pairs = M3_D_STATE // 2
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t_work * nh * n_pairs:
        return
    var bb = cell // (t_work * nh * n_pairs)
    var rem = cell - bb * t_work * nh * n_pairs
    var t = rem // (nh * n_pairs)
    var rem2 = rem - t * nh * n_pairs
    var hh = rem2 // n_pairs
    var j = rem2 - hh * n_pairs
    var e0 = 2 * j
    var e1 = 2 * j + 1
    var row = (bb * t_work + t) * nh + hh
    var base = row * n_state
    var k0r: Float32
    var k1r: Float32
    if t >= q0:
        # S12 (bias after the norm) + S11/S13 (m3_rot_kernel), new rows.
        var mm = bb * l + (t - q0)
        var q0v = ftz(
            ftz(bcc.unsafe_load(mm * n_state + e0))
            + ftz(c_bias.unsafe_load(hh * n_state + e0))
        )
        var q1v = ftz(
            ftz(bcc.unsafe_load(mm * n_state + e1))
            + ftz(c_bias.unsafe_load(hh * n_state + e1))
        )
        var k0v = ftz(
            ftz(bcb.unsafe_load(mm * n_state + e0))
            + ftz(b_bias.unsafe_load(hh * n_state + e0))
        )
        var k1v = ftz(
            ftz(bcb.unsafe_load(mm * n_state + e1))
            + ftz(b_bias.unsafe_load(hh * n_state + e1))
        )
        if j < r_ang:
            var th = ftz(theta_out.unsafe_load((mm * nh + hh) * r_ang + j))
            var cv = ftz(portable_cosf(th))
            var sv = ftz(portable_sinf(th))
            rotq_work.unsafe_store(
                base + e0,
                ftz(ftz(identical_mul(q0v, cv)) - ftz(identical_mul(q1v, sv))),
            )
            rotq_work.unsafe_store(
                base + e1,
                ftz(ftz(identical_mul(q0v, sv)) + ftz(identical_mul(q1v, cv))),
            )
            k0r = ftz(ftz(identical_mul(k0v, cv)) - ftz(identical_mul(k1v, sv)))
            k1r = ftz(ftz(identical_mul(k0v, sv)) + ftz(identical_mul(k1v, cv)))
        else:
            # STRUCTURAL identity: never-computed trig (DEVIATION 828).
            rotq_work.unsafe_store(base + e0, q0v)
            rotq_work.unsafe_store(base + e1, q1v)
            k0r = k0v
            k1r = k1v
        rotk_work.unsafe_store(base + e0, k0r)
        rotk_work.unsafe_store(base + e1, k1r)
    else:
        # A buffered row: rotated k arrived from the state buffer.
        k0r = rotk_work.unsafe_load(base + e0)
        k1r = rotk_work.unsafe_load(base + e1)
    # S15 (m3_kscale_kernel) over every working row.
    var sc = scale_work.unsafe_load(row)
    kscale_work.unsafe_store(base + e0, ftz(identical_mul(ftz(k0r), sc)))
    kscale_work.unsafe_store(base + e1, ftz(identical_mul(ftz(k1r), sc)))
    # The reports (m3_reports_kernel): the LAST working row's pre-scale k
    # and raw v (copies).
    if t == t_work - 1:
        k_last.unsafe_store((bb * nh + hh) * n_state + e0, k0r)
        k_last.unsafe_store((bb * nh + hh) * n_state + e1, k1r)
        if j < p_dim // 2:
            v_last.unsafe_store(
                (bb * nh + hh) * p_dim + e0, v_work.unsafe_load(row * p_dim + e0)
            )
            v_last.unsafe_store(
                (bb * nh + hh) * p_dim + e1, v_work.unsafe_load(row * p_dim + e1)
            )


def afn_m3_dacs_decay_kernel(
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    decay: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q + 1] (qk_s scratch)
    adt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * nc:
        return
    var bb = cell // (nh * nc)
    var rem = cell - bb * nh * nc
    var hh = rem // nc
    var c = rem - hh * nc
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var dbase = cell * qv
    # The cumsum, SERIAL ASCENDING; padded positions COPY the last real
    # value (m3_dacs_kernel).
    var run = Float32(0.0)
    for i in range(qv):
        if i < real:
            var v = ftz(adt_work.unsafe_load((bb * t_work + c0 + i) * nh + hh))
            if i == 0:
                run = v
            else:
                run = ftz(run + v)
        dacs.unsafe_store(dbase + i, run)
    # The state-pass decays (m3_state_decay_kernel): exp(last - cs_j) for
    # j < Q, exp(last) at j == Q. `run` is the last value; this thread
    # wrote the prefix it reads back.
    var dl = ftz(run)
    var ebase = cell * (qv + 1)
    for j in range(qv):
        var arg = ftz(dl - ftz(dacs.unsafe_load(dbase + j)))
        decay.unsafe_store(ebase + j, ftz(identical_exp(arg)))
    decay.unsafe_store(ebase + qv, ftz(identical_exp(dl)))


# ===========================================================================
# Launchers (ASYNCHRONOUS, the block waits once at the end of the call).
# ===========================================================================


def afn_m3_prep(
    ctx: DeviceContext,
    mut a_out: DeviceBuffer[DType.float32],
    mut dt_out: DeviceBuffer[DType.float32],
    mut dt_work: DeviceBuffer[DType.float32],
    mut adt_work: DeviceBuffer[DType.float32],
    mut sig_work: DeviceBuffer[DType.float32],
    mut v_work: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    mut dt_bias: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    q0: Int,
    nh: Int,
    dip: Int,
    c_dt: Int,
    c_a: Int,
    c_trap: Int,
    c_x: Int,
    neg_inf: Float32,
    neg_floor: Float32,
) raises:
    ctx.enqueue_function[afn_m3_prep_kernel](
        a_out.unsafe_ptr(),
        dt_out.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        adt_work.unsafe_ptr(),
        sig_work.unsafe_ptr(),
        v_work.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        dt_bias.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(dip),
        Int32(c_dt),
        Int32(c_a),
        Int32(c_trap),
        Int32(c_x),
        neg_inf,
        neg_floor,
        grid_dim=(_afn_grid(b * l * nh), 1, 1),
        block_dim=(AFN_M3_TPB, 1, 1),
    )


def afn_m3_scale_angle(
    ctx: DeviceContext,
    mut gamma_work: DeviceBuffer[DType.float32],
    mut betap_work: DeviceBuffer[DType.float32],
    mut scale_work: DeviceBuffer[DType.float32],
    mut theta_out: DeviceBuffer[DType.float32],
    mut dt_work: DeviceBuffer[DType.float32],
    mut sig_work: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    q0: Int,
    nh: Int,
    dip: Int,
    c_ang: Int,
) raises:
    var t_work = q0 + l
    ctx.enqueue_function[afn_m3_scale_angle_kernel](
        gamma_work.unsafe_ptr(),
        betap_work.unsafe_ptr(),
        scale_work.unsafe_ptr(),
        theta_out.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        sig_work.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(dip),
        Int32(c_ang),
        grid_dim=(_afn_grid(b * t_work * nh), 1, 1),
        block_dim=(AFN_M3_TPB, 1, 1),
    )


def afn_m3_rot_kscale(
    ctx: DeviceContext,
    mut rotq_work: DeviceBuffer[DType.float32],
    mut rotk_work: DeviceBuffer[DType.float32],
    mut kscale_work: DeviceBuffer[DType.float32],
    mut k_last: DeviceBuffer[DType.float32],
    mut v_last: DeviceBuffer[DType.float32],
    mut theta_out: DeviceBuffer[DType.float32],
    mut bcb: DeviceBuffer[DType.float32],
    mut bcc: DeviceBuffer[DType.float32],
    mut b_bias: DeviceBuffer[DType.float32],
    mut c_bias: DeviceBuffer[DType.float32],
    mut scale_work: DeviceBuffer[DType.float32],
    mut v_work: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    q0: Int,
    nh: Int,
) raises:
    var t_work = q0 + l
    ctx.enqueue_function[afn_m3_rot_kscale_kernel](
        rotq_work.unsafe_ptr(),
        rotk_work.unsafe_ptr(),
        kscale_work.unsafe_ptr(),
        k_last.unsafe_ptr(),
        v_last.unsafe_ptr(),
        theta_out.unsafe_ptr(),
        bcb.unsafe_ptr(),
        bcc.unsafe_ptr(),
        b_bias.unsafe_ptr(),
        c_bias.unsafe_ptr(),
        scale_work.unsafe_ptr(),
        v_work.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        grid_dim=(_afn_grid(b * t_work * nh * (M3_D_STATE // 2)), 1, 1),
        block_dim=(AFN_M3_TPB, 1, 1),
    )


def afn_m3_dacs_decay(
    ctx: DeviceContext,
    mut dacs: DeviceBuffer[DType.float32],
    mut decay: DeviceBuffer[DType.float32],
    mut adt_work: DeviceBuffer[DType.float32],
    b: Int,
    t_work: Int,
    nh: Int,
    nc: Int,
    qv: Int,
) raises:
    ctx.enqueue_function[afn_m3_dacs_decay_kernel](
        dacs.unsafe_ptr(),
        decay.unsafe_ptr(),
        adt_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_afn_grid(b * nh * nc), 1, 1),
        block_dim=(AFN_M3_TPB, 1, 1),
    )
