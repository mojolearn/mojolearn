# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-2 SSD core's math, written ONCE (Andrew, 2026-11-22: one body per operation; lane neural-blocks-3).

Every form of the chunked SSD scan calls these `@always_inline` bodies; none carries its own copy:

  - the Mamba-2 block's cell kernels (`mamba/impl/modules/ssd_minimal.mojo` S10-S19 forward,
    `mamba/impl/modules/mamba2.mojo::m2_skip_kernel` S20, `autograd/kernels/mamba2_ssd.mojo` and
    `autograd/kernels/mamba2.mojo::mamba2_d_skip_backward_kernel` backward): each kernel computes its thread
    index and calls `<kernel stem>_cell` with it;
  - their host restatement (`tools/mamba_host_gen.py`: the generated kernels call these same bodies);
  - the generic op (`autograd/ssm_m2_ops.mojo`, the public `ssd_scan` layer) on the host and the device:
    its element function calls the same body with its element index, so a model composed from the public
    layer runs the block's arithmetic in the block's order.

The tiled forward kernels (`m2_cb_g_tile_kernel`, `m2_ydiag_tile_kernel`, `m2_cstate_tile_kernel`,
`m2_yoff_tile_kernel`) and the backward tile (`mamba2_ydiag_xd_backward_tile_kernel`) stage operands in
threadgroup memory and run the SAME chains; they call the step bodies below (`m2_*_step`, `m2_fold2`).

Each `*_cell` body is its kernel's former text, the thread index made an argument: no float operation moved.
The seams, DEVIATION numbers and sabotage arms are documented where they were first written (the headers
of ssd_minimal.mojo and mamba2_ssd.mojo). The forward sabotage arms and the S12 lower-triangle default live
here because they are read inside the bodies; ssd_minimal.mojo imports them back for its own readers.
A body takes typed values and pointers and has no launch, allocation or GPU import in it.
"""
from std.sys.compile import is_defined
from checks.numerics import (
    ftz,
    identical_exp,
    identical_mul_add,
    identical_mul,
    identical_sigmoid,
    identical_softplus,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_STATE,
    M2_HEADDIM,
)

# ---------------------------------------------------------------- forward sabotage arms (moved from
# mamba/impl/modules/ssd_minimal.mojo, 2026-11-22; that file's header documents each)
comptime SAB_2712_UNBOUNDED = is_defined["MOJOLEARN_MAMBA_2712_UNBOUNDED"]()
comptime SAB_SEGSUM_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_SEGSUM_DESCENDING"
]()
comptime SAB_CHUNK_SIZE_128 = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_CHUNK_SIZE_128"
]()
comptime SAB_STATEPASS_MATRIX = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_STATEPASS_MATRIX"
]()
comptime SAB_STATEPASS_UNFUSED = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_STATEPASS_UNFUSED"
]()
comptime SAB_PAIR_DT_B = is_defined["MOJOLEARN_MAMBA2_SABOTAGE_PAIR_DT_B"]()
comptime SAB_FOLD_SERIAL_ZERO_SEED = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_FOLD_SERIAL_ZERO_SEED"
]()

#: lane nr-mamba (2026-10-04, roadmap B9) IDN_M2_CB_LOWER (IDENTICAL, default ON): S12's G = C.B folds
#: only j <= i and writes an explicit +0.0 above the diagonal (moved from ssd_minimal.mojo, whose comment
#: carries the full note and the tombstone of its _OFF arm).
comptime IDN_M2_CB_LOWER = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def m2_q_eff() -> Int:
    """THE chunk size the build runs. DEVIATION 783: Q is a profile constant with `K_LEAF_MIN`'s standing --
    every reduction boundary in S11-S18 moves with it, so this function is the ONE site the CHUNK_SIZE_128
    arm may touch, and everything downstream (chunk counts, stage shapes, fold boundaries) derives from its
    answer. (Moved from ssd_minimal.mojo, 2026-11-22.)"""
    comptime if SAB_CHUNK_SIZE_128:
        return 128
    return M2_CHUNK_SIZE


# ---------------------------------------------------------------- step bodies (the tiles call these)


@always_inline
def m2_fma_step(a: Float32, b: Float32, acc: Float32) -> Float32:
    """One gemm.fp32.v1 leaf step: `ftz(fma(a, b, acc))` (operands already flushed)."""
    return ftz(identical_mul_add(a, b, acc))


@always_inline
def m2_product(a: Float32, b: Float32) -> Float32:
    """One rounded product `ftz(a * b)` (S13's M = G o L, S15's B o decay, S10's x dt and dt A)."""
    return ftz(identical_mul(a, b))


@always_inline
def m2_fold2(leaf0: Float32, leaf1: Float32, two_leaves: Bool) -> Float32:
    """fold_balanced_tree at P = 2 over a k = Q contraction: one node when there are two leaves, none when one."""
    if two_leaves:
        return m2_sum(leaf0, leaf1)
    return ftz(leaf0)


@always_inline
def m2_exp_step(x: Float32) -> Float32:
    """One flushed exponential `ftz(exp(x))` (S12's L, S15's decay, S17's scale, S18's exp(dA_cs))."""
    return ftz(identical_exp(x))


@always_inline
def m2_sum(a: Float32, b: Float32) -> Float32:
    """One flushed add of flushed operands `ftz(ftz(a) + ftz(b))` (S19's Y = Y_diag + Y_off, the fold node)."""
    return ftz(ftz(a) + ftz(b))


@always_inline
def m2_yoff_y_cell(
    cell: Int,
    yoff: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    y_out: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    ydiag: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    pass_states: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, P, N]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_yoff_y_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    comptime n_state = M2_D_STATE
    var pn = M2_HEADDIM * n_state
    if cell >= b * t_work * nh * M2_HEADDIM:
        return
    var bb = cell // (t_work * nh * M2_HEADDIM)
    var rem = cell - bb * t_work * nh * M2_HEADDIM
    var t = rem // (nh * M2_HEADDIM)
    var rem2 = rem - t * nh * M2_HEADDIM
    var hh = rem2 // M2_HEADDIM
    var p = rem2 - hh * M2_HEADDIM
    var c = t // qv
    var i = t - c * qv
    var ti = bb * t_work + t

    # C . h_prev over n: one serial ascending leaf (k = 128).
    var acc = Float32(0.0)
    for n in range(n_state):
        acc = m2_fma_step(
                ftz(xbc.unsafe_load(ti * cd + di + n_state + n)),
                ftz(
                    pass_states.unsafe_load(
                        (((bb * nc + c) * nh + hh) * pn) + p * n_state + n
                    )
                ),
                acc)
    acc = ftz(acc)
    var sc = m2_exp_step(
            ftz(dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + i)))
    var yo = m2_product(acc, sc)
    yoff.unsafe_store(cell, yo)
    y_out.unsafe_store(cell, m2_sum(ydiag.unsafe_load(cell), yo))


@always_inline
def m2_statepass_cell(
    cell: Int,
    pass_states: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, P, N]
    h_last: MutPointer[Float32, MutAnyOrigin],  # [B, H, P, N]
    h_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, P, N] in/out
    cstate: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, P, N]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    b_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
    n_completed_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_statepass_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var n_completed = Int(n_completed_in)
    comptime n_state = M2_D_STATE
    var pn = M2_HEADDIM * n_state
    if cell >= b * nh * pn:
        return
    var bb = cell // (nh * pn)
    var rem = cell - bb * nh * pn
    var hh = rem // pn
    var i = rem - hh * pn  # (p, n) flattened

    var h = ftz(h_state.unsafe_load(cell))

    comptime if SAB_STATEPASS_MATRIX:
        # SABOTAGE (785's falsifier): ssd_minimal:64-69's decay-MATRIX
        # contraction. Multi-chunk decays associate as exp(sum of logs)
        # where the recurrence associates them as products of exps --
        # different bits from three chunks up, or two with a nonzero
        # initial state. states_cat[z]: z = 0 is the incoming h, z >= 1 is
        # chunk z-1's cstate; v[z] = dacs_last of chunk z-1, v[0] = 0.
        var boundary = h
        for z in range(1, nc + 1):
            # decay(z, c) = exp(sum_{k=c+1..z} v[k]); new = sum_c decay*state
            var acc = Float32(0.0)
            for cc in range(z + 1):
                var s = Float32(0.0)
                var k = cc + 1
                while k <= z:
                    s = ftz(
                        s
                        + ftz(
                            dacs.unsafe_load(
                                ((bb * nh + hh) * nc + (k - 1)) * qv
                                + (qv - 1)
                            )
                        )
                    )
                    k += 1
                var dec = m2_exp_step(s)
                var sv: Float32
                if cc == 0:
                    sv = ftz(h_state.unsafe_load(cell))
                else:
                    sv = ftz(
                        cstate.unsafe_load(
                            (((bb * nc + (cc - 1)) * nh + hh) * pn) + i
                        )
                    )
                acc = m2_fma_step(dec, sv, acc)
            # the state ENTERING chunk z (z < nc) is new_states[z]... the
            # ENTERING state for chunk c is the z = c row; z = nc is the
            # final state.
            if z <= nc - 1:
                pass_states.unsafe_store(
                    (((bb * nc + z) * nh + hh) * pn) + i, acc
                )
            if z == n_completed:
                boundary = acc
            if z == nc:
                h_last.unsafe_store(cell, acc)
        # chunk 0's entering state is the incoming h in this spelling too.
        pass_states.unsafe_store(
            (((bb * nc + 0) * nh + hh) * pn) + i, h
        )
        if n_completed > 0:
            h_state.unsafe_store(cell, boundary)
        return

    var boundary = h
    for c in range(nc):
        pass_states.unsafe_store((((bb * nc + c) * nh + hh) * pn) + i, h)
        var scale = m2_exp_step(
                ftz(
                    dacs.unsafe_load(
                        ((bb * nh + hh) * nc + c) * qv + (qv - 1)
                    )
                ))
        var cs = ftz(
            cstate.unsafe_load((((bb * nc + c) * nh + hh) * pn) + i)
        )
        comptime if SAB_STATEPASS_UNFUSED:
            # SABOTAGE: two roundings where the profile has one.
            var prod = m2_product(scale, h)
            h = ftz(prod + cs)
        else:
            h = m2_fma_step(scale, h, cs)
        if c == n_completed - 1:
            boundary = h
    h_last.unsafe_store(cell, h)
    if n_completed > 0:
        h_state.unsafe_store(cell, boundary)


@always_inline
def m2_cstate_cell(
    cell: Int,
    cstate: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, P, N]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    decay: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    dt: MutPointer[Float32, MutAnyOrigin],  # [B, T, H] (PAIR_DT_B arm only)
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_cstate_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    comptime n_state = M2_D_STATE
    if cell >= b * nc * nh * M2_HEADDIM * n_state:
        return
    var bb = cell // (nc * nh * M2_HEADDIM * n_state)
    var rem = cell - bb * nc * nh * M2_HEADDIM * n_state
    var c = rem // (nh * M2_HEADDIM * n_state)
    var rem2 = rem - c * nh * M2_HEADDIM * n_state
    var hh = rem2 // (M2_HEADDIM * n_state)
    var rem3 = rem2 - hh * M2_HEADDIM * n_state
    var p = rem3 // n_state
    var n = rem3 - p * n_state
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    # contract_leaf_size(k): see m2_ydiag_kernel.
    var leaf_len = qv
    if qv > 128:
        leaf_len = qv // 2

    var leaf0 = Float32(0.0)
    var leaf1 = Float32(0.0)
    for i in range(qv):
        var bd: Float32
        var xv: Float32
        if i < real:
            var ti = bb * t_work + c0 + i
            var dec = ftz(
                decay.unsafe_load(((bb * nh + hh) * nc + c) * qv + i)
            )
            bd = m2_product(ftz(xbc.unsafe_load(ti * cd + di + n)), dec)
            comptime if SAB_PAIR_DT_B:
                bd = m2_product(bd, ftz(dt.unsafe_load(ti * nh + hh)))
            xv = ftz(
                xd.unsafe_load((ti * nh + hh) * M2_HEADDIM + p)
            )
        else:
            # Padded row: B is exact +0.0, so B_decay is the +0.0 the
            # oracle's zero-filled matrix holds, and the product below is
            # an exact zero.
            bd = Float32(0.0)
            xv = Float32(0.0)
        comptime if SAB_FOLD_SERIAL_ZERO_SEED:
            leaf0 = m2_fma_step(xv, bd, leaf0)
        else:
            if i < leaf_len:
                leaf0 = m2_fma_step(xv, bd, leaf0)
            else:
                leaf1 = m2_fma_step(xv, bd, leaf1)
    var out: Float32
    comptime if SAB_FOLD_SERIAL_ZERO_SEED:
        out = ftz(leaf0)
    else:
        out = m2_fold2(leaf0, leaf1, leaf_len < qv)
    cstate.unsafe_store(cell, out)


@always_inline
def m2_decay_cell(
    cell: Int,
    decay: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    n_in: Int32,  # B * H * C * Q
    q_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_decay_kernel`'s thread body (one body per operation)."""
    var n = Int(n_in)
    var qv = Int(q_in)
    if cell >= n:
        return
    var row = cell // qv  # (b, h, c)
    var last = ftz(dacs.unsafe_load(row * qv + (qv - 1)))
    var d = ftz(last - ftz(dacs.unsafe_load(cell)))
    decay.unsafe_store(cell, m2_exp_step(d))


@always_inline
def m2_ydiag_cell(
    cell: Int,
    ydiag: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    cb_g: MutPointer[Float32, MutAnyOrigin],  # [B, C, Q, Q]
    seg_l: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    dt: MutPointer[Float32, MutAnyOrigin],  # [B, T, H] (PAIR_DT_B arm only)
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_ydiag_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * t_work * nh * M2_HEADDIM:
        return
    var bb = cell // (t_work * nh * M2_HEADDIM)
    var rem = cell - bb * t_work * nh * M2_HEADDIM
    var t = rem // (nh * M2_HEADDIM)
    var rem2 = rem - t * nh * M2_HEADDIM
    var hh = rem2 // M2_HEADDIM
    var p = rem2 - hh * M2_HEADDIM
    var c = t // qv
    var i = t - c * qv
    var gbase = ((bb * nc + c) * qv + i) * qv
    var lbase = (((bb * nc + c) * nh + hh) * qv) * qv + i * qv
    # contract_leaf_size(k): k = 256 -> two leaves of 128; k = 128 (the
    # CHUNK_SIZE_128 arm) -> ONE leaf of 128.
    var leaf_len = qv
    if qv > 128:
        leaf_len = qv // 2

    # Leaf partials: +0.0-seeded serial-ascending fma chains, one per leaf
    # (oracle_leaf_partial's loop, in registers).
    var leaf0 = Float32(0.0)
    var leaf1 = Float32(0.0)
    for jj in range(qv):
        var m_ij: Float32
        if jj <= i:
            m_ij = m2_product(
                    ftz(cb_g.unsafe_load(gbase + jj)),
                    ftz(seg_l.unsafe_load(lbase + jj)))
            comptime if SAB_PAIR_DT_B:
                # SABOTAGE: the fused chain scales the CB block by dt[j]
                # (dt bound to the B side) since X_d carries no dt.
                m_ij = m2_product(
                        m_ij,
                        ftz(
                            dt.unsafe_load(
                                ((bb * t_work) + (c * qv + jj)) * nh + hh
                            )
                        ))
        else:
            # STRUCTURAL zero above the diagonal (DEVIATION 782): +0.0,
            # never a computed product -- the oracle's zero-filled M.
            m_ij = Float32(0.0)
        # DEVIATION 2712 / 2713: X_d is [B, T, H, P] and this loop runs to
        # Q, so rows c * Q + jj >= T are PAST the sequence. Their m_ij is the
        # structural +0.0, and 0 x finite garbage is 0, which is why every
        # CUDA and HIP run and every cold Metal run carried the same bits;
        # 0 x NaN (Metal, warm process) is NaN, and the row past the last
        # allocation is an unmapped page on the MI325X (the launch fault).
        # A row past T contributes +0.0 x +0.0, the bits the record carries.
        var xv = Float32(0.0)
        # -D MOJOLEARN_MAMBA_2712_UNBOUNDED=1 restores the unbounded read: the
        # poison gate's sabotage 1, which must fail (the band is NaN there).
        if c * qv + jj < t_work or SAB_2712_UNBOUNDED:
            xv = ftz(
                xd.unsafe_load(
                    (((bb * t_work) + (c * qv + jj)) * nh + hh) * M2_HEADDIM + p
                )
            )
        comptime if SAB_FOLD_SERIAL_ZERO_SEED:
            # SABOTAGE: one serial chain across the whole k = Q, the fold
            # levels erased (the gemm lane's F5 alternative).
            leaf0 = m2_fma_step(m_ij, xv, leaf0)
        else:
            if jj < leaf_len:
                leaf0 = m2_fma_step(m_ij, xv, leaf0)
            else:
                leaf1 = m2_fma_step(m_ij, xv, leaf1)
    var out: Float32
    comptime if SAB_FOLD_SERIAL_ZERO_SEED:
        out = ftz(leaf0)
    else:
        out = m2_fold2(leaf0, leaf1, leaf_len < qv)
    ydiag.unsafe_store(cell, out)


@always_inline
def m2_cb_g_cell(
    cell: Int,
    cb_g: MutPointer[Float32, MutAnyOrigin],  # [B, C, Q, Q]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    b_in: Int32,
    t_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_cb_g_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * nc * qv * qv:
        return
    var bb = cell // (nc * qv * qv)
    var rem = cell - bb * nc * qv * qv
    var c = rem // (qv * qv)
    var rem2 = rem - c * qv * qv
    var i = rem2 // qv
    var j = rem2 - i * qv
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var acc = Float32(0.0)
    var live = i < real and j < real
    comptime if IDN_M2_CB_LOWER:
        # B9: above the diagonal is never read; explicit +0.0 (below).
        live = live and j <= i
    if live:
        var ti = bb * t_work + c0 + i
        var tj = bb * t_work + c0 + j
        for n in range(M2_D_STATE):
            acc = m2_fma_step(
                    ftz(xbc.unsafe_load(ti * cd + di + M2_D_STATE + n)),  # C
                    ftz(xbc.unsafe_load(tj * cd + di + n)),  # B
                    acc)
        acc = ftz(acc)
    else:
        # A padded row's B or C is exact +0.0 everywhere: the leaf is a
        # fold of exact zeros from a +0.0 seed, which is +0.0 -- written
        # directly, same bits as the fold (gemm section 9's argument).
        acc = Float32(0.0)
    cb_g.unsafe_store(((bb * nc + c) * qv + i) * qv + j, acc)


@always_inline
def m2_seg_l_cell(
    cell: Int,
    seg_l: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    da: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_seg_l_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * nc * nh * qv:
        return
    var bb = cell // (nc * nh * qv)
    var rem = cell - bb * nc * nh * qv
    var c = rem // (nh * qv)
    var rem2 = rem - c * nh * qv
    var hh = rem2 // qv
    var j = rem2 - hh * qv
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var lbase = (((bb * nc + c) * nh + hh) * qv) * qv
    # Above the diagonal: +0.0, STRUCTURAL -- written, never computed.
    for i in range(j):
        seg_l.unsafe_store(lbase + i * qv + j, Float32(0.0))
    var acc = Float32(0.0)
    seg_l.unsafe_store(lbase + j * qv + j, m2_exp_step(acc))
    for i in range(j + 1, qv):
        if i < real:
            comptime if SAB_SEGSUM_DESCENDING:
                var a2 = Float32(0.0)
                var k = i
                while k > j:
                    a2 = ftz(
                        a2
                        + ftz(
                            da.unsafe_load(
                                ((bb * t_work) + (c0 + k)) * nh + hh
                            )
                        )
                    )
                    k -= 1
                acc = a2
            else:
                acc = ftz(
                    acc
                    + ftz(
                        da.unsafe_load(((bb * t_work) + (c0 + i)) * nh + hh)
                    )
                )
        seg_l.unsafe_store(lbase + i * qv + j, m2_exp_step(acc))


@always_inline
def m2_chunk_cumsum_cell(
    cell: Int,
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    da: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_chunk_cumsum_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
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
    var run = Float32(0.0)
    for i in range(qv):
        if i < real:
            var v = ftz(da.unsafe_load(((bb * t_work) + (c0 + i)) * nh + hh))
            comptime if SAB_SEGSUM_DESCENDING:
                # SABOTAGE: rebuild THIS prefix by a descending fold --
                # the fold order clause undone, one variable.
                var acc = Float32(0.0)
                var k = i
                while k >= 0:
                    acc = ftz(
                        acc
                        + ftz(
                            da.unsafe_load(
                                ((bb * t_work) + (c0 + k)) * nh + hh
                            )
                        )
                    )
                    k -= 1
                run = acc
            else:
                if i == 0:
                    run = v
                else:
                    run = ftz(run + v)
        # else: COPY of the last real value, not an add.
        dacs.unsafe_store(((bb * nh + hh) * nc + c) * qv + i, run)


@always_inline
def m2_discretize_cell(
    cell: Int,
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    da: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    dt: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    a_out: MutPointer[Float32, MutAnyOrigin],  # [H]
    bt_in: Int32,  # B * T
    nh_in: Int32,
    cd_in: Int32,
):
    """`mamba/impl/modules/ssd_minimal.mojo::m2_discretize_kernel`'s thread body (one body per operation)."""
    var bt = Int(bt_in)
    var nh = Int(nh_in)
    var cd = Int(cd_in)
    if cell >= bt * nh:
        return
    var t = cell // nh
    var hh = cell - t * nh
    var dtv = ftz(dt.unsafe_load(t * nh + hh))
    # dA = dt * A: dt binds to A (789). PAIR_DT_B does not move this
    # product -- the fused chain agrees about dA.
    da.unsafe_store(
        cell, m2_product(dtv, ftz(a_out.unsafe_load(hh)))
    )
    for p in range(M2_HEADDIM):
        var xv = ftz(xbc.unsafe_load(t * cd + hh * M2_HEADDIM + p))
        comptime if SAB_PAIR_DT_B:
            # SABOTAGE (789's falsifier): the fused chain keeps dt OUT of
            # X_d (`chunk_state_ref`:1122 carries it as a separate einsum
            # factor); the dt factors reappear bound to the B side in the
            # Y_diag and chunk_state kernels below. ONE pairing decision
            # moves, in all three of its sites at once.
            xd.unsafe_store(cell * M2_HEADDIM + p, xv)
        else:
            xd.unsafe_store(
                cell * M2_HEADDIM + p, m2_product(xv, dtv)
            )


@always_inline
def m2_skip_cell(
    cell: Int,
    skip_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    y_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    silu_out: MutPointer[Float32, MutAnyOrigin],  # [M, CD]
    d_skip: MutPointer[Float32, MutAnyOrigin],  # [H]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    cd_in: Int32,
):
    """`mamba/impl/modules/mamba2.mojo::m2_skip_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var cd = Int(cd_in)
    var t_work = q0 + l
    if cell >= b * l * nh * M2_HEADDIM:
        return
    var bb = cell // (l * nh * M2_HEADDIM)
    var rem = cell - bb * l * nh * M2_HEADDIM
    var li = rem // (nh * M2_HEADDIM)
    var rem2 = rem - li * nh * M2_HEADDIM
    var hh = rem2 // M2_HEADDIM
    var p = rem2 - hh * M2_HEADDIM
    var xv = ftz(
        silu_out.unsafe_load((bb * l + li) * cd + hh * M2_HEADDIM + p)
    )
    var prod = m2_product(xv, ftz(d_skip.unsafe_load(hh)))
    var y = ftz(
        y_work.unsafe_load(
            (((bb * t_work) + (q0 + li)) * nh + hh) * M2_HEADDIM + p
        )
    )
    skip_out.unsafe_store(cell, ftz(y + prod))


@always_inline
def mamba2_reverse_chunk_state_cell(
    cell: Int,
    d_pass: MutPointer[Float32, MutAnyOrigin],
    d_cstate: MutPointer[Float32, MutAnyOrigin],
    d_scale_product: MutPointer[Float32, MutAnyOrigin],
    d_initial: MutPointer[Float32, MutAnyOrigin],
    direct_d_pass: MutPointer[Float32, MutAnyOrigin],
    d_final: MutPointer[Float32, MutAnyOrigin],
    pass_states: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_reverse_chunk_state_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = M2_HEADDIM * M2_D_STATE
    if cell >= b * nh * pn:
        return
    var bb = cell // (nh * pn)
    var rem = cell - bb * nh * pn
    var hh = rem // pn
    var i = rem - hh * pn
    var carry = ftz(d_final.unsafe_load(cell))
    var c = nc - 1
    while c >= 0:
        var state_idx = (((bb * nc + c) * nh + hh) * pn) + i
        var scale = m2_exp_step(
                ftz(
                    dacs.unsafe_load(
                        ((bb * nh + hh) * nc + c) * qv + (qv - 1)
                    )
                ))
        d_cstate.unsafe_store(state_idx, carry)
        d_scale_product.unsafe_store(
            state_idx,
            m2_product(carry, ftz(pass_states.unsafe_load(state_idx))),
        )
        var direct = ftz(direct_d_pass.unsafe_load(state_idx))
        carry = m2_fma_step(scale, carry, direct)
        d_pass.unsafe_store(state_idx, carry)
        c -= 1
    d_initial.unsafe_store(cell, carry)


@always_inline
def mamba2_scale_to_dacs_cell(
    row: Int,
    d_dacs_state: MutPointer[Float32, MutAnyOrigin],
    d_scale: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nc_in: Int32,
    nh_in: Int32,
    q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_scale_to_dacs_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var nc = Int(nc_in)
    var nh = Int(nh_in)
    var qv = Int(q_in)
    if row >= b * nc * nh:
        return
    var bb = row // (nc * nh)
    var rem = row - bb * nc * nh
    var c = rem // nh
    var h = rem - c * nh
    var dacs_idx = ((bb * nh + h) * nc + c) * qv + (qv - 1)
    var scale = m2_exp_step(ftz(dacs.unsafe_load(dacs_idx)))
    d_dacs_state.unsafe_store(
        dacs_idx, m2_product(ftz(d_scale.unsafe_load(row)), scale)
    )


@always_inline
def mamba2_merge_dacs_cell(
    i: Int,
    dst: MutPointer[Float32, MutAnyOrigin],
    yoff: MutPointer[Float32, MutAnyOrigin],
    state: MutPointer[Float32, MutAnyOrigin],
    decay: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_merge_dacs_kernel`'s thread body (one body per operation)."""
    if i < Int(n_in):
        dst.unsafe_store(
            i, ftz(ftz(ftz(yoff.unsafe_load(i)) + ftz(state.unsafe_load(i)))
                   + ftz(decay.unsafe_load(i)))
        )


@always_inline
def mamba2_da_cell_body(
    idx: Int,
    d_dt: MutPointer[Float32, MutAnyOrigin],
    d_da: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    bt_in: Int32, nh_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_da_cell_kernel`'s thread body (one body per operation)."""
    var bt = Int(bt_in); var nh = Int(nh_in)
    if idx >= bt * nh:
        return
    var hh = idx - (idx // nh) * nh
    d_dt.unsafe_store(idx, m2_product(
        ftz(d_da.unsafe_load(idx)), ftz(a.unsafe_load(hh))))


@always_inline
def m2_fold_tiles_cell(
    c: Int,
    out_a: MutPointer[Float32, MutAnyOrigin],  # columns [0, split)
    out_b: MutPointer[Float32, MutAnyOrigin],  # columns [split, cols)
    out_s: MutPointer[Float32, MutAnyOrigin],  # acc * scale, when has_scale
    scale: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],  # [tiles, cols]
    tiles_in: Int32, cols_in: Int32, split_in: Int32, has_scale: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::m2_fold_tiles_kernel`'s thread body (one body per operation)."""
    var tiles = Int(tiles_in); var cols = Int(cols_in); var split = Int(split_in)
    if c >= cols:
        return
    var acc = part.unsafe_load(c)
    for tile in range(1, tiles):
        acc = ftz(acc + ftz(part.unsafe_load(tile * cols + c)))
    if c < split:
        out_a.unsafe_store(c, acc)
        if has_scale != 0:
            out_s.unsafe_store(c, m2_product(acc, ftz(scale.unsafe_load(c))))
    else:
        out_b.unsafe_store(c - split, acc)


@always_inline
def m2_fold_rows_partial_cell(
    cell: Int,
    part: MutPointer[Float32, MutAnyOrigin],  # [tiles, cols]
    x: MutPointer[Float32, MutAnyOrigin],  # [rows, cols]
    y: MutPointer[Float32, MutAnyOrigin],  # [rows, cols] when is_dot
    rows_in: Int32, cols_in: Int32, per_in: Int32, tiles_in: Int32,
    is_dot: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::m2_fold_rows_partial_kernel`'s thread body (one body per operation)."""
    var rows = Int(rows_in); var cols = Int(cols_in)
    var per = Int(per_in); var tiles = Int(tiles_in)
    if cell >= tiles * cols:
        return
    var tile = cell // cols
    var c = cell - tile * cols
    var r0 = tile * per
    var r1 = r0 + per
    if r1 > rows:
        r1 = rows
    var acc = Float32(0.0)
    for r in range(r0, r1):
        var idx = r * cols + c
        if is_dot != 0:
            acc = m2_fma_step(
                ftz(x.unsafe_load(idx)), ftz(y.unsafe_load(idx)), acc)
        else:
            acc = ftz(acc + ftz(x.unsafe_load(idx)))
    part.unsafe_store(cell, acc)


@always_inline
def mamba2_ydiag_xd_backward_cell(
    cell: Int,
    d_xd: MutPointer[Float32, MutAnyOrigin],
    d_x: MutPointer[Float32, MutAnyOrigin],
    d_dt_xd: MutPointer[Float32, MutAnyOrigin],
    d_dt_merged: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    cb_g: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin],
    d_dt_da: MutPointer[Float32, MutAnyOrigin],
    d_xd_cstate: MutPointer[Float32, MutAnyOrigin],
    d_xd_total: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, di_in: Int32,
    cd_in: Int32, nc_in: Int32, q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_ydiag_xd_backward_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * t_work * nh:
        return
    var bb = cell // (t_work * nh)
    var rem = cell - bb * t_work * nh
    var tt = rem // nh
    var hh = rem - tt * nh
    var cc = tt // qv
    var jj = tt - cc * qv
    var real = t_work - cc * qv
    if real > qv:
        real = qv
    var ddt = Float32(0.0)
    var dtv = ftz(dt.unsafe_load(cell))
    for pp in range(M2_HEADDIM):
        var dxd = Float32(0.0)
        for ii in range(jj, real):
            var m = m2_product(
                ftz(cb_g.unsafe_load(((bb * nc + cc) * qv + ii) * qv + jj)),
                ftz(seg_l.unsafe_load(
                    (((bb * nc + cc) * nh + hh) * qv + ii) * qv + jj
                )))
            var dy = ftz(d_y.unsafe_load(
                ((bb * t_work + cc * qv + ii) * nh + hh)
                * M2_HEADDIM + pp
            ))
            dxd = m2_fma_step(m, dy, dxd)
        d_xd.unsafe_store(cell * M2_HEADDIM + pp, dxd)
        dxd = ftz(dxd + ftz(d_xd_cstate.unsafe_load(cell * M2_HEADDIM + pp)))
        d_xd_total.unsafe_store(cell * M2_HEADDIM + pp, dxd)
        var xp = ftz(xbc.unsafe_load(
            (bb * t_work + tt) * cd + hh * M2_HEADDIM + pp
        ))
        var out_idx = cell * M2_HEADDIM + pp
        d_x.unsafe_store(out_idx, m2_product(dxd, dtv))
        ddt = m2_fma_step(xp, dxd, ddt)
    d_dt_xd.unsafe_store(cell, ddt)
    d_dt_merged.unsafe_store(
        cell, ftz(ftz(d_dt_da.unsafe_load(cell)) + ddt)
    )


@always_inline
def mamba2_reverse_cumsum_cell(
    cell: Int,
    d_da: MutPointer[Float32, MutAnyOrigin],
    d_dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, nc_in: Int32, q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_reverse_cumsum_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * nh * nc:
        return
    var bb = cell // (nh * nc)
    var rem = cell - bb * nh * nc
    var hh = rem // nc
    var cc = rem - hh * nc
    var real = t_work - cc * qv
    if real > qv:
        real = qv
    var carry = Float32(0.0)
    var ii = qv - 1
    while ii >= 0:
        carry = ftz(carry + ftz(d_dacs.unsafe_load(
            ((bb * nh + hh) * nc + cc) * qv + ii
        )))
        if ii < real:
            d_da.unsafe_store(
                ((bb * t_work + cc * qv + ii) * nh + hh), carry
            )
        ii -= 1


@always_inline
def mamba2_ydiag_matrix_backward_cell(
    cell: Int,
    d_cb: MutPointer[Float32, MutAnyOrigin],
    d_seg: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    xd: MutPointer[Float32, MutAnyOrigin],
    cb: MutPointer[Float32, MutAnyOrigin],
    seg: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, nc_in: Int32, q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_ydiag_matrix_backward_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * nc * qv * qv:
        return
    var bb = cell // (nc * qv * qv)
    var rem = cell - bb * nc * qv * qv
    var cc = rem // (qv * qv)
    rem -= cc * qv * qv
    var ii = rem // qv
    var jj = rem - ii * qv
    var real = t - cc * qv
    if real > qv:
        real = qv
    var dcb = Float32(0.0)
    for hh in range(nh):
        var dm = Float32(0.0)
        if ii < real and jj <= ii:
            for pp in range(M2_HEADDIM):
                dm = m2_fma_step(
                    ftz(d_y.unsafe_load(((bb*t+cc*qv+ii)*nh+hh)*M2_HEADDIM+pp)),
                    ftz(xd.unsafe_load(((bb*t+cc*qv+jj)*nh+hh)*M2_HEADDIM+pp)), dm)
        var sidx = (((bb*nc+cc)*nh+hh)*qv+ii)*qv+jj
        d_seg.unsafe_store(sidx, m2_product(dm, ftz(cb.unsafe_load(cell))))
        dcb = m2_fma_step(dm, ftz(seg.unsafe_load(sidx)), dcb)
    d_cb.unsafe_store(cell, dcb)


@always_inline
def mamba2_seg_backward_cell(
    cell: Int,
    d_da_seg: MutPointer[Float32, MutAnyOrigin],
    d_da_total: MutPointer[Float32, MutAnyOrigin],
    d_seg: MutPointer[Float32, MutAnyOrigin],
    seg: MutPointer[Float32, MutAnyOrigin],
    d_da_s11: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, nc_in: Int32, q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_seg_backward_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * t * nh:
        return
    var bb = cell // (t * nh)
    var rem = cell - bb * t * nh
    var tt = rem // nh
    var hh = rem - tt * nh
    var cc = tt // qv
    var kk = tt - cc * qv
    var real = t - cc * qv
    if real > qv:
        real = qv
    var acc = Float32(0.0)
    for jj in range(kk):
        for ii in range(kk, real):
            var idx = (((bb * nc + cc) * nh + hh) * qv + ii) * qv + jj
            acc = m2_fma_step(
                ftz(d_seg.unsafe_load(idx)), ftz(seg.unsafe_load(idx)), acc)
    d_da_seg.unsafe_store(cell, acc)
    d_da_total.unsafe_store(cell, ftz(ftz(d_da_s11.unsafe_load(cell)) + acc))


@always_inline
def mamba2_cstate_dxd_db_cell(
    cell: Int,
    d_xd: MutPointer[Float32, MutAnyOrigin], d_b: MutPointer[Float32, MutAnyOrigin],
    d_b_total: MutPointer[Float32, MutAnyOrigin], d_b_cb: MutPointer[Float32, MutAnyOrigin],
    d_cstate: MutPointer[Float32, MutAnyOrigin], xd: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin], decay: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, di_in: Int32, cd_in: Int32,
    nc_in: Int32, q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_cstate_dxd_db_kernel`'s thread body (one body per operation)."""
    var b=Int(b_in); var t=Int(t_in); var nh=Int(nh_in); var di=Int(di_in)
    var cd=Int(cd_in); var nc=Int(nc_in); var qv=Int(q_in)
    var xd_cells=b*t*nh*M2_HEADDIM
    if cell < xd_cells:
        var bb=cell//(t*nh*M2_HEADDIM); var r=cell-bb*t*nh*M2_HEADDIM
        var tt=r//(nh*M2_HEADDIM); r-=tt*nh*M2_HEADDIM
        var hh=r//M2_HEADDIM; var pp=r-hh*M2_HEADDIM
        var cc=tt//qv; var ii=tt-cc*qv; var acc=Float32(0.0)
        var dec=ftz(decay.unsafe_load(((bb*nh+hh)*nc+cc)*qv+ii))
        for nn in range(M2_D_STATE):
            acc=m2_fma_step(
                ftz(d_cstate.unsafe_load((((bb*nc+cc)*nh+hh)*M2_HEADDIM+pp)*M2_D_STATE+nn)),
                m2_product(ftz(xbc.unsafe_load((bb*t+tt)*cd+di+nn)),dec),acc)
        d_xd.unsafe_store(cell,acc)
    var b_cells=b*t*M2_D_STATE
    if cell < b_cells:
        var bb=cell//(t*M2_D_STATE); var r=cell-bb*t*M2_D_STATE
        var tt=r//M2_D_STATE; var nn=r-tt*M2_D_STATE
        var cc=tt//qv; var ii=tt-cc*qv; var acc=Float32(0.0)
        for hh in range(nh):
            var dec=ftz(decay.unsafe_load(((bb*nh+hh)*nc+cc)*qv+ii))
            for pp in range(M2_HEADDIM):
                acc=m2_fma_step(
                    ftz(d_cstate.unsafe_load((((bb*nc+cc)*nh+hh)*M2_HEADDIM+pp)*M2_D_STATE+nn)),
                    m2_product(ftz(xd.unsafe_load(((bb*t+tt)*nh+hh)*M2_HEADDIM+pp)),dec),acc)
        d_b.unsafe_store(cell,acc)
        d_b_total.unsafe_store(cell,ftz(acc+ftz(d_b_cb.unsafe_load(cell))))


@always_inline
def mamba2_postconv_merge_cell(
    i: Int,
    d_c_total: MutPointer[Float32, MutAnyOrigin],
    d_x_total: MutPointer[Float32, MutAnyOrigin],
    d_c_yoff: MutPointer[Float32, MutAnyOrigin],
    d_c_cb: MutPointer[Float32, MutAnyOrigin],
    d_x_d: MutPointer[Float32, MutAnyOrigin],
    d_x_xd: MutPointer[Float32, MutAnyOrigin],
    c_cells_in: Int32,
    x_cells_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_postconv_merge_kernel`'s thread body (one body per operation)."""
    if i < Int(c_cells_in):
        d_c_total.unsafe_store(i, ftz(ftz(d_c_yoff.unsafe_load(i)) + ftz(d_c_cb.unsafe_load(i))))
    if i < Int(x_cells_in):
        d_x_total.unsafe_store(i, ftz(ftz(d_x_d.unsafe_load(i)) + ftz(d_x_xd.unsafe_load(i))))


@always_inline
def mamba2_decay_to_dacs_cell(
    row: Int,
    dst: MutPointer[Float32, MutAnyOrigin],
    d_decay: MutPointer[Float32, MutAnyOrigin],
    decay: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_decay_to_dacs_kernel`'s thread body (one body per operation)."""
    var rows = Int(rows_in)
    var qv = Int(q_in)
    if row >= rows:
        return
    var base = row * qv
    var last_sum = Float32(0.0)
    for ii in range(qv - 1):
        var g = m2_product(
            ftz(d_decay.unsafe_load(base + ii)),
            ftz(decay.unsafe_load(base + ii)))
        dst.unsafe_store(base + ii, ftz(-g))
        last_sum = ftz(last_sum + g)
    dst.unsafe_store(base + qv - 1, last_sum)


@always_inline
def mamba2_cb_backward_cell(
    cell: Int,
    d_b: MutPointer[Float32, MutAnyOrigin],
    d_c: MutPointer[Float32, MutAnyOrigin],
    d_cb: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, di_in: Int32, cd_in: Int32,
    nc_in: Int32, q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_cb_backward_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t = Int(t_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * t * M2_D_STATE:
        return
    var bb = cell // (t * M2_D_STATE)
    var rem = cell - bb * t * M2_D_STATE
    var tt = rem // M2_D_STATE
    var nn = rem - tt * M2_D_STATE
    var cc = tt // qv
    var pos = tt - cc * qv
    var real = t - cc * qv
    if real > qv:
        real = qv
    var db = Float32(0.0)
    var dc = Float32(0.0)
    for ii in range(pos, real):
        db = m2_fma_step(
            ftz(d_cb.unsafe_load(((bb*nc+cc)*qv+ii)*qv+pos)),
            ftz(xbc.unsafe_load((bb*t+cc*qv+ii)*cd+di+M2_D_STATE+nn)), db)
    for jj in range(pos + 1):
        dc = m2_fma_step(
            ftz(d_cb.unsafe_load(((bb*nc+cc)*qv+pos)*qv+jj)),
            ftz(xbc.unsafe_load((bb*t+cc*qv+jj)*cd+di+nn)), dc)
    d_b.unsafe_store(cell, db)
    d_c.unsafe_store(cell, dc)


@always_inline
def mamba2_s18_dc_ddacs_cell(
    cell: Int,
    d_c: MutPointer[Float32, MutAnyOrigin],
    d_dacs: MutPointer[Float32, MutAnyOrigin],
    d_yoff: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    pass_states: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_s18_dc_ddacs_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = M2_HEADDIM * M2_D_STATE
    var dc_cells = b * t_work * M2_D_STATE
    if cell < dc_cells:
        var bb = cell // (t_work * M2_D_STATE)
        var rem = cell - bb * t_work * M2_D_STATE
        var tt = rem // M2_D_STATE
        var nn = rem - tt * M2_D_STATE
        var cc = tt // qv
        var ii = tt - cc * qv
        var acc = Float32(0.0)
        for hh in range(nh):
            var scale = m2_exp_step(ftz(dacs.unsafe_load(
                ((bb * nh + hh) * nc + cc) * qv + ii
            )))
            for pp in range(M2_HEADDIM):
                var dy = ftz(d_yoff.unsafe_load(
                    ((bb * t_work + tt) * nh + hh) * M2_HEADDIM + pp
                ))
                var d_dot = m2_product(dy, scale)
                var pv = ftz(pass_states.unsafe_load(
                    (((bb * nc + cc) * nh + hh) * pn)
                    + pp * M2_D_STATE + nn
                ))
                acc = m2_fma_step(pv, d_dot, acc)
        d_c.unsafe_store(cell, acc)
    var da_cell = cell
    var da_cells = b * nh * nc * qv
    if da_cell < da_cells:
        var bb = da_cell // (nh * nc * qv)
        var rem = da_cell - bb * nh * nc * qv
        var hh = rem // (nc * qv)
        rem -= hh * nc * qv
        var cc = rem // qv
        var ii = rem - cc * qv
        var tt = cc * qv + ii
        var total = Float32(0.0)
        if tt < t_work:
            var scale = m2_exp_step(ftz(dacs.unsafe_load(da_cell)))
            for pp in range(M2_HEADDIM):
                var dot = Float32(0.0)
                for nn in range(M2_D_STATE):
                    dot = m2_fma_step(
                        ftz(xbc.unsafe_load(
                            (bb * t_work + tt) * cd + di + M2_D_STATE + nn
                        )),
                        ftz(pass_states.unsafe_load(
                            (((bb * nc + cc) * nh + hh) * pn)
                            + pp * M2_D_STATE + nn
                        )), dot)
                var dy = ftz(d_yoff.unsafe_load(
                    ((bb * t_work + tt) * nh + hh) * M2_HEADDIM + pp
                ))
                total = m2_fma_step(dy, dot, total)
            total = m2_product(total, scale)
        d_dacs.unsafe_store(da_cell, total)


@always_inline
def mamba2_s18_direct_dpass_cell(
    cell: Int,
    direct_d_pass: MutPointer[Float32, MutAnyOrigin],
    d_yoff: MutPointer[Float32, MutAnyOrigin],  # [B,T,H,P]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B,T,CD]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B,H,C,Q]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_s18_direct_dpass_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = M2_HEADDIM * M2_D_STATE
    if cell >= b * nc * nh * pn:
        return
    var bb = cell // (nc * nh * pn)
    var rem = cell - bb * nc * nh * pn
    var cc = rem // (nh * pn)
    rem -= cc * nh * pn
    var hh = rem // pn
    rem -= hh * pn
    var pp = rem // M2_D_STATE
    var nn = rem - pp * M2_D_STATE
    var leaf = qv
    if leaf > 128:
        leaf = 128
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    for ii in range(qv):
        var tt = cc * qv + ii
        if tt < t_work:
            var dy_idx = ((bb * t_work + tt) * nh + hh) * M2_HEADDIM + pp
            var dy = ftz(d_yoff.unsafe_load(dy_idx))
            var scale = m2_exp_step(ftz(
                dacs.unsafe_load(((bb * nh + hh) * nc + cc) * qv + ii)
            ))
            var d_dot = m2_product(dy, scale)
            var cv = ftz(xbc.unsafe_load((bb * t_work + tt) * cd + di + M2_D_STATE + nn))
            if ii < leaf:
                acc0 = m2_fma_step(cv, d_dot, acc0)
            else:
                acc1 = m2_fma_step(cv, d_dot, acc1)
    var result = acc0
    if qv > leaf:
        result = ftz(acc0 + acc1)
    direct_d_pass.unsafe_store(cell, result)


@always_inline
def mamba2_cstate_ddecay_cell(
    cell: Int,
    d_decay: MutPointer[Float32, MutAnyOrigin],
    d_cstate: MutPointer[Float32, MutAnyOrigin],
    xd: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, di_in: Int32,
    cd_in: Int32, nc_in: Int32, q_in: Int32,
):
    """`autograd/kernels/mamba2_ssd.mojo::mamba2_cstate_ddecay_kernel`'s thread body (one body per operation)."""
    var b = Int(b_in)
    var t = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    if cell >= b * nh * nc * qv:
        return
    var bb = cell // (nh * nc * qv)
    var rem = cell - bb * nh * nc * qv
    var hh = rem // (nc * qv)
    rem -= hh * nc * qv
    var cc = rem // qv
    var ii = rem - cc * qv
    var tt = cc * qv + ii
    var acc = Float32(0.0)
    if tt < t:
        for pp in range(M2_HEADDIM):
            var xv = ftz(xd.unsafe_load((bb * t * nh + tt * nh + hh) * M2_HEADDIM + pp))
            for nn in range(M2_D_STATE):
                var upstream = ftz(d_cstate.unsafe_load(
                    (((bb * nc + cc) * nh + hh) * M2_HEADDIM + pp)
                    * M2_D_STATE + nn
                ))
                var bv = ftz(xbc.unsafe_load((bb * t + tt) * cd + di + nn))
                acc = m2_fma_step(upstream, m2_product(xv, bv), acc)
    d_decay.unsafe_store(cell, acc)


@always_inline
def mamba2_d_skip_backward_cell(
    th: Int,
    d_scan: MutPointer[Float32, MutAnyOrigin],
    d_x_from_d: MutPointer[Float32, MutAnyOrigin],
    d_d_product: MutPointer[Float32, MutAnyOrigin],
    d_skip: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    d_weight: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    width_in: Int32,
    conv_width_in: Int32,
    head_dim_in: Int32,
):
    """`autograd/kernels/mamba2.mojo::mamba2_d_skip_backward_kernel`'s thread body (one body per operation)."""
    var cells = Int(cells_in)
    var width = Int(width_in)
    var conv_width = Int(conv_width_in)
    var p_dim = Int(head_dim_in)
    if th >= cells:
        return
    var t = th // (width // p_dim)
    var h = th % (width // p_dim)
    var acc = Float32(0.0)
    for p in range(p_dim):
        var cell = t * width + h * p_dim + p
        var ds = ftz(d_skip.unsafe_load(cell))
        var xv = ftz(xbc.unsafe_load(t * conv_width + h * p_dim + p))
        d_scan.unsafe_store(cell, ds)
        d_x_from_d.unsafe_store(
            cell, m2_product(ds, ftz(d_weight.unsafe_load(h)))
        )
        acc = m2_fma_step(ds, xv, acc)
    d_d_product.unsafe_store(th, acc)
