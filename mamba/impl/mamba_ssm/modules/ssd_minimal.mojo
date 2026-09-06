# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mamba_ssm/modules/ssd_minimal.py::segsum` (:23-32) and
`::ssd_minimal_discrete` (:34-78), state-spaces/mamba `e9594ce`, composed
discretize-first as its own test composes it (:94-103) -- ON THE DEVICE,
under profile `mojolearn.identical.mamba2.fp32.v1`
(`mamba/IDENTICAL_MAMBA2_CONTRACT.md`, commit e3b46e95). **COPY, DO NOT
IMPROVE.** Seams S10-S19 plus the `ssd.h_last` report stage; the block
around it is `mamba/impl/mamba_ssm/modules/mamba2.mojo`.

The host oracle `mamba/checks/mamba2_oracle.mojo` is the ANSWER, bit for
bit; this file is an independent transcription of the same order into
kernels, sharing only the seam functions (`ftz`, `identical_mul_add`,
`identical_exp`) -- the Mamba-1 lane's rule, unchanged.

WHY NO FLOAT CROSSES A THREAD BOUNDARY IN THIS FILE. Every kernel owns its
output cell entirely: the cumsum and the segsum columns are one thread per
serial chain; every contraction cell (S12's G, S14's Y_diag, S16's
chunk_states, S18's C.h) is one thread computing its gemm-v1 leaves and
fold IN REGISTERS (the `gemm_identical.mojo` flat-kernel argument); the
inter-chunk pass is one thread per (b, h, p, n) walking chunks serially --
the shipped kernel's own shape (`_state_passing_fwd_kernel`:72-84). No
shared memory, no warp primitive, no atomic, no cross-block reduction.
Clauses (b) and (c) of contract section 8 are properties of the SHAPE of
these kernels.

THE DEVIATIONS IMPLEMENTED HERE (contract section 9; cited, never
renumbered): 782 (pinned serial segsum, STRUCTURAL zeros -- the `-inf`
mask never exists in a buffer, and above-diagonal M entries enter S14's
fold as exact `+0.0` products, spelled identically to the oracle's
zero-filled matrix); 783 (`CHUNK_SIZE = 256` is part of the arithmetic;
`m2_q_eff()` below is the ONE place the sabotage arm may rebuild it); 784
(every intra-chunk contraction is a gemm.fp32.v1 cell: k = 128 one serial
ascending leaf, k = Q = 256 two leaves + one balanced-fold level -- the
leaf/fold spelling is `gemm_oracle.mojo`'s `oracle_leaf_partial` /
`fold_balanced_tree` at P = 2, written in registers); 785 (serial
inter-chunk pass, `h = ftz(fma(scale, h, cstate))`, mamba1 S9's fused
shape); 789 (discretize-first: dt binds to x and to A before ANY chunk
contraction, never to B); 790 (the working-sequence stage shapes -- see
the oracle's docstring; this file computes over `t_work` rows and the
block slices the card).

Padding (contract section 3): every intra-chunk structure is ALWAYS length
Q; a padded row's x/B/C/dt are exact `+0.0` in the working buffers (the
block zero-fills them), padded dacs positions COPY the last real value
(section 7), and a padded position contributes an exactly-zero product to
every fold.

`[[mojo-buffer-freed-at-last-use]]`: every buffer handed to a kernel here
is a FIELD of the caller's stages/state structs; no launcher allocates
scratch and returns without waiting.

THE SABOTAGE ARMS THIS FILE OWNS (contract 8f; each a compile-time
alternative spelling of ONE clause, OFF unless `-D`-named; the check
INVERTS its verdict when one is armed):

    MOJOLEARN_MAMBA2_SABOTAGE_SEGSUM_DESCENDING   S11's folds reversed
    MOJOLEARN_MAMBA2_SABOTAGE_CHUNK_SIZE_128      Q rebuilt at 128 (783's falsifier)
    MOJOLEARN_MAMBA2_SABOTAGE_STATEPASS_MATRIX    ssd_minimal:64-69's decay-matrix einsum
    MOJOLEARN_MAMBA2_SABOTAGE_STATEPASS_UNFUSED   mul then add at S17
    MOJOLEARN_MAMBA2_SABOTAGE_PAIR_DT_B           the fused chain's dt-to-B pairing (789's falsifier)
    MOJOLEARN_MAMBA2_SABOTAGE_FOLD_SERIAL_ZERO_SEED  serial chain replaces the k=256 balanced fold

The block file owns the other five (S6_BIAS_LAST, S6_TAPS_REVERSED,
CLAMP_BEFORE_SOFTPLUS, GATE_NORM_BEFORE, STEP_UPSTREAM_RECURRENCE).

EVERYTHING HERE IS RUN OWED. No kernel below has compiled or run; the
commands live in `mamba/checks/mamba2_check.mojo`'s header.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_exp, identical_mul_add
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_STATE,
    M2_HEADDIM,
)


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720's construction (the oracle's `pinned_mul`), spelled
    here so this file shares only `checks/numerics.mojo` with the host
    side."""
    return identical_mul_add(a, b, Float32(-0.0))


# ===========================================================================
# Sabotage arms (see header).
# ===========================================================================

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

comptime SSD_ANY_SABOTAGE = (
    SAB_SEGSUM_DESCENDING
    or SAB_CHUNK_SIZE_128
    or SAB_STATEPASS_MATRIX
    or SAB_STATEPASS_UNFUSED
    or SAB_PAIR_DT_B
    or SAB_FOLD_SERIAL_ZERO_SEED
)


def ssd_sabotage_name() -> String:
    comptime if SAB_SEGSUM_DESCENDING:
        return String("SEGSUM_DESCENDING")
    comptime if SAB_CHUNK_SIZE_128:
        return String("CHUNK_SIZE_128")
    comptime if SAB_STATEPASS_MATRIX:
        return String("STATEPASS_MATRIX")
    comptime if SAB_STATEPASS_UNFUSED:
        return String("STATEPASS_UNFUSED")
    comptime if SAB_PAIR_DT_B:
        return String("PAIR_DT_B")
    comptime if SAB_FOLD_SERIAL_ZERO_SEED:
        return String("FOLD_SERIAL_ZERO_SEED")
    return String("none")


def m2_q_eff() -> Int:
    """THE chunk size the build runs. DEVIATION 783: Q is a profile
    constant with `K_LEAF_MIN`'s standing -- every reduction boundary in
    S11-S18 moves with it, so this function is the ONE site the
    CHUNK_SIZE_128 arm may touch, and everything downstream (chunk counts,
    stage shapes, fold boundaries) derives from its answer."""
    comptime if SAB_CHUNK_SIZE_128:
        return 128
    return M2_CHUNK_SIZE


def m2_n_chunks(t_work: Int) -> Int:
    var qv = m2_q_eff()
    var nc = (t_work + qv - 1) // qv
    if nc < 1:
        nc = 1
    return nc


# ===========================================================================
# Launch geometry: an EXECUTION-plan quantity (DEVIATION 784's last
# sentence). No kernel reads block geometry in any expression that reaches
# a fold boundary or a seed.
# ===========================================================================

comptime MAMBA2_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + MAMBA2_TPB - 1) // MAMBA2_TPB
    if g < 1:
        return 1
    return g


# ===========================================================================
# S10 -- DISCRETIZE FIRST (DEVIATION 789): X_d = x * dt, dA = dt * A.
# One thread per (b, t, h), the P products in its own loop.
# xbc_work [B, T, CD] carries x | B | C; padded rows are exact +0.0.
# ===========================================================================


def m2_discretize_kernel(
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    da: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    dt: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    a_out: MutPointer[Float32, MutAnyOrigin],  # [H]
    bt_in: Int32,  # B * T
    nh_in: Int32,
    cd_in: Int32,
):
    var bt = Int(bt_in)
    var nh = Int(nh_in)
    var cd = Int(cd_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= bt * nh:
        return
    var t = cell // nh
    var hh = cell - t * nh
    var dtv = ftz(dt.unsafe_load(t * nh + hh))
    # dA = dt * A: dt binds to A (789). PAIR_DT_B does not move this
    # product -- the fused chain agrees about dA.
    da.unsafe_store(
        cell, ftz(pinned_mul(dtv, ftz(a_out.unsafe_load(hh))))
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
                cell * M2_HEADDIM + p, ftz(pinned_mul(xv, dtv))
            )


# ===========================================================================
# S11 -- the per-chunk cumsum, SERIAL ASCENDING; the chunk boundary is a
# hard reset; padded positions COPY the last real value (section 7).
# One thread per (b, h, c).
# ===========================================================================


def m2_chunk_cumsum_kernel(
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    da: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
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


# ===========================================================================
# S12 -- L = exp(segsum), STRUCTURAL zeros above the diagonal (DEVIATION
# 782). One thread per (b, c, h, j): the column-serial rebuild
# `seg[i][j] = ftz(seg[i-1][j] + dA[i])`, `seg[j][j] = +0.0`.
# ===========================================================================


def m2_seg_l_kernel(
    seg_l: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    da: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
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
    seg_l.unsafe_store(lbase + j * qv + j, ftz(identical_exp(acc)))
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
        seg_l.unsafe_store(lbase + i * qv + j, ftz(identical_exp(acc)))


# ===========================================================================
# S12 -- G = C.B, a gemm.fp32.v1 cell over n (k = 128, ONE serial
# ascending leaf; DEVIATION 784). One thread per (b, c, i, j), G = 1.
# ===========================================================================


def m2_cb_g_kernel(
    cb_g: MutPointer[Float32, MutAnyOrigin],  # [B, C, Q, Q]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    b_in: Int32,
    t_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    var b = Int(b_in)
    var t_work = Int(t_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
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
    if i < real and j < real:
        var ti = bb * t_work + c0 + i
        var tj = bb * t_work + c0 + j
        for n in range(M2_D_STATE):
            acc = ftz(
                identical_mul_add(
                    ftz(xbc.unsafe_load(ti * cd + di + M2_D_STATE + n)),  # C
                    ftz(xbc.unsafe_load(tj * cd + di + n)),  # B
                    acc,
                )
            )
        acc = ftz(acc)
    else:
        # A padded row's B or C is exact +0.0 everywhere: the leaf is a
        # fold of exact zeros from a +0.0 seed, which is +0.0 -- written
        # directly, same bits as the fold (gemm section 9's argument).
        acc = Float32(0.0)
    cb_g.unsafe_store(((bb * nc + c) * qv + i) * qv + j, acc)


# ===========================================================================
# S13 + S14 -- M = G ⊙ L (PRODUCT, structural +0.0 above the diagonal) and
# Y_diag = M . X_d: a gemm v1 cell at k = Q (two leaves of 128, one
# balanced-fold level at Q = 256; ONE leaf at the sabotaged Q = 128).
# One thread per (b, c, h, i, p); leaves and fold in registers.
# ===========================================================================


def m2_ydiag_kernel(
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
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
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
            m_ij = ftz(
                pinned_mul(
                    ftz(cb_g.unsafe_load(gbase + jj)),
                    ftz(seg_l.unsafe_load(lbase + jj)),
                )
            )
            comptime if SAB_PAIR_DT_B:
                # SABOTAGE: the fused chain scales the CB block by dt[j]
                # (dt bound to the B side) since X_d carries no dt.
                m_ij = ftz(
                    pinned_mul(
                        m_ij,
                        ftz(
                            dt.unsafe_load(
                                ((bb * t_work) + (c * qv + jj)) * nh + hh
                            )
                        ),
                    )
                )
        else:
            # STRUCTURAL zero above the diagonal (DEVIATION 782): +0.0,
            # never a computed product -- the oracle's zero-filled M.
            m_ij = Float32(0.0)
        var xv = ftz(
            xd.unsafe_load(
                (((bb * t_work) + (c * qv + jj)) * nh + hh) * M2_HEADDIM + p
            )
        )
        comptime if SAB_FOLD_SERIAL_ZERO_SEED:
            # SABOTAGE: one serial chain across the whole k = Q, the fold
            # levels erased (the gemm lane's F5 alternative).
            leaf0 = ftz(identical_mul_add(m_ij, xv, leaf0))
        else:
            if jj < leaf_len:
                leaf0 = ftz(identical_mul_add(m_ij, xv, leaf0))
            else:
                leaf1 = ftz(identical_mul_add(m_ij, xv, leaf1))
    var out: Float32
    comptime if SAB_FOLD_SERIAL_ZERO_SEED:
        out = ftz(leaf0)
    else:
        if leaf_len < qv:
            # fold_balanced_tree at P = 2: one arithmetic node.
            out = ftz(ftz(leaf0) + ftz(leaf1))
        else:
            # ONE leaf (k <= 128): no fold node exists.
            out = ftz(leaf0)
    ydiag.unsafe_store(cell, out)


# ===========================================================================
# S15 -- decay = exp(dA_cs_last - dA_cs), one subtraction then the exp.
# One thread per (b, h, c, i).
# ===========================================================================


def m2_decay_kernel(
    decay: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    n_in: Int32,  # B * H * C * Q
    q_in: Int32,
):
    var n = Int(n_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n:
        return
    var row = cell // qv  # (b, h, c)
    var last = ftz(dacs.unsafe_load(row * qv + (qv - 1)))
    var d = ftz(last - ftz(dacs.unsafe_load(cell)))
    decay.unsafe_store(cell, ftz(identical_exp(d)))


# ===========================================================================
# S15 (the product) + S16 -- B_decay = B ⊙ decay, then
# chunk_states = B_decay^T . X_d: gemm v1 cell at k = Q. One thread per
# (b, c, h, p, n).
# ===========================================================================


def m2_cstate_kernel(
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
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    comptime n_state = M2_D_STATE
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
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
            bd = ftz(
                pinned_mul(ftz(xbc.unsafe_load(ti * cd + di + n)), dec)
            )
            comptime if SAB_PAIR_DT_B:
                bd = ftz(
                    pinned_mul(bd, ftz(dt.unsafe_load(ti * nh + hh)))
                )
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
            leaf0 = ftz(identical_mul_add(xv, bd, leaf0))
        else:
            if i < leaf_len:
                leaf0 = ftz(identical_mul_add(xv, bd, leaf0))
            else:
                leaf1 = ftz(identical_mul_add(xv, bd, leaf1))
    var out: Float32
    comptime if SAB_FOLD_SERIAL_ZERO_SEED:
        out = ftz(leaf0)
    else:
        if leaf_len < qv:
            out = ftz(ftz(leaf0) + ftz(leaf1))
        else:
            out = ftz(leaf0)
    cstate.unsafe_store(cell, out)


# ===========================================================================
# S17 -- the SERIAL inter-chunk pass (DEVIATION 785): one thread per
# (b, h, p, n) walking chunks ascending, `h = ftz(fma(scale, h, cstate))`,
# `scale = ftz(exp(ftz(dA_cs_last)))`. Writes `pass.states` (the state
# ENTERING each chunk), `h_last` (after the FINAL padded chunk), and the
# resumption boundary (after the last COMPLETED chunk) back into h_state.
# ===========================================================================


def m2_statepass_kernel(
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
    var b = Int(b_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var n_completed = Int(n_completed_in)
    comptime n_state = M2_D_STATE
    var pn = M2_HEADDIM * n_state
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
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
                var dec = ftz(identical_exp(s))
                var sv: Float32
                if cc == 0:
                    sv = ftz(h_state.unsafe_load(cell))
                else:
                    sv = ftz(
                        cstate.unsafe_load(
                            (((bb * nc + (cc - 1)) * nh + hh) * pn) + i
                        )
                    )
                acc = ftz(identical_mul_add(dec, sv, acc))
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
        var scale = ftz(
            identical_exp(
                ftz(
                    dacs.unsafe_load(
                        ((bb * nh + hh) * nc + c) * qv + (qv - 1)
                    )
                )
            )
        )
        var cs = ftz(
            cstate.unsafe_load((((bb * nc + c) * nh + hh) * pn) + i)
        )
        comptime if SAB_STATEPASS_UNFUSED:
            # SABOTAGE: two roundings where the profile has one.
            var prod = ftz(pinned_mul(scale, h))
            h = ftz(prod + cs)
        else:
            h = ftz(identical_mul_add(scale, h, cs))
        if c == n_completed - 1:
            boundary = h
    h_last.unsafe_store(cell, h)
    if n_completed > 0:
        h_state.unsafe_store(cell, boundary)


# ===========================================================================
# S18 + S19 -- Y_off = (C . h_prev) ⊙ exp(dA_cs) (contract over n FIRST,
# k = 128 one leaf; scale AFTER), then Y = Y_diag + Y_off. One thread per
# (b, t, h, p), real rows only by construction.
# ===========================================================================


def m2_yoff_y_kernel(
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
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    comptime n_state = M2_D_STATE
    var pn = M2_HEADDIM * n_state
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
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
        acc = ftz(
            identical_mul_add(
                ftz(xbc.unsafe_load(ti * cd + di + n_state + n)),
                ftz(
                    pass_states.unsafe_load(
                        (((bb * nc + c) * nh + hh) * pn) + p * n_state + n
                    )
                ),
                acc,
            )
        )
    acc = ftz(acc)
    var sc = ftz(
        identical_exp(
            ftz(dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + i))
        )
    )
    var yo = ftz(pinned_mul(acc, sc))
    yoff.unsafe_store(cell, yo)
    y_out.unsafe_store(cell, ftz(ftz(ydiag.unsafe_load(cell)) + yo))


# ===========================================================================
# THE LAUNCHER: seams S10-S19 + h_last over the working sequence. All
# buffers are the CALLER's fields (the block's stages/state structs); this
# function only launches and synchronizes. Stage recording is the block's
# (one place, one prefix, contract section 7's order).
# ===========================================================================


def ssd_forward(
    ctx: DeviceContext,
    mut xd: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut dacs: DeviceBuffer[DType.float32],
    mut seg_l: DeviceBuffer[DType.float32],
    mut cb_g: DeviceBuffer[DType.float32],
    mut ydiag: DeviceBuffer[DType.float32],
    mut decay: DeviceBuffer[DType.float32],
    mut cstate: DeviceBuffer[DType.float32],
    mut pass_states: DeviceBuffer[DType.float32],
    mut yoff: DeviceBuffer[DType.float32],
    mut y_out: DeviceBuffer[DType.float32],
    mut h_last: DeviceBuffer[DType.float32],
    mut xbc_work: DeviceBuffer[DType.float32],
    mut dt_work: DeviceBuffer[DType.float32],
    mut a_out: DeviceBuffer[DType.float32],
    mut h_state: DeviceBuffer[DType.float32],
    b: Int,
    t_work: Int,
    di: Int,
    cd: Int,
    nh: Int,
) raises:
    """One SSD pass. `h_state` enters as the boundary/initial state and
    leaves as the boundary after the last COMPLETED chunk (unchanged when
    no chunk completed); `h_last` is the report stage. SYNCHRONIZES before
    returning."""
    var qv = m2_q_eff()
    var nc = m2_n_chunks(t_work)
    var n_completed = t_work // qv

    ctx.enqueue_function[m2_discretize_kernel](
        xd.unsafe_ptr(),
        da.unsafe_ptr(),
        xbc_work.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        a_out.unsafe_ptr(),
        Int32(b * t_work),
        Int32(nh),
        Int32(cd),
        grid_dim=(_grid(b * t_work * nh), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_chunk_cumsum_kernel](
        dacs.unsafe_ptr(),
        da.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nh * nc), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_seg_l_kernel](
        seg_l.unsafe_ptr(),
        da.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nc * nh * qv), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_cb_g_kernel](
        cb_g.unsafe_ptr(),
        xbc_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(di),
        Int32(cd),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nc * qv * qv), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_ydiag_kernel](
        ydiag.unsafe_ptr(),
        cb_g.unsafe_ptr(),
        seg_l.unsafe_ptr(),
        xd.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * t_work * nh * M2_HEADDIM), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_decay_kernel](
        decay.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b * nh * nc * qv),
        Int32(qv),
        grid_dim=(_grid(b * nh * nc * qv), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_cstate_kernel](
        cstate.unsafe_ptr(),
        xbc_work.unsafe_ptr(),
        decay.unsafe_ptr(),
        xd.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(di),
        Int32(cd),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nc * nh * M2_HEADDIM * M2_D_STATE), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_statepass_kernel](
        pass_states.unsafe_ptr(),
        h_last.unsafe_ptr(),
        h_state.unsafe_ptr(),
        cstate.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        Int32(n_completed),
        grid_dim=(_grid(b * nh * M2_HEADDIM * M2_D_STATE), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_yoff_y_kernel](
        yoff.unsafe_ptr(),
        y_out.unsafe_ptr(),
        ydiag.unsafe_ptr(),
        xbc_work.unsafe_ptr(),
        pass_states.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(di),
        Int32(cd),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * t_work * nh * M2_HEADDIM), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.synchronize()
