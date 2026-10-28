# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mamba_ssm/modules/ssd_minimal.py::segsum` (:23-32) and
`::ssd_minimal_discrete` (:34-78), state-spaces/mamba `e9594ce`, composed
discretize-first as its own test composes it (:94-103) -- ON THE DEVICE,
under profile `mojolearn.identical.mamba2.fp32.v1`
(`mamba/IDENTICAL_MAMBA2_CONTRACT.md`, commit e3b46e95). Arithmetic order is fixed
by that contract; a change needs a DEVIATION. Seams S10-S19 plus the `ssd.h_last` report stage; the block
around it is `mamba/impl/modules/mamba2.mojo`.

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
from std.sys.compile import is_defined, get_defined_int
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_exp, identical_mul_add, identical_mul
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
#: lane afn-mamba (2026-10-03): FAST + Apple + `-D MOJOLEARN_AFN_MAMBA2_SSD_MMA`
#: runs S12, S13/S14 and S15/S16 on simdgroup 8x8 tiles (afn_ssd_mma.mojo);
#: every other build takes the three cell kernels below unchanged.
from mamba.impl.modules.afn_defines import AFN_MAMBA2_SSD_MMA, IDN_M2_SSD_TILES
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from mamba.impl.modules.afn_ssd_mma import (
    afn_m2_cb_g_mma,
    afn_m2_cstate_mma,
    afn_m2_ydiag_mma,
    afn_ssd_mma_applies,
)
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_STATE,
    M2_HEADDIM,
)


# ===========================================================================
# Sabotage arms (see header).
# ===========================================================================

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

#: lane nr-mamba (2026-10-04, roadmap B9) IDN_M2_CB_LOWER (IDENTICAL, default
#: ON; `-D MOJOLEARN_IDN_M2_CB_LOWER_OFF` or `-D MOJOLEARN_IDN_ALL_OFF`
#: restores main): S12's G = C.B folds only j <= i and writes an explicit
#: +0.0 above the diagonal (never leaves the cell unwritten: cb.G is a
#: recorded, arena-allocated stage). Every reader (S13's M = G o L, the
#: backward's ydiag and cb gradients) reads j <= i only, so no output bit
#: moves; the RECORDED cb.G stage changes above the diagonal, on every column
#: together with the oracle (mamba/checks/mamba2_oracle.mojo, same define).
comptime IDN_M2_CB_LOWER = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # I08 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not is_defined["MOJOLEARN_IDN_M2_CB_LOWER_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
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
        cell, ftz(identical_mul(dtv, ftz(a_out.unsafe_load(hh))))
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
                cell * M2_HEADDIM + p, ftz(identical_mul(xv, dtv))
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


# NN37 schedules only the inherited lower-triangle arm's live cells.
# cb_g is explicitly zeroed first because traces/backward may read upper
# cells. The zero-fill launch and full memory traffic belong in its A/B.
# OFF, uncompiled/unmeasured; baseline lower-arm disabled means unavailable.
comptime NN37_TRIANGLE_TASKS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and IDN_M2_CB_LOWER and is_defined["MOJOLEARN_NN37_TRIANGLE_TASKS"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()


def nn37_m2_cb_triangle_kernel(
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
    var triangle = qv * (qv + 1) // 2
    var task = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if task >= b * nc * triangle:
        return
    var group = task // triangle
    var packed = task % triangle
    var bb = group // nc
    var c = group % nc
    # Integer upper-bound search for triangular row starts; never use a
    # vendor sqrt or precision-dependent inverse-triangle calculation.
    var low = 0
    var high = qv
    while low + 1 < high:
        var mid = (low + high) // 2
        if mid * (mid + 1) // 2 <= packed:
            low = mid
        else:
            high = mid
    var i = low
    var j = packed - i * (i + 1) // 2
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
    var live = i < real and j < real
    comptime if IDN_M2_CB_LOWER:
        # B9: above the diagonal is never read; explicit +0.0 (below).
        live = live and j <= i
    if live:
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
                identical_mul(
                    ftz(cb_g.unsafe_load(gbase + jj)),
                    ftz(seg_l.unsafe_load(lbase + jj)),
                )
            )
            comptime if SAB_PAIR_DT_B:
                # SABOTAGE: the fused chain scales the CB block by dt[j]
                # (dt bound to the B side) since X_d carries no dt.
                m_ij = ftz(
                    identical_mul(
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
                identical_mul(ftz(xbc.unsafe_load(ti * cd + di + n)), dec)
            )
            comptime if SAB_PAIR_DT_B:
                bd = ftz(
                    identical_mul(bd, ftz(dt.unsafe_load(ti * nh + hh)))
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
# lane nr-mamba (2026-10-04, roadmap B1): the S13/S14 and S16 cells as
# threadgroup tiles (IDN_M2_SSD_TILES in afn_defines.mojo; device columns
# only, the host column keeps the cell kernels above). Same chains, same
# leaves, same order, so the same bits. Every sabotage arm takes the cell
# kernels.
# ===========================================================================

comptime M2_SSD_TILED = (
    IDN_M2_SSD_TILES
    and not SAB_PAIR_DT_B
    and not SAB_FOLD_SERIAL_ZERO_SEED
    and not SAB_2712_UNBOUNDED
)
#: Y_diag tile: YD_ROWS rows i of one (b, c, h), all 64 p; X_d staged
#: YD_JC rows at a time. Page: (4 * 256 + 64 * 64) * 4 = 20,480 bytes.
# NN33 independent tile-geometry arms on the inherited I08 kernels.
# They do not change the 128-term leaves, zero terms or fold. Rows8 trades
# 512 lanes for reuse of each X tile across twice as many output rows;
# JC32 keeps its shared page at 16 KiB. CS16 reuses each staged B/decay
# tile across 16 value channels (18 KiB). Both fit inside the existing
# conservative 20 KiB resource guard. No dataset/shape-specific dispatch.
# All newly introduced switches are OFF, uncompiled and unmeasured.
# L11 (2026-10-07): the NN33 booleans became integer tile sweeps, one per
# kernel: -D MOJOLEARN_IDN_M2_YD_ROWS=4|8 (default 4; 8 also halves YD_JC to
# 32 to keep the page) and -D MOJOLEARN_IDN_M2_CS_PT=8|16 (default 8). The
# legal sets are asserted in core/six_lane_experiment_guards.mojo. Tiles
# move no bit; they may be tuned per vendor.
comptime _M2_TILE_SWEEP = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
# PROMOTED 2026-10-11 (grid g50ebe26a5, single-switch arm m2_yd_rows=8 vs today's defaults, one run per arm): mamba2-forward
# gaussian NV L40S 13.079 -> 5.688 ms (v1223), AMD MI325X 4.551 -> 4.388 ms (a1344); avg ratio 0.648; hash 0350b90b on both
# vendors. Default is now 8; -D MOJOLEARN_IDN_M2_YD_ROWS=4 restores the old tile.
comptime NN33_YDIAG_ROWS8 = _M2_TILE_SWEEP and get_defined_int["MOJOLEARN_IDN_M2_YD_ROWS", 8]() == 8
comptime NN33_CSTATE_P16 = _M2_TILE_SWEEP and get_defined_int["MOJOLEARN_IDN_M2_CS_PT", 8]() == 16
comptime M2_YD_ROWS = 8 if NN33_YDIAG_ROWS8 else 4
comptime M2_YD_JC = 32 if NN33_YDIAG_ROWS8 else 64
comptime M2_YD_QMAX = 256
comptime M2_YD_THREADS = M2_YD_ROWS * M2_HEADDIM
#: C_state tile: one (b, c, h) and CS_PT columns p, all 128 n (one thread
#: per n); B * decay staged CS_IC rows at a time. Page: (32 * 128 + 32 * 8)
#: * 4 = 17,408 bytes.
comptime M2_CS_PT = 16 if NN33_CSTATE_P16 else 8
comptime M2_CS_IC = 32


# ===========================================================================
# lane nv-occupancy (2026-10-28, idea 8 of neural-perf-ideas-20261028):
# MOJOLEARN_IDN_SSD_OCCUPANCY (IDENTICAL, default OFF; device columns only).
# The Oct 27 board read our own mamba2-forward 3x slower on the RTX 6000
# Ada than on the MI325X at the same bits. Three S10-S19 kernels issue
# warp-wide loads whose addresses differ across lanes by a whole row, so
# every load instruction becomes 32 (NVIDIA) or 64 (AMD) separate
# transactions served from L1/L2:
#   S10 discretize: one thread per (b, t, h) walks the 64 channels p, so
#       adjacent lanes read and write 64 words apart;
#   S12 G = C.B:    one thread per (i, j) reads row j of B, adjacent lanes
#       one row (cd words) apart, for all 128 steps of the chain;
#   S18 C.h_prev:   one thread per (t, h, p) reads row p of the entering
#       state, adjacent lanes 128 words apart, and recomputes the same
#       exp(dA_cs) once per channel (64 times per (t, h)).
# Under the define those three run as: one thread per (b, t, h, p) for S10
# (coalesced; the dA product is written by the p = 0 lane); a 16 x 16 cell
# tile for S12 with the 16 C rows and 16 B rows staged once in shared memory
# (row stride 129 so lanes that vary the row hit distinct banks); and a
# row tile for S18 that stages the (b, c, h) entering state [64 x 128] once
# in shared memory (stride 129) and the exp once per row.
# WHY NO BIT MOVES: every output cell keeps its own chain, the same
# operands (ftz applied to the loaded word, at staging instead of at use:
# ftz is a function of the word), the same +0.0 seed, the same ascending
# n order and the same final ftz; the structural zeros above the diagonal
# and on padded rows are written as before; only which thread computes a
# cell and where its operands are read from change. The tile sizes come
# from the warp/wavefront width and the shared-memory budget (a page must
# fit the column's limit, `lib_smem_page_fits_for`; Apple's 32 KiB column
# keeps the per-cell S18 kernel), never from a model or board dimension:
# the kernels take any (B, T, H) and any chunk count, and the row tiles
# are bounds-checked per row. The host column keeps the cell kernels (no
# shared memory; tools/mamba_host_gen.py substitutes the define to False).
# Every sabotage arm keeps the cell kernels. A/B: RUN OWED (nv first).
# ===========================================================================
comptime IDN_SSD_OCCUPANCY = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_SSD_OCCUPANCY"]()
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not SSD_ANY_SABOTAGE
    and not SAB_2712_UNBOUNDED
)
#: S12 tile side: 16 x 16 = 256 threads, 8 NVIDIA warps or 4 AMD
#: wavefronts. Each staged row is read by 16 cells of the tile instead of
#: once per cell. Page: 2 * 16 * 129 * 4 = 16,512 bytes.
comptime M2_CB_T = 16
comptime M2_OCC_LD = M2_D_STATE + 1
comptime M2_CB_PAGE = 2 * M2_CB_T * M2_OCC_LD * 4
comptime M2_OCC_CB = IDN_SSD_OCCUPANCY and lib_smem_page_fits_for[TARGET_COLUMN, M2_CB_PAGE]()


def m2_discretize_cell_kernel(
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    da: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    dt: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    a_out: MutPointer[Float32, MutAnyOrigin],  # [H]
    bt_in: Int32,  # B * T
    nh_in: Int32,
    cd_in: Int32,
):
    """`m2_discretize_kernel`'s cells, one thread per (b, t, h, p): the same
    two products of the same words, coalesced across p. The (b, t, h) dA
    cell is written once, by its p = 0 lane."""
    var bt = Int(bt_in)
    var nh = Int(nh_in)
    var cd = Int(cd_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= bt * nh * M2_HEADDIM:
        return
    var row = cell // M2_HEADDIM  # (b * t, h), m2_discretize_kernel's cell
    var p = cell - row * M2_HEADDIM
    var t = row // nh
    var hh = row - t * nh
    var dtv = ftz(dt.unsafe_load(t * nh + hh))
    if p == 0:
        da.unsafe_store(
            row, ftz(identical_mul(dtv, ftz(a_out.unsafe_load(hh))))
        )
    var xv = ftz(xbc.unsafe_load(t * cd + hh * M2_HEADDIM + p))
    xd.unsafe_store(cell, ftz(identical_mul(xv, dtv)))


def m2_cb_g_tile_kernel(
    cb_g: MutPointer[Float32, MutAnyOrigin],  # [B, C, Q, Q]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    b_in: Int32,
    t_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`m2_cb_g_kernel`'s cells for one M2_CB_T x M2_CB_T tile (i, j) of
    one (b, c): rows i0.. of C and j0.. of B staged once, each cell the
    same k = 128 serial ascending leaf over n."""
    var t_work = Int(t_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var nt = (qv + M2_CB_T - 1) // M2_CB_T
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var tj = blk % nt
    var r = blk // nt
    var ti = r % nt
    r = r // nt
    var c = r % nc
    var bb = r // nc
    var i0 = ti * M2_CB_T
    var j0 = tj * M2_CB_T
    var li = tid // M2_CB_T
    var lj = tid - li * M2_CB_T
    var i = i0 + li
    var j = j0 + lj
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    # Uniform over the block: a tile with no live cell writes the +0.0
    # every one of its cells gets from m2_cb_g_kernel (padded rows, and the
    # B9 zeros above the diagonal) and stages nothing.
    var tile_live = i0 < real and j0 < real
    comptime if IDN_M2_CB_LOWER:
        tile_live = tile_live and j0 <= i0 + M2_CB_T - 1
    if not tile_live:
        if i < qv and j < qv:
            cb_g.unsafe_store(((bb * nc + c) * qv + i) * qv + j, Float32(0.0))
        return
    var c_s = stack_allocation[
        M2_CB_T * M2_OCC_LD, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var b_s = stack_allocation[
        M2_CB_T * M2_OCC_LD, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var e = tid
    while e < M2_CB_T * M2_D_STATE:
        var rr = e // M2_D_STATE
        var n = e - rr * M2_D_STATE
        var cv = Float32(0.0)
        var bv = Float32(0.0)
        if i0 + rr < real:
            cv = ftz(
                xbc.unsafe_load(
                    (bb * t_work + c0 + i0 + rr) * cd + di + M2_D_STATE + n
                )
            )
        if j0 + rr < real:
            bv = ftz(xbc.unsafe_load((bb * t_work + c0 + j0 + rr) * cd + di + n))
        c_s[rr * M2_OCC_LD + n] = cv
        b_s[rr * M2_OCC_LD + n] = bv
        e += M2_CB_T * M2_CB_T
    barrier()
    if i >= qv or j >= qv:
        return
    var acc = Float32(0.0)
    var live = i < real and j < real
    comptime if IDN_M2_CB_LOWER:
        live = live and j <= i
    if live:
        for n in range(M2_D_STATE):
            acc = ftz(
                identical_mul_add(
                    c_s[li * M2_OCC_LD + n],  # C, ftz'd at staging
                    b_s[lj * M2_OCC_LD + n],  # B, ftz'd at staging
                    acc,
                )
            )
        acc = ftz(acc)
    else:
        acc = Float32(0.0)
    cb_g.unsafe_store(((bb * nc + c) * qv + i) * qv + j, acc)


# I08 AMD MI325X NEUTRAL/MIXED (2026-10-06, sourcecbcc8dcd3303):
# retained/base0.995/1.025 for B2,length513/1025,d_model32,d_inner64,heads1.
# The admitted retained G*L path is distinct (393216/655360 cells below cap);
# one same-process warmup/score of prefill forward/backward, identity reused.
# Generated caller scope only; no full-workload promotion. Evidence:
# overnight-ab-20261006/amd/normalized-measurements.json, I08.
comptime M2_RETAIN_GL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # PROMOTED 2026-10-11 (grid g50ebe26a5, arm m2_retain_gl=on vs today's defaults, one run per arm): mamba2-forward gaussian
    # NV L40S 13.079 -> 5.690 ms (v1242), AMD MI325X 4.551 -> 4.554 ms (a1350); avg ratio 0.660; hash 0350b90b on both vendors.
    # Default ON; -D MOJOLEARN_IDN_M2_RETAIN_GL_OFF restores the recompute path.
    and not is_defined["MOJOLEARN_IDN_M2_RETAIN_GL_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not SSD_ANY_SABOTAGE
)
comptime M2_RETAIN_GL_MAX_CELLS = 1<<22

def m2_retained_gl_kernel(retained: MutPointer[Float32,MutAnyOrigin],
    cb_g: MutPointer[Float32,MutAnyOrigin],seg_l: MutPointer[Float32,MutAnyOrigin],
    b_in: Int32,t_in: Int32,nh_in: Int32,nc_in: Int32,q_in: Int32):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var qv = Int(q_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    if cell>=Int(b_in)*nc*nh*qv*qv:
        return
    var j = cell%qv
    var i = (cell//qv)%qv
    var h = (cell//(qv*qv))%nh
    var c = (cell//(qv*qv*nh))%nc
    var bb = cell//(qv*qv*nh*nc)
    var value = Float32(0)
    if c*qv+i<Int(t_in) and j<=i:
        value = ftz(identical_mul(ftz(cb_g[((bb*nc+c)*qv+i)*qv+j]),ftz(seg_l[cell])))
    retained[cell] = value


def m2_ydiag_tile_kernel[RETAIN_M: Bool = False](
    ydiag: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    cb_g: MutPointer[Float32, MutAnyOrigin],  # [B, C, Q, Q]
    seg_l: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`m2_ydiag_kernel`'s cells for M2_YD_ROWS rows of one (b, c, h): the
    rows of M formed once in shared memory, X_d staged once per tile."""
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var n_it = (qv + M2_YD_ROWS - 1) // M2_YD_ROWS
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var it = blk % n_it
    var r = blk // n_it
    var hh = r % nh
    r = r // nh
    var c = r % nc
    var bb = r // nc
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    if it * M2_YD_ROWS >= real:
        return  # uniform over the block: no row of this tile is real
    var m_s = stack_allocation[
        M2_YD_ROWS * M2_YD_QMAX, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var x_s = stack_allocation[
        M2_YD_JC * M2_HEADDIM, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    # The tile's rows of M = G o L: the PRODUCT at j <= i, the structural
    # +0.0 above the diagonal; a row past the sequence is never stored.
    var e = tid
    while e < M2_YD_ROWS * qv:
        var rr = e // qv
        var jj = e - rr * qv
        var ii = it * M2_YD_ROWS + rr
        var mv = Float32(0.0)
        if ii < real and jj <= ii:
            comptime if RETAIN_M:
                mv = cb_g.unsafe_load((((bb*nc+c)*nh+hh)*qv+ii)*qv+jj)
            else:
                mv = ftz(
                    identical_mul(
                        ftz(cb_g.unsafe_load(((bb * nc + c) * qv + ii) * qv + jj)),
                        ftz(
                            seg_l.unsafe_load(
                                (((bb * nc + c) * nh + hh) * qv + ii) * qv + jj
                            )
                        ),
                    )
                )
        m_s[rr * M2_YD_QMAX + jj] = mv
        e += M2_YD_THREADS
    var ri = tid // M2_HEADDIM
    var p = tid - ri * M2_HEADDIM
    var i = it * M2_YD_ROWS + ri
    var active = i < real
    var leaf_len = qv
    if qv > 128:
        leaf_len = qv // 2
    var leaf0 = Float32(0.0)
    var leaf1 = Float32(0.0)
    var j0 = 0
    while j0 < qv:
        barrier()
        var ex = tid
        while ex < M2_YD_JC * M2_HEADDIM:
            var jr = ex // M2_HEADDIM
            var pp = ex - jr * M2_HEADDIM
            var jj = j0 + jr
            var xv = Float32(0.0)
            if jj < qv and c0 + jj < t_work:
                xv = ftz(
                    xd.unsafe_load(
                        (((bb * t_work) + (c0 + jj)) * nh + hh) * M2_HEADDIM + pp
                    )
                )
            x_s[ex] = xv
            ex += M2_YD_THREADS
        barrier()
        if active:
            for jr in range(M2_YD_JC):
                var jj = j0 + jr
                if jj < qv:
                    # every step, the j > i +0.0 ones included (the -0 rule)
                    var m_ij = m_s[ri * M2_YD_QMAX + jj]
                    var xv = x_s[jr * M2_HEADDIM + p]
                    if jj < leaf_len:
                        leaf0 = ftz(identical_mul_add(m_ij, xv, leaf0))
                    else:
                        leaf1 = ftz(identical_mul_add(m_ij, xv, leaf1))
        j0 += M2_YD_JC
    if active:
        var out: Float32
        if leaf_len < qv:
            out = ftz(ftz(leaf0) + ftz(leaf1))
        else:
            out = ftz(leaf0)
        ydiag.unsafe_store(
            (((bb * t_work) + (c0 + i)) * nh + hh) * M2_HEADDIM + p, out
        )


def m2_cstate_tile_kernel(
    cstate: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, P, N]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    decay: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """`m2_cstate_kernel`'s cells for one (b, c, h) and M2_CS_PT columns p:
    B * decay formed once per (i, n) in shared memory, one thread per n
    carrying the M2_CS_PT chains."""
    comptime n_state = M2_D_STATE
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    comptime n_pt = M2_HEADDIM // M2_CS_PT
    var blk = Int(block_idx.x)
    var n = Int(thread_idx.x)
    var pt = blk % n_pt
    var r = blk // n_pt
    var hh = r % nh
    r = r // nh
    var c = r % nc
    var bb = r // nc
    var pbase = pt * M2_CS_PT
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var bd_s = stack_allocation[
        M2_CS_IC * n_state, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var xv_s = stack_allocation[
        M2_CS_IC * M2_CS_PT, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var leaf_len = qv
    if qv > 128:
        leaf_len = qv // 2
    var leaf0 = SIMD[DType.float32, M2_CS_PT](0.0)
    var leaf1 = SIMD[DType.float32, M2_CS_PT](0.0)
    var i0 = 0
    while i0 < qv:
        barrier()
        # B * decay for rows i0 .. i0 + IC, this thread's n (+0.0 padded).
        for ir in range(M2_CS_IC):
            var i = i0 + ir
            var bd = Float32(0.0)
            if i < real:
                var ti = bb * t_work + c0 + i
                var dec = ftz(
                    decay.unsafe_load(((bb * nh + hh) * nc + c) * qv + i)
                )
                bd = ftz(
                    identical_mul(ftz(xbc.unsafe_load(ti * cd + di + n)), dec)
                )
            bd_s[ir * n_state + n] = bd
        var ex = n
        while ex < M2_CS_IC * M2_CS_PT:
            var ir = ex // M2_CS_PT
            var q = ex - ir * M2_CS_PT
            var i = i0 + ir
            var xv = Float32(0.0)
            if i < real:
                var ti = bb * t_work + c0 + i
                xv = ftz(xd.unsafe_load((ti * nh + hh) * M2_HEADDIM + pbase + q))
            xv_s[ex] = xv
            ex += n_state
        barrier()
        for ir in range(M2_CS_IC):
            var i = i0 + ir
            if i < qv:
                var bd = bd_s[ir * n_state + n]
                comptime for q in range(M2_CS_PT):
                    var xv = xv_s[ir * M2_CS_PT + q]
                    if i < leaf_len:
                        leaf0[q] = ftz(identical_mul_add(xv, bd, leaf0[q]))
                    else:
                        leaf1[q] = ftz(identical_mul_add(xv, bd, leaf1[q]))
        i0 += M2_CS_IC
    comptime for q in range(M2_CS_PT):
        var out: Float32
        if leaf_len < qv:
            out = ftz(ftz(leaf0[q]) + ftz(leaf1[q]))
        else:
            out = ftz(leaf0[q])
        cstate.unsafe_store(
            ((((bb * nc + c) * nh + hh) * M2_HEADDIM + pbase + q) * n_state) + n,
            out,
        )


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
            var prod = ftz(identical_mul(scale, h))
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


# NN36: OFF and unmeasured. Channel sharing is based on the existing
# fixed Mamba head layout, not an observed benchmark shape. Host computes
# the same explicitly rounded exp; no changed arithmetic profile.
# L11 (2026-10-07): NN36 and NI39 are two arms of ONE switch,
# -D MOJOLEARN_IDN_M2_YOFF_EXP=0|1|2: 0 recompute exp per cell (default),
# 1 = NN36 block-shared exp (GPU shared memory; host takes arm 0's
# expression), 2 = NI39 global per-position exp cache. Same exp, same bits.
comptime IDN_M2_YOFF_EXP_ARM = get_defined_int["MOJOLEARN_IDN_M2_YOFF_EXP", 0]()
comptime NN36_SHARED_DECAY = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and IDN_M2_YOFF_EXP_ARM == 1 and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()


def nn36_m2_yoff_shared_exp_kernel(
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
    # One block owns a logical token/head; one lane owns each value
    # channel. The same exp word is reused without changing the dot chain.
    var logical = Int(block_idx.x)
    var p = Int(thread_idx.x)
    var hh = logical % nh
    var ti = logical // nh
    var bb = ti // t_work
    var t = ti % t_work
    var c = t // qv
    var i = t % qv
    var shared_scale = stack_allocation[1, Float32, address_space=AddressSpace.SHARED]()
    if p == 0:
        shared_scale[0] = ftz(identical_exp(ftz(dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + i))))
    barrier()
    var cell = logical * M2_HEADDIM + p

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
    var sc = shared_scale[0]
    var yo = ftz(identical_mul(acc, sc))
    yoff.unsafe_store(cell, yo)
    y_out.unsafe_store(cell, ftz(ftz(ydiag.unsafe_load(cell)) + yo))


# NI39: cache exp(dacs) once per logical (batch, head, chunk, position),
# shared across the 64 independent output channels. The exact portable exp
# expression is unchanged; the host's uncached expression is the S counterpart.
# Default OFF. Allocation and the lifetime drain are part of whole-op A/B;
# no speed, identity, quality or full-workload claim has been measured.
comptime IDN_M2_YOFF_EXP_CACHE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_M2_YOFF_EXP_ARM == 2
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not SSD_ANY_SABOTAGE
)


def m2_yoff_exp_cache_kernel(
    cached: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin], cells_in: Int32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < Int(cells_in):
        cached.unsafe_store(cell, ftz(identical_exp(ftz(dacs.unsafe_load(cell)))))


def m2_yoff_y_kernel[CACHED_EXP: Bool = False](
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
    var sc: Float32
    comptime if CACHED_EXP:
        # In this specialization dacs holds exp(dacs), not the log prefix.
        sc = ftz(dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + i))
    else:
        sc = ftz(
            identical_exp(
                ftz(dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + i))
            )
        )
    var yo = ftz(identical_mul(acc, sc))
    yoff.unsafe_store(cell, yo)
    y_out.unsafe_store(cell, ftz(ftz(ydiag.unsafe_load(cell)) + yo))


# lane nv-occupancy (2026-10-28): MOJOLEARN_IDN_SSD_OCCUPANCY's S18/S19 row
# tile (see the block above m2_discretize_cell_kernel). One block owns
# M2_YO_ROWS rows of one (b, c, h): the entering state [P, N] staged once
# (stride 129: lanes vary p, so the 32 lanes of a warp read 32 banks), the
# C rows and exp(dA_cs) staged once per M2_YO_RL rows; each thread owns one
# (row, p) cell and runs the same k = 128 leaf, the same product and the
# same Y = Y_diag + Y_off add. Rows per block: the 8,192-word state stage is
# amortized over 32 * 64 * 128 chain steps (about 3%), and a Q = 256 chunk
# gives 8 blocks per (b, c, h), so even one (b, c) per head keeps more
# blocks than a 100+ SM/CU device has resident slots. Page:
# (64 * 129 + 4 * 128 + 4) * 4 = 35,088 bytes: fits NVIDIA's 48 KiB and
# AMD's 64 KiB; Apple's 32 KiB column keeps the cell kernel. Taken only on
# the default exp arm (MOJOLEARN_IDN_M2_YOFF_EXP=0).
comptime M2_YO_RL = 4
comptime M2_YO_ROWS = 32
comptime M2_YO_THREADS = M2_YO_RL * M2_HEADDIM
comptime M2_YO_PAGE = (M2_HEADDIM * M2_OCC_LD + M2_YO_RL * M2_D_STATE + M2_YO_RL) * 4
comptime M2_OCC_YOFF = (
    IDN_SSD_OCCUPANCY
    and IDN_M2_YOFF_EXP_ARM == 0
    and lib_smem_page_fits_for[TARGET_COLUMN, M2_YO_PAGE]()
)


def m2_yoff_tile_kernel(
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
    """`m2_yoff_y_kernel[False]`'s cells for M2_YO_ROWS rows of one (b, c, h)."""
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    comptime n_state = M2_D_STATE
    comptime pn = M2_HEADDIM * n_state
    var n_rt = (qv + M2_YO_ROWS - 1) // M2_YO_ROWS
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var rt = blk % n_rt
    var r = blk // n_rt
    var hh = r % nh
    r = r // nh
    var c = r % nc
    var bb = r // nc
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var r0 = rt * M2_YO_ROWS
    if r0 >= real:
        return  # uniform over the block: no row of this tile is real
    var rend = r0 + M2_YO_ROWS
    if rend > real:
        rend = real
    var rl = tid // M2_HEADDIM
    var p = tid - rl * M2_HEADDIM
    var s_s = stack_allocation[
        M2_HEADDIM * M2_OCC_LD, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var c_s = stack_allocation[
        M2_YO_RL * n_state, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var sc_s = stack_allocation[
        M2_YO_RL, Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    # The state ENTERING chunk c for head hh, [P, N] row-major in the stage.
    var sbase = ((bb * nc + c) * nh + hh) * pn
    var e = tid
    while e < pn:
        var pp = e // n_state
        var n = e - pp * n_state
        s_s[pp * M2_OCC_LD + n] = ftz(pass_states.unsafe_load(sbase + e))
        e += M2_YO_THREADS
    var i0 = r0
    while i0 < rend:
        barrier()  # the state stage (first pass) / the previous rows' reads
        var e2 = tid
        while e2 < M2_YO_RL * n_state:
            var rr = e2 // n_state
            var n = e2 - rr * n_state
            var cv = Float32(0.0)
            if i0 + rr < rend:
                cv = ftz(
                    xbc.unsafe_load(
                        (bb * t_work + c0 + i0 + rr) * cd + di + n_state + n
                    )
                )
            c_s[e2] = cv
            e2 += M2_YO_THREADS
        if tid < M2_YO_RL:
            var sc = Float32(0.0)
            if i0 + tid < rend:
                sc = ftz(
                    identical_exp(
                        ftz(
                            dacs.unsafe_load(
                                ((bb * nh + hh) * nc + c) * qv + i0 + tid
                            )
                        )
                    )
                )
            sc_s[tid] = sc
        barrier()
        var i = i0 + rl
        if i < rend:
            # C . h_prev over n: one serial ascending leaf (k = 128).
            var acc = Float32(0.0)
            for n in range(n_state):
                acc = ftz(
                    identical_mul_add(
                        c_s[rl * n_state + n], s_s[p * M2_OCC_LD + n], acc
                    )
                )
            acc = ftz(acc)
            var yo = ftz(identical_mul(acc, sc_s[rl]))
            var cell = ((bb * t_work + c0 + i) * nh + hh) * M2_HEADDIM + p
            yoff.unsafe_store(cell, yo)
            y_out.unsafe_store(cell, ftz(ftz(ydiag.unsafe_load(cell)) + yo))
        i0 += M2_YO_RL


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
    drain: Bool = True,
) raises:
    """One SSD pass. `h_state` enters as the boundary/initial state and
    leaves as the boundary after the last COMPLETED chunk (unchanged when
    no chunk completed); `h_last` is the report stage. `drain=True` retains
    the standalone/traced boundary. The production block passes False: all
    launches share its ordered context queue and the public output copy is
    the completion boundary."""
    var qv = m2_q_eff()
    var nc = m2_n_chunks(t_work)
    var n_completed = t_work // qv
    # Candidate owns temporary retained words until its explicit drain.
    var retained = List[DeviceBuffer[DType.float32]]()

    comptime if IDN_SSD_OCCUPANCY:
        ctx.enqueue_function[m2_discretize_cell_kernel](
            xd.unsafe_ptr(),
            da.unsafe_ptr(),
            xbc_work.unsafe_ptr(),
            dt_work.unsafe_ptr(),
            a_out.unsafe_ptr(),
            Int32(b * t_work),
            Int32(nh),
            Int32(cd),
            grid_dim=(_grid(b * t_work * nh * M2_HEADDIM), 1, 1),
            block_dim=(MAMBA2_TPB, 1, 1),
        )
    else:
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
    var mma = False
    comptime if AFN_MAMBA2_SSD_MMA:
        mma = afn_ssd_mma_applies(qv)
    if mma:
        afn_m2_cb_g_mma(ctx, cb_g, xbc_work, b, t_work, di, cd, nc, qv)
        afn_m2_ydiag_mma(ctx, ydiag, cb_g, seg_l, xd, b, t_work, nh, nc, qv)
    else:
        comptime if NN37_TRIANGLE_TASKS:
            cb_g.enqueue_fill(Float32(0.0))
            ctx.enqueue_function[nn37_m2_cb_triangle_kernel](
                cb_g.unsafe_ptr(),
                xbc_work.unsafe_ptr(),
                Int32(b),
                Int32(t_work),
                Int32(di),
                Int32(cd),
                Int32(nc),
                Int32(qv),
                grid_dim=(_grid(b * nc * (qv * (qv + 1) // 2)), 1, 1),
                block_dim=(MAMBA2_TPB, 1, 1),
            )
        elif M2_OCC_CB:
            var cb_nt = (qv + M2_CB_T - 1) // M2_CB_T
            ctx.enqueue_function[m2_cb_g_tile_kernel](
                cb_g.unsafe_ptr(),
                xbc_work.unsafe_ptr(),
                Int32(b),
                Int32(t_work),
                Int32(di),
                Int32(cd),
                Int32(nc),
                Int32(qv),
                grid_dim=(b * nc * cb_nt * cb_nt, 1, 1),
                block_dim=(M2_CB_T * M2_CB_T, 1, 1),
            )
        else:
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
        comptime if M2_SSD_TILED:
            var n_it = (qv + M2_YD_ROWS - 1) // M2_YD_ROWS
            var use_retained = False
            comptime if M2_RETAIN_GL:
                use_retained = b*nc*nh*qv*qv<=M2_RETAIN_GL_MAX_CELLS
            if use_retained:
                var cells = b*nc*nh*qv*qv
                retained.append(ctx.enqueue_create_buffer[DType.float32](cells))
                ctx.enqueue_function[m2_retained_gl_kernel](retained[0].unsafe_ptr(),cb_g.unsafe_ptr(),seg_l.unsafe_ptr(),Int32(b),Int32(t_work),Int32(nh),Int32(nc),Int32(qv),grid_dim=(_grid(cells),1,1),block_dim=(MAMBA2_TPB,1,1))
                ctx.enqueue_function[m2_ydiag_tile_kernel[True]](ydiag.unsafe_ptr(),retained[0].unsafe_ptr(),seg_l.unsafe_ptr(),xd.unsafe_ptr(),Int32(b),Int32(t_work),Int32(nh),Int32(nc),Int32(qv),grid_dim=(b*nc*nh*n_it,1,1),block_dim=(M2_YD_THREADS,1,1))
            else:
                ctx.enqueue_function[m2_ydiag_tile_kernel[False]](
                    ydiag.unsafe_ptr(),
                    cb_g.unsafe_ptr(),
                    seg_l.unsafe_ptr(),
                    xd.unsafe_ptr(),
                    Int32(b),
                    Int32(t_work),
                    Int32(nh),
                    Int32(nc),
                    Int32(qv),
                    grid_dim=(b * nc * nh * n_it, 1, 1),
                    block_dim=(M2_YD_THREADS, 1, 1),
                )
        else:
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
    if mma:
        afn_m2_cstate_mma(
            ctx, cstate, xbc_work, decay, xd, b, t_work, nh, di, cd, nc, qv
        )
    else:
        comptime if M2_SSD_TILED:
            ctx.enqueue_function[m2_cstate_tile_kernel](
                cstate.unsafe_ptr(),
                xbc_work.unsafe_ptr(),
                decay.unsafe_ptr(),
                xd.unsafe_ptr(),
                Int32(b),
                Int32(t_work),
                Int32(nh),
                Int32(di),
                Int32(cd),
                Int32(nc),
                Int32(qv),
                grid_dim=(b * nc * nh * (M2_HEADDIM // M2_CS_PT), 1, 1),
                block_dim=(M2_D_STATE, 1, 1),
            )
        else:
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
    comptime if NN36_SHARED_DECAY:
        ctx.enqueue_function[nn36_m2_yoff_shared_exp_kernel](
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
            grid_dim=(b * t_work * nh, 1, 1),
            block_dim=(M2_HEADDIM, 1, 1),
        )
    elif IDN_M2_YOFF_EXP_CACHE:
        var cache_index = len(retained)
        var cache_cells = b * nh * nc * qv
        retained.append(ctx.enqueue_create_buffer[DType.float32](cache_cells))
        ctx.enqueue_function[m2_yoff_exp_cache_kernel](
            retained[cache_index].unsafe_ptr(), dacs.unsafe_ptr(), Int32(cache_cells),
            grid_dim=(_grid(cache_cells), 1, 1), block_dim=(MAMBA2_TPB, 1, 1),
        )
        ctx.enqueue_function[m2_yoff_y_kernel[True]](
            yoff.unsafe_ptr(),
            y_out.unsafe_ptr(),
            ydiag.unsafe_ptr(),
            xbc_work.unsafe_ptr(),
            pass_states.unsafe_ptr(),
            retained[cache_index].unsafe_ptr(),
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
    elif M2_OCC_YOFF:
        ctx.enqueue_function[m2_yoff_tile_kernel](
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
            grid_dim=(b * nc * nh * ((qv + M2_YO_ROWS - 1) // M2_YO_ROWS), 1, 1),
            block_dim=(M2_YO_THREADS, 1, 1),
        )
    else:
        ctx.enqueue_function[m2_yoff_y_kernel[False]](
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
    # Materialization is a lifetime experiment: its buffer cannot be
    # released before the consumer drains, even on the otherwise async
    # production block path. Count this additional wait in whole-op timing.
    if drain or len(retained)>0:
        ctx.synchronize()
    _ = retained^
