# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-1 selective scan's math, written ONCE (Andrew, 2026-11-22: one body per operation).

Every form of the scan calls these `@always_inline` bodies; none carries its own copy:

  body                     callers
  m1_decay / m1_drive /    selective_scan_fwd_kernel and m1_persistent_scan_kernel (the block's forward),
  m1_state                 selective_scan_checkpoint_kernel, selective_scan_bwd_scan_kernel and
                           mamba_bwd_da_partial_kernel (the block's backward), and the generic op
                           (autograd/ssm_ops.mojo, the public SelectiveScan layer) on the host and the device
  m1_readout_term, m1_skip the forward readout S10 and the D skip S11 (both forward kernels, the op)
  m1_*_cell                one thread's whole body of each block kernel; the kernel computes its thread
                           index and calls it, the generic op's element function calls it with its element
                           index, so the block and a model composed from the public layer run the same
                           arithmetic in the same order
  m1_skip_grads            S11's backward pre-products (mamba_bwd_gate_kernel, the op)

The seams and their DEVIATION numbers are documented where they were first written:
`mamba/impl/ops/selective_scan_interface.mojo::selective_scan_fwd_kernel` (S5-S11) and
`autograd/kernels/mamba1_scan.mojo` (T1, T3, B13-B18). Moving them here moved no float operation: each body
is the kernel's former text with the thread index as an argument.

The sabotage arms live here because they live inside the bodies; the kernels' files import them (and so
re-export them to their old importers). A body takes typed values and pointers and has no launch,
allocation or GPU import in it, so it inlines into a device kernel, a host loop and the host
restatement of `tools/mamba_host_gen.py` alike.
"""
from std.math import exp2
from std.sys.compile import is_defined
from checks.numerics import ftz, identical_exp, identical_mul_add, identical_mul
from mamba.host.gen.gemm_identical_parts import (
    GEMM_FOLD_SLOTS,
    _fold_drain,
    _fold_push,
    _leaf_at,
    _leaf_bounds,
)
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED, NEURAL_CHAINS, neural_merge_chains

comptime M1P = MutPointer[Float32, MutAnyOrigin]

# ---------------------------------------------------------------- forward sabotage arms (moved from
# mamba/impl/ops/selective_scan_interface.mojo, 2026-11-22; that file documents each)
comptime SAB_S8_CUDA_PAIRING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S8_CUDA_PAIRING"
]()
comptime SAB_S9_UNFUSED = is_defined["MOJOLEARN_MAMBA_SABOTAGE_S9_UNFUSED"]()
comptime SAB_S11_D_FIRST = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S11_D_FIRST"
]()
comptime SAB_S10_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S10_DESCENDING"
]()
comptime SAB_S5_EXP2 = is_defined["MOJOLEARN_MAMBA_SABOTAGE_S5_EXP2"]()

comptime ANY_SABOTAGE = (
    SAB_S8_CUDA_PAIRING
    or SAB_S9_UNFUSED
    or SAB_S11_D_FIRST
    or SAB_S10_DESCENDING
    or SAB_S5_EXP2
)

# ---------------------------------------------------------------- backward sabotage arms read inside the
# bodies (moved from autograd/kernels/mamba1_scan.mojo, 2026-11-22; that file documents each and keeps the
# two atomic arms, which are launches, not bodies)
comptime SAB_BWD_S9B_FORWARD = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S9B_FORWARD"
]()
comptime SAB_BWD_S9B_DA_OFFSET = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S9B_DA_OFFSET"
]()
comptime SAB_BWD_S9B_UNFUSED = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S9B_UNFUSED"
]()
comptime SAB_BWD_T1_SEED_ADD = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_T1_SEED_ADD"
]()
comptime SAB_BWD_H_SUBTRACT = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_H_SUBTRACT"
]()
comptime SAB_BWD_S10_N_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S10_N_DESCENDING"
]()
comptime SAB_BWD_DDELTA_TWO_FOLDS = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_DDELTA_TWO_FOLDS"
]()
comptime SAB_BWD_DA_ASCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_DA_ASCENDING"
]()


# ================================================================ scalar seams
@always_inline
def m1_decay(dl: Float32, av: Float32) -> Float32:
    """S5 + S6: `deltaA = exp(delta * A)`, one product then the pinned exp."""
    var da_arg = ftz(identical_mul(dl, av))
    comptime if SAB_S5_EXP2:
        # SABOTAGE: MAX's and the CUDA kernel's exp2 substitution.
        return ftz(exp2(ftz(identical_mul(da_arg, Float32(1.4426950408889634)))))
    else:
        return ftz(identical_exp(da_arg))


@always_inline
def m1_drive(dl: Float32, bv: Float32, uv: Float32) -> Float32:
    """S7 + S8: `deltaB_u = (delta * B) * u`, the reference's pairing."""
    comptime if SAB_S8_CUDA_PAIRING:
        # SABOTAGE: `B * (delta * u)`, cuh:162 with :222.
        var du = ftz(identical_mul(dl, uv))
        return ftz(identical_mul(bv, du))
    else:
        var db = ftz(identical_mul(dl, bv))
        return ftz(identical_mul(db, uv))


@always_inline
def m1_state(da: Float32, h: Float32, dbu: Float32) -> Float32:
    """S9: `h = deltaA * h + deltaB_u`, ONE rounding (fused)."""
    comptime if SAB_S9_UNFUSED:
        # SABOTAGE: the reference's literal two roundings.
        return ftz(ftz(identical_mul(da, h)) + dbu)
    else:
        return ftz(identical_mul_add(da, h, dbu))


@always_inline
def m1_readout_term(cv: Float32, h: Float32, acc: Float32) -> Float32:
    """S10's one term: `acc += C[n] * h[n]`, fused, folded n ascending from +0.0."""
    return ftz(identical_mul_add(cv, h, acc))


@always_inline
def m1_skip(acc: Float32, uv: Float32, d_val: Float32) -> Float32:
    """S11: `out = y + u * D`, the product rounded on its own, D LAST."""
    var p = ftz(identical_mul(uv, d_val))
    return ftz(acc + p)


@always_inline
def m1_skip_grads(dskv: Float32, d_val: Float32, uv: Float32) -> Tuple[Float32, Float32]:
    """S11's backward pre-products: (du_d = dsk * D, pD = dsk * u); dD is the reduction of pD."""
    return (ftz(identical_mul(dskv, d_val)), ftz(identical_mul(dskv, uv)))


@always_inline
def m1_a_from_a_log(a_log: Float32) -> Float32:
    """S15: `A = -exp(A_log)`."""
    return -ftz(identical_exp(ftz(a_log)))


@always_inline
def m1_da_log(da: Float32, av: Float32) -> Float32:
    """S15's backward: `dA_log = dA * A`, A read from the recorded stage."""
    return ftz(identical_mul(ftz(da), ftz(av)))


# ================================================================ whole-thread bodies of the block kernels
@always_inline
def m1_scan_cell[DSTATE: Int](
    thread_id: Int, out_ptr: M1P, y_ptr: M1P, h_ptr: M1P, u_ptr: M1P, delta_ptr: M1P, a_ptr: M1P,
    b_ptr: M1P, c_ptr: M1P, d_ptr: M1P, batch: Int, seqlen: Int, dim: Int,
):
    """`selective_scan_fwd_kernel`'s thread: one `(batch, dim)` pair, the whole sequence in order; writes
    `y` (S10), `out` (S11) at every token and the state after the last token into `h_ptr` (in and out)."""
    if thread_id >= batch * dim:
        return
    var bb = thread_id // dim
    var d = thread_id - bb * dim
    if bb >= batch or d >= dim:
        return
    var state = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        state[n] = ftz(h_ptr.unsafe_load((bb * dim + d) * DSTATE + n))
    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))
    var d_val = ftz(d_ptr.unsafe_load(d))
    for li in range(seqlen):
        var t = bb * seqlen + li
        var uv = ftz(u_ptr.unsafe_load(t * dim + d))
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))
        comptime for n in range(DSTATE):
            var da = m1_decay(dl, a_vals[n])
            var dbu = m1_drive(dl, ftz(b_ptr.unsafe_load(t * DSTATE + n)), uv)
            state[n] = m1_state(da, state[n], dbu)
        var acc = Float32(0.0)
        comptime if SAB_S11_D_FIRST:
            # SABOTAGE: the CUDA kernel seeds its output accumulator with `D * u` (cuh:163).
            acc = ftz(identical_mul(d_val, uv))
        comptime if SAB_S10_DESCENDING:
            # SABOTAGE: the same terms, folded n descending.
            comptime for nn in range(DSTATE):
                var n = DSTATE - 1 - nn
                acc = m1_readout_term(ftz(c_ptr.unsafe_load(t * DSTATE + n)), state[n], acc)
        else:
            comptime for n in range(DSTATE):
                acc = m1_readout_term(ftz(c_ptr.unsafe_load(t * DSTATE + n)), state[n], acc)
        y_ptr.unsafe_store(t * dim + d, acc)
        comptime if SAB_S11_D_FIRST:
            out_ptr.unsafe_store(t * dim + d, acc)
        else:
            out_ptr.unsafe_store(t * dim + d, m1_skip(acc, uv, d_val))
    comptime for n in range(DSTATE):
        h_ptr.unsafe_store((bb * dim + d) * DSTATE + n, state[n])


@always_inline
def m1_checkpoint_cell[DSTATE: Int](
    thread_id: Int, hck_ptr: M1P, h_in_ptr: M1P, u_ptr: M1P, delta_ptr: M1P, a_ptr: M1P, b_ptr: M1P,
    batch: Int, seqlen: Int, dim: Int,
):
    """`selective_scan_checkpoint_kernel`'s thread: every `h[t, d, n]` for `t` in `[-1, L)`, slot 0 the
    initial state, the S5-S9 chain of the forward (the same bodies, so the same words)."""
    if thread_id >= batch * dim:
        return
    var bb = thread_id // dim
    var d = thread_id - bb * dim
    if bb >= batch or d >= dim:
        return
    var slots = seqlen + 1
    var state = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        state[n] = ftz(h_in_ptr.unsafe_load((bb * dim + d) * DSTATE + n))
    comptime for n in range(DSTATE):
        hck_ptr.unsafe_store(((bb * slots + 0) * dim + d) * DSTATE + n, state[n])
    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))
    for li in range(seqlen):
        var t = bb * seqlen + li
        var uv = ftz(u_ptr.unsafe_load(t * dim + d))
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))
        comptime for n in range(DSTATE):
            var da = m1_decay(dl, a_vals[n])
            var dbu = m1_drive(dl, ftz(b_ptr.unsafe_load(t * DSTATE + n)), uv)
            state[n] = m1_state(da, state[n], dbu)
        comptime for n in range(DSTATE):
            hck_ptr.unsafe_store(((bb * slots + (li + 1)) * dim + d) * DSTATE + n, state[n])


@always_inline
def m1_bwd_scan_cell[DSTATE: Int](
    thread_id: Int, dh_ptr: M1P, du_s_ptr: M1P, ddelta_ptr: M1P, w_ptr: M1P, dy_ptr: M1P, c_ptr: M1P,
    b_ptr: M1P, delta_ptr: M1P, a_ptr: M1P, u_ptr: M1P, hck_ptr: M1P, batch: Int, seqlen: Int, dim: Int,
):
    """`selective_scan_bwd_scan_kernel`'s thread: one `(batch, dim)` pair in REVERSE; T1 `dh` (fused, one
    rounding), B17 `du_s`, B18 `ddelta` (one interleaved chain, B term then A term) and `w = delta * u`."""
    if thread_id >= batch * dim:
        return
    var bb = thread_id // dim
    var d = thread_id - bb * dim
    if bb >= batch or d >= dim:
        return
    var slots = seqlen + 1
    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))
    var dh_state = SIMD[DType.float32, DSTATE](0.0)
    var da_carry = SIMD[DType.float32, DSTATE](0.0)
    for step in range(seqlen):
        var li = seqlen - 1 - step
        comptime if SAB_BWD_S9B_FORWARD:
            li = step
        var t = bb * seqlen + li
        var dyv = ftz(dy_ptr.unsafe_load(t * dim + d))
        var uv = ftz(u_ptr.unsafe_load(t * dim + d))
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))
        comptime for n in range(DSTATE):
            var cv = ftz(c_ptr.unsafe_load(t * DSTATE + n))
            var contrib = ftz(identical_mul(dyv, cv))
            var da_t = m1_decay(dl, a_vals[n])
            var dh_n = ftz(contrib)
            comptime if SAB_BWD_T1_SEED_ADD:
                dh_n = ftz(ftz(Float32(0.0)) + ftz(contrib))
            if step != 0:
                var afac = da_carry[n]
                comptime if SAB_BWD_S9B_DA_OFFSET:
                    afac = da_t
                comptime if SAB_BWD_S9B_UNFUSED:
                    dh_n = ftz(ftz(identical_mul(afac, dh_state[n])) + contrib)
                else:
                    dh_n = ftz(identical_mul_add(afac, dh_state[n], contrib))
            dh_state[n] = dh_n
            da_carry[n] = da_t
            dh_ptr.unsafe_store((t * dim + d) * DSTATE + n, dh_n)
        var acc_us = Float32(0.0)
        var acc_dd = Float32(0.0)
        var acc_dd_a = Float32(0.0)
        comptime for nn in range(DSTATE):
            var n = nn
            comptime if SAB_BWD_S10_N_DESCENDING:
                n = DSTATE - 1 - nn
            var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
            var dbb = ftz(identical_mul(dl, bv))
            acc_us = ftz(identical_mul_add(dh_state[n], dbb, acc_us))
            var ddbb = ftz(identical_mul(dh_state[n], uv))
            acc_dd = ftz(identical_mul_add(ddbb, bv, acc_dd))
            var d_arg: Float32
            comptime if SAB_BWD_H_SUBTRACT:
                var dbu_h = ftz(identical_mul(dbb, uv))
                var hcur = ftz(hck_ptr.unsafe_load(((bb * slots + (li + 1)) * dim + d) * DSTATE + n))
                var arec = ftz(ftz(hcur) - ftz(dbu_h))
                d_arg = ftz(identical_mul(dh_state[n], arec))
            else:
                var hprev = ftz(hck_ptr.unsafe_load(((bb * slots + li) * dim + d) * DSTATE + n))
                var d_da = ftz(identical_mul(dh_state[n], hprev))
                d_arg = ftz(identical_mul(d_da, da_carry[n]))
            comptime if SAB_BWD_DDELTA_TWO_FOLDS:
                acc_dd_a = ftz(identical_mul_add(d_arg, a_vals[n], acc_dd_a))
            else:
                acc_dd = ftz(identical_mul_add(d_arg, a_vals[n], acc_dd))
        var ddelta_v = acc_dd
        comptime if SAB_BWD_DDELTA_TWO_FOLDS:
            ddelta_v = ftz(ftz(acc_dd) + ftz(acc_dd_a))
        du_s_ptr.unsafe_store(t * dim + d, acc_us)
        ddelta_ptr.unsafe_store(t * dim + d, ddelta_v)
        w_ptr.unsafe_store(t * dim + d, ftz(identical_mul(dl, uv)))


@always_inline
def m1_dbc_cell[DSTATE: Int](
    cell: Int, dcm_ptr: M1P, dbm_ptr: M1P, dy_ptr: M1P, hck_ptr: M1P, w_ptr: M1P, dh_ptr: M1P,
    m: Int, l: Int, di: Int, leaf: Int, p_count: Int,
):
    """`mamba_bwd_dbc_kernel`'s thread: T3, `dC[t, n] = sum_d dy * h` and `dB[t, n] = sum_d w * dh` over
    the gemm contract's leaves of `di`, the leaves folded through the contract's tree."""
    if cell >= m * DSTATE:
        return
    var t = cell // DSTATE
    var n = cell - t * DSTATE
    if l < 1:
        return
    var bb = t // l
    var li = t - bb * l
    var slots = l + 1
    var stack_c = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ_c = 0
    var stack_b = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ_b = 0
    for tt in range(p_count):
        var bounds = _leaf_bounds(_leaf_at(tt, p_count), leaf, di)
        var acc_c = Float32(0.0)
        var acc_b = Float32(0.0)
        comptime if NEURAL_PROFILE_CHANGED:
            var ac = SIMD[DType.float32, NEURAL_CHAINS](0.0)
            var ab = SIMD[DType.float32, NEURAL_CHAINS](0.0)
            for d in range(bounds[0], bounds[1]):
                var lane = (d - bounds[0]) % NEURAL_CHAINS
                ac[lane] = ftz(identical_mul_add(ftz(dy_ptr.unsafe_load(t * di + d)), ftz(hck_ptr.unsafe_load(((bb * slots + li + 1) * di + d) * DSTATE + n)), ac[lane]))
                ab[lane] = ftz(identical_mul_add(ftz(w_ptr.unsafe_load(t * di + d)), ftz(dh_ptr.unsafe_load((t * di + d) * DSTATE + n)), ab[lane]))
            acc_c = neural_merge_chains[NEURAL_CHAINS](ac)
            acc_b = neural_merge_chains[NEURAL_CHAINS](ab)
        else:
            for d in range(bounds[0], bounds[1]):
                acc_c = ftz(identical_mul_add(
                    ftz(dy_ptr.unsafe_load(t * di + d)),
                    ftz(hck_ptr.unsafe_load(((bb * slots + (li + 1)) * di + d) * DSTATE + n)),
                    acc_c,
                ))
                acc_b = ftz(identical_mul_add(
                    ftz(w_ptr.unsafe_load(t * di + d)),
                    ftz(dh_ptr.unsafe_load((t * di + d) * DSTATE + n)),
                    acc_b,
                ))
        _ = _fold_push(stack_c, occ_c, ftz(acc_c))
        _ = _fold_push(stack_b, occ_b, ftz(acc_b))
    dcm_ptr.unsafe_store(cell, ftz(_fold_drain(stack_c, occ_c)))
    dbm_ptr.unsafe_store(cell, ftz(_fold_drain(stack_b, occ_b)))


@always_inline
def m1_da_partial_acc[DSTATE: Int](
    bb: Int, d: Int, dh_ptr: M1P, delta_ptr: M1P, a_ptr: M1P, u_ptr: M1P, b_ptr: M1P, hck_ptr: M1P,
    seqlen: Int, dim: Int,
) -> SIMD[DType.float32, DSTATE]:
    """`mamba_bwd_da_partial_kernel`'s fold for one `(batch, dim)` pair: `sum_t dh * h[t-1] * da * delta`
    with t descending; the kernel stores (or, sabotaged, atomically adds) the DSTATE partials."""
    var slots = seqlen + 1
    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))
    var acc = SIMD[DType.float32, DSTATE](0.0)
    for step in range(seqlen):
        var li = seqlen - 1 - step
        comptime if SAB_BWD_DA_ASCENDING:
            li = step
        var t = bb * seqlen + li
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))
        comptime for n in range(DSTATE):
            var dhv = ftz(dh_ptr.unsafe_load((t * dim + d) * DSTATE + n))
            var d_arg: Float32
            comptime if SAB_BWD_H_SUBTRACT:
                var uv = ftz(u_ptr.unsafe_load(t * dim + d))
                var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
                var dbb = ftz(identical_mul(dl, bv))
                var dbu_h = ftz(identical_mul(dbb, uv))
                var hcur = ftz(hck_ptr.unsafe_load(((bb * slots + (li + 1)) * dim + d) * DSTATE + n))
                var arec = ftz(ftz(hcur) - ftz(dbu_h))
                d_arg = ftz(identical_mul(dhv, arec))
            else:
                var da_t = m1_decay(dl, a_vals[n])
                var hprev = ftz(hck_ptr.unsafe_load(((bb * slots + li) * dim + d) * DSTATE + n))
                var d_da = ftz(identical_mul(dhv, hprev))
                d_arg = ftz(identical_mul(d_da, da_t))
            acc[n] = ftz(identical_mul_add(d_arg, dl, acc[n]))
    return acc


@always_inline
def m1_param_fold_cell(i: Int, out_ptr: M1P, partial_ptr: M1P, nb: Int, w: Int):
    """`mamba_bwd_param_fold_kernel`'s thread: the per-batch partials folded in batch order."""
    if i >= w:
        return
    var acc = Float32(0.0)
    for bb in range(nb):
        acc = ftz(ftz(acc) + ftz(partial_ptr.unsafe_load(bb * w + i)))
    out_ptr.unsafe_store(i, acc)
