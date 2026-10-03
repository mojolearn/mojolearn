# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-1 selective scan as a chunked parallel scan over time (lane
afn-mamba, 2026-10-03; `-D MOJOLEARN_AFN_MAMBA1_CHUNKSCAN`, FAST + Apple
only; `selective_scan_fn` dispatches here).

Main's `selective_scan_fwd_kernel` gives one thread the whole sequence of
one `(batch, dim)` pair: at the board shape (B 1, L 2048, d_inner 768)
that is 768 threads each walking 2048 steps of 16 states, a launch that
leaves most of an Apple GPU idle for the whole scan.

Here one THREADGROUP of 32 lanes (one simdgroup) owns a `(batch, dim)`
pair and the sequence is cut into 32 chunks of `ceil(L / 32)` steps:

  pass 1  lane c walks its chunk from h = 0, keeping the chunk's local end
          state S_c[n] and the product of its decays P_c[n] = prod exp(dt A_n)
          (both f32, the same per-step products as main's kernel);
  carry   lane 0 folds the 32 chunk summaries serially in threadgroup
          memory: h_start_{c+1} = fma(P_c, h_start_c, S_c) from the
          incoming state h_start_0 = h_in;
  pass 2  lane c re-walks its chunk from h_start_c with main's per-step
          arithmetic (deltaA = exp(dt A), deltaB_u = (dt B) u, h = fma(deltaA,
          h, deltaB_u), y = sum_n C_n h_n ascending from +0.0, out = y + u D)
          and writes y, out; the last chunk's lane stores the final state.

ONE launch instead of one, but 32x the parallelism, no scratch buffer and
no host step. The per-step arithmetic is main's; what moves is the FOLD
ORDER of the recurrence across a chunk boundary (the carry is injected as
P h_start + S instead of being threaded through every step), which is f32
reassociation noise. exp(dt A) is evaluated in both passes (a recompute,
never a readback: storing it would be L x d_inner x 16 floats).

The threadgroup barrier here orders THREADGROUP memory only, which is all
this kernel needs (every device write is by the lane that later reads it,
or never read in this launch).
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_exp, identical_mul, identical_mul_add

#: lanes per (batch, dim) threadgroup: one Metal simdgroup
comptime AFN_SCAN_LANES = 32


def afn_scan_chunked_kernel[
    DSTATE: Int
](
    out_ptr: MutPointer[Float32, MutAnyOrigin],
    y_ptr: MutPointer[Float32, MutAnyOrigin],
    h_ptr: MutPointer[Float32, MutAnyOrigin],
    u_ptr: MutPointer[Float32, MutAnyOrigin],
    delta_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32,
    seqlen_in: Int32,
    dim_in: Int32,
    chunk_in: Int32,
):
    var batch = Int(batch_in)
    var seqlen = Int(seqlen_in)
    var dim = Int(dim_in)
    var chunk = Int(chunk_in)
    var cell = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    # The whole threadgroup returns together (uniform), before any barrier.
    if cell >= batch * dim:
        return
    var bb = cell // dim
    var d = cell - bb * dim
    var nc = (seqlen + chunk - 1) // chunk

    var s_end = stack_allocation[
        AFN_SCAN_LANES * DSTATE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_prod = stack_allocation[
        AFN_SCAN_LANES * DSTATE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var s_start = stack_allocation[
        AFN_SCAN_LANES * DSTATE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))
    var d_val = ftz(d_ptr.unsafe_load(d))

    var t0 = lane * chunk
    var t1 = t0 + chunk
    if t1 > seqlen:
        t1 = seqlen

    # ---- pass 1: the chunk from h = 0; its end state and its decay product.
    var st = SIMD[DType.float32, DSTATE](0.0)
    var pr = SIMD[DType.float32, DSTATE](1.0)
    if lane < nc:
        for li in range(t0, t1):
            var t = bb * seqlen + li
            var uv = ftz(u_ptr.unsafe_load(t * dim + d))
            var dl = ftz(delta_ptr.unsafe_load(t * dim + d))
            comptime for n in range(DSTATE):
                var da = ftz(identical_exp(ftz(identical_mul(dl, a_vals[n]))))
                var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
                var dbu = ftz(identical_mul(ftz(identical_mul(dl, bv)), uv))
                st[n] = ftz(identical_mul_add(da, st[n], dbu))
                pr[n] = ftz(identical_mul(pr[n], da))
    comptime for n in range(DSTATE):
        s_end.unsafe_store(lane * DSTATE + n, st[n])
        s_prod.unsafe_store(lane * DSTATE + n, pr[n])
    barrier()

    # ---- the carry: h entering every chunk, from the incoming state.
    if lane == 0:
        var h = SIMD[DType.float32, DSTATE](0.0)
        comptime for n in range(DSTATE):
            h[n] = ftz(h_ptr.unsafe_load((bb * dim + d) * DSTATE + n))
        for c in range(nc):
            comptime for n in range(DSTATE):
                s_start.unsafe_store(c * DSTATE + n, h[n])
                h[n] = ftz(
                    identical_mul_add(
                        s_prod.unsafe_load(c * DSTATE + n),
                        h[n],
                        s_end.unsafe_load(c * DSTATE + n),
                    )
                )
    barrier()

    # ---- pass 2: the chunk from its entering state, with the outputs.
    if lane < nc:
        comptime for n in range(DSTATE):
            st[n] = s_start.unsafe_load(lane * DSTATE + n)
        for li in range(t0, t1):
            var t = bb * seqlen + li
            var uv = ftz(u_ptr.unsafe_load(t * dim + d))
            var dl = ftz(delta_ptr.unsafe_load(t * dim + d))
            comptime for n in range(DSTATE):
                var da = ftz(identical_exp(ftz(identical_mul(dl, a_vals[n]))))
                var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
                var dbu = ftz(identical_mul(ftz(identical_mul(dl, bv)), uv))
                st[n] = ftz(identical_mul_add(da, st[n], dbu))
            var acc = Float32(0.0)
            comptime for n in range(DSTATE):
                acc = ftz(
                    identical_mul_add(
                        ftz(c_ptr.unsafe_load(t * DSTATE + n)), st[n], acc
                    )
                )
            y_ptr.unsafe_store(t * dim + d, acc)
            var p = ftz(identical_mul(uv, d_val))
            out_ptr.unsafe_store(t * dim + d, ftz(acc + p))
        if lane == nc - 1:
            # The state after the last token: the walked value (the same
            # bits the last outputs were computed from).
            comptime for n in range(DSTATE):
                h_ptr.unsafe_store((bb * dim + d) * DSTATE + n, st[n])


def afn_selective_scan_chunked[
    DSTATE: Int
](
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    mut h_state: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut delta: DeviceBuffer[DType.float32],
    mut A: DeviceBuffer[DType.float32],
    mut B: DeviceBuffer[DType.float32],
    mut C: DeviceBuffer[DType.float32],
    mut D: DeviceBuffer[DType.float32],
    batch: Int,
    seqlen: Int,
    dim: Int,
) raises:
    """ONE launch: `batch * dim` threadgroups of `AFN_SCAN_LANES` lanes.
    ASYNCHRONOUS (the caller waits as it did for main's kernel)."""
    var total = batch * dim
    if total <= 0 or seqlen <= 0:
        return
    var chunk = (seqlen + AFN_SCAN_LANES - 1) // AFN_SCAN_LANES
    if chunk < 1:
        chunk = 1
    comptime kern = afn_scan_chunked_kernel[DSTATE]
    ctx.enqueue_function[kern](
        out.unsafe_ptr(),
        y.unsafe_ptr(),
        h_state.unsafe_ptr(),
        u.unsafe_ptr(),
        delta.unsafe_ptr(),
        A.unsafe_ptr(),
        B.unsafe_ptr(),
        C.unsafe_ptr(),
        D.unsafe_ptr(),
        Int32(batch),
        Int32(seqlen),
        Int32(dim),
        Int32(chunk),
        grid_dim=(total, 1, 1),
        block_dim=(AFN_SCAN_LANES, 1, 1),
    )
