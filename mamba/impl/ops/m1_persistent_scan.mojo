# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel.
"""Mamba-1 persistent chunked selective scan (lane/neural-fusions, L13).

`-D MOJOLEARN_IDN_M1_PERSISTENT_SCAN` (IDENTICAL only, default OFF; the switch
is read in `selective_scan_interface.mojo`, which calls `m1_persistent_scan`).

WHY. The shipped `selective_scan_fwd_kernel` runs one thread per `(batch,
channel)` pair and walks all `DSTATE` chains and the `y` fold inside it. At
batch 1 that is `d_inner` threads for the whole sequence (768 threads for a
d_model 384 block): a handful of warps per GPU, each serial over L tokens. The
NI38 window arm (`identical_scan_window.mojo`) spreads the chains but pays two
launches per 64-token window (2 L / 64 launches).

WHAT. ONE launch. A block owns `CH` channels of one batch row for the WHOLE
sequence (no inter-block dependence, so no look-back is needed: the chains of
different channels are independent). Thread `(c, n)` owns chain `(b, d, n)` and
walks it serially, ascending tokens, `TOK` tokens at a time, writing each
state into shared memory; after a barrier every thread of the block forms the
chunk's `y` and `out` cells from shared memory. The block then moves on to the
next chunk. Parallelism is `batch * d_inner * DSTATE` chains (16x the shipped
kernel) and the `y` fold is spread over the block.

BITS. Unchanged; this is an execution plan of the shipped arithmetic (contract
seams S5-S11, `selective_scan_fwd_kernel` line for line):
  S5/S6  da   = ftz(exp(ftz(dl * a)))
  S7/S8  dbu  = ftz(ftz(dl * b) * u)
  S9     h    = ftz(fma(da, h, dbu))                one rounding
  S10    y    = serial ascending n from +0.0, ftz(fma(ftz(C[n]), h[n], acc))
  S11    out  = ftz(y + ftz(u * D))                 D last, unfused
Every chain value is produced by exactly one thread in token order; the `y`
fold of a cell is one thread's ascending-n loop. `CH` and `TOK` decide only
which thread holds which value and when the barrier falls. The host column is
untouched (it computes the same graph).

PARAMETERS (execution plan only, no bit moves; sweep per vendor):
  MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH      channels per block, legal 2|4|8 (default 4)
  MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS  tokens per staged chunk, legal 16|32|64 (default 32)
Shared memory is `TOK * CH * DSTATE * 4` bytes (8 KiB at the defaults, 32 KiB
at the largest legal pair), within every column's 32 KiB floor. No rule reads
a shape: the block geometry is fixed and the grid covers any (batch, dim, L).
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import get_defined_int
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import (
    ftz, identical_exp, identical_mul,
    identical_mul_add,
)

comptime M1_PSCAN_CH = get_defined_int["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH", 4]()
comptime M1_PSCAN_TOK = get_defined_int["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS", 32]()


def m1_persistent_scan_kernel[DSTATE: Int, CH: Int, TOK: Int](
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
):
    comptime NTH = CH * DSTATE
    var batch = Int(batch_in)
    var seqlen = Int(seqlen_in)
    var dim = Int(dim_in)
    var groups = (dim + CH - 1) // CH
    var blk = Int(block_idx.x)
    # Block-uniform early exit, before any barrier.
    if blk >= batch * groups:
        return
    var bb = blk // groups
    var d0 = (blk - bb * groups) * CH
    var tid = Int(thread_idx.x)
    var cl = tid // DSTATE
    var n = tid - cl * DSTATE
    var d = d0 + cl
    var live = d < dim

    var hs = stack_allocation[
        TOK * CH * DSTATE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    var state = Float32(0.0)
    var av = Float32(0.0)
    if live:
        state = ftz(h_ptr.unsafe_load((bb * dim + d) * DSTATE + n))
        av = ftz(a_ptr.unsafe_load(d * DSTATE + n))

    var t0 = 0
    while t0 < seqlen:
        var count = min(TOK, seqlen - t0)
        # ---- chains: S5-S9, one thread per (d, n), ascending tokens.
        if live:
            for j in range(count):
                var t = bb * seqlen + t0 + j
                var uv = ftz(u_ptr.unsafe_load(t * dim + d))
                var dl = ftz(delta_ptr.unsafe_load(t * dim + d))
                var da = ftz(identical_exp(ftz(identical_mul(dl, av))))
                var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
                var db = ftz(identical_mul(dl, bv))
                var dbu = ftz(identical_mul(db, uv))
                state = ftz(identical_mul_add(da, state, dbu))
                hs.unsafe_store((j * CH + cl) * DSTATE + n, state)
        barrier()
        # ---- outputs: S10-S11, one thread per (token, channel) cell.
        var cell = tid
        while cell < count * CH:
            var j2 = cell // CH
            var c2 = cell - j2 * CH
            var dd = d0 + c2
            if dd < dim:
                var t2 = bb * seqlen + t0 + j2
                var acc = Float32(0.0)
                comptime for nn in range(DSTATE):
                    acc = ftz(identical_mul_add(
                        ftz(c_ptr.unsafe_load(t2 * DSTATE + nn)),
                        hs.unsafe_load((j2 * CH + c2) * DSTATE + nn),
                        acc,
                    ))
                y_ptr.unsafe_store(t2 * dim + dd, acc)
                var uv2 = ftz(u_ptr.unsafe_load(t2 * dim + dd))
                var p = ftz(identical_mul(uv2, ftz(d_ptr.unsafe_load(dd))))
                out_ptr.unsafe_store(t2 * dim + dd, ftz(acc + p))
            cell += NTH
        # The next chunk overwrites `hs`; every reader must be done first.
        barrier()
        t0 += count

    if live:
        h_ptr.unsafe_store((bb * dim + d) * DSTATE + n, state)


def m1_persistent_scan[DSTATE: Int](
    ctx: DeviceContext,
    mut output: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
    mut h: DeviceBuffer[DType.float32], mut u: DeviceBuffer[DType.float32],
    mut delta: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    mut bmat: DeviceBuffer[DType.float32], mut cmat: DeviceBuffer[DType.float32],
    mut dskip: DeviceBuffer[DType.float32], batch: Int, seqlen: Int, dim: Int,
) raises:
    """ASYNCHRONOUS; the caller synchronizes (selective_scan_fn does)."""
    comptime assert M1_PSCAN_CH == 2 or M1_PSCAN_CH == 4 or M1_PSCAN_CH == 8, (
        "MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH: legal set 2|4|8"
    )
    comptime assert M1_PSCAN_TOK == 16 or M1_PSCAN_TOK == 32 or M1_PSCAN_TOK == 64, (
        "MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS: legal set 16|32|64"
    )
    if batch <= 0 or dim <= 0:
        return
    var blocks = batch * ((dim + M1_PSCAN_CH - 1) // M1_PSCAN_CH)
    comptime kern = m1_persistent_scan_kernel[DSTATE, M1_PSCAN_CH, M1_PSCAN_TOK]
    ctx.enqueue_function[kern](
        output.unsafe_ptr(), y.unsafe_ptr(), h.unsafe_ptr(), u.unsafe_ptr(),
        delta.unsafe_ptr(), a.unsafe_ptr(), bmat.unsafe_ptr(), cmat.unsafe_ptr(),
        dskip.unsafe_ptr(), Int32(batch), Int32(seqlen), Int32(dim),
        grid_dim=(blocks, 1, 1), block_dim=(M1_PSCAN_CH * DSTATE, 1, 1),
    )
