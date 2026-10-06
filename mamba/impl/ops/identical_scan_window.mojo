# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel.
"""NI38 S subarm: bounded windows, parallel independent state chains.

This is deliberately a separate S experiment from the catalog's pending V
AFFINE composition. Each (batch, channel, state-index) chain walks a window
serially with the original da, (delta*B)*u and fused state update. A second
kernel forms each output's state-index fold in ascending order, then adds the
D skip last. Window boundaries store the exact carried state; no affine
summary, reassociation, request-length rule or approximate scan is introduced.

64 is a storage/scheduling window: changing it cannot change a numeric chain.
Scratch holds at most 64 tokens, allocated once and reused in ordered launches.
The unmodified host and backward checkpoint kernels are the S counterparts.
No compilation, identity, quality or whole-operation timing has been run.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_exp, identical_mul, identical_mul_add

comptime SCAN_WINDOW_TOKENS = 64
comptime SCAN_WINDOW_TPB = 128


def scan_window_state_kernel[DSTATE: Int](
    window: MutPointer[Float32, MutAnyOrigin],
    h: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    delta: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    bmat: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32, length_in: Int32, dim_in: Int32,
    first_in: Int32, count_in: Int32,
):
    var chain = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dim = Int(dim_in)
    if chain >= Int(batch_in) * dim * DSTATE:
        return
    var n = chain % DSTATE
    var pair = chain // DSTATE
    var d = pair % dim
    var bb = pair // dim
    var length = Int(length_in)
    var count = Int(count_in)
    var state = ftz(h.unsafe_load(chain))
    var av = ftz(a.unsafe_load(d * DSTATE + n))
    for j in range(count):
        var token = bb * length + Int(first_in) + j
        var uv = ftz(u.unsafe_load(token * dim + d))
        var dl = ftz(delta.unsafe_load(token * dim + d))
        var da = ftz(identical_exp(ftz(identical_mul(dl, av))))
        var bv = ftz(bmat.unsafe_load(token * DSTATE + n))
        var db = ftz(identical_mul(dl, bv))
        var dbu = ftz(identical_mul(db, uv))
        state = ftz(identical_mul_add(da, state, dbu))
        window.unsafe_store(((bb * count + j) * dim + d) * DSTATE + n, state)
    h.unsafe_store(chain, state)


def scan_window_output_kernel[DSTATE: Int](
    out: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    window: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    cmat: MutPointer[Float32, MutAnyOrigin],
    dskip: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32, length_in: Int32, dim_in: Int32,
    first_in: Int32, count_in: Int32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dim = Int(dim_in)
    var count = Int(count_in)
    if cell >= Int(batch_in) * count * dim:
        return
    var d = cell % dim
    var row = cell // dim
    var j = row % count
    var bb = row // count
    var token = bb * Int(length_in) + Int(first_in) + j
    var acc = Float32(0.0)
    comptime for n in range(DSTATE):
        acc = ftz(identical_mul_add(
            ftz(cmat.unsafe_load(token * DSTATE + n)),
            ftz(window.unsafe_load(cell * DSTATE + n)), acc,
        ))
    y.unsafe_store(token * dim + d, acc)
    var uv = ftz(u.unsafe_load(token * dim + d))
    var dv = ftz(dskip.unsafe_load(d))
    var skip = ftz(identical_mul(uv, dv))
    out.unsafe_store(token * dim + d, ftz(acc + skip))


def identical_selective_scan_window[DSTATE: Int](
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
    mut h: DeviceBuffer[DType.float32], mut u: DeviceBuffer[DType.float32],
    mut delta: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    mut bmat: DeviceBuffer[DType.float32], mut cmat: DeviceBuffer[DType.float32],
    mut dskip: DeviceBuffer[DType.float32], batch: Int, length: Int, dim: Int,
) raises:
    if batch <= 0 or length <= 0 or dim <= 0:
        return
    var capacity = min(length, SCAN_WINDOW_TOKENS)
    var window = ctx.enqueue_create_buffer[DType.float32](batch * capacity * dim * DSTATE)
    var chains = batch * dim * DSTATE
    var first = 0
    try:
        while first < length:
            var count = min(SCAN_WINDOW_TOKENS, length - first)
            ctx.enqueue_function[scan_window_state_kernel[DSTATE]](
                window.unsafe_ptr(), h.unsafe_ptr(), u.unsafe_ptr(), delta.unsafe_ptr(),
                a.unsafe_ptr(), bmat.unsafe_ptr(), Int32(batch), Int32(length),
                Int32(dim), Int32(first), Int32(count),
                grid_dim=((chains + SCAN_WINDOW_TPB - 1) // SCAN_WINDOW_TPB, 1, 1),
                block_dim=(SCAN_WINDOW_TPB, 1, 1),
            )
            var outputs = batch * count * dim
            ctx.enqueue_function[scan_window_output_kernel[DSTATE]](
                out.unsafe_ptr(), y.unsafe_ptr(), window.unsafe_ptr(), u.unsafe_ptr(),
                cmat.unsafe_ptr(), dskip.unsafe_ptr(), Int32(batch), Int32(length),
                Int32(dim), Int32(first), Int32(count),
                grid_dim=((outputs + SCAN_WINDOW_TPB - 1) // SCAN_WINDOW_TPB, 1, 1),
                block_dim=(SCAN_WINDOW_TPB, 1, 1),
            )
            first += count
        # The window is owned here, not by the caller. Include this completion in
        # the whole scan measurement; never release scratch while a reader is live.
        ctx.synchronize()
    except error:
        # Preserve the original failure while retiring queued scratch readers.
        try: ctx.synchronize()
        except: pass
        raise error
    _ = window^
