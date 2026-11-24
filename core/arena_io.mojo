# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shared ARENA RANGES runner transfer (lane py-shared, 2026-09-28).

An arena-style binding (x_prep, x_metrics, and any program of units over
one float32 arena) used to zero-fill a host arena, upload ALL of it, run,
and download ALL of it: inputs, output slots and scratch both ways. Here a
program says which words are INPUTS and which are OUTPUTS, and only those
cross the bus:

  upload_ranges: every arena word outside the input ranges is zeroed ON
    THE DEVICE (the host arena's words arrive zeroed, so the device sees
    the same words), then each input range is copied from the host arena,
    or device to device from a `DeviceStore` slot (a resident input the
    host never re-sends).
  download_ranges: each declared output range comes back into the host
    arena at the same offsets; every other host word keeps what the caller
    put there.

Range lists are Int32 words at an address, laid out by
`mojolearn._arena_io`:
  ins  : triples [lo, hi, src], ascending and disjoint within [0, arena_len);
         src = -1: the host arena's words [lo, hi); src >= 0: the first
         hi - lo words of store slot `src`.
  outs : quads [lo, hi, CNT, mult] (x_metrics' layout). CNT < 0: [lo, hi)
         comes back; CNT >= 0: the arena word CNT (an Int32 the program
         wrote; it comes back first) bounds the range to
         [lo, lo + min(hi - lo, mult * CNT)).
         CNT <= -2 (lane vdf-tail-a, 2026-11-24, an addition; a runner that
         passes `slots=True` to `check_out_ranges` only): [lo, hi) is copied
         device to device into the first hi - lo words of `DeviceStore` slot
         -CNT - 2 (`copy_out_ranges_to_store`) and never comes back to the
         host (a stage result the next device step reads, e.g. a scaler's
         NaN-filled block); mult is unused.
Where a word travels moves no bit: the words a stage reads are the words
it read before.
"""
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_zero import enqueue_zero_bytes
from core.device_store import DeviceStore

comptime ArenaFP = MutPointer[Float32, MutAnyOrigin]
comptime ArenaIP = MutPointer[Int32, MutAnyOrigin]
comptime IN_INTS = 3
comptime OUT_INTS = 4


def check_in_ranges(ins_addr: Int, nins: Int, arena_len: Int) raises:
    """Raises unless the input triples are ascending, disjoint and inside
    the arena."""
    if nins < 0 or (nins > 0 and ins_addr == 0):
        raise Error("arena ranges: invalid input range list")
    var r = ArenaIP(unsafe_from_address=ins_addr)
    var at = 0
    for k in range(nins):  # small-loop(nins: arena input ranges): validates the per-launch range list, a handful of entries
        var lo = Int(r.unsafe_load(IN_INTS * k))
        var hi = Int(r.unsafe_load(IN_INTS * k + 1))
        var src = Int(r.unsafe_load(IN_INTS * k + 2))
        if lo < at or hi < lo or hi > arena_len or src < -1:
            raise Error("arena ranges: input ranges must be ascending, disjoint and inside the arena")
        at = hi


def check_out_ranges(outs_addr: Int, nouts: Int, arena_len: Int, slots: Bool = False) raises:
    """Raises unless every output quad lies inside the arena. A slot output
    (CNT <= -2) is refused unless the runner copies them (`slots`)."""
    if nouts < 0 or (nouts > 0 and outs_addr == 0):
        raise Error("arena ranges: invalid output range list")
    var o = ArenaIP(unsafe_from_address=outs_addr)
    for k in range(nouts):  # small-loop(nouts: arena output ranges): validates the per-launch range list, a handful of entries
        var lo = Int(o.unsafe_load(OUT_INTS * k))
        var hi = Int(o.unsafe_load(OUT_INTS * k + 1))
        var cn = Int(o.unsafe_load(OUT_INTS * k + 2))
        if lo < 0 or hi < lo or hi > arena_len or cn >= arena_len or Int(o.unsafe_load(OUT_INTS * k + 3)) < 0:
            raise Error("arena ranges: output range outside the arena")
        if cn <= -2 and not slots:
            raise Error("arena ranges: this runner has no store slot outputs")


def upload_ranges(
    ctx: DeviceContext, df: DeviceBuffer[DType.float32], host_f: ArenaFP, arena_len: Int,
    ins_addr: Int, nins: Int, store: DeviceStore,
) raises:
    """Enqueue the arena's device image: zeros outside the input ranges,
    the inputs inside them (host words, or a store slot device to device).
    `check_in_ranges` must have passed."""
    var r = ArenaIP(unsafe_from_address=ins_addr)
    var base = Int(df.unsafe_ptr())
    var at = 0
    for k in range(nins + 1):
        var lo = arena_len
        var hi = arena_len
        var src = -1
        if k < nins:
            lo = Int(r.unsafe_load(IN_INTS * k))
            hi = Int(r.unsafe_load(IN_INTS * k + 1))
            src = Int(r.unsafe_load(IN_INTS * k + 2))
        if lo > at:
            enqueue_zero_bytes(ctx, MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=base + 4 * at), 4 * (lo - at))
        if hi > lo:
            if src >= 0:
                store.copy_into(ctx, src, df, lo, hi - lo)
            else:
                ctx.enqueue_copy(dst_buf=df.create_sub_buffer[DType.float32](lo, hi - lo), src_ptr=host_f + lo)
        at = max(at, hi)


def download_ranges(
    ctx: DeviceContext, df: DeviceBuffer[DType.float32], host_f: ArenaFP, outs_addr: Int, nouts: Int,
) raises:
    """Enqueue the declared output ranges back into the host arena (a
    count-bounded range first brings its count word back and waits).
    `check_out_ranges` must have passed. The caller synchronizes."""
    if nouts <= 0:
        return
    var o = ArenaIP(unsafe_from_address=outs_addr)
    var bounded = False
    # A slot output (CNT <= -2) stays on the device (`copy_out_ranges_to_store`).
    for k in range(nouts):
        var cn = Int(o.unsafe_load(OUT_INTS * k + 2))
        if cn >= 0:
            bounded = True
            ctx.enqueue_copy(dst_ptr=host_f + cn, src_buf=df.create_sub_buffer[DType.float32](cn, 1))
    if bounded:
        ctx.synchronize()
    for k in range(nouts):
        var lo = Int(o.unsafe_load(OUT_INTS * k))
        var hi = Int(o.unsafe_load(OUT_INTS * k + 1))
        var cn = Int(o.unsafe_load(OUT_INTS * k + 2))
        if cn <= -2:
            continue
        if cn >= 0:
            var c = Int(bitcast[DType.int32](host_f.unsafe_load(cn))) * Int(o.unsafe_load(OUT_INTS * k + 3))
            hi = lo + max(0, min(hi - lo, c))
        if hi > lo:
            ctx.enqueue_copy(dst_ptr=host_f + lo, src_buf=df.create_sub_buffer[DType.float32](lo, hi - lo))


def copy_out_ranges_to_store(
    ctx: DeviceContext, df: DeviceBuffer[DType.float32], outs_addr: Int, nouts: Int, store: DeviceStore,
) raises:
    """Enqueue every slot output quad (CNT <= -2, lane vdf-tail-a): arena
    words [lo, hi) device to device into the first hi - lo words of store
    slot -CNT - 2 (refused when the slot holds fewer). No wait: the runner's
    final synchronize covers them, before the arena is released.
    `check_out_ranges(..., slots=True)` must have passed."""
    if nouts <= 0:
        return
    var o = ArenaIP(unsafe_from_address=outs_addr)
    for k in range(nouts):  # small-loop(nouts: arena output ranges): one copy enqueue per declared slot range
        var cn = Int(o.unsafe_load(OUT_INTS * k + 2))
        if cn > -2:
            continue
        var lo = Int(o.unsafe_load(OUT_INTS * k))
        var hi = Int(o.unsafe_load(OUT_INTS * k + 1))
        if hi > lo:
            var view = df.create_sub_buffer[DType.float32](lo, hi - lo)
            store.copy_from(ctx, -cn - 2, view, hi - lo)
            _ = view^  # the queued copy holds its own reference


# TOMBSTONE: `upload_ranges_host` (lane fg-knn-nb G3, MOJOLEARN_XPREP_NO_SLOT_HOP: host spans
# straight into the arena) deleted 2026-10-09 (lane postmerge-act-2): slower on the average,
# gaussian-nb istella NV 1.25x / AMD 1.00x, taxi 1.02x / 1.03x (nv n0630, amd a1090),
# same digests; code recoverable at main 5c137b55e.
