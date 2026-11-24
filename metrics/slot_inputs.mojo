# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident inputs of the `_mojolearn_metrics` kernel entries (lane
vdf-xm-kernels, 2026-11-24; addendum 3 of the vd-functions briefs).

An input of a metric entry is either a HOST address (uploaded once, by
view of the caller's words, as the host-list entries of
`metrics/estimator.mojo` do) or a SLOT of the resident store the
`resident()` scope and the `DeviceRows` gather use
(python/mojolearn/_arena_io.py: the x_prep image's store,
`x_prep/store_slot.mojo`): the slot's device words are read IN PLACE, as a
non-owning view, never downloaded and never uploaded again.

ONE CONTEXT. A store buffer belongs to the context that made it
(core/device_store.mojo), so every call with at least one slot runs ALL of
its launches on the store's context (`x_prep_ctx()`); its host inputs are
uploaded on that context too. The kernels are the hand-tuned
metrics/impl entries, unchanged, on the same words: the bits are the bits
of the host-input entries.

THE VALUE CHECKS. The host-list entries scan their lists on the host
(finite floats, encoded label ranges, the binary labels). A slot's words
never visit the host, so this form runs the SAME checks on the device: one
grid-stride launch per input that stores the constant 1 into a flag word
(no atomics, no order; core/admitted_finite_status.mojo's rule) and ONE
read of eight status words per call. The refusal words are the host
entries' words.
"""
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast

from core.device_zero import enqueue_fill
from core.transfer import transfer_upload_ptr
from x_prep.store_slot import X_PREP_STORE, x_prep_ctx

comptime MetricSlotF32 = MutPointer[Float32, MutAnyOrigin]
comptime MetricSlotI32 = MutPointer[Int32, MutAnyOrigin]
comptime _HostF32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _HostI32 = MutPointer[Int32, MutUntrackedOrigin]

comptime SCAN_TPB = 256
comptime SCAN_MAX_BLOCKS = 65535
#: status words of one call: inputs scan into distinct words (see the callers)
comptime SCAN_WORDS = 8


def _scan_grid(total: Int) -> Int:
    var b = (total + SCAN_TPB - 1) // SCAN_TPB
    if b < 1:
        return 1
    if b > SCAN_MAX_BLOCKS:
        return SCAN_MAX_BLOCKS
    return b


def _scan_f32_kernel(src: MetricSlotF32, n: Int64, unit: Int32, status: MetricSlotI32, word: Int32):
    """status[word] = 1 when a value is not finite (exponent bits all set),
    or, with `unit`, lies outside [0, 1]. One thread per value (grid-stride);
    every flag store writes the constant 1."""
    var step = Int(grid_dim.x) * Int(block_dim.x)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var exp = UInt32(0x7F800000)
    while t < Int(n):
        var v = src.unsafe_load(t)
        var bits = bitcast[DType.uint32](v)
        if (bits & exp) == exp:
            status.unsafe_store(Int(word), Int32(1))
        elif unit != 0 and (v < 0 or v > 1):
            status.unsafe_store(Int(word), Int32(1))
        t += step


def _scan_i32_kernel(src: MetricSlotI32, n: Int64, lo: Int32, hi: Int32, status: MetricSlotI32, word: Int32):
    """status[word] = 1 when a label lies outside [lo, hi); status[word + 1]
    / [word + 2] = 1 when a 0 / a 1 occurs (the binary ranking's both
    classes). One thread per label (grid-stride); constant-1 stores only."""
    var step = Int(grid_dim.x) * Int(block_dim.x)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while t < Int(n):
        var v = src.unsafe_load(t)
        if v < lo or v >= hi:
            status.unsafe_store(Int(word), Int32(1))
        if v == 0:
            status.unsafe_store(Int(word) + 1, Int32(1))
        elif v == 1:
            status.unsafe_store(Int(word) + 2, Int32(1))
        t += step


struct MetricInputs(Movable):
    """The inputs of ONE metric entry call on the store's context: device
    views of slots (each slot's buffer reference held until the call ends,
    so a concurrent free cannot drop it under a launch), uploads of host
    words, and the call's status words. Keep it alive past the last launch
    that reads one of its views (`_ = inputs^` after the result read)."""
    var ctx: DeviceContext
    var held: List[DeviceBuffer[DType.float32]]
    var uploads_i: List[DeviceBuffer[DType.int32]]
    var status: List[DeviceBuffer[DType.int32]]

    def __init__(out self) raises:
        self.ctx = x_prep_ctx()
        self.held = List[DeviceBuffer[DType.float32]]()
        self.uploads_i = List[DeviceBuffer[DType.int32]]()
        self.status = List[DeviceBuffer[DType.int32]]()

    def _slot_address(mut self, slot: Int, n: Int) raises -> Int:
        # The slot's buffer reference, checked live and holding >= n words
        # under the store's lock (`DeviceStore._held`), kept for the call.
        if n < 1:
            raise Error("metrics: n must be positive, got " + String(n))
        var buf = X_PREP_STORE.get_or_create_ptr()[]._held(slot, n)
        var address = Int(buf.unsafe_ptr())
        self.held.append(buf^)
        return address

    def f32(mut self, slot: Int, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
        """`n` float32 words: slot `slot` read in place (slot >= 0), else the
        host words at `addr` uploaded once on this context."""
        if slot >= 0:
            var address = self._slot_address(slot, n)
            return DeviceBuffer[DType.float32](self.ctx, MetricSlotF32(unsafe_from_address=address),
                                               n if n > 0 else 1, owning=False)
        if n > 0 and addr == 0:
            raise Error("metrics: null input address")
        return transfer_upload_ptr[DType.float32](self.ctx, _HostF32(unsafe_from_address=addr), n, n if n > 0 else 1)

    def i32(mut self, slot: Int, addr: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
        """`n` int32 words (a slot holds 4-byte words: int32 bits are its
        bytes); as `f32`."""
        if slot >= 0:
            var address = self._slot_address(slot, n)
            return DeviceBuffer[DType.int32](self.ctx, MetricSlotI32(unsafe_from_address=address),
                                             n if n > 0 else 1, owning=False)
        if n > 0 and addr == 0:
            raise Error("metrics: null input address")
        return transfer_upload_ptr[DType.int32](self.ctx, _HostI32(unsafe_from_address=addr), n, n if n > 0 else 1)

    def _status_address(mut self) raises -> Int:
        if len(self.status) == 0:
            var st = self.ctx.enqueue_create_buffer[DType.int32](SCAN_WORDS)
            enqueue_fill(self.ctx, st, Int32(0))
            self.status.append(st^)
        return Int(self.status[0].unsafe_ptr())

    def scan_f32(mut self, buf: DeviceBuffer[DType.float32], n: Int, unit: Bool, word: Int) raises:
        """Queue the finite (and, with `unit`, [0, 1]) check of `n` values
        into status word `word`."""
        if n < 1:
            return
        var st = self._status_address()
        self.ctx.enqueue_function[_scan_f32_kernel](
            MetricSlotF32(unsafe_from_address=Int(buf.unsafe_ptr())), Int64(n), Int32(1 if unit else 0),
            MetricSlotI32(unsafe_from_address=st), Int32(word),
            grid_dim=_scan_grid(n), block_dim=SCAN_TPB,
        )

    def scan_i32(mut self, buf: DeviceBuffer[DType.int32], n: Int, lo: Int, hi: Int, word: Int) raises:
        """Queue the [lo, hi) check of `n` labels into status word `word`,
        with the 0 / 1 presence flags in `word + 1` / `word + 2`."""
        if n < 1:
            return
        var st = self._status_address()
        self.ctx.enqueue_function[_scan_i32_kernel](
            MetricSlotI32(unsafe_from_address=Int(buf.unsafe_ptr())), Int64(n), Int32(lo), Int32(hi),
            MetricSlotI32(unsafe_from_address=st), Int32(word),
            grid_dim=_scan_grid(n), block_dim=SCAN_TPB,
        )

    def flags(mut self) raises -> List[Int32]:
        """The SCAN_WORDS status words after every queued scan: one read."""
        var out = List[Int32](length=SCAN_WORDS, fill=Int32(0))
        if len(self.status) == 0:
            return out^
        var st = self.status[0]
        var h = self.ctx.enqueue_create_host_buffer[DType.int32](SCAN_WORDS)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=st)
        self.ctx.synchronize()
        for i in range(SCAN_WORDS):  # small-loop(SCAN_WORDS: eight status words): one read-back
            out[i] = h.unsafe_ptr()[i]
        _ = h^
        _ = st^
        return out^

    def keep_i32(mut self, var buf: DeviceBuffer[DType.int32]):
        """Hold a buffer the call made (an output) until the call ends."""
        self.uploads_i.append(buf^)
