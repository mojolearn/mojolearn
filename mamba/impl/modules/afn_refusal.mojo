# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Contract section 6's non-finite refusal with ONE readback per block call
(lane afn-mamba, 2026-10-03; `-D MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL`).

Main's Mamba-1 and Mamba-2 refusal downloads every named input to the host
(x on every call, the ten weights on the first call of a weights struct,
which the Python binding rebuilds on EVERY call, and the state) and scans
it there: thirteen device-to-host copies, thirteen waits and about 7 MB of
readback per forward at the board shape. The Mamba-3 refusal
(mamba3_refusal.mojo) reduces on the device but still reads back and waits
once per name.

Here every named buffer is one launch of a block-strided reduction into a
per-name row of partial codes; one final launch folds the rows into one
code per name; ONE copy and ONE wait bring the codes back. The code is
`2 * index + (0 for NaN, 1 for infinity)` (mamba3_refusal.mojo's scheme),
so the minimum is the first offending cell, and the raised text is
`_refuse_nonfinite_named`'s verbatim. Tested BY BITS, as the host spelling
is (row 49: Metal flushes compare operands).

Codes are Int32 (Metal has no 64-bit atomics and none are needed; a
buffer here is far below 2^30 elements, and a longer one is refused by
name rather than silently truncated).
"""
from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

comptime AFN_REFUSAL_THREADS = 256
comptime AFN_REFUSAL_BLOCKS = 32
comptime AFN_REFUSAL_NONE: Int32 = 0x7FFFFFFF


def afn_nonfinite_partial_kernel(
    part: MutPointer[Int32, MutAnyOrigin],  # [slots, AFN_REFUSAL_BLOCKS]
    values: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    slot_in: Int32,
):
    var red = stack_allocation[
        AFN_REFUSAL_THREADS, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var n = Int(n_in)
    var i = Int(block_idx.x) * AFN_REFUSAL_THREADS + tid
    var stride = Int(grid_dim.x) * AFN_REFUSAL_THREADS
    var best = AFN_REFUSAL_NONE
    while i < n:
        var bits = bitcast[DType.uint32](values.unsafe_load(i)) & UInt32(0x7FFFFFFF)
        if bits >= UInt32(0x7F800000):
            best = Int32(i) * 2
            if bits == UInt32(0x7F800000):
                best += 1
            break
        i += stride
    red.unsafe_store(tid, best)
    barrier()
    var active = AFN_REFUSAL_THREADS // 2
    while active > 0:
        if tid < active:
            var other = red.unsafe_load(tid + active)
            if other < red.unsafe_load(tid):
                red.unsafe_store(tid, other)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(
            Int(slot_in) * AFN_REFUSAL_BLOCKS + Int(block_idx.x), red.unsafe_load(0)
        )


def afn_nonfinite_fold_kernel(
    codes: MutPointer[Int32, MutAnyOrigin],  # [slots]
    part: MutPointer[Int32, MutAnyOrigin],  # [slots, AFN_REFUSAL_BLOCKS]
    n_slots_in: Int32,
):
    var s = Int(block_idx.x) * AFN_REFUSAL_THREADS + Int(thread_idx.x)
    if s >= Int(n_slots_in):
        return
    var best = AFN_REFUSAL_NONE
    for j in range(AFN_REFUSAL_BLOCKS):
        var c = part.unsafe_load(s * AFN_REFUSAL_BLOCKS + j)
        if c < best:
            best = c
    codes.unsafe_store(s, best)


struct AfnRefusalBatch(Movable):
    """The named buffers of one block call, reduced on the device, read
    back together by `finish`."""

    var part: DeviceBuffer[DType.int32]
    var codes: DeviceBuffer[DType.int32]
    var names: List[String]
    var cap: Int

    def __init__(out self, ctx: DeviceContext, cap: Int) raises:
        var c = cap
        if c < 1:
            c = 1
        self.cap = c
        self.part = ctx.enqueue_create_buffer[DType.int32](c * AFN_REFUSAL_BLOCKS)
        self.codes = ctx.enqueue_create_buffer[DType.int32](c)
        self.names = List[String]()
        # Every partial row is written by its launch; the fill covers the
        # rows of names never added (one launch, no wait).
        self.part.enqueue_fill(AFN_REFUSAL_NONE)

    def add(
        mut self,
        ctx: DeviceContext,
        name: String,
        mut values: DeviceBuffer[DType.float32],
        n: Int,
    ) raises:
        """One launch: the first non-finite cell of `values[:n]`, by name."""
        if n < 0 or n > len(values):
            raise Error("afn refusal: invalid buffer extent for " + name)
        if n >= (1 << 30):
            raise Error(
                "afn refusal: buffer '" + name + "' is too long for a 32-bit code"
            )
        if len(self.names) >= self.cap:
            raise Error("afn refusal: more names than the batch was sized for")
        var slot = len(self.names)
        self.names.append(name)
        if n == 0:
            return
        var blocks = (n + AFN_REFUSAL_THREADS - 1) // AFN_REFUSAL_THREADS
        if blocks > AFN_REFUSAL_BLOCKS:
            blocks = AFN_REFUSAL_BLOCKS
        ctx.enqueue_function[afn_nonfinite_partial_kernel](
            self.part.unsafe_ptr(),
            values.unsafe_ptr(),
            Int32(n),
            Int32(slot),
            grid_dim=(blocks, 1, 1),
            block_dim=(AFN_REFUSAL_THREADS, 1, 1),
        )

    def finish(mut self, ctx: DeviceContext) raises:
        """One fold launch, one copy, one wait; raises the first refusal in
        the order the names were added."""
        var n_slots = len(self.names)
        if n_slots == 0:
            return
        ctx.enqueue_function[afn_nonfinite_fold_kernel](
            self.codes.unsafe_ptr(),
            self.part.unsafe_ptr(),
            Int32(n_slots),
            grid_dim=((n_slots + AFN_REFUSAL_THREADS - 1) // AFN_REFUSAL_THREADS, 1, 1),
            block_dim=(AFN_REFUSAL_THREADS, 1, 1),
        )
        var host = ctx.enqueue_create_host_buffer[DType.int32](self.cap)
        var view = self.codes.create_sub_buffer[DType.int32](0, self.cap)
        ctx.enqueue_copy(dst_buf=host, src_buf=view)
        ctx.synchronize()
        _ = view^
        for s in range(n_slots):
            var code = host.unsafe_ptr().unsafe_load(s)
            if code == AFN_REFUSAL_NONE:
                continue
            var index = Int(code // 2)
            if code % 2 == 0:
                raise Error(
                    String("mamba: NaN in ")
                    + self.names[s]
                    + " at flat index "
                    + String(index)
                    + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
                    + " stage may record one)"
                )
            raise Error(
                String("mamba: infinity in ")
                + self.names[s]
                + " at flat index "
                + String(index)
                + " REFUSED (row 39)"
            )
        _ = host^
