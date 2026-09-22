# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A bump allocator of device and host SUB-BUFFERS over a few large parents.

NO REFERENCE FILE. CatBoost's `TCudaManager` hands buffers out of a
per-device memory pool (`cuda_lib/memory_pool.h`); this is the smallest
piece of that this repository needs, and it exists for one measured reason.

MEASURED ON THE M4 (2026-09-22, `enqueue_function` of a one-block kernel,
500 launches, nothing else in flight): 22 us a launch with up to a hundred
live device or host buffers in the context, 30 us at three hundred, 80 to
98 us at a thousand -- the Metal backend binds EVERY live allocation to
every dispatch (`-[AGXG16GFamilyComputeContext useResource:usage:]` is the
top of the enqueueing thread's profile). A thousand sub-buffers of ONE
parent cost the same 20 us as one buffer. The Ordered fit's batched
estimation keeps every task's buffers in flight at once, about a thousand
of them at the default four permutations; carved from an arena they are a
handful of parents.

Every sub-buffer starts on a 256-byte boundary and has EXACTLY the
requested length, so a whole-buffer copy to or from it moves the same
bytes a fresh allocation of that length would. Contents are NOT zeroed:
a caller may take from an arena only buffers whose every read cell it
writes first, the contract every pooled workspace in this tree already
states (DEVIATIONS 1890, 3041).

`reset` hands the same parents out again from the start: the caller must
have drained every use of every buffer taken since the last reset.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.sys.info import size_of

comptime ARENA_ALIGN = 256
comptime ARENA_MIN_CHUNK = 1 << 20


def _round_up(n: Int) -> Int:
    return (n + ARENA_ALIGN - 1) // ARENA_ALIGN * ARENA_ALIGN


struct BufferArena(Movable):
    """Device and host parents, filled front to back; a new parent at least
    twice everything handed out so far, so a fit holds O(log) parents."""

    var dev: List[DeviceBuffer[DType.uint8]]
    var dev_chunk: Int
    var dev_used: Int
    var dev_total: Int
    var host: List[HostBuffer[DType.uint8]]
    var host_chunk: Int
    var host_used: Int
    var host_total: Int

    def __init__(out self):
        self.dev = List[DeviceBuffer[DType.uint8]]()
        self.dev_chunk = 0
        self.dev_used = 0
        self.dev_total = 0
        self.host = List[HostBuffer[DType.uint8]]()
        self.host_chunk = 0
        self.host_used = 0
        self.host_total = 0

    def device[
        dtype: DType
    ](mut self, ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dtype]:
        """A device buffer of exactly `max(n, 1)` elements."""
        var count = n if n > 0 else 1
        var nbytes = _round_up(count * size_of[Scalar[dtype]]())
        while self.dev_chunk < len(self.dev) and (
            self.dev_used + nbytes > len(self.dev[self.dev_chunk])
        ):
            self.dev_chunk += 1
            self.dev_used = 0
        if self.dev_chunk == len(self.dev):
            var size = max(ARENA_MIN_CHUNK, max(nbytes, 2 * self.dev_total))
            self.dev.append(ctx.enqueue_create_buffer[DType.uint8](size))
            self.dev_total += size
            self.dev_used = 0
        var off = self.dev_used
        self.dev_used += nbytes
        return self.dev[self.dev_chunk].create_sub_buffer[dtype](
            off // size_of[Scalar[dtype]](), count
        )

    def host_buffer[
        dtype: DType
    ](mut self, ctx: DeviceContext, n: Int) raises -> HostBuffer[dtype]:
        """A host (staging) buffer of exactly `max(n, 1)` elements."""
        var count = n if n > 0 else 1
        var nbytes = _round_up(count * size_of[Scalar[dtype]]())
        while self.host_chunk < len(self.host) and (
            self.host_used + nbytes > len(self.host[self.host_chunk])
        ):
            self.host_chunk += 1
            self.host_used = 0
        if self.host_chunk == len(self.host):
            var size = max(ARENA_MIN_CHUNK, max(nbytes, 2 * self.host_total))
            self.host.append(ctx.enqueue_create_host_buffer[DType.uint8](size))
            self.host_total += size
            self.host_used = 0
        var off = self.host_used
        self.host_used += nbytes
        return self.host[self.host_chunk].create_sub_buffer[dtype](
            off // size_of[Scalar[dtype]](), count
        )

    def reset(mut self):
        """Hand the parents out again from the start (see the module
        docstring for the drain this needs)."""
        self.dev_chunk = 0
        self.dev_used = 0
        self.host_chunk = 0
        self.host_used = 0
