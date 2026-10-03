# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One arena allocation per Mamba block call, carved into sub-buffer views
(lane afn-mamba, 2026-10-03; `-D MOJOLEARN_AFN_MAMBA_ARENA`).

On Metal every live, separately allocated buffer is made resident on every
command encoder: a kernel launch costs about 0.25 us more per live
allocation (core/device_arena.mojo's measurement), and each `mamba_zeros`
of the stage structs also WAITS once (`mamba_zeros[wait=True]`, the Mamba-2
and Mamba-3 default), so a Mamba-3 forward paid about fifty allocations
and fifty waits before its first kernel. Here a block call takes ONE
`enqueue_create_buffer` of the summed size, fills it once (one launch),
and every weight, state piece, stage and the input are
`create_sub_buffer` views of it. The bytes a kernel reads and writes are
the same; only where a buffer's storage sits moves.

Lifetime: a view does not own the arena. The owner (the binding's run
function) keeps the `MambaArena` alive past the last use of every struct
built from it (`_ = arena^` after `_ = stages^`), exactly as
core/device_arena.mojo keeps its chunks alive under live views.

Guard band: `guard` extra floats follow every view's logical length, as
`mamba_device_alloc` gives every Mamba allocation under the poison build;
in production `guard == 0`. Under the poison define every view's body is
zero-filled on its own (the poison build is never timed).
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

#: a view's alignment, in floats (256 B)
comptime AFN_ARENA_ALIGN = 64
comptime _AFN_POISON = is_defined["MOJOLEARN_MAMBA_POISON"]()


struct MambaArena(Movable):
    var base: DeviceBuffer[DType.float32]
    var cap: Int
    var off: Int
    var guard: Int
    var views: Int

    @staticmethod
    def slot(n: Int, guard: Int) -> Int:
        """The arena floats one view of logical length `n` consumes."""
        var want = n
        if want < 1:
            want = 1
        want += guard
        return ((want + AFN_ARENA_ALIGN - 1) // AFN_ARENA_ALIGN) * AFN_ARENA_ALIGN

    @staticmethod
    def total(sizes: List[Int], guard: Int) -> Int:
        """The arena floats a list of logical lengths consumes."""
        var t = 0
        for i in range(len(sizes)):
            t += Self.slot(sizes[i], guard)
        return t

    def __init__(out self, ctx: DeviceContext, total: Int, guard: Int) raises:
        var cap = total
        if cap < AFN_ARENA_ALIGN:
            cap = AFN_ARENA_ALIGN
        self.base = ctx.enqueue_create_buffer[DType.float32](cap)
        self.cap = cap
        self.off = 0
        self.guard = guard
        self.views = 0
        comptime if _AFN_POISON:
            self.base.enqueue_fill(bitcast[DType.float32](UInt32(0x7FC00000)))
        else:
            # ONE fill for every zero-seeded stage of the call (a launch,
            # no wait; every reader is enqueued after it on the same
            # in-order context).
            self.base.enqueue_fill(Float32(0.0))

    def take(mut self, ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
        """A view of `max(n, 1) + guard` floats, 256 B aligned."""
        var want = n
        if want < 1:
            want = 1
        var need = Self.slot(n, self.guard)
        if self.off + need > self.cap:
            raise Error(
                "MambaArena.take: arena of "
                + String(self.cap)
                + " floats cannot hold a view of "
                + String(want)
                + " at offset "
                + String(self.off)
                + " (the size list and the constructor disagree)"
            )
        var view = self.base.create_sub_buffer[DType.float32](
            self.off, want + self.guard
        )
        comptime if _AFN_POISON:
            # The view's BODY is zero (mamba_zeros's meaning); the band
            # stays NaN. The poison build is never timed: wait here so the
            # local sub-buffer outlives its fill.
            var body = view.create_sub_buffer[DType.float32](0, want)
            body.enqueue_fill(Float32(0.0))
            ctx.synchronize()
            _ = body^
        self.off += need
        self.views += 1
        return view^
