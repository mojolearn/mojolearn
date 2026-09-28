# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TEAM: one fit, run by a whole thread block (lane linear, speed phase).

Pass 1 ran every fit on ONE device thread. The speed phase runs a fit on ONE
BLOCK of `LINEAR_TPB` threads instead, and the host binding runs the same
source as a team of one. Bits do not move, by construction:

  * every stored value (an element of an output, a scratch cell) is computed
    by exactly ONE thread, by the same sequence of operations the one-thread
    schedule used for it: a fold over rows stays a single thread's loop over
    ascending rows; only INDEPENDENT outputs (the columns of a gradient, the
    cells of a Gram matrix, the rows of a linear predictor) are dealt out
    across threads (`for j in range(t.tid, m, t.nt)`);
  * a value one thread stored is read by another only after `t.sync()`;
  * a scalar that DECIDES a branch or a loop bound is either computed by every
    thread from memory no thread is writing (the same bits everywhere), or
    computed by the lead thread and broadcast (`bcast`), so control flow is
    uniform and every barrier is reached by the whole block.

On the host `nt == 1`, `sync` is nothing and `bcast` returns its argument:
the host runs exactly the pass-1 sequence. On the device a fit that has not
been converted still runs on thread 0 alone, with a team of one
(x_linear/device.mojo), so conversions land one fit at a time.
"""
from std.sys.info import is_amd_gpu, is_apple_gpu, is_nvidia_gpu
from std.gpu import thread_idx, block_dim
from max.gpu.sync import barrier
from std.memory import bitcast
from x_linear.ops import FP

#: Threads per block of a team fit on the device.
comptime LINEAR_TPB = 256
#: Broadcast slots of a team.
comptime TEAM_SLOTS = 16
#: Row scratch of a team fit: at least TEAM_ROW_BUFS float32 buffers of n
#: words (x_linear/dispatch.mojo `team_rows` names more for a fit that needs
#: them); both bindings allocate them.
comptime TEAM_ROW_BUFS = 3


def team_work(n: Int, bufs: Int) -> Int:
    """Float32 words of a team's scratch: the slots, then `bufs` row buffers."""
    return TEAM_SLOTS + max(bufs, TEAM_ROW_BUFS) * n


@fieldwise_init
struct Team(ImplicitlyCopyable, Movable):
    var tid: Int
    var nt: Int
    var slot: FP  # TEAM_SLOTS float32 words for broadcasts
    var rw: FP  # row buffers of n words: row(k) = rw + k * n
    var n: Int

    @always_inline
    def lead(self) -> Bool:
        return self.tid == 0

    @always_inline
    def sync(self):
        comptime if is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu():
            if self.nt > 1:
                barrier()

    @always_inline
    def bcast(self, v: Float32, k: Int = 0) -> Float32:
        """The lead thread's `v` in every thread (slot k)."""
        if self.nt == 1:
            return v
        if self.lead():
            self.slot.unsafe_store(k, v)
        self.sync()
        var r = self.slot.unsafe_load(k)
        self.sync()
        return r

    @always_inline
    def bcast_int(self, v: Int, k: Int = 0) -> Int:
        if self.nt == 1:
            return v
        var r = self.bcast(bitcast[DType.float32](Int32(v)), k)
        return Int(bitcast[DType.int32](r))

    @always_inline
    def row(self, k: Int) -> FP:
        return self.rw + k * self.n


def team_at(tid: Int, nt: Int, scratch: FP, n: Int) -> Team:
    """The team over `scratch` (team_work(n, bufs) words): slots, then row buffers."""
    return Team(tid, nt, scratch, scratch + TEAM_SLOTS, n)


def device_team(scratch: FP, n: Int) -> Team:
    return team_at(Int(thread_idx.x), Int(block_dim.x), scratch, n)


def solo(scratch: FP, n: Int) -> Team:
    return team_at(0, 1, scratch, n)
