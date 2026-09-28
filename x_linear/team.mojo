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
from std.ffi import external_call
from std.memory import bitcast
from x_linear.ops import FP

#: A team's stored pointers (x_linear/team.mojo `Team`).
comptime TP = MutPointer[Float32, MutUntrackedOrigin]

#: Threads per block of a team fit on the device.
comptime LINEAR_TPB = 256
#: Broadcast slots of a team.
comptime TEAM_SLOTS = 16
#: Row scratch of a team fit: at least TEAM_ROW_BUFS float32 buffers of n
#: words (x_linear/dispatch.mojo `team_rows` names more for a fit that needs
#: them); both bindings allocate them.
comptime TEAM_ROW_BUFS = 3


def team_work(n: Int, bufs: Int, per_thread: Int) -> Int:
    """Float32 words of a team's scratch: the slots, `bufs` row buffers of n
    words, then `per_thread` words for each of LINEAR_TPB threads."""
    return TEAM_SLOTS + max(bufs, TEAM_ROW_BUFS) * n + LINEAR_TPB * per_thread


@always_inline
def team_barrier():
    """A block barrier that also orders DEVICE memory. A team shares its
    values through DEVICE memory (the fit's fw/res/iw buffers and the team
    scratch, broadcast slots included), never threadgroup memory. On NVIDIA `barrier()` is
    `bar.sync`, which orders global memory within the block, and on AMD it is
    `s_barrier` between workgroup-scope release/acquire fences, which cover
    global memory too. On Apple `barrier()` lowers to
    `air.wg.barrier(2, 1)`, Metal's `threadgroup_barrier(mem_threadgroup)`:
    it orders THREADGROUP memory only, so a thread could read a device word
    another thread stored before the barrier as its old value. The Metal
    column of every x_linear team fit disagreed with the CPU, CUDA and pass-1
    columns at 89aec9ed1 (x-lasso-lars, x-lasso-lars-pos, x-logistic-cv,
    x-bayes-ridge, ...). Here Apple gets `air.wg.barrier(3, 1)`,
    `threadgroup_barrier(mem_device | mem_threadgroup)`."""
    comptime if is_apple_gpu():
        external_call["air.wg.barrier", NoneType](Int32(3), Int32(1))
    else:
        barrier()


@fieldwise_init
struct Team(ImplicitlyCopyable, Movable):
    var tid: Int
    var nt: Int
    # POINTERS, not integer addresses (lane/linear-apple, 2026-09-28): a
    # pointer rebuilt from an integer (`FP(unsafe_from_address=...)`) is an
    # inttoptr into the GENERIC address space, which Metal's AIR does not
    # have; there every team row, slot and private word read or wrote the
    # wrong memory, and every x_linear team fit's Metal column disagreed
    # (SGD included, whose single problem runs on thread 0 alone). Kernel
    # pointer arguments keep their device address space through `+`.
    # (MutUntrackedOrigin: a struct field cannot expose AnyOrigin.)
    var slot_at: TP  # TEAM_SLOTS float32 words for broadcasts
    var rows_at: TP  # row buffers of n words: row(k)
    var n: Int
    var own_at: TP  # this thread's private words (team_work's per_thread)

    @always_inline
    def lead(self) -> Bool:
        return self.tid == 0

    @always_inline
    def sync(self):
        comptime if is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu():
            if self.nt > 1:
                team_barrier()

    @always_inline
    def bcast(self, v: Float32, k: Int = 0) -> Float32:
        """The lead thread's `v` in every thread (slot k)."""
        if self.nt == 1:
            return v
        var slot = self.slot_at.unsafe_origin_cast[MutAnyOrigin]()
        if self.lead():
            slot.unsafe_store(k, v)
        self.sync()
        var r = slot.unsafe_load(k)
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
        return self.rows_at.unsafe_origin_cast[MutAnyOrigin]() + k * self.n

    @always_inline
    def own(self) -> FP:
        return self.own_at.unsafe_origin_cast[MutAnyOrigin]()


def team_at(tid: Int, nt: Int, scratch: FP, n: Int, bufs: Int, per_thread: Int) -> Team:
    """The team over `scratch` (team_work(n, bufs, per_thread) words)."""
    var rw = scratch + TEAM_SLOTS
    var own = rw + max(bufs, TEAM_ROW_BUFS) * n + tid * per_thread
    return Team(
        tid, nt, scratch.unsafe_origin_cast[MutUntrackedOrigin](),
        rw.unsafe_origin_cast[MutUntrackedOrigin](), n,
        own.unsafe_origin_cast[MutUntrackedOrigin](),
    )


def device_team(scratch: FP, n: Int, bufs: Int, per_thread: Int) -> Team:
    return team_at(Int(thread_idx.x), Int(block_dim.x), scratch, n, bufs, per_thread)


def solo(scratch: FP, n: Int, bufs: Int, per_thread: Int) -> Team:
    return team_at(0, 1, scratch, n, bufs, per_thread)
