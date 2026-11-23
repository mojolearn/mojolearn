# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A session's device buffers carved from a few arena chunks (lane
neural-pass43, 2026-10-01; per-session handle since lane arena-session,
2026-11-23).

On Metal every live, separately allocated buffer is made resident on every
command encoder, so the host cost of a kernel launch grows with the number
of live allocations: measured on the M4 (2026-09-25) at about 0.25 us per
live DeviceBuffer per launch, 26 us with none and 270 us with 1,000, while
1,000 `create_sub_buffer` views of ONE allocation cost nothing. A resident
byte-LM session holds about 1,300 buffers (per layer 30 forward stages, 62
weight buffers and 48 backward stages, plus the trainer's own), and a train
step launches about 560 kernels: on the M3 Ultra the same attention kernel
read 4.6 ms inside a lone block and 11.6 ms inside the session, and every
small stage read 5-20x its AMD time.

So a stateful owner (a byte-LM trainer, its logits scratch slot, a Samba
session's Mamba-3 backward) owns a `DeviceArena` and hands the allocation
helpers (`_zeros` / `_upload` of modeling_llama, transformer_backward and
train_loop, `_m3_scratch`) a pointer to it (`arena_ptr`): those calls take
views of the arena's chunks instead of fresh buffers. A call without the
pointer (every caller with no such owner) allocates a fresh buffer and
never looks at any arena. Two owners at the same time each carve their own
arena; an owner rebuilt every step rewinds its arena (no lock).

Until 2026-11-23 the arena was ONE process-wide "active session" that every
helper consulted on every call (a slot lookup per allocation): two sessions
constructing at once, or one session's step allocating while another
constructed, took views of each other's arena and released each other's
chunks. That global, its session ids, `arena_begin` / `arena_end` /
`arena_active` / `arena_take` and the per-call lookup are deleted.

The CHUNKS stay in a process pool, locked only when an arena grows or
returns its chunks (an owner's construction, its first pass at a larger
shape, its teardown; never a rewound step): a view does not own its chunk (mamba/impl/modules/afn_arena.mojo's
lifetime rule), so a chunk is never freed while a view of it may be alive;
an arena's release marks its chunks free for the next arena on the same
context (the same stream, so the next arena's fills are ordered after the
last session's work). The bytes a kernel reads and writes are the same;
only where a buffer's storage sits moves. On for the Apple column, off
elsewhere; `-D MOJOLEARN_DEVICE_ARENA_OFF` turns it off on Apple.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.ctx_key import ctx_cache_key
from core.shared_slot import SharedSlot, shared_slot_unlock, shared_slot_wait_lock
from core.step_phase import step_count_device_alloc

#: Layout version of `_ChunkPool` (core/shared_slot.mojo rule). V4: the
#: chunk pool alone (no active session, owner ids or view count) plus its
#: lock word.
comptime _ARENA_LAYOUT = 4

#: a chunk's floats (256 MB) and a view's alignment (256 B)
comptime ARENA_CHUNK_FLOATS = 1 << 26
comptime ARENA_ALIGN_FLOATS = 64


struct _ChunkPool(Defaultable, Movable):
    var chunks: List[DeviceBuffer[DType.float32]]
    var chunk_n: List[Int]
    # the `ctx_cache_key` of the context each chunk was created on: the
    # pool is one slot per process and tier, reached by every neural
    # binding, and each binding runs its own context (its own stream or
    # queue), so a chunk is handed only to an arena on the same context
    var chunk_ctx: List[Int]
    var busy: List[Bool]
    #: the lock word (0 free, 1 held), taken around every List step
    var lock: List[Int64]

    def __init__(out self):
        self.chunks = List[DeviceBuffer[DType.float32]]()
        self.chunk_n = List[Int]()
        self.chunk_ctx = List[Int]()
        self.busy = List[Bool]()
        self.lock = List[Int64](length=1, fill=Int64(0))

    def word(self) -> MutPointer[Int64, MutUntrackedOrigin]:
        return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(self.lock.unsafe_ptr()))


comptime _ARENA_NAME_IDN: StaticString = "MojolearnDeviceArenaChunksIdenticalV4"
comptime _ARENA_NAME_OTHER: StaticString = "MojolearnDeviceArenaChunksFastV4"
comptime _ARENA_NAME: StaticString = _ARENA_NAME_IDN if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else _ARENA_NAME_OTHER
comptime _ARENA_CHUNKS = SharedSlot[_ChunkPool, _ARENA_NAME, _ARENA_LAYOUT]


# TOMBSTONE: MOJOLEARN_DEVICE_ARENA (NVIDIA/AMD opt-in arm, no A/B recorded in four weeks: the run-time switch since 2026-10-01, a build define since 2026-11-15) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
# Restore: git apply experiments/removed/MOJOLEARN_DEVICE_ARENA.patch; record in docs/TOMBSTONES.md.
comptime DEVICE_ARENA_ON = TARGET_COLUMN == COLUMN_APPLE and not is_defined["MOJOLEARN_DEVICE_ARENA_OFF"]()
"""Whether sessions carve their buffers from arenas: the Apple column, off
elsewhere (CUDA and HIP launches do not pay per live buffer). A build
constant since lane envgate-neural-lm (2026-11-15). `-D
MOJOLEARN_DEVICE_ARENA_OFF` turns it off on Apple."""


def _chunk_acquire(ctx: DeviceContext, key: Int, need: Int, mut index: Int) raises -> DeviceBuffer[DType.float32]:
    """A free pool chunk of at least `need` floats on context `key`, else a
    new chunk of max(need, ARENA_CHUNK_FLOATS), marked busy; `index` is its
    pool index. The returned buffer is a reference to the chunk, copied out
    under the lock (the List may grow under another arena right after)."""
    var pool = _ARENA_CHUNKS.get_or_create_ptr()
    var word = pool[].word()
    shared_slot_wait_lock(word)
    try:
        for i in range(len(pool[].chunks)):  # small-loop(chunks: arena chunks): a handful of grow-only device chunks, no data
            if not pool[].busy[i] and pool[].chunk_n[i] >= need and pool[].chunk_ctx[i] == key:
                pool[].busy[i] = True
                index = i
                return pool[].chunks[i]
        var size = need
        if size < ARENA_CHUNK_FLOATS:
            size = ARENA_CHUNK_FLOATS
        pool[].chunks.append(ctx.enqueue_create_buffer[DType.float32](size))
        pool[].chunk_n.append(size)
        pool[].chunk_ctx.append(key)
        pool[].busy.append(True)
        index = len(pool[].chunks) - 1
        return pool[].chunks[index]
    finally:
        shared_slot_unlock(word)


def _chunks_release(held: List[Int]) raises:
    """Mark the pool chunks `held` free. Never frees one: views of it may
    still be alive until their owner's fields are destroyed."""
    var pool = _ARENA_CHUNKS.get_or_create_ptr()
    var word = pool[].word()
    shared_slot_wait_lock(word)
    try:
        for j in range(len(held)):  # small-loop(held: an arena's chunks): a handful of chunk indices, no data
            pool[].busy[held[j]] = False
    finally:
        shared_slot_unlock(word)


struct DeviceArena(Movable):
    """One owner's arena: the pool chunks it holds and its fill point. The
    owner (a session, a trainer, a block's driver) constructs it, passes
    `arena_ptr(arena)` to the helpers that build its buffers, and keeps it
    as a field. An owner whose buffers are rebuilt every step `rewind`s it
    instead of releasing it: the next pass carves the same chunks again from
    their start, with no pool lock (the chunks stay on the owner's context,
    so the next pass's writes queue after the last pass's work on the same
    stream). The pool lock is taken only when the arena GROWS (a take that
    no held chunk fits) and when it is released or destroyed. Off (`on`
    False: every column but Apple, or `-D MOJOLEARN_DEVICE_ARENA_OFF`) it
    holds nothing and `arena_ptr` gives None."""

    var on: Bool
    #: pool indices of the chunks this arena holds, in the order taken
    var held: List[Int]
    #: references to those chunks, their floats and their context keys
    var chunks: List[DeviceBuffer[DType.float32]]
    var chunk_n: List[Int]
    var chunk_key: List[Int]
    #: the chunk views are taken from (an index into `chunks`, -1: none yet)
    var cur: Int
    var off: Int
    var views: Int

    def __init__(out self):
        """An arena that is on exactly when DEVICE_ARENA_ON."""
        self.on = DEVICE_ARENA_ON
        self.held = List[Int]()
        self.chunks = List[DeviceBuffer[DType.float32]]()
        self.chunk_n = List[Int]()
        self.chunk_key = List[Int]()
        self.cur = -1
        self.off = 0
        self.views = 0

    def __init__(out self, *, on: Bool):
        """An arena switched by the caller (the native checks on the CPU
        column, where DEVICE_ARENA_ON is False)."""
        self.on = on
        self.held = List[Int]()
        self.chunks = List[DeviceBuffer[DType.float32]]()
        self.chunk_n = List[Int]()
        self.chunk_key = List[Int]()
        self.cur = -1
        self.off = 0
        self.views = 0

    def take(mut self, ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
        """A view of `n` floats (at least 1): the current chunk when it fits
        and is on `ctx`, else the next held chunk that does, else a new
        chunk from the pool (the only path that takes the pool lock)."""
        if not self.on:
            raise Error("device arena: take from an arena that is off")
        var want = n
        if want < 1:
            want = 1
        var need = ((want + ARENA_ALIGN_FLOATS - 1) // ARENA_ALIGN_FLOATS) * ARENA_ALIGN_FLOATS
        var key = ctx_cache_key(ctx)
        var fits = self.cur >= 0 and self.chunk_key[self.cur] == key and self.off + need <= self.chunk_n[self.cur]
        if not fits:
            var pick = -1
            for i in range(self.cur + 1, len(self.chunks)):  # small-loop(chunks: this arena's chunks): a handful of chunk handles, no data
                if self.chunk_key[i] == key and self.chunk_n[i] >= need:
                    pick = i
                    break
            if pick < 0:
                var index = -1
                var chunk = _chunk_acquire(ctx, key, need, index)
                self.held.append(index)
                self.chunk_n.append(len(chunk))
                self.chunk_key.append(key)
                self.chunks.append(chunk^)
                pick = len(self.chunks) - 1
            self.cur = pick
            self.off = 0
        var view = self.chunks[self.cur].create_sub_buffer[DType.float32](self.off, want)
        self.off += need
        self.views += 1
        return view^

    def rewind(mut self):
        """Carve the held chunks again from their start. The caller has
        finished with every view taken since the last rewind (its pass is
        over); no lock, no chunk changes hands."""
        self.cur = -1
        self.off = 0

    def release(mut self) raises:
        """Return this arena's chunks to the pool now (the views already
        taken stay valid: a chunk is never freed)."""
        if len(self.held) > 0:
            _chunks_release(self.held)
        self.held = List[Int]()
        self.chunks = List[DeviceBuffer[DType.float32]]()
        self.chunk_n = List[Int]()
        self.chunk_key = List[Int]()
        self.cur = -1
        self.off = 0

    def stats(self) -> List[Int]:
        """[chunks held, views taken] for a probe."""
        var stats = List[Int]()
        stats.append(len(self.held))
        stats.append(self.views)
        return stats^

    def __deinit__(deinit self):
        if len(self.held) > 0:
            try:
                _chunks_release(self.held)
            except:
                pass


comptime ArenaPtr = UnsafePointer[DeviceArena, MutAnyOrigin]
"""The handle the allocation helpers take (`Optional[ArenaPtr]`, None for
every call outside a session's construction)."""


def arena_ptr(mut arena: DeviceArena) -> Optional[ArenaPtr]:
    """The helpers' handle to `arena`: None when it is off. The caller keeps
    `arena` in place (a local or a field it does not move) and named past
    the last helper call that receives the handle."""
    if not arena.on:
        return None
    return UnsafePointer(to=arena).unsafe_origin_cast[MutAnyOrigin]()


def arena_buffer(ctx: DeviceContext, n: Int, arena: Optional[ArenaPtr]) raises -> DeviceBuffer[DType.float32]:
    """`n` floats: a view of `arena` when given, else a fresh buffer. The
    words are unspecified either way (the helpers fill them)."""
    if arena:
        var p = arena.value()
        return p[].take(ctx, n)
    step_count_device_alloc()
    return ctx.enqueue_create_buffer[DType.float32](n)
