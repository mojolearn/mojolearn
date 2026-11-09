# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A session's device buffers carved from a few arena chunks (lane
neural-pass43, 2026-10-01).

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

So while a session CONSTRUCTS (`arena_begin` .. `arena_end`), the allocation
helpers (`_zeros` / `_upload` of modeling_llama, transformer_backward and
train_loop) take views of this module's chunks instead of fresh buffers.
The chunks live in a process global; `arena_release` marks a session's
chunks free for the next session of any shape and never frees them under
live views (the views die with their owner's fields, after `__deinit__`
runs). The bytes a kernel reads and writes are the same; only where a
buffer's storage sits moves. Default on for the Apple column, off
elsewhere; `MOJOLEARN_DEVICE_ARENA=0/1` forces either.
"""
from std.ffi import _Global
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.ctx_key import ctx_cache_key
from core.shared_slot import shared_slot_check, shared_slot_word

#: Layout version of `_ArenaPool` (core/shared_slot.mojo rule). V2: the
#: layout word and `chunk_ctx`, the context key a chunk was created on.
comptime _ARENA_LAYOUT = 2

#: a chunk's floats (256 MB) and a view's alignment (256 B)
comptime ARENA_CHUNK_FLOATS = 1 << 26
comptime ARENA_ALIGN_FLOATS = 64


struct _ArenaPool(Defaultable, Movable):
    # first field: the layout word (core/shared_slot.mojo)
    var layout: Int
    var chunks: List[DeviceBuffer[DType.float32]]
    var chunk_n: List[Int]
    # the `ctx_cache_key` of the context each chunk was created on: the
    # arena is one slot per process and tier, reached by every neural
    # binding, and each binding runs its own context (its own stream or
    # queue), so a chunk is handed only to a session on the same context
    var chunk_ctx: List[Int]
    var owner: List[Int]
    var active: Int
    var cur: Int
    var off: Int
    var next_id: Int
    var views: Int

    def __init__(out self):
        self.layout = shared_slot_word[_ArenaPool](_ARENA_LAYOUT)
        self.chunks = List[DeviceBuffer[DType.float32]]()
        self.chunk_n = List[Int]()
        self.chunk_ctx = List[Int]()
        self.owner = List[Int]()
        self.active = -1
        self.cur = -1
        self.off = 0
        self.next_id = 1
        self.views = 0


comptime _ARENA_NAME_IDN: StaticString = "MojolearnDeviceArenaIdenticalV2"
comptime _ARENA_NAME_OTHER: StaticString = "MojolearnDeviceArenaFastV2"
# typed StaticString arms: a conditional of two literals is a String value,
# which shared_slot_check (StaticString) does not take
comptime _ARENA_NAME: StaticString = _ARENA_NAME_IDN if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else _ARENA_NAME_OTHER
comptime DEVICE_ARENA = _Global[StorageType=_ArenaPool, name=_ARENA_NAME, init_fn=_ArenaPool.__init__]


def _arena() raises -> UnsafePointer[_ArenaPool, MutAnyOrigin]:
    """The process-wide arena, refused when its creator had another layout."""
    var pool = DEVICE_ARENA.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    shared_slot_check(pool[].layout, shared_slot_word[_ArenaPool](_ARENA_LAYOUT), _ARENA_NAME)
    return pool


def device_arena_on() -> Bool:
    """Whether sessions carve their buffers from arenas: the Apple column by
    default, off elsewhere (CUDA and HIP launches do not pay per live
    buffer); MOJOLEARN_DEVICE_ARENA=0/1 forces either."""
    var v = String(getenv("MOJOLEARN_DEVICE_ARENA"))
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        return v != "0"
    return v == "1"


def arena_begin() raises -> Int:
    """Start a session's arena: the allocation helpers take views from it
    until `arena_end`. Returns the arena id (-1 when arenas are off). A
    construction that raised without ending is simply superseded."""
    if not device_arena_on():
        return -1
    var pool = _arena()
    var id = pool[].next_id
    pool[].next_id += 1
    pool[].active = id
    pool[].cur = -1
    pool[].off = 0
    return id


def arena_end(id: Int) raises:
    """Stop taking views for arena `id` (its chunks stay owned)."""
    if id < 0:
        return
    var pool = _arena()
    if pool[].active == id:
        pool[].active = -1
        pool[].cur = -1
        pool[].off = 0


def arena_release(id: Int) raises:
    """Mark arena `id`'s chunks free for the next session. The chunks are
    not freed: views of them may still be alive until their owner's fields
    are destroyed, and the next session reuses them."""
    if id < 0:
        return
    var pool = _arena()
    for i in range(len(pool[].chunks)):  # small-loop(chunks: arena chunks): a handful of grow-only device chunks, no data
        if pool[].owner[i] == id:
            pool[].owner[i] = -1
    if pool[].active == id:
        pool[].active = -1
        pool[].cur = -1
        pool[].off = 0


def arena_active() raises -> Bool:
    """Whether an allocation helper should take a view right now."""
    if not device_arena_on():
        return False
    return _arena()[].active >= 0


def arena_take(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A view of `n` floats (at least 1) in the active arena: the current
    chunk when it fits, else a free chunk that fits, else a new chunk of
    max(n, ARENA_CHUNK_FLOATS)."""
    var pool = _arena()
    var id = pool[].active
    if id < 0:
        raise Error("device arena: no active arena")
    var want = n
    if want < 1:
        want = 1
    var need = ((want + ARENA_ALIGN_FLOATS - 1) // ARENA_ALIGN_FLOATS) * ARENA_ALIGN_FLOATS
    var key = ctx_cache_key(ctx)
    if pool[].cur < 0 or pool[].off + need > pool[].chunk_n[pool[].cur] or pool[].chunk_ctx[pool[].cur] != key:
        var pick = -1
        for i in range(len(pool[].chunks)):  # small-loop(chunks: arena chunks): a handful of grow-only device chunks, no data
            if pool[].owner[i] < 0 and pool[].chunk_n[i] >= need and pool[].chunk_ctx[i] == key:
                pick = i
                break
        if pick < 0:
            var size = need
            if size < ARENA_CHUNK_FLOATS:
                size = ARENA_CHUNK_FLOATS
            pool[].chunks.append(ctx.enqueue_create_buffer[DType.float32](size))
            pool[].chunk_n.append(size)
            pool[].chunk_ctx.append(key)
            pool[].owner.append(id)
            pick = len(pool[].chunks) - 1
        else:
            pool[].owner[pick] = id
        pool[].cur = pick
        pool[].off = 0
    var view = pool[].chunks[pool[].cur].create_sub_buffer[DType.float32](pool[].off, want)
    pool[].off += need
    pool[].views += 1
    return view^


def arena_stats() raises -> List[Int]:
    """[chunks, chunks in use, floats allocated, views taken] for a probe."""
    var pool = _arena()
    var used = 0
    var floats = 0
    for i in range(len(pool[].chunks)):  # small-loop(chunks: arena chunks): a handful of grow-only device chunks, no data
        floats += pool[].chunk_n[i]
        if pool[].owner[i] >= 0:
            used += 1
    var out = List[Int]()
    out.append(len(pool[].chunks))
    out.append(used)
    out.append(floats)
    out.append(pool[].views)
    return out^
