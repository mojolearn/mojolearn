# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Model selection on the device (lane cpu2-l4-modelsel, 2026-10-04; re-audit
lane L4): the fold-row gather, the cross_val_predict scatter, the binary
scorers' probability column and the parallel forest's offset merge.

THE RESIDENT FOLD STORE. `MselStore` holds device byte buffers named by an
integer id (the `core/device_store.mojo` pattern, in bytes instead of 4-byte
words, so a uint8 or int8 y and an odd row width fit): cross-validation puts
X, y and every fold's train and test indices once, and each fold of each
candidate is one device gather out of them (`device_take_rows`), downloaded
straight into the Array the estimator is handed. A host gather per fold per
candidate is gone; X crosses the bus once per search instead of never being
reused. Freed deterministically (`free` waits for the context, then drops the
buffer; the id is reused), never left to a garbage collector.

Bytes move as 4-byte words when the row width allows it, else as bytes: a
gather or a scatter is a byte copy, so every vendor writes the same bytes.
The casts of the scatter and the column read are the integer word moves of
`core/msel_convert.mojo`, shared with the host column (`core/msel_host.mojo`).

No shared memory (no page to gate). Grid-stride loops over Int indices, so
no element count is bounded by Int32; the one atomic-free status word is
set by any thread that sees an out-of-range index.
"""

from std.atomic import Atomic
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.ctx_key import ctx_cache_key
from core.device_zero import enqueue_fill, enqueue_fill_n
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.shared_slot import shared_slot_lock, shared_slot_unlock, SharedSlot
from core.msel_convert import (
    MSEL_ID_SLOT_BITS,
    MSEL_ID_SLOT_MASK,
    msel_itemsize,
    msel_load_f32,
)

comptime MSEL_TPB = 256
comptime MSEL_MAX_BLOCKS = 65535

comptime _U8 = MutPointer[UInt8, MutAnyOrigin]
comptime _U32 = MutPointer[UInt32, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _I32 = MutPointer[Int32, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]
comptime _F32 = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _stride() -> Int:
    return Int(grid_dim.x) * Int(block_dim.x)


def _grid(total: Int) -> Int:
    var b = (total + MSEL_TPB - 1) // MSEL_TPB
    if b < 1:
        return 1
    if b > MSEL_MAX_BLOCKS:
        return MSEL_MAX_BLOCKS
    return b


def _status(ctx: DeviceContext, mut d_status: DeviceBuffer[DType.int32]) raises -> Int:
    """Synchronizes and returns word 0 of the status buffer."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
    ctx.synchronize()
    var v = Int(h.unsafe_ptr()[0])
    _ = h^
    return v


# ===========================================================================
# The resident store
# ===========================================================================


#: Layout version of `MselStore` (core/shared_slot.mojo rule; the store is
#: one process-wide `_Global` read by every binding, core/msel_device_x.mojo).
#: V2: the layout word. 4 (2026-11-22): the lock word. 5 (2026-11-22, lane
#: r2-known): the table behind a pointer.
comptime MSEL_STORE_LAYOUT = 5


struct _MselTable(Movable):
    """The store's shared state, in heap storage of its own (the `MselStore`
    note, THE TABLE IS BEHIND A POINTER)."""
    var bufs: List[DeviceBuffer[DType.uint8]]
    #: bytes held by each id; -1 for a freed id
    var nbytes: List[Int]
    var released: List[Int]
    #: generation of each slot, raised on every reuse (MSEL_ID_SLOT_BITS)
    var gens: List[Int]
    var live: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.uint8]]()
        self.nbytes = List[Int]()
        self.released = List[Int]()
        self.gens = List[Int]()
        self.live = 0

    def check_held(self, id: Int, need: Int) raises -> Int:
        """`MselStore.check` with the store's lock held by the caller."""
        # Returns the id's slot. An id whose slot was freed and handed out
        # again carries the old generation and is refused, never read as the
        # new holder's bytes (MSEL_ID_SLOT_BITS).
        var slot = id & MSEL_ID_SLOT_MASK
        if (id < 0 or slot >= len(self.nbytes) or self.nbytes[slot] < 0
                or self.gens[slot] != (id >> MSEL_ID_SLOT_BITS)):
            raise Error(String("msel store: id ", id, " is not live"))
        if self.nbytes[slot] < need:
            raise Error(String("msel store: id ", id, " holds ", self.nbytes[slot], " bytes, ", need, " read"))
        return slot


struct MselStore(Defaultable, Movable):
    """The process-wide resident store. Its operations run with the GIL
    released (cross_val_score workers, two Python threads each running a
    search), so the id lists are guarded by `lock` word 0: two threads
    appending, freeing or reading ids at once corrupted them ("msel store:
    id N is not live", "holds 432 bytes, 640 read"; proving suite
    2026-11-22). The lock covers the bookkeeping only; allocation, copies,
    launches and synchronization run outside it, on the id's own buffer,
    which no other thread frees (each id has one owner).

    THE TABLE IS BEHIND A POINTER (lane r2-known, 2026-11-22; the pattern of
    `core/forest_export_run.mojo`). The id lists live in `_MselTable`, the one
    element of the heap List `cell`, and every method reaches them through
    `_table()` after `shared_slot_lock`. As fields of the store read through
    an immutable `self`, the compiler may load them once before the lock and
    keep the snapshot; memory reached through a pointer the method loads is
    ordered by the acquire compare-and-swap. A new field of shared state goes
    into `_MselTable`, never into the store struct."""
    #: word 0 guards everything in the table
    var lock: List[Int64]
    #: exactly one `_MselTable`, never grown, so its address is stable
    var cell: List[_MselTable]

    def __init__(out self):
        self.lock = List[Int64](length=1, fill=Int64(0))
        self.cell = List[_MselTable]()
        self.cell.append(_MselTable())

    def _word(self) -> MutPointer[Int64, MutUntrackedOrigin]:
        return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(self.lock.unsafe_ptr()))

    def _table(self) -> MutPointer[_MselTable, MutUntrackedOrigin]:
        return MutPointer[_MselTable, MutUntrackedOrigin](unsafe_from_address=Int(self.cell.unsafe_ptr()))

    def live_count(self) -> Int:
        """Ids held now, read under the lock."""
        var table = self._table()
        var word = self._word()
        shared_slot_lock(word)
        var n = table[].live
        shared_slot_unlock(word)
        return n

    def check(self, id: Int, need: Int) raises:
        """Raises unless `id` is live and holds at least `need` bytes."""
        var table = self._table()
        var word = self._word()
        shared_slot_lock(word)
        try:
            _ = table[].check_held(id, need)
        finally:
            shared_slot_unlock(word)

    def address(self, id: Int, need: Int) raises -> Int:
        """The device address of a live id holding at least `need` bytes,
        read under the lock (the kernels take it as a raw pointer)."""
        var table = self._table()
        var word = self._word()
        shared_slot_lock(word)
        var at = 0
        try:
            var slot = table[].check_held(id, need)
            at = Int(table[].bufs[slot].unsafe_ptr())
        finally:
            shared_slot_unlock(word)
        return at

    def _insert(self, var buf: DeviceBuffer[DType.uint8], n: Int) -> Int:
        var table = self._table()
        var word = self._word()
        shared_slot_lock(word)
        var slot: Int
        if len(table[].released) > 0:
            slot = table[].released.pop()
            # The reused id's one-byte placeholder (`free`) is dropped here.
            table[].bufs[slot] = buf^
            table[].nbytes[slot] = n
            table[].gens[slot] += 1
        else:
            table[].bufs.append(buf^)
            table[].nbytes.append(n)
            table[].gens.append(1)
            slot = len(table[].nbytes) - 1
        var id = (table[].gens[slot] << MSEL_ID_SLOT_BITS) | slot
        table[].live += 1
        shared_slot_unlock(word)
        return id

    def put(self, ctx: DeviceContext, src_addr: Int, n: Int) raises -> Int:
        """A new id holding the `n` host bytes at `src_addr`; waits for the
        copy (the host memory may change after this returns)."""
        if n < 1 or src_addr == 0:
            raise Error("msel store: put needs at least one byte at a non-null address")
        var buf = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=buf, src_ptr=_U8(unsafe_from_address=src_addr))
        ctx.synchronize()
        return self._insert(buf^, n)

    def alloc(self, ctx: DeviceContext, n: Int) raises -> Int:
        """A new id holding `n` zero bytes."""
        if n < 1:
            raise Error("msel store: alloc needs at least one byte")
        var buf = ctx.enqueue_create_buffer[DType.uint8](n)
        enqueue_fill(ctx, buf, UInt8(0))
        return self._insert(buf^, n)

    def read(self, ctx: DeviceContext, id: Int, dst_addr: Int, n: Int) raises:
        """The id's first `n` bytes into host memory, after every launch
        queued before it; waits."""
        if n > 0:
            if dst_addr == 0:
                self.check(id, n)
                raise Error("msel store: read into a null address")
            # The view of the id's bytes is taken under the lock; the copy
            # runs after it.
            var sub = Optional[DeviceBuffer[DType.uint8]]()
            var table = self._table()
            var word = self._word()
            shared_slot_lock(word)
            try:
                var slot = table[].check_held(id, n)
                sub = table[].bufs[slot].create_sub_buffer[DType.uint8](0, n)
            finally:
                shared_slot_unlock(word)
            ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=dst_addr), src_buf=sub.value())
            ctx.synchronize()
            _ = sub^
            return
        self.check(id, n)
        ctx.synchronize()

    def free(self, ctx: DeviceContext, id: Int) raises:
        """Release the id now: wait for the context, drop the buffer (after
        the lock: the freed buffer is swapped out for a one-byte placeholder
        and released once the lock is given back)."""
        self.check(id, 0)
        ctx.synchronize()
        var dropped = ctx.enqueue_create_buffer[DType.uint8](1)
        var table = self._table()
        var word = self._word()
        shared_slot_lock(word)
        try:
            var slot = table[].check_held(id, 0)
            swap(table[].bufs[slot], dropped)
            table[].nbytes[slot] = -1
            table[].released.append(slot)
            table[].live -= 1
        finally:
            shared_slot_unlock(word)
        _ = dropped^


# ===========================================================================
# take_rows: dst[r] = src[idx[r]], rows of `row_bytes` bytes
# ===========================================================================


def _take_words_kernel(
    src: _U32, n_src: Int64, words: Int64, idx: _I64, n_idx: Int64, dst: _U32, status: _I32,
):
    var w = Int(words)
    var total = Int(n_idx) * w
    var step = _stride()
    var t = _tid()
    while t < total:
        var r = t // w
        var c = t - r * w
        var s = Int(idx.unsafe_load(r))
        if s < 0 or s >= Int(n_src):
            status.unsafe_store(0, Int32(1))
        else:
            dst.unsafe_store(t, src.unsafe_load(s * w + c))
        t += step


def _take_bytes_kernel(
    src: _U8, n_src: Int64, width: Int64, idx: _I64, n_idx: Int64, dst: _U8, status: _I32,
):
    var w = Int(width)
    var total = Int(n_idx) * w
    var step = _stride()
    var t = _tid()
    while t < total:
        var r = t // w
        var c = t - r * w
        var s = Int(idx.unsafe_load(r))
        if s < 0 or s >= Int(n_src):
            status.unsafe_store(0, Int32(1))
        else:
            dst.unsafe_store(t, src.unsafe_load(s * w + c))
        t += step


# ---------------------------------------------------------------------------
# The take's download (lane fx-rows-rest, 2026-11-18). The gathered rows came
# down with one raw `enqueue_copy(dst_ptr=<the caller's fresh Array>)` after a
# first wait for the status word: a pageable device-to-host copy into pages
# never touched before, which the driver stages and faults in on ONE thread
# (call profile nvc-prof-20261118T182616Z-11b404, train_test_split of
# 1,000,000 x 220 float32: 892 MB down in 399 ms, 2.2 GB/s, on a host whose
# pinned link runs 27 GB/s; half of the 800 ms call). Now the status word and
# the rows share one wait, and the rows come down as a PIPELINE over chunks of
# MSEL_STAGE_BYTES through two pinned stages per take, kept per context
# (core/staged_download.mojo's transport, in bytes): the copy engine fills one
# stage at the link's pinned rate while the host drains the other into the
# caller's memory, the drain split into contiguous byte spans across the
# host's cores (so the first touch of the output's pages is spread over the
# cores too). Copies only: the same bytes land in the same places, and no byte
# of `dst` is written before the status word is read. Every take goes this
# way (a take smaller than one stage is one chunk and one wait where there
# were two). A transport rule from the sizes in front of it, for any rows and
# any row width: nothing here names a shape.
# ---------------------------------------------------------------------------

#: Bytes per pinned stage and per pipeline chunk (16 MiB: under a millisecond
#: of a pinned link per chunk, 32 MiB of pinned memory held per context). A
#: multiple of 4096 so every chunk but the last starts on a page boundary of
#: a page-aligned output.
comptime MSEL_STAGE_BYTES = 1 << 24
# fx-leftovers-product (2026-11-19, audit H2): the pool kept 2 stages, one
# take's pair, so a second take in flight at the same time (the binding
# releases the GIL) allocated and freed 2 x 16 MiB of pinned memory on every
# call, and a pinned free can serialize the device. The pool now keeps what
# concurrent takers actually made, up to 4 takers' pairs: a process that only
# ever runs one take at a time still makes and holds 2 stages (32 MiB), since
# stages are only made when the pool is empty.
comptime MSEL_STAGE_KEEP = 8
comptime MSEL_STAGE_LAYOUT = 1
comptime _HU8 = MutPointer[UInt8, MutUntrackedOrigin]


struct _MselStages(Defaultable, Movable):
    """The pinned stages kept between takes: each MSEL_STAGE_BYTES long, all
    made on the context `key` names (core/ctx_key.mojo: a stage is never
    handed to another context; a take on another context retires them)."""
    var bufs: List[HostBuffer[DType.uint8]]
    var key: Int
    #: word 0 guards `bufs` and `key`: takes run with the GIL released
    var lock: List[Int64]

    def __init__(out self):
        # side-effect free (core/shared_slot.mojo, first creation)
        self.bufs = List[HostBuffer[DType.uint8]]()
        self.key = 0
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime MSEL_STAGES = SharedSlot[_MselStages, "MojolearnMselTakeStagesV1", MSEL_STAGE_LAYOUT]


def _msel_stage_take(ctx: DeviceContext) raises -> HostBuffer[DType.uint8]:
    """A pooled pinned stage of this context, else a new one."""
    var pool = MSEL_STAGES.get_or_create_ptr()
    var key = ctx_cache_key(ctx)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](pool[].lock.unsafe_ptr())
    # Another context's stages are moved out under the lock and freed after
    # it (audit N3; the `retired` form of core/device_pool.mojo `pool_give`):
    # a pinned free is a driver call and must not run under the spin lock.
    # The empty list allocates nothing; it grows only on a context change.
    var retired = List[HostBuffer[DType.uint8]]()
    shared_slot_lock(lock)
    if pool[].key != key:
        while len(pool[].bufs) > 0:  # driver-loop: over the pooled stages of the retired context
            retired.append(pool[].bufs.pop())
        pool[].key = key
    if len(pool[].bufs) > 0:
        var kept = pool[].bufs.pop()
        shared_slot_unlock(lock)
        _ = retired^
        return kept^
    shared_slot_unlock(lock)
    _ = retired^
    return ctx.enqueue_create_host_buffer[DType.uint8](MSEL_STAGE_BYTES)


def _msel_stage_give(ctx: DeviceContext, var stage: HostBuffer[DType.uint8]) raises:
    """Return a stage to the pool (kept while it is this context's and the
    pool holds fewer than MSEL_STAGE_KEEP; dropped otherwise)."""
    var pool = MSEL_STAGES.get_or_create_ptr()
    var key = ctx_cache_key(ctx)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](pool[].lock.unsafe_ptr())
    # A stage the pool does not keep is freed after the lock (audit N3).
    # The empty list allocates nothing; it grows only for a dropped stage.
    var spare = List[HostBuffer[DType.uint8]]()
    shared_slot_lock(lock)
    if pool[].key == key and len(pool[].bufs) < MSEL_STAGE_KEEP:
        pool[].bufs.append(stage^)
    else:
        spare.append(stage^)
    shared_slot_unlock(lock)
    _ = spare^


def _msel_stage_enqueue(
    ctx: DeviceContext, mut d_out: DeviceBuffer[DType.uint8], total: Int, k: Int,
    mut stage: HostBuffer[DType.uint8],
) raises:
    """Chunk `k` of the first `total` bytes of `d_out` onto `stage`: a full
    chunk buffer to buffer, the last (shorter) one onto the stage's pointer."""
    var lo = k * MSEL_STAGE_BYTES
    if lo + MSEL_STAGE_BYTES <= total:
        var view = d_out.create_sub_buffer[DType.uint8](lo, MSEL_STAGE_BYTES)
        ctx.enqueue_copy(dst_buf=stage, src_buf=view)
        _ = view^
    else:
        var view = d_out.create_sub_buffer[DType.uint8](lo, total - lo)
        ctx.enqueue_copy(dst_ptr=stage.unsafe_ptr(), src_buf=view)
        _ = view^


def _msel_copy_out(dst: Int, src: Int, n: Int):
    """`n` bytes of a pinned stage at `src` into the caller's memory at
    `dst`. Spans are independent (each writes its own bytes), so the bytes do
    not depend on the split."""
    # parallel: byte copy over contiguous spans across the host's cores (grain floor in 4-byte words)
    if n <= 0:
        return
    var tasks = host_predict_task_count_work(n, n // 4)
    if tasks <= 1:
        unsafe_memcpy(dest=_HU8(unsafe_from_address=dst), src=_HU8(unsafe_from_address=src), count=n)
        return
    # Whole pages per span, so no two tasks first-touch one page.
    var span = ((host_predict_chunk(n, tasks) + 4095) // 4096) * 4096

    def _span(t: Int) {imm dst, imm src, imm n, imm span}:
        var lo = t * span
        if lo >= n:
            return
        var hi = min(lo + span, n)
        unsafe_memcpy(dest=_HU8(unsafe_from_address=dst + lo), src=_HU8(unsafe_from_address=src + lo),
                      count=hi - lo)

    host_parallelize(_span, tasks)


def _msel_take_download(
    ctx: DeviceContext, mut d_out: DeviceBuffer[DType.uint8], total: Int,
    mut d_status: DeviceBuffer[DType.int32], dst_addr: Int,
) raises -> Bool:
    """The take's status word and, when it is 0, the `total` gathered bytes
    of `d_out` into host memory at `dst_addr` (the pipeline above). False,
    and no byte of `dst` written, when the status word is set."""
    var nchunks = (total + MSEL_STAGE_BYTES - 1) // MSEL_STAGE_BYTES
    var st0 = _msel_stage_take(ctx)
    var st1 = _msel_stage_take(ctx)
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    var ok = False
    # fx-leftovers-product (2026-11-19, audit N2): an error between an enqueue
    # and its wait dropped a pinned stage (or the status buffer) with a copy
    # still writing into it. Nothing is moved or freed inside the `try`; the
    # `except` waits for the queued copies, names the three host buffers
    # after that wait, and raises the same error.
    try:
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
        # The first chunk rides the status word's wait; it lands on our stage,
        # never on `dst`, so a refused take still writes nothing.
        _msel_stage_enqueue(ctx, d_out, total, 0, st0)
        ctx.synchronize()
        ok = Int(h.unsafe_ptr()[0]) == 0
        if ok and dst_addr != 0:
            var a0 = Int(st0.unsafe_ptr())
            var a1 = Int(st1.unsafe_ptr())
            var k = 0
            while k < nchunks:  # driver-loop: over the download's chunks, two stages alternating
                if k > 0:
                    ctx.synchronize()
                if k + 1 < nchunks:
                    _msel_stage_enqueue(ctx, d_out, total, k + 1, st1)
                _msel_copy_out(dst_addr + k * MSEL_STAGE_BYTES, a0,
                               min(MSEL_STAGE_BYTES, total - k * MSEL_STAGE_BYTES))
                if k + 1 >= nchunks:
                    break
                ctx.synchronize()
                if k + 2 < nchunks:
                    _msel_stage_enqueue(ctx, d_out, total, k + 2, st0)
                _msel_copy_out(dst_addr + (k + 1) * MSEL_STAGE_BYTES, a1,
                               min(MSEL_STAGE_BYTES, total - (k + 1) * MSEL_STAGE_BYTES))
                k += 2
    except error:
        try:
            ctx.synchronize()
        except:
            pass
        _ = h.unsafe_ptr()
        _ = st0.unsafe_ptr()
        _ = st1.unsafe_ptr()
        raise error
    # The status buffer and both stages are named here, after the last wait
    # and the last copy out of either.
    _ = h^
    _msel_stage_give(ctx, st0^)
    _msel_stage_give(ctx, st1^)
    if ok and dst_addr == 0:
        raise Error("msel take_rows: null destination")
    return ok


def device_take_rows(
    ctx: DeviceContext, store: MselStore, src_id: Int, n_src: Int, row_bytes: Int,
    idx_id: Int, n_idx: Int, dst_addr: Int,
) raises -> Bool:
    """Rows `idx` (int64, the store's `idx_id`) of the `n_src` x `row_bytes`
    source (the store's `src_id`) into host memory at `dst_addr`
    (`n_idx * row_bytes` bytes). False, and no byte of `dst` written, when an
    index is out of range."""
    if n_idx < 1 or row_bytes < 1:
        return True
    var src_at = store.address(src_id, n_src * row_bytes)
    var idx_at = store.address(idx_id, n_idx * 8)
    var total = n_idx * row_bytes
    var d_out = ctx.enqueue_create_buffer[DType.uint8](total)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    # the kernels declare idx mutable; the store hands out an immutable view (box-run-2 compile fix)
    var idx = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=idx_at)
    var src_u8 = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src_at)
    if row_bytes % 4 == 0:
        var w = row_bytes // 4
        ctx.enqueue_function[_take_words_kernel](
            src_u8.bitcast[UInt32](), Int64(n_src), Int64(w),
            idx, Int64(n_idx), d_out.unsafe_ptr().bitcast[UInt32](), d_status.unsafe_ptr(),
            grid_dim=_grid(n_idx * w), block_dim=MSEL_TPB,
        )
    else:
        ctx.enqueue_function[_take_bytes_kernel](
            src_u8, Int64(n_src), Int64(row_bytes),
            idx, Int64(n_idx), d_out.unsafe_ptr(), d_status.unsafe_ptr(),
            grid_dim=_grid(total), block_dim=MSEL_TPB,
        )
    # One wait for the status word and the first chunk; pinned stages and a
    # copy-out across the host's cores (`_msel_take_download`). It was a wait
    # for the status word, then one pageable copy of all `total` bytes.
    var ok = _msel_take_download(ctx, d_out, total, d_status, dst_addr)
    _ = d_out^
    _ = d_status^
    return ok


# REMOVED (lane vdf-scope-cv, 2026-11-24): scatter_rows (`_scatter_kernel`,
# `device_scatter_rows`), cross_val_predict's placement; its operation 16 is retired.


# ===========================================================================
# proba_column: dst[i] = float32(src[i, col])
# ===========================================================================


def _column_kernel(src: _U8, code: Int32, n: Int64, k: Int64, col: Int64, dst: _F32):
    var step = _stride()
    var t = _tid()
    var kk = Int(k)
    var cc = Int(col)
    var cd = Int(code)
    while t < Int(n):
        dst.unsafe_store(t, msel_load_f32(src, cd, t * kk + cc))
        t += step


def device_proba_column(
    ctx: DeviceContext, src_addr: Int, code: Int, n: Int, k: Int, col: Int, dst_addr: Int,
) raises:
    """Column `col` of the host n x k float32 or float64 buffer as float32 at
    `dst_addr` (float64 narrowed to nearest even)."""
    if n < 1:
        return
    var d_src = ctx.enqueue_create_buffer[DType.uint8](n * k * msel_itemsize(code))
    var d_dst = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=_U8(unsafe_from_address=src_addr))
    ctx.enqueue_function[_column_kernel](
        d_src.unsafe_ptr(), Int32(code), Int64(n), Int64(k), Int64(col), d_dst.unsafe_ptr(),
        grid_dim=_grid(n), block_dim=MSEL_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_F32(unsafe_from_address=dst_addr), src_buf=d_dst)
    ctx.synchronize()
    _ = d_src^
    _ = d_dst^


# ===========================================================================
# rebase_offsets_i32: the parallel forest's merged tree offsets
# ===========================================================================


def _rebase_kernel(raw: _I32, lens: _I64, parts: Int64, n_out: Int64, dst: _I32, status: _I32):
    """dst[0] = 0; output o >= 1 is element j >= 1 of part s plus the sum of
    the last offsets of parts 0..s-1 (int64, refused past int32)."""
    var step = _stride()
    var o = _tid()
    while o < Int(n_out):
        if o == 0:
            dst.unsafe_store(0, Int32(0))
        else:
            var rem = o - 1
            var base = Int64(0)
            var start = 0
            var done = False
            for s in range(Int(parts)):
                var len_s = Int(lens.unsafe_load(s))
                var m = len_s - 1
                if rem < m:
                    var v = base + Int64(raw.unsafe_load(start + 1 + rem))
                    if v < Int64(-2147483648) or v > Int64(2147483647):
                        status.unsafe_store(0, Int32(1))
                    else:
                        dst.unsafe_store(o, Int32(v))
                    done = True
                    break
                rem -= m
                base += Int64(raw.unsafe_load(start + len_s - 1))
                start += len_s
            if not done:
                status.unsafe_store(0, Int32(1))
        o += step


def device_rebase_offsets_i32(
    ctx: DeviceContext, raw_addr: Int, n_raw: Int, lens_addr: Int, parts: Int, n_out: Int, dst_addr: Int,
) raises -> Bool:
    """The parts' int32 offset arrays laid end to end at `raw_addr` (`lens`
    int64, one length >= 1 per part) merged into one offset array of
    `n_out` = 1 + sum(len - 1) words at `dst_addr`. False when a merged
    offset leaves int32 or the sizes disagree."""
    if n_out < 1 or parts < 1 or n_raw < 1:
        return False
    var d_raw = ctx.enqueue_create_buffer[DType.int32](n_raw)
    var d_lens = ctx.enqueue_create_buffer[DType.int64](parts)
    var d_dst = ctx.enqueue_create_buffer[DType.int32](n_out)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_copy(dst_buf=d_raw, src_ptr=_I32(unsafe_from_address=raw_addr))
    ctx.enqueue_copy(dst_buf=d_lens, src_ptr=_I64(unsafe_from_address=lens_addr))
    ctx.enqueue_function[_rebase_kernel](
        d_raw.unsafe_ptr(), d_lens.unsafe_ptr(), Int64(parts), Int64(n_out), d_dst.unsafe_ptr(),
        d_status.unsafe_ptr(), grid_dim=_grid(n_out), block_dim=MSEL_TPB,
    )
    var ok = _status(ctx, d_status) == 0
    if ok:
        ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.synchronize()
    _ = d_raw^
    _ = d_lens^
    _ = d_dst^
    _ = d_status^
    return ok


# ===========================================================================
# Group splitters: GroupShuffleSplit's side table, shuffled GroupKFold's runs
# ===========================================================================
#
# Per-group row counts add with Int32 atomics (exact in any order: every sum
# is at most the row count, below 2^31 as in core/hotpath_device.mojo), and
# each group's entry is written by the one thread holding its position in
# the permutation, so the bytes do not depend on the schedule.


def _split_side_kernel(
    perm: _I64, m: Int64, n_test: Int64, n_side: Int64, counts: _I64, table: _I32, sums: _I32, status: _I32,
):
    var step = _stride()
    var i = _tid()
    while i < Int(n_side):
        var g = Int(perm.unsafe_load(i))
        if g < 0 or g >= Int(m):
            status.unsafe_store(0, Int32(1))
        else:
            var side = 1 if i < Int(n_test) else 0
            table.unsafe_store(g, Int32(side))
            _ = Atomic[Int32].fetch_add(sums + side, Int32(counts.unsafe_load(g)))
        i += step


def device_split_table_i32(
    ctx: DeviceContext, perm_addr: Int, m: Int, n_test: Int, n_train: Int,
    counts_addr: Int, table_addr: Int, sums_addr: Int,
) raises -> Bool:
    """`split_table_i32` on the device: table[g] = 1 for the first n_test
    groups of the int64 permutation, 0 for the next n_train, 2 for the
    rest; sums (int64, 2) = [train rows, test rows]. False when a permuted
    group is out of range."""
    var n_side = n_test + n_train
    var d_perm = ctx.enqueue_create_buffer[DType.int64](max(n_side, 1))
    var d_counts = ctx.enqueue_create_buffer[DType.int64](m)
    var d_table = ctx.enqueue_create_buffer[DType.int32](m)
    var d_sums = ctx.enqueue_create_buffer[DType.int32](2)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    enqueue_fill(ctx, d_sums, Int32(0))
    if n_side > 0:
        ctx.enqueue_copy(
            dst_buf=d_perm.create_sub_buffer[DType.int64](0, n_side), src_ptr=_I64(unsafe_from_address=perm_addr)
        )
    ctx.enqueue_copy(dst_buf=d_counts, src_ptr=_I64(unsafe_from_address=counts_addr))
    enqueue_fill_n(ctx, d_table, m, Int32(2))
    if n_side > 0:
        ctx.enqueue_function[_split_side_kernel](
            d_perm.unsafe_ptr(), Int64(m), Int64(n_test), Int64(n_side), d_counts.unsafe_ptr(),
            d_table.unsafe_ptr(), d_sums.unsafe_ptr(), d_status.unsafe_ptr(),
            grid_dim=_grid(n_side), block_dim=MSEL_TPB,
        )
    var ok = _status(ctx, d_status) == 0
    if ok:
        var h = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=table_addr), src_buf=d_table)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_sums)
        ctx.synchronize()
        var sp = _I64(unsafe_from_address=sums_addr)
        sp.unsafe_store(0, Int64(h.unsafe_ptr()[0]))
        sp.unsafe_store(1, Int64(h.unsafe_ptr()[1]))
        _ = h^
    _ = d_perm^
    _ = d_counts^
    _ = d_table^
    _ = d_sums^
    _ = d_status^
    return ok


def _group_perm_fold_kernel(
    perm: _I64, m: Int64, folds: Int64, counts: _I64, dst: _I32, sizes: _I32, status: _I32,
):
    var mm = Int(m)
    var K = Int(folds)
    var q = mm // K
    var r = mm - q * K
    var long_end = r * (q + 1)
    var step = _stride()
    var j = _tid()
    while j < mm:
        var f: Int
        if j < long_end:
            f = j // (q + 1)
        else:
            f = r + (j - long_end) // q
        var g = Int(perm.unsafe_load(j))
        if g < 0 or g >= mm:
            status.unsafe_store(0, Int32(1))
        else:
            dst.unsafe_store(g, Int32(f))
            _ = Atomic[Int32].fetch_add(sizes + f, Int32(counts.unsafe_load(g)))
        j += step


def device_group_fold_perm_i32(
    ctx: DeviceContext, counts_addr: Int, m: Int, n_folds: Int, perm_addr: Int, dst_addr: Int, sizes_addr: Int,
) raises -> Bool:
    """Shuffled GroupKFold: the permuted groups split into n_folds nearly
    equal runs (the first m % n_folds one longer); int32 dst[g] the fold of
    group g, int64 sizes[f] each fold's rows. Needs m >= n_folds >= 1.
    False when a permuted group is out of range."""
    var d_perm = ctx.enqueue_create_buffer[DType.int64](m)
    var d_counts = ctx.enqueue_create_buffer[DType.int64](m)
    var d_dst = ctx.enqueue_create_buffer[DType.int32](m)
    var d_sizes = ctx.enqueue_create_buffer[DType.int32](n_folds)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    enqueue_fill(ctx, d_sizes, Int32(0))
    ctx.enqueue_copy(dst_buf=d_perm, src_ptr=_I64(unsafe_from_address=perm_addr))
    ctx.enqueue_copy(dst_buf=d_counts, src_ptr=_I64(unsafe_from_address=counts_addr))
    ctx.enqueue_function[_group_perm_fold_kernel](
        d_perm.unsafe_ptr(), Int64(m), Int64(n_folds), d_counts.unsafe_ptr(), d_dst.unsafe_ptr(),
        d_sizes.unsafe_ptr(), d_status.unsafe_ptr(), grid_dim=_grid(m), block_dim=MSEL_TPB,
    )
    var ok = _status(ctx, d_status) == 0
    if ok:
        var h = ctx.enqueue_create_host_buffer[DType.int32](n_folds)
        ctx.enqueue_copy(dst_ptr=_I32(unsafe_from_address=dst_addr), src_buf=d_dst)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_sizes)
        ctx.synchronize()
        # the n_folds fold sizes widened to the caller's int64 (k scalars)
        var sp = _I64(unsafe_from_address=sizes_addr)
        for f in range(n_folds):
            sp.unsafe_store(f, Int64(h.unsafe_ptr()[f]))
        _ = h^
    _ = d_perm^
    _ = d_counts^
    _ = d_dst^
    _ = d_sizes^
    _ = d_status^
    return ok
