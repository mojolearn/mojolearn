# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident word buffers named by an integer id (lane py-shared,
2026-09-28): the shared store behind `mojolearn._arena_io.DeviceCache`.

A Python loop that hands a binding the same array call after call (a
cross-validation scorer, an imputer round, an optimizer's state, a model's
weights) uploads it once into a `DeviceStore` slot and passes the id; the
binding copies device to device, or launches on the slot's pointer. A slot
holds 4-byte WORDS (float32, or int32 bits: every copy is a byte copy, so
what a slot holds is exactly the bytes the host gave it).

EACH BINDING OWNS ITS OWN STORE, next to its own DeviceContext, because a
device buffer belongs to the context that made it:

    comptime MY_STORE = _Global[StorageType=DeviceStore, name="MojoMyStore<Tier>",
                                init_fn=DeviceStore.__init__]
    var s = MY_STORE.get_or_create_ptr()
    var id = s[].put(my_ctx(), host_addr, n_words)

and exports `<prefix>_dev_put(addr, n_words) -> id`, `<prefix>_dev_free(id)`
and `<prefix>_dev_live() -> count` (x_metrics/resident.mojo is the pattern,
three lines each), which `DeviceCache` calls by that prefix. A binding that
takes device-rows input (`mojolearn._arena_io.DeviceRows`, lane cpu4-misc)
also exports `<prefix>_dev_take_rows(src_id, row_words, idx_addr, n_idx) -> id`:
a new slot gathered on the device out of a resident one (`take_rows`).

FREED DETERMINISTICALLY: `free` waits for the context (so no queued launch
still reads the slot), then drops the buffer at once (the slot keeps a
shared one-word placeholder), and the id is reused. Outside FAST a large
freed buffer is kept as a spare for the next put of its length (G2,
STORE_SLOT_POOL, capped at STORE_POOL_KEEP_BYTES); `-D
MOJOLEARN_XPREP_ARENA_POOL_OFF=1` makes live device memory exactly the live
slots again. Where bytes live moves no bit.
"""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
# TOMBSTONE: MOJOLEARN_XPREP_PINNED_UPLOAD_OFF (slower) deleted 2026-10-09 by fd7e86b46; code recoverable at fd7e86b46^.
# Restore: git apply experiments/removed/MOJOLEARN_XPREP_PINNED_UPLOAD.patch; record in docs/TOMBSTONES.md.
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_zero import enqueue_fill

#: TOMBSTONE G1 (lane fg-knn-nb pinned upload stage, `MOJOLEARN_XPREP_PINNED_UPLOAD`;
#: deleted 2026-10-09 by lane postmerge-act-1, code recoverable at main 432d6e8ff):
#: `put`/`write` of >= 1M words went through a store-lifetime pinned stage of
#: two 32 MB halves (host memcpy of chunk i overlapping the DMA of chunk
#: i - 1). Post-merge A/B (one run per arm) found the plain pageable copy
#: FASTER on both vendors, gaussian-nb default -> `_OFF`: istella NV 129.8 ->
#: 88.7 ms (0.68x), AMD 46.4 -> 27.9 (0.60x); taxi NV 20.9 -> 19.3 (0.92x), AMD
#: 14.1 -> 12.9 (0.91x); same digests. The host memcpy into the stage plus
#: the pinned allocation cost more than the DMA gain. Every upload is one
#: `enqueue_copy` from the caller's pointer; both defines are refused by
#: core/six_lane_experiment_guards.mojo. No bit moves.

#: G2 (lane fg-knn-nb, slot half; the arena half is x_prep/device.mojo
#: X_PREP_POOL_ARENA): a freed slot's buffer of at least STORE_POOL_MIN_WORDS
#: is kept as a SPARE and handed to the next `put` of exactly its length
#: instead of a free now and a fresh allocation then. Cost reasoning: a
#: program's direct X slot is put and freed once per fit, so a repeated fit
#: of one shape paid a device allocation and a device free of the whole X
#: per call (tens of ms per GB on CUDA, more on ROCm, plus first-touch page
#: mapping). Idle spares are capped at STORE_POOL_KEEP_BYTES (2 GB, the
#: policy cap of core/device_pool.mojo), oldest dropped first. A spare's
#: words are whatever the last slot held; `put` overwrites all of them before
#: any read. Not in FAST (untouched by this lane). DEFAULT ON (storage only,
#: no bit moves); `-D MOJOLEARN_XPREP_ARENA_POOL_OFF=1` restores the
#: free-at-once store (and the fresh device arena, x_prep/device.mojo).
comptime STORE_SLOT_POOL = (
    GLOBAL_NUMERIC_MODE != NUMERIC_FAST
    and not is_defined["MOJOLEARN_XPREP_ARENA_POOL_OFF"]()
)
comptime STORE_POOL_MIN_WORDS = 1 << 20
comptime STORE_POOL_KEEP_BYTES = 2 * 1024 * 1024 * 1024

comptime StoreWP = MutPointer[Float32, MutAnyOrigin]
comptime _StoreU32 = MutPointer[UInt32, MutAnyOrigin]
comptime _StoreI64 = MutPointer[Int64, MutAnyOrigin]
comptime _StoreI32 = MutPointer[Int32, MutAnyOrigin]
comptime STORE_TPB = 256
comptime STORE_MAX_BLOCKS = 65535


def _store_take_kernel(
    src: _StoreU32, n_src: Int64, words: Int64, idx: _StoreI64, n_idx: Int64, dst: _StoreU32,
    status: _StoreI32,
):
    """dst[r, :] = src[idx[r], :] over rows of `words` 4-byte words: one
    thread per output word (grid-stride). A byte copy, so no bit moves; an
    out-of-range index sets status[0] and leaves its row unwritten."""
    var w = Int(words)
    var total = Int(n_idx) * w
    var step = Int(grid_dim.x) * Int(block_dim.x)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while t < total:
        var r = t // w
        var c = t - r * w
        var s = Int(idx.unsafe_load(r))
        if s < 0 or s >= Int(n_src):
            status.unsafe_store(0, Int32(1))
        else:
            dst.unsafe_store(t, src.unsafe_load(s * w + c))
        t += step


def _store_grid(total: Int) -> Int:
    var b = (total + STORE_TPB - 1) // STORE_TPB
    if b < 1:
        return 1
    if b > STORE_MAX_BLOCKS:
        return STORE_MAX_BLOCKS
    return b


struct DeviceStore(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.float32]]
    #: words held by each id; -1 for a freed id
    var words: List[Int]
    var released: List[Int]
    var live: Int
    #: G2: freed slot buffers kept for a put of the same length, and their bytes.
    var spare: List[DeviceBuffer[DType.float32]]
    var spare_bytes: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.words = List[Int]()
        self.released = List[Int]()
        self.live = 0
        self.spare = List[DeviceBuffer[DType.float32]]()
        self.spare_bytes = 0

    def _new_buffer(mut self, ctx: DeviceContext, n_words: Int) raises -> DeviceBuffer[DType.float32]:
        """G2: a spare of exactly `n_words` when one is idle, else a fresh
        allocation. Its words are unspecified either way."""
        comptime if STORE_SLOT_POOL:
            var n_spare = len(self.spare)
            for i in range(n_spare):  # small-loop(n_spare: idle spare slot buffers): a few buffer handles, never data
                if len(self.spare[i]) == n_words:
                    self.spare_bytes -= n_words * 4
                    return self.spare.pop(i)
        return ctx.enqueue_create_buffer[DType.float32](n_words)

    def _retire_buffer(mut self, var buf: DeviceBuffer[DType.float32]):
        """G2: keep a freed slot's buffer as a spare (the caller has waited
        on every use), or drop it. Over STORE_POOL_KEEP_BYTES the oldest
        spares are dropped first."""
        comptime if STORE_SLOT_POOL:
            var n = len(buf)
            var nb = n * 4
            if n >= STORE_POOL_MIN_WORDS and nb <= STORE_POOL_KEEP_BYTES:
                while len(self.spare) > 0 and self.spare_bytes + nb > STORE_POOL_KEEP_BYTES:
                    self.spare_bytes -= len(self.spare[0]) * 4
                    _ = self.spare.pop(0)
                self.spare_bytes += nb
                self.spare.append(buf^)
                return
        _ = buf^

    def _upload(mut self, ctx: DeviceContext, dst: DeviceBuffer[DType.float32], src_addr: Int, n_words: Int) raises:
        """The `n_words` host words at `src_addr` into `dst[0, n_words)`;
        waits. One copy from the caller's pointer (the G1 pinned stage was
        deleted 2026-10-09: slower on both vendors, see the TOMBSTONE above)."""
        if n_words <= 0:
            ctx.synchronize()
            return
        if n_words == len(dst):
            ctx.enqueue_copy(dst_buf=dst, src_ptr=StoreWP(unsafe_from_address=src_addr))
        else:
            ctx.enqueue_copy(
                dst_buf=dst.create_sub_buffer[DType.float32](0, n_words),
                src_ptr=StoreWP(unsafe_from_address=src_addr),
            )
        ctx.synchronize()

    # TOMBSTONE: `stage_upload` (x_prep G3 host spans, MOJOLEARN_XPREP_NO_SLOT_HOP) deleted 2026-10-09
    # (lane postmerge-act-2): G3 slower on the average, gaussian-nb istella NV 1.25x / AMD 1.00x,
    # taxi 1.02x / 1.03x (nv n0630, amd a1090); code recoverable at main 5c137b55e.

    def check(self, id: Int, need: Int) raises:
        """Raises unless `id` is live and holds at least `need` words."""
        if id < 0 or id >= len(self.words) or self.words[id] < 0:
            raise Error(String("device store: id ", id, " is not live"))
        if self.words[id] < need:
            raise Error(String("device store: id ", id, " holds ", self.words[id], " words, ", need, " read"))

    def put(mut self, ctx: DeviceContext, src_addr: Int, n_words: Int) raises -> Int:
        """A new slot holding the `n_words` host words at `src_addr`; waits
        for the copy (the host memory may change after this returns)."""
        if n_words < 1 or src_addr == 0:
            raise Error("device store: put needs at least one word at a non-null address")
        var buf = self._new_buffer(ctx, n_words)
        self._upload(ctx, buf, src_addr, n_words)
        return self._add(buf^, n_words)

    def _add(mut self, var buf: DeviceBuffer[DType.float32], n_words: Int) -> Int:
        """Files `buf` (n_words words) under a fresh or released id."""
        var id: Int
        if len(self.released) > 0:
            id = self.released.pop()
            self.bufs[id] = buf^
            self.words[id] = n_words
        else:
            self.bufs.append(buf^)
            self.words.append(n_words)
            id = len(self.words) - 1
        self.live += 1
        return id

    def take_rows(
        mut self, ctx: DeviceContext, src_id: Int, row_words: Int, idx_addr: Int, n_idx: Int
    ) raises -> Int:
        """A new slot holding rows `idx` (the `n_idx` host int64 words at
        `idx_addr`) of slot `src_id`, read as rows of `row_words` words:
        one device gather (lane cpu4-misc, device-rows input). Only the
        index words cross the bus; the rows never visit the host. Raises
        on an out-of-range index (one status word read back); waits."""
        if n_idx < 1 or row_words < 1 or idx_addr == 0:
            raise Error("device store: take_rows needs at least one index and one word a row")
        self.check(src_id, 0)
        var n_src = self.words[src_id] // row_words
        var total = n_idx * row_words
        var d_idx = ctx.enqueue_create_buffer[DType.int64](n_idx)
        ctx.enqueue_copy(dst_buf=d_idx, src_ptr=_StoreI64(unsafe_from_address=idx_addr))
        var d_status = ctx.enqueue_create_buffer[DType.int32](2)
        enqueue_fill(ctx, d_status, Int32(0))
        var out = ctx.enqueue_create_buffer[DType.float32](total)
        ctx.enqueue_function[_store_take_kernel](
            self.bufs[src_id].unsafe_ptr().bitcast[UInt32](), Int64(n_src), Int64(row_words),
            d_idx.unsafe_ptr(), Int64(n_idx), out.unsafe_ptr().bitcast[UInt32](), d_status.unsafe_ptr(),
            grid_dim=_store_grid(total), block_dim=STORE_TPB,
        )
        var h = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_status)
        ctx.synchronize()
        var bad = Int(h.unsafe_ptr()[0])
        _ = h^
        _ = d_idx^
        _ = d_status^
        if bad != 0:
            raise Error("device store: take_rows index out of range")
        return self._add(out^, total)

    def write(mut self, ctx: DeviceContext, id: Int, src_addr: Int, n_words: Int) raises:
        """Overwrite the slot's first `n_words` words from the host; waits."""
        self.check(id, n_words)
        if n_words > 0:
            var view = self.bufs[id].create_sub_buffer[DType.float32](0, n_words)
            self._upload(ctx, view, src_addr, n_words)
            _ = view^
        ctx.synchronize()

    def read(self, ctx: DeviceContext, id: Int, dst_addr: Int, n_words: Int) raises:
        """The slot's first `n_words` words into host memory, after every
        launch queued before it; waits."""
        self.check(id, n_words)
        if n_words > 0:
            ctx.enqueue_copy(
                dst_ptr=StoreWP(unsafe_from_address=dst_addr),
                src_buf=self.bufs[id].create_sub_buffer[DType.float32](0, n_words),
            )
        ctx.synchronize()

    def copy_into(self, ctx: DeviceContext, id: Int, dst: DeviceBuffer[DType.float32], at: Int, n_words: Int) raises:
        """Enqueue a device-to-device copy of the slot's first `n_words`
        words into `dst[at, at + n_words)` (no wait: stream order)."""
        self.check(id, n_words)
        if n_words > 0:
            ctx.enqueue_copy(
                dst_buf=dst.create_sub_buffer[DType.float32](at, n_words),
                src_buf=self.bufs[id].create_sub_buffer[DType.float32](0, n_words),
            )

    def ptr(self, id: Int, need: Int) raises -> StoreWP:
        """The slot's device pointer, for a binding's own launches."""
        self.check(id, need)
        return StoreWP(unsafe_from_address=Int(self.bufs[id].unsafe_ptr()))

    def free(mut self, ctx: DeviceContext, id: Int) raises:
        """Release the slot now: wait for the context, drop the buffer."""
        self.check(id, 0)
        ctx.synchronize()
        comptime if STORE_SLOT_POOL:
            # G2: the slot's buffer becomes a spare (or is dropped there).
            var old = self.bufs.pop(id)
            self.bufs.insert(id, ctx.enqueue_create_buffer[DType.float32](1))
            self._retire_buffer(old^)
        else:
            self.bufs[id] = ctx.enqueue_create_buffer[DType.float32](1)
        self.words[id] = -1
        self.released.append(id)
        self.live -= 1
