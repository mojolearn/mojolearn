# SPDX-License-Identifier: Apache-2.0
"""The x_decomp GPU image's member device store (lane vdf-scope-neural3, 2026-11-24): the device pool of the image
(x_decomp/resident.mojo `X_DECOMP_POOL`) as a `resident()` scope transport entry, so a cross-validation or a search
over XDecomp members (core/xdecomp_member.mojo) keeps X on the device ONCE and takes each fold's rows there.

A slot of this store IS a pool id: every resident operation of the image (XD_DEV_*, the decomposition kit
bindings/decomp_kit_run.mojo, the manifold kit bindings/mani_kit_run.mojo) reads it directly, with no copy and no
second id space. The entry has the x_prep transport numbering and signature (x_prep/transport_abi.mojo;
bindings/arena_cache_run.mojo ARENA_STORE_*):

  ARENA_STORE_PUT        a[0] host address, p[0] 4-byte words   -> a fresh pool id holding them (waits for the copy)
  ARENA_STORE_FREE       p[0] pool id                           -> 0 (the pool's free: the id waits in its class)
  ARENA_STORE_TAKE_ROWS  a[0] int64 row numbers, p[0] source pool id, p[1] words a row, p[2] row count
                         -> a fresh pool id holding those rows, gathered ON THE DEVICE (one launch of the
                            device store's word gather, core/device_store.mojo `_store_take_kernel`; only the
                            index words cross the bus, an out-of-range row is refused from one status word)
  ARENA_STORE_LIVE                                              -> the pool's live ids (a test readout)

No numerics: words are copied, never converted, so a taken id holds exactly the bytes a host gather of the same
rows would hold, and every kind's fit and transform compute the same bits from either. Exported as the capsule
`x_decomp_store_v1` by the GPU binding only (bindings/_mojolearn_x_decomp.mojo); the host image has no pool and
exports none, so its members answer no row index and keep the host path (the CPU column). Serial by nature: each
operation is one pool bookkeeping update and one copy or one gather launch on the image's one context.
"""
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import _store_take_kernel, _store_grid, STORE_TPB
from core.device_zero import enqueue_fill
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_decomp.cells import F32Ptr
from x_decomp.device import xd_ctx
from x_decomp.resident import pool_alloc, pool_free, pool_buffer, pool_live_count, dev_upload_run
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, XP_TRANSPORT_ABI, xp_validate_call

comptime _I64P = MutPointer[Int64, MutAnyOrigin]


struct _StoreWords(Defaultable, Movable):
    """The words each id this store made holds (a pool id's buffer is its
    size class, larger than the put): the exact row count of a take's source,
    so an index past X's rows is refused, not read from the class padding."""
    var words: List[Int]
    var lock: List[Int64]

    def __init__(out self):
        self.words = List[Int]()
        self.lock = List[Int64](length=1, fill=0)


comptime _WORDS = SharedSlot[_StoreWords, ("MojolearnXDecompStoreWordsIdenticalV1"
                                           if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                           else "MojolearnXDecompStoreWordsFastV1"), 2]


def _words_set(id: Int, words: Int) raises:
    var w = _WORDS.get_or_create_ptr()
    var lock = w[].lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    shared_slot_lock(lock)
    try:
        while len(w[].words) <= id:  # driver-loop: over pool ids up to this one (bookkeeping, a few per scope)
            w[].words.append(0)
        w[].words[id] = words
    finally:
        shared_slot_unlock(lock)


def _words_of(id: Int) raises -> Int:
    var w = _WORDS.get_or_create_ptr()
    var lock = w[].lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    shared_slot_lock(lock)
    try:
        if id < 0 or id >= len(w[].words) or w[].words[id] <= 0:
            raise Error("x_decomp store: pool id " + String(id) + " was not put through this store")
        return w[].words[id]
    finally:
        shared_slot_unlock(lock)


def xd_store_words(id: Int) raises -> Int:
    """The exact words pool id `id` holds as this store put or took it (`_StoreWords`), raising for an id the
    store did not make. A resident source's row bound (x_decomp/pca_randomized.mojo `RDev.held_words`)."""
    return _words_of(id)


def xd_store_take_rows(src_id: Int, row_words: Int, idx_addr: Int, n_idx: Int) raises -> Int:
    """A fresh pool id holding rows `idx` (the `n_idx` host int64 words at
    `idx_addr`) of pool id `src_id` read as rows of `row_words` words: one
    device gather, the rows never visit the host. Refuses an out-of-range
    row (one status word read back); waits."""
    if n_idx < 1 or row_words < 1 or idx_addr == 0:
        raise Error("x_decomp store: take_rows needs at least one index and one word a row")
    var total = n_idx * row_words
    var n_src = _words_of(src_id) // row_words
    var src = pool_buffer(src_id, n_src * row_words)
    var out_id = pool_alloc(total)
    var out = pool_buffer(out_id, total)
    var ctx = xd_ctx()
    var d_idx = ctx.enqueue_create_buffer[DType.int64](n_idx)
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=_I64P(unsafe_from_address=idx_addr))
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_function[_store_take_kernel](
        src.unsafe_ptr().bitcast[UInt32](), Int64(n_src), Int64(row_words),
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
    _ = src^  # keep the source id's buffer alive through the gather above
    _ = out^
    if bad != 0:
        pool_free(out_id)
        raise Error("IndexError: a row index is out of range for X's " + String(n_src) + " rows")
    _words_set(out_id, total)
    return out_id


def _xd_store_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == XP_DEV_PUT:
        if a[0] == 0 or p[0] < 1:
            raise Error("x_decomp store: a put needs at least one word at a non-null address")
        var id = pool_alloc(p[0])
        _ = dev_upload_run(id, F32Ptr(unsafe_from_address=a[0]), p[0])
        _words_set(id, p[0])
        return id
    if op == XP_DEV_FREE:
        _words_set(p[0], 0)
        pool_free(p[0])
        return 0
    if op == XP_DEV_TAKE_ROWS:
        return xd_store_take_rows(p[0], p[1], a[0], p[2])
    if op == XP_DEV_LIVE:
        return pool_live_count()
    raise Error("x_decomp store: transport operation " + String(op) + " is not a store operation")


def x_decomp_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                      context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The store's transport entry (module note)."""
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _xd_store_op(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XDecompStoreAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(XP_TRANSPORT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(x_decomp_store_v1)


comptime X_DECOMP_STORE_API = SharedSlot[XDecompStoreAPI, ("MojolearnXDecompStoreAPIIdenticalV1"
                                                           if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                                           else "MojolearnXDecompStoreAPIFastV1")]
