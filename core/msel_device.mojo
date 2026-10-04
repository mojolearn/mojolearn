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

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_zero import enqueue_fill
from core.msel_convert import (
    MSEL_I64,
    msel_itemsize,
    msel_load_f32,
    msel_load_f64_bits,
    msel_load_i64,
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


struct MselStore(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.uint8]]
    #: bytes held by each id; -1 for a freed id
    var nbytes: List[Int]
    var released: List[Int]
    var live: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.uint8]]()
        self.nbytes = List[Int]()
        self.released = List[Int]()
        self.live = 0

    def check(self, id: Int, need: Int) raises:
        """Raises unless `id` is live and holds at least `need` bytes."""
        if id < 0 or id >= len(self.nbytes) or self.nbytes[id] < 0:
            raise Error(String("msel store: id ", id, " is not live"))
        if self.nbytes[id] < need:
            raise Error(String("msel store: id ", id, " holds ", self.nbytes[id], " bytes, ", need, " read"))

    def _insert(mut self, var buf: DeviceBuffer[DType.uint8], n: Int) -> Int:
        var id: Int
        if len(self.released) > 0:
            id = self.released.pop()
            self.bufs[id] = buf^
            self.nbytes[id] = n
        else:
            self.bufs.append(buf^)
            self.nbytes.append(n)
            id = len(self.nbytes) - 1
        self.live += 1
        return id

    def put(mut self, ctx: DeviceContext, src_addr: Int, n: Int) raises -> Int:
        """A new id holding the `n` host bytes at `src_addr`; waits for the
        copy (the host memory may change after this returns)."""
        if n < 1 or src_addr == 0:
            raise Error("msel store: put needs at least one byte at a non-null address")
        var buf = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=buf, src_ptr=_U8(unsafe_from_address=src_addr))
        ctx.synchronize()
        return self._insert(buf^, n)

    def alloc(mut self, ctx: DeviceContext, n: Int) raises -> Int:
        """A new id holding `n` zero bytes."""
        if n < 1:
            raise Error("msel store: alloc needs at least one byte")
        var buf = ctx.enqueue_create_buffer[DType.uint8](n)
        enqueue_fill(ctx, buf, UInt8(0))
        return self._insert(buf^, n)

    def read(self, ctx: DeviceContext, id: Int, dst_addr: Int, n: Int) raises:
        """The id's first `n` bytes into host memory, after every launch
        queued before it; waits."""
        self.check(id, n)
        if n > 0:
            if dst_addr == 0:
                raise Error("msel store: read into a null address")
            ctx.enqueue_copy(
                dst_ptr=_U8(unsafe_from_address=dst_addr),
                src_buf=self.bufs[id].create_sub_buffer[DType.uint8](0, n),
            )
        ctx.synchronize()

    def free(mut self, ctx: DeviceContext, id: Int) raises:
        """Release the id now: wait for the context, drop the buffer."""
        self.check(id, 0)
        ctx.synchronize()
        self.bufs[id] = ctx.enqueue_create_buffer[DType.uint8](1)
        self.nbytes[id] = -1
        self.released.append(id)
        self.live -= 1


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
    store.check(src_id, n_src * row_bytes)
    store.check(idx_id, n_idx * 8)
    var total = n_idx * row_bytes
    var d_out = ctx.enqueue_create_buffer[DType.uint8](total)
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    var idx = store.bufs[idx_id].unsafe_ptr().bitcast[Int64]()
    if row_bytes % 4 == 0:
        var w = row_bytes // 4
        ctx.enqueue_function[_take_words_kernel](
            store.bufs[src_id].unsafe_ptr().bitcast[UInt32](), Int64(n_src), Int64(w),
            idx, Int64(n_idx), d_out.unsafe_ptr().bitcast[UInt32](), d_status.unsafe_ptr(),
            grid_dim=_grid(n_idx * w), block_dim=MSEL_TPB,
        )
    else:
        ctx.enqueue_function[_take_bytes_kernel](
            store.bufs[src_id].unsafe_ptr(), Int64(n_src), Int64(row_bytes),
            idx, Int64(n_idx), d_out.unsafe_ptr(), d_status.unsafe_ptr(),
            grid_dim=_grid(total), block_dim=MSEL_TPB,
        )
    var ok = _status(ctx, d_status) == 0
    if ok:
        if dst_addr == 0:
            raise Error("msel take_rows: null destination")
        ctx.enqueue_copy(dst_ptr=_U8(unsafe_from_address=dst_addr), src_buf=d_out)
        ctx.synchronize()
    _ = d_out^
    _ = d_status^
    return ok


# ===========================================================================
# scatter_rows: dst[idx[r]] = cast(src[r]), into a resident 64-bit block
# ===========================================================================


def _scatter_kernel(
    src: _U8, src_code: Int32, n_src: Int64, width: Int64, idx: _I64, n_dst: Int64,
    dst: _U64, dst_code: Int32, status: _I32,
):
    var w = Int(width)
    var total = Int(n_src) * w
    var code = Int(src_code)
    var to_int = Int(dst_code) == MSEL_I64
    var step = _stride()
    var t = _tid()
    while t < total:
        var r = t // w
        var c = t - r * w
        var d = Int(idx.unsafe_load(r))
        if d < 0 or d >= Int(n_dst):
            status.unsafe_store(0, Int32(1))
        else:
            var word: UInt64
            if to_int:
                word = bitcast[DType.uint64](msel_load_i64(src, code, t))
            else:
                word = msel_load_f64_bits(src, code, t)
            dst.unsafe_store(d * w + c, word)
        t += step


def device_scatter_rows(
    ctx: DeviceContext, store: MselStore, dst_id: Int, n_dst: Int, dst_code: Int,
    src_addr: Int, src_code: Int, n_src: Int, width: Int, idx_id: Int,
) raises -> Bool:
    """Row r of the host `n_src` x `width` buffer at `src_addr` (dtype
    `src_code`) into row idx[r] (int64, the store's `idx_id`) of the
    resident `n_dst` x `width` block `dst_id` of int64 (`dst_code` I64) or
    float64 words, cast on the device. False when an index is out of range
    (the block is then not to be read)."""
    if n_src < 1 or width < 1:
        return True
    store.check(dst_id, n_dst * width * 8)
    store.check(idx_id, n_src * 8)
    var nbytes = n_src * width * msel_itemsize(src_code)
    var d_src = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=_U8(unsafe_from_address=src_addr))
    var d_status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, d_status, Int32(0))
    ctx.enqueue_function[_scatter_kernel](
        d_src.unsafe_ptr(), Int32(src_code), Int64(n_src), Int64(width),
        store.bufs[idx_id].unsafe_ptr().bitcast[Int64](), Int64(n_dst),
        store.bufs[dst_id].unsafe_ptr().bitcast[UInt64](), Int32(dst_code), d_status.unsafe_ptr(),
        grid_dim=_grid(n_src * width), block_dim=MSEL_TPB,
    )
    var ok = _status(ctx, d_status) == 0
    _ = d_src^
    _ = d_status^
    return ok


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
