# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CUB / Thrust stand-ins the SVM port needs: flagged compaction, the
float key twiddle, gathers, fills, and the pinned serial sum.

NOT A PORT of any one file. Each function names the call it stands in for.
CUB and Thrust are OPEN (PORTING_RULES 0b-i), so these are written out
rather than substituted by a device-wide MAX primitive; MAX ships no device
`select`/`partition` anyway (VENDOR_LIBS.md).

THE COMPACTION (`cub::DeviceSelect::Flagged`, used by `workingset.cuh::
GatherAvailable`, `results.cuh::SelectByCoef` / `SelectUnboundSV` /
`SelectReduce`, and `thrust::copy_if` in `smosolver.cuh::
GetNonzeroDeltaAlpha`) is ORDER-PRESERVING: `out[k]` is the k-th flagged
input in index order. Ours is the three-pass decoupled scan the repository
already ports for CatBoost's one-bit reorder (`gbdt/gpu_util/kernel/
reorder_one_bit.mojo`: block scan, block-sums scan, carry) followed by a
scatter `out[offsets[i]] = in[i]` for flagged `i`. Integer scan and scatter:
no float arithmetic, so nothing here has a mode and nothing here can move a
bit -- the property the working-set identity rests on is that this is a
pure function of the flag bytes, and an exclusive prefix sum of 0/1 is.

THE TWIDDLE (`cub::DeviceRadixSort::SortPairs` over float keys): cub's
internal float key transform, negatives bit-inverted, non-negatives with
the sign bit set, so unsigned order is ascending float order with -0.0
before +0.0 and NaNs at the ends. `gbdt/gpu_util/kernel/radix_sort.mojo::
DeviceFloatSorter` spells the same map on the host; here it is a kernel
because the keys are already on the device.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from gbdt.gpu_util.kernel.reorder_one_bit import (
    REORDER_BLOCK,
    add_block_carry_kernel,
    block_scan_flags_kernel,
    scan_block_sums_kernel,
)
from original.numerics import ftz


comptime SEL_TPB = 256


def _grid(n: Int) -> Int:
    return (n + SEL_TPB - 1) // SEL_TPB


# ---------------------------------------------------------------------------
# elementwise kernels
# ---------------------------------------------------------------------------


def twiddle_keys_kernel(
    f: MutPointer[Float32, MutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    values: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """`keys[i] = twiddle(f[i])`, `values[i] = i` (their `f_idx =
    range(n_train)`, `workingset.cuh::Initialize`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var bits = bitcast[DType.uint32](f.unsafe_load(i))
        var key: UInt32
        if bits & UInt32(0x80000000) != UInt32(0):
            key = ~bits
        else:
            key = bits | UInt32(0x80000000)
        keys.unsafe_store(i, key)
        values.unsafe_store(i, UInt32(i))


def gather_u8_by_u32_kernel(
    dst: MutPointer[UInt8, MutAnyOrigin],
    src: MutPointer[UInt8, MutAnyOrigin],
    idx: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """`thrust::copy(make_permutation_iterator(av, idx), ...)`: the flags in
    sorted order (`workingset.cuh::GatherAvailable`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, src.unsafe_load(Int(idx.unsafe_load(i))))


def scatter_flagged_i32_kernel(
    dst: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        if flags.unsafe_load(i) != UInt8(0):
            dst.unsafe_store(Int(offsets.unsafe_load(i)), src.unsafe_load(i))


def scatter_flagged_u32_as_i32_kernel(
    dst: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """The sorted index payload is `UInt32` (the radix sort's value type);
    the working set is `Int32`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        if flags.unsafe_load(i) != UInt8(0):
            dst.unsafe_store(
                Int(offsets.unsafe_load(i)), Int32(Int(src.unsafe_load(i)))
            )


def scatter_flagged_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        if flags.unsafe_load(i) != UInt8(0):
            dst.unsafe_store(Int(offsets.unsafe_load(i)), src.unsafe_load(i))


def count_flagged_kernel(
    d_count: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[UInt8, MutAnyOrigin],
    n_in: Int32,
):
    """`d_num_selected = offsets[n-1] + flags[n-1]`."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        var n = Int(n_in)
        var last = Int32(Int(flags.unsafe_load(n - 1)) & 1)
        d_count.unsafe_store(0, offsets.unsafe_load(n - 1) + last)


def flag_nonzero_f32_kernel(
    flags: MutPointer[UInt8, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`[] (math_t a) { return a != 0; }` (`GetNonzeroDeltaAlpha`,
    `GetDualCoefs`, `GetSupportVectorIndices`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var x = v.unsafe_load(i)
        flags.unsafe_store(i, UInt8(1) if x != Float32(0.0) else UInt8(0))


def flag_nan_f32_kernel(
    flag: MutPointer[Int32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """DEVIATION 637 (`smosolver.mojo`): `flag[0] = 1` if any `v[i]` is
    NaN (`x != x`; inf is not flagged, its bits are the same on every
    vendor). Plain stores of the same value from many threads; reset by
    `set_i32_kernel` before each scan. No arithmetic, no mode."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var x = v.unsafe_load(i)
        if x != x:
            flag.unsafe_store(0, Int32(1))


def set_i32_kernel(v: MutPointer[Int32, MutAnyOrigin], value: Int32):
    """One-thread store of one Int32 (the NaN flag's reset)."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        v.unsafe_store(0, value)


def flag_free_kernel(
    flags: MutPointer[UInt8, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    C: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`[] (math_t a, math_t C) { return 0 < a && a < C; }`
    (`results.cuh::SelectUnboundSV`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var a = alpha.unsafe_load(i)
        var c = C.unsafe_load(i)
        var ok = Float32(0.0) < a and a < c
        flags.unsafe_store(i, UInt8(1) if ok else UInt8(0))


def fill_u8_kernel(
    flags: MutPointer[UInt8, MutAnyOrigin], value: UInt8, n_in: Int32
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        flags.unsafe_store(i, value)


def fill_f32_kernel(
    v: MutPointer[Float32, MutAnyOrigin], value: Float32, n_in: Int32
):
    """`thrust::fill` (`InitPenalty`) and `cudaMemsetAsync(alpha, 0)`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        v.unsafe_store(i, value)


def range_i32_kernel(v: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    """`raft::linalg::range`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        v.unsafe_store(i, Int32(i))


def svr_vec_indices_kernel(
    vec_idx: MutPointer[Int32, MutAnyOrigin],
    ws_idx: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_ws_in: Int32,
):
    """`GetVecIndices`, `kernelcache.cuh:804-808`.

    "For SVR we have duplicate set of training vectors, we return the
    original idx, which is simply ws_idx % n_rows." Their own spelling is
    `y < n ? y : y - n` rather than a modulo, which is the same thing given
    `ws_idx < 2 * n_rows` and is kept because a `%` would also accept an
    index they never produce.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_ws_in):
        var v = ws_idx.unsafe_load(i)
        vec_idx.unsafe_store(i, v if v < n_rows_in else v - n_rows_in)


def copy_i32_kernel(
    dst: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Int32, MutAnyOrigin],
    dst_off: Int32,
    src_off: Int32,
    n_in: Int32,
):
    """`raft::copy(dst + dst_off, src + src_off, n)`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(
            Int(dst_off) + i, src.unsafe_load(Int(src_off) + i)
        )


def gather_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    idx: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`thrust::transform(indices, ..., select_at_index(source))`
    (`kernelcache.cuh::selectValueSubset`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, src.unsafe_load(Int(idx.unsafe_load(i))))


def gather_rows_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    idx: MutPointer[Int32, MutAnyOrigin],
    n_sel_in: Int32,
    n_cols_in: Int32,
):
    """`ML::SVM::extractRows` for a DENSE matrix (`sparse_util.cuh`): row
    `idx[r]` of `x` to row `r` of `out`. Row-major both sides (theirs is
    column-major; the layout is a port detail recorded in DERIVATION_MAP)."""
    var k = Int(n_cols_in)
    var total = Int(n_sel_in) * k
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < total:
        var r = t // k
        var c = t - r * k
        var src_row = Int(idx.unsafe_load(r))
        dst.unsafe_store(t, x.unsafe_load(src_row * k + c))


def serial_sum_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """DEVIATION 632: `cub::DeviceReduce::Sum`'s stand-in at the two
    `CalcB` sites, an ASCENDING serial chain in one thread, every partial
    flushed. One thread and no fold so there is no shape to pin; `n` is
    the free-SV count or `n_train`, and this runs once per fit."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        var acc = Float32(0.0)
        for i in range(Int(n_in)):
            acc = ftz(acc + ftz(v.unsafe_load(i)))
        dst.unsafe_store(0, acc)


def serial_min_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`cub::DeviceReduce::Min` (`results.cuh::SelectReduce`): exact away
    from a +0.0/-0.0 pair, so a one-thread scan is the same answer as any
    fold; ON that pair (row 39) the strict `<` keeps the FIRST element in
    index order (the compaction preserves training-index order), which is
    the oracle's serial rule in `smo_oracle.mojo::_results`, and not a
    hardware `min`. A NaN at index 0 would persist and a later one would
    be dropped; none reaches here (DEVIATION 637 raises first)."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        var m = v.unsafe_load(0)
        for i in range(1, Int(n_in)):
            var x = v.unsafe_load(i)
            if x < m:
                m = x
        dst.unsafe_store(0, m)


def serial_max_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`cub::DeviceReduce::Max`; the twin of `serial_min_f32_kernel`, same
    row-39 note: strict `>`, first index wins a +0.0/-0.0 tie."""
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        var m = v.unsafe_load(0)
        for i in range(1, Int(n_in)):
            var x = v.unsafe_load(i)
            if x > m:
                m = x
        dst.unsafe_store(0, m)


# ---------------------------------------------------------------------------
# the flagged compaction: scratch + host driver
# ---------------------------------------------------------------------------


struct SelectScratch(Movable):
    """`cub_storage` + `d_num_selected`: the scan's offsets, the block sums
    and the one-int count, sized once for a capacity of `n` elements."""

    var offsets: DeviceBuffer[DType.int32]
    var block_sums: DeviceBuffer[DType.int32]
    var d_count: DeviceBuffer[DType.int32]
    var h_count: HostBuffer[DType.int32]
    var capacity: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        var cap = capacity
        if cap < 1:
            cap = 1
        self.capacity = cap
        self.offsets = ctx.enqueue_create_buffer[DType.int32](cap)
        self.block_sums = ctx.enqueue_create_buffer[DType.int32](
            (cap + REORDER_BLOCK - 1) // REORDER_BLOCK
        )
        self.d_count = ctx.enqueue_create_buffer[DType.int32](1)
        self.h_count = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.synchronize()

    def scan(
        mut self,
        ctx: DeviceContext,
        mut flags: DeviceBuffer[DType.uint8],
        n: Int,
    ) raises -> Int:
        """Exclusive prefix sum of the flag bytes into `offsets`; returns
        the count. DRAINS (reads one int back), exactly where theirs does
        (`d_num_selected.value(stream)` + `sync_stream`)."""
        if n > self.capacity:
            raise Error(
                "svm SelectScratch: n=" + String(n)
                + " exceeds capacity " + String(self.capacity)
            )
        if n <= 0:
            return 0
        var n_blocks = (n + REORDER_BLOCK - 1) // REORDER_BLOCK
        ctx.enqueue_function[block_scan_flags_kernel](
            flags.unsafe_ptr(), Int32(0), Int32(n),
            self.offsets.unsafe_ptr(), self.block_sums.unsafe_ptr(),
            grid_dim=n_blocks, block_dim=REORDER_BLOCK,
        )
        ctx.enqueue_function[scan_block_sums_kernel](
            self.block_sums.unsafe_ptr(), Int32(n_blocks),
            grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[add_block_carry_kernel](
            self.offsets.unsafe_ptr(), self.block_sums.unsafe_ptr(), Int32(n),
            grid_dim=n_blocks, block_dim=REORDER_BLOCK,
        )
        ctx.enqueue_function[count_flagged_kernel](
            self.d_count.unsafe_ptr(), self.offsets.unsafe_ptr(),
            flags.unsafe_ptr(), Int32(n),
            grid_dim=1, block_dim=1,
        )
        ctx.enqueue_copy(dst_ptr=self.h_count.unsafe_ptr(), src_buf=self.d_count)
        ctx.synchronize()
        return Int(self.h_count.unsafe_ptr().unsafe_load(0))

    def select_i32(
        mut self,
        ctx: DeviceContext,
        mut src: DeviceBuffer[DType.int32],
        mut flags: DeviceBuffer[DType.uint8],
        mut out: DeviceBuffer[DType.int32],
        n: Int,
    ) raises -> Int:
        """`cub::DeviceSelect::Flagged(src, flags, out, d_num_selected, n)`."""
        var count = self.scan(ctx, flags, n)
        if n > 0:
            ctx.enqueue_function[scatter_flagged_i32_kernel](
                out.unsafe_ptr(), src.unsafe_ptr(), flags.unsafe_ptr(),
                self.offsets.unsafe_ptr(), Int32(n),
                grid_dim=_grid(n), block_dim=SEL_TPB,
            )
        return count

    def select_u32_as_i32(
        mut self,
        ctx: DeviceContext,
        mut src: DeviceBuffer[DType.uint32],
        mut flags: DeviceBuffer[DType.uint8],
        mut out: DeviceBuffer[DType.int32],
        n: Int,
    ) raises -> Int:
        var count = self.scan(ctx, flags, n)
        if n > 0:
            ctx.enqueue_function[scatter_flagged_u32_as_i32_kernel](
                out.unsafe_ptr(), src.unsafe_ptr(), flags.unsafe_ptr(),
                self.offsets.unsafe_ptr(), Int32(n),
                grid_dim=_grid(n), block_dim=SEL_TPB,
            )
        return count

    def select_f32(
        mut self,
        ctx: DeviceContext,
        mut src: DeviceBuffer[DType.float32],
        mut flags: DeviceBuffer[DType.uint8],
        mut out: DeviceBuffer[DType.float32],
        n: Int,
    ) raises -> Int:
        var count = self.scan(ctx, flags, n)
        if n > 0:
            ctx.enqueue_function[scatter_flagged_f32_kernel](
                out.unsafe_ptr(), src.unsafe_ptr(), flags.unsafe_ptr(),
                self.offsets.unsafe_ptr(), Int32(n),
                grid_dim=_grid(n), block_dim=SEL_TPB,
            )
        return count


def read_scalar_f32(
    ctx: DeviceContext, mut scalar: DeviceBuffer[DType.float32]
) raises -> Float32:
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=scalar)
    ctx.synchronize()
    var v = h.unsafe_ptr().unsafe_load(0)
    _ = h^
    return v


def read_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """First `n` elements of a device buffer, drained."""
    var out = List[Float32]()
    if n <= 0:
        return out^
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    else:
        ctx.enqueue_copy(
            dst_ptr=h.unsafe_ptr(),
            src_buf=buf.create_sub_buffer[DType.float32](0, n),
        )
    ctx.synchronize()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def read_i32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    var out = List[Int32]()
    if n <= 0:
        return out^
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    else:
        ctx.enqueue_copy(
            dst_ptr=h.unsafe_ptr(),
            src_buf=buf.create_sub_buffer[DType.int32](0, n),
        )
    ctx.synchronize()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def upload_f32(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """A device buffer holding `values` (at least one element allocated)."""
    var n = len(values)
    var cap = n
    if cap < 1:
        cap = 1
    var d = ctx.enqueue_create_buffer[DType.float32](cap)
    var h = ctx.enqueue_create_host_buffer[DType.float32](cap)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, values[i])
    if n < cap:
        h.unsafe_ptr().unsafe_store(0, Float32(0.0))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return d^


def upload_i32(
    ctx: DeviceContext, values: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var n = len(values)
    var cap = n
    if cap < 1:
        cap = 1
    var d = ctx.enqueue_create_buffer[DType.int32](cap)
    var h = ctx.enqueue_create_host_buffer[DType.int32](cap)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, values[i])
    if n < cap:
        h.unsafe_ptr().unsafe_store(0, Int32(0))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return d^
