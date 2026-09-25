"""A stable LSD radix sort of (u32 key, u32 value) pairs with 8-bit digits,
for FAST builds.

The pinned sort (`gbdt/gpu_util/kernel/radix_sort.mojo`) reorders one bit
per pass: 32 passes of four launches for a full key. On Apple each launch
is a command buffer, so a 20k-element sort costs about 128 launches. This
sort takes four passes of three launches (digit counts per tile, one
exclusive scan, a stable scatter).

A stable sort by the full key is ONE permutation (ties keep input order),
so the result is the same pairs in the same order as the one-bit sort.

Layout: tiles of `FRS_TILE` elements, one thread per element. `counts`
holds `256 * n_tiles` int32, digit-major (`d * n_tiles + tile`), so the
exclusive scan of that array is each (digit, tile)'s first output slot.
The in-tile rank of an element is the number of earlier elements of its
tile with the same digit (a scan over the tile in shared memory), which is
what makes the scatter stable.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

comptime FRS_TILE = 256
comptime FRS_BINS = 256
comptime FRS_SCAN_TPB = 1024


def frs_counts_len(size: Int) -> Int:
    """int32 slots `counts` must hold for `size` elements."""
    return FRS_BINS * ((size + FRS_TILE - 1) // FRS_TILE)


def _frs_count_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    size: Int32,
    shift: Int32,
    n_tiles: Int32,
):
    var tile = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var n = Int(size)
    var h = stack_allocation[FRS_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var dg = stack_allocation[FRS_TILE, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var i = tile * FRS_TILE + t
    dg[t] = Int32((keys[i] >> UInt32(shift)) & 255) if i < n else Int32(-1)
    barrier()
    # thread t counts digit t over the tile: no atomics, exact
    var c = Int32(0)
    for j in range(FRS_TILE):
        if dg[j] == Int32(t):
            c += 1
    counts[t * Int(n_tiles) + tile] = c


def _frs_scan_kernel(counts: MutPointer[Int32, MutAnyOrigin], m: Int32):
    """Exclusive scan of `counts[0:m]` in one block."""
    var t = Int(thread_idx.x)
    var n = Int(m)
    var per = (n + FRS_SCAN_TPB - 1) // FRS_SCAN_TPB
    var lo = t * per
    var hi = lo + per
    if hi > n:
        hi = n
    var s = Int32(0)
    for j in range(lo, hi):
        s += counts[j]
    var sh = stack_allocation[FRS_SCAN_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    sh[t] = s
    barrier()
    var off = 1
    while off < FRS_SCAN_TPB:
        var add = Int32(0)
        if t >= off:
            add = sh[t - off]
        barrier()
        sh[t] += add
        barrier()
        off *= 2
    var run = sh[t] - s
    for j in range(lo, hi):
        var c = counts[j]
        counts[j] = run
        run += c


def _frs_scatter_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    vals: MutPointer[UInt32, MutAnyOrigin],
    out_keys: MutPointer[UInt32, MutAnyOrigin],
    out_vals: MutPointer[UInt32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    size: Int32,
    shift: Int32,
    n_tiles: Int32,
):
    var tile = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var n = Int(size)
    var dg = stack_allocation[FRS_TILE, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var i = tile * FRS_TILE + t
    var k = UInt32(0)
    var d = Int32(-1)
    if i < n:
        k = keys[i]
        d = Int32((k >> UInt32(shift)) & 255)
    dg[t] = d
    barrier()
    if i < n:
        var r = Int32(0)
        for j in range(t):
            if dg[j] == d:
                r += 1
        var dst = Int(counts[Int(d) * Int(n_tiles) + tile] + r)
        out_keys[dst] = k
        out_vals[dst] = vals[i]


def fast_radix_sort_pairs_u32(
    ctx: DeviceContext,
    size: Int,
    mut keys: DeviceBuffer[DType.uint32],
    mut values: DeviceBuffer[DType.uint32],
    mut temp_keys: DeviceBuffer[DType.uint32],
    mut temp_values: DeviceBuffer[DType.uint32],
    mut counts: DeviceBuffer[DType.int32],
) raises:
    """Sort `keys` ascending carrying `values`, stably, in place (four
    passes, so the answer ends where it started). `counts` holds at least
    `frs_counts_len(size)` slots."""
    if size <= 0:
        return
    var n_tiles = (size + FRS_TILE - 1) // FRS_TILE
    var m = FRS_BINS * n_tiles
    for p in range(4):
        var shift = Int32(8 * p)
        if p % 2 == 0:
            ctx.enqueue_function[_frs_count_kernel](
                keys.unsafe_ptr(), counts.unsafe_ptr(), Int32(size), shift,
                Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )
        else:
            ctx.enqueue_function[_frs_count_kernel](
                temp_keys.unsafe_ptr(), counts.unsafe_ptr(), Int32(size), shift,
                Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )
        ctx.enqueue_function[_frs_scan_kernel](
            counts.unsafe_ptr(), Int32(m), grid_dim=1, block_dim=FRS_SCAN_TPB,
        )
        if p % 2 == 0:
            ctx.enqueue_function[_frs_scatter_kernel](
                keys.unsafe_ptr(), values.unsafe_ptr(), temp_keys.unsafe_ptr(),
                temp_values.unsafe_ptr(), counts.unsafe_ptr(), Int32(size),
                shift, Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )
        else:
            ctx.enqueue_function[_frs_scatter_kernel](
                temp_keys.unsafe_ptr(), temp_values.unsafe_ptr(), keys.unsafe_ptr(),
                values.unsafe_ptr(), counts.unsafe_ptr(), Int32(size),
                shift, Int32(n_tiles), grid_dim=n_tiles, block_dim=FRS_TILE,
            )
