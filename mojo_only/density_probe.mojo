"""What indexed cindex reads really cost on THIS device, by leaf density.

    pixi run -e bench mojo run -I . mojo_only/density_probe.mojo

The shape sweep measured the density cliff from outside (per-level cost
11.5 -> 27.8 ms at constant built rows). This probe measures the
mechanism from inside, on the exact access shape the hist kernels use:
`ci[col * R + idx[i]]` where idx is an ASCENDING, IRREGULARLY-GAPPED
subset of rows -- what a leaf's rows look like after stable partition.
Strided subsets would let the prefetcher cheat, so membership is hashed.

Three arms, alternated in one process:

  DENSE    sum ci[col*R + i] over all R rows        (the ceiling)
  INDEXED  sum ci[col*R + idx[i]] at density 1/k    (the hist's read)
  GATHER   out[i] = ci[col*R + idx[i]], then DENSE over out
           (their GatherBins arm's true cost: the SAME amplified read,
            plus a dense write, plus the dense re-read the hist does)

The decision this feeds, cited: their `ELoadFromCompressedIndexPolicy`
picks GatherBins only above 2 stats (`split_properties_helper.cpp:
1338-1341`) -- on their hardware one amplified pass beats gather's
triple at <= 2 stats. If Apple's amplification curve is steep enough
that INDEXED at 1/32-1/64 loses to GATHER's read+write+reread, then
flipping that threshold ON THIS VENDOR is a kernel-matrix policy row
over THEIR OWN two arms, not an invention. If INDEXED holds up, the
compaction idea stays deferred and this file is the negative result.

Times are per-arm medians over reps; every buffer touch is written to a
per-block partial so no load can be dead-coded. Useful bytes at density
1/k are 4 * R/k per column; the printed GB/s is USEFUL bandwidth, so
amplification shows up as the dense/indexed ratio directly.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.time import perf_counter_ns

comptime R = 400_000
comptime COLS = 500
comptime BLOCK = 256
comptime REPS = 5


def _sum_dense_kernel(
    ci: MutPointer[UInt32, MutAnyOrigin],
    partials: MutPointer[UInt32, MutAnyOrigin],
    n_rows: Int32,
):
    var col = Int(block_idx.y)
    var base = col * Int(n_rows)
    var acc = UInt32(0)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * 64
    while i < Int(n_rows):
        acc += ci[unsafe_offset = base + i]
        i += stride
    var red = stack_allocation[
        BLOCK, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    red[unsafe_offset = Int(thread_idx.x)] = acc
    barrier()
    if thread_idx.x == 0:
        var s = UInt32(0)
        for t in range(BLOCK):
            s += red[unsafe_offset=t]
        partials[
            unsafe_offset = col * 64 + Int(block_idx.x)
        ] = s


def _sum_indexed_kernel(
    ci: MutPointer[UInt32, MutAnyOrigin],
    idx: MutPointer[UInt32, MutAnyOrigin],
    partials: MutPointer[UInt32, MutAnyOrigin],
    n_rows: Int32,
    m: Int32,
):
    var col = Int(block_idx.y)
    var base = col * Int(n_rows)
    var acc = UInt32(0)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * 64
    while i < Int(m):
        acc += ci[unsafe_offset = base + Int(idx[unsafe_offset=i])]
        i += stride
    var red = stack_allocation[
        BLOCK, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    red[unsafe_offset = Int(thread_idx.x)] = acc
    barrier()
    if thread_idx.x == 0:
        var s = UInt32(0)
        for t in range(BLOCK):
            s += red[unsafe_offset=t]
        partials[
            unsafe_offset = col * 64 + Int(block_idx.x)
        ] = s


def _gather_kernel(
    ci: MutPointer[UInt32, MutAnyOrigin],
    idx: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
    n_rows: Int32,
    m: Int32,
):
    var col = Int(block_idx.y)
    var base = col * Int(n_rows)
    var dst_base = col * Int(m)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * 64
    while i < Int(m):
        dst[unsafe_offset = dst_base + i] = ci[
            unsafe_offset = base + Int(idx[unsafe_offset=i])
        ]
        i += stride


def main() raises:
    var ctx = DeviceContext()

    var ci = ctx.enqueue_create_buffer[DType.uint32](R * COLS)
    var partials = ctx.enqueue_create_buffer[DType.uint32](COLS * 64)
    var hci = ctx.enqueue_create_host_buffer[DType.uint32](R * COLS)
    var state = UInt64(0x9E3779B97F4A7C15)
    for i in range(R * COLS):
        state = state * 6364136223846793005 + 1442695040888963407
        hci.unsafe_ptr().unsafe_store(i, UInt32((state >> 33) & 0xFF))
    ctx.enqueue_copy(dst_buf=ci, src_ptr=hci.unsafe_ptr())
    ctx.synchronize()

    print("density probe:", R, "rows x", COLS, "uint32 columns,",
          REPS, "reps, hashed membership, ascending indices")

    # densities 1/1 (identity control), 1/2, 1/8, 1/32, 1/64
    var ks = [1, 2, 8, 32, 64]

    for ki in range(len(ks)):
        var k = ks[ki]
        # hashed ascending subset: position j joins iff hash(j) % k == 0
        var members = List[UInt32]()
        var h = UInt64(12345)
        for j in range(R):
            h = (UInt64(j) * 0x9E3779B97F4A7C15) ^ (h >> 7)
            if Int(h % UInt64(k)) == 0:
                members.append(UInt32(j))
        var m = len(members)
        if m == 0:
            continue
        var idx = ctx.enqueue_create_buffer[DType.uint32](m)
        var hidx = ctx.enqueue_create_host_buffer[DType.uint32](m)
        for i in range(m):
            hidx.unsafe_ptr().unsafe_store(i, members[i])
        ctx.enqueue_copy(dst_buf=idx, src_ptr=hidx.unsafe_ptr())
        var gathered = ctx.enqueue_create_buffer[DType.uint32](m * COLS)
        ctx.synchronize()

        var useful_gb = Float64(4 * m * COLS) / 1e9

        var t_dense = Float64(1e18)
        var t_indexed = Float64(1e18)
        var t_gather = Float64(1e18)
        for _ in range(REPS):
            # DENSE over the gathered-size region (the re-read gather pays)
            var t0 = perf_counter_ns()
            ctx.enqueue_function[_sum_dense_kernel](
                gathered.unsafe_ptr(), partials.unsafe_ptr(), Int32(m),
                grid_dim=(64, COLS), block_dim=(BLOCK, 1, 1),
            )
            ctx.synchronize()
            var d = Float64(perf_counter_ns() - t0) / 1e6
            if d < t_dense:
                t_dense = d

            t0 = perf_counter_ns()
            ctx.enqueue_function[_sum_indexed_kernel](
                ci.unsafe_ptr(), idx.unsafe_ptr(), partials.unsafe_ptr(),
                Int32(R), Int32(m),
                grid_dim=(64, COLS), block_dim=(BLOCK, 1, 1),
            )
            ctx.synchronize()
            d = Float64(perf_counter_ns() - t0) / 1e6
            if d < t_indexed:
                t_indexed = d

            t0 = perf_counter_ns()
            ctx.enqueue_function[_gather_kernel](
                ci.unsafe_ptr(), idx.unsafe_ptr(), gathered.unsafe_ptr(),
                Int32(R), Int32(m),
                grid_dim=(64, COLS), block_dim=(BLOCK, 1, 1),
            )
            ctx.synchronize()
            d = Float64(perf_counter_ns() - t0) / 1e6
            if d < t_gather:
                t_gather = d

        # GatherBins' full price: amplified gather + dense re-read the
        # hist then does. (The dense WRITE is inside t_gather already.)
        var t_gather_total = t_gather + t_dense
        print(
            "  density 1/", k, " m", m,
            " useful", useful_gb, "GB:",
            " dense", t_dense, "ms (", useful_gb / (t_dense / 1e3), "GB/s)",
            " indexed", t_indexed, "ms (", useful_gb / (t_indexed / 1e3),
            "GB/s)  gather+reread", t_gather_total, "ms  -> indexed is",
            t_indexed / t_gather_total, "x of gather-arm cost",
        )
