from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer, DeviceFunction
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, block_dim


def touch_kernel(p: MutPointer[Float32, MutAnyOrigin], n: Int32, v: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        p[i] = p[i] + v


def zero_kernel(p: MutPointer[UInt8, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        p[i] = 0


def bench(name: String, mode: Int, ctx: DeviceContext, kfn: DeviceFunction, zfn: DeviceFunction, buf: DeviceBuffer[DType.float32], small: DeviceBuffer[DType.uint8], hsmall: HostBuffer[DType.uint8], zbuf: DeviceBuffer[DType.uint8], n: Int) raises:
    var r = 1500
    var t0 = perf_counter_ns()
    for _ in range(r):
        if mode == 0:
            ctx.enqueue_memset(small, UInt8(0))
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
        elif mode == 1:
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
            ctx.enqueue_memset(small, UInt8(0))
        elif mode == 2:
            ctx.enqueue_copy(dst_buf=small, src_buf=hsmall)
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
        elif mode == 3:
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
        elif mode == 4:
            ctx.enqueue_function(zfn, zbuf, Int32(4096), grid_dim=(16), block_dim=(256))
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
        elif mode == 5:
            ctx.enqueue_memset(small, UInt8(0))
            ctx.enqueue_memset(small, UInt8(0))
            ctx.enqueue_memset(small, UInt8(0))
        elif mode == 6:
            ctx.enqueue_copy(dst_buf=small, src_buf=hsmall)
            ctx.enqueue_copy(dst_buf=hsmall, src_buf=small)
            ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
    var t1 = perf_counter_ns()
    ctx.synchronize()
    var t2 = perf_counter_ns()
    print(name, ": ", Float64(t1 - t0) / Float64(r) / 1000.0, " us host per group; incl drain ", Float64(t2 - t0) / Float64(r) / 1000.0, " us")


def main() raises:
    var ctx = DeviceContext()
    var n = 1024
    var hsmall = ctx.enqueue_create_host_buffer[DType.uint8](4096)
    var zbuf = ctx.enqueue_create_buffer[DType.uint8](4096)
    var zfn = ctx.compile_function[zero_kernel]()
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var big = ctx.enqueue_create_buffer[DType.uint8](64 * 1024 * 1024)
    var sub = big.create_sub_buffer[DType.uint8](0, 4 * 1024 * 1024)
    ctx.synchronize()
    var kfn = ctx.compile_function[touch_kernel]()
    for _ in range(50):
        ctx.enqueue_memset(sub, UInt8(0))
    ctx.synchronize()
    var reps = 2000
    var t0 = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_memset(sub, UInt8(0))
    var t1 = perf_counter_ns()
    ctx.synchronize()
    var t2 = perf_counter_ns()
    print("memset 4MB sub-buffer: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us host; incl drain ", Float64(t2 - t0) / Float64(reps) / 1000.0, " us")
    var small = big.create_sub_buffer[DType.uint8](0, 4096)
    t0 = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_memset(small, UInt8(0))
    t1 = perf_counter_ns()
    ctx.synchronize()
    t2 = perf_counter_ns()
    print("memset 4KB sub-buffer: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us host; incl drain ", Float64(t2 - t0) / Float64(reps) / 1000.0, " us")
    t0 = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_memset(small, UInt8(0))
        ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
        ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
    t1 = perf_counter_ns()
    ctx.synchronize()
    t2 = perf_counter_ns()
    print("memset+2 launches: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us host per triple; incl drain ", Float64(t2 - t0) / Float64(reps) / 1000.0, " us")
    t0 = perf_counter_ns()
    for i in range(reps):
        var v = big.create_sub_buffer[DType.uint8](0, 4096 + (i % 7) * 64)
        ctx.enqueue_memset(v, UInt8(0))
        _ = v^
    t1 = perf_counter_ns()
    ctx.synchronize()
    t2 = perf_counter_ns()
    print("create_sub_buffer+memset: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us host; incl drain ", Float64(t2 - t0) / Float64(reps) / 1000.0, " us")

    bench("memset+launch", 0, ctx, kfn, zfn, buf, small, hsmall, zbuf, n)
    bench("2 launches+memset", 1, ctx, kfn, zfn, buf, small, hsmall, zbuf, n)
    bench("h2d copy+2 launches", 2, ctx, kfn, zfn, buf, small, hsmall, zbuf, n)
    bench("3 launches", 3, ctx, kfn, zfn, buf, small, hsmall, zbuf, n)
    bench("zero-kernel+2 launches", 4, ctx, kfn, zfn, buf, small, hsmall, zbuf, n)
    bench("3 memsets", 5, ctx, kfn, zfn, buf, small, hsmall, zbuf, n)
    bench("h2d+d2h copies+launch", 6, ctx, kfn, zfn, buf, small, hsmall, zbuf, n)
    _ = buf^
    _ = big^
