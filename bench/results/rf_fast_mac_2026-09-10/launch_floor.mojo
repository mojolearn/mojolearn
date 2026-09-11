from max.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, block_dim


def touch_kernel(p: MutPointer[Float32, MutAnyOrigin], n: Int32, v: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        p[i] = p[i] + v


def main() raises:
    var ctx = DeviceContext()
    var n = 1024
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    var small = ctx.enqueue_create_buffer[DType.float32](64)
    var hsmall = ctx.enqueue_create_host_buffer[DType.float32](64)
    ctx.enqueue_memset(buf, Float32(0))
    ctx.synchronize()
    var kfn = ctx.compile_function[touch_kernel]()
    # warm
    for _ in range(100):
        ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
    ctx.synchronize()
    var reps = 5000
    var t0 = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
    var t1 = perf_counter_ns()
    ctx.synchronize()
    var t2 = perf_counter_ns()
    print("enqueue_function: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us/launch host; drain ", Float64(t2 - t1) / 1e6, " ms total; per launch incl drain ", Float64(t2 - t0) / Float64(reps) / 1000.0, " us")
    # small H2D upload per launch
    t0 = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_copy(dst_buf=small, src_buf=hsmall)
        ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
    t1 = perf_counter_ns()
    ctx.synchronize()
    t2 = perf_counter_ns()
    print("copy+launch: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us/pair host; incl drain ", Float64(t2 - t0) / Float64(reps) / 1000.0, " us")
    # launch + sync each (the worst case)
    reps = 500
    t0 = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(kfn, buf, Int32(n), Float32(1.0), grid_dim=(n // 256), block_dim=(256))
        ctx.synchronize()
    t1 = perf_counter_ns()
    print("launch+sync: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us each")
    # D2H small readback + sync
    t0 = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_copy(dst_buf=hsmall, src_buf=small)
        ctx.synchronize()
    t1 = perf_counter_ns()
    print("d2h copy+sync: ", Float64(t1 - t0) / Float64(reps) / 1000.0, " us each")
    ctx.enqueue_copy(dst_buf=host, src_buf=buf)
    ctx.synchronize()
    print("check ", host[0])
