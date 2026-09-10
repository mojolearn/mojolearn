# Times the host-side passes a tree fit takes on X today, at 2M x 20 f32.
# Pass A: bindings/_mojolearn_trees.mojo::_copy_f32  (scalar List.append)
# Pass B: builder.mojo::upload_dataset  (scalar store into pinned host buffer)
# Pass C: enqueue_copy host->device + synchronize
# Pass D: what a fused version would do: one SIMD f64->f32 cast written straight
#         into the pinned buffer, then the same DMA.
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

comptime N = 2_000_000 * 20

def ms(t0: Int, t1: Int) -> Float64:
    return Float64(t1 - t0) / 1e6

def main() raises:
    var src = alloc[Float32](N)
    var src64 = alloc[Float64](N)
    for i in range(N):
        src.unsafe_store(i, Float32(i % 977) * 0.125)
        src64.unsafe_store(i, Float64(i % 977) * 0.125)
    var ctx = DeviceContext()
    var d_data = ctx.enqueue_create_buffer[DType.float32](N)
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()

    var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(src))
    for rep in range(4):
        # A
        var t0 = perf_counter_ns()
        var out = List[Float32](capacity=N)
        for i in range(N):
            out.append(p[i])
        var t1 = perf_counter_ns()
        # B
        for i in range(len(out)):
            h_data.unsafe_ptr().unsafe_store(i, out[i])
        var t2 = perf_counter_ns()
        # B2: same List -> pinned copy but SIMD-8 through raw pointers
        var op = out.unsafe_ptr()
        var hp0 = h_data.unsafe_ptr()
        var j = 0
        while j < N - (N % 8):
            hp0.unsafe_store[width=8](j, op.unsafe_load[width=8](j))
            j += 8
        var t2b = perf_counter_ns()
        # B3: scalar store into a plain (non-pinned) malloc'd buffer
        var plain = alloc[Float32](N)
        for i in range(len(out)):
            plain.unsafe_store(i, out[i])
        var t2c = perf_counter_ns()
        plain.free()
        # C
        ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
        ctx.synchronize()
        var t3 = perf_counter_ns()
        # D: fused f64 -> f32 cast straight into the pinned buffer, SIMD 8
        var hp = h_data.unsafe_ptr()
        var i = 0
        var body = N - (N % 8)
        while i < body:
            hp.unsafe_store[width=8](i, src64.unsafe_load[width=8](i).cast[DType.float32]())
            i += 8
        while i < N:
            hp.unsafe_store(i, src64.unsafe_load(i).cast[DType.float32]())
            i += 1
        var t4 = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
        ctx.synchronize()
        var t5 = perf_counter_ns()
        print("rep", rep,
              "A list_append", ms(t0, t1),
              "B pinned_store", ms(t1, t2), "B2 simd_pinned", ms(t2, t2b), "B3 scalar_plain", ms(t2b, t2c),
              "C dma", ms(t2c, t3),
              "| D fused_cast_into_pinned", ms(t3, t4),
              "dma", ms(t4, t5), "ms")
        _ = out[0]
    src.free(); src64.free()
