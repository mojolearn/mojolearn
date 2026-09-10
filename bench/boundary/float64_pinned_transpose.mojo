# SPDX-License-Identifier: Apache-2.0
"""WP1b staging-only A/B; imports the production transpose, no duplicate.

Build (only in a granted slot):
 tools/with_build_lock.sh nice -n 19 pixi run mojo build -I . -I bindings \
   bench/boundary/float64_pinned_transpose.mojo -o /tmp/wp1b-transpose
Run under the same build lock, nice -n 19. No fit or DMA is timed.
"""
from _mojolearn import _tiled_transpose_to_f32
from hostptr import copy_f32
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.time import perf_counter_ns


def run_case(ctx: DeviceContext, nr: Int, nc: Int) raises:
    var n = nr * nc
    var source = alloc[Float64](n)
    var temporary = alloc[Float32](n)
    var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(source))
    var tmp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(temporary))
    # Asymmetric coordinates expose a wrong transpose, and fractional values
    # exercise actual Float64-to-Float32 rounding rather than exact integers.
    for r in range(nr):
        for c in range(nc):
            var v = Float64((r*73+c*131)%65521) / 997.0 - 30.0
            if (r*nc+c)%4093 == 0:
                v = bitcast[DType.float64](UInt64(0x8000000000000000))
            src.unsafe_store(r*nc+c, v)
    var direct = ctx.enqueue_create_host_buffer[DType.float32](n)
    var staged = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var dp = direct.unsafe_ptr()
    var sp = staged.unsafe_ptr()
    var base_first = Float64(0)
    var base_last = Float64(0)
    var base_min = Float64(1e30)
    var direct_min = Float64(1e30)
    for rep in range(6):
        var direct_ms = Float64(0)
        var base_ms = Float64(0)
        for step in range(2):
            var arm = (rep + step) % 2
            var t0 = perf_counter_ns()
            if arm == 0:
                _tiled_transpose_to_f32(src, dp, nr, nc)
            else:
                _tiled_transpose_to_f32(src, tmp, nr, nc)
                copy_f32(tmp, sp, n)
            var elapsed = Float64(perf_counter_ns()-t0)/1e6
            if arm == 0:
                direct_ms = elapsed
            else:
                base_ms = elapsed
        # Untimed exact bits against BOTH other arm and independent address
        # oracle. Neither matching each other nor repeated sums is sufficient.
        for c in range(nc):
            for r in range(nr):
                var i = c*nr+r
                var expected = src.unsafe_load(r*nc+c).cast[DType.float32]()
                if bitcast[DType.uint32](dp.unsafe_load(i)) != bitcast[DType.uint32](expected) or bitcast[DType.uint32](sp.unsafe_load(i)) != bitcast[DType.uint32](expected):
                    raise Error("WP1b per-cell bits differ")
        print("WP1b rows", nr, "features", nc, "rep", rep,
              "warmup", rep == 0, "direct_pinned_ms", direct_ms,
              "malloc_then_simd_pinned_ms", base_ms, "bits_exact", True)
        if rep > 0:
            base_min = min(base_min, base_ms)
            direct_min = min(direct_min, direct_ms)
            if rep == 1:
                base_first = base_ms
            base_last = base_ms
    print("WP1b summary rows", nr, "features", nc,
          "direct_min_ms", direct_min, "baseline_min_ms", base_min,
          "baseline_endpoint_ratio", base_last/base_first,
          "scope", "host_staging_only_no_DMA_or_fit")
    source.free()
    temporary.free()


def main() raises:
    var ctx = DeviceContext()
    print("WP1b actual helper _mojolearn._tiled_transpose_to_f32 tiles128x64")
    run_case(ctx, 1_000_000, 28)
    run_case(ctx, 2_000_000, 20)
