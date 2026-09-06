# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-only, unrun index-layout qualification and component timing.

Run IDENTICAL with -I .; imports transfer helpers from gemv_serial_layout_main.
MOJOLEARN_KNN_LAYOUT_LARGE=1 enables rows32/index100000/features32 timing;
override ROWS/COLS/D with the same prefix. SAMPLES defaults9, minimum7.
Norms and selection use production kernels. Timing labels distinguish distance
only, transpose+distance, and unchanged selection; these are NOT end-to-end
kNN measurements. Allocations and norm calculation are excluded.
Large fixtures default to independently seeded pseudorandom queries/index
(PROFILE=3). PROFILE=0 retains the original duplicate-heavy fixture.
MOJOLEARN_KNN_LAYOUT_DUMP=1 prints every distance UInt32 for full cross-device
comparison; otherwise only small fixtures emit cells. Local comparisons always
check EVERY cell, including the large fixture.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from bench.gemv_serial_layout_main import _env_int, _upload, _read, _same
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.row_norms import NORM_TPB, row_norm_kernel
from neighbors.checks.pinned_distance_tile import PINNED_TILE_TPB, pinned_distance_tile_kernel
from neighbors.checks.transposed_index_distance_candidate import transposed_index_distance_into
from neighbors.checks.select_radix_identical import radix_topk_identical_kernel
from neighbors.impl.matrix.detail.select_radix import SELECT_BLOCK


def _values(rows: Int, d: Int, profile: Int, salt: Int = 3) -> List[Float32]:
    var out = List[Float32]()
    for i in range(rows):
        for p in range(d):
            # Repeated rows deliberately produce exact neighbor ties.
            var v = Float32(((i % 19) * 17 + p * 11) % 61 - 30) / Float32(32)
            if profile == 3:
                # Same dyadic generator as lanes_price_main._price_u01.
                # Independent query/index salts avoid deliberate self matches.
                var z = UInt64(i) * 0x9E3779B97F4A7C15 + UInt64(p + 1) * 0xBF58476D1CE4E5B9 + UInt64(salt + 1) * 0x94D049BB133111EB
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                z = z ^ (z >> 31)
                v = Float32(Int((z >> 40) & UInt64(0xFFFF))) / Float32(65536.0)
            elif profile == 1:
                v = Float32(4096) + Float32((i + p) % 3) / Float32(1024)
            elif profile == 2:
                var bits = UInt32(0)
                if (i + p) % 4 == 1:
                    bits = UInt32(2147483648)
                elif (i + p) % 4 == 2:
                    bits = UInt32(1)
                elif (i + p) % 4 == 3:
                    bits = UInt32(8388608)
                v = bitcast[DType.float32](bits)
            out.append(v)
    return out^


def _distance(
    ctx: DeviceContext, mut z: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
    mut yt: DeviceBuffer[DType.float32], mut qn: DeviceBuffer[DType.float32],
    mut yn: DeviceBuffer[DType.float32], r: Int, n: Int, d: Int, root: Int, arm: Int,
) raises:
    if arm == 0:
        ctx.enqueue_function[pinned_distance_tile_kernel](
            z.unsafe_ptr(), q.unsafe_ptr(), y.unsafe_ptr(), qn.unsafe_ptr(), yn.unsafe_ptr(),
            Int32(r), Int32(n), Int32(d), Int32(root),
            grid_dim=((r * n + PINNED_TILE_TPB - 1) // PINNED_TILE_TPB, 1, 1),
            block_dim=(PINNED_TILE_TPB, 1, 1),
        )
    else:
        transposed_index_distance_into(ctx, z, q, y, yt, qn, yn, r, n, d, root, arm == 1)


def _select(
    ctx: DeviceContext, mut z: DeviceBuffer[DType.float32],
    mut ov: DeviceBuffer[DType.float32], mut oi: DeviceBuffer[DType.uint32],
    mut bv: DeviceBuffer[DType.float32], mut bi: DeviceBuffer[DType.uint32],
    r: Int, n: Int, k: Int,
) raises:
    ctx.enqueue_function[radix_topk_identical_kernel](
        z.unsafe_ptr(), ov.unsafe_ptr(), oi.unsafe_ptr(), bv.unsafe_ptr(), bi.unsafe_ptr(),
        Int32(n), Int32(k), Int32(n), Int32(1),
        grid_dim=(r, 1, 1), block_dim=(SELECT_BLOCK, 1, 1),
    )


def _case(r: Int, n: Int, d: Int, profile: Int, root: Int, timing: Bool, samples: Int) raises:
    var hq = _values(r, d, profile, 5)
    var hy = _values(n, d, profile, 3)
    var k = min(8, n)
    with DeviceContext() as ctx:
        var q = _upload(ctx, hq)
        var y = _upload(ctx, hy)
        var yt = ctx.enqueue_create_buffer[DType.float32](n * d)
        var qn = ctx.enqueue_create_buffer[DType.float32](r)
        var yn = ctx.enqueue_create_buffer[DType.float32](n)
        var z = ctx.enqueue_create_buffer[DType.float32](r * n)
        var ov = ctx.enqueue_create_buffer[DType.float32](r * k)
        var oi = ctx.enqueue_create_buffer[DType.uint32](r * k)
        var bv = ctx.enqueue_create_buffer[DType.float32](2 * r * n)
        var bi = ctx.enqueue_create_buffer[DType.uint32](2 * r * n)
        var hi = ctx.enqueue_create_host_buffer[DType.uint32](r * k)
        ctx.enqueue_function[row_norm_kernel](qn.unsafe_ptr(), q.unsafe_ptr(), Int32(d), Int32(0),
            grid_dim=(r, 1, 1), block_dim=(NORM_TPB, 1, 1))
        ctx.enqueue_function[row_norm_kernel](yn.unsafe_ptr(), y.unsafe_ptr(), Int32(d), Int32(0),
            grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1))
        ctx.synchronize()
        var expected = List[Float32]()
        var expected_vals = List[Float32]()
        var expected_idx = List[UInt32]()
        for repeat in range(2):
            for arm in range(3):
                z.enqueue_fill(Float32(-987654))
                ov.enqueue_fill(Float32(-987654))
                oi.enqueue_fill(UInt32(4294967295))
                ctx.synchronize()
                _distance(ctx, z, q, y, yt, qn, yn, r, n, d, root, arm)
                ctx.synchronize()
                var got = _read(ctx, z)
                for i in range(len(got)):
                    if got[i] == Float32(-987654):
                        raise Error("distance poison survived")
                _select(ctx, z, ov, oi, bv, bi, r, n, k)
                ctx.synchronize()
                var vals = _read(ctx, ov)
                ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=oi)
                ctx.synchronize()
                if repeat == 0 and arm == 0:
                    expected = got.copy()
                    expected_vals = vals.copy()
                    for i in range(r * k):
                        expected_idx.append(hi.unsafe_ptr().unsafe_load(i))
                else:
                    _same(expected, got, "distance")
                    _same(expected_vals, vals, "selected distance")
                for i in range(r * k):
                    var ix = hi.unsafe_ptr().unsafe_load(i)
                    if ix != expected_idx[i] or ix >= UInt32(n) or vals[i] == Float32(-987654):
                        raise Error("selected index mismatch or poison")
                    if profile == 2 and ix != UInt32(i % k):
                        raise Error("zero-distance ties must select lowest indices")
        _same(hq, _read(ctx, q), "query mutated")
        _same(hy, _read(ctx, y), "index mutated")
        var ht = _read(ctx, yt)
        for j in range(n):
            for p in range(d):
                if bitcast[DType.uint32](hy[j * d + p]) != bitcast[DType.uint32](ht[p * n + j]):
                    raise Error("index transpose changed bits")
        if r * n <= 4096 or String(getenv("MOJOLEARN_KNN_LAYOUT_DUMP")) == "1":
            for i in range(r * n):
                print("CELL", r, n, d, profile, root, i, bitcast[DType.uint32](expected[i]))
        print("BITGATE PASS", r, n, d, profile, root, "distance_cells", r * n, "selected_pairs", r * k)
        if timing:
            print("TIMING components_only norms_and_allocations_excluded scratch_bytes", 4 * n * d)
            for warm in range(2):
                for arm in range(3):
                    _distance(ctx, z, q, y, yt, qn, yn, r, n, d, root, arm)
                    ctx.synchronize()
                _select(ctx, z, ov, oi, bv, bi, r, n, k)
                ctx.synchronize()
            for sample in range(samples):
                for position in range(4):
                    var arm = (sample + position) % 4
                    ctx.synchronize()
                    var start = perf_counter_ns()
                    if arm < 3:
                        _distance(ctx, z, q, y, yt, qn, yn, r, n, d, root, arm)
                    else:
                        _select(ctx, z, ov, oi, bv, bi, r, n, k)
                    ctx.synchronize()
                    var elapsed = Float64(perf_counter_ns() - start) / Float64(1000000)
                    var label = String("legacy_distance")
                    if arm == 1:
                        label = "transpose_plus_distance"
                    elif arm == 2:
                        label = "prepared_distance"
                    elif arm == 3:
                        label = "unchanged_selection"
                    print("SAMPLE", sample, label, elapsed)
            _same(expected, _read(ctx, z), "post-timing distance")
        _ = q^
        _ = y^
        _ = yt^
        _ = qn^
        _ = yn^
        _ = z^
        _ = ov^
        _ = oi^
        _ = bv^
        _ = bi^
        _ = hi^


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("index-layout gate requires IDENTICAL")
    var samples = _env_int("MOJOLEARN_KNN_LAYOUT_SAMPLES", 9)
    if samples < 7:
        raise Error("at least seven samples required")
    for profile in range(4):
        for root in range(2):
            _case(1, 1, 1, profile, root, False, samples)
            _case(3, 33, 31, profile, root, False, samples)
            _case(5, 65, 129, profile, root, False, samples)
    if String(getenv("MOJOLEARN_KNN_LAYOUT_LARGE")) == "1":
        var r = _env_int("MOJOLEARN_KNN_LAYOUT_ROWS", 32)
        var n = _env_int("MOJOLEARN_KNN_LAYOUT_COLS", 100000)
        var d = _env_int("MOJOLEARN_KNN_LAYOUT_D", 32)
        if r <= 0 or n <= 0 or d <= 0 or r > 1024 or n > 1000000 or d > 4096 or r * n > 16000000 or n * d > 16000000:
            raise Error("fixture exceeds bounded positive dimensions/storage")
        var profile = _env_int("MOJOLEARN_KNN_LAYOUT_PROFILE", 3)
        if profile != 0 and profile != 3:
            raise Error("large fixture PROFILE must be 0 (duplicates) or 3 (random)")
        _case(r, n, d, profile, 1, True, samples)
    print("KNN INDEX LAYOUT QUALIFICATION PASS")
