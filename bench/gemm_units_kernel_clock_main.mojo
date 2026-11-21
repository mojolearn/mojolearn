# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE KERNEL CLOCK OF THE MATRIX UNITS: FP32 vs TF32 vs BF16 ON DEVICE-RESIDENT OPERANDS
(lane inv-bf16-kernel, 2026-11-20; rows `kernel-clock-units-*` of bench/results/paper_compare/owed-pre0867.json).

WHY. The board's gemm row (`mojolearn.linalg.matmul` on two float32 [4096, 4096] host arrays) is a transport row:
its clock holds two 64 MB uploads, one 64 MB download into fresh host pages and one product, and the product is
about a tenth of it. A/B abf2 moved it 64.8 -> 60.0 ms (TF32) and 58.8 ms (BF16): both arms removed the same
4.8 ms, which is most of the FP32 product, and what is left of the row cannot show which unit is faster. This
program times the PRODUCT alone, operands and output resident on the device, and times the transport stages
beside it (never inside it), so the three precisions are compared on the multiply.

BUILD: a `fast` build on NVIDIA or AMD with the matrix-units route and the resident bf16 arm:
    -D MOJOLEARN_NUMERIC_IDENTICAL -D MOJOLEARN_MODE_FAST_ROWS -D MOJOLEARN_GEMM_FAST_BF16_RESIDENT
In any other build it prints one refusal line and exits (an identical build refuses the define at compile time).

ARMS, per shape, C[m x n] = A[m x k] . B[n x k]^T (the forward form of a layer: B is the weight):
    fp32-exact     gemm/checks/gemm_identical.mojo::identical_gemm_into, vendor route closed: the identical kernel
    tf32-unit      gemm/fast_matrix_units.mojo::vendor_matmul_gemm_into: the default of `fast` (one launch)
    bf16-percall   both operands narrowed from FP32 by two launches, then the bf16 product: the launches of the
                   deleted arm MOJOLEARN_GEMM_FAST_BF16_UNITS (three launches)
    bf16-resident  gemm/fast_bf16_resident.mojo::bf16_resident_product over images narrowed before the clock
                   (one launch); `narrow_once_ms` is the one-time cost, reported beside it
Each arm: one warm-up call outside the clock, then KC_CALLS consecutive calls, each [launch, synchronize] inside its
own clock, arms alternated call by call so drift is shared; the line carries the median, the minimum and the
maximum. Quality: max |C_arm - C_fp32| / max |C_fp32| over the whole output, reduced on the device (one thread a
row), for every arm. TFLOP/s = 2 m n k / median.

TRANSPORT, per shape, beside the product (the product's own functions, gemm/host_transport.mojo): `upload_ms` (A
from pageable host memory through the pinned stage), `download_fresh_ms` (C into host pages never touched: what a
public call that returns a new array pays) and `download_ms` (C into the same pages again). These are the
`upload_ms_separate` of the row.

THE CHAIN ROW (the MLP work): a three-product inference chain on resident weights,
    H1 = relu(X W1^T + b1), H2 = relu(H1 W2^T + b2), Y = H2 W3^T + b3,
as chain-fp32 (exact products, FP32 epilogue launches), chain-tf32 (unit products, FP32 epilogue launches: the
launches of the default route) and chain-bf16-resident (weight images resident, X narrowed by one launch, each
hidden epilogue stores the bf16 image the next product reads, FP32 output: the same number of launches as
chain-tf32). Quality: max |Y_arm - Y_fp32| / max |Y_fp32|.

SHAPES (KC_SHAPES below): sizes only, each with its reason. No board dataset, no data shape: operands are a
device-filled pseudo-random field in [-0.5, 0.5).

ONE DEVICE QUEUE. A device queue is serial by nature and a timing must own the device, so this program is one
process on one GPU and uses no host cores beyond its own thread; the host loops here are over shapes, arms and
calls (driver loops), and one reduction over m row maxima. It gets the smallest host that holds its GPU.
"""
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gemm.contract import OP_NT
from gemm.checks.gemm_identical import identical_gemm_into as fp32_gemm_into
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats as fp32_workspace_max_floats
from gemm.fast_matrix_units import vendor_matmul_gemm_into
from gemm.fast_bf16_resident import (
    BF16_EPI_RELU,
    Bf16Image,
    Bf16Resident,
    GEMM_FAST_BF16_RESIDENT,
    bf16_image_narrow,
    bf16_resident_epilogue,
    bf16_resident_product,
)
from gemm.host_transport import GemmHostLease, gemm_down_f32, gemm_up_f32

#: consecutive scored calls an arm (AGENTS.md "Small rows": the median of N = 20 consecutive calls)
comptime KC_CALLS = 20
comptime KC_TPB = 256
comptime KC_ARMS = 4

comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _HP = MutPointer[Float32, MutUntrackedOrigin]


def kc_fill_kernel(p: _FP, count: Int64, salt: UInt32):
    """p[i] = a fixed pseudo-random value in [-0.5, 0.5): 16 bits of a multiply-xorshift hash of (i, salt)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count):
        return
    var x = (UInt32(i) + salt) * UInt32(2654435761)
    x = x ^ (x >> UInt32(15))
    x = x * UInt32(2246822519)
    x = x ^ (x >> UInt32(13))
    p.unsafe_store(i, Float32(Int(x & UInt32(65535)) - 32768) / Float32(65536.0))


def kc_diff_kernel(stats: _FP, c: _FP, ref_c: _FP, m: Int64, n: Int64):
    """One thread a row r: stats[2 r] = max_j |c - ref_c|, stats[2 r + 1] = max_j |ref_c| over the row's n cells."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(m):
        return
    var worst = Float32(0.0)
    var scale = Float32(0.0)
    var base = r * Int(n)
    for j in range(Int(n)):
        var want = ref_c.unsafe_load(base + j)
        var d = c.unsafe_load(base + j) - want
        if d < Float32(0.0):
            d = -d
        if want < Float32(0.0):
            want = -want
        if d > worst:
            worst = d
        if want > scale:
            scale = want
    stats.unsafe_store(2 * r, worst)
    stats.unsafe_store(2 * r + 1, scale)


def kc_bias_relu_kernel(c: _FP, bias: _FP, n: Int64, count: Int64, relu: Int64):
    """c[i] = act(c[i] + bias[i mod n]) in place, FP32: the epilogue launch of the FP32 and TF32 chains."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count):
        return
    var v = c.unsafe_load(i) + bias.unsafe_load(i % Int(n))
    if Int(relu) != 0:
        if v < Float32(0.0):
            v = Float32(0.0)
    c.unsafe_store(i, v)


def _fill(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], count: Int, salt: Int) raises:
    var p = buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[kc_fill_kernel](
        p, Int64(count), UInt32(salt),
        grid_dim=((count + KC_TPB - 1) // KC_TPB, 1, 1), block_dim=(KC_TPB, 1, 1),
    )


def _bias_relu(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32], mut bias: DeviceBuffer[DType.float32],
    m: Int, n: Int, relu: Bool,
) raises:
    var pc = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pb = bias.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[kc_bias_relu_kernel](
        pc, pb, Int64(n), Int64(m * n), Int64(1 if relu else 0),
        grid_dim=((m * n + KC_TPB - 1) // KC_TPB, 1, 1), block_dim=(KC_TPB, 1, 1),
    )


def _rel_diff(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32], mut ref_c: DeviceBuffer[DType.float32], m: Int, n: Int
) raises -> Float64:
    """max |c - ref_c| / max |ref_c| over m x n cells: the row maxima on the device, their maximum here."""
    var stats = ctx.enqueue_create_buffer[DType.float32](2 * m)
    var host = ctx.enqueue_create_host_buffer[DType.float32](2 * m)
    var ps = stats.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pc = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pr = ref_c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[kc_diff_kernel](
        ps, pc, pr, Int64(m), Int64(n),
        grid_dim=((m + KC_TPB - 1) // KC_TPB, 1, 1), block_dim=(KC_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=stats)
    ctx.synchronize()
    var worst = Float64(0.0)
    var scale = Float64(0.0)
    for r in range(m):  # driver-loop: over the m row maxima the device reduced (2 m floats, never the cells)
        var d = Float64(host.unsafe_ptr().unsafe_load(2 * r))
        var s = Float64(host.unsafe_ptr().unsafe_load(2 * r + 1))
        if d > worst:
            worst = d
        if s > scale:
            scale = s
    _ = stats^
    _ = host^
    if scale <= Float64(0.0):
        return Float64(-1.0)
    return worst / scale


def _sorted(samples: List[Int]) -> List[Int]:
    """The samples ascending (an insertion sort of KC_CALLS values)."""
    var s = samples.copy()
    for i in range(1, len(s)):  # driver-loop: over the KC_CALLS timings of one arm
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    return s^


def _ms(ns: Int) -> Float64:
    return Float64(ns) / Float64(1000000.0)


def _arm_name(arm: Int) -> String:
    if arm == 0:
        return String("fp32-exact")
    if arm == 1:
        return String("tf32-unit")
    if arm == 2:
        return String("bf16-percall")
    return String("bf16-resident")


def _report(row: String, arm: String, m: Int, n: Int, k: Int, samples: List[Int], rel: Float64, launches: Int):
    var s = _sorted(samples)
    var median = (s[(len(s) - 1) // 2] + s[len(s) // 2]) // 2
    var tflops = Float64(2.0) * Float64(m) * Float64(n) * Float64(k) / (Float64(median) * Float64(1000.0))
    print(
        "KCLOCK", row, "arm", arm, "m", m, "n", n, "k", k, "calls", len(s),
        "median_ms", _ms(median), "min_ms", _ms(s[0]), "max_ms", _ms(s[len(s) - 1]),
        "tflops", tflops, "launches_per_call", launches, "max_rel_diff_vs_fp32", rel,
    )


def run_product(ctx: DeviceContext, name: String, m: Int, n: Int, k: Int, host_block: _HP, host_cells: Int) raises:
    """One shape: the four product arms on resident operands, then the transport stages beside them."""
    comptime if GEMM_FAST_BF16_RESIDENT:
        var a = ctx.enqueue_create_buffer[DType.float32](m * k)
        var b = ctx.enqueue_create_buffer[DType.float32](n * k)
        var c_ref = ctx.enqueue_create_buffer[DType.float32](m * n)
        var c = ctx.enqueue_create_buffer[DType.float32](m * n)
        var ws = ctx.enqueue_create_buffer[DType.float32](max(1, fp32_workspace_max_floats(m, n, k)))
        _fill(ctx, a, m * k, 17)
        _fill(ctx, b, n * k, 31)
        # the reference output, and the resident images (narrowed ONCE, timed as narrow_once_ms)
        fp32_gemm_into[False](ctx, c_ref, a, b, ws, m, n, k, OP_NT)
        ctx.synchronize()
        var img_a = Bf16Image(ctx, m * k)
        var img_b = Bf16Image(ctx, n * k)
        var scratch_a = Bf16Image(ctx, m * k)
        var scratch_b = Bf16Image(ctx, n * k)
        ctx.synchronize()
        var t_narrow = Int(perf_counter_ns())
        bf16_image_narrow(ctx, img_a, a, m * k)
        bf16_image_narrow(ctx, img_b, b, n * k)
        ctx.synchronize()
        var narrow_ns = Int(perf_counter_ns()) - t_narrow

        var samples = List[List[Int]]()
        var rels = List[Float64]()
        for arm in range(KC_ARMS):  # driver-loop: over the four arms
            samples.append(List[Int]())
            rels.append(Float64(-1.0))
        # rep 0 is every arm's warm-up (outside the record) and the call whose output is compared
        for rep in range(KC_CALLS + 1):  # driver-loop: over the scored calls
            for arm in range(KC_ARMS):  # driver-loop: over the four arms, alternated call by call
                var t0 = Int(perf_counter_ns())
                if arm == 0:
                    fp32_gemm_into[False](ctx, c, a, b, ws, m, n, k, OP_NT)
                elif arm == 1:
                    if not vendor_matmul_gemm_into(ctx, c, a, b, ws, m, n, k, OP_NT):
                        raise Error("kernel clock: the matrix-unit product refused an OP_NT product")
                elif arm == 2:
                    bf16_image_narrow(ctx, scratch_a, a, m * k)
                    bf16_image_narrow(ctx, scratch_b, b, n * k)
                    bf16_resident_product(ctx, c, scratch_a, scratch_b, m, n, k, True)
                else:
                    bf16_resident_product(ctx, c, img_a, img_b, m, n, k, True)
                ctx.synchronize()
                var dt = Int(perf_counter_ns()) - t0
                if rep == 0:
                    rels[arm] = _rel_diff(ctx, c, c_ref, m, n)
                else:
                    samples[arm].append(dt)
        for arm in range(KC_ARMS):  # driver-loop: over the four arms
            _report(name, _arm_name(arm), m, n, k, samples[arm], rels[arm], 3 if arm == 2 else 1)
        print("KCLOCK", name, "narrow_once_ms", _ms(narrow_ns), "cells", (m + n) * k,
              "note", "both operands to resident bf16 images, one time, outside every product clock")

        # TRANSPORT, beside the product: the public call's own upload and download functions.
        if m * k <= host_cells and m * n <= host_cells:
            var lease = GemmHostLease()
            # C into host pages never touched (a block of this shape's own, left to the process: a bench), then
            # into the same pages again; the shared block is the upload source
            var fresh_raw = alloc[Float32](m * n)
            var fresh = _HP(unsafe_from_address=Int(fresh_raw))
            var t_df = Int(perf_counter_ns())
            gemm_down_f32(ctx, lease, c_ref, fresh, m * n)
            var down_fresh_ns = Int(perf_counter_ns()) - t_df
            var t_d = Int(perf_counter_ns())
            gemm_down_f32(ctx, lease, c_ref, fresh, m * n)
            var down_ns = Int(perf_counter_ns()) - t_d
            gemm_down_f32(ctx, lease, a, host_block, m * k)
            var t_u = Int(perf_counter_ns())
            gemm_up_f32(ctx, lease, a, host_block, m * k)
            ctx.synchronize()
            var up_ns = Int(perf_counter_ns()) - t_u
            lease.release()
            print(
                "KCLOCK", name, "transport", "upload_ms", _ms(up_ns), "upload_mb", Float64(m * k) * 4.0 / 1048576.0,
                "download_fresh_ms", _ms(down_fresh_ns), "download_ms", _ms(down_ns),
                "download_mb", Float64(m * n) * 4.0 / 1048576.0,
                "note", "download_fresh is into host pages touched for the first time; download is the same pages again",
            )
        else:
            print("KCLOCK", name, "transport", "skipped", "host block holds", host_cells, "floats")
        ctx.synchronize()
        _ = a^
        _ = b^
        _ = c^
        _ = c_ref^
        _ = ws^
        _ = img_a^
        _ = img_b^
        _ = scratch_a^
        _ = scratch_b^


def run_chain(ctx: DeviceContext, name: String, m: Int, d: Int, h: Int, o: Int) raises:
    """The MLP inference chain on resident weights: m rows, d features, two hidden layers of h, o outputs."""
    comptime if GEMM_FAST_BF16_RESIDENT:
        var x = ctx.enqueue_create_buffer[DType.float32](m * d)
        var w1 = ctx.enqueue_create_buffer[DType.float32](h * d)
        var w2 = ctx.enqueue_create_buffer[DType.float32](h * h)
        var w3 = ctx.enqueue_create_buffer[DType.float32](o * h)
        var b1 = ctx.enqueue_create_buffer[DType.float32](h)
        var b2 = ctx.enqueue_create_buffer[DType.float32](h)
        var b3 = ctx.enqueue_create_buffer[DType.float32](o)
        var h1 = ctx.enqueue_create_buffer[DType.float32](m * h)
        var h2 = ctx.enqueue_create_buffer[DType.float32](m * h)
        var y = ctx.enqueue_create_buffer[DType.float32](m * o)
        var y_ref = ctx.enqueue_create_buffer[DType.float32](m * o)
        var ws_n = max(fp32_workspace_max_floats(m, h, d), max(fp32_workspace_max_floats(m, h, h), fp32_workspace_max_floats(m, o, h)))
        var ws = ctx.enqueue_create_buffer[DType.float32](max(1, ws_n))
        _fill(ctx, x, m * d, 3)
        _fill(ctx, w1, h * d, 5)
        _fill(ctx, w2, h * h, 7)
        _fill(ctx, w3, o * h, 11)
        _fill(ctx, b1, h, 13)
        _fill(ctx, b2, h, 19)
        _fill(ctx, b3, o, 23)
        # the resident weight images: narrowed once, by their owner, before any call
        var owner = Bf16Resident()
        var i1 = owner.image_of(ctx, w1, h * d)
        var i2 = owner.image_of(ctx, w2, h * h)
        var i3 = owner.image_of(ctx, w3, o * h)
        var img_x = Bf16Image(ctx, m * d)
        var img_h1 = Bf16Image(ctx, m * h)
        var img_h2 = Bf16Image(ctx, m * h)
        ctx.synchronize()

        var samples = List[List[Int]]()
        var rels = List[Float64]()
        for arm in range(3):  # driver-loop: over the three chain arms
            samples.append(List[Int]())
            rels.append(Float64(-1.0))
        for rep in range(KC_CALLS + 1):  # driver-loop: over the scored calls
            for arm in range(3):  # driver-loop: over the three chain arms, alternated call by call
                var t0 = Int(perf_counter_ns())
                if arm == 0:
                    fp32_gemm_into[False](ctx, h1, x, w1, ws, m, h, d, OP_NT)
                    _bias_relu(ctx, h1, b1, m, h, True)
                    fp32_gemm_into[False](ctx, h2, h1, w2, ws, m, h, h, OP_NT)
                    _bias_relu(ctx, h2, b2, m, h, True)
                    fp32_gemm_into[False](ctx, y, h2, w3, ws, m, o, h, OP_NT)
                    _bias_relu(ctx, y, b3, m, o, False)
                elif arm == 1:
                    if not vendor_matmul_gemm_into(ctx, h1, x, w1, ws, m, h, d, OP_NT):
                        raise Error("kernel clock: the matrix-unit product refused layer 1")
                    _bias_relu(ctx, h1, b1, m, h, True)
                    if not vendor_matmul_gemm_into(ctx, h2, h1, w2, ws, m, h, h, OP_NT):
                        raise Error("kernel clock: the matrix-unit product refused layer 2")
                    _bias_relu(ctx, h2, b2, m, h, True)
                    if not vendor_matmul_gemm_into(ctx, y, h2, w3, ws, m, o, h, OP_NT):
                        raise Error("kernel clock: the matrix-unit product refused layer 3")
                    _bias_relu(ctx, y, b3, m, o, False)
                else:
                    # X arrives FP32 (the public boundary): one narrow, the launch that brings the batch in
                    bf16_image_narrow(ctx, img_x, x, m * d)
                    owner.product(ctx, h1, img_x, i1, m, h, d, True)
                    bf16_resident_epilogue(ctx, img_h1, h1, b1, m, h, BF16_EPI_RELU)
                    owner.product(ctx, h2, img_h1, i2, m, h, h, True)
                    bf16_resident_epilogue(ctx, img_h2, h2, b2, m, h, BF16_EPI_RELU)
                    owner.product(ctx, y, img_h2, i3, m, o, h, True)
                    _bias_relu(ctx, y, b3, m, o, False)
                ctx.synchronize()
                var dt = Int(perf_counter_ns()) - t0
                if rep == 0:
                    if arm == 0:
                        ctx.enqueue_copy(dst_buf=y_ref, src_buf=y)
                        ctx.synchronize()
                        rels[arm] = Float64(0.0)
                    else:
                        rels[arm] = _rel_diff(ctx, y, y_ref, m, o)
                else:
                    samples[arm].append(dt)
        # work of the chain in multiply-adds, as one (m, n, k) triple for the TFLOP/s column: m rows x (h d + h h + o h)
        var per_row = h * d + h * h + o * h
        _report(name, String("chain-fp32"), m, per_row, 1, samples[0], rels[0], 6)
        _report(name, String("chain-tf32"), m, per_row, 1, samples[1], rels[1], 6)
        _report(name, String("chain-bf16-resident"), m, per_row, 1, samples[2], rels[2], 7)
        print("KCLOCK", name, "chain", "rows", m, "features", d, "hidden", h, "outputs", o,
              "weight_narrows_total", owner.narrows, "note",
              "n is the multiply-adds a row (h d + h h + o h) and k is 1 on the chain lines; the bf16 chain's 7th launch is the narrow of X")
        ctx.synchronize()
        _ = x^
        _ = w1^
        _ = w2^
        _ = w3^
        _ = b1^
        _ = b2^
        _ = b3^
        _ = h1^
        _ = h2^
        _ = y^
        _ = y_ref^
        _ = ws^
        _ = img_x^
        _ = img_h1^
        _ = img_h2^
        _ = owner^


def main() raises:
    comptime if not GEMM_FAST_BF16_RESIDENT:
        print("KCLOCK REFUSED: build with -D MOJOLEARN_NUMERIC_IDENTICAL -D MOJOLEARN_MODE_FAST_ROWS -D MOJOLEARN_GEMM_FAST_BF16_RESIDENT (NVIDIA or AMD column)")
        return
    var ctx = DeviceContext()
    print("KCLOCK_DEVICE", ctx.name(), "calls", KC_CALLS)
    # One pageable host block for the transport stages: 4096 x 4096 floats (64 MB), the public gemm row's operand.
    # Larger shapes skip the transport lines (their rate is the same GB/s; the product lines are what they are for).
    var host_cells = 4096 * 4096
    var raw = alloc[Float32](host_cells)
    var host_block = _HP(unsafe_from_address=Int(raw))
    var only = String(getenv("MOJOLEARN_KCLOCK_ONLY"))

    # THE SQUARE LADDER: the public gemm row's shape with a factor of 8 in work on each side, so the unit's
    # rate is read where the product is 0.2 ms, 1.5 ms and 12 ms of TF32 arithmetic, not at one size.
    if only.byte_length() == 0 or only == "square":
        run_product(ctx, "square-2048", 2048, 2048, 2048, host_block, host_cells)
        run_product(ctx, "square-4096", 4096, 4096, 4096, host_block, host_cells)
        run_product(ctx, "square-8192", 8192, 8192, 8192, host_block, host_cells)
    # NOT SQUARE: the same work as square-4096 (2^36 multiply-adds) in three aspects: a long K (a weight
    # gradient's form: few cells, long sums), a tall batch against a narrow layer, and a wide output (a head).
    if only.byte_length() == 0 or only == "aspect":
        run_product(ctx, "longk-1024x1024x65536", 1024, 1024, 65536, host_block, host_cells)
        run_product(ctx, "tall-65536x1024x1024", 65536, 1024, 1024, host_block, host_cells)
        run_product(ctx, "wide-2048x32768x1024", 2048, 32768, 1024, host_block, host_cells)
    # LAYER-SIZED PRODUCTS: what one launch of a training step or an inference chunk multiplies. Each is a
    # generic layer form at two sizes a factor of 4 or more apart, so the launch-bound end and the
    # arithmetic-bound end are both on the table: a hidden layer at a 4096-row chunk (width 256 and 1024), a
    # transformer feed-forward up-projection at 8192 tokens (width 384 -> 1024 and 768 -> 3072), and the
    # 512 x 4096 x 4096 projection of bench/results/gemm-ceiling-20260930 (stored torch TF32 / BF16 kernel rates
    # on the L40S: 102.9 / 186.0 TFLOP/s, the comparison this arm is for).
    if only.byte_length() == 0 or only == "layer":
        run_product(ctx, "hidden-4096x256x256", 4096, 256, 256, host_block, host_cells)
        run_product(ctx, "hidden-4096x1024x1024", 4096, 1024, 1024, host_block, host_cells)
        run_product(ctx, "ffn-8192x1024x384", 8192, 1024, 384, host_block, host_cells)
        run_product(ctx, "ffn-8192x3072x768", 8192, 3072, 768, host_block, host_cells)
        run_product(ctx, "proj-512x4096x4096", 512, 4096, 4096, host_block, host_cells)
    # THE CHAIN: an MLP inference chunk end to end on resident weights, at a small and a large width and at a
    # 4096-row chunk and a whole 262144-row batch (the chunk is launch-bound; the batch is not).
    if only.byte_length() == 0 or only == "chain":
        run_chain(ctx, "chain-4096x128-h256", 4096, 128, 256, 64)
        run_chain(ctx, "chain-4096x512-h1024", 4096, 512, 1024, 64)
        run_chain(ctx, "chain-262144x128-h256", 262144, 128, 256, 64)
        run_chain(ctx, "chain-65536x512-h1024", 65536, 512, 1024, 64)
    ctx.synchronize()
    print("KCLOCK_DONE")
