# SPDX-License-Identifier: Apache-2.0
"""M3-only correctness/lifecycle fixture. M2 compiles; never time this driver.
Pinned proposal source: 9ab2d3d3fb770498ef025db08f595a0149792bb7.
Compile: mojo build -j 1 --target-cpu apple-m1 --target-accelerator metal:1 -I . -D MOJOLEARN_APPLE_FAST_CALLPATH_CANDIDATES
         bench/apple_callpath_quality_main.mojo -o apple-callpath-quality
"""
from std.memory import bitcast
from experiments.apple_callpath.context_owner import CallpathContext
from experiments.apple_callpath.resident_slot import ResidentCallSlot, wait_pair, CALLPATH_ENABLED
from experiments.apple_callpath.packed_readback import PackedReadback
from experiments.apple_callpath.minmax_adapter import enqueue_minmax_transform
from preprocessing.minmax import minmax_transform_kernel
from metrics.checks.device_io import upload_f32, download_f32


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def values(n: Int, round: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):
        var j = (i + round) % 8
        var bits = UInt32(0x3F800000)
        if j == 0:
            bits = 0x80000000  # negative zero
        elif j == 1:
            bits = 0x00000001  # subnormal
        elif j == 2:
            bits = 0x80000001
        elif j == 3:
            bits = 0x7FC00123  # quiet NaN payload
        elif j == 4:
            bits = 0x7F800000
        elif j == 5:
            bits = 0xFF800000
        elif j == 6:
            bits = 0xC0200000
        out[i] = bitcast[DType.float32](bits)
    return out^


def exact(a: List[Float32], b: List[Float32]) raises:
    require(len(a) == len(b), "length differs")
    for i in range(len(a)):
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), "output word differs")


def transfer_case(ctx: CallpathContext, foreign: CallpathContext, n: Int) raises:
    var first = ResidentCallSlot(ctx, n, n, 3, 3)
    var second = ResidentCallSlot(ctx, n, n, 3, 3)
    var packed = PackedReadback(ctx, 2 * n)
    var out1 = List[Float32](length=n, fill=Float32(0))
    var out2 = List[Float32](length=n, fill=Float32(0))
    for round in range(3):
        var input1 = values(n, round)
        var input2 = values(n, round + 1)
        first.stage(input1)
        second.stage(input2)
        var wrong_context = False
        try:
            first.begin(foreign)
        except:
            wrong_context = True
        require(wrong_context and first.phase == 0, "foreign context accepted or changed slot")
        first.begin(ctx)
        var occupied = False
        try:
            first.begin(ctx)
        except:
            occupied = True
        second.begin(ctx)
        if n > 0:
            ctx.device.enqueue_copy(dst_buf=first.device_output, src_buf=first.device_input)
            ctx.device.enqueue_copy(dst_buf=second.device_output, src_buf=second.device_input)
        var scratch = values(3, round)
        var scratch_out = List[Float32](length=3, fill=Float32(0))
        var indexes = List[Int32](capacity=3)
        indexes.append(Int32(16777217))
        indexes.append(Int32(-2147483647))
        indexes.append(Int32(round))
        var indexes_out = List[Int32](length=3, fill=Int32(0))
        ctx.device.enqueue_copy(dst_buf=first.scratch, src_ptr=scratch.unsafe_ptr())
        ctx.device.enqueue_copy(dst_ptr=scratch_out.unsafe_ptr(), src_buf=first.scratch)
        ctx.device.enqueue_copy(dst_buf=first.index_scratch, src_ptr=indexes.unsafe_ptr())
        ctx.device.enqueue_copy(dst_ptr=indexes_out.unsafe_ptr(), src_buf=first.index_scratch)
        first.seal(ctx)
        second.seal(ctx)
        var premature = False
        try:
            first.collect_into(out1)
        except:
            premature = True
        var restage = False
        try:
            first.stage(input1)
        except:
            restage = True
        wait_pair(ctx, first, second)
        first.collect_into(out1)
        second.collect_into(out2)
        require(occupied and premature and restage, "occupied/visibility guard failed")
        exact(input1, out1)
        exact(input2, out2)
        exact(scratch, scratch_out)
        for i in range(3):
            require(indexes[i] == indexes_out[i], "integer scratch lost bits")
        var a = packed.append(ctx, first.device_output, 0, n)
        var b = packed.append(ctx, second.device_output, 0, n)
        var overfull = False
        try:
            _ = packed.append(ctx, first.device_output, 0, 1)
        except:
            overfull = True
        packed.finish(ctx)
        packed.collect_into(a, out1)
        packed.collect_into(b, out2)
        exact(input1, out1)
        exact(input2, out2)
        require(overfull, "packed capacity guard failed")
        packed.reset(ctx)
        var after_reset = False
        try:
            packed.collect_into(0, out1)
        except:
            after_reset = True
        require(after_reset, "readback readable after reset")
    # Simulate failure after begin: drain must release the claim only after wait.
    var again = values(n, 7)
    first.stage(again)
    first.begin(ctx)
    first.drain(ctx)
    require(first.phase == 0 and not first.staged, "drain did not clear slot")
    first.stage(again)
    first.begin(ctx)
    if n > 0:
        ctx.device.enqueue_copy(dst_buf=first.device_output, src_buf=first.device_input)
    first.seal(ctx)
    first.wait(ctx)
    first.collect_into(out1)
    exact(again, out1)
    _ = packed^
    _ = first^
    _ = second^
    ctx.device.synchronize()
    print("CALLPATH-QUALITY C1+C2+C3 n=", n, " status=PASS", sep="")


def minmax_case(ctx: CallpathContext, n: Int) raises:
    var scale_values = List[Float32](capacity=3)
    scale_values.append(Float32(2))
    scale_values.append(Float32(-0.5))
    scale_values.append(Float32(0.0001))
    var offset_values = List[Float32](capacity=3)
    offset_values.append(Float32(1))
    offset_values.append(Float32(-2))
    offset_values.append(Float32(0))
    var scale = upload_f32(ctx.device, scale_values)
    var offset = upload_f32(ctx.device, offset_values)
    var slot = ResidentCallSlot(ctx, n, n, 0, 0)
    var result = List[Float32](length=n, fill=Float32(0))
    for variant in range(3):
        var input = values(n, variant)
        var baseline_in = upload_f32(ctx.device, input)
        var baseline_out = ctx.device.enqueue_create_buffer[DType.float32](max(n, 1))
        var inverse = Int32(1 if variant == 2 else 0)
        var clip = Int32(1 if variant == 1 else 0)
        if n > 0:
            ctx.device.enqueue_function[minmax_transform_kernel](
                baseline_in.unsafe_ptr(), scale.unsafe_ptr(), offset.unsafe_ptr(), baseline_out.unsafe_ptr(),
                Int32(n), Int32(3), inverse, clip, Float32(-1), Float32(1),
                grid_dim=(n + 255) // 256, block_dim=256)
        var reference = download_f32(ctx.device, baseline_out, n)
        slot.stage(input)
        enqueue_minmax_transform(ctx, slot, scale, offset, 3, inverse, clip, Float32(-1), Float32(1))
        slot.wait(ctx)
        slot.collect_into(result)
        exact(reference, result)
        _ = baseline_in^
        _ = baseline_out^
        ctx.device.synchronize()
    _ = slot^
    _ = scale^
    _ = offset^
    ctx.device.synchronize()
    print("CALLPATH-QUALITY C4 n=", n, " status=PASS", sep="")


def run_quality() raises:
    comptime if not CALLPATH_ENABLED:
        raise Error("requires opt-in FAST Apple build")
    var original = CallpathContext()
    var slot_before_move = ResidentCallSlot(original, 0, 0, 0, 0)
    var slab_before_move = PackedReadback(original, 0)
    var ctx = original^
    # Slots constructed before a move must still recognize the same context.
    slot_before_move.check_context(ctx)
    slab_before_move.check_context(ctx)
    var foreign = CallpathContext()
    var rejected_slot = False
    var rejected_slab = False
    try:
        slot_before_move.check_context(foreign)
    except:
        rejected_slot = True
    try:
        slab_before_move.check_context(foreign)
    except:
        rejected_slab = True
    require(rejected_slot and rejected_slab, "foreign context accepted after owner move")
    transfer_case(ctx, foreign, 0)
    transfer_case(ctx, foreign, 1)
    transfer_case(ctx, foreign, 257)
    transfer_case(ctx, foreign, 4099)
    minmax_case(ctx, 0)
    minmax_case(ctx, 777)
    ctx.device.synchronize()
    foreign.device.synchronize()
    print("CALLPATH-QUALITY status=PASS variants=C1,C2,C3,C4 first_read=all_words mode=FAST vendor=Apple reach=C1+C2+C3+C4 timing=NONE")
