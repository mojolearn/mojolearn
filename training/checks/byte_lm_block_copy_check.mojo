# SPDX-License-Identifier: Apache-2.0
"""Independent nine-copy oracle, guarded tails, and observed offset sabotage."""
from std.memory import bitcast
from std.os import getenv
from max.gpu.host import DeviceContext, DeviceBuffer
from training.byte_lm_block_copy import byte_block_copy
from training.byte_lm_config import ByteConfig
from training.checks.train_loop import _upload, _copy_into, download_f32


def pattern(n: Int, salt: Int) -> List[Float32]:
    var values = List[Float32]()
    # Signed zeros, infinities, subnormals and NaN payloads must copy as bits.
    var bits: List[UInt32] = [0, 0x80000000, 1, 0x80000001,
        0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x3f800000]
    for i in range(n):
        if i % 17 < len(bits):
            values.append(bitcast[DType.float32](bits[(i + salt) % len(bits)]))
        else:
            values.append(bitcast[DType.float32](UInt32(0x3f000000 + i * 71 + salt)))
    return values^


def same(a: List[Float32], b: List[Float32], label: String) raises:
    if len(a) != len(b):
        raise Error(label + ": length differs")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(label + ": bits differ at " + String(i))


def check(ctx: DeviceContext, offsets: List[Int], label: String) raises:
    var initial = pattern(offsets[9] + 7, 3)
    var flat = _upload(ctx, initial)
    var fused_flat = _upload(ctx, initial)
    var reference_flat = _upload(ctx, initial)
    var initial0 = pattern(offsets[1] - offsets[0] + 5, 19)
    var b0 = _upload(ctx, initial0)
    var r0 = _upload(ctx, initial0)
    var initial1 = pattern(offsets[2] - offsets[1] + 5, 20)
    var b1 = _upload(ctx, initial1)
    var r1 = _upload(ctx, initial1)
    var initial2 = pattern(offsets[3] - offsets[2] + 5, 21)
    var b2 = _upload(ctx, initial2)
    var r2 = _upload(ctx, initial2)
    var initial3 = pattern(offsets[4] - offsets[3] + 5, 22)
    var b3 = _upload(ctx, initial3)
    var r3 = _upload(ctx, initial3)
    var initial4 = pattern(offsets[5] - offsets[4] + 5, 23)
    var b4 = _upload(ctx, initial4)
    var r4 = _upload(ctx, initial4)
    var initial5 = pattern(offsets[6] - offsets[5] + 5, 24)
    var b5 = _upload(ctx, initial5)
    var r5 = _upload(ctx, initial5)
    var initial6 = pattern(offsets[7] - offsets[6] + 5, 25)
    var b6 = _upload(ctx, initial6)
    var r6 = _upload(ctx, initial6)
    var initial7 = pattern(offsets[8] - offsets[7] + 5, 26)
    var b7 = _upload(ctx, initial7)
    var r7 = _upload(ctx, initial7)
    var initial8 = pattern(offsets[9] - offsets[8] + 5, 27)
    var b8 = _upload(ctx, initial8)
    var r8 = _upload(ctx, initial8)
    # Every refusal occurs before a launch. Both directions share the guard,
    # but call both specializations so an unwired branch cannot pass by proxy.
    for fault in range(5):
        var invalid = offsets.copy()
        var expected = String("byte block copy: offset out of range")
        if fault == 0:
            invalid = List[Int]()
            expected = "byte block copy: expected ten offsets"
        elif fault == 1:
            invalid[0] = -1
        elif fault == 2:
            invalid[9] = len(flat) + 1
        elif fault == 3:
            invalid[1] = invalid[0] - 1
            expected = "byte block copy: invalid tensor range"
        else:
            invalid[1] = invalid[0] + len(b0) + 1
            expected = "byte block copy: invalid tensor range"
        for direction in range(2):
            var message = String("")
            try:
                if direction == 0:
                    byte_block_copy[False](ctx, flat, b0, b1, b2, b3, b4, b5, b6, b7, b8, invalid)
                else:
                    byte_block_copy[True](ctx, flat, b0, b1, b2, b3, b4, b5, b6, b7, b8, invalid)
            except e:
                message = String(e)
            if message.find(expected) < 0:
                raise Error(label + ": wrong bounds refusal: " + message)
    byte_block_copy[False](ctx, flat, b0, b1, b2, b3, b4, b5, b6, b7, b8, offsets)
    _copy_into(ctx, r0, flat, 0, offsets[0], offsets[1] - offsets[0])
    _copy_into(ctx, r1, flat, 0, offsets[1], offsets[2] - offsets[1])
    _copy_into(ctx, r2, flat, 0, offsets[2], offsets[3] - offsets[2])
    _copy_into(ctx, r3, flat, 0, offsets[3], offsets[4] - offsets[3])
    _copy_into(ctx, r4, flat, 0, offsets[4], offsets[5] - offsets[4])
    _copy_into(ctx, r5, flat, 0, offsets[5], offsets[6] - offsets[5])
    _copy_into(ctx, r6, flat, 0, offsets[6], offsets[7] - offsets[6])
    _copy_into(ctx, r7, flat, 0, offsets[7], offsets[8] - offsets[7])
    _copy_into(ctx, r8, flat, 0, offsets[8], offsets[9] - offsets[8])
    ctx.synchronize()
    same(download_f32(ctx, b0, len(b0)), download_f32(ctx, r0, len(r0)), label + " unpack 0 including tail")
    same(download_f32(ctx, b1, len(b1)), download_f32(ctx, r1, len(r1)), label + " unpack 1 including tail")
    same(download_f32(ctx, b2, len(b2)), download_f32(ctx, r2, len(r2)), label + " unpack 2 including tail")
    same(download_f32(ctx, b3, len(b3)), download_f32(ctx, r3, len(r3)), label + " unpack 3 including tail")
    same(download_f32(ctx, b4, len(b4)), download_f32(ctx, r4, len(r4)), label + " unpack 4 including tail")
    same(download_f32(ctx, b5, len(b5)), download_f32(ctx, r5, len(r5)), label + " unpack 5 including tail")
    same(download_f32(ctx, b6, len(b6)), download_f32(ctx, r6, len(r6)), label + " unpack 6 including tail")
    same(download_f32(ctx, b7, len(b7)), download_f32(ctx, r7, len(r7)), label + " unpack 7 including tail")
    same(download_f32(ctx, b8, len(b8)), download_f32(ctx, r8, len(r8)), label + " unpack 8 including tail")
    same(download_f32(ctx, flat, len(flat)), initial, label + " read-only flat")
    # Independent pack data: a roundtrip alone could hide paired offset bugs.
    var g0 = _upload(ctx, pattern(len(b0), 67))
    var g1 = _upload(ctx, pattern(len(b1), 68))
    var g2 = _upload(ctx, pattern(len(b2), 69))
    var g3 = _upload(ctx, pattern(len(b3), 70))
    var g4 = _upload(ctx, pattern(len(b4), 71))
    var g5 = _upload(ctx, pattern(len(b5), 72))
    var g6 = _upload(ctx, pattern(len(b6), 73))
    var g7 = _upload(ctx, pattern(len(b7), 74))
    var g8 = _upload(ctx, pattern(len(b8), 75))
    var trial_offsets = offsets.copy()
    if getenv("MOJOLEARN_BLOCK_COPY_SABOTAGE") == "1":
        # Shift all flat destinations one slot, staying in bounds and retaining
        # all range lengths. The oracle below must reject the wrong placement.
        for i in range(10):
            trial_offsets[i] += 1
    byte_block_copy[True](ctx, fused_flat, g0, g1, g2, g3, g4, g5, g6, g7, g8, trial_offsets)
    _copy_into(ctx, reference_flat, g0, offsets[0], 0, offsets[1] - offsets[0])
    _copy_into(ctx, reference_flat, g1, offsets[1], 0, offsets[2] - offsets[1])
    _copy_into(ctx, reference_flat, g2, offsets[2], 0, offsets[3] - offsets[2])
    _copy_into(ctx, reference_flat, g3, offsets[3], 0, offsets[4] - offsets[3])
    _copy_into(ctx, reference_flat, g4, offsets[4], 0, offsets[5] - offsets[4])
    _copy_into(ctx, reference_flat, g5, offsets[5], 0, offsets[6] - offsets[5])
    _copy_into(ctx, reference_flat, g6, offsets[6], 0, offsets[7] - offsets[6])
    _copy_into(ctx, reference_flat, g7, offsets[7], 0, offsets[8] - offsets[7])
    _copy_into(ctx, reference_flat, g8, offsets[8], 0, offsets[9] - offsets[8])
    ctx.synchronize()
    same(download_f32(ctx, g0, len(g0)), pattern(len(g0), 67), label + " pack source 0")
    same(download_f32(ctx, g1, len(g1)), pattern(len(g1), 68), label + " pack source 1")
    same(download_f32(ctx, g2, len(g2)), pattern(len(g2), 69), label + " pack source 2")
    same(download_f32(ctx, g3, len(g3)), pattern(len(g3), 70), label + " pack source 3")
    same(download_f32(ctx, g4, len(g4)), pattern(len(g4), 71), label + " pack source 4")
    same(download_f32(ctx, g5, len(g5)), pattern(len(g5), 72), label + " pack source 5")
    same(download_f32(ctx, g6, len(g6)), pattern(len(g6), 73), label + " pack source 6")
    same(download_f32(ctx, g7, len(g7)), pattern(len(g7), 74), label + " pack source 7")
    same(download_f32(ctx, g8, len(g8)), pattern(len(g8), 75), label + " pack source 8")
    same(download_f32(ctx, fused_flat, len(fused_flat)),
         download_f32(ctx, reference_flat, len(reference_flat)),
         label + " pack including prefix and tail")
    print("PASS", label, "unpack+pack bits and canaries")


def main() raises:
    var ctx = DeviceContext()
    # Uneven counts hit empty, single element and partial/full/multiple blocks.
    var irregular: List[Int] = [3, 3, 4, 11, 266, 522, 779, 781, 798, 1311]
    check(ctx, irregular, "uneven")
    var empty: List[Int] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    check(ctx, empty, "all empty")
    var shapes: List[ByteConfig] = [ByteConfig(), ByteConfig(1, 7, 24, 3, 1, 8, 40, 3, 257)]
    for shape in shapes:
        var full = shape.offsets()
        for layer in range(shape.n_layers):
            var offsets = List[Int]()
            for j in range(10):
                offsets.append(full[1 + 9 * layer + j])
            check(ctx, offsets, shape.profile() + " layer " + String(layer))
    print("PASS byte block copy; production dispatch unchanged")
