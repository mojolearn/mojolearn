# SPDX-License-Identifier: Apache-2.0
"""Host-only byte gate for WP6; no GPU, numeric mode, or timing claim."""
from std.memory import bitcast
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, u32_ptr, copy_f32, read_f32, read_i32


def pattern(i: Int) -> UInt32:
    # Includes both zeros, infinities, subnormals and distinct NaN payloads.
    var special: List[UInt32] = [0, 0x80000000, 0x7F800000, 0xFF800000,
                                 1, 0x80000001, 0x7FC12345, 0x7F812345]
    if i < len(special):
        return special[i]
    return UInt32(i) * UInt32(2654435761)


def main() raises:
    var cells = 0
    var sizes: List[Int] = [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 2049]
    for n in sizes:
        for offset in range(8):
            var src = alloc[Float32](n + 16)
            var dst = alloc[Float32](n + 16)
            for i in range(n + 16):
                src.unsafe_store(i, bitcast[DType.float32](pattern(i)))
                dst.unsafe_store(i, bitcast[DType.float32](UInt32(0xDEADBEEF)))
            copy_f32(src + offset, dst + offset, n)
            for i in range(n + 16):
                var want = UInt32(0xDEADBEEF)
                if offset <= i < offset + n:
                    want = pattern(i)
                if bitcast[DType.uint32](dst.unsafe_load(i)) != want:
                    raise Error("SIMD copy changed bytes or a boundary sentinel")
                cells += 1
            # Explicit allocations remain live across integer-address borrows.
            var result = read_f32(Int(src + offset), n)
            for i in range(n):
                if bitcast[DType.uint32](result[i]) != pattern(i + offset):
                    raise Error("memcpy read changed bytes")
                cells += 1
            var labels = read_i32(Int(src + offset), n)
            for i in range(n):
                if bitcast[DType.uint32](labels[i]) != pattern(i + offset):
                    raise Error("int32 memcpy changed label bits")
                cells += 1
            copy_f32(dst + offset, dst + offset, n)
            for i in range(n):
                if bitcast[DType.uint32](dst.unsafe_load(i + offset)) != pattern(i + offset):
                    raise Error("same-span copy changed bytes")
            src.free()
            dst.free()
    var refusals = 0
    try:
        _ = f32_ptr(0)
    except:
        refusals += 1
    try:
        _ = f64_ptr(0)
    except:
        refusals += 1
    try:
        _ = i32_ptr(0)
    except:
        refusals += 1
    try:
        _ = u32_ptr(0)
    except:
        refusals += 1
    var owner = alloc[Float32](1)
    try:
        _ = read_f32(Int(owner), -1)
    except:
        refusals += 1
    owner.free()
    if refusals != 5:
        raise Error("pointer/length refusal missing")
    print("PASS hostptr cells", cells, "refusals", refusals)
