# SPDX-License-Identifier: Apache-2.0
# Embedding input storage/layout only. Lookup and duplicate-gradient folds
# remain in the existing embedding kernels.
from std.memory import bitcast
from core.hotpath_impl import _refused


def embedding_pack(addr: Int, shape: List[Int], strides: List[Int], item: Int,
                   swapped: Bool, code: Int, count: Int) raises -> List[UInt8]:
    var width = 4 if code == 9 else item
    if count > 9223372036854775807 // width:
        raise Error("OverflowError: embedding input byte count exceeds the native address range")
    var result = List[UInt8](length=count * width, fill=0)
    var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    for i in range(count):
        var remain = i
        var offset = 0
        for j in range(len(shape) - 1, -1, -1):
            offset += (remain % shape[j]) * strides[j]
            remain //= shape[j]
        var raw = UInt64(0)
        for k in range(item):
            raw |= UInt64(source[offset + (item - 1 - k if swapped else k)]) << UInt64(8 * k)
        if code == 9:
            # The old buffer door promoted signed16 to signed32 exactly.
            raw = UInt64(bitcast[DType.uint32](Int32(bitcast[DType.int16](UInt16(raw)))))
        elif code == 10 and raw > UInt64(9223372036854775807):
            # uint64 was first stored in an int64 Array, before narrowing.
            raise Error("OverflowError: int too big to convert")
        for k in range(width):
            result[i * width + k] = UInt8(raw >> UInt64(8 * k))
    return result^


def _embedding_refusal[source: DType](addr: Int, count: Int) raises:
    var words = MutPointer[Scalar[source], MutUntrackedOrigin](unsafe_from_address=addr)
    for i in range(count):
        var value = words[i]
        if _refused[source, DType.int32, 1](value):
            comptime if source.is_floating_point():
                if value != value:
                    raise Error("ValueError: cannot convert float NaN to integer")
            raise Error("OverflowError: embedding index does not fit int32")
    raise Error("RuntimeError: the native index cast refused without an invalid input")


def embedding_cast_refusal(addr: Int, code: Int, count: Int) raises:
    if code == 0:
        _embedding_refusal[DType.float32](addr, count)
    elif code == 1:
        _embedding_refusal[DType.float64](addr, count)
    elif code == 3:
        _embedding_refusal[DType.int64](addr, count)
    elif code == 4:
        _embedding_refusal[DType.uint32](addr, count)
    elif code == 8:
        _embedding_refusal[DType.float16](addr, count)
    else:
        raise Error("RuntimeError: an exactly representable native index cast refused")
