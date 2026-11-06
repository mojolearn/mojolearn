# SPDX-License-Identifier: Apache-2.0
# Classifier target check read straight from a float label buffer.
#
# The public classifier target admission refuses a float label that is NaN or
# infinite ("Input y contains NaN or infinity") or finite but not integral
# ("Unknown label type: continuous"), at the first such label in C order.
# `bindings/label_integer_storage.mojo` hands an exact float32/float64 ndarray
# or Array here by address, shape and byte strides, so the check is one native
# pass over the buffer instead of one Python object and three scalar checks per
# label (`bindings/prep_lda_admission.mojo`, `lda_native_labels`).
from std.math import isfinite, floor


comptime LABELS_FLOAT_OK = 0
comptime LABELS_FLOAT_NONFINITE = 1
comptime LABELS_FLOAT_CONTINUOUS = 2


def label_float_target_refusal(address: Int, wide: Bool, shape: List[Int], strides: List[Int]) -> Int:
    """LABELS_FLOAT_OK when every element is finite and integral; otherwise
    the refusal of the first refused element in C order (the order `tolist`
    and the flattened label walk visit). `wide` reads float64, else float32;
    `strides` are in bytes and may be negative. Rank >= 1."""
    var rank = len(shape)
    var total = 1
    for k in range(rank):
        total *= shape[k]
    if total == 0 or rank == 0:
        return LABELS_FLOAT_OK
    var index = List[Int](length=rank, fill=0)
    var offset = 0
    for _ in range(total):
        var number: Float64
        if wide:
            number = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address + offset)[0]
        else:
            number = Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address + offset)[0])
        if not isfinite(number):
            return LABELS_FLOAT_NONFINITE
        if floor(number) != number:
            return LABELS_FLOAT_CONTINUOUS
        var axis = rank - 1
        while axis >= 0:
            index[axis] += 1
            offset += strides[axis]
            if index[axis] < shape[axis]:
                break
            offset -= strides[axis] * shape[axis]
            index[axis] = 0
            axis -= 1
    return LABELS_FLOAT_OK
