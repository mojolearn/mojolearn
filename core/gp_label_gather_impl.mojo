# SPDX-License-Identifier: Apache-2.0
# Float label buffer gather for the Gaussian process classifier's label input
# (bindings/gp_labels_native.mojo, `_gp_float_storage_labels`): a float32 or
# float64 label buffer of any rank and byte strides read once in C order (the
# order `tolist` and the flattened label walk visit) into a contiguous float64
# vector, in place of one Python float per label.


def label_float_gather_f64(address: Int, wide: Bool, shape: List[Int], strides: List[Int], out_address: Int):
    """Copy every element of the buffer at `address` (`wide` float64, else
    float32; `strides` in bytes, may be negative; rank >= 1) in C order into
    the float64 vector at `out_address`, widening float32 exactly."""
    var rank = len(shape)
    var total = 1
    for k in range(rank):
        total *= shape[k]
    if total == 0 or rank == 0:
        return
    var out = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=out_address)
    var index = List[Int](length=rank, fill=0)
    var offset = 0
    for i in range(total):
        if wide:
            out[i] = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address + offset)[0]
        else:
            out[i] = Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address + offset)[0])
        var axis = rank - 1
        while axis >= 0:
            index[axis] += 1
            offset += strides[axis]
            if index[axis] < shape[axis]:
                break
            offset -= strides[axis] * shape[axis]
            index[axis] = 0
            axis -= 1
