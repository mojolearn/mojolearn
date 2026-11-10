# SPDX-License-Identifier: Apache-2.0
"""Native shape/admission decisions for existing training device tensors."""
from bindings.tensor_view import TensorView, base_format


def optimizer_device_transfer_span(address: Int, count: Int) raises:
    if count < 0 or count > 9223372036854775807 // 4:
        raise Error("ValueError: training device transfer count is out of range")
    if count > 0 and (address <= 0 or address > 9223372036854775807 - (count * 4 - 1)):
        raise Error("ValueError: training device transfer requires complete native storage")


def optimizer_device_extent(value: Int) raises -> Int:
    if value < 0:
        raise Error("ValueError: training device shape dimensions must be nonnegative")
    return value


def optimizer_device_count(shape: List[Int]) raises -> Int:
    var size = 1
    for axis in range(len(shape)):  # driver-loop: native tensor shape metadata
        var extent = optimizer_device_extent(shape[axis])
        if extent != 0 and size > 9223372036854775807 // extent:
            raise Error("OverflowError: training device shape exceeds native indexing")
        size *= extent
    return size


def optimizer_device_reshape(shape: List[Int], count: Int) raises:
    if optimizer_device_count(shape) != count:
        raise Error("ValueError: training device reshape does not hold the array's floats")


def optimizer_device_upload(view: TensorView) raises -> Int:
    if base_format(view.fmt) != "f" or view.itemsize != 4:
        raise Error("TypeError: mojolearn.train_to_device: a must be float32")
    var count = optimizer_device_count(view.shape)
    if count < 1:
        raise Error("ValueError: mojolearn.train_to_device: a must be nonempty")
    if count > 9223372036854775807 // 4 or view.nbytes != count * 4 or view.addr == 0:
        raise Error("ValueError: training device upload requires complete float32 storage")
    return count


def optimizer_device_output(shape: List[Int], count: Int) raises:
    # A rebound public size may request a prefix, as before, but cannot make
    # the unchanged native copy write beyond its newly allocated result.
    if count > optimizer_device_count(shape):
        raise Error("ValueError: training device download exceeds its output shape")
