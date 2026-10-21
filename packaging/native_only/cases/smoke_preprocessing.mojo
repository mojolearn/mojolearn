# SPDX-License-Identifier: Apache-2.0
# Native estimator fit, transform and inverse; independent closed-form oracle.
from std.math import sqrt
from preprocessing.estimator import minmax_fit_refusing, minmax_transform_refusing, standard_fit_refusing, standard_transform_refusing

comptime FP = MutPointer[Float32, MutUntrackedOrigin]


def ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def close(got: Float32, want: Float32, name: String) raises:
    if not (abs(got - want) <= Float32(2e-6)):
        raise Error("native-only scaler: " + name + " differs from closed form")


def main() raises:
    # One varying column and one constant column exercise both scale rules.
    var x: List[Float32] = [1, -3, 3, -3, 5, -3, 7, -3]
    var minmax = List[Float32](length=10, fill=0)
    var standard = List[Float32](length=6, fill=0)
    var output = List[Float32](length=8, fill=0)
    var inverse = List[Float32](length=8, fill=0)
    minmax_fit_refusing(ptr(x), 4, 2, -1, 1, ptr(minmax))
    var wanted: List[Float32] = [1, -3, 7, -3, 6, 0, Float32(1)/3, 2, -Float32(4)/3, 5]
    for i in range(10):
        close(minmax[i], wanted[i], "minmax fitted state")
    minmax_transform_refusing(ptr(x), ptr(minmax) + 6, ptr(minmax) + 8, ptr(output), 4, 2, 0, 0, -1, 1)
    for r in range(4):
        close(output[2*r], Float32(2*r)/3 - 1, "minmax transform")
        close(output[2*r+1], -1, "constant-column transform")
    minmax_transform_refusing(ptr(output), ptr(minmax) + 6, ptr(minmax) + 8, ptr(inverse), 4, 2, 1, 0, -1, 1)
    for i in range(8):
        close(inverse[i], x[i], "minmax inverse")
    standard_fit_refusing(ptr(x), 4, 2, 1, 1, ptr(standard))
    var expected: List[Float32] = [4, -3, 5, 0, sqrt(Float32(5)), 1]
    for i in range(6):
        close(standard[i], expected[i], "standard fitted state")
    standard_transform_refusing(ptr(x), ptr(standard), ptr(standard) + 4, ptr(output), 4, 2, 0, 1, 1)
    for r in range(4):
        close(output[2*r], Float32(2*r-3)/sqrt(Float32(5)), "standard transform")
        close(output[2*r+1], 0, "standard constant-column transform")
    standard_transform_refusing(ptr(output), ptr(standard), ptr(standard) + 4, ptr(inverse), 4, 2, 1, 1, 1)
    for i in range(8):
        close(inverse[i], x[i], "standard inverse")
    _ = x^
    _ = minmax^
    _ = standard^
    _ = output^
    _ = inverse^
    print("NATIVE_ONLY_PASS preprocessing")
