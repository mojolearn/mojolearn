# SPDX-License-Identifier: Apache-2.0
"""One dependent native KernelPCA centering chain with independent expected values.

CPU-only correctness reproducer, not a timing or identity run. The chain is
serial because each operation consumes the preceding output; existing Xn ops
retain their normal host parallel dispatch. This deliberately fails on the
current dense-fit sequence. Arrays have explicit guard words around the full
output extents that the real boundary allocates.
"""
from core.neighbors_native_driver import XnDriver
from x_neighbors.native_descriptor_host import xn_native_v1


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def main() raises:
    var driver = XnDriver(xn_native_v1)
    var matrix: List[Float32] = [1.0, 2.0, 2.0, 4.0]
    var sums: List[Float32] = [-9999.0, 0.0, 0.0, -9999.0]
    var columns: List[Float32] = [-9999.0, 0.0, 0.0, -9999.0]
    var second: List[Float32] = [-9999.0, 0.0, 0.0, -9999.0]
    var total: List[Float32] = [-9999.0, 0.0, 0.0, -9999.0]
    var centered: List[Float32] = [-9999.0, 0.0, 0.0, 0.0, 0.0, -9999.0]
    var empty: InlineArray[Float64, 0] = []
    var divisor: InlineArray[Float64, 1] = [2.0]
    var shape: InlineArray[Int, 2] = [2, 2]
    var row_shape: InlineArray[Int, 2] = [1, 2]
    var count: InlineArray[Int, 1] = [2]
    var a: InlineArray[Int, 2] = [Int(matrix.unsafe_ptr()), Int(sums.unsafe_ptr() + 1)]
    _ = driver.run("colsum", a, shape, empty)
    a = [Int(sums.unsafe_ptr() + 1), Int(columns.unsafe_ptr() + 1)]
    _ = driver.run("scale_div", a, count, divisor)
    a = [Int(columns.unsafe_ptr() + 1), Int(second.unsafe_ptr() + 1)]
    _ = driver.run("colsum", a, row_shape, empty)
    a = [Int(second.unsafe_ptr() + 1), Int(total.unsafe_ptr() + 1)]
    _ = driver.run("scale_div", a, count, divisor)
    var center_addresses: InlineArray[Int, 5] = [Int(matrix.unsafe_ptr()), Int(columns.unsafe_ptr() + 1),
        Int(columns.unsafe_ptr() + 1), Int(total.unsafe_ptr() + 1), Int(centered.unsafe_ptr() + 1)]
    _ = driver.run("kpca_center", center_addresses, shape, empty)
    require(sums[0] == -9999.0 and sums[3] == -9999.0, "first colsum canary overwritten")
    require(columns[0] == -9999.0 and columns[3] == -9999.0, "columns canary overwritten")
    require(second[0] == -9999.0 and second[3] == -9999.0, "second colsum canary overwritten")
    require(total[0] == -9999.0 and total[3] == -9999.0, "total canary overwritten")
    require(centered[0] == -9999.0 and centered[5] == -9999.0, "centered matrix canary overwritten")
    print("KPCA_REPRO canaries=pass source=2x2 columns=", columns[1], columns[2])
    print("KPCA_REPRO total_extent=2 observed=", total[1], total[2], " expected_scalar=2.25")
    print("KPCA_REPRO centered=", centered[1], centered[2], centered[3], centered[4],
          " expected=0.25,-0.25,-0.25,0.25")
    _ = matrix
    require(total[1] == Float32(2.25), "dense KernelPCA total is not the independent global kernel mean")
    require(centered[1] == Float32(0.25) and centered[2] == Float32(-0.25)
        and centered[3] == Float32(-0.25) and centered[4] == Float32(0.25), "dense kernel centering differs")
