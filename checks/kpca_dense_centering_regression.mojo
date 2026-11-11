# SPDX-License-Identifier: Apache-2.0
"""Regression for the product's shared dense KernelPCA global-mean helper.

One dependent native chain, with the same independent 2x2 oracle and guarded
storage as the frozen failing reproducer. Existing Xn and x_decomp operations
retain their normal parallel dispatch; no timing or identity is performed.
"""
from core.neighbors_native_driver import XnDriver
from x_neighbors.native_descriptor_host import xn_native_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from core.kpca_native_plan import KPCANeighborDriver, kpca_fit_global_mean


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def main() raises:
    var driver = XnDriver(xn_native_v1)
    var matrix: List[Float32] = [1.0, 2.0, 2.0, 4.0]
    var sums: List[Float32] = [-9999.0, 0.0, 0.0, -9999.0]
    var columns: List[Float32] = [-9999.0, 0.0, 0.0, -9999.0]
    var total: List[Float32] = [-9999.0, 0.0, -9999.0]
    var centered: List[Float32] = [-9999.0, 0.0, 0.0, 0.0, 0.0, -9999.0]
    var empty: Array[Float64, 0] = []
    var divisor: Array[Float64, 1] = [2.0]
    var shape: Array[Int, 2] = [2, 2]
    var count: Array[Int, 1] = [2]
    var a: Array[Int, 2] = [Int(matrix.unsafe_ptr()), Int(sums.unsafe_ptr() + 1)]
    _ = driver.run("colsum", a, shape, empty)
    a = [Int(sums.unsafe_ptr() + 1), Int(columns.unsafe_ptr() + 1)]
    _ = driver.run("scale_div", a, count, divisor)
    var native = KPCANeighborDriver(driver.copy())
    kpca_fit_global_mean(native, x_decomp_host_v1, Int(columns.unsafe_ptr() + 1),
        Int(total.unsafe_ptr() + 1), 2)
    var center_addresses: Array[Int, 5] = [Int(matrix.unsafe_ptr()), Int(columns.unsafe_ptr() + 1),
        Int(columns.unsafe_ptr() + 1), Int(total.unsafe_ptr() + 1), Int(centered.unsafe_ptr() + 1)]
    _ = driver.run("kpca_center", center_addresses, shape, empty)
    require(sums[0] == -9999.0 and sums[3] == -9999.0, "first colsum canary overwritten")
    require(columns[0] == -9999.0 and columns[3] == -9999.0, "columns canary overwritten")
    require(total[0] == -9999.0 and total[2] == -9999.0, "total canary overwritten")
    require(centered[0] == -9999.0 and centered[5] == -9999.0, "centered matrix canary overwritten")
    print("KPCA_REGRESSION canaries=pass source=2x2 columns=", columns[1], columns[2])
    print("KPCA_REGRESSION total_extent=1 observed=", total[1], " expected_scalar=2.25")
    print("KPCA_REGRESSION centered=", centered[1], centered[2], centered[3], centered[4],
          " expected=0.25,-0.25,-0.25,0.25")
    _ = matrix
    require(total[1] == Float32(2.25), "dense KernelPCA total is not the independent global kernel mean")
    require(centered[1] == Float32(0.25) and centered[2] == Float32(-0.25)
        and centered[3] == Float32(-0.25) and centered[4] == Float32(0.25), "dense kernel centering differs")
    print("KPCA_REGRESSION independent_mean_and_centering=pass")
