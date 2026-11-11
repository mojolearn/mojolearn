# SPDX-License-Identifier: Apache-2.0
"""Original fixtures on actual GPU linalg operations, with unchanged settings.

The production device contexts/pools are shared, so graph submissions are
ordered. Independent input decoding and numerical oracle cells use the host
pool; the numerical device operations retain their existing parallel kernels.
"""
from core.hostptr import f32_ptr
from decomposition.linalg_door_impl import qr_r_device_into, svdvals_device_into
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from checks.native_tests.linalg_numerical_cases import LINALG_NUMERICAL_FIXTURES, LinalgNumericalCalls, fixture, require


struct DeviceLinalgNumerical(LinalgNumericalCalls):
    @staticmethod
    def qr(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        qr_r_device_into(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def singular(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        svdvals_device_into(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def eigen(source: Int, values: Int, vectors: Int, n: Int) raises:
        DevExec.eigh(F32Ptr(unsafe_from_address=source), F32Ptr(unsafe_from_address=values),
                     F32Ptr(unsafe_from_address=vectors), n, 1)


def run_checks() raises:
    var failed = 0
    for index in range(LINALG_NUMERICAL_FIXTURES):
        try:
            fixture[DeviceLinalgNumerical](index)
        except error:
            failed += 1
            print("FAIL linalg_numerical_gpu fixture", index, String(error))
    require(failed == 0, "linalg_numerical_gpu failed " + String(failed) + " fixtures")
    print("PASS linalg_numerical_gpu:", LINALG_NUMERICAL_FIXTURES, "native numerical fixtures")
