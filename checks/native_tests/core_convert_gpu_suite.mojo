# SPDX-License-Identifier: Apache-2.0
"""The device narrowing and transposes the GPU base binding takes
(core/hotpath_device_run.mojo `hpdev_try_cast_f64_to_f32`,
`hpdev_try_transpose_to_f32`), every fixture in order.

Both entries use the one process-wide device context of the base binding,
so fixtures are submitted in order; each operation is one thread per element.
A False return (a sabotage build, or a size outside the device twin) fails the
fixture: the default GPU build takes the device route for every size here.
"""
from core.hotpath_device_run import hpdev_try_cast_f64_to_f32, hpdev_try_transpose_to_f32
from checks.native_tests.core_convert_cases import CONVERT_FIXTURES, ConvertCalls, require, run_convert_fixture


struct DeviceConvert(ConvertCalls):
    @staticmethod
    def cast(src: Int, dst: Int, n: Int) raises:
        require(hpdev_try_cast_f64_to_f32(src, dst, n), "device cast_f64_to_f32 not taken for n=" + String(n))

    @staticmethod
    def cast_colmajor(src: Int, dst: Int, rows: Int, cols: Int) raises:
        require(hpdev_try_transpose_to_f32(src, dst, rows, cols, True), "device cast_colmajor not taken")

    @staticmethod
    def transpose(src: Int, dst: Int, rows: Int, cols: Int) raises:
        require(hpdev_try_transpose_to_f32(src, dst, rows, cols, False), "device transpose_f32 not taken")


def run_checks() raises:
    var failed = 0
    for fixture in range(CONVERT_FIXTURES):
        try:
            run_convert_fixture[DeviceConvert](fixture)
        except error:
            failed += 1
            print("FAIL core_convert_gpu fixture", fixture, String(error))
    require(failed == 0, "core_convert_gpu failed " + String(failed) + " of " + String(CONVERT_FIXTURES))
    print("PASS core_convert_gpu:", CONVERT_FIXTURES, "fixtures, ordered on the base binding's device context")
