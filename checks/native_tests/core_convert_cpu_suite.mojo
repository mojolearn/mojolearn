# SPDX-License-Identifier: Apache-2.0
"""The host narrowing and transposes, every fixture through both host routes,
run across host cores.

Route 0 is the host bindings' column (core/host_helpers_impl.mojo
`cast_f64_to_f32`, core/core_host_impl.mojo `transpose_to_f32`); route 1 is
the base binding's host loop (core/mojolearn_shell_impl.mojo
`cast_f64_to_f32_host`, `tiled_transpose_to_f32`).
"""
from core.core_host_impl import transpose_to_f32
from core.host_helpers_impl import cast_f64_to_f32
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, f64_ptr
from core.mojolearn_shell_impl import cast_f64_to_f32_host, tiled_transpose_to_f32
from checks.native_tests.core_convert_cases import CONVERT_FIXTURES, ConvertCalls, require, run_convert_fixture


struct HostHelpersConvert(ConvertCalls):
    @staticmethod
    def cast(src: Int, dst: Int, n: Int) raises:
        cast_f64_to_f32(f64_ptr(src), f32_ptr(dst), n)

    @staticmethod
    def cast_colmajor(src: Int, dst: Int, rows: Int, cols: Int) raises:
        transpose_to_f32[DType.float64](f64_ptr(src), f32_ptr(dst), rows, cols)

    @staticmethod
    def transpose(src: Int, dst: Int, rows: Int, cols: Int) raises:
        transpose_to_f32[DType.float32](f32_ptr(src), f32_ptr(dst), rows, cols)


struct HostShellConvert(ConvertCalls):
    @staticmethod
    def cast(src: Int, dst: Int, n: Int) raises:
        cast_f64_to_f32_host(f64_ptr(src), f32_ptr(dst), n)

    @staticmethod
    def cast_colmajor(src: Int, dst: Int, rows: Int, cols: Int) raises:
        tiled_transpose_to_f32[DType.float64](f64_ptr(src), f32_ptr(dst), rows, cols)

    @staticmethod
    def transpose(src: Int, dst: Int, rows: Int, cols: Int) raises:
        tiled_transpose_to_f32[DType.float32](f32_ptr(src), f32_ptr(dst), rows, cols)


def run_checks() raises:
    comptime UNITS = 2 * CONVERT_FIXTURES
    var failures = List[Int](length=UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        var fixture = unit % CONVERT_FIXTURES
        try:
            if unit < CONVERT_FIXTURES:
                run_convert_fixture[HostHelpersConvert](fixture)
            else:
                run_convert_fixture[HostShellConvert](fixture)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL core_convert_cpu route", unit // CONVERT_FIXTURES, "fixture", fixture, String(error))
    host_parallelize(task, UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_convert_cpu failed " + String(failed) + " of " + String(UNITS))
    print("PASS core_convert_cpu:", CONVERT_FIXTURES, "fixtures x 2 host routes, parallel")
