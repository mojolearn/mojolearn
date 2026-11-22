# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""NumPy's contract for `linalg.qr`, `eigh`, `svdvals` and `svd` on the device column (lane py2mojo-D6, 2026-11-22;
ported from python/mojolearn/tests/test_linalg_decompositions.py, its eight numeric NumPy-contract tests as 18 units).
The cases, their inputs, references, tolerances and what stays in Python are in linalg_numpy_cases.mojo; this file
runs them on the device image (x_decomp_device_v1) and the device qr_r / svdvals entries, in order on the one shared
device context. A file of its own because the device image does not compile in the host column build.
"""
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from decomposition.linalg_door_impl import qr_r_device_into, svdvals_device_into
from x_decomp.descriptor_schema import XD_OPERATION_COUNT
from x_decomp.descriptor_device import x_decomp_device_v1
from std.testing import TestSuite
from checks.native_tests.testsuite_serial import check_cases_serial
from checks.native_tests.linalg_numpy_cases import NumpyLinalgCalls, numpy_unit

comptime NUMPY_CONTRACT_UNITS = 18


struct DeviceNumpyLinalg(NumpyLinalgCalls):
    @staticmethod
    def qr(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        qr_r_device_into(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def singular(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        svdvals_device_into(source, rows, cols, f32_ptr(dst))


def _unit(unit: Int) raises:
    numpy_unit[DeviceNumpyLinalg](EntryRef(x_decomp_device_v1, entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0,
                                                                              XD_OPERATION_COUNT)), unit)


def test_numpy_contract() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    check_cases_serial("numpy_contract", NUMPY_CONTRACT_UNITS, _unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
