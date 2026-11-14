# SPDX-License-Identifier: Apache-2.0
"""The common GradientBoosting fit cases on the device column.

One unit at a time on the physical GPU (each device fit owns the device for
its whole graph; its kernels stay parallel). The host-only units are the CPU
suite's. Predictions read the device-fitted model text with the host reader
(`gbdt_public_fits_cases.predict`); the device-resident predict has its own
check (`gbdt_resident_ns_predict_gpu`).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.gbdt_entry import GbdtImage
from gbdt.mojolearn_gbdt_run import gbdt_fit_native_run
from checks.native_tests.gbdt_public_fits_cases import model_dim as text_model_dim
from checks.native_tests.gbdt_public_fits_cases import GBDT_COMMON_UNITS, common_unit, require


struct _DeviceFit(GbdtImage):
    """The GPU image's column-major fit (`bindings/_mojolearn_gbdt.mojo` `_GbdtGpuImage`)."""
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        require(not row_major, "the check fits column-major")
        return gbdt_fit_native_run(False, args)

    @staticmethod
    def row_major() -> Bool:
        return False

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return text_model_dim(text)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(GBDT_COMMON_UNITS):  # one device graph at a time (see the docstring)
        try:
            common_unit[_DeviceFit](unit)
        except error:
            failed += 1
            print("FAIL gbdt_public_fits_gpu unit", unit, String(error))
    require(failed == 0, "gbdt_public_fits_gpu failed " + String(failed) + " units")
    print("PASS gbdt_public_fits_gpu:", GBDT_COMMON_UNITS, "original GradientBoosting fit units")
