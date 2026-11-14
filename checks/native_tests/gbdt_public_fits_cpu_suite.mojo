# SPDX-License-Identifier: Apache-2.0
"""The GradientBoosting fit cases on the host column, across host cores.

Every unit (`gbdt_public_fits_cases.mojo`: the common units, then the
host-only units of the `*_on_the_host_when_built` tests) is an independent
fixture with its own inputs and fits; they run across the host cores with a
private failure slot each. Each host fit keeps its own feature-parallel work.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.gbdt_entry import GbdtImage
from gbdt.mojolearn_gbdt_host_run import gbdt_fit_native_run
from gbdt.host.gbdt_oracle import GBDT_ORACLE_HOST_SABOTAGE
from checks.native_tests.gbdt_public_fits_cases import model_dim as text_model_dim
from checks.native_tests.gbdt_public_fits_cases import (
    GBDT_COMMON_UNITS, GBDT_HOST_UNITS, common_unit, host_unit, require,
)


struct _HostFit(GbdtImage):
    """The host image's fit (`bindings/_mojolearn_gbdt_host.mojo` `_GbdtHostImage`)."""
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        require(not row_major, "the host fit is column-major")
        return gbdt_fit_native_run(args)

    @staticmethod
    def row_major() -> Bool:
        return False

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return text_model_dim(text)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # `assert not module.gbdt_host_sabotage()`: a sabotage build is not this check's
    comptime assert not GBDT_ORACLE_HOST_SABOTAGE, "a sabotage build loaded outside the gate"
    comptime UNITS = GBDT_COMMON_UNITS + GBDT_HOST_UNITS
    var failures = List[Int](length=UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < GBDT_COMMON_UNITS:
                common_unit[_HostFit](unit)
            else:
                host_unit[_HostFit](unit - GBDT_COMMON_UNITS)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gbdt_public_fits_cpu unit", unit, String(error))
    host_parallelize(task, UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gbdt_public_fits_cpu failed " + String(failed) + " units")
    print("PASS gbdt_public_fits_cpu:", UNITS, "original GradientBoosting fit units")
