# SPDX-License-Identifier: Apache-2.0
from dbscan.checks.dbscan_eps_filter_suite import run_checks
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


# Lane sp-dbscan-filter (2026-11-20): the GEMM filter of the brute epsilon pass
# (`dbscan/impl/brute_eps_filter.mojo`) against the exact pass, every bit word and degree
# (`pixi run check-native-gpu dbscan_eps_filter_gpu`).
# Device column only: the suite launches kernels, so the CPU build never instantiates it.
def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("dbscan_eps_filter_gpu", cpu=False, gpu=True, serial="", run=run_cases)
