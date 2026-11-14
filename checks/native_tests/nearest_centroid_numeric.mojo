# SPDX-License-Identifier: Apache-2.0
"""Existing NC numerical graphs, ordered on the one physical GPU.

Only one GPU job runs at a time. Each graph uses the shared DeviceContext, so
submissions are ordered here as in the PCA/TSVD numerical suites; kernels
retain their parallel execution. The separate parallel run_checks and device
concurrency fixtures remain unchanged, with their Metal failures still open.
This wrapper obtains numerical/refusal evidence, never concurrency clearance.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.native_tests.nearest_centroid_descriptor_suite import fixture, empty_case
from x_neighbors.native_descriptor_device import xn_native_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was nearest_centroid_numeric_gpu_suite.mojo) ----
def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var cases = report.cases(11)
    for unit in range(11):
        try:
            if unit < 8:
                fixture(xn_native_v1, unit)
            else:
                empty_case(xn_native_v1, unit - 8)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("nearest_centroid_numeric", cpu=False, gpu=True, serial="", run=run_cases)
