# SPDX-License-Identifier: Apache-2.0
"""The nearest_centroid_native native check."""
from checks.native_tests.nearest_centroid_descriptor_suite import run_checks
from x_neighbors.native_descriptor_host import xn_native_v1
from x_neighbors.native_descriptor_device import xn_native_v1 as xn_native_v1_gpu
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report


# ---- The host column entry (was nearest_centroid_native_cpu.mojo) ----
def _run_cpu(mut report: Report) raises:
    run_checks(xn_native_v1)


# ---- The device column entry (was nearest_centroid_native_gpu.mojo) ----
def _run_gpu(mut report: Report) raises:
    run_checks(xn_native_v1_gpu)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("nearest_centroid_native", cpu=True, gpu=True, serial="", run=run_cases)
