# SPDX-License-Identifier: Apache-2.0
"""Native KMeans CPU fixtures; independent owners run across host workers."""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.host_parallel import host_parallelize
from core.kmeans_native_api import KMeansNativeDriver
from core.kmeans_native_host import kmeans_host_native_v1
from checks.native_tests.kmeans_kernel_cases import fixture
from core.kmeans_native_device import kmeans_device_native_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The host column (was kmeans_native_cpu_suite.mojo) ----
comptime KMEANS_CPU_FIXTURES = 6


def _run_cpu(mut report: Report) raises:
    comptime assert TARGET_COLUMN == COLUMN_CPU
    var cases = report.cases(KMEANS_CPU_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            fixture(KMeansNativeDriver(kmeans_host_native_v1), unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, KMEANS_CPU_FIXTURES)


# ---- The device column (was kmeans_native_gpu_suite.mojo) ----
# Native GPU KMeans fixtures; one shared device context is exercised in order.
comptime KMEANS_GPU_FIXTURES = 9


def _run_gpu(mut report: Report) raises:
    comptime assert TARGET_COLUMN != COLUMN_CPU
    var cases = report.cases(KMEANS_GPU_FIXTURES)
    for unit in range(KMEANS_GPU_FIXTURES):
        try:
            fixture(KMeansNativeDriver(kmeans_device_native_v1), unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("kmeans_native", cpu=True, gpu=True, serial="", run=run_cases)
