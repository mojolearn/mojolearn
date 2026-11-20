# SPDX-License-Identifier: Apache-2.0
"""Root-only ordered numerical GPU upgrade batch.

One graph at a time on the physical GPU; all existing kernels remain parallel.
This deliberately does not qualify shared DeviceContext concurrency (#7544).
OneClass resident-off builds report the reduced actual case total, not 127.
"""
from x_neighbors.ocsvm_dev import OCSVM_IDN_RES
from checks.native_tests.gpr_score_gpu_suite import run_checks as suite_0
from checks.native_tests.gpc_score_gpu_suite import run_checks as suite_1
from checks.native_tests.linear_svc_native_gpu_suite import run_checks as suite_2
from checks.native_tests.labels_threshold_gpu_suite import run_checks as suite_3
from checks.native_tests.hdbscan_native_gpu_suite import run_checks as suite_4
from checks.native_tests.linear_svr_native_gpu_suite import run_checks as suite_5
from checks.native_tests.svc_accuracy_native_gpu_suite import run_checks as suite_6
from checks.native_tests.svr_native_gpu_suite import run_checks as suite_7
from checks.native_tests.ipca_native_gpu_suite import run_checks as suite_8
from checks.native_tests.optics_native_gpu_suite import run_checks as suite_9
from checks.native_tests.ocsvm_native_gpu_suite import run_checks as suite_10

comptime CONSOLIDATED_GPU_CASES = 117 + (10 if OCSVM_IDN_RES else 0)


def main() raises:
    var failed = 0
    try:
        suite_0()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 0, String(problem))
    try:
        suite_1()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 1, String(problem))
    try:
        suite_2()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 2, String(problem))
    try:
        suite_3()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 3, String(problem))
    try:
        suite_4()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 4, String(problem))
    try:
        suite_5()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 5, String(problem))
    try:
        suite_6()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 6, String(problem))
    try:
        suite_7()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 7, String(problem))
    try:
        suite_8()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 8, String(problem))
    try:
        suite_9()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 9, String(problem))
    try:
        suite_10()
    except problem:
        failed += 1
        print("FAIL consolidated_native_gpu group", 10, String(problem))
    if failed:
        raise Error("Consolidated native GPU groups failed: " + String(failed))
    print("PASS consolidated_native_gpu:", CONSOLIDATED_GPU_CASES, "fixtures")
