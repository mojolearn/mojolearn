# SPDX-License-Identifier: Apache-2.0
"""Measure production forest fits using the retained benchmark data generator.
One excluded fit warms the same DeviceContext before one scored full fit.
"""
from max.gpu.host import DeviceContext
from gemm.checks.gemm_step_arms import gemm_step_env_int
from ensemble.bench.rf_bench import run_arm


def main() raises:
    var ctx = DeviceContext()
    var rows = gemm_step_env_int("AB_ROWS", 100000)
    var features = gemm_step_env_int("AB_FEATURES", 32)
    run_arm(ctx, "A07-warmup", rows, features, 128, False, False, False, 1)
    run_arm(ctx, "A07-score", rows, features, 128, True, True, True, 1)
