# SPDX-License-Identifier: Apache-2.0
"""Focused mode/shape gate for the Apple production-batch GEMM dispatch."""
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.testing import assert_true
from gemm.checks.gemm_identical import (
    PLAN_TUNED_64_4X4,
    PLAN_TUNED_128_8X8,
    choose_gemm_plan,
)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN == COLUMN_APPLE:
        assert_true(choose_gemm_plan(1024, 768, 768) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(2048, 768, 768) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(2048, 2048, 768) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(2048, 768, 3072) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(32768, 1024, 768) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(32768, 2304, 768) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(32768, 3072, 768) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(768, 768, 2048) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(768, 2048, 2048) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(768, 3072, 2048) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(767, 3072, 2048) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 3072, 1023) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(1023, 768, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(2048, 767, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(32768, 1024, 769) == PLAN_TUNED_128_8X8)
    else:
        assert_true(choose_gemm_plan(32768, 1024, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(32768, 2304, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(32768, 3072, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 768, 2048) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 2048, 2048) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 3072, 2048) == PLAN_TUNED_128_8X8)
    print("production GEMM dispatch gate: PASS")
