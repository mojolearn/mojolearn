# SPDX-License-Identifier: Apache-2.0
"""Focused mode/shape gate for the production-batch GEMM dispatch.

lane/no-bench-tuning (2026-10-04): the Apple rules, the AMD short-contraction
rule and the NVIDIA fold-stack rule are range rules now, so this gate checks
NEIGHBORS of the board shapes (640, 767, 769, 896, 1024 around d_model 768;
k = 640..1024 around 768) and the band edges, not only the board rows."""
from checks.kernel_matrix import COLUMN_APPLE, COLUMN_NVIDIA, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.testing import assert_true
from gemm.checks.gemm_identical import (
    PLAN_TUNED_64_4X4,
    PLAN_TUNED_128_8X8,
    amd_short_contract_large_output,
    choose_gemm_plan,
    contract_partition,
    gemm_default_ksplit_leaves,
    gemm_kpack_fold_slots_for,
)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN == COLUMN_APPLE:
        # The measured board rows (d_model 768) still take the 64x64 plan.
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
        # Neighbors of 768 inside the 512..1024 band take it too.
        assert_true(choose_gemm_plan(2048, 767, 768) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(2048, 769, 769) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(2048, 640, 640) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(4096, 896, 2048) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(2048, 1024, 1024) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(2048, 512, 4096) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(32768, 1024, 769) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(767, 3072, 2048) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(640, 2048, 2048) == PLAN_TUNED_64_4X4)
        assert_true(choose_gemm_plan(896, 1024, 1024) == PLAN_TUNED_64_4X4)
        # Outside the band (or too few rows / too short k) keep the 128 tile.
        assert_true(choose_gemm_plan(511, 3072, 2048) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 3072, 1023) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(1023, 768, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(2048, 511, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(32768, 1100, 1100) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(2048, 1025, 1025) == PLAN_TUNED_128_8X8)
    else:
        assert_true(choose_gemm_plan(32768, 1024, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(32768, 2304, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(32768, 3072, 768) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 768, 2048) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 2048, 2048) == PLAN_TUNED_128_8X8)
        assert_true(choose_gemm_plan(768, 3072, 2048) == PLAN_TUNED_128_8X8)
    # AMD short-contraction band (a pure host rule, checked on every column):
    # 4 < P <= 8 leaves of 128, i.e. 512 < k <= 1024, with a large output.
    assert_true(amd_short_contract_large_output(4096, 768, 768))
    assert_true(amd_short_contract_large_output(2048, 1024, 768))
    assert_true(amd_short_contract_large_output(4096, 768, 513))
    assert_true(amd_short_contract_large_output(4096, 768, 640))
    assert_true(amd_short_contract_large_output(4096, 768, 896))
    assert_true(amd_short_contract_large_output(4096, 768, 1024))
    assert_true(not amd_short_contract_large_output(4096, 768, 512))
    assert_true(not amd_short_contract_large_output(4096, 768, 1025))
    assert_true(not amd_short_contract_large_output(4096, 768, 3072))
    assert_true(not amd_short_contract_large_output(2048, 768, 768))
    assert_true(not amd_short_contract_large_output(1024, 4096, 768))
    # The band's leaf bounds are what the rule's comment says they are.
    assert_true(contract_partition(640)[1] == 5)
    assert_true(contract_partition(1024)[1] == 8)
    # NVIDIA fold-stack class: the measured GPT-3-small dWeight shape keeps
    # FS4 under the generalized rule, and the class is a pure leaf bound.
    assert_true(gemm_kpack_fold_slots_for(6, 0) == 4)
    assert_true(gemm_kpack_fold_slots_for(8, 0) == 4)
    assert_true(gemm_kpack_fold_slots_for(9, 0) == 8)
    assert_true(gemm_kpack_fold_slots_for(24, 8) == 4)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN == COLUMN_NVIDIA:
        assert_true(
            gemm_kpack_fold_slots_for(
                contract_partition(2048)[1],
                gemm_default_ksplit_leaves(768, 768, 2048),
            ) == 4
        )
    print("production GEMM dispatch gate: PASS")
