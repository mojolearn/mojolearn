# SPDX-License-Identifier: Apache-2.0
"""Focused mode/shape gate for the production-batch GEMM dispatch.

lane/no-bench-tuning (2026-10-04): the Apple rules, the AMD short-contraction
rule and the NVIDIA fold-stack rule are range rules now, so this gate checks
NEIGHBORS of the board shapes (640, 767, 769, 896, 1024 around d_model 768;
k = 640..1024 around 768) and the band edges, not only the board rows."""
from checks.kernel_matrix import COLUMN_AMD, COLUMN_APPLE, COLUMN_NVIDIA, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from std.testing import assert_true
from gemm.checks.gemm_identical import (
    GEMM_IDENTICAL_MFMA,
    IDN_GEMM_AMD_BAND_MFMA,
    IDN_GEMM_MFMA_NO_LONE_GROUP,
    IDN_GEMM_MFMA_REUSE_WS,
    PLAN_SPLIT_32_2X2,
    PLAN_SPLIT_64_4X4,
    PLAN_TUNED_32_2X2,
    PLAN_TUNED_64_4X4,
    PLAN_TUNED_128_8X8,
    _mfma_group_leaves,
    choose_gemm_plan_tiles,
    gemm_mfma16_waves,
    gemm_nv_step_kpack_rows,
    gemm_tile_min_blocks,
    amd_short_contract_large_output,
    choose_gemm_plan,
    contract_partition,
    gemm_default_ksplit_leaves,
    gemm_kpack_fold_slots_for,
    identical_gemm_workspace_max_floats,
)


def _check_mfma_groups(m: Int, n: Int, k: Int) raises:
    """lane/nr-gemm: the matrix-core group size never resolves to ONE
    group (that is the all-leaves launch), and where the AMD band runs the
    matrix-core body grouped, the shipped workspace holds its nodes."""
    var p = contract_partition(k)[1]
    var g = _mfma_group_leaves(m, n, k)
    comptime if IDN_GEMM_MFMA_NO_LONE_GROUP:
        if g > 0:
            assert_true((p + g - 1) // g >= 2)
    comptime if (
        GEMM_IDENTICAL_MFMA
        and IDN_GEMM_AMD_BAND_MFMA
        and IDN_GEMM_MFMA_REUSE_WS
        and TARGET_COLUMN == COLUMN_AMD
    ):
        if (
            choose_gemm_plan(m, n, k) != PLAN_TUNED_128_8X8
            and amd_short_contract_large_output(m, n, k)
            and g > 0
        ):
            assert_true(
                identical_gemm_workspace_max_floats(m, n, k)
                >= m * n * ((p + g - 1) // g)
            )


def _check_small_bodies(m: Int, n: Int, k: Int) raises:
    """lane/nr-gemm: the small-tile matrix-core rule (AMD body, a pure host
    rule checked on every column) and the NVIDIA packed step-down rule."""
    var p = contract_partition(k)[1]
    var big = m * n * p + 1
    var plans: List[Int] = [
        PLAN_TUNED_64_4X4, PLAN_TUNED_32_2X2, PLAN_SPLIT_64_4X4, PLAN_SPLIT_32_2X2,
    ]
    for pi in range(len(plans)):
        var plan = plans[pi]
        var split = plan == PLAN_SPLIT_64_4X4 or plan == PLAN_SPLIT_32_2X2
        var r = gemm_mfma16_waves(m, n, k, plan, big)
        var nw = -r if r < 0 else r
        assert_true(r == 0 or nw == 1 or nw == 2 or nw == 4)
        assert_true(r == 0 or ((r < 0) == split))
        if n < 64:
            assert_true(r == 0)
        if split:
            # A split launch never runs without its leaf-major workspace.
            assert_true(gemm_mfma16_waves(m, n, k, plan, m * n * p - 1) == 0)
    # Plans the rule does not serve keep their plan.
    assert_true(gemm_mfma16_waves(m, n, k, PLAN_TUNED_128_8X8, big) == 0)
    var rows = gemm_nv_step_kpack_rows(m, n, k)
    assert_true(rows == 0 or rows == 64 or rows == 128)
    if rows != 0:
        assert_true(choose_gemm_plan_tiles(m, n, k) == PLAN_TUNED_128_8X8)
        var plan2 = choose_gemm_plan(m, n, k)
        assert_true(plan2 == PLAN_TUNED_64_4X4 or plan2 == PLAN_TUNED_32_2X2)
        # The chosen packed tile fills the device (128 or 64 rows x 64 cols
        # on NVIDIA's narrow packed geometry).
        assert_true(((m + rows - 1) // rows) * ((n + 63) // 64) >= gemm_tile_min_blocks())


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
        comptime if is_defined["MOJOLEARN_LEGACY_SHAPE_APPLE_PLAN64"]():
            # The removed d_model == 768 rule's boundaries.
            assert_true(choose_gemm_plan(767, 3072, 2048) == PLAN_TUNED_128_8X8)
            assert_true(choose_gemm_plan(2048, 767, 768) == PLAN_TUNED_128_8X8)
            assert_true(choose_gemm_plan(768, 3072, 1023) == PLAN_TUNED_128_8X8)
            assert_true(choose_gemm_plan(1023, 768, 768) == PLAN_TUNED_128_8X8)
            assert_true(choose_gemm_plan(32768, 1024, 769) == PLAN_TUNED_128_8X8)
        else:
            # merge 2026-10-05: the IDENTICAL lane's narrow-side band rule.
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
    # lane/nr-gemm: MFMA group sizes and the AMD band's workspace on board
    # neighbors (k = 256..1152 around 384/768/1024, short and long outputs).
    var ms: List[Int] = [384, 1024, 2048, 2049, 4096, 8192]
    var ns: List[Int] = [256, 384, 640, 768, 1024, 1025, 3072]
    var ks: List[Int] = [256, 384, 385, 512, 513, 640, 768, 896, 1024, 1025, 1152, 2048]
    for mi in range(len(ms)):
        for ni in range(len(ns)):
            for ki in range(len(ks)):
                _check_mfma_groups(ms[mi], ns[ni], ks[ki])
                _check_small_bodies(ms[mi], ns[ni], ks[ki])
    # Narrow outputs and the edges of the 64-column tile.
    _check_small_bodies(2048, 63, 384)
    _check_small_bodies(2048, 64, 384)
    _check_small_bodies(2048, 65, 384)
    _check_small_bodies(17, 4096, 384)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN == COLUMN_AMD:
        if gemm_tile_min_blocks() == 1024:
            # Hand counts at the shipped 1024-block floor (board neighbors):
            # 2048 x 384: 64-row tiles 192, 32-row 384, so the one-wave tile.
            assert_true(gemm_mfma16_waves(2048, 384, 384, PLAN_TUNED_32_2X2, 0) == 1)
            assert_true(gemm_mfma16_waves(2048, 385, 384, PLAN_TUNED_32_2X2, 0) == 1)
            # 4096 x 1024: 64 x 16 = 1024 blocks of 64 rows.
            assert_true(gemm_mfma16_waves(4096, 1024, 384, PLAN_TUNED_64_4X4, 0) == 4)
            # 4096 x 512: 64-row 512, 32-row 1024 blocks.
            assert_true(gemm_mfma16_waves(4096, 512, 384, PLAN_TUNED_64_4X4, 0) == 2)
            # Split: 384 x 384 x 2048 (dW), P = 16: 36 x 16 = 576 64-row
            # blocks, 72 x 16 = 1152 32-row blocks.
            assert_true(
                gemm_mfma16_waves(384, 384, 2048, PLAN_SPLIT_64_4X4, 384 * 384 * 16) == -2
            )
    print("production GEMM dispatch gate: PASS")
