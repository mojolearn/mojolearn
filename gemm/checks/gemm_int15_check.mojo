# SPDX-License-Identifier: Apache-2.0
# Standalone wrapper of `gemm/checks/gemm_int15_check_suite.mojo`: the checks, fixtures and helpers live there
# (a module with main() cannot be imported; split by lane fx-checks-orphans, 2026-11-18).
# Import from the suite, never from this file.
from max.gpu.host import DeviceContext
from std.sys import has_accelerator
from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_int15_apple import int15_apple_sabotage_name
from gemm.checks.gemm_int15 import int15_plan_dispatch_name, int15_sabotage_name
from gemm.contract import GEMM_ORACLE_HOST_SABOTAGE
from gemm.checks.gemm_int15_check_suite import (
    HAS_UNIT,
    check_int15_pieces_cover_every_code,
    check_int15_sum_to_float_is_correctly_rounded,
    check_int15_quantizer_is_the_rule,
    check_int15_bounds_are_where_the_contract_says,
    check_int15_pieces_oracle_matches_oracle,
    check_int15_device_integers_match_host,
    check_int15_device_conversions_match_host,
    check_int15_device_matches_oracle,
    check_int15_row_scales,
    check_int15_plans_agree,
    check_int15_planted_worst_cases,
    check_int15_large_product_is_written_whole,
    check_int15_is_batch_invariant,
    check_int15_unit_loads_state_their_alignment,
    check_int15_device_refuses_above_max_k,
    _gate,
)


def main() raises:
    print(
        "== gemm/checks/gemm_int15_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int15_sabotage_name()
        + "  apple unit sabotage: " + int15_apple_sabotage_name()
        + "  host sabotage: " + String(GEMM_ORACLE_HOST_SABOTAGE) + " =="
    )
    print("   profile: mojolearn.identical.gemm.int15i64.v1")
    print("   contract: gemm/IDENTICAL_LOWBIT_CONTRACT.md (THE FIFTEEN-BIT PROFILE)")
    print("   column: " + column_name(TARGET_COLUMN) + "  int15 dispatch: " + int15_plan_dispatch_name())
    var ran = 0
    var failed = 0
    try:
        check_int15_pieces_cover_every_code()
        _gate(String("check_int15_pieces_cover_every_code"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_pieces_cover_every_code"), ran, failed, String(e))
    try:
        check_int15_sum_to_float_is_correctly_rounded()
        _gate(String("check_int15_sum_to_float_is_correctly_rounded"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_sum_to_float_is_correctly_rounded"), ran, failed, String(e))
    try:
        check_int15_quantizer_is_the_rule()
        _gate(String("check_int15_quantizer_is_the_rule"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_quantizer_is_the_rule"), ran, failed, String(e))
    try:
        check_int15_bounds_are_where_the_contract_says()
        _gate(String("check_int15_bounds_are_where_the_contract_says"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_bounds_are_where_the_contract_says"), ran, failed, String(e))
    try:
        check_int15_pieces_oracle_matches_oracle()
        _gate(String("check_int15_pieces_oracle_matches_oracle"), ran, failed, String(""))
    except e:
        _gate(String("check_int15_pieces_oracle_matches_oracle"), ran, failed, String(e))
    comptime if not has_accelerator():
        print("   no accelerator: the device gates did not run")
    else:
        var ctx = DeviceContext()
        try:
            check_int15_device_integers_match_host(ctx)
            _gate(String("check_int15_device_integers_match_host"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_integers_match_host"), ran, failed, String(e))
        try:
            check_int15_device_conversions_match_host(ctx)
            _gate(String("check_int15_device_conversions_match_host"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_conversions_match_host"), ran, failed, String(e))
        try:
            check_int15_device_matches_oracle(ctx)
            _gate(String("check_int15_device_matches_oracle"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_matches_oracle"), ran, failed, String(e))
        try:
            check_int15_plans_agree(ctx)
            _gate(String("check_int15_plans_agree"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_plans_agree"), ran, failed, String(e))
        try:
            check_int15_row_scales(ctx)
            _gate(String("check_int15_row_scales"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_row_scales"), ran, failed, String(e))
        try:
            check_int15_planted_worst_cases(ctx)
            _gate(String("check_int15_planted_worst_cases"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_planted_worst_cases"), ran, failed, String(e))
        try:
            check_int15_large_product_is_written_whole(ctx)
            _gate(String("check_int15_large_product_is_written_whole"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_large_product_is_written_whole"), ran, failed, String(e))
        try:
            check_int15_is_batch_invariant(ctx)
            _gate(String("check_int15_is_batch_invariant"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_is_batch_invariant"), ran, failed, String(e))
        comptime if HAS_UNIT:
            try:
                check_int15_unit_loads_state_their_alignment(ctx)
                _gate(String("check_int15_unit_loads_state_their_alignment"), ran, failed, String(""))
            except e:
                _gate(String("check_int15_unit_loads_state_their_alignment"), ran, failed, String(e))
        try:
            check_int15_device_refuses_above_max_k(ctx)
            _gate(String("check_int15_device_refuses_above_max_k"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_device_refuses_above_max_k"), ran, failed, String(e))
        _ = ctx^
    print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
    if failed > 0:
        raise Error(String(failed) + " gate(s) failed")
