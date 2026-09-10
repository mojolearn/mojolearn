# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the DBSCAN checks."""

from dbscan.checks.dbscan_check import (
    check_dbscan,
    check_dbscan_batching_agrees,
    check_dbscan_duplicate_equals_weight_two,
    check_dbscan_eps_sensitivity,
    check_dbscan_manhattan_changes_the_labels,
    check_dbscan_manhattan_neighborhood,
    check_dbscan_manhattan_refused_on_the_ball_cover,
    check_dbscan_max_mbytes_moves_the_batch,
    check_dbscan_rbc_matches_brute,
    check_dbscan_rbc_two_loop_arms,
    check_dbscan_tiny_budget_agrees,
    check_dbscan_uniform_weight_matches_unweighted,
    check_dbscan_weighted_degree_matches_host_oracle,
    check_dbscan_weighted_fold_is_pinned,
    check_exclusive_scan_beyond_the_old_cap,
    check_fused_eps_agrees_with_materialized,
)


def main() raises:
    check_fused_eps_agrees_with_materialized()
    check_dbscan()
    check_dbscan_eps_sensitivity()
    check_exclusive_scan_beyond_the_old_cap()
    check_dbscan_batching_agrees()
    check_dbscan_rbc_matches_brute()
    check_dbscan_rbc_two_loop_arms()
    check_dbscan_max_mbytes_moves_the_batch()
    check_dbscan_tiny_budget_agrees()
    # The metric arm (DEVIATION 27). Cheapest first: the fold pin is a
    # comptime assertion plus a host fold, the neighborhood is one kernel,
    # and the label check is two fits.
    #
    # THE FOUR WEIGHTED GATES WERE BLOCKED ON THE TOOLCHAIN, NOT DROPPED.
    # Building any of them used to raise `DeadArgumentElimination surveyUse
    # failed`, a compiler assertion at -O2 and above, and take the whole
    # dbscan build down, which is why `sample_weight` shipped IMPLEMENTED
    # AND UNGATED until 2026-09-01, and why this file built at `-O1` from
    # then until 2026-09-09.
    #
    # THE TRIGGER WAS ONE LOOP, AND IT IS GONE (2026-09-09, NVIDIA L40S,
    # Mojo 1.0.0 ed45d567): `_host_weighted_degree_strided`'s `while k <
    # len(cols): ...; k += width`, a `len()` re-read in the condition of a
    # loop whose step is a runtime `Int` argument. The bound is now read
    # once before the loop. The reduced repro is
    # `dbscan/checks/compiler_repro_dead_arg_elim.mojo`; the four 2026-09-01
    # rewrites were never the cure and their sites say so.
    #
    # THE BISECT, kept because the handle is reusable. The IMPORT above is
    # what pulls a gate into codegen, so an import and its call must BOTH
    # be commented to disable one. One entry point per gate showed the fold
    # gate and the degree-oracle gate assert alone (both call the strided
    # host fold) while the uniform and duplicate gates build clean, which
    # is what the 2026-09-01 record's "two independent triggers" were.
    #
    # MEASURED: with the fix, this file at -O3 and at -O1 prints
    # byte-identical output (17 gates) in both the FAST and the IDENTICAL
    # build on the L40S. The Apple M4 run of the default -O3 task is owed.
    # A three-vendor leg for the weighted numbers is still owed.
    check_dbscan_weighted_fold_is_pinned()
    check_dbscan_manhattan_neighborhood()
    check_dbscan_manhattan_changes_the_labels()
    check_dbscan_manhattan_refused_on_the_ball_cover()
    # The weighted core-point test (DEVIATION 28) and its wiring
    # (DEVIATION 29).
    check_dbscan_weighted_degree_matches_host_oracle()
    check_dbscan_uniform_weight_matches_unweighted()
    check_dbscan_duplicate_equals_weight_two()
