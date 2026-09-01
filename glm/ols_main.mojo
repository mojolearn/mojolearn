# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the OLS checks. Both halves of them.

The first three are the ACCURACY checks: a planted model recovered within
1%, the target scaled by 5, and the least-squares residual beating the
truth's. All three are TOLERANCE checks and would pass on a build whose
summation order differs on every vendor, which is exactly why the later
groups exist.

The next seven (2026-09-01) are the PROPERTY gates on the two shapes
`ols.cuh:112-113` used to refuse and on sample weights: interpolation, the
minimum-norm property, a float64 Gaussian-elimination oracle, the
single-column closed form, `A^T (A w - b) ~ 0`, the duplicate-row identity,
the operand restore, and the host-rescale equality the Python surface rests
on. Each runs its own NEGATIVE CONTROL in the same process and fails if the
control passes.

The rest are DEVIATION 527's identity properties, and each asserts something
DIFFERENT under `NUMERIC_FAST` and `NUMERIC_IDENTICAL`. Run both arms:

    tools/with_build_lock.sh     pixi run mojo run -I . glm/ols_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo

or `pixi run check-linalg-identity`, which runs this file and its
neighbours in both modes under the shared build lock. Every line printed
carries the mode the binary COMPILED IN, read from the comptime constant --
four sessions share this checkout and the mode line is one line in a file
all of them edit.

The certificate itself is `glm/ols_trace_main.mojo`.
"""

from glm.checks.ols_check import (
    check_ols_arms_are_pinned,
    check_ols_beats_truth_on_noise,
    check_ols_card_hashes_raw_bytes,
    check_ols_card_is_emitted,
    check_ols_dispatch_routes_special_shapes,
    check_ols_duplicating_a_row_equals_doubling_its_weight,
    check_ols_exact,
    check_ols_host_surface_takes_the_guard,
    check_ols_is_launch_invariant,
    check_ols_normal_equation_residual_is_zero,
    check_ols_rank_guard_is_absolute,
    check_ols_refuses_over_capacity,
    check_ols_sample_weight_host_rescale_matches_device,
    check_ols_sample_weight_restores_its_operands,
    check_ols_scale_invariant,
    check_ols_single_column_matches_the_closed_form,
    check_ols_wide_is_the_minimum_norm_solution,
    _mode_name,
)


def main() raises:
    print("== glm/ols_main.mojo [" + _mode_name() + "] ==")
    check_ols_exact()
    check_ols_scale_invariant()
    check_ols_beats_truth_on_noise()
    check_ols_dispatch_routes_special_shapes()
    check_ols_wide_is_the_minimum_norm_solution()
    check_ols_single_column_matches_the_closed_form()
    check_ols_normal_equation_residual_is_zero()
    check_ols_duplicating_a_row_equals_doubling_its_weight()
    check_ols_sample_weight_restores_its_operands()
    check_ols_sample_weight_host_rescale_matches_device()
    check_ols_arms_are_pinned()
    check_ols_refuses_over_capacity()
    check_ols_is_launch_invariant()
    check_ols_host_surface_takes_the_guard()
    check_ols_rank_guard_is_absolute()
    check_ols_card_hashes_raw_bytes()
    check_ols_card_is_emitted()
