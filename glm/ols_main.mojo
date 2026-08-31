# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the OLS checks. Both halves of them.

The first four are the ACCURACY checks: a planted model recovered within
1%, the target scaled by 5, the least-squares residual beating the truth's,
and `ols.cuh:112-113`'s shape refusal. Three of those four are TOLERANCE
checks and would pass on a build whose summation order differs on every
vendor, which is exactly why the second group exists.

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

from glm.original.ols_check import (
    check_ols_arms_are_pinned,
    check_ols_beats_truth_on_noise,
    check_ols_card_hashes_raw_bytes,
    check_ols_card_is_emitted,
    check_ols_dispatch_guard,
    check_ols_exact,
    check_ols_host_surface_takes_the_guard,
    check_ols_is_launch_invariant,
    check_ols_rank_guard_is_absolute,
    check_ols_refuses_over_capacity,
    check_ols_scale_invariant,
    _mode_name,
)


def main() raises:
    print("== glm/ols_main.mojo [" + _mode_name() + "] ==")
    check_ols_exact()
    check_ols_scale_invariant()
    check_ols_beats_truth_on_noise()
    check_ols_dispatch_guard()
    check_ols_arms_are_pinned()
    check_ols_refuses_over_capacity()
    check_ols_is_launch_invariant()
    check_ols_host_surface_takes_the_guard()
    check_ols_rank_guard_is_absolute()
    check_ols_card_hashes_raw_bytes()
    check_ols_card_is_emitted()
