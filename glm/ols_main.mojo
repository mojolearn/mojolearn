"""Entry point for the OLS checks."""

from glm.mojo_only.ols_check import (
    check_ols_beats_truth_on_noise,
    check_ols_exact,
    check_ols_scale_invariant,
)


def main() raises:
    check_ols_exact()
    check_ols_scale_invariant()
    check_ols_beats_truth_on_noise()
