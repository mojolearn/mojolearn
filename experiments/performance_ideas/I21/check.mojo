# SPDX-License-Identifier: Apache-2.0
"""Actual E/M-step and whole-EM contract gates, including covariance,
empty/collapsed components, iterations, likelihood and geometry invariance.
Rollback builds isolate fused Cholesky and one-drain state lifetime; the
historical stacked E-step regression is a separately opt-in control."""
from experiments.performance_ideas.I21.component_check import check_component_batches
from mixture.checks.gmm_check import check_estep_vs_oracle, check_mstep_vs_oracle, check_recovers_planted_parameters, check_iteration_count_is_identical, check_collapse_is_identical, check_launch_invariance

def main() raises:
    check_component_batches()
    check_estep_vs_oracle()
    check_mstep_vs_oracle()
    check_recovers_planted_parameters()
    check_iteration_count_is_identical()
    check_collapse_is_identical()
    check_launch_invariance()
    print("I21 PASS complete_EM_state likelihood_iterations covariance_empty_geometry")
