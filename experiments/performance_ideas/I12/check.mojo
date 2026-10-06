# SPDX-License-Identifier: Apache-2.0
"""Focused independent coordinates, host iterate replay, launch geometry,
coefficient prediction, refusal names, and optimization quality gates.
Gram profiles are independent; the full speculative schedule has explicit
first-acceptance, status, logical-count and selected-gradient witnesses."""
from max.gpu.host import DeviceContext
from experiments.performance_ideas.I12.trials_check import check_exact_trial_schedule
from glm.checks.logistic_check import check_logistic_device_equals_host, check_logistic_is_a_minimizer, check_owlqn_is_a_minimizer
from glm.checks.multinomial_check import check_softmax_device_equals_host, check_softmax_is_a_minimizer
from solver.checks.cd_check import check_cd_refuses_by_name, check_cd_recovers_the_planted_support, check_cd_device_equals_oracle, check_cd_is_launch_invariant, check_cd_elasticnet_arms_reach, check_cd_predict_matches_host

def main() raises:
    var ctx=DeviceContext()
    check_exact_trial_schedule(ctx)
    check_cd_refuses_by_name()
    check_cd_recovers_the_planted_support()
    check_cd_device_equals_oracle()
    check_cd_is_launch_invariant()
    check_cd_elasticnet_arms_reach()
    check_cd_predict_matches_host()
    check_logistic_device_equals_host()
    check_logistic_is_a_minimizer()
    check_owlqn_is_a_minimizer()
    check_softmax_device_equals_host()
    check_softmax_is_a_minimizer()
    print("I12 PASS accepted_coordinate_iterates GLM_line_search_minimizers prediction_replay")
