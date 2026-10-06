# SPDX-License-Identifier: Apache-2.0
"""Actual batched ARIMA likelihood/gradient, complete LBFGS fit, batch
composition invariance, refusal handling and public forecasting oracle.
Candidate and rollback preserve model initialization/differencing. Affine
scan/filter profile revisions remain conditional; no Kalman approximation."""
# I23 experiment qualification pending: compile/fixtures do not establish
# four-column identity or NVIDIA+AMD full-operation speed. Existing promoted
# defaults stay unchanged; this campaign attributes explicit experiment arms.
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from arima.checks.fit_check import check_grad_matches_float64, check_lbfgs_rules_match_glm, check_fit_is_batch_composition_invariant, check_fit_refuses_by_name
from arima.checks.arima_check import check_grad_device_equals_oracle, check_predict_device_equals_oracle, check_kalman_launch_invariant

def main() raises:
    var ctx=DeviceContext()
    var trace=IdentityTrace.disabled()
    check_grad_device_equals_oracle(ctx,trace)
    check_predict_device_equals_oracle(ctx,trace)
    check_kalman_launch_invariant(ctx)
    check_grad_matches_float64(ctx)
    check_lbfgs_rules_match_glm(ctx)
    check_fit_is_batch_composition_invariant(ctx)
    check_fit_refuses_by_name(ctx)
    print("I23 PASS independent_series complete_fit gradient_forecast_refusals")
