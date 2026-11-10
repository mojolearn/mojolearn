# SPDX-License-Identifier: Apache-2.0
"""Importable native Prophet GPU fitted-state checks."""
from sequence.exec_device import DeviceExec
from sequence.fit_team_impl import prophet_fit_team_impl
from checks.native_tests.prophet_state_cases import (
    PROPHET_STATE_FIXTURES, ProphetState, ProphetFitInputs,
    require, prepare_fit, run_queries,
)


def fixture(fixture_id: Int) raises:
    var ex = DeviceExec()
    var state = ProphetState(fixture_id)
    if fixture_id == 0:
        var inputs = ProphetFitInputs()
        prepare_fit(ex, inputs, state)
        var addresses = inputs.addresses(state)
        var ip = state.ints(120)
        ip.append(2000)
        var fp: List[Float64] = [0.05]
        require(prophet_fit_team_impl(ex, addresses, ip, fp) == 1, "Prophet full default fit count")
        inputs.guards()
        state.guards()
        _ = inputs
    run_queries(ex, state)
    _ = state


def run_checks() raises:
    # Shared sequence DeviceContext: submit independent states in order; every
    # production fit/prediction uses its existing parallel device kernels.
    var failed = 0
    for fixture_id in range(PROPHET_STATE_FIXTURES):
        try:
            fixture(fixture_id)
        except error:
            failed += 1
            print("FAIL prophet_state_gpu fixture", fixture_id, String(error))
    require(failed == 0, "prophet_state_gpu failed " + String(failed) + " fixtures")
    print("PASS prophet_state_gpu:", PROPHET_STATE_FIXTURES, "native fitted-state fixtures")
