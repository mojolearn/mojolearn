# SPDX-License-Identifier: Apache-2.0
"""Importable native Prophet CPU fitted-state checks."""
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from sequence.ops_impl import prophet_fit_impl
from checks.native_tests.prophet_state_cases import (
    PROPHET_STATE_FIXTURES, ProphetState, ProphetFitInputs,
    require, prepare_fit, run_queries,
)


def fixture(fixture_id: Int) raises:
    var ex = HostExec()
    var state = ProphetState(fixture_id)
    if fixture_id == 0:
        var inputs = ProphetFitInputs()
        prepare_fit(ex, inputs, state)
        var addresses = inputs.addresses(state)
        var ip = state.ints(120)
        ip.append(2000)
        var fp: List[Float64] = [0.05]
        require(prophet_fit_impl(ex, addresses, ip, fp) == 1, "Prophet full default fit count")
        inputs.guards()
        state.guards()
        _ = inputs
    run_queries(ex, state)
    _ = state


def run_checks() raises:
    var failures = List[Int](length=PROPHET_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            fixture(fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL prophet_state_cpu fixture", fixture_id, String(error))
    host_parallelize(task, PROPHET_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "prophet_state_cpu failed " + String(failed) + " fixtures")
    print("PASS prophet_state_cpu:", PROPHET_STATE_FIXTURES, "parallel native fitted-state fixtures")
