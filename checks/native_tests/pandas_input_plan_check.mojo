# SPDX-License-Identifier: Apache-2.0
"""Independent native preparation fixtures use the existing host worker pool."""
from core.host_parallel import host_parallelize
from checks.native_tests.pandas_input_plan_cases import PANDAS_PLAN_FIXTURES, require, run_pandas_plan_fixture
from checks.native_tests.runner import Check, Column, Report


def run_checks() raises:
    var failures = List[Int](length=PANDAS_PLAN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            run_pandas_plan_fixture(fixture_id)
        except error:
            print("FAIL pandas_input_plan fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, PANDAS_PLAN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "pandas_input_plan failed " + String(failed) + " fixtures")
    print("PASS pandas_input_plan:", PANDAS_PLAN_FIXTURES, "native metadata/conversion/ownership fixtures, parallel")


# Registered by lane checks-register (2026-11-16): it was a standalone main()
# that no CI job ran. One process per check name in CI keeps any fresh-process
# precondition.
# Host column: the suite is host work (its fixtures run across host
# cores inside run_checks; the check counts as one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("pandas_input_plan_check", cpu=True, gpu=False, serial="", run=run_cases)
