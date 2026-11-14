# SPDX-License-Identifier: Apache-2.0
"""SmallMLPTrainer program fixtures on the cpu column (training_small_mlp_cases.mojo).

The fixtures run one after another: the program store and the session
registry they open into (autograd/small_mlp_program.mojo `_STORE`,
autograd/ops_api.mojo `ops_session_open`) are process lists appended
without a lock, so concurrent opens from pool workers would race. The work
inside each fixture is the program's own parallel kernels.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from autograd.ops_api import OPS_HOST
from sequence.exec import HostExec
from checks.native_tests.training_small_mlp_cases import (
    SMALL_MLP_FIXTURES, adamw_fixture, loss_and_grads_fixture, sessions_fixture,
)
from checks.native_tests.training_check_common import require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(SMALL_MLP_FIXTURES):  # ordered: see the module note
        try:
            if unit == 0:
                adamw_fixture[HostExec, OPS_HOST](HostExec())
            elif unit == 1:
                loss_and_grads_fixture[HostExec, OPS_HOST](HostExec())
            else:
                sessions_fixture[HostExec, OPS_HOST](HostExec(), HostExec())
        except error:
            failed += 1
            print("FAIL training_small_mlp_cpu fixture", unit, String(error))
    require(failed == 0, "training_small_mlp_cpu failed " + String(failed) + " fixtures")
    print("PASS training_small_mlp_cpu:", SMALL_MLP_FIXTURES, "fixtures (AdamW equations, cross-entropy, sessions forward)")
