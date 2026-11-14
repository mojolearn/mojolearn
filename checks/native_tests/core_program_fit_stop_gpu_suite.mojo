# SPDX-License-Identifier: Apache-2.0
"""Program.fit's device stopping criterion on device sessions.

Serial by nature: program_fit_parameters/program_fit_native reach a session
through the process-wide session table (autograd/ops_api.mojo
_OPS_SESSIONS), whose open/close append to and pop from unlocked lists, so
concurrent sessions from host tasks would race on that table. Four tiny fits
(40 x 3, at most 12 epochs) run one after another.
"""
from autograd.ops_api import OPS_DEVICE
from checks.native_tests.ops_higher_cases import require
from checks.native_tests.core_program_fit_stop_cases import fit_run, check_tol_stop, check_every_epoch
from sequence.exec_device import DeviceExec


def run_checks() raises:
    var failed = 0
    try:
        check_tol_stop(fit_run[DeviceExec, OPS_DEVICE](DeviceExec(), 12, Float32(1e3), 2),
                       fit_run[DeviceExec, OPS_DEVICE](DeviceExec(), 3, Float32(0.0), 0))
    except error:
        print("FAIL core_program_fit_stop_gpu fixture 0", String(error))
        failed += 1
    try:
        check_every_epoch(fit_run[DeviceExec, OPS_DEVICE](DeviceExec(), 4, Float32(0.0), 0),
                          fit_run[DeviceExec, OPS_DEVICE](DeviceExec(), 4, Float32(-1e30), 1))
    except error:
        print("FAIL core_program_fit_stop_gpu fixture 1", String(error))
        failed += 1
    require(failed == 0, "core_program_fit_stop_gpu failed " + String(failed) + " of 2")
    print("PASS core_program_fit_stop_gpu: 2 fixtures (session table: serial)")
