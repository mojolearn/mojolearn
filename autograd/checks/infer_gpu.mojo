# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.infer_cases import run_infer_checks


def main() raises:
    var ex = DeviceExec()
    run_infer_checks(ex)
