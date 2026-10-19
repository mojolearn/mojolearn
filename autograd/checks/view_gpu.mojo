# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.view_cases import run_view_checks


def main() raises:
    var ex = DeviceExec()
    run_view_checks(ex)
