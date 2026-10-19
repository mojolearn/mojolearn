# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.data_cases import run_data_checks


def main() raises:
    var ex = DeviceExec()
    run_data_checks(ex)
