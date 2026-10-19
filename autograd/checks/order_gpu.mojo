# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.order_cases import run_order_checks


def main() raises:
    var ex = DeviceExec()
    run_order_checks(ex)
