# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.losses import run_loss_checks


def run_checks() raises:
    var ex = DeviceExec()
    run_loss_checks(ex)
