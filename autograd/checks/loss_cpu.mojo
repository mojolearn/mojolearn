# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.losses import run_loss_checks


def run_checks() raises:
    var ex = HostExec()
    run_loss_checks(ex)
