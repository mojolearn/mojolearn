# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.losses import run_loss_checks


def main() raises:
    var ex = HostExec()
    run_loss_checks(ex)
