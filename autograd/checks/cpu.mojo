# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.gradients import run_gradient_checks
from autograd.checks.lifecycle import run_lifecycle_checks


def main() raises:
    var ex = HostExec()
    run_gradient_checks(ex)
    run_lifecycle_checks(ex)
