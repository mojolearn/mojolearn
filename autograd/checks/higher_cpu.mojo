# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.higher import run_higher_checks


def main() raises:
    var ex = HostExec()
    run_higher_checks(ex)
