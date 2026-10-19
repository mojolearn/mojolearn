# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.view_cases import run_view_checks


def main() raises:
    var ex = HostExec()
    run_view_checks(ex)
