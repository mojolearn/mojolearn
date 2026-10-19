# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.order_cases import run_order_checks


def main() raises:
    var ex = HostExec()
    run_order_checks(ex)
