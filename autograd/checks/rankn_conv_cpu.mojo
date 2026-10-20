# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.rankn_conv_cases import run_rankn_conv_checks


def main() raises:
    var ex = HostExec()
    run_rankn_conv_checks(ex)
