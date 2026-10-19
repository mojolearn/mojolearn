# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.fused_cases import run_fused_cases


def main() raises:
    var ex = HostExec()
    run_fused_cases(ex)
