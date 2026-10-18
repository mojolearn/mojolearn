# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.mlp_cases import run_mlp_cases


def main() raises:
    var ex = HostExec()
    run_mlp_cases(ex)
