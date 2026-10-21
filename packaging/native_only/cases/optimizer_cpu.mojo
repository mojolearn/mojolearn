# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.optim_cpu import run_optim_cases


def main() raises:
    var ex = HostExec()
    run_optim_cases(ex)
    ex.sync()
    print("NATIVE_ONLY_PASS optimizer_cpu")
