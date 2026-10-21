# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.optim_cpu import run_optim_cases


def main() raises:
    var ex = DeviceExec()
    run_optim_cases(ex)
    ex.sync()
    print("NATIVE_ONLY_PASS optimizer_gpu")
