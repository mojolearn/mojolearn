# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.fold_cases import run_fold_cases


def main() raises:
    var ex = DeviceExec()
    run_fold_cases(ex)
