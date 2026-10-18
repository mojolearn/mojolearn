# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.mlp_cases import run_mlp_cases


def main() raises:
    var ex = DeviceExec()
    run_mlp_cases(ex)
