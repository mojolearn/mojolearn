# SPDX-License-Identifier: Apache-2.0
# The GEMM route check (autograd/checks/gemm_cases.mojo): every route's
# words equal bit for bit, the float64 oracle, the host's words reported.
from sequence.exec_device import DeviceExec
from sequence.exec import HostExec
from autograd.checks.gemm_cases import run_gemm_cases


def main() raises:
    var dev = DeviceExec()
    var host = HostExec()
    run_gemm_cases(dev, host)
