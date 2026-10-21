# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from packaging.native_only.cases.autograd_conv_core import check_conv_chains


def main() raises:
    var ex = DeviceExec()
    check_conv_chains(ex)
    ex.sync()
    print("NATIVE_ONLY_PASS autograd_conv_gpu")
