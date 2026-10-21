# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from packaging.native_only.cases.autograd_core import check_tape


def main() raises:
    var ex = DeviceExec()
    check_tape(ex)
    ex.sync()
    print("NATIVE_ONLY_PASS autograd_gpu")
