# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from packaging.native_only.cases.sequential_core import check_model


def main() raises:
    var ex = DeviceExec()
    check_model(ex)
    ex.sync()
    print("NATIVE_ONLY_PASS sequential_gpu")
