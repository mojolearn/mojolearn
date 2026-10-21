# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from packaging.native_only.cases.sequence_optimizer_core import check_sequence_optimizers


def main() raises:
    var ex = DeviceExec()
    check_sequence_optimizers(ex)
    ex.sync()
    print("NATIVE_ONLY_PASS sequence_optimizer_gpu")
