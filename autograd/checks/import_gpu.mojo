# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.import_cases import run_import_checks


def main() raises:
    var ex = DeviceExec()
    run_import_checks(ex)
