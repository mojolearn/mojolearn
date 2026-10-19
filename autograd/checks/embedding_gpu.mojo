# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.embedding_cases import run_embedding_checks


def main() raises:
    var ex = DeviceExec()
    run_embedding_checks(ex)
