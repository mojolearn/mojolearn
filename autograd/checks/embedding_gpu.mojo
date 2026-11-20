# SPDX-License-Identifier: Apache-2.0
from sequence.exec_device import DeviceExec
from autograd.checks.embedding_cases import run_embedding_checks


def run_checks() raises:
    var ex = DeviceExec()
    run_embedding_checks(ex)
