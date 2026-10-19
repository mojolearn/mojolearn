# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.infer_cases import run_infer_checks


def main() raises:
    var ex = HostExec()
    run_infer_checks(ex)
