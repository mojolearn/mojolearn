# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.fold_cases import run_fold_cases


def main() raises:
    var ex = HostExec()
    run_fold_cases(ex)
