# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.import_cases import run_import_checks


def main() raises:
    var ex = HostExec()
    run_import_checks(ex)
