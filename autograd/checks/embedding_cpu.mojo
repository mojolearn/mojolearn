# SPDX-License-Identifier: Apache-2.0
from sequence.exec import HostExec
from autograd.checks.embedding_cases import run_embedding_checks


def main() raises:
    var ex = HostExec()
    run_embedding_checks(ex)
