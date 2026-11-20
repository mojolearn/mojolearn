# SPDX-License-Identifier: Apache-2.0
"""Standalone entry point for the reusable native test suite
(mamba_step_backward_glue_check_suite.mojo).

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba_step_backward_glue_check.mojo
"""
from mamba.checks.mamba_step_backward_glue_check_suite import run_checks


def main() raises:
    run_checks()
