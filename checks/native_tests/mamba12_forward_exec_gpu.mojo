# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the Mamba-1/2 forward plumbing suite (GPU)."""
from checks.native_tests.mamba12_forward_exec_gpu_suite import run_checks


def main() raises:
    run_checks()
