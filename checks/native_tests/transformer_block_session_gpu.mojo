# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the Transformer block session suite (GPU column)."""
from checks.native_tests.transformer_block_session_gpu_suite import run_checks


def main() raises:
    run_checks()
