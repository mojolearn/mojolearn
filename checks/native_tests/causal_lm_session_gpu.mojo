# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the CausalLM session table suite (GPU column)."""
from checks.native_tests.causal_lm_session_gpu_suite import run_checks


def main() raises:
    run_checks()
