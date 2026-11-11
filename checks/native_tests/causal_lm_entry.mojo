# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the CausalLM entry table suite."""
from checks.native_tests.causal_lm_entry_suite import run_checks


def main() raises:
    run_checks()
