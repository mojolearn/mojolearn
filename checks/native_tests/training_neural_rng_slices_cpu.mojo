# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the neural RNG slice suite (CPU column)."""
from checks.native_tests.training_neural_rng_slices_cpu_suite import run_checks


def main() raises:
    run_checks()
