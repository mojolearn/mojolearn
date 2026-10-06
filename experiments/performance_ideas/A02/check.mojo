# SPDX-License-Identifier: Apache-2.0
"""Actual dispatcher one/two-page arms plus bounded deeper resource probe."""
from gemm.experiments.profile_check import run_profile
from gemm.experiments.bounded_staging_check import run_checks


def main() raises:
    run_profile()
    run_checks()
