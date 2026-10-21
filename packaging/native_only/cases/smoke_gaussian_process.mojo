# SPDX-License-Identifier: Apache-2.0
# Native fit and posterior prediction reproduce planted training observations.
from gaussian_process.checks.gp_check import check_posterior_recovers_training


def main() raises:
    check_posterior_recovers_training()
    print("NATIVE_ONLY_PASS gaussian_process")
