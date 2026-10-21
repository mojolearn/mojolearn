# SPDX-License-Identifier: Apache-2.0
# Native EM fit consumes means, covariance and component weights against planted separated mixture.
from mixture.checks.gmm_check import check_recovers_planted_parameters


def main() raises:
    check_recovers_planted_parameters()
    print("NATIVE_ONLY_PASS gaussian_mixture")
