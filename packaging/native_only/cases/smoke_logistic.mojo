# SPDX-License-Identifier: Apache-2.0
# Native binary logistic fit loop and consumed fitted coefficients/training accuracy.
from glm.checks.logistic_check import check_logistic_planted_separable


def main() raises:
    check_logistic_planted_separable()
    print("NATIVE_ONLY_PASS logistic")
