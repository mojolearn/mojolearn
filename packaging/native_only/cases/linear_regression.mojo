# SPDX-License-Identifier: Apache-2.0
# Native OLS fit and consumed coefficients against planted exact model; Python estimator intercept/validation not exercised.
from glm.checks.ols_check import check_ols_exact


def main() raises:
    check_ols_exact()
    print("NATIVE_ONLY_PASS linear_regression")
