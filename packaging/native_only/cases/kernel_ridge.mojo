# SPDX-License-Identifier: Apache-2.0
# Native kernel ridge fitted prediction against planted linear function.
from kernel_methods.checks.km_check import check_kernel_ridge_planted_linear


def main() raises:
    check_kernel_ridge_planted_linear()
    print("NATIVE_ONLY_PASS kernel_ridge")
