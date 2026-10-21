# SPDX-License-Identifier: Apache-2.0
# Native factorization and triangular solves with consumed residual against original equation.
from cholesky.checks.cholesky_check import check_cho_solve_residual


def main() raises:
    check_cho_solve_residual()
    print("NATIVE_ONLY_PASS cholesky")
