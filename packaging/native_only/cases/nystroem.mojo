# SPDX-License-Identifier: Apache-2.0
# Native full-landmark transformation reconstructs exact kernel.
from kernel_methods.checks.km_check import check_nystroem_full_equals_exact_kernel


def main() raises:
    check_nystroem_full_equals_exact_kernel()
    print("NATIVE_ONLY_PASS nystroem")
