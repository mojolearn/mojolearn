# SPDX-License-Identifier: Apache-2.0
# Native seeded random features and consumed transform against independent scalar oracle.
from kernel_methods.checks.km_check import check_rbf_sampler_vs_oracle


def main() raises:
    check_rbf_sampler_vs_oracle()
    print("NATIVE_ONLY_PASS rbf_sampler")
