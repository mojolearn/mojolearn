# SPDX-License-Identifier: Apache-2.0
# Native IVF index build and all-list search consumed against brute force.
from ivf.checks.ivf_check import check_nprobe_equals_nlists_is_brute_force


def main() raises:
    check_nprobe_equals_nlists_is_brute_force()
    print("NATIVE_ONLY_PASS ivf")
