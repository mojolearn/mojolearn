# SPDX-License-Identifier: Apache-2.0
# Native whole fit_transform:8 rows,3 neighbors,4 epochs; consumed shape/finite/noncollapsed layout, seeded replay and unsupported-dimension refusal. This is not an independent numerical-accuracy oracle.
from umap.checks.estimator_check import main as check_0


def main() raises:
    check_0()
    print("NATIVE_ONLY_PASS umap")
