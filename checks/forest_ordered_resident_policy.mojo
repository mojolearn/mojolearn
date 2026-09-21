# SPDX-License-Identifier: Apache-2.0
"""Pure policy gate for NVIDIA IDENTICAL ordered resident inference."""
from core.forest_inference_model import forest_ordered_resident_policy
from checks.kernel_matrix import COLUMN_APPLE, COLUMN_NVIDIA, COLUMN_AMD


def main() raises:
    if not forest_ordered_resident_policy[COLUMN_NVIDIA, True, False, False]():
        raise Error("NVIDIA IDENTICAL must select ordered resident")
    if forest_ordered_resident_policy[COLUMN_NVIDIA, False, False, False]():
        raise Error("NVIDIA non-IDENTICAL must retain its existing policy")
    if not forest_ordered_resident_policy[COLUMN_APPLE, True, False, False]():
        raise Error("Apple IDENTICAL must select ordered resident")
    if not forest_ordered_resident_policy[COLUMN_APPLE, False, False, False]():
        raise Error("Apple FAST must select ordered resident")
    if forest_ordered_resident_policy[COLUMN_AMD, True, False, False]():
        raise Error("AMD must not select NVIDIA's default")
    if not forest_ordered_resident_policy[COLUMN_APPLE, True, True, False]():
        raise Error("explicit experimental force did not select the route")
    if forest_ordered_resident_policy[COLUMN_NVIDIA, True, True, True]():
        raise Error("restore switch must win over default and force")
    if forest_ordered_resident_policy[COLUMN_APPLE, False, False, True]():
        raise Error("restore switch must disable Apple FAST default")
    print("PASS forest ordered resident policy")
