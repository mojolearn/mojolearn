# SPDX-License-Identifier: Apache-2.0
"""Pure policy gate for default ordered resident forest inference."""
from core.forest_inference_model import forest_ordered_resident_policy
from checks.kernel_matrix import COLUMN_APPLE, COLUMN_NVIDIA, COLUMN_AMD
from checks.numerics import NUMERIC_FAST, NUMERIC_DETERMINISTIC, NUMERIC_IDENTICAL


def main() raises:
    if not forest_ordered_resident_policy[COLUMN_NVIDIA, NUMERIC_IDENTICAL, False, False]():
        raise Error("NVIDIA IDENTICAL must select ordered resident")
    if not forest_ordered_resident_policy[COLUMN_NVIDIA, NUMERIC_FAST, False, False]():
        raise Error("NVIDIA FAST must select ordered resident")
    if not forest_ordered_resident_policy[COLUMN_APPLE, NUMERIC_IDENTICAL, False, False]():
        raise Error("Apple IDENTICAL must select ordered resident")
    if not forest_ordered_resident_policy[COLUMN_APPLE, NUMERIC_FAST, False, False]():
        raise Error("Apple FAST must select ordered resident")
    if not forest_ordered_resident_policy[COLUMN_AMD, NUMERIC_IDENTICAL, False, False]():
        raise Error("AMD IDENTICAL must select ordered resident")
    if not forest_ordered_resident_policy[COLUMN_AMD, NUMERIC_FAST, False, False]():
        raise Error("AMD FAST must select ordered resident")
    if forest_ordered_resident_policy[COLUMN_AMD, NUMERIC_DETERMINISTIC, False, False]():
        raise Error("DETERMINISTIC must retain its prior route")
    if not forest_ordered_resident_policy[COLUMN_APPLE, NUMERIC_IDENTICAL, True, False]():
        raise Error("explicit experimental force did not select the route")
    if forest_ordered_resident_policy[COLUMN_NVIDIA, NUMERIC_IDENTICAL, True, True]():
        raise Error("restore switch must win over default and force")
    if forest_ordered_resident_policy[COLUMN_APPLE, NUMERIC_FAST, False, True]():
        raise Error("restore switch must disable Apple FAST default")
    if forest_ordered_resident_policy[COLUMN_AMD, NUMERIC_IDENTICAL, False, True]():
        raise Error("restore switch must disable AMD IDENTICAL default")
    print("PASS forest ordered resident policy")
