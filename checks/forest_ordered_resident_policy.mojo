# SPDX-License-Identifier: Apache-2.0
"""Pure policy gate for default ordered resident forest inference."""
from core.forest_inference_model import forest_ordered_resident_policy
from checks.kernel_matrix import COLUMN_APPLE, COLUMN_NVIDIA, COLUMN_AMD
from checks.numerics import NUMERIC_FAST, NUMERIC_IDENTICAL


def run_checks() raises:
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
    if not forest_ordered_resident_policy[COLUMN_APPLE, NUMERIC_IDENTICAL, True, False]():
        raise Error("explicit experimental force did not select the route")
    if forest_ordered_resident_policy[COLUMN_NVIDIA, NUMERIC_IDENTICAL, True, True]():
        raise Error("restore switch must win over default and force")
    if forest_ordered_resident_policy[COLUMN_APPLE, NUMERIC_FAST, False, True]():
        raise Error("restore switch must disable Apple FAST default")
    if forest_ordered_resident_policy[COLUMN_AMD, NUMERIC_IDENTICAL, False, True]():
        raise Error("restore switch must disable AMD IDENTICAL default")
    print("PASS forest ordered resident policy")


# Registered by lane checks-toplevel (2026-11-16): a standalone main() under
# checks/ that no CI job ran. One process per check name in CI keeps any
# fresh-process precondition. Host column: the suite is host work and does not branch on the target.
from checks.native_tests.runner import Check, Column, Report


def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("forest_ordered_resident_policy", cpu=True, gpu=False, serial="", run=run_cases)
