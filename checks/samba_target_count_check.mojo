# SPDX-License-Identifier: Apache-2.0
"""Exact ignore-index counts, including sizes around the former SIMD width."""
from core.samba_count_impl import samba_target_count


def run_checks() raises:
    var cases = 0
    for n in [1, 7, 8, 9, 16, 31, 32, 33, 65, 128]:
        for pattern in range(4):
            var targets = List[Int32](length=n, fill=3)
            var expected = 0
            for i in range(n):
                var ignored = pattern == 1 or (pattern == 2 and i % 3 == 0)
                targets[i] = Int32(-100 if ignored else (-99 if pattern == 3 else 3))
                if not ignored:
                    expected += 1
            var actual = samba_target_count(Int(targets.unsafe_ptr()), n)
            _ = targets^  # the raw address does not retain the caller's owner
            if actual != expected:
                raise Error("Samba target count: n=" + String(n) + ", pattern="
                            + String(pattern) + ", expected=" + String(expected)
                            + ", actual=" + String(actual))
            cases += 1
    print("Samba target count:", cases, "cases passed")


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
    return Check("samba_target_count_check", cpu=True, gpu=False, serial="", run=run_cases)
