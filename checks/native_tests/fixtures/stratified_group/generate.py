# SPDX-License-Identifier: Apache-2.0
"""Writes the class labels of checks/native_tests/core_stratified_group_fp64.mojo.

Moved from python/mojolearn/tests/test_stratified_group_fp64_identity.py
(2026-11-14): the labels are the identity harness's metrics labels
(tools/identity_lanes/metrics.py `_metrics_labels`) of the harness fixtures
(tools/identity_break.py `fixture`, NumPy PCG64 draws) at the former test's
(fixture, rows) pairs. They are stored so the Mojo check needs no NumPy
stream; the groups ((row // 7) % group_count) and the fold-plan oracle are
computed in Mojo.

Run from the repository root:  python checks/native_tests/fixtures/stratified_group/generate.py
Files: <fixture>-<rows>.i32 (little-endian int32 labels, one per row).
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CASES = (("base", 600), ("wide", 600), ("ties", 600), ("hashed", 600), ("base", 577), ("base", 631))


def _harness():
    spec = importlib.util.spec_from_file_location("identity_break_fixture_source", ROOT / "tools" / "identity_break.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics_labels():
    namespace = {"np": np}
    source = (ROOT / "tools" / "identity_lanes" / "metrics.py").read_text()
    start = source.index("def _metrics_labels(")
    end = source.index("\ndef ", start + 1)
    exec(compile(source[start:end], "metrics.py[_metrics_labels]", "exec"), namespace)
    return namespace["_metrics_labels"]


def main():
    harness = _harness()
    labels_of = _metrics_labels()
    for name, rows in CASES:
        values, _, _ = harness.fixture(name)
        labels, _ = labels_of(values, rows)
        np.ascontiguousarray(labels, dtype="<i4").tofile(HERE / f"{name}-{rows}.i32")


if __name__ == "__main__":
    main()
