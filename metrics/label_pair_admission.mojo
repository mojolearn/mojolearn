# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics bodies' input admission, written once (lane r3-small, 2026-11-22).

The device column (`metrics/estimator.mojo`) and the host column
(`metrics/host/*`, `metrics/mojolearn_metrics_host_run.mojo`) each carried a
word-for-word copy of these checks; the copies are gone and both columns
import them from here, so a refusal is the same sentence on every column.
A leaf module: no imports, so the host oracle can use it without a cycle
(`classification_oracle` imports `metrics_oracle`).
"""


def metrics_check_pair(y_true: List[Int32], y_pred: List[Int32], n: Int) raises:
    """Both label arrays must hold at least `n` entries. A short list is a
    read past the end, which is a crash on a good day."""
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    if len(y_true) < n or len(y_pred) < n:
        raise Error(
            "metrics: labels_true holds " + String(len(y_true))
            + " and labels_pred holds " + String(len(y_pred))
            + " entries, both must hold at least n = " + String(n)
        )


def metrics_check_float_pair(a: List[Float32], b: List[Float32], n: Int) raises:
    """Both float arrays must hold at least `n` entries."""
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    if len(a) < n or len(b) < n:
        raise Error(
            "metrics: the two float arrays hold " + String(len(a)) + " and "
            + String(len(b)) + " entries, both must hold at least n = "
            + String(n)
        )


def metrics_check_range(lower: Int32, upper: Int32) raises:
    """The shared label range must not be empty."""
    if upper < lower:
        raise Error(
            "metrics: upper_class_range (" + String(upper)
            + ") is below lower_class_range (" + String(lower) + ")"
        )
