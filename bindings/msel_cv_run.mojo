# SPDX-License-Identifier: Apache-2.0
"""Typed model-selection helpers (RUN): the perf_counter clock and the numeric dtype test."""
from std.time import perf_counter_ns


def msel_seconds() -> Float64:
    """`time.perf_counter()`."""
    return Float64(perf_counter_ns()) / 1e9


def _numeric_dtype(dtype: String) -> Bool:
    return dtype == "<f4" or dtype == "<f8" or dtype == "<i4" or dtype == "<i8" or dtype == "<u4" or dtype == "<u1"
