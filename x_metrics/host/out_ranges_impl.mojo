# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host x_metrics binding's output-range check, typed and Python free
(moved word for word from `bindings/_mojolearn_x_metrics_host.mojo`)."""
from x_metrics.common import IP


def check_out_ranges(oa: Int, no: Int, n: Int) raises:
    """Each of the `no` output records at `oa` (four int32 words: lo, hi,
    count, unused) must lie inside the `n`-word arena."""
    for k in range(no):
        var lo = Int(IP(unsafe_from_address=oa).unsafe_load(4 * k))
        var hi = Int(IP(unsafe_from_address=oa).unsafe_load(4 * k + 1))
        var cn = Int(IP(unsafe_from_address=oa).unsafe_load(4 * k + 2))
        if lo < 0 or hi < lo or hi > n or cn >= n:
            raise Error("x_metrics: output range outside the arena")
