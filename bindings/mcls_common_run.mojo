# SPDX-License-Identifier: Apache-2.0
# RUN pieces shared by the classification metric doors: typed only, no
# Python import (tools/python_free_dirs.sh). Lane metric-programs-2
# (2026-11-12): split from mcls_common.mojo.

comptime MC_NONE = -1
comptime MC_INT32_MAX = 2147483647


def mc_view_kind(dtype: String) raises -> Int:
    """The integer item kind of an array typestr: 1 signed, 2 unsigned
    (bool bytes read as unsigned), 0 none (floats, str, object, uint64)."""
    if dtype.byte_length() < 3:
        return 0
    var tail = String(dtype[byte=1:dtype.byte_length()])
    if tail == "i1" or tail == "i2" or tail == "i4" or tail == "i8":
        return 1
    if tail == "u1" or tail == "u2" or tail == "u4" or tail == "b1":
        return 2
    return 0
