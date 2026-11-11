# SPDX-License-Identifier: Apache-2.0
"""The existing classification metrics signed-label encoding entry."""
from core.cast_in import CastIn, CAST_KIND_SIGNED
from core.label_cast import host_unique_inverse_cast


def score_encode_signed(addr: Int, item: Int, count: Int, classes_addr: Int, codes_addr: Int) raises -> Int:
    """The ORDER RULE over `count` signed little-endian integer labels of
    `item` bytes (contiguous): the sorted distinct values written to
    `classes_addr` (items of the same width) and one dense int32 code per
    row; returns k. The host label encoder called natively: no Python call
    into the base binding's `prepare_targets` entry, the same classes and
    codes (exact integers)."""
    var src = CastIn(addr, CAST_KIND_SIGNED, item, count, 1, item, item)
    var k = host_unique_inverse_cast(src, 1, classes_addr, codes_addr)
    if k < 0:
        raise Error("RuntimeError: prepare_targets refused integer labels")
    return k

