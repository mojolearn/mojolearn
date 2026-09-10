# SPDX-License-Identifier: Apache-2.0
"""Register-column ownership: exact layout and adversarial tail qualification.

Use IDENTICAL, with/without MOJOLEARN_KNN_IDENTICAL_COALESCED_COLUMNS.
Every device cell and selected output is checked against the existing scalar
pinned kernel; no tolerance or output fingerprint substitutes for equality.
"""
from bench.knn_index_layout_main import _case
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.pinned_distance_tile import (
    RT_TPB, RT_COLS, RT_TILE_COLS, RT_COALESCED_COLUMNS, _rt_column,
)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("column-ownership qualification requires IDENTICAL")
    # A bijection proves no duplicate writer or missing full-tile column.
    for block in range(2):
        var seen = List[Int]()
        for i in range(RT_TILE_COLS):
            seen.append(0)
        for lane in range(RT_TPB):
            for slot in range(RT_COLS):
                var col = _rt_column(block, lane, slot) - block * RT_TILE_COLS
                if col < 0 or col >= RT_TILE_COLS:
                    raise Error("column ownership escaped tile")
                seen[col] += 1
        for col in range(RT_TILE_COLS):
            if seen[col] != 1:
                raise Error("column ownership is not bijective")
    # Cross lane, register-slot, block, and query-row boundaries. Profiles
    # cover duplicates, cancellation, signed zero/subnormal, independent data.
    for profile in range(4):
        for root in range(2):
            _case(1, 1, 1, profile, root, False, 7)
            _case(7, 127, 8, profile, root, False, 7)
            _case(9, 129, 17, profile, root, False, 7)
            _case(9, 511, 32, profile, root, False, 7)
            _case(17, 513, 33, profile, root, False, 7)
    print("COALESCED DISTANCE PASS", "enabled", RT_COALESCED_COLUMNS,
          "cases", 40, "ownership_cells", 2 * RT_TILE_COLS)
