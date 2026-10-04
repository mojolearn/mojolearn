# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shape rule of the tiled multinomial gradient (lane fam2-linear,
2026-10-04), in a file with no device import so the device objective
(`glm/impl/qn/glm_base.mojo` QN_TILED_MULTI), the host column
(`glm/host/qn_oracle.mojo`) and the check mirror
(`glm/checks/multinomial_check.mojo`) read ONE rule: all take the tile order
for the same fits, or none does.

Default ON: `C > 1` gradients wider than the row-coalesced kernel's 256
cells (`qn_coalesced_applies` on NVIDIA / AMD) take the tile order on every
vendor. `-D MOJOLEARN_QN_TILED_MULTI_OFF` (or the master
`MOJOLEARN_IDN_ALL_OFF`) restores one block per cell; pass it to the host
build too. Candidate arm (default OFF): `-D MOJOLEARN_QN_TILED_MULTI_ALL`
tiles every `C > 1` gradient, the narrow ones included (also to the host
build).
"""

from std.sys.compile import is_defined

comptime QN_TILED_MULTI_ON = not (
    is_defined["MOJOLEARN_QN_TILED_MULTI_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime QN_TILED_MULTI_ALL = is_defined["MOJOLEARN_QN_TILED_MULTI_ALL"]()
#: rows per tile (= QNT_ROWS, HOST_QNT_ROWS)
comptime QNTM_ROWS = 256
#: narrower gradients keep the row-coalesced / one-block-per-cell order
comptime QNTM_MIN_CELLS = 256
#: the tile partials' workspace bound, in floats (256 MB)
comptime QNTM_MAX_WS = 1 << 26


def qn_tiled_multi_tiles(n_rows: Int) -> Int:
    return (n_rows + QNTM_ROWS - 1) // QNTM_ROWS


def qn_tiled_multi_shape(n_rows: Int, d: Int, c: Int) -> Bool:
    """True when the `C > 1` gradient takes the tile order (IDENTICAL
    builds; the device gate adds the mode test)."""
    comptime if QN_TILED_MULTI_ON:
        if c <= 1 or d < 1 or n_rows < 1:
            return False
        if qn_tiled_multi_tiles(n_rows) * d * c > QNTM_MAX_WS:
            return False
        comptime if QN_TILED_MULTI_ALL:
            return True
        return d * c > QNTM_MIN_CELLS
    return False
