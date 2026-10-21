# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shape rule of the tiled multinomial gradient (lane fam2-linear,
2026-10-04), in a file with no device import so the device objective
(`glm/impl/qn/glm_base.mojo` QN_TILED_MULTI), the host column
(`glm/host/qn_oracle.mojo`) and the check mirror
(`glm/checks/multinomial_check.mojo`) read ONE rule: all take the tile order
for the same fits, or none does.

Every `C > 1` gradient takes the tile order on every vendor (lane
debt-block-per-n, 2026-10-21). The narrow gradients (`d * c <= 256` cells)
used to keep the STATS_TPB-chain order (the row-coalesced kernel or
`xtdz_multi_kernel`, one block per cell walking all n rows), and a gradient
whose tile partials passed QNTM_MAX_WS fell back to `xtdz_multi_kernel`;
now the device runs the cells in workspace-bounded chunks
(`qntm_chunk_cells`), which leaves every chain and fold as it is. BITS: the
narrow multinomial gradients move to the tile order on NVIDIA, AMD, Apple
and the host column together. `-D MOJOLEARN_QN_TILED_MULTI_OFF` and the
candidate `-D MOJOLEARN_QN_TILED_MULTI_ALL` are retired (refused in
core/six_lane_experiment_guards.mojo); `-D MOJOLEARN_IDN_ALL_OFF` no
longer reaches this rule.
"""
# TOMBSTONE: MOJOLEARN_QN_TILED_MULTI_OFF (retired rollback) deleted 2026-10-21 by lane debt-block-per-n; code recoverable at 28b8ed11b.
# Restore: git apply experiments/removed/MOJOLEARN_QN_TILED_MULTI_OFF.patch; record in docs/TOMBSTONES.md.

#: rows per tile (= QNT_ROWS, HOST_QNT_ROWS)
comptime QNTM_ROWS = 256
#: the tile partials' workspace bound, in floats (256 MB): the cells run in
#: chunks of at most this many partials
comptime QNTM_MAX_WS = 1 << 26


def qn_tiled_multi_tiles(n_rows: Int) -> Int:
    return (n_rows + QNTM_ROWS - 1) // QNTM_ROWS


def qntm_chunk_cells(n_rows: Int, cells: Int) -> Int:
    """Gradient cells per chunk of `qntm_tiled_into`: all of them when
    their tile partials fit QNTM_MAX_WS, else as many as fit (at least
    one). A workspace bound, not a shape rule: it never changes a bit."""
    var tiles = max(qn_tiled_multi_tiles(n_rows), 1)
    return max(1, min(cells, QNTM_MAX_WS // tiles))


def qntm_ws_floats(n_rows: Int, cells: Int) -> Int:
    """Floats of tile partials one chunk of `qntm_tiled_into` stores."""
    return max(qn_tiled_multi_tiles(n_rows), 1) * qntm_chunk_cells(n_rows, cells)


def qn_tiled_multi_shape(n_rows: Int, d: Int, c: Int) -> Bool:
    """True when the `C > 1` gradient takes the tile order (IDENTICAL
    builds; the device gate adds the mode test): every non-empty one."""
    return c > 1 and d >= 1 and n_rows >= 1
