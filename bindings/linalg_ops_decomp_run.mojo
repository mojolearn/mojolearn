# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the linalg decompositions (bindings/linalg_ops_decomp.mojo):
the TSQR route test. Typed Mojo only, no Python import
(tools/python_free_dirs.sh); moved unchanged (lane rs-decomposition,
2026-11-12)."""

comptime TS_MAX_N = 512


def _lin_tsqr_on(rows: Int, cols: Int) -> Bool:
    """`_tsqr_on(rows, cols)`: tall and at most TS_MAX_N wide."""
    return cols >= 1 and cols <= TS_MAX_N and rows >= cols
