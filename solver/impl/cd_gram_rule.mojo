# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shape rule of the IDENTICAL Gram coordinate descent (lane/fam-linear,
2026-10-04), in a file with no device import so the device solver
(`solver/impl/cd.mojo` CD_IDN_GRAM) and the host column
(`solver/host/cd_oracle.mojo`, `bindings/_mojolearn_solver_host.mojo`) read
ONE rule: both take the Gram sweeps for the same fits, or neither does.

`-D MOJOLEARN_CD_IDN_GRAM_OFF` (or the master `MOJOLEARN_IDN_ALL_OFF`)
restores the row sweeps; pass it to the host build too. The default serves
n_cols <= 64, where the Gram (n_cols^2 cells of n_rows products) costs a few
row sweeps; `-D MOJOLEARN_CD_IDN_GRAM_WIDE` raises the bound to 256 for the
A/B on wide data (also to the host build).
"""

from std.sys.compile import is_defined

comptime CD_IDN_GRAM_ON = not (
    is_defined["MOJOLEARN_CD_IDN_GRAM_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime CD_IDN_GRAM_MAX_COLS = 256 if is_defined["MOJOLEARN_CD_IDN_GRAM_WIDE"]() else 64


def cd_idn_gram_shape(n_rows: Int, n_cols: Int) -> Bool:
    """True when the fit takes the Gram sweeps (given the define is on)."""
    comptime if CD_IDN_GRAM_ON:
        return n_cols >= 1 and n_cols <= CD_IDN_GRAM_MAX_COLS and n_rows >= 4 * n_cols
    return False
