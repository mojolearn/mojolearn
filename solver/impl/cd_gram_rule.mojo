# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shape rule of the IDENTICAL Gram coordinate descent (lane/fam-linear,
2026-10-04), in a file with no device import so the device solver
(`solver/impl/cd.mojo` CD_IDN_GRAM) and the host column
(`solver/host/cd_oracle.mojo`, `bindings/_mojolearn_solver_host.mojo`) read
ONE rule: both take the Gram sweeps for the same fits, or neither does.

`-D MOJOLEARN_CD_IDN_GRAM_OFF` (or the master `MOJOLEARN_IDN_ALL_OFF`)
restores the row sweeps; pass it to the host build too. The default serves
the fits where the Gram's build plus sweeps is no more work than the row
sweeps over CD_IDN_GRAM_PAYBACK sweeps (`cd_idn_gram_cost_ok`, up to the
kernel ceiling 256); `-D MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF` restores the
old n_cols <= 64, `-D MOJOLEARN_CD_IDN_GRAM_WIDE` the fixed 256 (also to the
host build).
"""

from std.sys.compile import is_defined

comptime CD_IDN_GRAM_ON = not (
    is_defined["MOJOLEARN_CD_IDN_GRAM_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
# lane fam2-linear: the width bound as candidate arms for the orchestrator's
# A/B (default 64; pass the same define to the device and the host build):
#   -D MOJOLEARN_CD_IDN_GRAM_COLS_128   n_cols <= 128
#   -D MOJOLEARN_CD_IDN_GRAM_COLS_256   n_cols <= 256 (= the older _WIDE)
# 256 is the kernel's ceiling (one block of CD_IDN_GRAM_TPB = 256 threads).
comptime CD_IDN_GRAM_MAX_COLS = (
    256
    if (is_defined["MOJOLEARN_CD_IDN_GRAM_WIDE"]() or is_defined["MOJOLEARN_CD_IDN_GRAM_COLS_256"]())
    else (128 if is_defined["MOJOLEARN_CD_IDN_GRAM_COLS_128"]() else 64)
)


#: lane/no-bench-tuning-2 (2026-10-04): the 64-column default was a guess.
#: The default is now a work rule in (rows, cols), up to the kernel ceiling
#: 256: the Gram costs one build of n_rows * n_cols^2 / 2 products (the
#: symmetric half) plus n_cols^2 per sweep; a row sweep costs 2 * n_rows *
#: n_cols (the dot and the residual update per coordinate). The Gram is taken
#: when it is no more work over CD_IDN_GRAM_PAYBACK sweeps:
#:   n_rows n_cols^2 / 2 + T n_cols^2 <= 2 T n_rows n_cols
#:   <=>  n_cols (n_rows + 2T) <= 4 T n_rows,
#: plus the existing n_rows >= 4 n_cols (a tall system) and n_cols <= 256.
#: T = 64 sweeps (a fit that converges sooner pays a build it did not need;
#: one that runs longer saves more). The bound approaches 4T = 256 for tall
#: data and shrinks for short data (n_rows 1024: n_cols <= 227; n_rows 400:
#: n_cols <= 100 by tallness, 193 by work). Bits move for
#: the fits whose route flips (n_cols 65..255 on tall data), on the device and
#: the host column together: both read this function.
#: `-D MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF=1` restores n_cols <= 64 (pass it
#: to the host build too); the explicit COLS_128 / COLS_256 / WIDE arms keep
#: their fixed bounds.
comptime CD_IDN_GRAM_PAYBACK = 64
comptime CD_IDN_GRAM_FIXED_BOUND = (
    is_defined["MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF"]()
    or is_defined["MOJOLEARN_CD_IDN_GRAM_WIDE"]()
    or is_defined["MOJOLEARN_CD_IDN_GRAM_COLS_256"]()
    or is_defined["MOJOLEARN_CD_IDN_GRAM_COLS_128"]()
)
comptime CD_IDN_GRAM_KERNEL_MAX_COLS = 256


def cd_idn_gram_cost_ok(n_rows: Int, n_cols: Int) -> Bool:
    """The work rule above (no define read: the checks call it directly)."""
    if n_cols < 1 or n_cols > CD_IDN_GRAM_KERNEL_MAX_COLS or n_rows < 4 * n_cols:
        return False
    var t = CD_IDN_GRAM_PAYBACK
    return n_cols * (n_rows + 2 * t) <= 4 * t * n_rows


def cd_idn_gram_shape(n_rows: Int, n_cols: Int) -> Bool:
    """True when the fit takes the Gram sweeps (given the define is on)."""
    comptime if CD_IDN_GRAM_ON:
        comptime if CD_IDN_GRAM_FIXED_BOUND:
            return n_cols >= 1 and n_cols <= CD_IDN_GRAM_MAX_COLS and n_rows >= 4 * n_cols
        return cd_idn_gram_cost_ok(n_rows, n_cols)
    return False
