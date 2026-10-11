# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lu-delayed-update (2026-10-11): MOJOLEARN_IDN_LU_DELAYED_UPDATE's switch
and widths, shared by the device half (x_decomp/lu_delayed.mojo, where THE
ORDER is stated) and the host column (x_decomp/host.mojo `lu_delayed_host`).
Constants only: no GPU import, no host threads.
"""

from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: The switch: IDENTICAL builds with `-D MOJOLEARN_IDN_LU_DELAYED_UPDATE`.
#: Needs LU_FAST_STEP1 (the one-launch panel; IDENTICAL default) on the
#: device; the host twin keys on the same two so the columns agree.
#: (LU_FAST_STEP1's IDENTICAL condition is spelled here, not imported: this
#: file is the host column's and imports nothing GPU; x_decomp/lu_delayed.mojo
#: asserts the two agree.)
comptime IDN_LU_DELAYED_UPDATE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_GATES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
    and not is_defined["MOJOLEARN_LU_FAST_STEP1_OFF"]()
    and is_defined["MOJOLEARN_IDN_LU_DELAYED_UPDATE"]()
)
#: The inner panel width (main's LU_PANEL_NB; NB128's loss was widening it).
comptime LDP = 32
#: The outer block, the big update's k. 256 = two contract leaves: wide enough
#: that the trailing square is touched 8x less often than at 32, narrow enough
#: that the in-block updates (n x <= 224 at k = 32) stay a small share. A
#: cost rule on the GEMM's leaf size, not on any matrix dimension.
comptime LDU_NB = 256
#: The host fold buffer: leaves of one BU call (k <= LDU_NB).
comptime LDU_MAX_LEAVES = 8
