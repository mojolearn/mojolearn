# SPDX-License-Identifier: Apache-2.0
"""Compile-time defaults for the distribution being built."""

from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU

# The native package builder changes only this distribution setting to True.
# Existing wheel and measurement builds keep their explicit mode flags.
comptime NATIVE_PACKAGE = False

# THE FAST-ROWS SET (lanes fx-mode-policy-core, fx-fast-selector, fx-fast-set-build).
# `fast` is `identical` with named relaxations switched on, so a fast set off
# Apple is THE IDENTICAL BUILD with the contract off:
#     -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_MODE_FAST_ROWS [-D <row define> ...]
# It compiles the identical kernels (`identical_mode_enabled()` stays True, so
# `GLOBAL_NUMERIC_MODE`, every kernel-form test, the reported mode code, the
# slot names and the entry-table keys are the identical set's), and
# checks/mode_policy.mojo turns the contract off (`POLICY_BUILD_FAST_ROWS`),
# which is what lets a row define relax a seam. With the flag and no row
# define the set is the identical kernels. A row define without the flag is a
# compile error (`_row` there): a relaxation never changes an identical build.
# How it is built and installed today: as a measurement wheel whose one GPU
# set (and host set) is the fast-rows set, in the identical tier directory
# (`tools/release.py candidate --owed-arm fast-rows` or `--defines
# MOJOLEARN_MODE_FAST_ROWS,<row define>`; the build scripts add
# NUMERIC_IDENTICAL for that tier and bindings/build_defines.sh adds the
# rest). The selector answers the name `fast` with that set
# (python/mojolearn/_backend.py, `_FAST_ANSWER`), the default name runs it
# too, and `python -m mojolearn verify` refuses to label it identical when a
# row is on (the exported mode policy mask). It is raced against the plain
# wheel of the same commit as an ordinary B arm.
# NOT built: the flag in a legacy fast build (no NUMERIC_IDENTICAL; the fast
# tier scripts, Apple's shipped fast set). That build compiles the old fast
# arms, and the table reads it as POLICY_BUILD_LEGACY_FAST, so the flag would
# name a set that is not the identical kernels: refused below.
# Owed before one wheel can carry the identical set AND a fast-rows set side
# by side (the fast tier directory): a tier identity separate from the kernel
# form. A fast-rows binary reports mode code 1, publishes under the
# `...Identical...` SharedSlot names and the mode-1 entry-table keys
# (`Int(GLOBAL_NUMERIC_MODE)` in bindings/), so in the fast directory the
# loader refuses it (`_check_tier_mode`) and beside the identical set it
# would take that set's slots.
comptime MODE_FAST_ROWS = is_defined["MOJOLEARN_MODE_FAST_ROWS"]()


def identical_mode_enabled() -> Bool:
    comptime if NATIVE_PACKAGE:
        comptime assert not (
            is_defined["MOJOLEARN_NUMERIC_FAST"]()
            and is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
        ), "select only one numeric mode"
        comptime assert not (MODE_FAST_ROWS and is_defined["MOJOLEARN_NUMERIC_FAST"]()), "MOJOLEARN_MODE_FAST_ROWS is the identical kernels with the contract off; it is not combined with MOJOLEARN_NUMERIC_FAST"
        comptime if is_defined["MOJOLEARN_NUMERIC_FAST"]():
            comptime assert TARGET_HAS_APPLE_GPU, "fast mode requires an Apple GPU target"
            return False
        return True
    comptime assert not MODE_FAST_ROWS or is_defined["MOJOLEARN_NUMERIC_IDENTICAL"](), "MOJOLEARN_MODE_FAST_ROWS is the identical kernels with the contract off: build it as the identical tier (-D MOJOLEARN_NUMERIC_IDENTICAL=1), not as a legacy fast build"
    return is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
