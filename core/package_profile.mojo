# SPDX-License-Identifier: Apache-2.0
"""Compile-time defaults for the distribution being built."""

from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU

# The native package builder changes only this distribution setting to True.
# Existing wheel and measurement builds keep their explicit mode flags.
comptime NATIVE_PACKAGE = False

# THE SECOND FAST SET (declared, inert: no build passes it yet; lane fx-fast-selector).
# `fast` is `identical` with named relaxations switched on. Off Apple no fast
# set is built, and the selector answers the name `fast` with the identical
# set (python/mojolearn/_backend.py, `_FAST_ANSWER`). When a relaxation is
# measured on a vendor, that vendor's second set is built as a fast build
# (no MOJOLEARN_NUMERIC_IDENTICAL) with `-D MOJOLEARN_MODE_FAST_ROWS` plus the
# relaxation's own define, into the fast tier directory, and the selector then
# picks it because it is there. With the flag and no relaxation define the set
# must be the identical kernels (the mode policy table reads this constant).
# The flag in an identical build is a compile error: a relaxation never
# changes an identical build.
# Owed before such a set can be built: the Apple-target assert below must
# admit MODE_FAST_ROWS, and the release's mode list (tools/RELEASE_MODES.md,
# MOJOLEARN_RELEASE_MODES) must select `fast` for that vendor.
comptime MODE_FAST_ROWS = is_defined["MOJOLEARN_MODE_FAST_ROWS"]()


def identical_mode_enabled() -> Bool:
    comptime if MODE_FAST_ROWS:
        comptime assert not is_defined["MOJOLEARN_NUMERIC_IDENTICAL"](), "MOJOLEARN_MODE_FAST_ROWS builds a fast set; an identical build refuses it"
    comptime if NATIVE_PACKAGE:
        comptime assert not (
            is_defined["MOJOLEARN_NUMERIC_FAST"]()
            and is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
        ), "select only one numeric mode"
        comptime if is_defined["MOJOLEARN_NUMERIC_FAST"]():
            comptime assert TARGET_HAS_APPLE_GPU, "fast mode requires an Apple GPU target"
            return False
        return True
    return is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
