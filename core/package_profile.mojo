# SPDX-License-Identifier: Apache-2.0
"""Compile-time defaults for the distribution being built."""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

# The native package builder changes only this distribution setting to True.
# Existing wheel and measurement builds keep their explicit mode flags.
comptime NATIVE_PACKAGE = False


def identical_mode_enabled() -> Bool:
    comptime if NATIVE_PACKAGE:
        comptime assert not (
            is_defined["MOJOLEARN_NUMERIC_FAST"]()
            and is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
        ), "select only one numeric mode"
        comptime if is_defined["MOJOLEARN_NUMERIC_FAST"]():
            comptime assert has_apple_gpu_accelerator(), "fast mode requires an Apple GPU target"
            return False
        return True
    return is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
