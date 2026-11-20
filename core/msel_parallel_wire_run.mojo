# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/msel_parallel_wire.mojo` (wave 8 run split): the native CV
wire's nesting bound as typed Mojo. Imports no Python."""


def _depth(depth: Int) raises:
    if depth > 64:
        raise Error("ValueError: native CV configuration exceeds 64 nested levels")
