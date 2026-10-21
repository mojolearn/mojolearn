# SPDX-License-Identifier: Apache-2.0
"""Typed finite check moved word for word from bindings/_mojolearn_neural_host.mojo."""
from std.math import isfinite


def _finite(values: List[Float32], what: String) raises:
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error(what + " is not finite")
