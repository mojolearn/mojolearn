# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GLM target-range check in native code (lane fam2-linear, 2026-10-04).

The Python layer walked every target (`yv.tolist()`, `min(y)`, `sum(y)`)
before a Poisson / Gamma / Tweedie fit: data work in Python on a GPU route.
The check is now the fit's own: the device grid (`x_linear/device.mojo`
`glm_ydom_kernel`, one thread a row, three flags) and the CPU binding
(`glm_ydom_host`) decide it from the same three facts, by bits:

    has_neg   some y < 0
    has_pos   some y > 0
    has_zero  some y == 0

scikit-learn's rule (their `_check_y`; the message in `_expansion_linear.py` `_GLMBase.fit`): for 1 <= power < 2
the range fails when `min(y) < 0 or sum(y) <= 0`; with no negative target the
sum is `<= 0` exactly when no target is positive, so that is `has_neg or not
has_pos`. For power >= 2 it fails when `min(y) <= 0`: `has_neg or has_zero`.
A failed range returns `res[d + 2] = -1` (the converged word) and no fit; the
Python layer raises the same ValueError.

Lane cpu2-l10-linear (2026-10-04): the check is the only route. The
`MOJOLEARN_XLIN_GLM_YDOM_OFF` arm (and its `MOJOLEARN_IDN_ALL_OFF` hook) that
left the Python walk in place is removed: a fix, not an optimization
(CPU data work on a GPU route). This file has no device import: both
bindings read it.
"""

from std.memory import bitcast
from x_linear.ops import FP

comptime XLIN_GLM_DEV_YDOM = True
"""Always on (no `_OFF` arm): kept as a name for its comptime callers."""
#: `res[d + 2]` of a fit refused for its targets' range
comptime GLM_YDOM_REFUSED = Float32(-1)


def glm_ydom_bad(power: Float32, has_neg: Bool, has_pos: Bool, has_zero: Bool) -> Bool:
    """True when the targets are outside the loss's range (see the header)."""
    if power >= Float32(2):
        return has_neg or has_zero
    if power >= Float32(1):
        return has_neg or not has_pos
    return False


def glm_ydom_host(y: FP, n: Int, power: Float32) -> Bool:
    """The CPU column's check over y[0:n] (a CPU-only install; the device
    route runs `glm_ydom_kernel`)."""
    var has_neg = False
    var has_pos = False
    var has_zero = False
    for i in range(n):
        var b = bitcast[DType.uint32](y.unsafe_load(i))
        if (b & UInt32(0x7FFFFFFF)) == UInt32(0):
            has_zero = True
        elif (b >> 31) != UInt32(0):
            has_neg = True
        else:
            has_pos = True
    return glm_ydom_bad(power, has_neg, has_pos, has_zero)
