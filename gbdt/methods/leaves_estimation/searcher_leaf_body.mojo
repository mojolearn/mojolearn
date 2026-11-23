# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The greedy structure searcher's own leaf value, written once (AGENTS.md
"One body per operation").

GPU-free: no `max.gpu`, no `DeviceContext`, no kernel. The device's
`compute_leaf_values_kernel` (`leaves_estimation.mojo`, one thread a leaf)
and the CPU column's searcher leaves (`gbdt/host/gbdt_oracle.mojo`
`_rmse_leaf_value`, which `gbdt_host_boost` and `gbdt_losses_host_fit` take
under Simple and under DEVIATION 64) call `searcher_leaf_value` with the
same final-partition sums, so the two columns write the same leaf with the
same statements.

Lane gbdt-cpu-rest-4-simple (2026-11-23): before this module the kernel held
the statements inline and the host restated them term for term; the
statements are unchanged, so no bit is expected to move on either column.
"""

#: `TLeavesEstimationConfig::MinLeafWeight` (`leaves_estimation_config.h:11`,
#: the literal `1e-20` on every path, `:60`).
comptime SEARCHER_MIN_LEAF_WEIGHT = Float32(1e-20)


@always_inline
def searcher_leaf_value(w: Float32, g: Float32, l2: Float32) -> Float32:
    """One leaf from its sums: `w` the weight plane (stat 0), `g` the gradient
    plane (stat 1), `l2` the regularizer.

    `(*Der2AtPoint)[i] += lambda` (`pointwise_oracle.cpp:86-89`): the
    regularizer is folded into the Hessian, and the guard is on that Hessian
    (`descent_helpers.cpp:87`); the step is `g / (hessian + 1e-20)`, `+g`
    because their `der` is already the negative gradient; `RegularizeImpl`
    (`oracle_interface.h:43-53`) then zeroes an underweight leaf, reading the
    WEIGHT, not the Hessian. See `compute_leaf_values_kernel` for the account
    of each statement."""
    var hessian = w + l2
    if hessian <= Float32(0.0):
        return Float32(0.0)
    var v = g / (hessian + Float32(1e-20))
    if w < SEARCHER_MIN_LEAF_WEIGHT:
        v = Float32(0.0)
    return v
