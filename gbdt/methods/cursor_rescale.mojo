# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learn cursor's leaf update, one statement for the device kernel
(`kernel_add_model_value.mojo`) and the host oracles that restate it
(`gbdt/host/gbdt_oracle_*`; their per-row add is
`gbdt/host/gbdt_oracle.mojo::apply_scaled_leaves_host`): `cursor + round(leaf * rate)`, the rescaled
leaf formed once per leaf. Host-safe: no GPU imports."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.soft_f64 import sf64_from_f32, sf64_mul, sf64_to_f32


@always_inline
def rescaled_leaf(raw: Float32, learning_rate: Float32) -> Float32:
    """THE CURSOR UPDATE SEAM (IDENTITY_PATHS row 9), formed once per leaf.

    Under IDENTICAL the learn cursor takes `cursor + round(leaf * rate)`:
    their `Rescale(step)` then add (`doc_parallel_boosting.h:388`), the value
    the model stores (the host rescale, DEVIATION 256) and predict adds, and
    the same statement as TREE_OP_APPLY's `leaf_scaled`
    (`_leaf_out_gated_kernel`: the correctly rounded product through soft
    float, so no backend's contraction or flush can reach it). It replaces
    `fma(leaf, rate, cursor)` (fb11, 2026-11-17): one rounding where the
    model and TREE_OP_APPLY take two, so the greedy searcher's learn cursor
    sat a last bit off the model it trained and off the pointwise searcher's
    (`fit_pointwise_check` W2, NVIDIA L40S: tree 0 identical, 13 of 20
    losses off in the 7th digit). Bits change on every column together: this
    kernel and the host oracles that restate it (`gbdt/host/gbdt_oracle_*`).
    Under FAST the plain product."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return sf64_to_f32(sf64_mul(sf64_from_f32(raw), sf64_from_f32(learning_rate)))
    return raw * learning_rate


def rescaled_leaves(leaves: List[Float32], learning_rate: Float32) -> List[Float32]:
    """`rescaled_leaf` over a tree's leaf values (leaves x dims), once per
    tree: the values the stored model keeps and the cursor adds."""
    var out = List[Float32](capacity=len(leaves))
    for i in range(len(leaves)):  # driver-loop: over one tree's leaf values
        out.append(rescaled_leaf(leaves[i], learning_rate))
    return out^
