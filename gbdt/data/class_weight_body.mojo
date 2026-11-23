# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The class-weight fold of one row, written ONCE (AGENTS.md "One body per
operation"; lane gbdt-ordered-weights, 2026-11-23).

Their `MakeClassificationWeights` (`private/libs/target/data_providers.cpp:
168`): `rawWeights[i] * classWeights[targetClass[i]]`, the class the dense
code for the multiclass family and the binarized target (`y > 0.5`)
otherwise. The device's `class_weight_fold_kernel`
(`gbdt/data/target_prep.mojo`) and the host's row-weight column
(`gbdt/host/gbdt_host_entries_impl.mojo::_entry_row_weights_range`, which
the Plain and Ordered host fits read) call these two bodies, so the class
and the single product are the same statements on every column.

GPU-free: no `max.gpu` import, so the host oracles may import it.
"""


@always_inline
def class_weight_slot(v: Float32, multiclass: Bool) -> Int:
    """The class-weight index of a target value: the dense class code for
    the multiclass family, `1 if v > 0.5 else 0` otherwise."""
    if multiclass:
        return Int(Int32(v))
    return 1 if v > Float32(0.5) else 0


@always_inline
def class_weighted(weight: Float32, class_weight: Float32) -> Float32:
    """The row's weight times its class weight: ONE product, as their
    `rawWeights[i] * classWeights[...]` (the group weight is 1 here)."""
    return weight * class_weight
