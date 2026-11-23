# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One step of the non-symmetric tree walk (`ComputeNonSymmetricDecision
TreeBinsImpl`, `add_model_value.cu:353-395`), the ONE body every column calls
(lane gbdt-cpu-rest-4-cat-ns, 2026-11-23): the device kernel
`compute_non_symmetric_decision_tree_bins_kernel`
(`gbdt/models/kernel/add_bin_values.mojo`), the host fit's training and
held-out bins (`gbdt/host/gbdt_oracle_depthwise.mojo::_ns_bins_range`) and the
host predict (`core/gbdt_host_predict.mojo::_apply_non_symmetric_block`).

GPU-FREE: integers only, no launch, no allocation, no import. The tree is the
flat pre-order `TTreeNode` array whose `left_subtree` / `right_subtree` are
LEAF counts; `bin` accumulates the leaves to the left of the row's leaf.
"""


@always_inline
def ns_node_goes_right(feature_val: UInt32, node_bin: UInt32, take_bin: Bool) -> Bool:
    """The node's predicate on the row's (shifted, masked) feature value:
    `==` for a TakeBin (one-hot) node, `>` for a TakeGreater one."""
    if take_bin:
        return feature_val == node_bin
    return feature_val > node_bin


@always_inline
def ns_walk_step(
    goes_right: Bool,
    left_subtree: Int,
    right_subtree: Int,
    mut bin: Int,
    mut node: Int,
) -> Bool:
    """Advances (`bin`, `node`) by one node and returns `stop`:

        right: bin += left_subtree; stop = right_subtree == 1;
               else node += left_subtree
        left:  stop = left_subtree == 1; else node += 1
    """
    if goes_right:
        bin += left_subtree
        if right_subtree == 1:
            return True
        node += left_subtree
        return False
    if left_subtree == 1:
        return True
    node += 1
    return False
