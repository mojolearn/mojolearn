# SPDX-License-Identifier: Apache-2.0
"""The held-out loss word of one boosting round, the same quantity on the device and on the host.

`_test_loss` (gbdt/methods/doc_parallel_boosting.mojo) returns `1 - NDCG` for the LightGBM ranking objectives
(lambdarank, rank_xendcg) and `-functionValue / test_weight` for every other objective. The device-rounds op
TREE_OP_HELDOUT_LOSS writes the same value as a binary64 bit pattern through this helper, in soft float64
(checks/soft_f64.mojo: correctly rounded, so the bits are the host expression's on every vendor). Before the 0.9.0
review it always divided, which for a ranking arm recorded `-NDCG / n`."""
from checks.soft_f64 import SF64_ONE, sf64_div, sf64_from_f32, sf64_neg, sf64_sub


@always_inline
def heldout_loss_bits(fv: Float32, denominator_bits: UInt64, ranking: Bool) -> UInt64:
    """`1 - double(fv)` when `ranking`, else `-double(fv) / denominator` (denominator as binary64 bits)."""
    var value = sf64_from_f32(fv)
    if ranking:
        return sf64_sub(SF64_ONE, value)
    return sf64_div(sf64_neg(value), denominator_bits)
