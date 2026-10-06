# SPDX-License-Identifier: Apache-2.0
"""T15 host counterpart of requested native RF impurity importance.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Same ascending-node, feature, tree and feature folds as the incumbent native
compute_feature_importances_host; explicit IEEE64 operations match the device.
"""
from ensemble.host.rf_oracle import RfHostForest
from checks.soft_f64 import (SF64_INF, SF64_ONE, sf64_add, sf64_div,
    sf64_from_f32, sf64_from_int, sf64_gt, sf64_mul, sf64_to_f32)


def rf_host_importance(forest: RfHostForest, features: Int, output_addr: Int) raises:
    if features < 1 or output_addr == 0:
        raise Error("RF feature importance requires positive feature count and an output buffer")
    var accumulated = List[UInt64](length=features,fill=UInt64(0))
    for t in range(forest.n_trees):
        var lo = Int(forest.offsets[t])
        var hi = Int(forest.offsets[t+1])
        if lo == hi or forest.counts[lo] <= 0:
            continue
        var finite = List[UInt64](length=features,fill=UInt64(0))
        var infinite = List[UInt64](length=features,fill=UInt64(0))
        var has_infinite = False
        for j in range(lo,hi):
            if forest.left_child[j] == -1:
                continue
            var c = Int(forest.colid[j])
            var contribution = sf64_mul(sf64_from_f32(forest.metrics[j]),sf64_from_int(Int(forest.counts[j])))
            if contribution == SF64_INF:
                infinite[c] = sf64_add(infinite[c],SF64_ONE)
                has_infinite = True
            elif (contribution & SF64_INF) != SF64_INF and sf64_gt(contribution,UInt64(0)):
                finite[c] = sf64_add(finite[c],contribution)
        var total = UInt64(0)
        for c in range(features):
            total = sf64_add(total,infinite[c] if has_infinite else finite[c])
        if sf64_gt(total,UInt64(0)):
            for c in range(features):
                accumulated[c] = sf64_add(accumulated[c],sf64_div(infinite[c] if has_infinite else finite[c],total))
    var total = UInt64(0)
    for c in range(features):
        total = sf64_add(total,accumulated[c])
    var output = MutPointer[Float32,MutUntrackedOrigin](unsafe_from_address=output_addr)
    for c in range(features):
        output[c] = sf64_to_f32(sf64_div(accumulated[c],total)) if sf64_gt(total,UInt64(0)) else Float32(0)
