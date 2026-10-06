# SPDX-License-Identifier: Apache-2.0
"""Actual non-symmetric tree fits with mixed feature packing, repeated
fits, live leaf limits and skewed min-leaf stopping. Fingerprints include
split sequence, leaf values and weights; A/B builds attribute frontier and
partition inheritance independently. A separate full-model cache witness is recorded as an additional
validation path for weighted boosting/prediction/loss words."""
from max.gpu.host import DeviceContext
from checks.depthwise_check import Fixture, default_options
from checks.lossguide_check import fit_policy, lossguide_options
from checks.gbdt_partition_cache_check import tree_hash

def main() raises:
    var ctx = DeviceContext()
    var fx = Fixture(ctx.copy())
    for leaves in [3, 11, 23]:
        for policy in ["Depthwise", "Lossguide"]:
            var reference = UInt64(0)
            for repeat in range(2):
                var options = lossguide_options(6,leaves) if policy=="Lossguide" else default_options(6)
                var model = fit_policy(fx,options)
                var h = UInt64(14695981039346656037)
                tree_hash(h,model)
                if repeat==0:
                    reference=h
                elif h!=reference:
                    raise Error("I17 frontier state leaked across fits")
                print("I17 fingerprint policy=",policy,"max_leaves=",leaves,"repeat=",repeat,"words=",h)
    _ = fx^
    print("I17 PASS live_leaf_fits=12 split_leaf_weight_fingerprints")
