# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The device-rounds held-out loss word (TREE_OP_HELDOUT_LOSS, gbdt/targets/heldout_loss_bits.mojo) is the bits of
`_test_loss`'s host expression: `1 - double(fv)` for the LightGBM ranking objectives, `-double(fv) / test_weight`
for the others (0.9.0 integration review: the device op always divided, so a ranking arm would record -NDCG/n)."""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, require
from gbdt.targets.heldout_loss_bits import heldout_loss_bits

comptime GBDT_HELDOUT_LOSS_FIXTURES = 2


def fixture(unit: Int) raises:
    var values: List[Float32] = [0.0, 0.8125, 0.333333343, 1.0, 0.9999999, 1.5e-8, -3.25, 1234.5]
    var weights: List[Float64] = [1.0, 7.0, 3.0, 1000.0, 0.1]
    if unit == 0:
        # ranking: 1 - NDCG, whatever the row weight
        for i in range(len(values)):
            var want = bitcast[DType.uint64](Float64(1.0) - Float64(values[i]))
            for w in range(len(weights)):
                var got = heldout_loss_bits(values[i], bitcast[DType.uint64](weights[w]), True)
                require(got == want, "ranking held-out loss is 1 - NDCG, never divided by the weight")
    else:
        # pointwise: -functionValue / weight
        for i in range(len(values)):
            for w in range(len(weights)):
                var want = bitcast[DType.uint64](-Float64(values[i]) / weights[w])
                var got = heldout_loss_bits(values[i], bitcast[DType.uint64](weights[w]), False)
                require(got == want, "pointwise held-out loss is -fv / weight")


def test_gbdt_heldout_loss_bits() raises:
    check_cases("gbdt_heldout_loss_bits", GBDT_HELDOUT_LOSS_FIXTURES, fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
