# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Direct derivative/metric oracles, before a tree sees the gradients. Published LightGBM
v4.6 equations evaluated in FP64 with Random::NextFloat, default sigmoid table and gain.
The FP32 implementation shares no code with the calculation of the expected constants."""
from std.sys.compile import is_defined
from std.math import isfinite
from std.testing import TestSuite
from max.gpu.host import DeviceContext
from checks.native_tests.kit import upload, download
from checks.native_tests.testsuite_cases import require
from gbdt.host.lightgbm_ranking import HostLightGBMRank
from gbdt.targets.kernel.lightgbm_ranking import LightGBMRankWorkspace, launch_lightgbm_ndcg, launch_lightgbm_ranking
from gbdt.targets.ranking_options import OBJECTIVE_LAMBDARANK, OBJECTIVE_RANK_XENDCG


def _evaluate(objective: Int, groups: List[Int], labels: List[Float32], scores: List[Float32],
              weights: List[Float32], iteration: Int, cutoff: Int = 30) raises -> Tuple[List[Float32], List[Float32], Float32]:
    var n = len(labels)
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        var ws = HostLightGBMRank(groups, n)
        var d = List[Float32]()
        var h = List[Float32]()
        ws.derivatives(objective, labels, scores, weights, 5, iteration, d, h)
        return (d^, h^, ws.ndcg(labels, scores, cutoff))
    else:
        var ctx = DeviceContext()
        var ws = LightGBMRankWorkspace(ctx, n, len(groups))
        var offsets: List[UInt32] = [0]
        var qids = List[UInt32]()
        var sizes = List[UInt32]()
        for q in range(len(groups)):  # small-loop: query metadata of the fixture
            sizes.append(UInt32(groups[q]))
            offsets.append(offsets[q] + sizes[q])
            qids.extend(List[UInt32](length=groups[q], fill=UInt32(q)))
        var dl = upload[DType.float32](ctx, labels)
        var ds = upload[DType.float32](ctx, scores)
        var dw = upload[DType.float32](ctx, weights if len(weights) > 0 else [Float32(1)])
        var doff = upload[DType.uint32](ctx, offsets)
        var dsize = upload[DType.uint32](ctx, sizes)
        var dq = upload[DType.uint32](ctx, qids)
        var d = ctx.enqueue_create_buffer[DType.float32](n)
        var h = ctx.enqueue_create_buffer[DType.float32](n)
        var m = ctx.enqueue_create_buffer[DType.float32](1)
        launch_lightgbm_ranking(ctx, ws, objective, dl, ds, dw, len(weights) > 0, doff, dsize, dq,
                               n, len(groups), 5, iteration, d, h)
        launch_lightgbm_ndcg(ctx, ws, dl, ds, doff, dsize, dq, n, len(groups), m, cutoff)
        var metric = download[DType.float32](ctx, m, 1)
        return (download[DType.float32](ctx, d, n), download[DType.float32](ctx, h, n), metric[0])


def _close(got: List[Float32], expected: List[Float32]) raises:
    require(len(got) == len(expected), "ranking derivative count")
    for i in range(len(expected)):  # small-loop: at most six expected words
        require(abs(got[i] - expected[i]) < Float32(0.000025), "ranking oracle row " + String(i))


def test_grouped_derivatives_and_query_metric() raises:
    var groups: List[Int] = [3, 1, 2]
    var y: List[Float32] = [0, 2, 1, 4, 2, 0]
    var scores: List[Float32] = [0.7, -0.3, 0.1, 99, 0, 0]
    var w: List[Float32] = [1, 0.5, 2, 3, 1, 1]
    var lambda_ = _evaluate(OBJECTIVE_LAMBDARANK, groups, y, scores, w, 0)
    _close(lambda_[0], [-0.40393324, 0.20083525, 0.00452548, 0, 0.22659824, -0.22659824])
    _close(lambda_[1], [0.11776065, 0.06093609, 0.15971022, 0, 0.11329912, 0.11329912])
    require(abs(lambda_[2] - Float32(0.86229422)) < Float32(0.000002), "NDCG averages queries equally")
    var xe = _evaluate(OBJECTIVE_RANK_XENDCG, groups, y, scores, w, 0)
    _close(xe[0], [-0.27569832, 0.13922975, -0.00552236, 0, 0.39514680, -0.39514680])
    _close(xe[1], [0.24952770, 0.07754787, 0.40869271, 0, 0.25, 0.25])
    var next = _evaluate(OBJECTIVE_RANK_XENDCG, groups, y, scores, w, 1)
    _close(next[0], [-0.36807267, 0.20481625, -0.08311968, 0, 0.31828584, -0.31828584])


def test_perfect_ranking_has_exactly_unit_ndcg() raises:
    var result = _evaluate(OBJECTIVE_LAMBDARANK, [2], [2, 1], [2, 1], [], 0)
    require(result[2] == Float32(1), "a perfectly ordered query must have zero ranking loss")


def test_metric_cutoff_is_independent_of_training_cutoff() raises:
    # Reuse the ideal-gain cache initialized at the derivative's cutoff of 30.
    # One query ranks grade 1 over grade 2, one is perfect and an empty-gain query scores one.
    var top = _evaluate(OBJECTIVE_LAMBDARANK, [2, 2, 1], [1, 2, 2, 1, 0],
                        [0, 0, 0, 0, 0], [], 0, 1)
    require(abs(top[2] - Float32(7.0 / 9.0)) < Float32(1e-6), "NDCG@1 across distinct queries")
    var full = _evaluate(OBJECTIVE_LAMBDARANK, [2, 2, 1], [1, 2, 2, 1, 0],
                         [0, 0, 0, 0, 0], [], 0, 2)
    require(abs(full[2] - Float32(0.93223586)) < Float32(1e-6), "NDCG@2 includes the second grade")
    _close(top[0], full[0])
    _close(top[1], full[1])


def test_cutoff_and_stable_ties() raises:
    # The documents at ranks 30 and 31 interact with each top-30 document but never each other.
    var labels = List[Float32](length=32, fill=0)
    labels[30] = 1
    labels[31] = 2
    var result = _evaluate(OBJECTIVE_LAMBDARANK, [32], labels, List[Float32](length=32, fill=0), [], 0)
    _close([result[0][30], result[0][31]], [0.26893362, 0.82027589])
    _close([result[1][30], result[1][31]], [0.13446681, 0.41013794])
    require(result[2] == 0, "both relevant rows are below the NDCG cutoff")


def test_saturated_softmax_and_zero_relevance() raises:
    var xe = _evaluate(OBJECTIVE_RANK_XENDCG, [3], [0, 0, 0], [1000, 0, -1], [], 0)
    require(xe[2] == 1, "all-zero relevance has unit NDCG")
    for i in range(3):
        require(isfinite(xe[0][i]) and isfinite(xe[1][i]), "a saturated softmax must have finite derivatives")
    require(abs(xe[0][0] + xe[0][1] + xe[0][2]) < Float32(0.000005), "query derivatives sum to zero")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
