# SPDX-License-Identifier: Apache-2.0
"""Exact host binary-ranking gate, including signed-zero score ties.

    pixi run mojo run -I . metrics/checks/host_binary_ranking_check.mojo
"""
from std.memory import bitcast

from metrics.host.classification_oracle import host_binary_ranking


def _same(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def main() raises:
    var y: List[Int32] = [0, 1, 1, 0]
    var scores: List[Float32] = [
        bitcast[DType.float32](UInt32(0x80000000)), Float32(0.0),
        Float32(2.0), Float32(2.0),
    ]
    var auc = host_binary_ranking(y, scores, 4, False)
    if auc[1] != 2 or not _same(auc[0][0], Float32(0.5)):
        raise Error("host binary-ranking AUC moved")
    var curve = host_binary_ranking(y, scores, 4, True)
    if curve[1] != 2:
        raise Error("host binary-ranking tie grouping moved")
    var expected: List[Float32] = [
        Float32(0.5), Float32(0.5), Float32(1.0),
        Float32(1.0), Float32(0.5), Float32(0.0),
        Float32(0.0), Float32(2.0),
    ]
    var positions: List[Int] = [0, 1, 2, 5, 6, 7, 10, 11]
    for i in range(len(expected)):
        if not _same(curve[0][positions[i]], expected[i]):
            raise Error("host binary-ranking curve moved at checked position")
    print("PASS host binary ranking: AUC, tie groups, curve and signed zero")
