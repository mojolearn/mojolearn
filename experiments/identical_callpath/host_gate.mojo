# SPDX-License-Identifier: Apache-2.0
"""Untimed host-column check against frozen NVIDIA/AMD fixture digests.

Uses shipped host implementations, never device adapters or reconstructed
algorithm arithmetic. Run only as host identity verification on Linux boxes.
The scalar fixtures/order match identity_gate.mojo and primitive_gate.mojo.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from preprocessing.host.scaler_oracle import host_minmax_transform, host_standard_transform
from core.knn_host_predict import host_row_norms
from decomposition.host.pca_oracle import host_column_mean, host_shift_columns
from core.classical_host_predict import host_gemm_nt
from gemm.host.identical_gemm import gemm_oracle, OP_NT


def absorb(values: List[Float32], mut digest: UInt64):
    for i in range(len(values)):
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def scaler_gate() raises:
    var rows = 257
    var cols = 3
    var scales: List[Float32] = [1.0, 0.5, 2.0]
    var offsets: List[Float32] = [0.0, -1.0, 1.0]
    var inputs = List[List[Float32]]()
    for slot in range(2):
        var values = List[Float32](length=rows * cols, fill=0.0)
        for i in range(rows * cols):
            values[i] = Float32((i + slot * 11) % 37 - 18) * 0.125
        values[0] = bitcast[DType.float32](UInt32(0x80000000))
        values[1] = bitcast[DType.float32](UInt32(1))
        values[2] = bitcast[DType.float32](UInt32(0x80000001))
        inputs.append(values^)
    var digest = UInt64(14695981039346656037)
    var comparisons = 0
    # Preserve GPU gate digest order, including its two independent wait modes.
    for _ in range(2):
        for inverse in range(2):
            for first in range(2):
                for slot in range(2):
                    var result = host_minmax_transform(inputs[slot], scales, offsets,
                        rows, cols, inverse, first, -1.0, 1.0)
                    absorb(result, digest)
                    comparisons += 1
                for second in range(2):
                    for slot in range(2):
                        var result = host_standard_transform(inputs[slot], offsets, scales,
                            rows, cols, inverse, first, second)
                        absorb(result, digest)
                        comparisons += 1
    if comparisons != 48 or digest != UInt64(138198733663681093):
        print("CALLPATH_HOST_SCALERS status=DIFFER comparisons=", comparisons, " digest=", digest)
        raise Error("host scalers differ from frozen436bc1aaa GPU digests")
    print("CALLPATH_HOST_SCALERS status=PASS comparisons=", comparisons, " digest=", digest)


def primitive_gate() raises:
    var rows = 257
    var matrix = List[Float32](length=rows * 3, fill=0.0)
    var right: List[Float32] = [1.0, 1.0, 1.0, 2.0, -1.0, 1.0, -1.0, 0.0, 2.0]
    var digest = UInt64(14695981039346656037)
    var comparisons = 0
    for turn in range(2):
        for row in range(rows):
            for col in range(3):
                matrix[row * 3 + col] = Float32(col + 1 + 3 * turn)
        var norms = host_row_norms(matrix, rows, 3)
        var means = host_column_mean(matrix, rows, 3)
        var core_product = host_gemm_nt(matrix, right, rows, 3, 3)
        var identical_product = gemm_oracle(matrix, right, OP_NT, rows, 3, 3)
        absorb(norms, digest)
        absorb(means, digest)
        absorb(core_product, digest)
        absorb(identical_product, digest)
        var centered = host_shift_columns(matrix, means, rows, 3, -1.0)
        absorb(centered, digest)
        var restored = host_shift_columns(centered, means, rows, 3, 1.0)
        absorb(restored, digest)
        comparisons += 6
    if comparisons != 12 or digest != UInt64(12251619760217914789):
        print("CALLPATH_HOST_PRIMITIVES status=DIFFER comparisons=", comparisons, " digest=", digest)
        raise Error("host primitives differ from frozen436bc1aaa GPU digests")
    print("CALLPATH_HOST_PRIMITIVES status=PASS comparisons=", comparisons, " adapters=5 digest=", digest)


def main() raises:
    comptime assert is_defined["MOJOLEARN_COLUMN_CPU"](), "host gate requires the actual CPU column"
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "host gate requires IDENTICAL mode"
    scaler_gate()
    primitive_gate()
