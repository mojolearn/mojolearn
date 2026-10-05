# SPDX-License-Identifier: Apache-2.0
"""GPU primitive-adapter smoke gate with exactly representable fixtures.

This checks all five shared adapters and dirty-slot reuse. It is not a broad
numerical identity gate or a benchmark. Run only on authorized GPU boxes.
"""
from core.identical_callpath import IdenticalCallSession
from experiments.identical_callpath.families import (
    enqueue_row_norms, enqueue_column_means, enqueue_shift_columns,
    enqueue_core_gemm_nt, enqueue_identical_gemm,
)
from experiments.identical_callpath.identity_gate import compare
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats
from gemm.checks.gemm_oracle import OP_NT


def main() raises:
    var session = IdenticalCallSession()
    var rows = 257
    var left = session.reserve_f32(rows * 3)
    var right = session.reserve_f32(9)
    var core_output = session.reserve_f32(rows * 3)
    var identical_output = session.reserve_f32(rows * 3)
    var norm_output = session.reserve_f32(rows)
    var means_output = session.reserve_f32(3)
    var scratch = session.reserve_f32(identical_gemm_workspace_max_floats(rows, 3, 3))
    var matrix = List[Float32](length=rows * 3, fill=0.0)
    var right_matrix: List[Float32] = [1.0, 1.0, 1.0, 2.0, -1.0, 1.0, -1.0, 0.0, 2.0]
    var expected_norms = List[Float32](length=rows, fill=14.0)
    var expected_means: List[Float32] = [1.0, 2.0, 3.0]
    var expected_gram = List[Float32](length=rows * 3, fill=0.0)
    var expected_centered = List[Float32](length=rows * 3, fill=0.0)
    var norms = List[Float32](length=rows, fill=0.0)
    var means = List[Float32](length=3, fill=0.0)
    var gram = List[Float32](length=rows * 3, fill=0.0)
    var centered = List[Float32](length=rows * 3, fill=0.0)
    var digest = UInt64(14695981039346656037)
    var comparisons = 0
    for turn in range(2):
        # Different second-round data exposes stale uploads/readbacks and dirty
        # outputs. 257 rows and 771 cells exercise nonmultiple kernel tails.
        for col in range(3):
            expected_means[col] = Float32(col + 1 + 3 * turn)
        for row in range(rows):
            expected_norms[row] = 14.0 if turn == 0 else 77.0
            for col in range(3):
                matrix[row * 3 + col] = expected_means[col]
            expected_gram[row * 3] = 6.0 if turn == 0 else 15.0
            expected_gram[row * 3 + 1] = 3.0 if turn == 0 else 9.0
            expected_gram[row * 3 + 2] = 5.0 if turn == 0 else 8.0
        session.stage_f32(left, matrix)
        session.stage_f32(right, right_matrix)
        session.begin()
        session.upload_f32(left)
        session.upload_f32(right)
        enqueue_row_norms(session, norm_output, left, rows, 3)
        enqueue_column_means(session, means_output, left, rows, 3)
        enqueue_core_gemm_nt(session, core_output, left, right, rows, 3, 3)
        enqueue_identical_gemm(session, identical_output, left, right, scratch, rows, 3, 3, OP_NT)
        session.readback_f32(norm_output)
        session.readback_f32(means_output)
        session.readback_f32(core_output)
        session.readback_f32(identical_output)
        session.finish()
        session.collect_f32(norm_output, norms)
        compare(norms, expected_norms, digest)
        session.collect_f32(means_output, means)
        compare(means, expected_means, digest)
        session.collect_f32(core_output, gram)
        compare(gram, expected_gram, digest)
        session.collect_f32(identical_output, gram)
        compare(gram, expected_gram, digest)
        comparisons += 4
        session.begin()
        enqueue_shift_columns(session, left, means_output, rows, 3)
        session.readback_f32(left)
        session.finish()
        session.collect_f32(left, centered)
        compare(centered, expected_centered, digest)
        session.begin()
        enqueue_shift_columns(session, left, means_output, rows, 3, True)
        session.readback_f32(left)
        session.finish()
        session.collect_f32(left, centered)
        compare(centered, matrix, digest)
        comparisons += 2
    if comparisons != 12:
        raise Error("incomplete primitive coverage")
    print("CALLPATH_PRIMITIVES status=PASS comparisons=", comparisons,
          " adapters=5 reuse_rounds=2 digest=", digest)
