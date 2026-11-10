# SPDX-License-Identifier: Apache-2.0
"""Native accuracy COUNT stages; the existing metrics planner owns scheduling.

Stage order and arena offsets follow the previous score program exactly.
Building these dependent stage records is bounded by the fold depth, never
by label data; the existing kernels parallelize the row/group work.
"""
from core.metrics_program_plan import MetricProgramPlan


struct LabelAccuracyPlan(Movable):
    var program: MetricProgramPlan
    var result_offset: Int

    def __init__(out self, var program: MetricProgramPlan, result_offset: Int):
        self.program = program^
        self.result_offset = result_offset


def _group(mut program: MetricProgramPlan, key: Int, rows: Int, classes: Int,
           weights: Int) raises -> Tuple[Int, Int]:
    var off = program.alloc(classes + 1)
    var order = program.scratch(rows)
    var sums = program.alloc(classes)
    program.stage(0, 1, [key, rows, classes, off, order])
    if classes:
        program.stage(1, classes, [off, order, -1, 1, weights, sums, 1])
    return (off, sums)


def _counts(mut program: MetricProgramPlan, off: Int, classes: Int) raises:
    var counts = program.alloc(classes)
    if classes:
        program.stage(56, classes, [off, classes, counts])


def label_accuracy_plan(rows: Int, classes: Int, weighted: Bool) raises -> LabelAccuracyPlan:
    var program = MetricProgramPlan()
    var actual = program.input(rows)
    var predicted = program.input(rows)
    var matched = program.scratch(rows)
    program.stage(2, rows, [actual, predicted, matched, classes, 1])
    var weights = -1
    if weighted:
        weights = program.input(rows)
    var hits = _group(program, matched, rows, classes, weights)
    var truths = _group(program, actual, rows, classes, weights)
    var predictions = _group(program, predicted, rows, classes, weights)
    if weighted:
        var zero = program.scratch(rows)
        program.stage(2, rows, [actual, actual, zero, 1, 2])
        _ = _group(program, zero, rows, 1, weights)
    else:
        _counts(program, hits[0], classes)
        _counts(program, truths[0], classes)
        _counts(program, predictions[0], classes)
    var result_at = program.alloc(11)
    var true_positive = hits[1] if weighted else hits[0]
    var total = rows
    if weighted:
        var blocks = max(1, (rows + 255) // 256)
        var sums = program.scratch(4 * blocks)
        program.stage(59, blocks, [7, matched, weights, rows, sums, classes])
        var stride = 1
        while stride < blocks:  # dependent fold levels, each stage runs its independent units
            program.stage(59, (blocks - stride + 2 * stride - 1) // (2 * stride), [8, sums, stride, blocks])
            stride *= 2
        total = sums
    program.stage(59, 1, [2, true_positive, classes, 1 if weighted else 0, total, result_at])
    return LabelAccuracyPlan(program^, result_at)
