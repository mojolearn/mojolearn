# SPDX-License-Identifier: Apache-2.0
"""Typed class-weight validation and existing Float64-to-Float32 publication."""
from std.math import isfinite
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def class_weight_kind(kind: Int) raises:
    if kind != 3 and kind != 4:
        raise Error("class weights must be finite nonnegative real numbers")


def class_weight_value(value: Float64) raises:
    if not isfinite(value) or value < 0:
        raise Error("class weights must be finite nonnegative real numbers")


struct ClassWeightInputs(Movable):
    var values: List[Float64]
    var all_one: Bool

    def __init__(out self):
        self.values = List[Float64]()
        self.all_one = True

    def append(mut self, value: Float64, exact_one: Bool) raises:
        # UNPACK calls this after each exact scalar conversion. Checking here
        # preserves the earlier invalid-value error over a later conversion
        # error. No Python object or scalar comparison enters this native state.
        class_weight_value(value)
        self.append_validated(value, exact_one)

    def append_validated(mut self, value: Float64, exact_one: Bool):
        self.values.append(value)
        self.all_one = self.all_one and exact_one


struct ClassWeightPlan(Movable):
    var all_one: Bool
    var words: List[Float32]

    def __init__(out self, all_one: Bool, var words: List[Float32]):
        self.all_one = all_one
        self.words = words^


def class_weight_all_one(values: List[Float64]) raises -> Bool:
    var count = len(values)
    var tasks = host_predict_task_count_work(max(1, count), max(1, count) * 3)
    var chunk = host_predict_chunk(max(1, count), tasks)
    var bad = List[Int](length=tasks, fill=0)
    var units = List[Int](length=tasks, fill=1)
    var source = values.unsafe_ptr()
    var bp = bad.unsafe_ptr()
    var up = units.unsafe_ptr()
    def task(worker: Int) {imm source, imm bp, imm up, imm count, imm chunk}:
        var valid = True
        var all_one = True
        for i in range(worker * chunk, min((worker + 1) * chunk, count)):
            var value = source[i]
            valid = valid and isfinite(value) and value >= 0
            all_one = all_one and value == 1
        bp[worker] = Int(not valid)
        up[worker] = Int(all_one)
    host_parallelize(task, tasks)
    var all_one = True
    for worker in range(tasks):  # fixed worker summaries after the join
        if bad[worker]:
            raise Error("class weights must be finite nonnegative real numbers")
        all_one = all_one and units[worker] != 0
    _ = values
    return all_one


def class_weight_plan(values: List[Float64], all_one: Bool) raises -> ClassWeightPlan:
    # Exact source equality was captured at admission. Extended precision
    # 1+epsilon may round to Float64/Float32 1 while still requiring row weights.
    if all_one:
        return ClassWeightPlan(True, List[Float32]())
    var count = len(values)
    var words = List[Float32](unsafe_uninit_length=count)
    var tasks = host_predict_task_count_work(max(1, count), max(1, count) * 4)
    var chunk = host_predict_chunk(max(1, count), tasks)
    var bad = List[Int](length=tasks, fill=0)
    var positive = List[Int](length=tasks, fill=0)
    var source = values.unsafe_ptr()
    var dst = words.unsafe_ptr()
    var bp = bad.unsafe_ptr()
    var pp = positive.unsafe_ptr()
    def task(worker: Int) {imm source, imm dst, imm bp, imm pp, imm count, imm chunk}:
        var failed = False
        var has_positive = False
        for i in range(worker * chunk, min((worker + 1) * chunk, count)):
            var value = source[i]
            var narrowed = Float32(value)
            failed = failed or not isfinite(narrowed) or (value > 0 and narrowed == 0)
            dst[i] = narrowed
            has_positive = has_positive or narrowed > 0
        bp[worker] = Int(failed)
        pp[worker] = Int(has_positive)
    host_parallelize(task, tasks)
    var has_positive = False
    for worker in range(tasks):  # worker summaries, not a serial data pass
        if bad[worker]:
            raise Error("class weights must remain finite and nonzero when positive in Float32")
        has_positive = has_positive or positive[worker] != 0
    if not has_positive:
        raise Error("class weights must have positive total")
    _ = values
    return ClassWeightPlan(False, words^)
