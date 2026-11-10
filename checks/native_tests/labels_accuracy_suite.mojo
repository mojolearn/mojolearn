# SPDX-License-Identifier: Apache-2.0
"""Known score results through the real native COUNT planner and host kernels."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.labels_accuracy_plan import label_accuracy_plan
from x_metrics.host.program import run_program_host


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _score(unit: Int) raises:
    # Neighboring fold boundaries, odd final leaves, and several fold levels.
    var sizes: List[Int] = [4, 255, 256, 257, 513, 1025]
    var rows = sizes[unit % 6]
    var variant = unit // 6
    var weighted = variant >= 3
    var plan = label_accuracy_plan(rows, 7, weighted)
    var arena = List[UInt32](length=plan.program.size, fill=0)
    var words = arena.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var actual = plan.program.input_offsets[0]
    var predicted = plan.program.input_offsets[1]
    var weight_at = plan.program.input_offsets[2] if weighted else 0
    var hits = 0
    var total_weight = 0
    var hit_weight = 0
    # Each fixture owns its arena. Rows are small; fixtures parallelize the
    # independent inputs, and the production runner owns stage parallelism.
    for i in range(rows):
        var code = (i * 3 + 2) % 7
        var match = (i % 4) < (variant % 3 + 1)
        words[actual + i] = UInt32(code)
        # Out-of-range codes are misses, even when both vectors contain -1.
        words[predicted + i] = UInt32(code if match else -1)
        hits += Int(match)
        if weighted:
            var weight = 1 + i % 4
            words[weight_at + i] = bitcast[DType.uint32](Float32(weight))
            total_weight += weight
            if match:
                hit_weight += weight
    run_program_host(Int(words), plan.program.size, Int(plan.program.stages.unsafe_ptr()),
                     len(plan.program.stages) // 16)
    var at = plan.result_offset
    var bits = UInt64(words[at + 8]) | (UInt64(words[at + 9]) << 32)
    var expected = Float64(hit_weight) / Float64(total_weight) if weighted else Float64(hits) / Float64(rows)
    _require(bits == bitcast[DType.uint64](expected), "independent matched fraction")
    _require(words[at] == 0, "nonzero denominator")
    _ = plan^
    _ = arena^


def _edge(unit: Int) raises:
    if unit == 0:
        var message = String()
        try:
            _ = label_accuracy_plan(2147483647, 2, False)
        except problem:
            message = String(problem)
        _require(message == "mojolearn metrics: the problem exceeds the Int32 arena bound", "arena overflow refusal")
        return
    var plan = label_accuracy_plan(4, 2, unit == 1)
    var arena = List[UInt32](length=plan.program.size, fill=0)
    var words = arena.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var actual = plan.program.input_offsets[0]
    var predicted = plan.program.input_offsets[1]
    for i in range(4):
        words[actual + i] = UInt32(-1) if unit == 2 else UInt32(i % 2)
        words[predicted + i] = words[actual + i]
    run_program_host(Int(words), plan.program.size, Int(plan.program.stages.unsafe_ptr()),
                     len(plan.program.stages) // 16)
    var at = plan.result_offset
    if unit == 1:
        _require(words[at] != 0, "zero weights set denominator status")
    else:
        var bits = UInt64(words[at + 8]) | (UInt64(words[at + 9]) << 32)
        _require(bits == (UInt64(0) if unit == 2 else UInt64(0x3FF0000000000000)), "invalid codes miss; all valid codes match")
    _ = plan^
    _ = arena^


def run_checks() raises:
    comptime FIXTURES = 40
    var failures = List[Int](length=FIXTURES, fill=0)
    var failed_at = failures.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    @parameter
    def task(unit: Int):
        try:
            if unit < 36:
                _score(unit)
            else:
                _edge(unit - 36)
        except problem:
            print("FAIL labels_accuracy fixture", unit, String(problem))
            failed_at[unit] = 1
    host_parallelize(task, FIXTURES)
    var count = 0
    for value in failures:
        count += value
    _require(count == 0, "native label accuracy failed " + String(count) + " of " + String(FIXTURES))
    print("PASS labels_accuracy:", FIXTURES, "native fixtures, parallel")
