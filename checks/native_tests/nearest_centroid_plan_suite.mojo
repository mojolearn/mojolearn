# SPDX-License-Identifier: Apache-2.0
"""Independent prior/offset oracles for typed NearestCentroid preparation."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.nearest_centroid_plan import nc_uniform, nc_narrow_priors, nc_normalize_priors, nc_class_offsets, nc_power_steps

comptime NC_PLAN_FIXTURES = 20


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def offsets_case(unit: Int) raises:
    var sizes: List[Int] = [0, 1, 2, 3, 257, 65537, 131073, 262271]
    var count = sizes[unit]
    var source = List[Int64](unsafe_uninit_length=count)
    var sp = source.unsafe_ptr()
    var tasks = host_predict_task_count_work(max(1, count), max(1, count))
    var chunk = host_predict_chunk(count, tasks)
    def fill(worker: Int) {imm sp, imm count, imm chunk}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            sp[i] = Int64(i % 3)
    host_parallelize(fill, tasks)
    var result = nc_class_offsets(Int(source.unsafe_ptr()), count)
    require(len(result) == count + 1 and result[0] == 0, "prefix extent/origin")
    var rp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(result.unsafe_ptr()))
    var flags = List[Int](length=tasks, fill=0)
    var fp = flags.unsafe_ptr()
    def check(worker: Int) {imm rp, imm count, imm chunk, imm fp}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            var n = i + 1
            var expected = 3 * (n // 3) + (1 if n % 3 == 2 else 0)
            if Int(rp[n]) != expected:
                fp[worker] = 1
    host_parallelize(check, tasks)
    for flag in flags:  # metadata: worker summaries
        require(flag == 0, "integer class prefix differs from closed form")
    _ = source^
    _ = result^


def prior_case(unit: Int) raises:
    if unit < 4:
        var count = 1 if unit == 0 else (3 if unit == 1 else (65537 if unit == 2 else 262271))
        var values = List[Float64](length=count, fill=1.0 / Float64(count))
        require(nc_uniform(Int(values.unsafe_ptr()), count), "uniform prior refused")
        values[count - 1] = 2.0
        require(not nc_uniform(Int(values.unsafe_ptr()), count), "changed prior was cached")
        return
    if unit == 4:
        var values: List[Float64] = [2.0, 1.0, 1.0]
        nc_normalize_priors(Int(values.unsafe_ptr()), 3)
        require(values[0] == 0.5 and values[1] == 0.25 and values[2] == 0.25, "prior normalization")
    elif unit == 5:
        var values: List[Float64] = [0.5, 0.500001]
        nc_normalize_priors(Int(values.unsafe_ptr()), 2)
        require(values[0] == 0.5 and values[1] == 0.500001, "isclose must skip rescale")
    elif unit == 6:
        var values: List[Float64] = [0.25, -0.0, 1e-300, 2.0]
        var words = List[Float32](length=4, fill=0)
        nc_narrow_priors(Int(values.unsafe_ptr()), Int(words.unsafe_ptr()), 4)
        require(bitcast[DType.uint32](words[0]) == UInt32(0x3e800000), "quarter bits")
        require(bitcast[DType.uint32](words[1]) == UInt32(0x80000000), "negative zero bits")
        require(words[2] == 0 and words[3] == 2, "query prior narrowing")
    else:
        var steps = nc_power_steps(257)
        require(steps[0] == 512 and steps[1] == 45, "median network metadata")
        require(nc_uniform(0, 0), "empty prior uniformity")
        nc_narrow_priors(0, 0, 0)


def refusal_case(unit: Int) raises:
    var values: List[Float64] = [0.0, 0.0]
    var expected = String("ZeroDivisionError: float division by zero")
    if unit == 1:
        values[0] = -1.0
        values[1] = bitcast[DType.float64](UInt64(0x7ff0000000000000))
        expected = "priors must be non-negative"
    elif unit == 2:
        values[0] = 1e308
        values[1] = 1e308
        expected = "OverflowError: math range error"
    var refused = False
    try:
        nc_normalize_priors(0 if unit == 3 else Int(values.unsafe_ptr()), 0 if unit == 3 else 2)
    except error:
        refused = String(error) == expected
    require(refused, "prior refusal order/message changed")


def run_checks() raises:
    var failures = List[Int](length=NC_PLAN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            if unit < 8:
                offsets_case(unit)
            elif unit < 16:
                prior_case(unit - 8)
            else:
                refusal_case(unit - 16)
        except error:
            fp[unit] = 1
            print("FAIL nearest_centroid_plan fixture", unit, String(error))
    host_parallelize(run, NC_PLAN_FIXTURES)
    var total = 0
    for failed in failures:  # fixture summaries
        total += failed
    require(total == 0, "nearest_centroid_plan failed " + String(total) + " fixtures")
    print("PASS nearest_centroid_plan:", NC_PLAN_FIXTURES, "native prior/offset fixtures")
