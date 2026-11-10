# SPDX-License-Identifier: Apache-2.0
"""Typed schedule correctness with independent closed forms and literal words.

Schedules produce host scalar metadata for CPU and GPU optimizers alike, so
this is one host check. It is not the precision-seam identity verifier; that
existing registration and Decimal oracle remain untouched. Independent
fixtures and scalar steps use the host pool; product table calls keep their
existing bounded run implementation.
"""
from std.math import cos, pow
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from training.optimizer_public_common import PublicStep, step_lr
from sequence.lr_table import LrSpec, lr_spec_fill, lr_spec_value, LR_ONE_BASED, LR_BEYOND, LR_EXACT_ONLY

comptime LR_FIXTURES = 25
comptime SENTINEL = Float32(31415.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def schedule_spec(family: Int, exact: Bool = False) -> LrSpec:
    var spec = LrSpec(20, 7, 0, 0, 0.1, 0.5, 0, 0)
    if family == 1:
        spec = LrSpec(20, 1, 0, 0, 0.05, 0.9, 0, 0)
    elif family == 2:
        spec = LrSpec(10, 0, 0, 40, 0.2, 25, 1e4, 0.3)
    if exact:
        spec.kind += LR_EXACT_ONLY
    return spec


def closed_form(family: Int, step: Int) -> Float64:
    var epoch = step - 1
    if family == 0:
        return 0.1 * pow(Float64(0.5), Float64(epoch // 7))
    if family == 1:
        return 0.05 * pow(Float64(0.9), Float64(epoch))
    # Independent restatement of the original Python torch-rule oracle.
    var initial = Float64(0.2) / 25.0
    var low = initial / 1e4
    var phase_end = Float64(0.3) * 40.0 - 1.0
    var start_value = initial
    var end_value = Float64(0.2)
    var start = Float64(0)
    var end = phase_end
    if Float64(epoch) > phase_end:
        start_value = 0.2
        end_value = low
        start = phase_end
        end = 39.0
    var progress = (Float64(epoch) - start) / (end - start)
    return end_value + (start_value - end_value) * 0.5 * (cos(3.141592653589793 * progress) + 1.0)


def near_reference(actual: Float32, family: Int, step: Int) raises:
    var expected = closed_form(family, step)
    var tolerance = Float64(1e-6) if family == 2 else Float64(1e-7)
    # The original math.isclose relative bounds, no new absolute tolerance.
    require(abs(Float64(actual) - expected) <= tolerance * max(abs(Float64(actual)), abs(expected)),
        "independent LR closed form family " + String(family) + " step " + String(step))


def numerical_fixture(fixture_id: Int) raises:
    var family = fixture_id // 6
    var exact = (fixture_id // 3) % 2 != 0
    var form = fixture_id % 3
    var spec = schedule_spec(family, exact)
    var first_step = 9 if form == 2 else 1
    var count = 21 if form == 2 else 40
    var values = List[Float32](length=count + 2, fill=SENTINEL)
    var vp = values.unsafe_ptr()
    if form == 0:
        var failures = List[Int](length=count, fill=0)
        var fp = failures.unsafe_ptr()
        def step_task(index: Int) {imm vp, imm fp, imm spec, imm first_step}:
            try:
                var result = lr_spec_value(spec, first_step + index)
                if result.status != 0:
                    fp.unsafe_store(index, 1)
                else:
                    vp.unsafe_store(index + 1, result.value)
            except error:
                print("FAIL scalar schedule step", first_step + index, String(error))
                fp.unsafe_store(index, 1)
        host_parallelize(step_task, count)
        for failure in failures:
            require(failure == 0, "native scalar schedule refused")
    else:
        var result = lr_spec_fill(spec, first_step, count, f32_ptr(Int(values.unsafe_ptr()) + 4))
        require(result[0] == 0 and result[1] == count, "native schedule fill count/status")
    for index in range(count):
        near_reference(values[index + 1], family, first_step + index)
    require(values[0] == SENTINEL and values[count + 1] == SENTINEL, "schedule fill overwrote guards")


def boundary_fixture(fixture_id: Int) raises:
    var values = List[Float32](length=6, fill=SENTINEL)
    var dst = f32_ptr(Int(values.unsafe_ptr()) + 4)
    if fixture_id == 18:
        var result = lr_spec_fill(schedule_spec(0), 0, 4, dst)
        require(result[0] == LR_ONE_BASED and result[1] == 0, "schedule one-based status")
        for value in values:
            require(value == SENTINEL, "refused start wrote output")
    elif fixture_id == 19:
        var result = lr_spec_fill(schedule_spec(0), -1, 0, dst)
        require(result[0] == 0 and result[1] == 0, "empty schedule fill contract")
        for value in values:
            require(value == SENTINEL, "empty schedule wrote output")
    else:
        var result = lr_spec_fill(schedule_spec(2), 40, 4, dst)
        require(result[0] == LR_BEYOND and result[1] == 2, "OneCycle first failing index")
        near_reference(values[1], 2, 40)
        near_reference(values[2], 2, 41)
        require(values[0] == SENTINEL and values[3] == SENTINEL and values[4] == SENTINEL and values[5] == SENTINEL, "OneCycle refusal overwrote tail/guards")
    _ = values


def literal_fixture(fixture_id: Int) raises:
    var family = fixture_id - 21
    var spec = LrSpec(family, 2, -1 if family == 0 else 6, 0, 0.5, 0.125, 0, 0)
    var steps: List[Int] = [1, 2, 3, 4, 5, 6, 7, 10]
    var expected: List[Float32] = [0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    if family == 1:
        expected = [0.25, 0.5, 0.40625, 0.3125, 0.21875, 0.125, 0.125, 0.125]
    elif family == 2:
        # Cosine endpoints/midpoint are exact dyadic words; no shared cosine
        # helper is used to compute these expected values.
        steps = [1, 2, 4, 6, 10]
        expected = [0.25, 0.5, 0.3125, 0.125, 0.125]
    for index in range(len(steps)):
        var result = lr_spec_value(spec, steps[index])
        require(result.status == 0 and bitcast[DType.uint32](result.value) == bitcast[DType.uint32](expected[index]), "literal warmup/tail schedule")


def optimizer_rate_fixture() raises:
    var request = PublicStep()
    request.hyper.lr = 0.125
    request.t = 2
    require(step_lr(request) == 0.125, "unscheduled typed optimizer rate")
    request.has_sched = True
    request.sched = schedule_spec(1)
    near_reference(Float32(step_lr(request)), 1, 3)
    request.t = -1
    var refused = False
    try:
        _ = step_lr(request)
    except:
        refused = True
    require(refused, "typed optimizer accepted non-one-based schedule step")
    request.sched = schedule_spec(2)
    request.t = 41
    refused = False
    try:
        _ = step_lr(request)
    except:
        refused = True
    require(refused, "typed optimizer accepted exhausted OneCycle schedule")


def main() raises:
    var failures = List[Int](length=LR_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            if fixture_id < 18:
                numerical_fixture(fixture_id)
            elif fixture_id < 21:
                boundary_fixture(fixture_id)
            elif fixture_id < 24:
                literal_fixture(fixture_id)
            else:
                optimizer_rate_fixture()
        except error:
            print("FAIL lr_schedule fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, LR_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "lr_schedule failed " + String(failed) + " fixtures")
    print("PASS lr_schedule:", LR_FIXTURES, "parallel native fixtures; host scalar metadata for CPU/GPU")
