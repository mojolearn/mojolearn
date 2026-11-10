# SPDX-License-Identifier: Apache-2.0
"""Direct typed optimizers checked against independent Float64 equations.

Migrates the five elementwise families from test_x_sequence_optim.py, keeping
both configurations, six steps, 4x3 shape and original 2e-5/2e-6 tolerances.
Adafactor's factored matrix/vector algorithm is a separate fixture batch.
"""
from std.math import pow, sqrt
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec
from sequence.recurrent import OptConfig, OptState, opt_step
from sequence.ops import OPT_RMSPROP, OPT_ADAGRAD, OPT_LION, OPT_ADAMAX, OPT_NADAM

comptime SEQUENCE_OPTIMIZER_FIXTURES = 10
comptime CELLS = 12  # former 4x3 parameter matrix
comptime STEPS = 6


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


@fieldwise_init
struct Fixture(ImplicitlyCopyable, Movable):
    var kind: Int
    var flags: Int
    var lr: Float64
    var f1: Float64
    var f2: Float64
    var eps: Float64
    var decay: Float64
    var f7: Float64
    var initial: Float64


def fixture_config(fixture_id: Int) -> Fixture:
    if fixture_id == 0:
        return Fixture(OPT_RMSPROP, 0, .01, 0, .99, 1e-8, 0, 0, 0)
    if fixture_id == 1:
        return Fixture(OPT_RMSPROP, 1, .003, 0, .99, 1e-8, .01, .7, 0)
    if fixture_id == 2:
        return Fixture(OPT_ADAGRAD, 0, .01, 0, 0, 1e-10, 0, 0, 0)
    if fixture_id == 3:
        return Fixture(OPT_ADAGRAD, 0, .05, .1, 0, 1e-10, .01, 0, .1)
    if fixture_id == 4:
        return Fixture(OPT_LION, 0, .01, .9, .99, 1e-8, 0, 0, 0)
    if fixture_id == 5:
        return Fixture(OPT_LION, 0, .003, .95, .98, 1e-8, .1, 0, 0)
    if fixture_id == 6:
        return Fixture(OPT_ADAMAX, 0, .002, .9, .999, 1e-8, 0, 0, 0)
    if fixture_id == 7:
        return Fixture(OPT_ADAMAX, 0, .01, .8, .99, 1e-8, .05, 0, 0)
    if fixture_id == 8:
        return Fixture(OPT_NADAM, 0, .002, .9, .999, 1e-8, 0, .004, 0)
    return Fixture(OPT_NADAM, 1, .01, .9, .999, 1e-8, .05, .01, 0)


def fixture_data() -> Tuple[List[Float32], List[Float32]]:
    var params = List[Float32](length=CELLS, fill=0)
    var gradients = List[Float32](length=STEPS * CELLS, fill=0)
    var pp = params.unsafe_ptr()
    var gp = gradients.unsafe_ptr()
    def cell(index: Int) {imm pp, imm gp}:
        pp.unsafe_store(index, Float32((index * 7) % 19 - 9) * .0625)
        for epoch in range(STEPS):
            gp.unsafe_store(epoch * CELLS + index, Float32(((epoch + 1) * 17 + index * 13) % 31 - 15) * .03125)
    host_parallelize(cell, CELLS)
    return (params^, gradients^)


def expected_values(cfg: Fixture, params: List[Float32], gradients: List[Float32]) -> List[Float64]:
    # Four planes: parameter and the three optimizer state buffers. Each cell
    # has its own ordered recurrence; cells are independent pool work units.
    var result = List[Float64](length=4 * CELLS, fill=0)
    var rp = result.unsafe_ptr()
    var pp = params.unsafe_ptr()
    var gp = gradients.unsafe_ptr()
    def cell(index: Int) {imm cfg, imm rp, imm pp, imm gp}:
        var p = Float64(pp.unsafe_load(index))
        var first = Float64(0)
        var second = cfg.initial if cfg.kind == OPT_ADAGRAD else Float64(0)
        var third = Float64(0)
        var mu_product = Float64(1)
        for step in range(1, STEPS + 1):  # dependent optimizer recurrence
            var g = Float64(gp.unsafe_load((step - 1) * CELLS + index))
            if cfg.kind == OPT_RMSPROP:
                g += cfg.decay * p
                second = cfg.f2 * second + (1.0 - cfg.f2) * g * g
                var variance = second
                if cfg.flags != 0:
                    third = cfg.f2 * third + (1.0 - cfg.f2) * g
                    variance -= third * third
                var denominator = sqrt(variance) + cfg.eps
                if cfg.f7 > 0:
                    first = cfg.f7 * first + g / denominator
                    p -= cfg.lr * first
                else:
                    p -= cfg.lr * g / denominator
            elif cfg.kind == OPT_ADAGRAD:
                g += cfg.decay * p
                second += g * g
                var rate = cfg.lr / (1.0 + Float64(step - 1) * cfg.f1)
                p -= rate * g / (sqrt(second) + cfg.eps)
            elif cfg.kind == OPT_LION:
                var direction = cfg.f1 * first + (1.0 - cfg.f1) * g
                var sign = Float64(1) if direction > 0 else (Float64(-1) if direction < 0 else Float64(0))
                p = p * (1.0 - cfg.lr * cfg.decay) - cfg.lr * sign
                first = cfg.f2 * first + (1.0 - cfg.f2) * g
            elif cfg.kind == OPT_ADAMAX:
                g += cfg.decay * p
                first += (1.0 - cfg.f1) * (g - first)
                second = max(cfg.f2 * second, abs(g) + cfg.eps)
                p -= cfg.lr / (1.0 - pow(cfg.f1, Float64(step))) * first / second
            else:  # NAdam, coupled or decoupled decay
                if cfg.flags != 0:
                    p *= 1.0 - cfg.lr * cfg.decay
                else:
                    g += cfg.decay * p
                var mu = cfg.f1 * (1.0 - .5 * pow(.96, Float64(step) * cfg.f7))
                var mu_next = cfg.f1 * (1.0 - .5 * pow(.96, Float64(step + 1) * cfg.f7))
                mu_product *= mu
                first += (1.0 - cfg.f1) * (g - first)
                second = cfg.f2 * second + (1.0 - cfg.f2) * g * g
                var denominator = sqrt(second / (1.0 - pow(cfg.f2, Float64(step)))) + cfg.eps
                p -= cfg.lr * (1.0 - mu) / (1.0 - mu_product) * g / denominator
                p -= cfg.lr * mu_next / (1.0 - mu_product * mu_next) * first / denominator
        rp.unsafe_store(index, p)
        rp.unsafe_store(CELLS + index, first)
        rp.unsafe_store(2 * CELLS + index, second)
        rp.unsafe_store(3 * CELLS + index, third)
    host_parallelize(cell, CELLS)
    _ = params
    _ = gradients
    return result^


def run_sequence_optimizer_fixture[E: Exec](mut ex: E, fixture_id: Int) raises:
    var fixture = fixture_config(fixture_id)
    var cfg = OptConfig(fixture.kind, fixture.flags, Float32(fixture.f1), Float32(fixture.f2),
        Float32(fixture.eps), Float32(fixture.decay), Float32(fixture.f7))
    var data = fixture_data()
    var expected = expected_values(fixture, data[0], data[1])
    var initial = List[Float32](length=5 * CELLS, fill=0)
    for index in range(CELLS):  # bounded layout initialization, one fixture pool unit
        initial[index] = data[0][index]
        initial[3 * CELLS + index] = Float32(fixture.initial)
    var actual = List[Float32](length=5 * CELLS, fill=0)
    var storage = ex.alloc(5 * CELLS)
    try:
        ex.upload(storage, f32_ptr(Int(initial.unsafe_ptr())), 5 * CELLS)
        var scalar_state = OptState()
        for step in range(1, STEPS + 1):  # dependent optimizer state progression
            ex.upload(storage + CELLS, f32_ptr(Int(data[1].unsafe_ptr()) + 4 * (step - 1) * CELLS), CELLS)
            opt_step(ex, cfg, scalar_state, step, Float32(fixture.lr), storage, storage + CELLS,
                storage + 2 * CELLS, storage + 3 * CELLS, storage + 4 * CELLS, CELLS)
        ex.download(f32_ptr(Int(actual.unsafe_ptr())), storage, 5 * CELLS)
        ex.sync()
    except error:
        ex.free(storage)
        raise error
    ex.free(storage)
    _ = initial
    _ = data
    for plane in range(4):  # parameter and three state buffers
        var offset = 0 if plane == 0 else (plane + 1) * CELLS
        for index in range(CELLS):
            var want = expected[plane * CELLS + index]
            var got = Float64(actual[offset + index])
            # Original NumPy reference tolerances, unchanged. The additional
            # state comparison checks the same equations, not an output hash.
            require(abs(got - want) <= 2e-6 + 2e-5 * abs(want),
                "sequence optimizer fixture " + String(fixture_id) + " plane " + String(plane) + " cell " + String(index))
