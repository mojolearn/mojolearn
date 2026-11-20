# SPDX-License-Identifier: Apache-2.0
"""SmallMLPTrainer's captured program against independent Float64 equations
(lane tmig-training, 2026-11-14). Ported from
`python/mojolearn/tests/test_small_mlp_numerical_edges.py` (the whole file)
and the native functions of `test_neural_public_coverage_p1.py`
(`test_small_mlp_loss_and_grads_is_cross_entropy_without_a_step`,
`test_small_mlp_predict_logits_sessions_concatenates`,
`test_mlp_inference_sessions_matches_reference`).

The entry is autograd/small_mlp_program.mojo (`small_mlp_program_open`,
`small_mlp_program_run`, `small_mlp_sessions_forward`), the functions the
host and device descriptor entries call; no Python and no binding.

Fixtures:
  0  AdamW: from planted nonzero moments at step 4 (lr .01, betas .75/.875,
     eps 1e-6, weight decay .125), two whole steps; the parameters and both
     moments after each step match the Float64 AdamW equations at rtol 2e-5,
     atol 2e-7, and the negative controls (moments reset, the sign of the
     update reversed) are outside that tolerance. The step counter is
     checked through the bias correction of the second step.
  1  Loss and gradients without a step: the loss is the Float64 mean
     cross-entropy of relu(x W1^T + b1) W2^T + b2 at rel 1e-5, every
     gradient is finite, and the weights and moments come down unchanged.
  2  The sessions forward of 12 rows matches the Float64 logits at rtol
     1e-5, atol 1e-6 and the program's own forward (predict_logits) at
     atol 1e-6.
  3  The parallel owner's shard gradient (test_parallel_mlp_native.py
     `test_gradient_matches_independent_calculus_and_preserves_input`):
     loss and the four gradients against the independent Float64 calculus.
  4  The parallel owner's apply (`test_native_adamw_update_matches_independent_first_two_step_equations`):
     two APPLY steps against the first two AdamW steps.

Inputs: fixture 0 keeps the Python test's closed-form values; fixtures 1 and
2 regenerate the Python test's NumPy draws (normal * 0.1 weights, uniform
inputs, labels in 0..2) with the counter-based generator; the assertions
are relations to the reference, so no NumPy stream is needed.
"""
from std.math import exp, log, sqrt
from sequence.exec_trait import Exec
from autograd.small_mlp_program import (
    SMALL_MLP_PROGRAM_STEP,
    SMALL_MLP_PROGRAM_DOWNLOAD,
    SMALL_MLP_PROGRAM_APPLY,
    SMALL_MLP_PROGRAM_CLOSE,
    SMALL_MLP_PROGRAM_FORWARD,
    SMALL_MLP_PROGRAM_GRADS,
    SMALL_MLP_PROGRAM_TRAIN,
    small_mlp_program_open,
    small_mlp_program_run,
    small_mlp_sessions_forward,
)
from checks.native_tests.training_check_common import (
    close64,
    all_close,
    all_close64,
    same_words,
    all_finite,
    uniform,
    uniform_int,
    to_f64,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from autograd.ops_api import OPS_HOST, OPS_DEVICE
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime SMALL_MLP_FIXTURES = 5
comptime IN = 8
comptime HID = 16
comptime OUT = 3
comptime TOTAL = 195
comptime UPDATE_RTOL = 2e-5
comptime UPDATE_ATOL = 2e-7


struct Weights(Movable):
    """The four parameters in registry order [w1 16x8, b1 16, w2 3x16, b2 3]
    and the two AdamW moments, 195 floats each (the DOWNLOAD layout)."""
    var w1: List[Float32]
    var b1: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]
    var m: List[Float32]
    var v: List[Float32]

    def __init__(out self):
        self.w1 = List[Float32](length=HID * IN, fill=0)
        self.b1 = List[Float32](length=HID, fill=0)
        self.w2 = List[Float32](length=OUT * HID, fill=0)
        self.b2 = List[Float32](length=OUT, fill=0)
        self.m = List[Float32](length=TOTAL, fill=0)
        self.v = List[Float32](length=TOTAL, fill=0)

    def addresses(self) -> List[Int]:
        return [Int(self.w1.unsafe_ptr()), Int(self.b1.unsafe_ptr()), Int(self.w2.unsafe_ptr()),
                Int(self.b2.unsafe_ptr()), Int(self.m.unsafe_ptr()), Int(self.v.unsafe_ptr())]

    def flat(self) -> List[Float32]:
        """The parameters packed in registry order (195 floats)."""
        var result = List[Float32](capacity=TOTAL)
        for value in self.w1:
            result.append(value)
        for value in self.b1:
            result.append(value)
        for value in self.w2:
            result.append(value)
        for value in self.b2:
            result.append(value)
        return result^


struct Step(Movable):
    """The STEP outputs: one loss, rows x 3 logits, the four gradients and
    the input gradient."""
    var loss: List[Float32]
    var logits: List[Float32]
    var dw1: List[Float32]
    var db1: List[Float32]
    var dw2: List[Float32]
    var db2: List[Float32]
    var dx: List[Float32]

    def __init__(out self, rows: Int):
        self.loss = List[Float32](length=1, fill=0)
        self.logits = List[Float32](length=rows * OUT, fill=0)
        self.dw1 = List[Float32](length=HID * IN, fill=0)
        self.db1 = List[Float32](length=HID, fill=0)
        self.dw2 = List[Float32](length=OUT * HID, fill=0)
        self.db2 = List[Float32](length=OUT, fill=0)
        self.dx = List[Float32](length=rows * IN, fill=0)

    def gradients(self) -> List[Float32]:
        """The four gradients packed in registry order (195 floats)."""
        var result = List[Float32](capacity=TOTAL)
        for value in self.dw1:
            result.append(value)
        for value in self.db1:
            result.append(value)
        for value in self.dw2:
            result.append(value)
        for value in self.db2:
            result.append(value)
        return result^


def _open[E: Exec & Movable & Deinitable, KIND: Int](var ex: E, w: Weights, t: Int, lr: Float32,
        beta1: Float32, beta2: Float32, eps: Float32, wd: Float32) raises -> Int:
    var f: List[Float32] = [lr, beta1, beta2, eps, wd]
    return small_mlp_program_open[E, KIND](ex^, w.addresses(), [t], f)


def _download[E: Exec & Movable & Deinitable, KIND: Int](handle: Int) raises -> Weights:
    var w = Weights()
    var n = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_DOWNLOAD, w.addresses(), [handle], List[Float32]())
    require(n == TOTAL, "download returned " + String(n) + " floats, not 195")
    return w^


def _step[E: Exec & Movable & Deinitable, KIND: Int](handle: Int, x: List[Float32], y: List[Int32],
        rows: Int, mode: Int, want_dx: Bool) raises -> Step:
    var result = Step(rows)
    var a: List[Int] = [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(result.loss.unsafe_ptr()),
                        Int(result.logits.unsafe_ptr()), Int(result.dw1.unsafe_ptr()), Int(result.db1.unsafe_ptr()),
                        Int(result.dw2.unsafe_ptr()), Int(result.db2.unsafe_ptr()), Int(result.dx.unsafe_ptr())]
    var p: List[Int] = [handle, rows, mode, 1 if want_dx else 0, 1]
    var n = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_STEP, a, p, List[Float32]())
    require(n == rows * OUT, "step returned " + String(n) + " values, not rows * 3")
    _ = x
    _ = y
    return result^


def _close[E: Exec & Movable & Deinitable, KIND: Int](handle: Int) raises:
    _ = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_CLOSE, List[Int](), [handle], List[Float32]())


# ---------------------------------------------------------------- fixture 0: AdamW


struct AdamW(Movable):
    var parameters: List[Float64]
    var m: List[Float64]
    var v: List[Float64]

    def __init__(out self):
        self.parameters = List[Float64](length=TOTAL, fill=0)
        self.m = List[Float64](length=TOTAL, fill=0)
        self.v = List[Float64](length=TOTAL, fill=0)


def adamw_reference(before: Weights, gradients: List[Float32], t: Int, lr: Float64, beta1: Float64,
        beta2: Float64, eps: Float64, wd: Float64, reset_moments: Bool, ascent: Bool) -> AdamW:
    """Independent FP64 AdamW equations (the Python test's `adamw_reference`):
    step `t` is the one being taken (the completed count + 1)."""
    var result = AdamW()
    var parameters = before.flat()
    var c1 = 1.0 - beta1 ** Float64(t)
    var c2 = 1.0 - beta2 ** Float64(t)
    for i in range(TOTAL):  # small-loop(oracle cells): 195 independent scalars
        var g = Float64(gradients[i])
        var first = 0.0 if reset_moments else Float64(before.m[i])
        var second = 0.0 if reset_moments else Float64(before.v[i])
        first = beta1 * first + (1.0 - beta1) * g
        second = beta2 * second + (1.0 - beta2) * g * g
        var update = lr * (first / c1) / (sqrt(second / c2) + eps)
        var decayed = Float64(parameters[i]) * (1.0 - lr * wd)
        result.parameters[i] = decayed + update if ascent else decayed - update
        result.m[i] = first
        result.v[i] = second
    return result^


def _linspace(start: Float64, stop: Float64, count: Int) -> List[Float32]:
    """NumPy's float32 linspace: start + i * step in float64, the last value
    exactly `stop`, then rounded once to float32."""
    var values = List[Float32](length=count, fill=0)
    var step = (stop - start) / Float64(count - 1)
    for i in range(count):  # small-loop(fixture fill): 195 values
        values[i] = Float32(start + Float64(i) * step)
    values[count - 1] = Float32(stop)
    return values^


def _edges_weights() -> Weights:
    """`((arange(prod(shape)) % 7 - 3) / 32).reshape(shape)` per weight, with
    the planted moments of the Python test."""
    var w = Weights()
    for i in range(HID * IN):  # small-loop(fixture fill)
        w.w1[i] = Float32(i % 7 - 3) / 32
    for i in range(HID):  # small-loop(fixture fill)
        w.b1[i] = Float32(i % 7 - 3) / 32
    for i in range(OUT * HID):  # small-loop(fixture fill)
        w.w2[i] = Float32(i % 7 - 3) / 32
    for i in range(OUT):  # small-loop(fixture fill)
        w.b2[i] = Float32(i % 7 - 3) / 32
    w.m = _linspace(0.01, 0.03, TOTAL)
    w.v = _linspace(0.02, 0.04, TOTAL)
    return w^


def adamw_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex: E) raises:
    comptime rows = 7
    var lr = Float32(.01)
    var beta1 = Float32(.75)
    var beta2 = Float32(.875)
    var eps = Float32(1e-6)
    var wd = Float32(.125)
    var planted = _edges_weights()
    # Nonzero carried state and a noninitial step make dropped moments and
    # reset bias-correction counters observable.
    var completed = 4
    var handle = _open[E, KIND](ex^, planted, completed, lr, beta1, beta2, eps, wd)
    try:
        for batch in range(2):  # dependent training recurrence
            var before = _download[E, KIND](handle)
            var x = List[Float32](length=rows * IN, fill=0)
            for i in range(rows * IN):  # small-loop(fixture fill)
                x[i] = Float32(i % 13 - 6 + batch) / 8
            var y = List[Int32](length=rows, fill=0)
            for i in range(rows):  # small-loop(fixture fill)
                y[i] = Int32((i + batch) % 3)
            var result = _step[E, KIND](handle, x, y, rows, SMALL_MLP_PROGRAM_TRAIN, True)
            var after = _download[E, KIND](handle)
            var gradients = result.gradients()
            var t = completed + 1
            var want = adamw_reference(before, gradients, t, Float64(lr), Float64(beta1), Float64(beta2),
                                       Float64(eps), Float64(wd), False, False)
            require(all_close(after.flat(), want.parameters, UPDATE_RTOL, UPDATE_ATOL),
                    "batch " + String(batch) + ": parameters are not the AdamW equations")
            require(all_close(after.m, want.m, UPDATE_RTOL, UPDATE_ATOL),
                    "batch " + String(batch) + ": first moment is not the AdamW equations")
            require(all_close(after.v, want.v, UPDATE_RTOL, UPDATE_ATOL),
                    "batch " + String(batch) + ": second moment is not the AdamW equations")
            # Negative controls: the tolerance sees missing carried moments and
            # an optimizer sign reversal.
            var reset = adamw_reference(before, gradients, t, Float64(lr), Float64(beta1), Float64(beta2),
                                        Float64(eps), Float64(wd), True, False)
            var ascent = adamw_reference(before, gradients, t, Float64(lr), Float64(beta1), Float64(beta2),
                                         Float64(eps), Float64(wd), False, True)
            require(not all_close64(want.parameters, reset.parameters, UPDATE_RTOL, UPDATE_ATOL),
                    "negative control: reset moments are inside the parameter tolerance")
            require(not all_close64(want.m, reset.m, UPDATE_RTOL, UPDATE_ATOL),
                    "negative control: a reset first moment is inside the tolerance")
            require(not all_close64(want.v, reset.v, UPDATE_RTOL, UPDATE_ATOL),
                    "negative control: a reset second moment is inside the tolerance")
            require(not all_close64(want.parameters, ascent.parameters, UPDATE_RTOL, UPDATE_ATOL),
                    "negative control: an ascent step is inside the parameter tolerance")
            completed = t
    except error:
        _close[E, KIND](handle)
        raise error
    _close[E, KIND](handle)


# ---------------------------------------------------------------- fixtures 1 and 2: the public draws


comptime ROWS_M = 12


def _drawn_weights() -> Weights:
    """normal * 0.1 weights (here uniform with the same spread), zero biases,
    zero moments: `SmallMLPTrainer(W1, B1, W2, B2)`."""
    var w = Weights()
    for i in range(HID * IN):  # small-loop(fixture fill)
        w.w1[i] = uniform(0, i, -0.17, 0.17)
    for i in range(OUT * HID):  # small-loop(fixture fill)
        w.w2[i] = uniform(1, i, -0.17, 0.17)
    return w^


def _drawn_x() -> List[Float32]:
    var x = List[Float32](length=ROWS_M * IN, fill=0)
    for i in range(ROWS_M * IN):  # small-loop(fixture fill)
        x[i] = uniform(2, i, 0.0, 1.0)
    return x^


def _drawn_y() -> List[Int32]:
    var y = List[Int32](length=ROWS_M, fill=0)
    for i in range(ROWS_M):  # small-loop(fixture fill)
        y[i] = Int32(uniform_int(3, i, 3))
    return y^


def logits_reference(w: Weights, x: List[Float32], rows: Int) -> List[Float64]:
    """relu(x W1^T + b1) W2^T + b2 in Float64."""
    var result = List[Float64](length=rows * OUT, fill=0)
    for r in range(rows):  # small-loop(oracle rows): fixture-sized
        var hidden = List[Float64](length=HID, fill=0)
        for h in range(HID):
            var acc = Float64(w.b1[h])
            for i in range(IN):
                acc += Float64(x[r * IN + i]) * Float64(w.w1[h * IN + i])
            hidden[h] = max(acc, 0.0)
        for o in range(OUT):
            var acc = Float64(w.b2[o])
            for h in range(HID):
                acc += hidden[h] * Float64(w.w2[o * HID + h])
            result[r * OUT + o] = acc
    return result^


def loss_and_grads_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex: E) raises:
    var w = _drawn_weights()
    var x = _drawn_x()
    var y = _drawn_y()
    var handle = _open[E, KIND](ex^, w, 0, Float32(1e-3), Float32(.9), Float32(.999), Float32(1e-8), Float32(0))
    try:
        var result = _step[E, KIND](handle, x, y, ROWS_M, SMALL_MLP_PROGRAM_GRADS, False)
        var z = logits_reference(w, x, ROWS_M)
        var loss = Float64(0)
        for r in range(ROWS_M):  # small-loop(oracle rows)
            var top = max(z[r * OUT], max(z[r * OUT + 1], z[r * OUT + 2]))
            var total = Float64(0)
            for o in range(OUT):
                total += exp(z[r * OUT + o] - top)
            loss += -(z[r * OUT + Int(y[r])] - top - log(total))
        loss /= Float64(ROWS_M)
        require(close64(Float64(result.loss[0]), loss, 1e-5, 1e-12),
                "loss " + String(result.loss[0]) + " is not the mean cross-entropy " + String(loss))
        require(all_finite(result.gradients()), "a gradient is not finite")
        # Without a step: the weights and moments come down as they went up.
        var after = _download[E, KIND](handle)
        require(same_words(after.flat(), w.flat()), "loss and gradients moved the weights")
        require(same_words(after.m, w.m) and same_words(after.v, w.v), "loss and gradients moved the moments")
    except error:
        _close[E, KIND](handle)
        raise error
    _close[E, KIND](handle)


def sessions_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex: E, var program_ex: E) raises:
    var w = _drawn_weights()
    var x = _drawn_x()
    var y = _drawn_y()
    var got = List[Float32](length=ROWS_M * OUT, fill=0)
    var a: List[Int] = [Int(got.unsafe_ptr()), Int(x.unsafe_ptr()), Int(w.w1.unsafe_ptr()),
                        Int(w.b1.unsafe_ptr()), Int(w.w2.unsafe_ptr()), Int(w.b2.unsafe_ptr())]
    var n = small_mlp_sessions_forward[E, KIND](ex^, a, [ROWS_M, IN, HID, OUT])
    require(n == ROWS_M * OUT, "sessions forward returned " + String(n) + " values")
    var want = logits_reference(w, x, ROWS_M)
    require(all_close(got, want, 1e-5, 1e-6), "sessions forward is not the reference logits")
    var handle = _open[E, KIND](program_ex^, w, 0, Float32(1e-3), Float32(.9), Float32(.999), Float32(1e-8), Float32(0))
    try:
        var forward = _step[E, KIND](handle, x, y, ROWS_M, SMALL_MLP_PROGRAM_FORWARD, False)
        require(all_close(got, to_f64(forward.logits), 1e-7, 1e-6),
                "sessions forward is not the program's predict_logits")
    except error:
        _close[E, KIND](handle)
        raise error
    _close[E, KIND](handle)
    _ = w
    _ = x


# ---------------------------------------------------------------- fixtures 3 and 4: the parallel owner
# Ported from test_parallel_mlp_native.py: a ParallelNeural shard's
# 'gradient' is ONE program run in loss-and-gradients mode and 'apply' is the
# program's APPLY (bindings/parallel_mlp_native_door.mojo). The integer label
# dtypes and the Fortran-ordered input of the Python test are boundary
# admission and stay there.


def _parallel_weights() -> Weights:
    """`model_fixture`: zeros except w1[0, 0] = 1, b1[0] = .5 and
    w2[:, 0] = [.25, -.25, .125]."""
    var w = Weights()
    w.w1[0] = 1.0
    w.b1[0] = 0.5
    w.w2[0 * HID] = 0.25
    w.w2[1 * HID] = -0.25
    w.w2[2 * HID] = 0.125
    return w^


def parallel_gradient_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex: E) raises:
    """The independent calculus: loss at rel 2e-6 / abs 2e-7, gradients at
    rtol 3e-6, atol 3e-7, the weights unchanged."""
    comptime rows = 3
    var w = _parallel_weights()
    var x = List[Float32](length=rows * IN, fill=0)
    x[0 * IN] = 1.0
    x[1 * IN] = -0.25
    x[2 * IN] = 0.5
    var y: List[Int32] = [0, 1, 2]
    var handle = _open[E, KIND](ex^, w, 0, Float32(.0625), Float32(.5), Float32(.75), Float32(1e-8), Float32(.25))
    try:
        var result = _step[E, KIND](handle, x, y, rows, SMALL_MLP_PROGRAM_GRADS, False)
        # independent Float64 calculus (the Python test's `independent_gradient`)
        var z = List[Float64](length=rows * HID, fill=0)
        var h = List[Float64](length=rows * HID, fill=0)
        var dl = List[Float64](length=rows * OUT, fill=0)
        var loss = Float64(0)
        for r in range(rows):  # small-loop(oracle rows): three rows
            for k in range(HID):
                var acc = Float64(w.b1[k])
                for i in range(IN):
                    acc += Float64(x[r * IN + i]) * Float64(w.w1[k * IN + i])
                z[r * HID + k] = acc
                h[r * HID + k] = max(acc, 0.0)
            var logits = List[Float64](length=OUT, fill=0)
            var top = Float64(-1e300)
            for o in range(OUT):
                var acc = Float64(w.b2[o])
                for k in range(HID):
                    acc += h[r * HID + k] * Float64(w.w2[o * HID + k])
                logits[o] = acc
                top = max(top, acc)
            var total = Float64(0)
            for o in range(OUT):
                total += exp(logits[o] - top)
            for o in range(OUT):
                var p = exp(logits[o] - top) / total
                if o == Int(y[r]):
                    loss += -log(p)
                    p -= 1.0
                dl[r * OUT + o] = p / Float64(rows)
        loss /= Float64(rows)
        var want = List[Float64](length=TOTAL, fill=0)
        for r in range(rows):  # small-loop(oracle rows)
            for k in range(HID):
                var dh = Float64(0)
                for o in range(OUT):
                    dh += dl[r * OUT + o] * Float64(w.w2[o * HID + k])
                if z[r * HID + k] <= 0:
                    dh = 0
                for i in range(IN):
                    want[k * IN + i] += dh * Float64(x[r * IN + i])
                want[128 + k] += dh
            for o in range(OUT):
                for k in range(HID):
                    want[144 + o * HID + k] += dl[r * OUT + o] * h[r * HID + k]
                want[192 + o] += dl[r * OUT + o]
        require(abs(Float64(result.loss[0]) - loss) <= max(2e-6 * abs(loss), 2e-7),
                "loss " + String(result.loss[0]) + " is not the independent calculus " + String(loss))
        require(all_close(result.gradients(), want, 3e-6, 3e-7), "gradients are not the independent calculus")
        var after = _download[E, KIND](handle)
        require(same_words(after.flat(), w.flat()), "the gradient moved the weights")
    except error:
        _close[E, KIND](handle)
        raise error
    _close[E, KIND](handle)


def parallel_apply_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex: E) raises:
    """Two APPLY steps of constant .125 gradients against the first two AdamW
    steps: parameters at rtol 3e-6, atol 3e-7, moments at atol 1e-8."""
    var w = _parallel_weights()
    var lr = Float64(.0625)
    var beta1 = Float64(.5)
    var beta2 = Float64(.75)
    var eps = Float64(1e-8)
    var wd = Float64(.25)
    var handle = _open[E, KIND](ex^, w, 0, Float32(lr), Float32(beta1), Float32(beta2), Float32(eps), Float32(wd))
    try:
        var gradients = List[Float32](length=TOTAL, fill=0.125)
        var expected = to_f64(w.flat())
        var m = Float64(0)
        var v = Float64(0)
        for step in range(1, 3):  # dependent optimizer recurrence
            var n = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_APPLY, [Int(gradients.unsafe_ptr())], [handle], List[Float32]())
            require(n == TOTAL, "apply returned " + String(n) + " values")
            m = beta1 * m + (1.0 - beta1) * 0.125
            v = beta2 * v + (1.0 - beta2) * 0.125 * 0.125
            for i in range(TOTAL):  # small-loop(oracle cells)
                expected[i] = expected[i] * (1.0 - lr * wd) - lr * (m / (1.0 - beta1 ** Float64(step))) / (
                    sqrt(v / (1.0 - beta2 ** Float64(step))) + eps)
            var after = _download[E, KIND](handle)
            require(all_close(after.flat(), expected, 3e-6, 3e-7), "step " + String(step) + ": parameters are not AdamW")
            require(all_close(after.m, List[Float64](length=TOTAL, fill=m), 0.0, 1e-8),
                    "step " + String(step) + ": first moment is not AdamW")
            require(all_close(after.v, List[Float64](length=TOTAL, fill=v), 0.0, 1e-8),
                    "step " + String(step) + ": second moment is not AdamW")
        _ = gradients
    except error:
        _close[E, KIND](handle)
        raise error
    _close[E, KIND](handle)


# ---- The host column (was training_small_mlp_cpu_suite.mojo) ----
# SmallMLPTrainer program fixtures on the cpu column (training_small_mlp_cases.mojo).
#
# The fixtures run one after another: the program store and the session
# registry they open into (autograd/small_mlp_program.mojo `_STORE`,
# autograd/ops_api.mojo `ops_session_open`) are process lists appended
# without a lock, so concurrent opens from pool workers would race. The work
# inside each fixture is the program's own parallel kernels.
def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(SMALL_MLP_FIXTURES)
    for unit in range(SMALL_MLP_FIXTURES):  # ordered: see the module note
        try:
            if unit == 0:
                adamw_fixture[HostExec, OPS_HOST](HostExec())
            elif unit == 1:
                loss_and_grads_fixture[HostExec, OPS_HOST](HostExec())
            elif unit == 2:
                sessions_fixture[HostExec, OPS_HOST](HostExec(), HostExec())
            elif unit == 3:
                parallel_gradient_fixture[HostExec, OPS_HOST](HostExec())
            else:
                parallel_apply_fixture[HostExec, OPS_HOST](HostExec())
        except error:
            cases.fail(unit, error)


# ---- The device column (was training_small_mlp_gpu_suite.mojo) ----
# SmallMLPTrainer program fixtures on the gpu column (training_small_mlp_cases.mojo).
#
# The fixtures run one after another: the program store and the session
# registry they open into (autograd/small_mlp_program.mojo `_STORE`,
# autograd/ops_api.mojo `ops_session_open`) are process lists appended
# without a lock, so concurrent opens from pool workers would race. The work
# inside each fixture is the program's own parallel kernels.
def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(SMALL_MLP_FIXTURES)
    for unit in range(SMALL_MLP_FIXTURES):  # ordered: see the module note
        try:
            if unit == 0:
                adamw_fixture[DeviceExec, OPS_DEVICE](DeviceExec())
            elif unit == 1:
                loss_and_grads_fixture[DeviceExec, OPS_DEVICE](DeviceExec())
            elif unit == 2:
                sessions_fixture[DeviceExec, OPS_DEVICE](DeviceExec(), DeviceExec())
            elif unit == 3:
                parallel_gradient_fixture[DeviceExec, OPS_DEVICE](DeviceExec())
            else:
                parallel_apply_fixture[DeviceExec, OPS_DEVICE](DeviceExec())
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("training_small_mlp", cpu=True, gpu=True, serial="", run=run_cases)
