# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""SmallMLPTrainer's whole-step relations on its captured program (lane
tests-to-mojo-neural, 2026-11-15; moved from
python/mojolearn/tests/test_small_mlp_surface.py
`test_predict_matches_the_logits_of_the_next_step`,
`test_train_step_equals_loss_and_grads_then_apply_gradients` and
`test_train_steps_equals_consecutive_train_steps`).

The entry is autograd/small_mlp_program.mojo (`small_mlp_program_open`,
`small_mlp_program_run`), with the words bindings/small_mlp_native_door.mojo
passes: STEP [handle, rows, mode, want_dx, k] over nine addresses, APPLY
over the 195 gradients in registry order, DOWNLOAD into the parameters and
both moments. The trainer is the Python test's `trainer()`: each tensor of
[w1 16x8, b1 16, w2 3x16, b2 3] is ((0..n-1) % 7 - 3) / 32, zero moments,
step 0, the public defaults lr 1e-3, betas 0.9 / 0.999, eps 1e-8, weight
decay 0.01. A batch of r rows is x = ((0..8r-1) % 13 - 6) / 8, y = row % 3.
Every value is dyadic, so the inputs are the Python test's bit for bit.

  0  predict equals the next step's logits: on one program, a FORWARD of a
     5-row batch gives the words of the following TRAIN step's logits.
  1  one train step equals loss-and-gradients then apply: on a 9-row batch
     a TRAIN step on one program and a GRADS step then an APPLY of its
     gradients on a second give the same loss word, the same parameter and
     moment words, and the same FORWARD logits afterwards.
  2  train_steps(k = 3) equals three train steps: three TRAIN steps of the
     consecutive 4-row slices of a 12-row batch give the losses and the
     parameter and moment words of one STEP with k = 3 over the 12 rows.

The step counter, the returned dict and the refusals stay in the Python
tests (they are the door's).
"""
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
)
from checks.native_tests.training_check_common import same_words
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from autograd.ops_api import OPS_HOST, OPS_DEVICE
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from std.testing import TestSuite
from checks.native_tests.testsuite_serial import CPU_COLUMN_BUILD
from checks.native_tests.training_check_common import require


comptime IN = 8
comptime HID = 16
comptime OUT = 3
comptime TOTAL = 195


def _pattern(n: Int) -> List[Float32]:
    var values = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(fixture fill)
        values[i] = Float32((i % 7) - 3) / 32.0
    return values^


struct State(Movable):
    """The DOWNLOAD layout: [w1, b1, w2, b2] and the two AdamW moments."""
    var w1: List[Float32]
    var b1: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]
    var m: List[Float32]
    var v: List[Float32]

    def __init__(out self):
        """The Python test's `weights()`, zero moments."""
        self.w1 = _pattern(HID * IN)
        self.b1 = _pattern(HID)
        self.w2 = _pattern(OUT * HID)
        self.b2 = _pattern(OUT)
        self.m = List[Float32](length=TOTAL, fill=0)
        self.v = List[Float32](length=TOTAL, fill=0)

    def addresses(self) -> List[Int]:
        return [Int(self.w1.unsafe_ptr()), Int(self.b1.unsafe_ptr()), Int(self.w2.unsafe_ptr()),
                Int(self.b2.unsafe_ptr()), Int(self.m.unsafe_ptr()), Int(self.v.unsafe_ptr())]

    def same(self, other: State) -> Bool:
        return (same_words(self.w1, other.w1) and same_words(self.b1, other.b1)
                and same_words(self.w2, other.w2) and same_words(self.b2, other.b2)
                and same_words(self.m, other.m) and same_words(self.v, other.v))


struct Batch(Movable):
    var x: List[Float32]
    var y: List[Int32]
    var rows: Int

    def __init__(out self, rows: Int):
        """The Python test's `batch(rows)`."""
        self.rows = rows
        self.x = List[Float32](length=rows * IN, fill=0)
        self.y = List[Int32](length=rows, fill=0)
        for i in range(rows * IN):  # small-loop(fixture fill)
            self.x[i] = Float32((i % 13) - 6) / 8.0
        for r in range(rows):  # small-loop(fixture fill)
            self.y[r] = Int32(r % 3)

    def slice(self, start: Int, count: Int) -> Batch:
        var part = Batch(count)
        for i in range(count * IN):  # small-loop(fixture copy)
            part.x[i] = self.x[start * IN + i]
        for r in range(count):  # small-loop(fixture copy)
            part.y[r] = self.y[start + r]
        return part^


struct Step(Movable):
    """The STEP outputs: k losses, the last step's logits, the four
    gradients in registry order (one 195-float buffer) and the input
    gradient slot."""
    var losses: List[Float32]
    var logits: List[Float32]
    var grads: List[Float32]
    var dx: List[Float32]

    def __init__(out self, rows: Int, k: Int):
        self.losses = List[Float32](length=k, fill=-7)
        self.logits = List[Float32](length=rows * OUT, fill=-7)
        self.grads = List[Float32](length=TOTAL, fill=-7)
        self.dx = List[Float32](length=rows * IN, fill=-7)


def _open[E: Exec & Movable & Deinitable, KIND: Int](var ex: E, state: State) raises -> Int:
    var f: List[Float32] = [1e-3, 0.9, 0.999, 1e-8, 0.01]
    var handle = small_mlp_program_open[E, KIND](ex^, state.addresses(), [0], f)
    _ = state
    return handle


def _close[E: Exec & Movable & Deinitable, KIND: Int](handle: Int) raises:
    _ = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_CLOSE, List[Int](), [handle], List[Float32]())


def _download[E: Exec & Movable & Deinitable, KIND: Int](handle: Int) raises -> State:
    var state = State()
    var n = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_DOWNLOAD, state.addresses(), [handle],
                                           List[Float32]())
    require(n == TOTAL, "download returned " + String(n) + " floats, not 195")
    return state^


def _step[E: Exec & Movable & Deinitable, KIND: Int](handle: Int, batch: Batch, mode: Int, k: Int) raises -> Step:
    """`k` runs of `mode` over `batch`, `batch.rows // k` rows each."""
    var rows = batch.rows // k
    var result = Step(rows, k)
    var g = Int(result.grads.unsafe_ptr())
    var a: List[Int] = [Int(batch.x.unsafe_ptr()), Int(batch.y.unsafe_ptr()), Int(result.losses.unsafe_ptr()),
                        Int(result.logits.unsafe_ptr()), g, g + 4 * HID * IN, g + 4 * (HID * IN + HID),
                        g + 4 * (HID * IN + HID + OUT * HID), Int(result.dx.unsafe_ptr())]
    var n = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_STEP, a, [handle, rows, mode, 0, k], List[Float32]())
    require(n == rows * OUT, "step returned " + String(n) + " values, not rows * 3")
    _ = batch
    return result^


def predict_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex: E) raises:
    var start = State()
    var batch = Batch(5)
    var handle = _open[E, KIND](ex^, start)
    try:
        var predicted = _step[E, KIND](handle, batch, SMALL_MLP_PROGRAM_FORWARD, 1)
        var trained = _step[E, KIND](handle, batch, SMALL_MLP_PROGRAM_TRAIN, 1)
        require(same_words(predicted.logits, trained.logits),
                "predict_logits differs from the logits of the next train step")
    except error:
        _close[E, KIND](handle)
        raise error
    _close[E, KIND](handle)


def apply_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex_a: E, var ex_b: E) raises:
    var start = State()
    var batch = Batch(9)
    var a = _open[E, KIND](ex_a^, start)
    var b: Int
    try:
        b = _open[E, KIND](ex_b^, start)
    except error:
        _close[E, KIND](a)
        raise error
    try:
        var whole = _step[E, KIND](a, batch, SMALL_MLP_PROGRAM_TRAIN, 1)
        var parts = _step[E, KIND](b, batch, SMALL_MLP_PROGRAM_GRADS, 1)
        var n = small_mlp_program_run[E, KIND](SMALL_MLP_PROGRAM_APPLY, [Int(parts.grads.unsafe_ptr())], [b],
                                               List[Float32]())
        require(n == TOTAL, "apply returned " + String(n) + " values")
        require(same_words(whole.losses, parts.losses), "train_step loss " + String(whole.losses[0])
                + " differs from loss_and_grads " + String(parts.losses[0]))
        require(_download[E, KIND](a).same(_download[E, KIND](b)),
                "train_step and loss_and_grads + apply_gradients leave different parameters or moments")
        var after_a = _step[E, KIND](a, batch, SMALL_MLP_PROGRAM_FORWARD, 1)
        var after_b = _step[E, KIND](b, batch, SMALL_MLP_PROGRAM_FORWARD, 1)
        require(same_words(after_a.logits, after_b.logits), "the two updated models predict different logits")
        _ = parts
    except error:
        _close[E, KIND](a)
        _close[E, KIND](b)
        raise error
    _close[E, KIND](a)
    _close[E, KIND](b)


def steps_fixture[E: Exec & Movable & Deinitable, KIND: Int](var ex_a: E, var ex_b: E) raises:
    var start = State()
    var batch = Batch(12)
    var a = _open[E, KIND](ex_a^, start)
    var b: Int
    try:
        b = _open[E, KIND](ex_b^, start)
    except error:
        _close[E, KIND](a)
        raise error
    try:
        var multi = _step[E, KIND](a, batch, SMALL_MLP_PROGRAM_TRAIN, 3)
        for i in range(3):  # dependent optimizer recurrence: step i reads step i - 1's weights
            var one = _step[E, KIND](b, batch.slice(4 * i, 4), SMALL_MLP_PROGRAM_TRAIN, 1)
            require(multi.losses[i].to_bits() == one.losses[0].to_bits(),
                    "train_steps loss " + String(i) + " is " + String(multi.losses[i]) + ", consecutive "
                    + String(one.losses[0]))
        require(_download[E, KIND](a).same(_download[E, KIND](b)),
                "train_steps and three consecutive train steps leave different parameters or moments")
    except error:
        _close[E, KIND](a)
        _close[E, KIND](b)
        raise error
    _close[E, KIND](a)
    _close[E, KIND](b)


# The fixtures run one after another on both columns: the program store
# and the session registry they open into (autograd/small_mlp_program.mojo
# `_STORE`, autograd/ops_api.mojo `ops_session_open`) are process lists
# appended without a lock (training_small_mlp.mojo has the same note). The
# work inside each fixture is the program's own parallel kernels.
# ---- The host and device columns ----
# The relation fixtures on the host column (HostExec) or the device column (DeviceExec), one after another (TestSuite
# runs the tests in order; the note above says why they are ordered).


def test_predict() raises:
    comptime if CPU_COLUMN_BUILD:
        predict_fixture[HostExec, OPS_HOST](HostExec())
    else:
        predict_fixture[DeviceExec, OPS_DEVICE](DeviceExec())


def test_apply() raises:
    comptime if CPU_COLUMN_BUILD:
        apply_fixture[HostExec, OPS_HOST](HostExec(), HostExec())
    else:
        apply_fixture[DeviceExec, OPS_DEVICE](DeviceExec(), DeviceExec())


def test_steps() raises:
    comptime if CPU_COLUMN_BUILD:
        steps_fixture[HostExec, OPS_HOST](HostExec(), HostExec())
    else:
        steps_fixture[DeviceExec, OPS_DEVICE](DeviceExec(), DeviceExec())


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    TestSuite.discover_tests[__functions_in_module()]().run()
