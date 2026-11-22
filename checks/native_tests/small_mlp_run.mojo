# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Native state and lifetime checks of SmallMLPTrainer's RUN file
(bindings/small_mlp_native_run.mojo) on the host program entry.

What is checked, with typed values only: a program opens from the host state
once and is reused; one call of k steps equals k calls of one step; a sync
leaves the program open and a settle-with-close hands the truth back to the
host copy, after which the next step opens again and continues from it; a
gradient applied through the program advances the step by one and matches on
two trainers; sync and close of a slot with no program are no-ops; a closed
slot keeps no program and a released handle refuses.

Serial by nature: the program's session registry is driven by one caller at
a time in the product (the door holds the GIL across every program call), so
the trainers of this suite take turns; the cases are a few 8-16-3 steps.
Not covered here: a program settled on another image when the route changes
(one process of this suite publishes one image).
"""
from std.testing import TestSuite
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.neural_entry_table import NeuralEntryKey, neural_ref, NEURAL_TABLE_MLP_PROGRAM
from core.small_mlp_state_impl import (
    MLP_STORE, MLP_TOTAL, MlpState, mlp_offset, mlp_store_create, mlp_store_index, mlp_store_release,
)
from sequence.small_mlp_entry_host import publish_neural_small_mlp_host
from bindings.small_mlp_native_run import (
    SMALL_MLP_RUN_GRADS,
    SMALL_MLP_RUN_TRAIN,
    small_mlp_run_apply,
    small_mlp_run_close,
    small_mlp_run_settle,
    small_mlp_run_steps,
    small_mlp_run_sync,
)

comptime _ROWS = 4


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error("small_mlp_run: " + message)


def _state() -> MlpState:
    var state = MlpState()
    for i in range(MLP_TOTAL):  # small-loop(the 195 fixture parameters)
        state.params[i] = Float32((i * 37) % 23 - 11) * Float32(0.01)
    state.lr = Float64(Float32(0.001))
    state.beta1 = Float64(Float32(0.9))
    state.beta2 = Float64(Float32(0.999))
    state.eps = Float64(Float32(0.00000001))
    state.weight_decay = Float64(Float32(0.01))
    state.schedule = String("{}")
    return state^


def _inputs(k: Int) -> List[Float32]:
    """`k` copies of one batch of `_ROWS` rows (minibatch i is rows
    i * _ROWS to (i + 1) * _ROWS)."""
    var x = List[Float32](length=k * _ROWS * 8, fill=Float32(0))
    for i in range(k * _ROWS * 8):  # small-loop(fixture cells)
        var cell = i % (_ROWS * 8)
        x[i] = Float32((cell * 13) % 17) * Float32(0.0625)
    return x^


def _labels(k: Int) -> List[Int32]:
    var y = List[Int32](length=k * _ROWS, fill=Int32(0))
    for i in range(k * _ROWS):  # small-loop(fixture labels)
        y[i] = Int32((i % _ROWS) % 3)
    return y^


def _step(index: Int, program: EntryRef, k: Int, mode: Int, mut gradients: List[Float32]) raises:
    """`k` steps on `k` copies of the fixture batch; the last step's packed
    gradient lands in `gradients` (MLP_TOTAL cells)."""
    var x = _inputs(k)
    var y = _labels(k)
    var losses = List[Float32](length=k, fill=Float32(0))
    var logits = List[Float32](length=_ROWS * 3, fill=Float32(0))
    var dx = List[Float32](length=_ROWS * 8, fill=Float32(0))
    var ga = Int(gradients.unsafe_ptr())
    var addresses: List[Int] = [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(losses.unsafe_ptr()),
        Int(logits.unsafe_ptr()), ga + 4 * mlp_offset(0), ga + 4 * mlp_offset(1), ga + 4 * mlp_offset(2),
        ga + 4 * mlp_offset(3), Int(dx.unsafe_ptr())]
    small_mlp_run_steps(index, program, addresses, _ROWS, k, mode, False)
    for i in range(k):  # small-loop(step losses)
        require(losses[i] == losses[i] and losses[i] > Float32(0), "a step loss is not a positive finite number")
    _ = dx^
    _ = logits^
    _ = losses^
    _ = y^
    _ = x^


def _same_state(a: Int, b: Int, what: String) raises:
    var store = MLP_STORE.get_or_create_ptr()
    require(store[].slots[a].state.step == store[].slots[b].state.step, what + ": step counters differ")
    for i in range(MLP_TOTAL):  # small-loop(the 195 parameters and moments)
        require(store[].slots[a].state.params[i] == store[].slots[b].state.params[i], what + ": parameters differ")
        require(store[].slots[a].state.m[i] == store[].slots[b].state.m[i], what + ": first moments differ")
        require(store[].slots[a].state.v[i] == store[].slots[b].state.v[i], what + ": second moments differ")


# One ordered story over two slots of the shared MLP store (serial by nature, step 4 of the TestSuite recipe).
def test_run_lifetime_in_order() raises:
    publish_neural_small_mlp_host()
    var program = neural_ref[NEURAL_TABLE_MLP_PROGRAM](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))
    var first = mlp_store_create(_state())
    var second = mlp_store_create(_state())
    var a = mlp_store_index(first)
    var b = mlp_store_index(second)
    var store = MLP_STORE.get_or_create_ptr()
    var ga = List[Float32](length=MLP_TOTAL, fill=Float32(0))
    var gb = List[Float32](length=MLP_TOTAL, fill=Float32(0))

    # A slot with no program: sync, close and settle are no-ops.
    require(store[].slots[a].program < 0, "a new slot holds a program")
    small_mlp_run_sync(a)
    small_mlp_run_close(a)
    small_mlp_run_settle(a, True)
    require(store[].slots[a].program < 0, "a no-op close opened a program")

    # Two calls of one step equal one call of two steps; the program opens
    # once and is reused.
    _step(a, program, 1, SMALL_MLP_RUN_TRAIN, ga)
    var opened = store[].slots[a].program
    require(opened >= 0, "a step did not open the program")
    require(store[].slots[a].program_entry.word == program.word, "the slot does not remember the opening entry")
    _step(a, program, 1, SMALL_MLP_RUN_TRAIN, ga)
    require(store[].slots[a].program == opened, "a second step opened a second program")
    _step(b, program, 2, SMALL_MLP_RUN_TRAIN, gb)
    require(store[].slots[a].state.step == 2 and store[].slots[b].state.step == 2, "the step counter did not advance by k")

    # A sync brings the weights down and leaves the program open.
    small_mlp_run_settle(a, False)
    small_mlp_run_settle(b, False)
    require(store[].slots[a].program == opened, "a sync closed the program")
    _same_state(a, b, "k steps in one call")
    var moved = False
    var fresh = _state()
    for i in range(MLP_TOTAL):  # small-loop(the 195 parameters)
        moved = moved or store[].slots[a].state.params[i] != fresh.params[i]
    require(moved, "two training steps left the parameters unchanged")

    # Settle with close: the host copy is the truth; the next step opens
    # again from it and continues exactly where the open program does.
    small_mlp_run_settle(a, True)
    require(store[].slots[a].program < 0, "a settle with close left the program open")
    _step(a, program, 1, SMALL_MLP_RUN_TRAIN, ga)
    require(store[].slots[a].program >= 0, "a step after a close did not open the program")
    _step(b, program, 1, SMALL_MLP_RUN_TRAIN, gb)
    small_mlp_run_settle(a, False)
    small_mlp_run_settle(b, False)
    _same_state(a, b, "a step after close and reopen")

    # Loss and gradients without a step, then the same gradient applied.
    _step(a, program, 1, SMALL_MLP_RUN_GRADS, ga)
    _step(b, program, 1, SMALL_MLP_RUN_GRADS, gb)
    require(store[].slots[a].state.step == 3, "a gradient-only run advanced the step counter")
    var nonzero = False
    for i in range(MLP_TOTAL):  # small-loop(the 195 gradient cells)
        require(ga[i] == gb[i], "two trainers in the same state returned different gradients")
        nonzero = nonzero or ga[i] != Float32(0)
    require(nonzero, "the gradient is all zeros")
    require(small_mlp_run_apply(a, program, Int(ga.unsafe_ptr())) == 4, "apply did not return the new step count")
    require(small_mlp_run_apply(b, program, Int(gb.unsafe_ptr())) == 4, "apply did not return the new step count")
    small_mlp_run_settle(a, False)
    small_mlp_run_settle(b, False)
    _same_state(a, b, "an applied gradient")
    _ = gb^
    _ = ga^

    # Close and release: no program is left, and the handle refuses.
    small_mlp_run_close(a)
    small_mlp_run_close(b)
    require(store[].slots[a].program < 0 and store[].slots[b].program < 0, "close left a program")
    require(mlp_store_release(first) and mlp_store_release(second), "a live handle did not release")
    require(not mlp_store_release(first), "a released handle released twice")
    var refused = False
    try:
        _ = mlp_store_index(first)
    except:
        refused = True
    require(refused, "a released handle still addresses a slot")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
