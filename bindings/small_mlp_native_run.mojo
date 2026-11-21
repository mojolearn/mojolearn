# SPDX-License-Identifier: Apache-2.0
"""RUN of SmallMLPTrainer: the trainer's captured program on the executor,
on typed values only.

The door (bindings/small_mlp_native_door.mojo) unpacks the caller's buffers
and resolves the trainer's route once; every function here takes the store
index of the trainer's state (core/small_mlp_state_impl.mojo), the resolved
program entry of the route's image (core/neural_entry_table.mojo family
NEURAL_TABLE_MLP_PROGRAM) and plain addresses and words. Nothing here imports
Python or holds a Python object; the caller decides the GIL and keeps every
addressed buffer alive until the call returns.

An open program remembers the entry that opened it (`MlpSlot.program_entry`):
its download and close go through that entry, whatever route the operation
that triggers them resolved, so a trainer whose route changed (a new process
mode) settles on the image that holds its weights and opens on the new one.

One trainer step is one program call; there are no independent units inside
an operation to split across cores (the program's kernels are the parallel
part), so this file is serial by nature.
"""
from core.entry_table import EntryRef
from core.neural_entry_table import neural_entry_call
from core.small_mlp_state_impl import MLP_STORE, MLP_TOTAL, mlp_require_state

# The program's operations (autograd/small_mlp_program.mojo): (op, addresses, ints[, floats]).
comptime SMALL_MLP_RUN_OPEN = 0
comptime SMALL_MLP_RUN_STEP = 1
comptime SMALL_MLP_RUN_DOWNLOAD = 2
comptime SMALL_MLP_RUN_APPLY = 3
comptime SMALL_MLP_RUN_CLOSE = 4

# The program's STEP modes.
comptime SMALL_MLP_RUN_FORWARD = 0
comptime SMALL_MLP_RUN_GRADS = 1
comptime SMALL_MLP_RUN_TRAIN = 2


def _run_error(error: Error) -> Error:
    """The program refuses a non-finite loss on the device; that refusal
    surfaces as a RuntimeError naming the step."""
    var message = String(error)
    if message.startswith("RuntimeError: ") or message.startswith("ImportError: ") or message.startswith("ValueError: "):
        return Error(message)
    if message.find("nonfinite") >= 0 or message.find("not finite") >= 0:
        return Error("RuntimeError: SmallMLPTrainer step returned an invalid result: " + message)
    return Error("RuntimeError: " + message)


def small_mlp_run_call(program: EntryRef, op: Int, addresses: List[Int], ints: List[Int],
                       floats: List[Float64]) raises -> Int:
    """One call of a program image's descriptor entry: plain words, one
    indirect call."""
    try:
        return neural_entry_call(program, op, addresses, ints, floats)
    except error:
        raise _run_error(error)


def _state_addresses(index: Int) raises -> List[Int]:
    var store = MLP_STORE.get_or_create_ptr()
    return [
        store[].slots[index].state.params_addr(0), store[].slots[index].state.params_addr(1),
        store[].slots[index].state.params_addr(2), store[].slots[index].state.params_addr(3),
        store[].slots[index].state.m_addr(), store[].slots[index].state.v_addr()]


def small_mlp_run_sync(index: Int) raises:
    """Bring an open program's weights and moments into the store."""
    var store = MLP_STORE.get_or_create_ptr()
    var handle = store[].slots[index].program
    if handle < 0:
        return
    if small_mlp_run_call(store[].slots[index].program_entry, SMALL_MLP_RUN_DOWNLOAD, _state_addresses(index),
                          [handle], List[Float64]()) != MLP_TOTAL:
        raise Error("RuntimeError: SmallMLPTrainer state download returned an invalid result")


def small_mlp_run_close(index: Int) raises:
    """Close an open program on the image that opened it."""
    var store = MLP_STORE.get_or_create_ptr()
    var handle = store[].slots[index].program
    if handle < 0:
        return
    var opened = store[].slots[index].program_entry
    store[].slots[index].program = -1
    _ = small_mlp_run_call(opened, SMALL_MLP_RUN_CLOSE, List[Int](), [handle], List[Float64]())


def small_mlp_run_settle(index: Int, close: Bool) raises:
    """Before the host state is read (or replaced): sync an open program,
    and close it when the host copy becomes the truth again."""
    small_mlp_run_sync(index)
    if close:
        small_mlp_run_close(index)


def small_mlp_run_open(index: Int, program: EntryRef) raises -> Int:
    """The trainer's program on the route's image, opened from the host
    state on first use (the weights and moments upload once). A program open
    on another image (the route changed) is settled there first."""
    var store = MLP_STORE.get_or_create_ptr()
    if store[].slots[index].program >= 0:
        if store[].slots[index].program_entry.word == program.word:
            return store[].slots[index].program
        small_mlp_run_settle(index, True)
    mlp_require_state(store[].slots[index].state)
    ref s = store[].slots[index].state
    var handle = small_mlp_run_call(program, SMALL_MLP_RUN_OPEN, _state_addresses(index), [s.step],
        [s.lr, s.beta1, s.beta2, s.eps, s.weight_decay])
    store[].slots[index].program = handle
    store[].slots[index].program_entry = program
    return handle


def small_mlp_run_steps(index: Int, program: EntryRef, addresses: List[Int], rows: Int, k: Int,
                        mode: Int, want_dx: Bool) raises:
    """`k` runs of the program (mode 0 forward, 1 loss and gradients, 2 the
    whole step). `addresses` is the STEP span in the program's order: x, y,
    losses, logits, the four gradients, the input gradient. The step counter
    advances under mode 2; a refused step updates nothing."""
    var store = MLP_STORE.get_or_create_ptr()
    var handle = small_mlp_run_open(index, program)
    var written = small_mlp_run_call(program, SMALL_MLP_RUN_STEP, addresses,
                                     [handle, rows, mode, 1 if want_dx else 0, k], List[Float64]())
    if written != rows * 3:
        raise Error("RuntimeError: SmallMLPTrainer step returned an invalid result")
    if mode == SMALL_MLP_RUN_TRAIN:
        store[].slots[index].state.step += k


def small_mlp_run_apply(index: Int, program: EntryRef, gradients: Int) raises -> Int:
    """One AdamW update on the program's weights and moments from the packed
    gradient at `gradients` (the update a train step applies); the new step
    count. The host copy comes down only on export."""
    var store = MLP_STORE.get_or_create_ptr()
    var handle = small_mlp_run_open(index, program)
    var applied = small_mlp_run_call(program, SMALL_MLP_RUN_APPLY, [gradients], [handle], List[Float64]())
    if applied != MLP_TOTAL:
        raise Error("RuntimeError: SmallMLPTrainer optimizer did not advance exactly one step")
    store[].slots[index].state.step += 1
    return store[].slots[index].state.step
