# SPDX-License-Identifier: Apache-2.0
# Immutable tape operations for the common descriptor transport. The provider
# retains the exact boxed owner and reserves/releases under its admission GIL;
# only the numerical call runs with that GIL released. No Python execution here.
from transformer.transformer_binding_impl import (
    TransformerForwardTape, _transformer_run_session, _transformer_saved_backward,
)
from transformer.block_options import BlockOptions
from training.neural_identical_experiments import IDN_SAMBA_FORWARD_TAPE


def transformer_tape_operation(op: Int, a: List[Int], words: List[Int]) raises -> Int:
    # 0 reserve, 1 release, 2 forward, 3 backward, 4 close.
    if len(words) == 0 or words[0] == 0:
        raise Error("transformer forward tape: null native owner")
    var tape = MutPointer[TransformerForwardTape, MutUntrackedOrigin](unsafe_from_address=words[0])
    if op == 0:
        if tape[].busy:
            raise Error("transformer forward tape: busy")
        tape[].busy = True
        return 0
    if op == 1:
        tape[].busy = False
        return 0
    if not tape[].busy:
        raise Error("transformer forward tape: native operation requires a reserved owner")
    if op == 5:
        # Private operation-owned tape storage may start the next microbatch
        # only after the preceding immutable snapshot was consumed/closed.
        if len(a) != 0 or len(words) != 1 or not tape[].consumed:
            raise Error("transformer private tape reset requires a consumed snapshot")
        tape[] = TransformerForwardTape([])
        tape[].busy = True
        return 0
    if op == 4:
        tape[].close()
        return 0
    comptime if not IDN_SAMBA_FORWARD_TAPE:
        raise Error("transformer forward tape: IDENTICAL opt-in is disabled")
    if tape[].consumed:
        raise Error("transformer forward tape: busy, closed or already consumed")
    if len(a) != 11 or len(words) != (9 if op == 2 else 1):
        raise Error("transformer forward tape: invalid native descriptor arity")
    for i in range(11):  # small-loop(eleven admitted tape buffer addresses)
        if a[i] == 0:
            raise Error("transformer forward tape: null address at slot " + String(i))
    if op == 2:
        if len(tape[].shape) != 0:
            raise Error("transformer forward tape: forward snapshot already initialized")
        var p = List[Int]()
        for i in range(1, 9):  # small-loop(eight immutable shape scalars)
            p.append(words[i])
        tape[].shape = p.copy()
        var expanded = List[Int]()
        for i in range(10):  # small-loop(input and nine immutable weight addresses)
            expanded.append(a[i])
        expanded.append(0)
        expanded.append(0)
        expanded.append(a[10])
        try:
            _ = _transformer_run_session[True, True](tape[].session, expanded,
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[1], 0, p[7], BlockOptions())
        except problem:
            tape[].close()
            raise problem
        return 0
    if op != 3 or len(tape[].shape) != 8:
        raise Error("transformer forward tape: invalid forward snapshot")
    # A failing backward also consumes the immutable snapshot, exactly as the
    # original public native entry. Release of busy remains the provider's job.
    tape[].consumed = True
    try:
        _transformer_saved_backward(tape[], a)
    except problem:
        tape[].close()
        raise problem
    tape[].close()
    return 0
