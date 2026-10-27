# SPDX-License-Identifier: Apache-2.0
# Same single-use native Mamba tape owners and numerical helpers as the public
# binding. The common provider owns reservation, GIL lifetime and object roots.
from mamba.impl.mamba_shell_impl import (
    Mamba3ForwardTape, Mamba3PrefillSession, IDN_SAMBA_M3_FORWARD_TAPE, _m3_prefill_run, _m3_backward_tape_run,
)


def mamba_tape_operation(op: Int, a: List[Int], words: List[Int]) raises -> Int:
    # 0 reserve, 1 release, 2 forward, 3 backward, 4 close.
    if len(words) == 0 or words[0] == 0:
        raise Error("mamba3 forward tape: null native owner")
    var tape = MutPointer[Mamba3ForwardTape, MutUntrackedOrigin](unsafe_from_address=words[0])
    if op == 0:
        if tape[].owner.busy:
            raise Error("mamba3 forward tape: busy")
        tape[].owner.busy = True
        return 0
    if op == 1:
        tape[].owner.busy = False
        return 0
    if not tape[].owner.busy:
        raise Error("mamba3 forward tape: native operation requires a reserved owner")
    if op == 5:
        if len(a) != 0 or len(words) != 1 or not tape[].consumed:
            raise Error("mamba3 private tape reset requires a consumed snapshot")
        tape[] = Mamba3ForwardTape(Mamba3PrefillSession())
        tape[].owner.busy = True
        return 0
    if op == 4:
        tape[].consumed = True
        tape[].owner.usable = False
        tape[].owner.release()
        return 0
    comptime if not IDN_SAMBA_M3_FORWARD_TAPE:
        raise Error("mamba3 forward tape: experiment disabled")
    if tape[].consumed:
        raise Error("mamba3 backward tape: consumed or closed")
    if len(a) != 11 or len(words) != (4 if op == 2 else 1):
        raise Error("mamba3 forward tape: invalid native descriptor arity")
    for i in range(11):  # small-loop(eleven admitted tape buffer addresses)
        if a[i] == 0:
            raise Error("mamba3 forward tape: null buffer at slot " + String(i))
    if op == 2:
        if tape[].owner.stages_valid:
            raise Error("mamba3 forward tape: forward snapshot already initialized")
        if words[1] < 1 or words[2] < 1 or words[3] < 1:
            raise Error("mamba3 forward tape: positive B, L and d_model required")
        var expanded = List[Int](length=25, fill=0)
        for i in range(10):  # small-loop(input and nine immutable weight addresses)
            expanded[i] = a[i]
        expanded[20] = a[10]
        try:
            _ = _m3_prefill_run(tape[].owner, expanded, words[1], words[2], words[3], force_retain=True)
        except problem:
            tape[].consumed = True
            tape[].owner.usable = False
            tape[].owner.release()
            raise problem
        return 0
    if op != 3 or not tape[].owner.usable or not tape[].owner.stages_valid:
        raise Error("mamba3 backward tape: invalid forward snapshot")
    # Nonfinite dy consumes an immutable tape too: unlike a reusable prefill
    # session, this contract never permits retry on partially mutated stages.
    tape[].consumed = True
    try:
        _m3_backward_tape_run(tape[].owner, a)
    except problem:
        tape[].owner.usable = False
        tape[].owner.release()
        raise problem
    tape[].owner.usable = False
    tape[].owner.release()
    return 0
