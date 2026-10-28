# SPDX-License-Identifier: Apache-2.0
# Budget-aware Transformer SESSION tickets, not the standalone retained tape.
# Appended block-table operations 36..40 select these local operations 0..4.
# Native provider reservation holds the same busy flag as the existing binding;
# numerical work and failure cleanup call the original session implementations.
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from transformer.transformer_binding_impl import (
    TransformerSession, _session_forward_tape_run, _session_backward_tape_run,
)


def transformer_session_tape_operation(op: Int, a: List[Int], words: List[Int]) raises -> Int:
    if op < 0 or op > 4 or len(words) < 1 or words[0] == 0:
        raise Error("transformer session tape: invalid native descriptor")
    var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=words[0])
    if op == 0 or op == 1:
        if len(a) != 0 or len(words) != 1:
            raise Error("transformer session tape: reservation requires only its owner")
        if op == 0:
            if owner[].busy:
                raise Error("transformer tape: session is busy")
            # A closed session may still receive discard_tape, which has
            # always been a no-op after validating its cookie. Execution
            # operations independently reject closed owners below.
            owner[].busy = True
        else:
            owner[].busy = False
        return 0
    if not owner[].busy:
        raise Error("transformer session tape: execution requires a reserved owner")
    if op == 4:
        if len(a) != 0 or len(words) != 3:
            raise Error("transformer tape: owner and epoch required")
        if words[1] != words[0]:
            raise Error("transformer tape: foreign session")
        # An older ticket must not discard a more recent forward.
        if not owner[].closed and words[2] == owner[].tape_generation:
            owner[].invalidate_tape()
        return 0
    if owner[].closed:
        raise Error("transformer tape: session is busy or closed")
    if op == 2:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("transformer tape: IDENTICAL only")
        # The final address is a provider-owned four-Int report buffer; it is
        # not passed to the numerical helper or substituted for a tensor.
        if len(a) != 12 or len(words) != 11 or a[11] == 0:
            raise Error("transformer tape: expected 11 buffers, a report and ten scalars")
        var inputs = List[Int]()
        for i in range(11):  # small-loop(original tape tensor address vector)
            if a[i] == 0:
                raise Error("transformer tape: null buffer address")
            inputs.append(a[i])
        var p = List[Int]()
        for i in range(1, 11):  # small-loop(original shape/window/budget/cost words)
            p.append(words[i])
        if p[0] <= 0 or p[1] <= 0 or p[7] < 0 or p[8] < 0 or p[9] < 0:
            raise Error("transformer tape: invalid shape/window/activation budget/cost")
        try:
            _session_forward_tape_run(words[0], inputs, p)
        except problem:
            owner[].clear()
            raise problem
        var report = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[11])
        report[0] = words[0]
        report[1] = owner[].tape_generation
        report[2] = Int(owner[].tape.value().retained)
        report[3] = owner[].tape.value().retained_bytes
        return 0
    if len(a) != 11 or len(words) != 3:
        raise Error("transformer tape backward: expected 11 addresses and owner/epoch")
    var generation = words[2]
    if words[1] != words[0] or generation != owner[].tape_generation or not owner[].tape:
        raise Error("transformer tape: stale, foreign or already consumed ticket")
    for address in a:  # small-loop(original tape gradient address vector)
        if address == 0:
            raise Error("transformer tape backward: null buffer address")
    try:
        _session_backward_tape_run(words[0], a, generation)
    except problem:
        owner[].clear()
        raise problem
    return 0
