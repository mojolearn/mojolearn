# SPDX-License-Identifier: Apache-2.0
# Resident whole-model reservation and execution on the original CausalLM and
# Transformer session owners. Group admission/release is GIL-held; numerical
# execution is GIL-released by the common provider, as in the public binding.
from core.native_session_group import NativeSessionGroup
from transformer.transformer_binding_impl import (
    CausalLMSession, TransformerDecodeSession, _require_session_open,
    _lm_open_run, _lm_open_run_int15, _lm_run, _lm_forward_run, LM_RESIDENT_GENERATE,
)


def _lm_group_reserve(group: NativeSessionGroup) raises:
    if group.owner == 0 or len(group.layers) == 0:
        raise Error("causal lm resident: missing model or decoder layers")
    var owner = MutPointer[CausalLMSession, MutUntrackedOrigin](unsafe_from_address=group.owner)
    if owner[].busy or not owner[].is_open() or not owner[].usable:
        raise Error("causal lm session: busy, not open or lost after a failed call")
    var run = group.operation == 1
    if (not run and group.operation != 0) or len(group.addresses) != (3 if run else 2) or len(group.parameters) != (3 if run else 1):
        raise Error("causal lm resident: invalid native group descriptor")
    if group.addresses[0] == 0 or group.addresses[1] == 0:
        raise Error("causal lm resident: null input or output")
    var length = group.parameters[0]
    var steps = group.parameters[1] if run else 1
    if length < 1 or steps < 1:
        raise Error("causal lm resident: positive length and step count required")
    var held = -1
    for i in range(len(group.layers)):  # driver-loop: validate every original layer before setting any busy flag
        if group.layers[i] == 0:
            raise Error("causal lm resident: null decoder owner")
        var layer = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=group.layers[i])
        _require_session_open(layer[])
        if layer[].b != owner[].b or layer[].dm != owner[].d:
            raise Error("causal lm resident: decoder dimensions differ from the model")
        var cached = layer[].kv.value().s
        if run:
            if cached != group.parameters[2]:
                raise Error("causal lm resident: caller and decoder cached lengths differ")
        elif held >= 0 and held != cached:
            raise Error("causal_lm_session_forward: the layers hold different cached lengths")
        held = cached
        if cached + length + steps - 1 > layer[].smax:
            raise Error("causal lm resident: additional positions do not fit the cache")
    for address in group.layers:  # driver-loop: original whole-model session reservation, no layer locks acquired
        MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=address)[].busy = True
    owner[].busy = True


def _lm_group_release(group: NativeSessionGroup):
    for address in group.layers:  # driver-loop: release original grouped decoder reservation
        MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=address)[].busy = False
    MutPointer[CausalLMSession, MutUntrackedOrigin](unsafe_from_address=group.owner)[].busy = False


def _lm_group_execute(group: NativeSessionGroup) raises -> Int:
    var owner = MutPointer[CausalLMSession, MutUntrackedOrigin](unsafe_from_address=group.owner)
    if not owner[].busy:
        raise Error("causal lm resident: execution requires a reserved group")
    try:
        if group.operation == 1:
            _lm_run(owner[], group.layers, group.addresses[0], group.addresses[1], group.addresses[2],
                group.parameters[0], group.parameters[1])
        else:
            _lm_forward_run(owner[], group.layers, group.addresses[0], group.addresses[1], group.parameters[0])
    except problem:
        for address in group.layers:  # driver-loop: original failure marks the complete session group unusable
            MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=address)[].usable = False
        owner[].usable = False
        raise problem
    var result = 0
    for address in group.layers:  # driver-loop: original last-layer cached-length return
        result = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=address)[].kv.value().s
    return result


def transformer_lm_resident_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(p) == 0 or p[0] == 0:
        raise Error("causal lm resident: null native owner")
    if op >= 30:
        if len(p) != 1 or len(a) != 0 or len(f) != 0:
            raise Error("causal lm resident: invalid group lifecycle descriptor")
        var group = MutPointer[NativeSessionGroup, MutUntrackedOrigin](unsafe_from_address=p[0])
        if op == 30:
            _lm_group_reserve(group[])
            return 0
        if op == 31:
            _lm_group_release(group[])
            return 0
        if op == 32:
            return _lm_group_execute(group[])
        raise Error("causal lm resident: unknown group operation")
    var owner = MutPointer[CausalLMSession, MutUntrackedOrigin](unsafe_from_address=p[0])
    if op == 25 or op == 26 or op == 29:
        if len(p) != 1 or len(a) != 0 or len(f) != 0:
            raise Error("causal lm resident: invalid lifecycle descriptor")
        if op == 26:
            owner[].busy = False
            return 0
        if owner[].busy:
            raise Error("causal lm session: busy")
        if op == 29:
            owner[].usable = False
            owner[].release()
            return 0
        if owner[].is_open():
            raise Error("causal lm session: already open; close it first")
        owner[].busy = True
        return 0
    var packed = op == 28
    if (op != 27 and not packed) or len(a) != (5 if packed else 3) or len(p) != 4 or len(f) != 1:
        raise Error("causal lm session open: invalid descriptor")
    if a[0] == 0 or (packed and (a[2] == 0 or a[3] == 0 or a[4] == 0)):
        raise Error("causal lm session open: null input address")
    # a[1] (the final norm) is 0 for a model without one; admitted only by
    # the MOJOLEARN_IDN_LM_RESIDENT_GENERATE experiment.
    comptime if not LM_RESIDENT_GENERATE:
        if a[1] == 0:
            raise Error("causal lm session open: null input address")
    if not owner[].busy:
        raise Error("causal lm session open: owner must be reserved")
    try:
        if packed:
            _lm_open_run_int15(owner[], a[0], a[1], [a[2], a[3], a[4]], p[1], p[2], p[3], Float32(f[0]))
        else:
            _lm_open_run(owner[], a[0], a[1], a[2], p[1], p[2], p[3], Float32(f[0]))
    except problem:
        owner[].release()
        raise problem
    owner[].usable = True
    return 0
