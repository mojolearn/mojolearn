# SPDX-License-Identifier: Apache-2.0
# Executed-path control-flow VJPs. Branch decisions are discrete: predicates
# run without recording, while selected operations and loop-carried state use
# one recording tape. No state copy or reset severs an iteration's parents.
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape
from autograd.program_control import _signature
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.tape import Tape
from autograd.import_api import ImportRules


def _parameter_leaf[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], handle: Int, mut leaves: Dict[Int, Int], mut temps: List[Int],
) raises -> Int:
    if handle in leaves:
        return leaves[handle]
    var slot = s._slot(handle)
    if s.owned[slot] == 0:
        raise Error("ValueError: control-flow parameters must be uploaded tensors")
    var grad = not s.int_words[slot]
    var t = s.tape.variable_nd(s.ex, FP(unsafe_from_address=s.owned[slot]), s.shapes[slot], grad)
    var h = s._add(t, 0, grad, 0)
    s.int_words[s._slot(h)] = not grad
    temps.append(h)
    leaves[handle] = h
    return h


def _replay_handles[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram, inputs: List[Int], mut leaves: Dict[Int, Int],
    mut temps: List[Int], recording: Bool,
) raises -> List[Int]:
    if len(inputs) != p.n_in:
        raise Error("ValueError: control-flow state has the wrong number of tensors")
    s.tape.record()
    var params = List[Int]()
    for j in range(len(p.params)):  # driver-loop: over parameter references
        var slot = s._slot(p.params[j])
        if not same_shape(s.shapes[slot], p.param_shapes[j]):
            raise Error("ValueError: control-flow parameter changed shape")
        params.append(_parameter_leaf(s, p.params[j], leaves, temps))
    if not recording:
        s.tape.no_grad()
    var values = inputs.copy()
    for k in range(p.n_nodes()):  # driver-loop: over the executed graph nodes
        var args = List[Int]()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over node references
            var value_ref = p.args[j]
            args.append(params[-value_ref - 1] if value_ref < 0 else values[value_ref])
        var ints = List[Int]()
        for j in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: over integer node options
            ints.append(p.ints[j])
        var floats = List[Float32]()
        for j in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: over scalar node options
            floats.append(p.fs[j])
        var h = s._program_op(k, p.code[k], args, ints, floats)
        temps.append(h)
        values.append(h)
    var outputs = List[Int]()
    for j in range(len(p.outs)):  # driver-loop: over output references
        var value_ref = p.outs[j]
        outputs.append(params[-value_ref - 1] if value_ref < 0 else values[value_ref])
    s.tape.record()
    return outputs^


def _decision[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], condition: OpsProgram, state: List[Int],
    mut leaves: Dict[Int, Int], mut temps: List[Int],
) raises -> Bool:
    var outs = _replay_handles(s, condition, state, leaves, temps, False)
    if len(outs) != 1 or s.shape_nd(outs[0]).count() != 1:
        raise Error("ValueError: control-flow condition must return one scalar")
    var value = s.scalar(outs[0])
    if (bitcast[DType.uint32](value) & UInt32(0x7fffffff)) >= UInt32(0x7f800000):
        raise Error("ValueError: control-flow condition must be finite")
    return value != 0
