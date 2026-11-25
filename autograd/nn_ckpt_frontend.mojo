# SPDX-License-Identifier: Apache-2.0
"""Typed pieces of `Session.checkpoint` (lane activation-ckpt, 2026-11-25). The door
(providers/python/autograd_ops_frontend_pyapi_door.mojo, group activation-ckpt) converts the Python arguments and
calls `session_checkpoint`; no Python object reaches this file.

`checkpoint_session` is the whole public operation on one session: the segment's description goes into the node's
words (autograd/nn_ckpt.mojo `ckpt_encode`), one `OpsSession.op` of OPS_CHECKPOINT runs it (logged in an inference
session, so `Session.capture` keeps the checkpoint, its segment and its parameters, and Program.fit replays it
checkpointed), then, for a segment with several outputs, free views take the packed column apart: operation 40
(`split`'s piece) and 33 (`reshape`), both logged too. Returns one handle per output of the segment.
"""
from sequence.exec_trait import Exec
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram
from autograd.nn_ckpt import OPS_CHECKPOINT, ckpt_encode, ckpt_check_program, ckpt_check_binding
from autograd.rankn import ShapeN


def checkpoint_session[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], segment: OpsProgram, inputs: List[Int], params: List[Int], counter: Int, recompute: Bool,
) raises -> List[Int]:
    """The checkpointed call of `segment` on `inputs`, its parameters bound to `params` (handles of `s`, in the
    segment's parameter order); `counter` this call's RNG key (the node's last word, which a training replay
    advances); `recompute` False gives the plain form (the same nodes kept on the tape)."""
    if counter < 0:
        raise Error("ValueError: checkpoint: counter must be >= 0")
    ckpt_check_program(segment)
    var shapes = List[ShapeN](capacity=len(inputs) + len(params))
    var ids = List[Bool](capacity=len(inputs) + len(params))
    var hs = List[Int](capacity=len(inputs) + len(params))
    for k in range(len(inputs)):  # driver-loop: over the segment's inputs (metadata)
        shapes.append(s.shape_nd(inputs[k]))
        ids.append(s.is_integer(inputs[k]))
        hs.append(inputs[k])
    for j in range(len(params)):  # driver-loop: over the bound parameters (metadata)
        shapes.append(s.shape_nd(params[j]))
        ids.append(s.is_integer(params[j]))
        hs.append(params[j])
    if len(inputs) != segment.n_in:
        raise Error("ValueError: checkpoint: the segment takes " + String(segment.n_in) + " inputs; got "
                    + String(len(inputs)))
    if len(params) != len(segment.params):
        raise Error("ValueError: checkpoint: the segment reads " + String(len(segment.params))
                    + " parameters; params names " + String(len(params)))
    # The binding is checked here, before the node is made, so a refusal names the caller's arguments; the op
    # checks it again when it runs (a replayed program's node).
    ckpt_check_binding(segment, shapes, ids)
    var words = ckpt_encode(segment, recompute, counter)
    var packed = s.op(OPS_CHECKPOINT, hs, words[0], words[1])
    var n_out = len(segment.outs)
    var result = List[Int](capacity=n_out)
    if n_out == 1:
        result.append(packed)
        return result^
    var no_fs = List[Float32]()
    for k in range(n_out):  # driver-loop: over the packed pieces (views)
        var ph: List[Int] = [packed]
        var pi: List[Int] = [0, k, 0]
        for q in range(n_out):  # small-loop(pieces: the segment's outputs)
            pi.append(segment.out_shapes[q].count())
        var flat = s.op(40, ph, pi, no_fs)
        var vh: List[Int] = [flat]
        var vi: List[Int] = [0]
        var sh = segment.out_shapes[k]
        for d in range(sh.ndim):  # small-loop(ndim: at most 8)
            vi.append(sh.d[d])
        result.append(s.op(33, vh, vi, no_fs))
    return result^


def session_checkpoint[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, seg_sid: Int, pid: Int, inputs: List[Int], params: List[Int], params_given: Bool, counter: Int,
    recompute: Bool,
) raises -> List[Int]:
    """`Session.checkpoint`: the segment is program `pid` of session `seg_sid` (this session or another of the same
    executor kind; the door refuses a segment of another device). Without `params` the segment reads its own
    parameters, which must then be tensors of this session."""
    var s = ops_session[E, KIND](sid)
    var segment: OpsProgram
    if seg_sid == sid:
        segment = s[].program(pid)
    else:
        var c = ops_session[E, KIND](seg_sid)
        segment = c[].program(pid)
    if params_given:
        return checkpoint_session(s[], segment, inputs, params, counter, recompute)
    if seg_sid != sid and len(segment.params) > 0:
        raise Error("ValueError: checkpoint: the segment was captured in another Session and reads "
                    + String(len(segment.params)) + " parameters; pass params= with this Session's tensors for"
                    + " them, in the segment's parameter order")
    var own = segment.params.copy()
    return checkpoint_session(s[], segment, inputs, own, counter, recompute)
