# SPDX-License-Identifier: Apache-2.0
"""Typed pieces of `Session.scan` (lane nn-blocks-scan, 2026-11-25). The door
(providers/python/autograd_ops_frontend_pyapi_door.mojo, group nn-blocks-scan) converts the Python arguments and
calls `session_scan`; no Python object reaches this file.

`scan_session` is the whole public operation on one session: the cell's description goes into the scan node's
words (autograd/nn_scan.mojo `scan_encode`), one `OpsSession.op` of OPS_SCAN runs it (logged in an inference
session, so `Session.capture` keeps the scan, its cell and its parameters, and a saved program carries both), then
free views take the packed column apart: operation 40 (`split`'s piece) and 33 (`reshape`), both logged too.
Returns [ys, final state 1, .., final state k]: ys is (steps, *the cell's step output shape), the outputs of every
step in time order (steps first, as torch's recurrent layers and jax.lax.scan stack them).
"""
from sequence.exec_trait import Exec
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram
from autograd.nn_scan import OPS_SCAN, OPS_SCAN_CKPT, scan_encode, scan_check_binding, scan_check_cell
from autograd.rankn import ShapeN


def scan_session[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], cell: OpsProgram, xs: Int, states: List[Int], lengths: Int, params: List[Int],
    reverse: Bool, counter: Int = 0, checkpoint_every: Int = 0,
) raises -> List[Int]:
    """The scan of `cell` over `xs` from `states`, the cell's parameters bound to `params` (handles of `s`, in the
    cell's parameter order); `lengths` a handle of one float length per sequence, or 0; `counter` this call's RNG
    key (the scan node's last word, which a training replay advances; autograd/nn_scan.mojo); `checkpoint_every`
    k >= 1 keeps the state at every k-th step and recomputes the steps between in the backward (activation-ckpt,
    autograd/nn_ckpt.mojo), 0 keeps every step."""
    if counter < 0:
        raise Error("ValueError: scan: counter must be >= 0")
    if checkpoint_every < 0:
        raise Error("ValueError: scan: checkpoint_every must be a positive number of steps, or None")
    # The binding is checked here, before the node is made, so a refusal names the caller's arguments; the op
    # checks it again when it runs (a replayed program's node).
    var state_shapes = List[ShapeN](capacity=len(states))
    var state_ids = List[Bool](capacity=len(states))
    for k in range(len(states)):  # driver-loop: over state tensors (metadata)
        state_shapes.append(s.shape_nd(states[k]))
        state_ids.append(s.is_integer(states[k]))
    var param_shapes = List[ShapeN](capacity=len(params))
    var param_ids = List[Bool](capacity=len(params))
    for j in range(len(params)):  # driver-loop: over bound parameters (metadata)
        param_shapes.append(s.shape_nd(params[j]))
        param_ids.append(s.is_integer(params[j]))
    var lengths_count = -1
    if lengths != 0:
        lengths_count = s.shape_nd(lengths).count()
    var xs_shape = s.shape_nd(xs)
    scan_check_cell(cell)
    scan_check_binding(cell, xs_shape, s.is_integer(xs), state_shapes, state_ids, lengths_count, param_shapes,
                       param_ids)
    var words = scan_encode(cell, reverse, lengths != 0, counter, checkpoint_every)
    var hs = List[Int](capacity=2 + len(states) + len(params))
    hs.append(xs)
    for k in range(len(states)):  # driver-loop: over state tensors (handles)
        hs.append(states[k])
    if lengths != 0:
        hs.append(lengths)
    for j in range(len(params)):  # driver-loop: over bound parameters (handles)
        hs.append(params[j])
    # a scan with checkpoint_every has its own code (OPS_SCAN_CKPT): the plain scan's entry tests nothing for it
    var packed = s.op(OPS_SCAN_CKPT if checkpoint_every > 0 else OPS_SCAN, hs, words[0], words[1])
    var steps = xs_shape.d[0]
    var n_state = len(states)
    var y_shape = cell.out_shapes[n_state]
    var no_fs = List[Float32]()
    # the pieces: steps * |y| words, then |state k| words each
    var sizes = List[Int](capacity=n_state + 1)
    sizes.append(steps * y_shape.count())
    for k in range(n_state):  # driver-loop: over state tensors (piece sizes)
        sizes.append(cell.in_shapes[k].count())
    var result = List[Int](capacity=n_state + 1)
    for k in range(n_state + 1):  # driver-loop: over the packed pieces (views)
        var flat = packed
        if n_state > 0:
            var ph: List[Int] = [packed]
            var pi: List[Int] = [0, k, 0]
            for q in range(len(sizes)):  # small-loop(pieces: the outputs and the state tensors)
                pi.append(sizes[q])
            flat = s.op(40, ph, pi, no_fs)
        var vh: List[Int] = [flat]
        var vi: List[Int] = [0]
        if k == 0:
            vi.append(steps)
            for d in range(y_shape.ndim):  # small-loop(ndim: at most 8)
                vi.append(y_shape.d[d])
        else:
            var sh = cell.in_shapes[k - 1]
            for d in range(sh.ndim):  # small-loop(ndim: at most 8)
                vi.append(sh.d[d])
        result.append(s.op(33, vh, vi, no_fs))
    return result^


def session_scan[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, cell_sid: Int, pid: Int, xs: Int, states: List[Int], lengths: Int, params: List[Int],
    params_given: Bool, reverse: Bool, counter: Int = 0, checkpoint_every: Int = 0,
) raises -> List[Int]:
    """`Session.scan`: the cell is program `pid` of session `cell_sid` (this session or another of the same
    executor kind; the door refuses a cell of another device). Without `params` the cell reads its own
    parameters, which must then be tensors of this session."""
    var s = ops_session[E, KIND](sid)
    var cell: OpsProgram
    if cell_sid == sid:
        cell = s[].program(pid)
    else:
        var c = ops_session[E, KIND](cell_sid)
        cell = c[].program(pid)
    if params_given:
        return scan_session(s[], cell, xs, states, lengths, params, reverse, counter, checkpoint_every)
    if cell_sid != sid and len(cell.params) > 0:
        raise Error("ValueError: scan: the cell was captured in another Session and reads "
                    + String(len(cell.params)) + " parameters; pass params= with this Session's tensors for them,"
                    + " in the cell's parameter order")
    var own = cell.params.copy()
    return scan_session(s[], cell, xs, states, lengths, own, reverse, counter, checkpoint_every)
