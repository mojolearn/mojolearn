# SPDX-License-Identifier: Apache-2.0
# The public scan over a user-defined cell (lane nn-blocks-scan, 2026-11-25): session code OPS_SCAN = 118.
#
# A cell is a captured Program (cell(state_1, ..., state_k, x_t) -> (next_state_1, ..., next_state_k, y_t)),
# held as DATA: its description (autograd/program.mojo `program_words`) travels inside the scan node's integer and
# float arguments, so a program that holds a scan saves, loads and replays it like any other node (the nested cell is
# part of the node's words), and nothing calls back into Python while it runs.
#
# The engine is the session itself (autograd/ops_api.mojo `OpsSession._scan`, group `nn-blocks-scan`): a driver loop
# over steps that replays the cell's nodes through `_op_run_tensors` with the step's handles, so every step is the
# same launch sequence the cell's operations make one by one, on the device for a device session and on the host
# for a host session (same kernels, same fold order, same bits on every column). The state stays in executor
# buffers; nothing is read back. In a recording session every replayed node is on the tape, so the backward is
# backward through time over the tape (no new rule): the gradient reaches the initial state, xs (through each
# step's view) and every bound parameter. Memory: the tape keeps every step's node outputs (the per-step states
# and the cell's intermediates), O(steps x the cell's values), as an unrolled loop does.
#
# This module holds what needs no session: the codes, the description words of the node, their decoding and the
# binding checks (shared by the session's op and its frontend, autograd/nn_scan_frontend.mojo).
#
# The node (session code 118): handles [xs, state_1 .. state_k, (lengths), param_1 .. param_m]; ints
# [SCAN_FORMAT, k, reverse, has_lengths, m, counter, n_words, the cell's program words ...]; floats the cell's floats.
# A dropout inside the cell draws a new mask at every step, as torch's does: step t's dropout node k keeps its
# captured seed words and takes the counter (captured counter) + ((counter * steps + t) * nodes + k), so every
# (call counter, step, node) has its own mask, the same on every replay, and the tape keeps each step's mask for the
# backward.
# Its result is one (n, 1) column: the per-step outputs in time order, each flattened, then the final states, each
# flattened; the frontend takes the pieces apart with free views (operations 40 and 33), so the op keeps the
# session's one-result shape and a capture keeps every piece.
from std.sys.compile import is_defined
from autograd.program import (
    OpsProgram, program_words, program_from_words, shape_text, same_shape, PROGRAM_IN_FLOAT, PROGRAM_PARAM_WEIGHT,
)
from autograd.rankn import ShapeN

comptime OPS_SCAN = 118
#: the version word of the node's integer arguments
comptime SCAN_FORMAT = 1
comptime SCAN_HEADER = 7

# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-nn_scan_*, lane ops-scan-cell).
# Off in every build; each arm's patch switches one on.
#   STATE_LAG: step t (t >= 2) is fed the state of step t - 2 (forward, loss, gradients and parameters move);
#   INIT_GRAD: the initial state is detached (its gradient is zero; the parameter and xs gradients of a loss that
#   reads the initial state's path move too).
# The plain name faults the device executor only, _HOST the host executor only.
comptime SCAN_SABOTAGE_STATE_LAG = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_STATE_LAG"]()
comptime SCAN_SABOTAGE_STATE_LAG_HOST = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_STATE_LAG_HOST"]()
comptime SCAN_SABOTAGE_INIT_GRAD = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_INIT_GRAD"]()
comptime SCAN_SABOTAGE_INIT_GRAD_HOST = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_INIT_GRAD_HOST"]()


struct ScanSpec(Movable):
    """The decoded scan node: the cell and its options."""
    var cell: OpsProgram
    var n_state: Int
    var reverse: Bool
    var has_lengths: Bool
    var n_params: Int
    #: the call's dropout counter (Session.scan counter=)
    var counter: Int

    def __init__(out self, var cell: OpsProgram, n_state: Int, reverse: Bool, has_lengths: Bool, n_params: Int,
                 counter: Int):
        self.cell = cell^
        self.n_state = n_state
        self.reverse = reverse
        self.has_lengths = has_lengths
        self.n_params = n_params
        self.counter = counter


def scan_host_code_name(code: Int) -> String:
    """The name of a cell operation a scan refuses (it keeps a position or a choice on the host, or is a scan),
    else the empty string."""
    if code >= 81 and code <= 84:
        return "a block step op"
    if code == 85 or code == 86:
        return "a generation op"
    if code == OPS_SCAN:
        return "a scan"
    return ""


def scan_check_cell(cell: OpsProgram) raises:
    """The cell's own signature: at least one input (the step input last), one output per state plus the step
    output, floating-point inputs, weight parameters, and only operations on device values."""
    if cell.n_in < 1:
        raise Error("ValueError: scan: the cell takes its state tensors and then one step input")
    var n_state = cell.n_in - 1
    if len(cell.outs) != n_state + 1:
        raise Error("ValueError: scan: the cell takes " + String(n_state) + " state tensors and a step input, so it"
                    + " must return " + String(n_state + 1) + " tensors (the next state tensors, then the step"
                    + " output); it returns " + String(len(cell.outs)))
    for k in range(cell.n_in):  # driver-loop: over cell inputs (metadata)
        if cell.in_kinds[k] != PROGRAM_IN_FLOAT:
            raise Error("ValueError: scan: the cell's input " + String(k)
                        + " holds integer ids; a scan carries floating-point values")
    for k in range(n_state):  # driver-loop: over cell state outputs (metadata)
        if not same_shape(cell.out_shapes[k], cell.in_shapes[k]):
            raise Error("ValueError: scan: the cell's next state " + String(k) + " is "
                        + shape_text(cell.out_shapes[k]) + " but state " + String(k) + " is "
                        + shape_text(cell.in_shapes[k]) + ": each state tensor must keep its shape")
    for j in range(len(cell.params)):  # driver-loop: over cell parameters (metadata)
        if cell.param_kind(j) != PROGRAM_PARAM_WEIGHT:
            raise Error("ValueError: scan: the cell reads declared state (Session.state), which it would write on"
                        + " the host's schedule; a scan cell reads weights only")
    for k in range(cell.n_nodes()):  # driver-loop: over cell nodes (codes)
        var name = scan_host_code_name(cell.code[k])
        if name.byte_length() > 0:
            raise Error("ValueError: scan: the cell holds " + name + " (node " + String(k)
                        + "), which reads a host value; a scan cell must be built from operations on device values")


def scan_check_binding(
    cell: OpsProgram, xs: ShapeN, xs_ids: Bool, states: List[ShapeN], states_ids: List[Bool], lengths: Int,
    params: List[ShapeN], params_ids: List[Bool],
) raises:
    """The binding of a cell to its operands, checked once per call before any step runs. `lengths` is the
    lengths tensor's element count, or -1 when none is given."""
    scan_check_cell(cell)
    var n_state = cell.n_in - 1
    if xs.ndim < 2:
        raise Error("ValueError: scan: xs must have the steps first and at least one more dimension (rank 2 or"
                    + " more); got rank " + String(xs.ndim))
    if xs_ids:
        raise Error("ValueError: scan: xs must hold floating-point values")
    var step_dims = List[Int]()
    for d in range(1, xs.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        step_dims.append(xs.d[d])
    var step = ShapeN.of(step_dims)
    var want = cell.in_shapes[n_state]
    var fits = same_shape(step, want) or (xs.ndim == 2 and step.count() == want.count())
    if not fits:
        raise Error("ValueError: scan: each step of xs is " + shape_text(step) + " but the cell's step input is "
                    + shape_text(want))
    if len(states) != n_state:
        raise Error("ValueError: scan: the cell carries " + String(n_state) + " state tensors; init_state gives "
                    + String(len(states)))
    for k in range(n_state):  # driver-loop: over state tensors (metadata)
        if states_ids[k]:
            raise Error("ValueError: scan: init_state " + String(k) + " must hold floating-point values")
        if not same_shape(states[k], cell.in_shapes[k]):
            raise Error("ValueError: scan: init_state " + String(k) + " is " + shape_text(states[k])
                        + " but the cell's state " + String(k) + " is " + shape_text(cell.in_shapes[k]))
    if len(params) != len(cell.params):
        raise Error("ValueError: scan: the cell reads " + String(len(cell.params)) + " parameters; params names "
                    + String(len(params)))
    for j in range(len(params)):  # driver-loop: over bound parameters (metadata)
        if not same_shape(params[j], cell.param_shapes[j]):
            raise Error("ValueError: scan: params[" + String(j) + "] is " + shape_text(params[j])
                        + " but the cell's parameter " + String(j) + " is " + shape_text(cell.param_shapes[j]))
        _ = params_ids[j]
    if lengths >= 0:
        if xs.ndim < 3:
            raise Error("ValueError: scan: lengths needs xs as (steps, batch, ...): rank 3 or more")
        var batch = xs.d[1]
        if lengths != batch:
            raise Error("ValueError: scan: lengths must hold one length per sequence (" + String(batch)
                        + " values); got " + String(lengths))
        for k in range(n_state):  # driver-loop: over state tensors (metadata)
            if cell.in_shapes[k].d[0] != batch or cell.in_shapes[k].ndim < 2:
                raise Error("ValueError: scan: lengths needs the batch first in every state tensor; state "
                            + String(k) + " is " + shape_text(cell.in_shapes[k]))
        var y = cell.out_shapes[n_state]
        if y.d[0] != batch or y.ndim < 2:
            raise Error("ValueError: scan: lengths needs the batch first in the step output; it is "
                        + shape_text(y))


def scan_encode(cell: OpsProgram, reverse: Bool, has_lengths: Bool, counter: Int) -> Tuple[List[Int], List[Float32]]:
    """The node's integer and float arguments (the layout in the module note)."""
    var got = program_words(cell)
    var ints = List[Int]()
    ints.append(SCAN_FORMAT)
    ints.append(cell.n_in - 1)
    ints.append(1 if reverse else 0)
    ints.append(1 if has_lengths else 0)
    ints.append(len(cell.params))
    ints.append(counter)
    ints.append(len(got[0]))
    for i in range(len(got[0])):  # driver-loop: over the cell's description words, once per call
        ints.append(Int(got[0][i]))
    return (ints^, got[1].copy())


def scan_decode(ints: List[Int], fs: List[Float32], n_handles: Int) raises -> ScanSpec:
    """The node's arguments read back and checked against its handle count."""
    if len(ints) < SCAN_HEADER or ints[0] != SCAN_FORMAT:
        raise Error("ValueError: ops: scan: the node's arguments are not a scan description")
    var n_state = ints[1]
    var reverse = ints[2] != 0
    var has_lengths = ints[3] != 0
    var n_params = ints[4]
    var counter = ints[5]
    var n_words = ints[6]
    if n_state < 0 or n_params < 0 or counter < 0 or n_words < 1 or SCAN_HEADER + n_words != len(ints):
        raise Error("ValueError: ops: scan: a damaged scan description")
    var words = List[Int64](capacity=n_words)
    for i in range(n_words):  # driver-loop: over the cell's description words, once per call
        words.append(Int64(ints[SCAN_HEADER + i]))
    var cell = program_from_words(words, fs, "ops: scan: the cell")
    if cell.n_in != n_state + 1 or len(cell.params) != n_params:
        raise Error("ValueError: ops: scan: the description's counts disagree with its cell")
    var want = 1 + n_state + (1 if has_lengths else 0) + n_params
    if n_handles != want:
        raise Error("ValueError: ops: scan takes " + String(want) + " tensors (xs, the state tensors, "
                    + ("lengths, " if has_lengths else "") + "the cell's parameters); got " + String(n_handles))
    return ScanSpec(cell^, n_state, reverse, has_lengths, n_params, counter)
