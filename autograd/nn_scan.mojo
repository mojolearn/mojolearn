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
# and the cell's intermediates), O(steps x the cell's values), as an unrolled loop does; with checkpoint_every = k
# (format 3, lane activation-ckpt) it keeps the state every k steps and one segment's values while the backward
# recomputes it (autograd/nn_ckpt.mojo: the formula, the bits).
#
# This module holds what needs no session: the codes, the description words of the node, their decoding and the
# binding checks (shared by the session's op and its frontend, autograd/nn_scan_frontend.mojo).
#
# The node (session code 118): handles [xs, state_1 .. state_k, (lengths), param_1 .. param_m]; ints
# [SCAN_FORMAT, fingerprint low 32 bits, fingerprint high 32 bits, k, reverse, has_lengths, m, n_words, the cell's
# program words ..., key]; floats the cell's floats. The FINGERPRINT (FNV-1a over every integer after the first three
# except the key, and every float's bits; `scan_fingerprint`) is written once when the node is made; the session keeps
# each decoded cell under it (`ScanCache`, one dictionary read per call), so a scan replayed every minibatch parses its
# cell and splits its nodes' arguments once per session, not once per call (lane fable-followups, 2026-11-25:
# AGENTS.md "No refactor adds cost on a hot path"). The key is outside the fingerprint because a replay advances it.
# The KEY is the node's last integer, the word a training replay advances for every random node
# (autograd/program_runtime.mojo, PreparedReplayNode.random: the same rule as a top-level dropout's counter, from the
# captured value: captured + step * stride + offset + node), so a scan inside Program.fit draws new masks every
# minibatch, a resumed fit continues the same stream, and an inference replay sees the captured key. Eagerly it is
# Session.scan's `counter=` (as Session.dropout's `counter`: a new one every training step).
# A dropout inside the cell draws its mask from a seed derived from (its captured seed, the key, its cell node, the
# time index) by a 64-bit mix (`scan_dropout_seed`, splitmix64 rounds), not a linear offset, so no two (key, node,
# step) share a mask whatever the sequence lengths; its captured counter is kept. The derivation is host integer
# arithmetic once per dropout node per step, the same words for the host and the device column; the tape keeps each
# step's mask for the backward and a replay derives the same seeds.
# Its result is one (n, 1) column: the per-step outputs in time order, each flattened, then the final states, each
# flattened; the frontend takes the pieces apart with free views (operations 40 and 33), so the op keeps the
# session's one-result shape and a capture keeps every piece.
from std.sys.compile import is_defined
from autograd.program import (
    OpsProgram, program_words, program_from_words, shape_text, same_shape, PROGRAM_IN_FLOAT, PROGRAM_PARAM_WEIGHT,
)
from autograd.rankn import ShapeN
from autograd.nn_cbw import OPS_CUSTOM_VJP, _fnv, _split_nodes
from std.memory import bitcast

comptime OPS_SCAN = 118
#: the version word of the node's integer arguments (2: the fingerprint words, lane fable-followups 2026-11-25)
comptime SCAN_FORMAT = 2
comptime SCAN_HEADER = 8
#: lane activation-ckpt (2026-11-25): a scan with `checkpoint_every` (Session.scan): format 3, one more header word
#: (the segment length k >= 1) after n_words; a scan without it keeps format 2 and its words unchanged. The forward,
#: the backward that recomputes each segment and the memory saved are autograd/nn_ckpt.mojo's note.
comptime SCAN_FORMAT_CKPT = 3
#: The session code of a scan with checkpoint_every (lane fable-finalize): its own code, tested after every other
#: operation's, so the plain scan's entry (`_scan`, code 118) has no checkpoint test in it. Same handles and words
#: as OPS_SCAN with format 3.
comptime OPS_SCAN_CKPT = 120
#: autograd/nn_ckpt.mojo OPS_CHECKPOINT (not imported: nn_ckpt imports this module)
comptime _SCAN_CKPT_CODE = 119
#: autograd/ops_api.mojo OPS_DROPOUT (not imported: ops_api imports this module); ints [seed lo, seed hi, counter]
comptime _SCAN_DROPOUT_CODE = 21

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
#   DROP_KEY: a dropout inside the cell reuses its captured seed at every step and every key (the masks repeat).
comptime SCAN_SABOTAGE_DROP_KEY = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_DROP_KEY"]()
comptime SCAN_SABOTAGE_DROP_KEY_HOST = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_DROP_KEY_HOST"]()


struct ScanCell(Movable):
    """One decoded scan node, kept by its session for every later call with the same fingerprint: the cell, its
    options, each cell node's integer and float arguments split once, and the cell's dropout nodes with their
    captured seeds. The per-call RNG key is not here (it is the node's last word, read per call)."""
    var cell: OpsProgram
    var n_state: Int
    var reverse: Bool
    var has_lengths: Bool
    var n_params: Int
    #: the node's fingerprint and word counts (a dictionary hit is taken only when the counts agree too)
    var fingerprint: Int
    var n_ints: Int
    var n_fs: Int
    #: each cell node's arguments; a dropout node's two seed words are rewritten before every step that runs it
    var node_ints: List[List[Int]]
    var node_fs: List[List[Float32]]
    var drop_nodes: List[Int]
    var drop_seed: List[UInt64]
    #: Session-local rule index per custom node, -1 for a primitive. Empty for ordinary cells.
    var custom_ids: List[Int]
    #: lane activation-ckpt: the checkpoint segment length (format 3), 0 for a plain scan (format 2)
    var every: Int

    def __init__(out self):
        self.cell = OpsProgram()
        self.n_state = 0
        self.reverse = False
        self.has_lengths = False
        self.n_params = 0
        self.every = 0
        self.fingerprint = 0
        self.n_ints = 0
        self.n_fs = 0
        self.node_ints = List[List[Int]]()
        self.node_fs = List[List[Float32]]()
        self.custom_ids = List[Int]()
        self.drop_nodes = List[Int]()
        self.drop_seed = List[UInt64]()


struct ScanCache(Movable):
    """A session's decoded scan cells (`ScanCell`), indexed by fingerprint. Cleared when the session closes."""
    var cells: List[ScanCell]
    var by_key: Dict[Int, Int]

    def __init__(out self):
        self.cells = List[ScanCell]()
        self.by_key = Dict[Int, Int]()

    def clear(mut self):
        self.cells.clear()
        self.by_key.clear()

    def find(self, ints: List[Int], fs: List[Float32]) raises -> Int:
        """The index of the decoded cell these node words name, or -1 (one dictionary read, no parse)."""
        if len(ints) < SCAN_HEADER + 2 or (ints[0] != SCAN_FORMAT and ints[0] != SCAN_FORMAT_CKPT):
            raise Error("ValueError: ops: scan: the node's arguments are not a scan description")
        var found = self.by_key.find(scan_key_of(ints))
        if found:
            var i = found.value()
            if self.cells[i].n_ints == len(ints) and self.cells[i].n_fs == len(fs):
                return i
        return -1

    def add(mut self, var cell: ScanCell) -> Int:
        """Keep a decoded cell; returns its index."""
        var key = cell.fingerprint
        self.cells.append(cell^)
        self.by_key[key] = len(self.cells) - 1
        return len(self.cells) - 1


def scan_host_code_name(code: Int) -> String:
    """The name of a cell operation a scan refuses (it keeps a position or a choice on the host, or is a scan),
    else the empty string."""
    if code >= 81 and code <= 84:
        return "a block step op"
    if code == 85 or code == 86:
        return "a generation op"
    if code == OPS_SCAN or code == OPS_SCAN_CKPT:
        return "a scan"
    if code == OPS_CUSTOM_VJP:
        return "a custom operation"
    if code == _SCAN_CKPT_CODE:
        # activation-ckpt: likewise (the nested instance has no checkpoint branch); a scan checkpoints its own
        # steps with checkpoint_every
        return "a checkpoint"
    return ""


# Replace a same-shaped unary custom rule in a scanned cell by the ordinary identity gradient.
comptime SCAN_SABOTAGE_CUSTOM_RULE = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_CUSTOM_RULE"]()
comptime SCAN_SABOTAGE_CUSTOM_RULE_HOST = is_defined["MOJOLEARN_NN_SCAN_SABOTAGE_CUSTOM_RULE_HOST"]()


def scan_check_custom_program(p: OpsProgram, role: String) raises:
    """A cell's custom rule may only use operations on device values, in both directions."""
    for k in range(p.n_nodes()):  # driver-loop: over custom program nodes at bind
        var name = scan_host_code_name(p.code[k])
        if name.byte_length() > 0:
            raise Error("ValueError: scan: a custom operation's " + role + " program holds " + name
                        + " (node " + String(k) + "); scan cells operate on device values")


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
        if cell.code[k] == OPS_CUSTOM_VJP:
            continue  # The session checks both embedded graphs before it caches their custom rule.
        var name = scan_host_code_name(cell.code[k])
        if name.byte_length() > 0:
            raise Error("ValueError: scan: the cell holds " + name + " (node " + String(k)
                        + "), which reads a host value; a scan cell must be built from operations on device values")


def scan_check_binding(
    cell: OpsProgram, xs: ShapeN, xs_ids: Bool, states: List[ShapeN], states_ids: List[Bool], lengths: Int,
    params: List[ShapeN], params_ids: List[Bool],
) raises:
    """The binding of a cell to its operands, checked once per call before any step runs. `lengths` is the
    lengths tensor's element count, or -1 when none is given. The cell's own signature (`scan_check_cell`) is
    checked by the caller: once per decoded cell in the session, once per call in the frontend."""
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


def _splitmix64(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def scan_dropout_seed(seed: UInt64, key: Int, node: Int, t: Int) -> UInt64:
    """The 64-bit dropout seed of cell node `node` at time index `t` of a scan with RNG key `key`, from the
    node's captured seed: four splitmix64 rounds, one word each, so it does not depend on the sequence length."""
    var h = _splitmix64(seed)
    h = _splitmix64(h ^ UInt64(key))
    h = _splitmix64(h ^ UInt64(node))
    return _splitmix64(h ^ UInt64(t))


def scan_key_of(ints: List[Int]) -> Int:
    """The fingerprint a scan node's words carry (words 1 and 2)."""
    return (ints[1] & 0xFFFFFFFF) | ((ints[2] & 0xFFFFFFFF) << 32)


def scan_fingerprint(ints: List[Int], fs: List[Float32]) -> Int:
    """FNV-1a over the node's integers after the first three, except the last (the key a replay advances), and
    every float's bits (the hash autograd/nn_cbw.mojo uses for a custom operation's key)."""
    var h = UInt64(0xCBF29CE484222325)
    for j in range(3, len(ints) - 1):  # driver-loop: over node words, once per node made or decoded
        h = _fnv(h, bitcast[DType.uint64](Int64(ints[j])))
    for j in range(len(fs)):  # driver-loop: over float words, once per node made or decoded
        h = _fnv(h, UInt64(bitcast[DType.uint32](fs[j])))
    return Int(bitcast[DType.int64](h))


def scan_encode(
    cell: OpsProgram, reverse: Bool, has_lengths: Bool, key: Int, every: Int = 0,
) -> Tuple[List[Int], List[Float32]]:
    """The node's integer and float arguments (the layout in the module note), its fingerprint written in.
    `every` >= 1 (activation-ckpt): format 3, the scan checkpoints every `every` steps."""
    var got = program_words(cell)
    var ints = List[Int](capacity=SCAN_HEADER + len(got[0]) + 2)
    ints.append(SCAN_FORMAT_CKPT if every > 0 else SCAN_FORMAT)
    ints.append(0)
    ints.append(0)
    ints.append(cell.n_in - 1)
    ints.append(1 if reverse else 0)
    ints.append(1 if has_lengths else 0)
    ints.append(len(cell.params))
    ints.append(len(got[0]))
    if every > 0:
        ints.append(every)
    for i in range(len(got[0])):  # driver-loop: over the cell's description words, once per node made
        ints.append(Int(got[0][i]))
    ints.append(key)
    var fs = got[1].copy()
    var fp = scan_fingerprint(ints, fs)
    ints[1] = fp & 0xFFFFFFFF
    ints[2] = (fp >> 32) & 0xFFFFFFFF
    return (ints^, fs^)


def scan_decode(ints: List[Int], fs: List[Float32]) raises -> ScanCell:
    """The node's arguments read back, checked (format, counts, fingerprint, the cell's signature) and split per
    cell node: once per session for each distinct scan node (OpsSession keeps the result in its ScanCache)."""
    if len(ints) < SCAN_HEADER + 2 or (ints[0] != SCAN_FORMAT and ints[0] != SCAN_FORMAT_CKPT):
        raise Error("ValueError: ops: scan: the node's arguments are not a scan description")
    var n_state = ints[3]
    var reverse = ints[4] != 0
    var has_lengths = ints[5] != 0
    var n_params = ints[6]
    var n_words = ints[7]
    # activation-ckpt: format 3 holds the checkpoint segment length after n_words
    var head = SCAN_HEADER + (1 if ints[0] == SCAN_FORMAT_CKPT else 0)
    var every = ints[SCAN_HEADER] if ints[0] == SCAN_FORMAT_CKPT else 0
    if n_state < 0 or n_params < 0 or n_words < 1 or head + n_words + 1 != len(ints):
        raise Error("ValueError: ops: scan: a damaged scan description")
    if ints[0] == SCAN_FORMAT_CKPT and every < 1:
        raise Error("ValueError: ops: scan: a damaged scan description (checkpoint_every must be >= 1)")
    var fp = scan_fingerprint(ints, fs)
    if (fp & 0xFFFFFFFF) != (ints[1] & 0xFFFFFFFF) or ((fp >> 32) & 0xFFFFFFFF) != (ints[2] & 0xFFFFFFFF):
        raise Error("ValueError: ops: scan: a damaged scan description (its fingerprint does not match its words)")
    var words = List[Int64](capacity=n_words)
    for i in range(n_words):  # driver-loop: over the cell's description words, once per decoded node
        words.append(Int64(ints[head + i]))
    var cell = program_from_words(words, fs, "ops: scan: the cell")
    if cell.n_in != n_state + 1 or len(cell.params) != n_params:
        raise Error("ValueError: ops: scan: the description's counts disagree with its cell")
    scan_check_cell(cell)
    var got = ScanCell()
    _split_nodes(cell, got.node_ints, got.node_fs)
    for k in range(cell.n_nodes()):  # driver-loop: collect the nodes needing per-session bindings
        if cell.code[k] == _SCAN_DROPOUT_CODE and len(got.node_ints[k]) >= 3:
            got.drop_nodes.append(k)
            got.drop_seed.append((UInt64(got.node_ints[k][1]) << 32) | UInt64(got.node_ints[k][0] & 0xFFFFFFFF))
        elif cell.code[k] == OPS_CUSTOM_VJP and len(got.custom_ids) == 0:
            got.custom_ids = List[Int](length=cell.n_nodes(), fill=-1)
    got.cell = cell^
    got.n_state = n_state
    got.reverse = reverse
    got.has_lengths = has_lengths
    got.n_params = n_params
    got.every = every
    got.fingerprint = scan_key_of(ints)
    got.n_ints = len(ints)
    got.n_fs = len(fs)
    return got^


def scan_call_key(ints: List[Int], cell: ScanCell, n_handles: Int) raises -> Int:
    """Per call: the handle count against the decoded cell, and this call's RNG key (the node's last word)."""
    var want = 1 + cell.n_state + (1 if cell.has_lengths else 0) + cell.n_params
    if n_handles != want:
        raise Error("ValueError: ops: scan takes " + String(want) + " tensors (xs, the state tensors, "
                    + ("lengths, " if cell.has_lengths else "") + "the cell's parameters); got " + String(n_handles))
    var key = ints[len(ints) - 1]
    if key < 0:
        raise Error("ValueError: ops: scan: a damaged scan description (negative key)")
    return key
