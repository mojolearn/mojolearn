# SPDX-License-Identifier: Apache-2.0
# ACTIVATION CHECKPOINTING (lane activation-ckpt, Andrew 2026-11-25): train a larger model in a fixed memory budget.
#
# WHAT THE USER MARKS. A segment is a captured Program (inputs -> outputs, reading its parameters like a scan cell):
#   Session.checkpoint(segment, *inputs, params=None, counter=0, recompute=True)   session code OPS_CHECKPOINT = 119
# runs the segment where the call is, like torch.utils.checkpoint.checkpoint(fn, *inputs): in a recording session
# the forward keeps only the segment's inputs (and parameters) and its outputs; the backward recomputes the
# segment's forward from the inputs and then runs its backward. Its first user is Session.scan(...,
# checkpoint_every=k) (autograd/nn_scan.mojo format 3), which keeps the carried state at every k-th step and
# recomputes the k steps in between, last segment first. `recompute=False` runs the same nodes on the tape with the
# same output packing (the plain form, against which the checkpointed form is compared); an inference session runs
# the segment directly (there is no backward to recompute for).
#
# BITS: the checkpointed run equals the plain run bit for bit, on every column. The recompute replays the same
# nodes with the same kernels in the same order on the session's executor, and a dropout inside the segment
# replays the SAME mask: its seed is derived from (its captured seed, the call's key, its node, the step) by
# autograd/nn_scan.mojo scan_dropout_seed, never drawn again. The gradients are folded in the plain graph's order:
# the checkpoint is ONE custom tape node (tag NN_CKPT_TAG) at the segment's place in the graph, so every consumer
# outside the segment contributes before it and after it exactly as before; its rule writes each input's whole
# gradient in place (writes_total) and CONTINUES that gradient's fold: the recomputed leaf of an input points at the
# input's gradient (Tape.variable_carry) and starts touched when the tape says the gradient already held a
# contribution (Tape.edge_holds_gradient), so the segment's terms are added one by one in decreasing node order,
# never summed apart and added as one. Inside a scan the same holds across segments: a boundary state's gradient
# is carried into the earlier segment's recompute, and a step output that is also a boundary state has its own
# term (the stacked output's slice) folded in first, as the plain graph's stack node (the highest index) does.
# The outputs are seeded through the same packing node the plain form records (one output: the output itself;
# several: one (n, 1) column per output and one concat). An input that no output reaches is not an edge of the node
# (it gets no gradient from it, as in the plain graph); a recomputed leaf that a segment's outputs do not reach is
# released before that segment's backward (Tape.release_leaf_gradient), so nothing is zero-filled that the plain
# graph would not touch. Fused-epilogue choices the tape makes from a node's consumer (a linear's act' applied by
# its sole linear consumer, a pass-through gradient buffer borrowed) are value-identical by the tape's own
# invariant; the gradients a user reads of the parameters and the leaves, and every training step, are the plain
# run's words.
#
# MEMORY (the formula; values and gradient slots are float32 words). Plain: the tape keeps every node of the
# segment, its value and, for a node that needs a gradient, a gradient slot: A = sum over the segment's nodes of
# |v| (1 + g(v)), g(v) = 1 when v needs a gradient. Checkpointed: the tape keeps the outputs' copy, O = sum |y|
# (and its gradient slot), plus the record (addresses only); the inputs are kept either way. The recompute runs on
# the session's own scratch tape (`ckpt_tape`, reused by every checkpoint and every call, returned at close), whose
# peak is ONE segment's A. So for N checkpointed segments of equal A in one graph:
#     saved = N (A - 2 O) - A          (plain N A; checkpointed 2 N O + A)
# A scan over T steps with per-step nodes a (= the step's A: views, the cell's nodes, the lengths' where nodes),
# state words s and step output words y, checkpointed every k steps (P = ceil(T / k) segments):
#     plain        T a + |stack|
#     checkpointed (P - 1) s (the boundary states, values only) + |stack| (the stacked output, a copy)
#                  + k a (the scratch tape's peak: one segment's recompute) + 2 s (two boundary gradients, the
#                  session's `ckpt_gtape`)
#     saved        = (T - k) a - (P + 1) s, largest near k = sqrt(T s / a).
# The price is one more forward of every checkpointed node in the backward (and, here, one copy of each output),
# as torch.utils.checkpoint pays.
#
# WORDS (the node's integer and float arguments, so a captured program saves and loads the segment as data):
#   ints [CKPT_FORMAT, fingerprint low 32, fingerprint high 32, n_out, n_params, recompute, n_words,
#         the segment's program words ..., key]; floats the segment's floats.
# The fingerprint is autograd/nn_scan.mojo scan_fingerprint (FNV-1a over every word after the first three except the
# key, and every float); the session keeps each decoded segment under it (CkptCache: one dictionary read per call).
# The key is the node's last word, advanced by a training replay like a dropout's counter
# (autograd/program_replay_metadata.mojo `random`), so Program.fit draws new masks every minibatch; eagerly it is
# Session.checkpoint's `counter=`.
#
# The handles of the node: the segment's inputs in order, then one tensor per parameter (Program.params order).
# Refused at bind, by name: a segment that holds a scan, a custom operation, a checkpoint, a block step or a
# generation op (the nested runner has no branch for them, or they keep host values); an output that is an input
# or a parameter returned unchanged; declared state (Session.state) as a parameter; an integer (ids) output.
from std.sys.compile import is_defined
from autograd.program import (
    OpsProgram, program_words, program_from_words, shape_text, same_shape, PROGRAM_IN_FLOAT, PROGRAM_PARAM_WEIGHT,
)
from autograd.rankn import ShapeN
from autograd.nn_cbw import _split_nodes
from autograd.nn_scan import scan_fingerprint, scan_key_of, scan_host_code_name

comptime OPS_CHECKPOINT = 119
#: the tape tag of a checkpointed segment's node (CUSTOM + NN_CKPT_TAG), clear of every rule set's tags
comptime NN_CKPT_TAG = 6880
comptime CKPT_FORMAT = 1
comptime CKPT_HEADER = 7
#: autograd/ops_api.mojo OPS_DROPOUT (not imported: ops_api imports this module); ints [seed lo, seed hi, counter]
comptime _CKPT_DROPOUT_CODE = 21
#: record kinds: a Session.checkpoint segment, a scan with checkpoint_every
comptime CKPT_KIND_SEGMENT = 0
comptime CKPT_KIND_SCAN = 1

# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-nn_ckpt_*, lane ops-composed-cases
# case file composed-checkpoint.json). Off in every build; each arm's patch switches one on. Each faults the
# RECOMPUTE only (the backward of a checkpointed segment or of a checkpointed scan), so the forward and the plain
# form stay and the checkpointed gradients move away from the plain ones:
#   FRESH_MASK: a dropout inside the recomputed segment draws its mask from the next key (a fresh draw);
#   SKIP_NODE: the recompute skips one node (in a scan, one node of every recomputed step): the first node whose
#   output has its first operand's shape is replaced by that operand.
# The plain name faults the device executor only, _HOST the host executor only.
comptime CKPT_SABOTAGE_FRESH_MASK = is_defined["MOJOLEARN_NN_CKPT_SABOTAGE_FRESH_MASK"]()
comptime CKPT_SABOTAGE_FRESH_MASK_HOST = is_defined["MOJOLEARN_NN_CKPT_SABOTAGE_FRESH_MASK_HOST"]()
comptime CKPT_SABOTAGE_SKIP_NODE = is_defined["MOJOLEARN_NN_CKPT_SABOTAGE_SKIP_NODE"]()
comptime CKPT_SABOTAGE_SKIP_NODE_HOST = is_defined["MOJOLEARN_NN_CKPT_SABOTAGE_SKIP_NODE_HOST"]()


struct CkptSeg(Movable):
    """One decoded checkpoint node, kept by its session for every later call with the same fingerprint: the
    segment, its counts, each node's arguments split once, and its dropout nodes with their captured seeds."""
    var program: OpsProgram
    var n_out: Int
    var n_params: Int
    var recompute: Bool
    var fingerprint: Int
    var n_ints: Int
    var n_fs: Int
    var node_ints: List[List[Int]]
    var node_fs: List[List[Float32]]
    var drop_nodes: List[Int]
    var drop_seed: List[UInt64]

    def __init__(out self):
        self.program = OpsProgram()
        self.n_out = 0
        self.n_params = 0
        self.recompute = True
        self.fingerprint = 0
        self.n_ints = 0
        self.n_fs = 0
        self.node_ints = List[List[Int]]()
        self.node_fs = List[List[Float32]]()
        self.drop_nodes = List[Int]()
        self.drop_seed = List[UInt64]()


struct CkptRecord(Copyable, Movable):
    """One checkpointed node of the recorded graph. The operands are DISTINCT tape tensors (a tensor passed twice
    is one leaf in the recompute, as it is one tensor in the plain graph): their value addresses (tape storage or
    uploaded tensors, alive until the graph resets), shapes, kinds and edge of the custom node (-1: not an edge: no
    gradient, or no output reaches it). `role` maps the node's handles (in the node's order) to operands."""
    var kind: Int
    #: the decoded segment (CkptState.segs) or scan cell (OpsSession.scans.cells) index
    var cache: Int
    #: the tape index of the custom node (a later graph after a tape reset is told apart by it)
    var node: Int
    var key: Int
    var op_fp: List[Int]
    var op_shape: List[ShapeN]
    var op_ids: List[Bool]
    var op_edge: List[Int]
    var role: List[Int]
    #: the outputs' total words and the output shape of the node
    var total: Int
    # scan only (CKPT_KIND_SCAN; autograd/ops_api.mojo `_scan_ckpt_run`)
    #: address of the boundary states: segment p's end states (p = 0 .. n_seg - 2), n_bd words each
    var bstore: Int
    var n_bd: Int
    #: per distinct boundary tensor b: the first state k whose output it is, and its word offset in a row
    var bd_k: List[Int]
    var bd_off: List[Int]
    #: per state k: its distinct boundary tensor
    var state_bd: List[Int]
    #: per boundary p and tensor b (row-major): whether the end state needed a gradient in the forward
    var bd_req: List[Bool]
    #: the distinct boundary tensor the step output is (the cell returns it as a state too), or -1
    var y_alias: Int
    var steps: Int
    var every: Int
    var n_seg: Int

    def __init__(out self):
        self.kind = CKPT_KIND_SEGMENT
        self.cache = 0
        self.node = -1
        self.key = 0
        self.op_fp = List[Int]()
        self.op_shape = List[ShapeN]()
        self.op_ids = List[Bool]()
        self.op_edge = List[Int]()
        self.role = List[Int]()
        self.total = 0
        self.bstore = 0
        self.n_bd = 0
        self.bd_k = List[Int]()
        self.bd_off = List[Int]()
        self.state_bd = List[Int]()
        self.bd_req = List[Bool]()
        self.y_alias = -1
        self.steps = 0
        self.every = 0
        self.n_seg = 0


struct CkptState(Movable):
    """A session's decoded checkpoint segments (by fingerprint) and the checkpointed nodes of the recorded graph."""
    var segs: List[CkptSeg]
    var by_key: Dict[Int, Int]
    var records: List[CkptRecord]

    def __init__(out self):
        self.segs = List[CkptSeg]()
        self.by_key = Dict[Int, Int]()
        self.records = List[CkptRecord]()

    def clear_records(mut self):
        self.records.clear()

    def clear(mut self):
        self.segs.clear()
        self.by_key.clear()
        self.records.clear()

    def find(self, ints: List[Int], fs: List[Float32]) raises -> Int:
        """The index of the decoded segment these node words name, or -1 (one dictionary read, no parse)."""
        if len(ints) < CKPT_HEADER + 2 or ints[0] != CKPT_FORMAT:
            raise Error("ValueError: ops: checkpoint: the node's arguments are not a checkpoint description")
        var found = self.by_key.find(scan_key_of(ints))
        if found:
            var i = found.value()
            if self.segs[i].n_ints == len(ints) and self.segs[i].n_fs == len(fs):
                return i
        return -1

    def add(mut self, var seg: CkptSeg) -> Int:
        var key = seg.fingerprint
        self.segs.append(seg^)
        self.by_key[key] = len(self.segs) - 1
        return len(self.segs) - 1

    def admit(mut self, node: Int) -> Int:
        """The record index the next record takes at tape index `node`: a node index at or below the last
        record's starts a new graph (a reset tape restarts its indices), whose records replace the old ones."""
        var last = len(self.records) - 1
        if last >= 0 and node <= self.records[last].node:
            self.records.clear()
        return len(self.records)


def ckpt_check_program(p: OpsProgram) raises:
    """The segment's own signature: at least one output, each computed by the segment (not an input or a
    parameter returned unchanged), weight parameters only, and only operations the nested runner has."""
    if len(p.outs) < 1:
        raise Error("ValueError: checkpoint: the segment returns no output")
    for k in range(len(p.outs)):  # driver-loop: over the segment's outputs (metadata)
        var rf = p.outs[k]
        if rf < 0:
            raise Error("ValueError: checkpoint: the segment's output " + String(k) + " is a parameter returned"
                        + " unchanged; a checkpointed output must be computed by the segment")
        if rf < p.n_in:
            raise Error("ValueError: checkpoint: the segment's output " + String(k) + " is its input " + String(rf)
                        + " returned unchanged; a checkpointed output must be computed by the segment")
    for j in range(len(p.params)):  # driver-loop: over the segment's parameters (metadata)
        if p.param_kind(j) != PROGRAM_PARAM_WEIGHT:
            raise Error("ValueError: checkpoint: the segment reads declared state (Session.state), which it would"
                        + " write again when the backward recomputes it; a segment reads weights only")
    for k in range(p.n_nodes()):  # driver-loop: over the segment's nodes (codes)
        var name = scan_host_code_name(p.code[k])
        if name.byte_length() > 0:
            raise Error("ValueError: checkpoint: the segment holds " + name + " (node " + String(k)
                        + "); a checkpointed segment is built from operations on device values (a scan"
                        + " checkpoints its own steps with checkpoint_every)")


def ckpt_encode(p: OpsProgram, recompute: Bool, key: Int) -> Tuple[List[Int], List[Float32]]:
    """The node's integer and float arguments (the module note's layout), its fingerprint written in."""
    var got = program_words(p)
    var ints = List[Int](capacity=CKPT_HEADER + len(got[0]) + 1)
    ints.append(CKPT_FORMAT)
    ints.append(0)
    ints.append(0)
    ints.append(len(p.outs))
    ints.append(len(p.params))
    ints.append(1 if recompute else 0)
    ints.append(len(got[0]))
    for i in range(len(got[0])):  # driver-loop: over the segment's description words, once per node made
        ints.append(Int(got[0][i]))
    ints.append(key)
    var fs = got[1].copy()
    var fp = scan_fingerprint(ints, fs)
    ints[1] = fp & 0xFFFFFFFF
    ints[2] = (fp >> 32) & 0xFFFFFFFF
    return (ints^, fs^)


def ckpt_decode(ints: List[Int], fs: List[Float32]) raises -> CkptSeg:
    """The node's arguments read back, checked (format, counts, fingerprint, the segment's signature) and split
    per node: once per session for each distinct checkpoint node (OpsSession keeps it in CkptState)."""
    if len(ints) < CKPT_HEADER + 2 or ints[0] != CKPT_FORMAT:
        raise Error("ValueError: ops: checkpoint: the node's arguments are not a checkpoint description")
    var n_out = ints[3]
    var n_params = ints[4]
    var recompute = ints[5] != 0
    var n_words = ints[6]
    if n_out < 1 or n_params < 0 or n_words < 1 or CKPT_HEADER + n_words + 1 != len(ints):
        raise Error("ValueError: ops: checkpoint: a damaged checkpoint description")
    var fp = scan_fingerprint(ints, fs)
    if (fp & 0xFFFFFFFF) != (ints[1] & 0xFFFFFFFF) or ((fp >> 32) & 0xFFFFFFFF) != (ints[2] & 0xFFFFFFFF):
        raise Error("ValueError: ops: checkpoint: a damaged checkpoint description (its fingerprint does not match"
                    + " its words)")
    var words = List[Int64](capacity=n_words)
    for i in range(n_words):  # driver-loop: over the segment's description words, once per decoded node
        words.append(Int64(ints[CKPT_HEADER + i]))
    var p = program_from_words(words, fs, "ops: checkpoint: the segment")
    if len(p.outs) != n_out or len(p.params) != n_params:
        raise Error("ValueError: ops: checkpoint: the description's counts disagree with its segment")
    ckpt_check_program(p)
    var got = CkptSeg()
    _split_nodes(p, got.node_ints, got.node_fs)
    for k in range(p.n_nodes()):  # driver-loop: over the segment's nodes (dropout ones), once per decoded node
        if p.code[k] == _CKPT_DROPOUT_CODE and len(got.node_ints[k]) >= 3:
            got.drop_nodes.append(k)
            got.drop_seed.append((UInt64(got.node_ints[k][1]) << 32) | UInt64(got.node_ints[k][0] & 0xFFFFFFFF))
    got.program = p^
    got.n_out = n_out
    got.n_params = n_params
    got.recompute = recompute
    got.fingerprint = scan_key_of(ints)
    got.n_ints = len(ints)
    got.n_fs = len(fs)
    return got^


def ckpt_call_key(ints: List[Int]) raises -> Int:
    """This call's RNG key (the node's last word)."""
    var key = ints[len(ints) - 1]
    if key < 0:
        raise Error("ValueError: ops: checkpoint: a damaged checkpoint description (negative key)")
    return key


def ckpt_check_binding(
    p: OpsProgram, shapes: List[ShapeN], ids: List[Bool],
) raises:
    """The node's operands (the inputs, then the parameters) against the segment: counts, shapes and kinds,
    checked once per call before anything runs."""
    var n_in = p.n_in
    var want = n_in + len(p.params)
    if len(shapes) != want:
        raise Error("ValueError: checkpoint: the segment takes " + String(n_in) + " inputs and "
                    + String(len(p.params)) + " parameters; got " + String(len(shapes)) + " tensors")
    for k in range(n_in):  # driver-loop: over the segment's inputs (metadata)
        if not same_shape(shapes[k], p.in_shapes[k]):
            raise Error("ValueError: checkpoint: input " + String(k) + " is " + shape_text(shapes[k])
                        + " but the segment's input " + String(k) + " is " + shape_text(p.in_shapes[k]))
        var want_ids = p.in_kinds[k] != PROGRAM_IN_FLOAT
        if ids[k] != want_ids:
            if want_ids:
                raise Error("ValueError: checkpoint: input " + String(k) + " must be an ids tensor (Session.ids)")
            raise Error("ValueError: checkpoint: input " + String(k) + " must be a float tensor, not ids")
    for j in range(len(p.params)):  # driver-loop: over the bound parameters (metadata)
        if not same_shape(shapes[n_in + j], p.param_shapes[j]):
            raise Error("ValueError: checkpoint: params[" + String(j) + "] is " + shape_text(shapes[n_in + j])
                        + " but the segment's parameter " + String(j) + " is " + shape_text(p.param_shapes[j]))
