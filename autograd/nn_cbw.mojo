# SPDX-License-Identifier: Apache-2.0
"""USER-DEFINED BACKWARD RULES FOR A CAPTURED OPERATION (sub-lane nn-blocks-cbw,
2026-11-25; Andrew's block 5 contract in briefs-20261122/nn-blocks-cbw.md).

A user packages two captured programs built from existing session operations:

  forward   (x_0 .. x_{n-1}) -> (y_0 .. y_{m-1}, extra_0 ..)   the op's outputs, then
            any further tensors it wants to keep for the backward
  backward  (saved_0 .. saved_{s-1}, g_0 .. g_{m-1}) -> one gradient per float input
            of the forward, in input order (an ids input takes no gradient)

`saved` names, by index into the forward's outputs, the tensors the backward
receives first (MyTanh: forward x -> y, saved [0], backward (y, g) -> g * (1 - y * y)).
It is a vector-Jacobian product: g_k is the incoming gradient of output k from
the rest of the model, never assumed to be 1.

This module holds the data and the checks; the session half (running the
forward, recording the tape node, the BackwardRules wrapper `CbwRules` whose
rule runs the backward program) lives in autograd/ops_api.mojo between the
`nn-blocks-cbw begin/end` markers, because it calls the session's own operation
dispatch (`OpsSession._program_op`), which a module imported by ops_api cannot
reach without an import cycle.

BINDING happens once, in `cbw_bind` (Session.custom_op, or the first replay of a
loaded program that holds the operation): counts, shapes and kinds of the inputs,
outputs, saved tensors and gradients are checked against each other and a
mismatch is refused by name. The bound operation is kept by the session under
the fingerprint of its words; a call looks it up with one dictionary read.

WORDS (the integer and float arguments of one OPS_CUSTOM_VJP node, so a program
that holds the operation saves and loads both graphs with no new record field):
  ints [NN_CBW_FORMAT, key low 32 bits, key high 32 bits, n_out, n_saved,
        saved..., n_fwd_words, forward program_words..., n_fwd_floats,
        n_bwd_words, backward program_words..., n_bwd_floats]
  floats  the forward program's float words, then the backward's
The key is FNV-1a over every word after the first three and every float's bits.

Same bits on every vendor: the forward and the backward run the session's own
operations (each already identical across vendors and the host column); the
gradient words are copied into the tape's destinations with OP_COPY; the tape
accumulates contributions in its canonical order (decreasing consumer index,
increasing edge). No new kernel, no vendor branch, no size rule.

Launches. Forward: the forward program's operations (plus one OP_COPY per output
when the op returns several outputs, or returns an input unchanged). Backward:
the backward program's operations plus one OP_COPY per differentiable input.

First-version limits: the backward is built from existing operations (a new
device kernel needs a compiled Mojo extension); first order only (a recorded
backward, create_graph=True, through a custom rule is refused by name); neither
program may read session tensors as parameters (pass them as inputs, which also
gives them gradients).
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from autograd.program import (
    OpsProgram, PROGRAM_IN_FLOAT, PROGRAM_IN_IDS, program_words, program_from_words, same_shape, shape_text,
)
from autograd.rankn import ShapeN

# The Session operation code (autograd/ops_api.mojo OPS_*), the tape tag of the
# recorded node (CUSTOM + NN_CBW_TAG; sub-lane cbw's kernel opcode range
# 6860..6879, so no other rule set claims it) and the words' format.
comptime OPS_CUSTOM_VJP = 126
comptime NN_CBW_TAG = 6860
comptime NN_CBW_FORMAT = 1

# Fault injection (verify's sabotage arms, test builds only, never a product
# build). DROP_PATH: the rule claims to write each input's whole gradient
# (writes_total), so its copy overwrites what another consumer of the same
# input already contributed (the second path's term is lost). SEED_ONE: the
# backward program receives ones as the incoming gradient instead of the
# upstream words (a rule that assumes 1). Both are in the shared session code,
# so one define faults the host column and the device column alike.
comptime NN_CBW_SABOTAGE_DROP_PATH = is_defined["MOJOLEARN_NN_CBW_SABOTAGE_DROP_PATH"]()
comptime NN_CBW_SABOTAGE_SEED_ONE = is_defined["MOJOLEARN_NN_CBW_SABOTAGE_SEED_ONE"]()


struct CbwOp(Copyable, Movable):
    """One bound custom operation: both programs, the saved outputs, the
    returned output count, the forward input of each gradient, the node words
    and each program's per-node arguments split once at binding."""
    var forward: OpsProgram
    var backward: OpsProgram
    var saved: List[Int]
    var n_out: Int
    var grad_inputs: List[Int]
    var words: List[Int]
    var floats: List[Float32]
    var key: Int
    var f_ints: List[List[Int]]
    var f_fs: List[List[Float32]]
    var b_ints: List[List[Int]]
    var b_fs: List[List[Float32]]

    def __init__(out self):
        self.forward = OpsProgram()
        self.backward = OpsProgram()
        self.saved = List[Int]()
        self.n_out = 0
        self.grad_inputs = List[Int]()
        self.words = List[Int]()
        self.floats = List[Float32]()
        self.key = 0
        self.f_ints = List[List[Int]]()
        self.f_fs = List[List[Float32]]()
        self.b_ints = List[List[Int]]()
        self.b_fs = List[List[Float32]]()


struct CbwRecord(Copyable, Movable):
    """One custom node of the recorded graph: the bound op, the tape index of
    its node (a later graph after a tape reset is told apart by it), the saved
    tensors' storage (tape storage, alive until the tape resets), their shapes
    and kinds, and the shapes of the outputs and inputs at this call."""
    var op: Int
    var node: Int
    var saved: List[Int]
    var saved_shapes: List[ShapeN]
    var saved_words: List[Bool]
    var out_shapes: List[ShapeN]
    var out_offsets: List[Int]
    var in_shapes: List[ShapeN]

    def __init__(out self):
        self.op = 0
        self.node = -1
        self.saved = List[Int]()
        self.saved_shapes = List[ShapeN]()
        self.saved_words = List[Bool]()
        self.out_shapes = List[ShapeN]()
        self.out_offsets = List[Int]()
        self.in_shapes = List[ShapeN]()


struct CbwState(Movable):
    """A session's bound custom operations (id = index + 1), their index by
    key, the custom nodes of the current recorded graph, and the output shapes
    of the last forward (for the views of a several-output call)."""
    var ops: List[CbwOp]
    var by_key: Dict[Int, Int]
    var records: List[CbwRecord]
    var last_out_shapes: List[ShapeN]

    def __init__(out self):
        self.ops = List[CbwOp]()
        self.by_key = Dict[Int, Int]()
        self.records = List[CbwRecord]()
        self.last_out_shapes = List[ShapeN]()

    def clear_records(mut self):
        self.records.clear()

    def clear(mut self):
        self.ops.clear()
        self.by_key.clear()
        self.records.clear()
        self.last_out_shapes.clear()

    def holds_rules(self) -> Bool:
        """The recorded graph may hold a custom node (a backward must take the
        rule wrapper). A graph without one takes the wrapper identically."""
        return len(self.records) != 0

    def find(self, ints: List[Int]) raises -> Int:
        """The index of the bound op these node words name, or -1."""
        if len(ints) < 5 or ints[0] != NN_CBW_FORMAT:
            raise Error("ValueError: custom: a damaged custom operation record")
        var key = cbw_key_of(ints)
        var found = self.by_key.find(key)
        if found:
            var i = found.value()
            if len(self.ops[i].words) == len(ints):
                return i
        return -1

    def add(mut self, var op: CbwOp) -> Int:
        """Keep a bound op; returns its index (an op bound twice is kept once)."""
        var found = self.by_key.find(op.key)
        if found:
            var i = found.value()
            if len(self.ops[i].words) == len(op.words):
                return i
        var key = op.key
        self.ops.append(op^)
        self.by_key[key] = len(self.ops) - 1
        return len(self.ops) - 1


def cbw_key_of(ints: List[Int]) -> Int:
    """The key node words carry (words 1 and 2)."""
    return (ints[1] & 0xFFFFFFFF) | ((ints[2] & 0xFFFFFFFF) << 32)


def _fnv(h: UInt64, word: UInt64) -> UInt64:
    var x = h
    for b in range(8):  # small-loop(8 bytes of one word): fingerprint, no data
        x = (x ^ ((word >> UInt64(8 * b)) & UInt64(0xFF))) * UInt64(0x100000001B3)
    return x


def _split_nodes(p: OpsProgram, mut ints: List[List[Int]], mut fs: List[List[Float32]]):
    """Each node's integer and float arguments as its own list, once at binding."""
    ints.clear()
    fs.clear()
    for k in range(p.n_nodes()):  # driver-loop: over program nodes, once per binding
        var ki = List[Int](capacity=p.i_at[k + 1] - p.i_at[k])
        for j in range(p.i_at[k], p.i_at[k + 1]):  # small-loop(ints: a node's integer arguments): description words
            ki.append(p.ints[j])
        var kf = List[Float32](capacity=p.f_at[k + 1] - p.f_at[k])
        for j in range(p.f_at[k], p.f_at[k + 1]):  # small-loop(fs: a node's float arguments): description words
            kf.append(p.fs[j])
        ints.append(ki^)
        fs.append(kf^)


def _check_program(p: OpsProgram, which: String) raises:
    if len(p.params) != 0:
        raise Error("ValueError: custom_op: the " + which + " program reads " + String(len(p.params))
                    + " session tensors as parameters; capture it with every tensor it reads as an input "
                    + "(make constants with full)")
    for k in range(len(p.outs)):  # driver-loop: over program outputs
        if p.outs[k] < 0:
            raise Error("ValueError: custom_op: the " + which + " program returns a parameter")


def cbw_bind(var forward: OpsProgram, var backward: OpsProgram, saved: List[Int], n_out: Int) raises -> CbwOp:
    """Check the two programs' signatures against each other and bind them.

    Refusals (ValueError, by name): parameters in either program; an output
    count outside 1 .. the forward's outputs; a saved index that names no
    forward output; a forward output neither returned nor saved; the backward's
    input count, each saved input's shape, each incoming gradient's kind and
    shape; the backward's output count and each gradient's shape against the
    forward's float inputs."""
    _check_program(forward, "forward")
    _check_program(backward, "backward")
    var n_fwd_out = len(forward.outs)
    if n_out < 1 or n_out > n_fwd_out:
        raise Error("ValueError: custom_op: outputs must be between 1 and the forward program's "
                    + String(n_fwd_out) + " outputs, got " + String(n_out))
    var named = List[Bool](length=n_fwd_out, fill=False)
    for j in range(len(saved)):  # driver-loop: over saved tensors
        if saved[j] < 0 or saved[j] >= n_fwd_out:
            raise Error("ValueError: custom_op: saved entry " + String(j) + " names forward output "
                        + String(saved[j]) + ", which does not exist (the forward has " + String(n_fwd_out) + " outputs)")
        named[saved[j]] = True
    for k in range(n_out, n_fwd_out):  # driver-loop: over forward outputs past the returned ones
        if not named[k]:
            raise Error("ValueError: custom_op: forward output " + String(k) + " is neither returned nor saved")
    var n_saved = len(saved)
    if backward.n_in != n_saved + n_out:
        raise Error("ValueError: custom_op: the backward program takes " + String(backward.n_in)
                    + " inputs; it must take " + String(n_saved + n_out) + " (the " + String(n_saved)
                    + " saved tensors, then one incoming gradient per output)")
    for j in range(n_saved):  # driver-loop: over saved tensors
        if not same_shape(backward.in_shapes[j], forward.out_shapes[saved[j]]):
            raise Error("ValueError: custom_op: backward input " + String(j) + " (saved tensor " + String(j)
                        + ", forward output " + String(saved[j]) + ") has shape " + shape_text(backward.in_shapes[j])
                        + "; the forward saves shape " + shape_text(forward.out_shapes[saved[j]]))
    for k in range(n_out):  # driver-loop: over returned outputs
        var i = n_saved + k
        if backward.in_kinds[i] != PROGRAM_IN_FLOAT:
            raise Error("ValueError: custom_op: backward input " + String(i) + " (the gradient of output "
                        + String(k) + ") must be a float input, not ids")
        if not same_shape(backward.in_shapes[i], forward.out_shapes[k]):
            raise Error("ValueError: custom_op: backward input " + String(i) + " (the gradient of output "
                        + String(k) + ") has shape " + shape_text(backward.in_shapes[i]) + "; output "
                        + String(k) + " has shape " + shape_text(forward.out_shapes[k]))
    var grad_inputs = List[Int]()
    for i in range(forward.n_in):  # driver-loop: over forward inputs
        if forward.in_kinds[i] == PROGRAM_IN_FLOAT:
            grad_inputs.append(i)
    if len(grad_inputs) == 0:
        raise Error("ValueError: custom_op: the forward program has no float input to differentiate")
    if len(backward.outs) != len(grad_inputs):
        raise Error("ValueError: custom_op: the backward program returns " + String(len(backward.outs))
                    + " gradients; the forward has " + String(len(grad_inputs)) + " float inputs")
    for j in range(len(grad_inputs)):  # driver-loop: over gradients
        var i = grad_inputs[j]
        if not same_shape(backward.out_shapes[j], forward.in_shapes[i]):
            raise Error("ValueError: custom_op: gradient " + String(j) + " has shape "
                        + shape_text(backward.out_shapes[j]) + "; forward input " + String(i)
                        + " has shape " + shape_text(forward.in_shapes[i]))
    # the node words (module docstring)
    var fw = program_words(forward)
    var bw = program_words(backward)
    var words = List[Int]()
    words.append(NN_CBW_FORMAT)
    words.append(0)
    words.append(0)
    words.append(n_out)
    words.append(n_saved)
    for j in range(n_saved):  # driver-loop: over saved tensors
        words.append(saved[j])
    words.append(len(fw[0]))
    for j in range(len(fw[0])):  # driver-loop: over forward description words
        words.append(Int(fw[0][j]))
    words.append(len(fw[1]))
    words.append(len(bw[0]))
    for j in range(len(bw[0])):  # driver-loop: over backward description words
        words.append(Int(bw[0][j]))
    words.append(len(bw[1]))
    var floats = List[Float32](capacity=len(fw[1]) + len(bw[1]))
    for j in range(len(fw[1])):  # driver-loop: over forward float words
        floats.append(fw[1][j])
    for j in range(len(bw[1])):  # driver-loop: over backward float words
        floats.append(bw[1][j])
    var h = UInt64(0xCBF29CE484222325)
    for j in range(3, len(words)):  # driver-loop: over node words, once per binding
        h = _fnv(h, bitcast[DType.uint64](Int64(words[j])))
    for j in range(len(floats)):  # driver-loop: over float words, once per binding
        h = _fnv(h, UInt64(bitcast[DType.uint32](floats[j])))
    var key = Int(bitcast[DType.int64](h))
    words[1] = key & 0xFFFFFFFF
    words[2] = (key >> 32) & 0xFFFFFFFF
    var op = CbwOp()
    _split_nodes(forward, op.f_ints, op.f_fs)
    _split_nodes(backward, op.b_ints, op.b_fs)
    op.forward = forward^
    op.backward = backward^
    op.saved = saved.copy()
    op.n_out = n_out
    op.grad_inputs = grad_inputs^
    op.key = cbw_key_of(words)
    op.words = words^
    op.floats = floats^
    return op^


struct _Words(Movable):
    var at: Int

    def __init__(out self):
        self.at = 0

    def take(mut self, ints: List[Int]) raises -> Int:
        if self.at >= len(ints):
            raise Error("ValueError: custom: a damaged custom operation record (it ends early)")
        var v = ints[self.at]
        self.at += 1
        return v


def cbw_from_words(ints: List[Int], fs: List[Float32]) raises -> CbwOp:
    """The op a node's words describe, bound again (the same checks as
    Session.custom_op), refused when the words do not give back their key."""
    if len(ints) < 5 or ints[0] != NN_CBW_FORMAT:
        raise Error("ValueError: custom: a damaged custom operation record")
    var r = _Words()
    r.at = 3
    var n_out = r.take(ints)
    var n_saved = r.take(ints)
    if n_saved < 0 or n_saved > len(ints):
        raise Error("ValueError: custom: a damaged custom operation record (saved count)")
    var saved = List[Int](capacity=n_saved)
    for _ in range(n_saved):  # driver-loop: over saved tensors
        saved.append(r.take(ints))
    var nfw = r.take(ints)
    if nfw < 1 or nfw > len(ints):
        raise Error("ValueError: custom: a damaged custom operation record (forward words)")
    var fwi = List[Int64](capacity=nfw)
    for _ in range(nfw):  # driver-loop: over forward description words
        fwi.append(Int64(r.take(ints)))
    var nff = r.take(ints)
    var nbw = r.take(ints)
    if nbw < 1 or nbw > len(ints):
        raise Error("ValueError: custom: a damaged custom operation record (backward words)")
    var bwi = List[Int64](capacity=nbw)
    for _ in range(nbw):  # driver-loop: over backward description words
        bwi.append(Int64(r.take(ints)))
    var nbf = r.take(ints)
    if r.at != len(ints) or nff < 0 or nbf < 0 or nff + nbf != len(fs):
        raise Error("ValueError: custom: a damaged custom operation record (word counts)")
    var fwf = List[Float32](capacity=nff)
    for j in range(nff):  # driver-loop: over forward float words
        fwf.append(fs[j])
    var bwf = List[Float32](capacity=nbf)
    for j in range(nbf):  # driver-loop: over backward float words
        bwf.append(fs[nff + j])
    var forward = program_from_words(fwi, fwf, "custom: forward program")
    var backward = program_from_words(bwi, bwf, "custom: backward program")
    var op = cbw_bind(forward^, backward^, saved, n_out)
    if op.key != cbw_key_of(ints):
        raise Error("ValueError: custom: a damaged custom operation record (its key does not match its words)")
    return op^


def cbw_program_holds(p: OpsProgram) -> Bool:
    """Whether a captured program holds a custom operation node."""
    for k in range(p.n_nodes()):  # driver-loop: over program nodes
        if p.code[k] == OPS_CUSTOM_VJP:
            return True
    return False
