# SPDX-License-Identifier: Apache-2.0
"""CAPTURED PROGRAMS of an operation session (lane ag-program, 2026-10-19).

An inference session (`OpsSession`, record off) logs every operation call
it runs (`OpsLog`: the operation code, the input handles, the integer and
float arguments and the result handle). `OpsSession.capture(inputs,
outputs)` walks that log back from the named outputs to the named inputs
and keeps the operations the outputs depend on, in their order, as an
`OpsProgram`:

  values   0 .. n_in - 1        the program's inputs (their recorded shapes)
           n_in + k             the result of kept operation k
           -(j + 1)             parameter j: a session tensor made by
                                `upload` (a trainable weight, or any array
                                not named as an input), read IN PLACE at
                                each run, so an optimizer step between runs
                                is seen by the next run
  nodes    code, value refs of its inputs, its integer and float arguments
  outputs  value refs and their recorded shapes

`OpsSession.run_program` replays the nodes on the session's executor in
ONE binding call: the inputs are uploaded through their CastIn
descriptors into a scratch tape the session keeps for replays (its storage
reused run after run), every node runs through the same `OpsSession` entry
the eager call ran (so the same kernels with the same arguments: the same
words), and the outputs are downloaded into the caller's arrays. The host
loop over nodes is a driver loop over model structure; every element of
work is on the executor.

Index arguments (lane ag-program-embedding): `embedding` and `take`
called with an ids TENSOR (a session tensor of int32 words: one uploaded
with `OpsSession.upload_ids`, or an input of the program) record the ids as
a value ref like any other input, so a program holds them; each input has a
kind (PROGRAM_IN_FLOAT: cast to float32 at each run; PROGRAM_IN_IDS: cast
to int32 words, as `upload_ids` casts them). Not captured: the same
operations called with a caller ARRAY of ids, whose integer arguments carry
the array's address, gone by the time the program runs.

The description round-trips through integer and float words
(`program_words` / `program_from_words`), the extension record of a
CKPT_SECTION_PROGRAM checkpoint (autograd/model_checkpoint.mojo), so a
program saved on one install loads and runs on any other, the CPU-only
install included.
"""
from autograd.rankn import RN_MAX_RANK, ShapeN
from autograd.rankn_api import rn_dims, rn_shape

# The description's record version (the first extension word of a saved program).
# Version 2 (lane ag-program-embedding) adds one kind word per input after
# the input shapes; a program whose inputs are all PROGRAM_IN_FLOAT is still
# written as version 1, word for word what it was, and both versions load.
comptime PROGRAM_RECORD_VERSION_1 = 1
comptime PROGRAM_RECORD_VERSION = 2
# The kind of a program input: float32 values, or int32 words (embedding and
# take ids), each cast from the caller's array at every run.
comptime PROGRAM_IN_FLOAT = 0
comptime PROGRAM_IN_IDS = 1
# The largest operation code a program may hold (autograd/ops_api.mojo OPS_*):
# OPS_PAD 73, the last of the import family (lane ag-import-kernels), so a
# saved imported model that uses those operations loads again. Code 47 is
# reserved and refused below.
comptime PROGRAM_MAX_CODE = 73


struct OpsLog(Movable):
    """The operation calls of an inference session since its last reset,
    flattened: entry k's input handles are hs[h_at[k] : h_at[k + 1]], its
    integers ints[i_at[k] : i_at[k + 1]], its floats fs[f_at[k] : f_at[k + 1]]."""
    var code: List[Int]
    var res: List[Int]
    var h_at: List[Int]
    var hs: List[Int]
    var i_at: List[Int]
    var ints: List[Int]
    var f_at: List[Int]
    var fs: List[Float32]

    def __init__(out self):
        self.code = List[Int]()
        self.res = List[Int]()
        self.h_at = List[Int]()
        self.hs = List[Int]()
        self.i_at = List[Int]()
        self.ints = List[Int]()
        self.f_at = List[Int]()
        self.fs = List[Float32]()
        self.h_at.append(0)
        self.i_at.append(0)
        self.f_at.append(0)

    def count(self) -> Int:
        return len(self.code)

    def clear(mut self):
        self.code.clear()
        self.res.clear()
        self.h_at.clear()
        self.hs.clear()
        self.i_at.clear()
        self.ints.clear()
        self.f_at.clear()
        self.fs.clear()
        self.h_at.append(0)
        self.i_at.append(0)
        self.f_at.append(0)

    def add(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], res: Int):
        self.code.append(code)
        self.res.append(res)
        for k in range(len(hs)):  # small-loop(hs: an op's input handles): argument words, no data
            self.hs.append(hs[k])
        for k in range(len(ints)):  # small-loop(ints: an op's integer arguments): argument words, no data
            self.ints.append(ints[k])
        for k in range(len(fs)):  # small-loop(fs: an op's float arguments): argument words, no data
            self.fs.append(fs[k])
        self.h_at.append(len(self.hs))
        self.i_at.append(len(self.ints))
        self.f_at.append(len(self.fs))


struct OpsProgram(Copyable, Movable):
    """A captured inference program (the value encoding is the module
    docstring's)."""
    var live: Bool
    var n_in: Int
    var in_shapes: List[ShapeN]
    #: PROGRAM_IN_FLOAT or PROGRAM_IN_IDS per input
    var in_kinds: List[Int]
    #: session handles of the parameters, and the shapes they had at capture
    var params: List[Int]
    var param_shapes: List[ShapeN]
    var code: List[Int]
    var a_at: List[Int]
    var args: List[Int]
    var i_at: List[Int]
    var ints: List[Int]
    var f_at: List[Int]
    var fs: List[Float32]
    var outs: List[Int]
    var out_shapes: List[ShapeN]

    def __init__(out self):
        self.live = True
        self.n_in = 0
        self.in_shapes = List[ShapeN]()
        self.in_kinds = List[Int]()
        self.params = List[Int]()
        self.param_shapes = List[ShapeN]()
        self.code = List[Int]()
        self.a_at = List[Int]()
        self.args = List[Int]()
        self.i_at = List[Int]()
        self.ints = List[Int]()
        self.f_at = List[Int]()
        self.fs = List[Float32]()
        self.outs = List[Int]()
        self.out_shapes = List[ShapeN]()
        self.a_at.append(0)
        self.i_at.append(0)
        self.f_at.append(0)

    def n_nodes(self) -> Int:
        return len(self.code)

    def n_values(self) -> Int:
        """Inputs plus node results (the non-negative refs)."""
        return self.n_in + len(self.code)

    def end_node(mut self):
        """Close the node whose code, args, ints and floats were just appended."""
        self.a_at.append(len(self.args))
        self.i_at.append(len(self.ints))
        self.f_at.append(len(self.fs))


def same_shape(a: ShapeN, b: ShapeN) -> Bool:
    if a.ndim != b.ndim:
        return False
    for k in range(a.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if a.d[k] != b.d[k]:
            return False
    return True


def shape_text(s: ShapeN) -> String:
    var out = String("(")
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): message text, no data
        if k > 0:
            out += ", "
        out += String(s.d[k])
    return out + ")"


def program_refuses_node(code: Int, n_args: Int) -> Bool:
    """Operations a program cannot hold: the reserved code 47, which names
    no operation, and embedding and take (autograd/ops_api.mojo
    OPS_EMBEDDING = 32, OPS_TAKE = 41) called with a caller array of ids
    (one tensor argument; the integer arguments carry the array's address).
    With the ids as a tensor (two tensor arguments) they are held."""
    if code == 47:
        return True
    if code == 32 or code == 41:
        return n_args != 2
    return False


# ---------------------------------------------------------------- words
def _put_shape(mut xi: List[Int64], s: ShapeN):
    xi.append(Int64(s.ndim))
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape words, no data
        xi.append(Int64(s.d[k]))


def program_words(p: OpsProgram) -> Tuple[List[Int64], List[Float32]]:
    """The program's description as checkpoint extension words:
    ints [record version, n_in, (ndim, dims...) per input, (version 2
    only) one kind word per input, n_params,
    (ndim, dims...) per parameter, n_nodes, per node (code, n_args, the
    refs, n_ints, the ints, n_floats), n_out, the output refs, (ndim,
    dims...) per output]; floats every node's float arguments in node order."""
    var xi = List[Int64]()
    var any_ids = False
    for k in range(len(p.in_kinds)):  # driver-loop: over program inputs
        if p.in_kinds[k] != PROGRAM_IN_FLOAT:
            any_ids = True
    xi.append(Int64(PROGRAM_RECORD_VERSION if any_ids else PROGRAM_RECORD_VERSION_1))
    xi.append(Int64(p.n_in))
    for k in range(p.n_in):  # driver-loop: over program inputs
        _put_shape(xi, p.in_shapes[k])
    if any_ids:
        for k in range(p.n_in):  # driver-loop: over program inputs
            xi.append(Int64(p.in_kinds[k]))
    xi.append(Int64(len(p.params)))
    for k in range(len(p.params)):  # driver-loop: over program parameters
        _put_shape(xi, p.param_shapes[k])
    var nn = p.n_nodes()
    xi.append(Int64(nn))
    for k in range(nn):  # driver-loop: over program nodes
        xi.append(Int64(p.code[k]))
        xi.append(Int64(p.a_at[k + 1] - p.a_at[k]))
        for j in range(p.a_at[k], p.a_at[k + 1]):  # small-loop(args: a node's input refs): description words, no data
            xi.append(Int64(p.args[j]))
        xi.append(Int64(p.i_at[k + 1] - p.i_at[k]))
        for j in range(p.i_at[k], p.i_at[k + 1]):  # small-loop(ints: a node's integer arguments): description words, no data
            xi.append(Int64(p.ints[j]))
        xi.append(Int64(p.f_at[k + 1] - p.f_at[k]))
    xi.append(Int64(len(p.outs)))
    for k in range(len(p.outs)):  # driver-loop: over program outputs
        xi.append(Int64(p.outs[k]))
    for k in range(len(p.outs)):  # driver-loop: over program outputs
        _put_shape(xi, p.out_shapes[k])
    return (xi^, p.fs.copy())


struct _Reader(Movable):
    var at: Int
    var n: Int

    def __init__(out self, n: Int):
        self.at = 0
        self.n = n

    def take(mut self, xi: List[Int64], what: String) raises -> Int:
        if self.at >= self.n:
            raise Error(what + ": the program record ends early")
        var v = Int(xi[self.at])
        self.at += 1
        return v

    def shape(mut self, xi: List[Int64], what: String) raises -> ShapeN:
        var nd = self.take(xi, what)
        if nd < 1 or nd > RN_MAX_RANK:
            raise Error(what + ": a program shape has 1 to 8 dimensions")
        var dims = List[Int](capacity=nd)
        for _ in range(nd):  # small-loop(ndim: tensor rank, at most 8): shape words, no data
            dims.append(self.take(xi, what))
        return rn_shape(dims, what)


def program_from_words(xi: List[Int64], xf: List[Float32], what: String) raises -> OpsProgram:
    """The program `program_words` wrote, checked word by word (every ref
    names an input, a parameter or an earlier node). Its parameters are the
    caller's to fill: `params` holds one 0 per parameter shape."""
    var r = _Reader(len(xi))
    var version = r.take(xi, what)
    if version != PROGRAM_RECORD_VERSION_1 and version != PROGRAM_RECORD_VERSION:
        raise Error(what + ": an unknown program record version")
    var p = OpsProgram()
    var n_in = r.take(xi, what)
    if n_in < 1:
        raise Error(what + ": a program has at least one input")
    p.n_in = n_in
    for _ in range(n_in):  # driver-loop: over program inputs
        p.in_shapes.append(r.shape(xi, what))
    for _ in range(n_in):  # driver-loop: over program inputs
        var kind = PROGRAM_IN_FLOAT
        if version == PROGRAM_RECORD_VERSION:
            kind = r.take(xi, what)
            if kind != PROGRAM_IN_FLOAT and kind != PROGRAM_IN_IDS:
                raise Error(what + ": a program input has an unknown kind")
        p.in_kinds.append(kind)
    var n_p = r.take(xi, what)
    if n_p < 0:
        raise Error(what + ": a damaged parameter count")
    for _ in range(n_p):  # driver-loop: over program parameters
        p.params.append(0)
        p.param_shapes.append(r.shape(xi, what))
    var nn = r.take(xi, what)
    if nn < 0:
        raise Error(what + ": a damaged node count")
    var nf_total = 0
    for k in range(nn):  # driver-loop: over program nodes
        var code = r.take(xi, what)
        if code < 1 or code > PROGRAM_MAX_CODE or code == 47:
            raise Error(what + ": node " + String(k) + " holds an unknown operation code " + String(code))
        p.code.append(code)
        var na = r.take(xi, what)
        if na < 0 or na > len(xi):
            raise Error(what + ": a damaged argument count")
        if program_refuses_node(code, na):
            raise Error(what + ": node " + String(k) + " reads a caller's index array, which a program cannot hold")
        for _ in range(na):  # small-loop(args: a node's input refs): description words, no data
            var rf = r.take(xi, what)
            if rf < -n_p or rf >= n_in + k:
                raise Error(what + ": node " + String(k) + " reads a value that is not before it")
            p.args.append(rf)
        var ni = r.take(xi, what)
        if ni < 0 or ni > len(xi):
            raise Error(what + ": a damaged integer count")
        for _ in range(ni):  # small-loop(ints: a node's integer arguments): description words, no data
            p.ints.append(r.take(xi, what))
        var nf = r.take(xi, what)
        if nf < 0 or nf_total + nf > len(xf):
            raise Error(what + ": a damaged float count")
        for j in range(nf):  # small-loop(fs: a node's float arguments): description words, no data
            p.fs.append(xf[nf_total + j])
        nf_total += nf
        p.end_node()
    if nf_total != len(xf):
        raise Error(what + ": the program record holds extra float words")
    var no = r.take(xi, what)
    if no < 1:
        raise Error(what + ": a program has at least one output")
    for _ in range(no):  # driver-loop: over program outputs
        var rf = r.take(xi, what)
        if rf < -n_p or rf >= n_in + nn:
            raise Error(what + ": an output names no value of the program")
        p.outs.append(rf)
    for _ in range(no):  # driver-loop: over program outputs
        p.out_shapes.append(r.shape(xi, what))
    if r.at != len(xi):
        raise Error(what + ": the program record holds extra words")
    return p^
