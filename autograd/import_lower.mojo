# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The lowering of an import graph onto session operations, in Mojo
(docs/plans/MODEL_IMPORT.md section 7, step 2 of the native import boundary).

`onnx_import(handle, session, names, flat, addrs, source)` takes a graph read
by autograd/import_reader.mojo (or built by `onnx_from_words` below) and an
inference `OpsSession` and does what importers/_lower.py and _program.py did
in Python, in the same order with the same arguments, so the captured
program's words are the Python lowering's word for word:

* resolve: the shapes (autograd/import_resolve.mojo `_resolve`) with the
  caller's input shapes (`names`, and `flat` as (rank, dims...) runs);
* the refusal pass: every node the outputs need matched against the table
  and its supported subset, with the shape resolution's problems, all
  listed, in graph order, before anything is uploaded (the same texts);
* the lowering: initializers uploaded through the cast path (float16,
  float64 and integers cast on the executor; a bfloat16 tensor or a `Gemm`
  `transB` weight through the weight import's records), the parameter-only
  subgraphs run once as session operations and persisted (read back and
  uploaded unchanged: a byte copy), the graph inputs as placeholders (an
  integer input read only as `Gather` indices an ids tensor), one or a few
  session operations per remaining node;
* the capture, then the reset and the release of every upload the program
  does not read.

`addrs` holds the address of each buffer of the graph (the model file and
external data files the caller maps, the reader's own bytes): the reader
keeps byte ranges only. Returns [status, ...]:
  0: [0, program_info, inputs [(name, dims, dtype name)], outputs [(name,
     dims, 1 for int32 words else 0)]]
  4: [4, message] (a fatal resolve error)
  5: [5, problems [(name, op or None, reason)]]

`onnx_from_words(words, strings, floats, addrs)` builds a graph handle from
the word layout `onnx_info` returns (header, then the status-0 body; every
buffer an external one at its address): the torch.export reader's GraphIR
takes the same lowering through it.

Description only on the host: names, shapes, axes and attribute numbers.
Element values are never computed here; every value computation, one-element
constants included, is a session operation on the executor."""
# cpu-route: model graph lowered on the host. The lowering is the import step
# before the runtime: it walks nodes, axes and integer shape constants
# (description) and emits one session call per operation; the only element
# bytes it touches are handed to the executor's upload unchanged.
from std.memory import bitcast
from std.python import Python, PythonObject

from autograd.import_reader import (
    AK_FLOAT, AK_FLOATS, AK_INT, AK_INTS, AK_STRING, AK_STRINGS, AK_TENSOR, AK_TENSORS, AttrVal,
    DT_BF16, DT_BOOL, DT_F16, DT_F32, DT_F64, DT_I32, DT_I64, GNode, GValue, ONNX_OK, OnnxGraph,
    OnnxPtr, S_EMPTY, TRec, _ONNX_GRAPHS, _last_attr, dtype_name, fatal_text, fmt_tuple, is_fatal,
    is_int_dtype, itemsize, onnx_graph, py_repr,
)
from autograd.import_resolve import (
    MaybeInts, RCtx, _RProblem, _ail, _as, _fatal, _in, _numel, _resolve, _slice, _slice_range,
)
from autograd.ops_api import (
    OPS_ABS, OPS_ADD, OPS_ARGMAX, OPS_AVGPOOL2D, OPS_CEIL, OPS_COMPARE, OPS_CONCAT, OPS_CONV2D,
    OPS_COS, OPS_DIV, OPS_ELU, OPS_EMBEDDING, OPS_ERF, OPS_EXP, OPS_EXPAND, OPS_FLOOR, OPS_FULL,
    OPS_GELU, OPS_HARD_SIGMOID, OPS_HARD_SWISH, OPS_LAYER_NORM, OPS_LEAKY_RELU, OPS_LINEAR, OPS_LOG,
    OPS_LOGICAL, OPS_LOG_SOFTMAX, OPS_LOG_SOFTMAX_N, OPS_MATMUL, OPS_MAX, OPS_MAXIMUM, OPS_MAXPOOL2D,
    OPS_MEAN_N, OPS_MINIMUM, OPS_MUL, OPS_NARROW, OPS_PAD, OPS_PERMUTE, OPS_POW, OPS_RECIPROCAL,
    OPS_RELU, OPS_RESHAPE, OPS_RESHAPE_N, OPS_ROUND, OPS_SCALE, OPS_SIGMOID, OPS_SIGN, OPS_SIN,
    OPS_SOFTMAX, OPS_SOFTMAX_N, OPS_SOFTPLUS, OPS_SQRT, OPS_SUB, OPS_SUM_N, OPS_TAKE, OPS_TANH,
    OPS_TRANSPOSE, OPS_WHERE, OpsSession, ops_session,
)
from autograd.pyapi import program_info
from autograd.rankn_api import rn_dims, rn_shape
from autograd.weight_import import WI_RECORD, session_load_check, wi_source
from core.cast_in import CAST_KIND_BOOL, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CastIn
from sequence.exec_trait import Exec
from sequence.ops import FP

comptime IMPORT_OK = 0
comptime IMPORT_BAD_VALUE = 4
comptime IMPORT_PROBLEMS = 5

# the largest tensor rank of a session (autograd/rankn.mojo RN_MAX_RANK)
comptime MAX_RANK = 8
# autograd/weight_import.mojo: the largest source rank, the layouts, the C-order stride code
comptime _WI_MAX_RANK = 4
comptime _WI_LAYOUT_AS_IS = 2
comptime _WI_LAYOUT_TRANSPOSE = 3
comptime _WI_STRIDE_C = 0
# NumPy's dtype number of int64 (NPY_LONG on the 64-bit platforms mojolearn builds for):
# the constant Gather's index array, as importers/_lower.py handed it to Session.take
comptime _NPY_INT64 = 7

comptime _NO_HANDLE = -1


# ------------------------------------------------------------ small helpers --

def _pyint_list(p: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(p)):  # cpu-route: model graph lowered on the host (argument words)
        out.append(Int(py=p[i]))
    return out^


def _contains(v: List[Int], x: Int) -> Bool:
    for i in range(len(v)):  # cpu-route: model graph lowered on the host (a short list)
        if v[i] == x:
            return True
    return False


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # cpu-route: model graph lowered on the host (one per axis)
        if a[i] != b[i]:
            return False
    return True


def _all_equal(v: List[Int]) -> Bool:
    """`len(set(v)) == 1`."""
    if len(v) == 0:
        return False
    for i in range(1, len(v)):  # cpu-route: model graph lowered on the host (one per axis)
        if v[i] != v[0]:
            return False
    return True


def _at(v: List[Int], i: Int) raises -> Int:
    """v[i] with Python's negative index; a fatal error out of range."""
    var j = i + len(v) if i < 0 else i
    if j < 0 or j >= len(v):
        raise _fatal("index " + String(i) + " out of range for " + fmt_tuple(v))
    return v[j]


def _range(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(a, b):  # cpu-route: model graph lowered on the host (one per axis)
        out.append(i)
    return out^


def _cat1(head: Int, v: List[Int]) -> List[Int]:
    """[head] + v."""
    var out = List[Int](capacity=len(v) + 1)
    out.append(head)
    for i in range(len(v)):  # cpu-route: model graph lowered on the host (one per axis)
        out.append(v[i])
    return out^


def _session_shape(shape: List[Int]) -> List[Int]:
    """The shape a session holds for a model value: at least two axes, a
    scalar (1, 1) and a [n] vector (1, n)."""
    if len(shape) >= 2:
        return shape.copy()
    var out = List[Int]()
    for _ in range(2 - len(shape)):  # cpu-route: model graph lowered on the host (padding axes)
        out.append(1)
    for i in range(len(shape)):  # cpu-route: model graph lowered on the host (one per axis)
        out.append(shape[i])
    return out^


def _norm(axis: Int, rank: Int) raises -> Int:
    """importers/_lower.py _norm: a refusal (not fatal) when out of range."""
    var hi = rank if rank > 1 else 1
    if not (-hi <= axis and axis < hi):
        raise Error("axis " + String(axis) + " is out of range for rank " + String(rank))
    return axis + rank if axis < 0 else axis


def _sax(axis: Int, rank: Int) raises -> Int:
    """The session axis of model axis `axis` of a rank-`rank` value."""
    return _norm(axis, rank) + (2 - rank if rank < 2 else 0)


def _af(g: OnnxGraph, n: GNode, name: String, default: Float64) raises -> Float64:
    """A number attribute (n.attrs.get(name, default)) as a Python float."""
    var k = _last_attr(g.st, n.attrs, name)
    if k < 0:
        return default
    if n.attrs[k].kind == AK_FLOAT:
        return n.attrs[k].f
    if n.attrs[k].kind == AK_INT:
        return Float64(n.attrs[k].i)
    raise _fatal("attribute '" + name + "' of " + g.st.text(n.name) + " is not a number")


def _ai(g: OnnxGraph, n: GNode, name: String, default: Int) raises -> Int:
    """An int attribute (n.attrs.get(name, default))."""
    var k = _last_attr(g.st, n.attrs, name)
    if k < 0:
        return default
    if n.attrs[k].kind != AK_INT:
        raise _fatal("attribute '" + name + "' of " + g.st.text(n.name) + " is not an int")
    return n.attrs[k].i


def _has_in(n: GNode, i: Int) -> Bool:
    """`len(n.inputs) > i and n.inputs[i]`."""
    return len(n.inputs) > i and n.inputs[i] != S_EMPTY


def _ints(c: RCtx, g: OnnxGraph, name: Int) -> MaybeInts:
    """importers/_lower.py _Graph.ints: the folded integers of `name` or an
    integer initializer's, else none (nothing recorded)."""
    if name == S_EMPTY:
        return MaybeInts(False, List[Int]())
    var got = c.f_slot.get(name)
    if got:
        return MaybeInts(True, c.f_vals[got.value()].copy())
    var k = g.init_of(name)
    if k >= 0 and is_int_dtype(g.inits[k].dtype):
        return MaybeInts(True, g.inits[k].ints.copy())
    return MaybeInts(False, List[Int]())


def _is_float(dt: Int) -> Bool:
    return dt == DT_F32 or dt == DT_F16 or dt == DT_BF16 or dt == DT_F64


def _desc_inputs(op: String) -> List[Int]:
    """The inputs of an op that are description (static integers folded at import), not data."""
    var d = List[Int]()
    if (op == "Reshape" or op == "Expand" or op == "Squeeze" or op == "Unsqueeze" or op == "Split"
            or op == "ReduceSum" or op == "ReduceMean" or op == "ReduceMax" or op == "ReduceMin"
            or op == "ReduceL2" or op == "Trilu" or op == "Gather"):
        d.append(1)
    elif op == "Slice":
        d.append(1)
        d.append(2)
        d.append(3)
        d.append(4)
    elif op == "Pad":
        d.append(1)
        d.append(3)
    elif op == "ConstantOfShape" or op == "Shape" or op == "Size":
        d.append(0)
    elif op == "Dropout":
        d.append(1)
        d.append(2)
    elif op == "Range":
        d.append(0)
        d.append(1)
        d.append(2)
    return d^


# --------------------------------------------------------- the op table --

def _unary_code(op: String) -> Int:
    """The session operation of a one-input elementwise op, 0 when none
    (Not is OPS_LOGICAL kind 0, below)."""
    if op == "Relu":
        return OPS_RELU
    if op == "Sigmoid":
        return OPS_SIGMOID
    if op == "Tanh":
        return OPS_TANH
    if op == "Exp":
        return OPS_EXP
    if op == "Log":
        return OPS_LOG
    if op == "Sqrt":
        return OPS_SQRT
    if op == "Reciprocal":
        return OPS_RECIPROCAL
    if op == "Erf":
        return OPS_ERF
    if op == "Softplus":
        return OPS_SOFTPLUS
    if op == "Abs":
        return OPS_ABS
    if op == "Floor":
        return OPS_FLOOR
    if op == "Ceil":
        return OPS_CEIL
    if op == "Round":
        return OPS_ROUND
    if op == "Sin":
        return OPS_SIN
    if op == "Cos":
        return OPS_COS
    if op == "Sign":
        return OPS_SIGN
    if op == "HardSwish":
        return OPS_HARD_SWISH
    return 0


def _is_unary(op: String) -> Bool:
    return _unary_code(op) != 0 or op == "Not"


def _is_binary(op: String) -> Bool:
    return (op == "Add" or op == "Sub" or op == "Mul" or op == "Div" or op == "Equal" or op == "Greater"
        or op == "Less" or op == "GreaterOrEqual" or op == "LessOrEqual" or op == "And" or op == "Or")


def _is_reduce(op: String) -> Bool:
    return op == "ReduceSum" or op == "ReduceMean" or op == "ReduceMax" or op == "ReduceMin" or op == "ReduceL2"


def _in_table(op: String) -> Bool:
    """The ops importers/_lower.py's _TABLE lowers."""
    if _is_unary(op) or _is_binary(op) or _is_reduce(op):
        return True
    return (op == "Reshape" or op == "Flatten" or op == "Squeeze" or op == "Unsqueeze" or op == "Shape"
        or op == "Size" or op == "Identity" or op == "Dropout" or op == "Cast" or op == "Neg"
        or op == "LeakyRelu" or op == "Elu" or op == "HardSigmoid" or op == "Gelu" or op == "Pow"
        or op == "Max" or op == "Min" or op == "Where" or op == "Clip" or op == "MatMul" or op == "Gemm"
        or op == "Softmax" or op == "LogSoftmax" or op == "LayerNormalization" or op == "RMSNormalization"
        or op == "BatchNormalization" or op == "InstanceNormalization" or op == "GroupNormalization"
        or op == "Transpose" or op == "Expand" or op == "Concat" or op == "Split" or op == "Slice"
        or op == "Gather" or op == "ArgMax" or op == "GlobalAveragePool" or op == "GlobalMaxPool"
        or op == "Conv" or op == "MaxPool" or op == "AveragePool" or op == "Pad" or op == "Trilu"
        or op == "ConstantOfShape" or op == "Constant")


# ------------------------------------------------------------ the analysis --

struct _An(Movable):
    """importers/_lower.py _Graph: which values are constant, which nodes
    the outputs need, who reads each value, the ids inputs."""
    var inputs: List[Int]            # g.inputs slots that are not initializers
    var const: Dict[Int, Bool]
    var needed: List[Int]            # node slots, graph order
    var want: Dict[Int, Bool]
    var consumers: Dict[Int, Int]    # value -> number of needed-node reads
    var outs: Dict[Int, Bool]        # graph output names
    var index_inputs: Dict[Int, Bool]

    def __init__(out self):
        self.inputs = List[Int]()
        self.const = Dict[Int, Bool]()
        self.needed = List[Int]()
        self.want = Dict[Int, Bool]()
        self.consumers = Dict[Int, Int]()
        self.outs = Dict[Int, Bool]()
        self.index_inputs = Dict[Int, Bool]()

    def used(self, name: Int) -> Bool:
        """A needed node or a graph output reads value `name`."""
        return name != S_EMPTY and (name in self.consumers or name in self.outs)


def _analyse(g: OnnxGraph, c: RCtx) raises -> _An:
    var a = _An()
    for i in range(len(g.inputs)):  # cpu-route: model graph lowered on the host (graph inputs)
        if g.init_of(g.inputs[i].name) < 0:
            a.inputs.append(i)
    for i in range(len(g.inits)):  # cpu-route: model graph lowered on the host (initializers)
        a.const[g.inits[i].name] = True
    for i in range(len(c.f_name)):  # cpu-route: model graph lowered on the host (folded values)
        a.const[c.f_name[i]] = True
    for k in range(len(g.nodes)):  # cpu-route: model graph lowered on the host (graph nodes)
        var all_c = True
        for j in range(len(g.nodes[k].inputs)):  # cpu-route: model graph lowered on the host (node inputs)
            var x = g.nodes[k].inputs[j]
            if x != S_EMPTY and x not in a.const:
                all_c = False
        if all_c:
            for j in range(len(g.nodes[k].outputs)):  # cpu-route: model graph lowered on the host (node outputs)
                if g.nodes[k].outputs[j] != S_EMPTY:
                    a.const[g.nodes[k].outputs[j]] = True
    # the nodes the outputs need, back from the outputs; a folded value needs no node
    for i in range(len(g.outputs)):  # cpu-route: model graph lowered on the host (graph outputs)
        a.want[g.outputs[i].name] = True
        a.outs[g.outputs[i].name] = True
    var rev = List[Int]()
    var k = len(g.nodes) - 1
    while k >= 0:  # cpu-route: model graph lowered on the host (graph nodes, last to first)
        var hit = False
        for j in range(len(g.nodes[k].outputs)):  # cpu-route: model graph lowered on the host (node outputs)
            var o = g.nodes[k].outputs[j]
            if o != S_EMPTY and o in a.want and o not in c.f_slot:
                hit = True
        if hit:
            rev.append(k)
            var desc = _desc_inputs(g.st.text(g.nodes[k].op))
            for j in range(len(g.nodes[k].inputs)):  # cpu-route: model graph lowered on the host (node inputs)
                if g.nodes[k].inputs[j] != S_EMPTY and not _contains(desc, j):
                    a.want[g.nodes[k].inputs[j]] = True
        k -= 1
    var r = len(rev) - 1
    while r >= 0:  # cpu-route: model graph lowered on the host (needed nodes)
        a.needed.append(rev[r])
        r -= 1
    for q in range(len(a.needed)):  # cpu-route: model graph lowered on the host (needed nodes)
        ref n = g.nodes[a.needed[q]]
        for j in range(len(n.inputs)):  # cpu-route: model graph lowered on the host (node inputs)
            var x = n.inputs[j]
            if x != S_EMPTY:
                a.consumers[x] = a.consumers.get(x, 0) + 1
    # integer model inputs that needed nodes read only as Gather indices: ids tensors
    for q in range(len(a.inputs)):  # cpu-route: model graph lowered on the host (graph inputs)
        ref v = g.inputs[a.inputs[q]]
        var name = v.name
        if not (v.dtype == DT_I64 or v.dtype == DT_I32) or name not in a.consumers or name in a.outs:
            continue
        var only = True
        for t in range(len(a.needed)):  # cpu-route: model graph lowered on the host (the input's readers)
            ref n = g.nodes[a.needed[t]]
            if not _contains(n.inputs, name):
                continue
            if not (g.st.text(n.op) == "Gather" and len(n.inputs) > 1 and n.inputs[1] == name and n.inputs[0] != name):
                only = False
        if only:
            a.index_inputs[name] = True
    return a^


# ------------------------------------------------------------ subset checks --

@fieldwise_init
struct _Win(Copyable, Movable):
    """(kernel, strides, pads, dilations) of a Conv or pool node, the pads explicit."""
    var k: List[Int]
    var st: List[Int]
    var pads: List[Int]
    var dil: List[Int]


def _window(g: OnnxGraph, c: RCtx, n: GNode) raises -> _Win:
    """importers/_lower.py _window: auto_pad resolved from the static shapes
    (VALID no padding; SAME_UPPER / SAME_LOWER to ceil(size / stride)
    outputs, the odd count at the end or at the start)."""
    var s = c.shape(_in(n, 0))
    var spatial = len(s) - 2 if len(s) > 2 else 0
    var k = List[Int]()
    if not _ail(g, n, "kernel_shape", k):
        if g.st.text(n.op) == "Conv":
            var ws = c.shape(_in(n, 1))
            k = _slice(ws, 2, len(ws))
    var st = List[Int]()
    if not _ail(g, n, "strides", st):
        st = List[Int](length=spatial, fill=1)
    var pads = List[Int]()
    if not _ail(g, n, "pads", pads):
        pads = List[Int](length=2 * spatial, fill=0)
    var dil = List[Int]()
    if not _ail(g, n, "dilations", dil):
        dil = List[Int](length=spatial, fill=1)
    var auto = _as(g, n, "auto_pad", "NOTSET")
    if auto == "VALID":
        pads = List[Int](length=2 * spatial, fill=0)
    elif (auto == "SAME_UPPER" or auto == "SAME_LOWER") and len(k) == spatial and len(st) == spatial:
        var before = List[Int]()
        var after = List[Int]()
        for i in range(spatial):  # cpu-route: model graph lowered on the host (one per spatial axis)
            var d = s[2 + i]
            if st[i] == 0:
                raise _fatal("strides of " + g.st.text(n.name) + " hold 0")
            var di = _at(dil, i)
            var total = (-(-d // st[i]) - 1) * st[i] + (k[i] - 1) * di + 1 - d
            if total < 0:
                total = 0
            var first = total // 2 if auto == "SAME_UPPER" else total - total // 2
            before.append(first)
            after.append(total - first)
        pads = before^
        for i in range(len(after)):  # cpu-route: model graph lowered on the host (one per spatial axis)
            pads.append(after[i])
    return _Win(k^, st^, pads^, dil^)


def _pad_counts(pads: List[Int]) raises -> List[Int]:
    """Session.pad counts of a rank-4 (N, C, H, W) value from 2-D ONNX
    window pads (top, left, bottom, right)."""
    var out = List[Int](length=8, fill=0)
    out[2] = _at(pads, 0)
    out[3] = _at(pads, 1)
    out[6] = _at(pads, 2)
    out[7] = _at(pads, 3)
    return out^


def _q(g: OnnxGraph, name: Int) -> String:
    """'name' as the messages quote a value."""
    return "'" + g.st.text(name) + "'"


def _extra_outputs(g: OnnxGraph, a: _An, n: GNode, what: String) -> List[String]:
    var out = List[String]()
    for j in range(1, len(n.outputs)):  # cpu-route: model graph lowered on the host (node outputs)
        if a.used(n.outputs[j]):
            out.append("the " + what + " output is not supported")
            return out^
    return out^


def _subset(g: OnnxGraph, c: RCtx, a: _An, n: GNode, op: String) raises -> List[String]:
    """The subset check of node n (importers/_lower.py _c_*): its problems.
    A plain Error is a refusal that replaces them (Python's _Refuse)."""
    var out = List[String]()
    if _is_binary(op):
        if op == "Div":
            var dt = c.dtype(n.outputs[0]) if c.has(n.outputs[0]) else -1
            if dt == DT_I64 or dt == DT_I32:
                out.append("integer division of tensor values is not supported")
        return out^
    if _is_reduce(op):
        var axes = n.inputs[1] if len(n.inputs) > 1 else S_EMPTY
        if axes != S_EMPTY and not _ints(c, g, axes).ok:
            out.append("the axes are computed from the model's inputs (static axes are supported)")
        return out^
    if op == "Cast":
        var to = c.dtype(n.outputs[0])
        var src = c.dtype(_in(n, 0))
        if _is_float(src) and not _is_float(to):
            out.append("a cast of floating-point values to " + dtype_name(to) + " is not supported")
        elif (src == DT_I64 or src == DT_I32) and to == DT_BOOL:
            out.append("a cast of integer values to bool is not supported")
        return out^
    if op == "Gelu":
        var ap = _as(g, n, "approximate", "none")
        if ap != "none" and ap != "tanh":
            out.append("approximate=" + py_repr(ap) + " (only 'none' and 'tanh' are supported)")
        return out^
    if op == "LayerNormalization":
        out = _extra_outputs(g, a, n, "Mean or InvStdDev")
        var s = c.shape(_in(n, 0))
        var r = len(s)
        var ax = _ai(g, n, "axis", -1)
        if r == 0 or _norm(ax, r) != r - 1:
            out.append("axis=" + String(ax) + " normalizes more than the last axis of " + fmt_tuple(s)
                       + " (only the last axis is supported)")
            return out^
        for j in range(1, min(3, len(n.inputs))):  # cpu-route: model graph lowered on the host (scale and bias)
            var i = n.inputs[j]
            if i != S_EMPTY and _numel(c.shape(i)) != s[r - 1]:
                out.append(_q(g, i) + " of shape " + fmt_tuple(c.shape(i)) + " is not one value per feature ("
                           + String(s[r - 1]) + ")")
        if not _af(g, n, "epsilon", 1e-5) > 0:
            out.append("epsilon must be positive")
        return out^
    if op == "RMSNormalization":
        out = _extra_outputs(g, a, n, "InvStdDev")
        var s = c.shape(_in(n, 0))
        var r = len(s)
        var ax = _ai(g, n, "axis", -1)
        if r == 0 or _norm(ax, r) != r - 1:
            out.append("axis=" + String(ax) + " normalizes more than the last axis of " + fmt_tuple(s)
                       + " (only the last axis is supported)")
        elif _numel(c.shape(_in(n, 1))) != s[r - 1]:
            out.append("scale of shape " + fmt_tuple(c.shape(_in(n, 1))) + " is not one value per feature ("
                       + String(s[r - 1]) + ")")
        return out^
    if op == "BatchNormalization":
        out = _extra_outputs(g, a, n, "running statistics")
        if _ai(g, n, "training_mode", 0) != 0:
            out.append("training_mode=1 (only inference is supported)")
        var s = c.shape(_in(n, 0))
        if len(s) < 2:
            out.append("input " + fmt_tuple(s) + " has no channel axis")
            return out^
        for j in range(1, min(5, len(n.inputs))):  # cpu-route: model graph lowered on the host (the four statistics)
            var i = n.inputs[j]
            if i == S_EMPTY or i not in a.const:
                out.append(_q(g, i) + " is computed from the model's inputs (constant statistics are supported)")
            elif _numel(c.shape(i)) != s[1]:
                out.append(_q(g, i) + " of shape " + fmt_tuple(c.shape(i)) + " is not one value per channel ("
                           + String(s[1]) + ")")
        return out^
    if op == "InstanceNormalization":
        var s = c.shape(_in(n, 0))
        if len(s) < 3:
            out.append("input " + fmt_tuple(s) + " needs rank 3 or more")
            return out^
        for j in range(1, min(3, len(n.inputs))):  # cpu-route: model graph lowered on the host (scale and bias)
            var i = n.inputs[j]
            if i != S_EMPTY and _numel(c.shape(i)) != s[1]:
                out.append(_q(g, i) + " of shape " + fmt_tuple(c.shape(i)) + " is not one value per channel ("
                           + String(s[1]) + ")")
        return out^
    if op == "GroupNormalization":
        var s = c.shape(_in(n, 0))
        if len(s) < 3:
            out.append("input " + fmt_tuple(s) + " needs rank 3 or more")
            return out^
        var gk = _last_attr(g.st, n.attrs, "num_groups")
        var gtext = String("None")
        var gr = 0
        var ok = False
        if gk >= 0:
            if n.attrs[gk].kind == AK_INT:
                gr = n.attrs[gk].i
                gtext = String(gr)
                ok = True
            elif n.attrs[gk].kind == AK_FLOAT:
                gtext = String(n.attrs[gk].f)
            else:
                raise _fatal("attribute 'num_groups' of " + g.st.text(n.name) + " is not an int")
        if not ok or gr < 1 or s[1] % gr != 0:
            out.append("num_groups=" + gtext + " does not divide " + String(s[1]) + " channels")
            return out^
        for j in range(1, min(3, len(n.inputs))):  # cpu-route: model graph lowered on the host (scale and bias)
            var i = n.inputs[j]
            if i != S_EMPTY:
                var p = _numel(c.shape(i))
                if p != s[1] and p != gr:
                    out.append(_q(g, i) + " of shape " + fmt_tuple(c.shape(i)) + " is neither per channel ("
                               + String(s[1]) + ") nor per group (" + String(gr) + ")")
        return out^
    if op == "Split":
        var s = c.shape(_in(n, 0))
        var ax = _norm(_ai(g, n, "axis", 0), len(s))
        for j in range(len(n.outputs)):  # cpu-route: model graph lowered on the host (node outputs)
            var o = n.outputs[j]
            if o != S_EMPTY and c.has(o) and c.shape(o)[ax] == 0:
                out.append("an empty piece is not supported")
                return out^
        return out^
    if op == "Slice":
        if len(n.inputs) > 4 and n.inputs[4] != S_EMPTY:
            var steps = _ints(c, g, n.inputs[4])
            if steps.ok:
                var bad = False
                for j in range(len(steps.v)):  # cpu-route: model graph lowered on the host (one per sliced axis)
                    if steps.v[j] != 1:
                        bad = True
                if bad:
                    out.append("steps " + fmt_tuple(steps.v) + " (only steps of 1 are supported)")
        if _contains(c.shape(n.outputs[0]), 0):
            out.append("an empty slice is not supported")
        return out^
    if op == "Gather":
        var idx = _in(n, 1)
        if not _ints(c, g, idx).ok and idx not in a.index_inputs:
            out.append("the indices are computed from the model's inputs (constant indices, or an integer "
                       + "model input read only as indices, are supported)")
        return out^
    if op == "ArgMax":
        if _ai(g, n, "select_last_index", 0) != 0:
            out.append("select_last_index=1 is not supported")
        return out^
    if op == "Conv":
        var s = c.shape(_in(n, 0))
        if len(s) != 3 and len(s) != 4:
            out.append("input " + fmt_tuple(s) + ": only 1-D and 2-D convolutions are supported")
            return out^
        var w = _window(g, c, n)
        var group = _ai(g, n, "group", 1)
        var oc = c.shape(_in(n, 1))[0]
        if group < 1 or s[1] % group != 0 or oc % group != 0:
            out.append("group=" + String(group) + " does not divide " + String(s[1]) + " input and "
                       + String(oc) + " output channels")
        var dbad = False
        for j in range(len(w.dil)):  # cpu-route: model graph lowered on the host (one per spatial axis)
            if w.dil[j] != 1:
                dbad = True
        if dbad:
            out.append("dilations=" + fmt_tuple(w.dil) + " (only 1 is supported)")
        if not _all_equal(w.st):
            out.append("strides=" + fmt_tuple(w.st) + " (only equal strides are supported)")
        var pbad = False
        for j in range(len(w.pads)):  # cpu-route: model graph lowered on the host (one per side)
            if w.pads[j] < 0:
                pbad = True
        if pbad:
            out.append("pads=" + fmt_tuple(w.pads) + " (negative pads are not supported)")
        return out^
    if op == "MaxPool" or op == "AveragePool":
        out = _extra_outputs(g, a, n, "Indices")
        var s = c.shape(_in(n, 0))
        if len(s) != 4:
            out.append("input " + fmt_tuple(s) + ": only 2-D pooling is supported")
            return out^
        var w = _window(g, c, n)
        if not _all_equal(w.k):
            out.append("kernel_shape=" + fmt_tuple(w.k) + " (only a square kernel is supported)")
        if not _all_equal(w.st):
            out.append("strides=" + fmt_tuple(w.st) + " (only equal strides are supported)")
        if len(w.k) > 0:
            var pbad = False
            for j in range(len(w.pads)):  # cpu-route: model graph lowered on the host (one per side)
                if w.pads[j] < 0 or w.pads[j] >= w.k[0]:
                    pbad = True
            if pbad:
                # a pad of the kernel's size or more makes windows of padding only
                out.append("pads=" + fmt_tuple(w.pads) + " (each pad must be between 0 and kernel_shape - 1)")
        var dbad = False
        for j in range(len(w.dil)):  # cpu-route: model graph lowered on the host (one per spatial axis)
            if w.dil[j] != 1:
                dbad = True
        if dbad:
            out.append("dilations=" + fmt_tuple(w.dil) + " (only 1 is supported)")
        if _ai(g, n, "storage_order", 0) != 0:
            out.append("storage_order=1 is not supported")
        if _ai(g, n, "ceil_mode", 0) != 0 and len(w.k) > 0 and len(w.st) > 0:
            # ceil_mode changes nothing when every window fits exactly
            var fl = List[Int]()
            for i in range(2):  # cpu-route: model graph lowered on the host (two spatial axes)
                var sti = _at(w.st, i)
                if sti == 0:
                    raise _fatal("strides of " + g.st.text(n.name) + " hold 0")
                fl.append((s[2 + i] + _at(w.pads, i) + _at(w.pads, 2 + i) - _at(w.k, i)) // sti + 1)
            var os = c.shape(n.outputs[0])
            if not _same(_slice(os, 2, len(os)), fl):
                out.append("ceil_mode=1 with a partial last window is not supported")
        return out^
    if op == "Pad":
        var mode = _as(g, n, "mode", "constant")
        if mode != "constant":
            out.append("mode=" + py_repr(mode) + " (only 'constant' is supported)")
        if _has_in(n, 1) and not _ints(c, g, n.inputs[1]).ok:
            out.append("the pads are computed from the model's inputs (static pads are supported)")
        return out^
    if op == "Trilu":
        var s = c.shape(_in(n, 0))
        if len(s) < 2:
            out.append("input " + fmt_tuple(s) + " needs rank 2 or more")
            return out^
        if _has_in(n, 1) and not _ints(c, g, n.inputs[1]).ok:
            out.append("k is computed from the model's inputs (a static k is supported)")
        return out^
    if op == "Dropout":
        out = _extra_outputs(g, a, n, "mask")
        if len(n.inputs) > 2 and n.inputs[2] != S_EMPTY:
            var tm = _ints(c, g, n.inputs[2])
            var on = not tm.ok
            for j in range(len(tm.v)):  # cpu-route: model graph lowered on the host (one flag)
                if tm.v[j] != 0:
                    on = True
            if on:
                out.append("training_mode is not false (only inference is supported)")
        return out^
    return out^


# ------------------------------------------------------------ the refusal pass --

@fieldwise_init
struct _Found(Copyable, Movable):
    var key: Int
    var name: String
    var has_op: Bool
    var op: String
    var why: String


def _check(g: OnnxGraph, c: RCtx, a: _An, rprobs: List[_RProblem], source: String) raises -> List[_Found]:
    """importers/_lower.py check: every problem, in graph order (a stable
    sort on the node position; value problems first)."""
    var order = Dict[Int, Int]()
    for k in range(len(g.nodes)):  # cpu-route: model graph lowered on the host (graph nodes)
        order[g.nodes[k].name] = k
    var n_nodes = len(order)
    var input_names = Dict[Int, Bool]()
    for i in range(len(g.inputs)):  # cpu-route: model graph lowered on the host (graph inputs)
        input_names[g.inputs[i].name] = True
    var found = List[_Found]()
    var flagged = Dict[Int, Bool]()
    for i in range(len(rprobs)):  # cpu-route: model graph lowered on the host (shape problems)
        ref p = rprobs[i]
        var key: Int
        if p.op >= 0:
            key = order.get(p.name, n_nodes)
            flagged[p.name] = True
        else:
            key = -1 if (p.name in input_names or p.name not in a.outs) else n_nodes
        found.append(_Found(key, g.st.text(p.name), p.op >= 0, g.st.text(p.op) if p.op >= 0 else String(""),
                            p.reason))
    if len(a.inputs) == 0:
        found.append(_Found(-1, source, False, String(""), String("the model has no inputs")))
    for q in range(len(a.inputs)):  # cpu-route: model graph lowered on the host (graph inputs)
        ref v = g.inputs[a.inputs[q]]
        if v.dtype == DT_BF16:
            found.append(_Found(-1, g.st.text(v.name), False, String(""),
                                String("bfloat16 inputs are not supported (pass the input as float32)")))
        if len(v.dims) > MAX_RANK:
            found.append(_Found(-1, g.st.text(v.name), False, String(""),
                                "rank " + String(len(v.dims)) + " (at most " + String(MAX_RANK) + " is supported)"))
    for q in range(len(a.needed)):  # cpu-route: model graph lowered on the host (needed nodes)
        ref n = g.nodes[a.needed[q]]
        var k = order[n.name]
        if n.name in flagged:
            continue
        var op = g.st.text(n.op)
        var dom = g.st.text(n.domain)
        var nm = g.st.text(n.name)
        var default_dom = dom.byte_length() == 0 or dom == "ai.onnx"
        if not default_dom:
            found.append(_Found(k, nm, True, op, "operations of domain '" + dom + "' are not supported"))
            continue
        if not _in_table(op):
            found.append(_Found(k, nm, True, op, String("no kernel; see mojolearn.importers.supported_ops()")))
            continue
        var missing = False
        for j in range(len(n.inputs)):  # cpu-route: model graph lowered on the host (node values)
            if n.inputs[j] != S_EMPTY and not c.has(n.inputs[j]):
                missing = True
        for j in range(len(n.outputs)):  # cpu-route: model graph lowered on the host (node values)
            if n.outputs[j] != S_EMPTY and not c.has(n.outputs[j]):
                missing = True
        if missing:
            continue  # an upstream problem is already listed
        var why = List[String]()
        for j in range(len(n.outputs)):  # cpu-route: model graph lowered on the host (node outputs)
            var o = n.outputs[j]
            if o != S_EMPTY and a.used(o):
                var os = c.shape(o)
                if len(os) > MAX_RANK:
                    why.append(_q(g, o) + " has rank " + String(len(os)) + " (at most " + String(MAX_RANK)
                               + " is supported)")
                elif _contains(os, 0):
                    why.append(_q(g, o) + " is empty")
        for j in range(len(n.inputs)):  # cpu-route: model graph lowered on the host (node inputs)
            var i = n.inputs[j]
            var t = g.init_of(i)
            if t < 0:
                continue
            if g.inits[t].numel() == 0 and i in a.want:
                why.append(_q(g, i) + " is an empty tensor")
            elif g.inits[t].dtype == DT_BF16 and len(g.inits[t].dims) > _WI_MAX_RANK:
                why.append(_q(g, i) + " is a bfloat16 tensor of more than " + String(_WI_MAX_RANK) + " dimensions")
        try:
            var more = _subset(g, c, a, n, op)
            for j in range(len(more)):  # cpu-route: model graph lowered on the host (one per problem)
                why.append(more[j])
        except e:
            var msg = String(e)
            if is_fatal(msg):
                raise Error(msg)
            why.append(msg)
        for j in range(len(why)):  # cpu-route: model graph lowered on the host (one per problem)
            found.append(_Found(k, nm, True, op, why[j]))
    # stable insertion sort on the key (Python's list.sort)
    for i in range(1, len(found)):  # cpu-route: model graph lowered on the host (one per problem)
        var j = i
        while j > 0 and found[j - 1].key > found[j].key:  # cpu-route: model graph lowered on the host (one per problem)
            var tmp = found[j - 1].copy()
            found[j - 1] = found[j].copy()
            found[j] = tmp^
            j -= 1
    return found^


# ------------------------------------------------------------ the lowering --

struct _Low[E: Exec & Movable & Deinitable](Movable):
    """One pass over the needed nodes, in order, through session `p`
    (importers/_lower.py _Lowerer: the same calls in the same order)."""
    var p: UnsafePointer[OpsSession[Self.E], MutUntrackedOrigin]
    var g: OnnxPtr
    var c: RCtx
    var a: _An
    var addrs: List[Int]
    var h: Dict[Int, Int]        # value name -> session handle
    var ht: Dict[Int, Int]       # Gemm transB weight name -> its transposed upload
    var owned: List[Int]         # handles made with an upload (uploads and persisted constants)
    var owned_set: Dict[Int, Bool]
    var cur_const: Bool
    var int32_out: Dict[Int, Bool]

    def __init__(out self, p: UnsafePointer[OpsSession[Self.E], MutUntrackedOrigin], g: OnnxPtr,
                 var c: RCtx, var a: _An, var addrs: List[Int]):
        self.p = p
        self.g = g
        self.c = c^
        self.a = a^
        self.addrs = addrs^
        self.h = Dict[Int, Int]()
        self.ht = Dict[Int, Int]()
        self.owned = List[Int]()
        self.owned_set = Dict[Int, Bool]()
        self.cur_const = False
        self.int32_out = Dict[Int, Bool]()

    # ---------------------------------------------------------- session calls
    def op(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]) raises -> Int:
        return self.p[].op(code, hs, ints, fs)

    def op1(mut self, code: Int, x: Int) raises -> Int:
        return self.op(code, [x], List[Int](), List[Float32]())

    def op2(mut self, code: Int, x: Int, y: Int) raises -> Int:
        return self.op(code, [x, y], List[Int](), List[Float32]())

    def opk(mut self, code: Int, x: Int, ints: List[Int]) raises -> Int:
        return self.op(code, [x], ints, List[Float32]())

    def opf(mut self, code: Int, x: Int, f: Float64) raises -> Int:
        return self.op(code, [x], List[Int](), [Float32(f)])

    def shape_h(self, h: Int) raises -> List[Int]:
        """Session.shape."""
        return rn_dims(self.p[].shape_nd(h))

    def reshape(mut self, x: Int, dims: List[Int]) raises -> Int:
        """Session.reshape (no -1 in the dims)."""
        if len(dims) == 2:
            return self.opk(OPS_RESHAPE, x, [dims[0], dims[1]])
        return self.opk(OPS_RESHAPE_N, x, _cat1(0, dims))

    def full(mut self, dims: List[Int], v: Float64) raises -> Int:
        return self.op(OPS_FULL, List[Int](), dims, [Float32(v)])

    def scale(mut self, x: Int, f: Float64) raises -> Int:
        return self.opf(OPS_SCALE, x, f)

    def split(mut self, x: Int, sizes: List[Int], axis: Int) raises -> List[Int]:
        """Session.split by sizes: one narrow per piece, every piece made."""
        var out = List[Int]()
        for k in range(len(sizes)):  # driver-loop: over the pieces of one split
            var ints = [axis, k, 0]
            for j in range(len(sizes)):  # cpu-route: model graph lowered on the host (piece lengths)
                ints.append(sizes[j])
            out.append(self.opk(OPS_NARROW, x, ints))
        return out^

    def concat(mut self, hs: List[Int], axis: Int) raises -> Int:
        return self.op(OPS_CONCAT, hs, [axis], List[Float32]())

    def reduce(mut self, code: Int, x: Int, axes: List[Int]) raises -> Int:
        """Session.sum / mean / max with keepdims=True."""
        return self.opk(code, x, _cat1(1, axes))

    def pad(mut self, x: Int, counts: List[Int], v: Float64) raises -> Int:
        return self.op(OPS_PAD, [x], counts, [Float32(v)])

    def conv2d(mut self, x: Int, w: Int, b: Int, stride: Int, padding: Int, groups: Int) raises -> Int:
        var hs = [x, w]
        if b != _NO_HANDLE:
            hs.append(b)
        return self.op(OPS_CONV2D, hs, [stride, padding, groups], List[Float32]())

    def where(mut self, cond: Int, x: Int, y: Int) raises -> Int:
        return self.op(OPS_WHERE, [cond, x, y], List[Int](), List[Float32]())

    def _upload_any(mut self, addr: Int, kind: Int, size: Int, dims: List[Int]) raises -> Int:
        """Session.tensor of a C-order array of `dims` at `addr` (cast on the executor)."""
        if len(dims) <= 2:
            var rows = 1
            var cols = 1
            if len(dims) == 2:
                rows = dims[0]
                cols = dims[1]
            elif len(dims) == 1:
                rows = dims[0]
            return self.p[].upload(CastIn(addr, kind, size, rows, cols, cols * size, size), False)
        var last = dims[len(dims) - 1]
        var src = CastIn(addr, kind, size, _numel(dims) // last, last, last * size, size)
        return self.p[].upload_nd(src, rn_shape(dims, "ops: tensor"), False)

    def _upload_f32_zeros(mut self, dims: List[Int]) raises -> Int:
        var buf = List[Float32](length=_numel(dims), fill=Float32(0))
        var h = self._upload_any(Int(buf.unsafe_ptr()), CAST_KIND_FLOAT, 4, dims)
        _ = buf^
        return h

    # ---------------------------------------------------------- values
    def own(mut self, h: Int) -> Int:
        self.owned.append(h)
        self.owned_set[h] = True
        return h

    def persist(mut self, h: Int) raises -> Int:
        """Constant h as an uploaded tensor: its words read back and uploaded
        unchanged (a byte copy at import), so the program reads it as a
        parameter instead of computing it at every run."""
        if h in self.owned_set:
            return h
        var dims = self.shape_h(h)
        var buf = List[Float32](length=_numel(dims), fill=Float32(0))
        self.p[].download(h, FP(unsafe_from_address=Int(buf.unsafe_ptr())))
        var nh = self._upload_any(Int(buf.unsafe_ptr()), CAST_KIND_FLOAT, 4, dims)
        _ = buf^
        return self.own(nh)

    def k(mut self, h: Int) raises -> Int:
        return self.persist(h)

    def is_const(self, name: Int) -> Bool:
        return name in self.a.const

    def kc(mut self, h: Int, name: Int) raises -> Int:
        """`L.k(h) if name in G.const and not L.cur_const else h`."""
        if self.is_const(name) and not self.cur_const:
            return self.k(h)
        return h

    def get(mut self, name: Int) raises -> Int:
        """The handle of value `name`; a constant read by a node that depends
        on the inputs is a parameter (persisted when it was computed)."""
        var got = self.h.get(name)
        var h: Int
        if got:
            h = got.value()
        else:
            h = self._upload_value(name)
            self.h[name] = h
        if not self.cur_const and name in self.a.const and h not in self.owned_set:
            h = self.persist(h)
            self.h[name] = h
        return h

    def _upload_value(mut self, name: Int) raises -> Int:
        var t = self.g[].init_of(name)
        if t >= 0:
            return self.upload(self.g[].inits[t].copy(), False)
        var v = _ints(self.c, self.g[], name)
        if v.ok:
            return self.upload_ints(v.v, self.c.shape(name))
        raise Error("the lowering reached value '" + self.g[].st.text(name) + "' before the node that computes it")

    def upload(mut self, t: TRec, transpose: Bool) raises -> Int:
        """Initializer t as a session tensor of its session shape, cast on the
        executor; transpose loads a 2-D (a, b) initializer as (b, a) through
        the weight import's layout code (a bfloat16 tensor goes that way too)."""
        var shape = _session_shape(t.dims)
        if t.buf < 0 or t.buf >= len(self.addrs):
            raise _fatal("tensor '" + self.g[].st.text(t.name) + "' names buffer " + String(t.buf))
        var base = self.addrs[t.buf] + t.off
        if t.dtype == DT_BF16 or transpose:
            var target: List[Int]
            if transpose:
                target = [t.dims[1], t.dims[0]]
            else:
                target = shape.copy()
            var h = self.own(self._upload_f32_zeros(target))
            var src = t.dims.copy()
            if len(src) == 0:
                src.append(1)
            var code: Int
            var size: Int
            if t.dtype == DT_F32:
                code = 11
                size = 4
            elif t.dtype == DT_F16:
                code = 23
                size = 2
            elif t.dtype == DT_F64:
                code = 12
                size = 8
            elif t.dtype == DT_BF16:
                code = 4096
                size = 2
            else:
                raise _fatal("tensor '" + self.g[].st.text(t.name) + "' is not a float tensor")
            var rec = [h, 0, _WI_LAYOUT_TRANSPOSE if transpose else _WI_LAYOUT_AS_IS, base, 0, code, size, len(src)]
            for j in range(_WI_MAX_RANK):  # cpu-route: model graph lowered on the host (one per source axis)
                rec.append(src[j] if j < len(src) else 0)
            rec.append(_WI_STRIDE_C)
            for _ in range(_WI_MAX_RANK):  # cpu-route: model graph lowered on the host (stride words)
                rec.append(0)
            if len(rec) != WI_RECORD:
                raise _fatal("the weight import record holds " + String(len(rec)) + " words")
            var ws = wi_source(rec, 0, self.g[].st.text(t.name))
            var view = session_load_check(ws, rec[2], rn_dims(self.p[].shape_nd(h)))
            self.p[].load_into(h, view)
            return h
        # a view of the raw bytes, cast to float32 on the executor by the upload
        var kind: Int
        if t.dtype == DT_F32 or t.dtype == DT_F16 or t.dtype == DT_F64:
            kind = CAST_KIND_FLOAT
        elif t.dtype == DT_BOOL:
            kind = CAST_KIND_BOOL
        else:
            kind = CAST_KIND_SIGNED
        return self.own(self._upload_any(base, kind, itemsize(t.dtype), shape))

    def upload_ints(mut self, values: List[Int], shape: List[Int]) raises -> Int:
        """Folded description integers (a shape read as data) as a tensor."""
        var buf = values.copy()
        var h = self._upload_any(Int(buf.unsafe_ptr()), CAST_KIND_SIGNED, 8, _session_shape(shape))
        _ = buf^
        return self.own(h)

    def const_scalar(mut self, value: Float64) raises -> Int:
        """A (1, 1) constant of an attribute number (epsilon, a fill), as a parameter."""
        return self.persist(self.full([1, 1], value))

    def fit(mut self, h: Int, name: Int) raises -> Int:
        """h reshaped to value name's session shape when the call left it in
        another shape of the same size."""
        var want = _session_shape(self.c.shape(name))
        var have = self.shape_h(h)
        if _same(have, want):
            return h
        if _numel(have) != _numel(want):
            raise Error("the lowering made " + fmt_tuple(have) + " for '" + self.g[].st.text(name)
                        + "', the graph says " + fmt_tuple(want))
        return self.reshape(h, want)

    def rank(self, name: Int) raises -> Int:
        return len(self.c.shape(name))

    def x(mut self, n: GNode, k: Int) raises -> Int:
        return self.get(_in(n, k))

    # ---------------------------------------------------------- the pass
    def run(mut self, mut inputs: List[Int], mut outputs: List[Int]) raises:
        for q in range(len(self.a.inputs)):  # driver-loop: over graph inputs
            var name = self.g[].inputs[self.a.inputs[q]].name
            var shp = _session_shape(self.c.shape(name))
            var h: Int
            if name in self.a.index_inputs:
                # an ids placeholder (id 0 is in range of every table); capture names it an
                # ids input, and it is freed after the capture with the other unread uploads
                var z = List[Int32](length=_numel(shp), fill=Int32(0))
                var rows = shp[0]
                var cols = shp[1]
                if len(shp) > 2:
                    cols = shp[len(shp) - 1]
                    rows = _numel(shp) // cols
                var src = CastIn(Int(z.unsafe_ptr()), CAST_KIND_SIGNED, 4, rows, cols, cols * 4, 4)
                var d = shp.copy()
                if len(shp) == 2:
                    d = [src.rows, src.cols]
                h = self.own(self.p[].upload_ids(src, rn_shape(d, "ops: ids")))
                _ = z^
            else:
                h = self.full(shp, 0.0)  # a placeholder; capture names it an input
            self.h[name] = h
            inputs.append(h)
        for q in range(len(self.a.needed)):  # driver-loop: over graph nodes
            var slot = self.a.needed[q]
            var all_c = True
            for j in range(len(self.g[].nodes[slot].inputs)):  # cpu-route: model graph lowered on the host (node inputs)
                var i = self.g[].nodes[slot].inputs[j]
                if i != S_EMPTY and i not in self.a.const:
                    all_c = False
            self.cur_const = all_c
            var outs = self.lower(slot)
            ref n = self.g[].nodes[slot]
            var m = len(outs) if len(outs) < len(n.outputs) else len(n.outputs)
            for j in range(m):  # cpu-route: model graph lowered on the host (node outputs)
                if n.outputs[j] != S_EMPTY and outs[j] != _NO_HANDLE:
                    self.h[n.outputs[j]] = outs[j]
        self.cur_const = False
        for i in range(len(self.g[].outputs)):  # driver-loop: over graph outputs
            outputs.append(self.get(self.g[].outputs[i].name))

    def lower(mut self, slot: Int) raises -> List[Int]:
        """The session calls of node `slot` (importers/_lower.py _l_*):
        one handle per output, _NO_HANDLE for an output it leaves unset."""
        var n = self.g[].nodes[slot].copy()
        var op = self.g[].st.text(n.op)
        var res = List[Int]()
        var rest = len(n.outputs) - 1
        var uc = _unary_code(op)
        if uc != 0:
            res.append(self.op1(uc, self.x(n, 0)))
            return res^
        if op == "Not":
            res.append(self.opk(OPS_LOGICAL, self.x(n, 0), [0]))
            return res^
        if _is_binary(op):
            var ha = self.x(n, 0)
            var hb = self.x(n, 1)
            if op == "Add":
                res.append(self.op2(OPS_ADD, ha, hb))
            elif op == "Sub":
                res.append(self.op2(OPS_SUB, ha, hb))
            elif op == "Mul":
                res.append(self.op2(OPS_MUL, ha, hb))
            elif op == "Div":
                res.append(self.op2(OPS_DIV, ha, hb))
            elif op == "And" or op == "Or":
                res.append(self.op(OPS_LOGICAL, [ha, hb], [1 if op == "And" else 2], List[Float32]()))
            else:
                var kind = 0
                if op == "Greater":
                    kind = 1
                elif op == "Less":
                    kind = 2
                elif op == "GreaterOrEqual":
                    kind = 3
                elif op == "LessOrEqual":
                    kind = 4
                res.append(self.op(OPS_COMPARE, [ha, hb], [kind], List[Float32]()))
            return res^
        if _is_reduce(op):
            res.append(self._reduce(n, op))
            return res^
        if op == "Reshape" or op == "Flatten" or op == "Squeeze" or op == "Unsqueeze":
            # the resolved shape, row-major
            res.append(self.fit(self.x(n, 0), n.outputs[0]))
            return res^
        if op == "Shape" or op == "Size":
            # every output is folded at import
            for _ in range(len(n.outputs)):  # cpu-route: model graph lowered on the host (node outputs)
                res.append(_NO_HANDLE)
            return res^
        if op == "Identity" or op == "Dropout" or op == "Cast":
            res.append(self.x(n, 0))
            for _ in range(rest):  # cpu-route: model graph lowered on the host (node outputs)
                res.append(_NO_HANDLE)
            return res^
        if op == "Neg":
            res.append(self.scale(self.x(n, 0), -1.0))
            return res^
        if op == "LeakyRelu":
            res.append(self.opf(OPS_LEAKY_RELU, self.x(n, 0), _af(self.g[], n, "alpha", 0.01)))
            return res^
        if op == "Elu":
            res.append(self.opf(OPS_ELU, self.x(n, 0), _af(self.g[], n, "alpha", 1.0)))
            return res^
        if op == "HardSigmoid":
            var hx = self.x(n, 0)
            var al = _af(self.g[], n, "alpha", 0.2)
            var be = _af(self.g[], n, "beta", 0.5)
            res.append(self.op(OPS_HARD_SIGMOID, [hx], List[Int](), [Float32(al), Float32(be)]))
            return res^
        if op == "Gelu":
            var hx = self.x(n, 0)
            var ap = _as(self.g[], n, "approximate", "none")
            if ap != "none" and ap != "tanh":
                raise Error("approximate must be 'none' or 'tanh'")
            res.append(self.opk(OPS_GELU, hx, [0 if ap == "none" else 1]))
            return res^
        if op == "Pow":
            var e = _ints(self.c, self.g[], _in(n, 1))
            if e.ok and len(e.v) == 1 and self.rank(n.inputs[1]) <= self.rank(n.inputs[0]):
                # an integer exponent is description: the number goes in the call
                res.append(self.opf(OPS_POW, self.x(n, 0), Float64(e.v[0])))
            else:
                var hx = self.x(n, 0)
                var he = self.x(n, 1)
                res.append(self.op2(OPS_POW, hx, he))
            return res^
        if op == "Max" or op == "Min":
            var code = OPS_MAXIMUM if op == "Max" else OPS_MINIMUM
            var ins = List[Int]()
            for j in range(len(n.inputs)):  # cpu-route: model graph lowered on the host (the operands)
                if n.inputs[j] != S_EMPTY:
                    ins.append(n.inputs[j])
            var h = self.get(ins[0])
            for j in range(1, len(ins)):  # driver-loop: over the operands of one node
                var hj = self.get(ins[j])
                h = self.op2(code, h, hj)
            res.append(self.fit(h, n.outputs[0]))
            return res^
        if op == "Where":
            var hc = self.x(n, 0)
            var ha = self.x(n, 1)
            var hb = self.x(n, 2)
            res.append(self.where(hc, ha, hb))
            return res^
        if op == "Clip":
            var h = self.x(n, 0)
            if _has_in(n, 1):
                var lo = self.x(n, 1)
                h = self.op2(OPS_MAXIMUM, h, lo)
            if _has_in(n, 2):
                var hi = self.x(n, 2)
                h = self.op2(OPS_MINIMUM, h, hi)
            res.append(self.fit(h, n.outputs[0]))
            return res^
        if op == "MatMul":
            var a_ = _in(n, 0)
            var b_ = _in(n, 1)
            var ha = self.get(a_)
            var hb = self.get(b_)
            if self.rank(b_) == 1:
                # a [k] right operand is a (k, 1) column (its session row is (1, k))
                var col = self.reshape(hb, [self.c.shape(b_)[0], 1])
                hb = self.kc(col, b_)
            res.append(self.fit(self.op2(OPS_MATMUL, ha, hb), n.outputs[0]))
            return res^
        if op == "Gemm":
            res.append(self._gemm(n))
            return res^
        if op == "Softmax" or op == "LogSoftmax":
            var xn = _in(n, 0)
            var r = self.rank(xn)
            var ax = _sax(_ai(self.g[], n, "axis", -1), r)
            var srank = len(_session_shape(self.c.shape(xn)))
            var hx = self.get(xn)
            if ax == srank - 1:
                res.append(self.op1(OPS_SOFTMAX if op == "Softmax" else OPS_LOG_SOFTMAX, hx))
            else:
                res.append(self.opk(OPS_SOFTMAX_N if op == "Softmax" else OPS_LOG_SOFTMAX_N, hx, [ax]))
            return res^
        if op == "LayerNormalization":
            var xn = _in(n, 0)
            var xs = self.c.shape(xn)
            var d = xs[len(xs) - 1]
            var w = self._row_param(_in(n, 1), d)
            var b: Int
            if _has_in(n, 2):
                b = self._row_param(n.inputs[2], d)
            else:
                b = self.persist(self.full([1, d], 0.0))
            var hx = self.get(xn)
            var eps = _af(self.g[], n, "epsilon", 1e-5)
            if not eps > 0:
                raise Error("eps must be positive")
            res.append(self.op(OPS_LAYER_NORM, [hx, w, b], List[Int](), [Float32(eps)]))
            for _ in range(rest):  # cpu-route: model graph lowered on the host (node outputs)
                res.append(_NO_HANDLE)
            return res^
        if op == "RMSNormalization":
            var xn = _in(n, 0)
            var sh = _session_shape(self.c.shape(xn))
            var hx = self.get(xn)
            var sq = self.op2(OPS_MUL, hx, hx)
            var ms = self.reduce(OPS_MEAN_N, sq, [len(sh) - 1])  # a mean on the executor through the session
            var ep = self.const_scalar(_af(self.g[], n, "epsilon", 1e-5))
            var den = self.op1(OPS_SQRT, self.op2(OPS_ADD, ms, ep))
            var q_ = self.op2(OPS_DIV, hx, den)
            var w = self._row_param(_in(n, 1), sh[len(sh) - 1])
            res.append(self.op2(OPS_MUL, q_, w))
            for _ in range(rest):  # cpu-route: model graph lowered on the host (node outputs)
                res.append(_NO_HANDLE)
            return res^
        if op == "BatchNormalization":
            var sh = self.c.shape(_in(n, 0))
            var ch = sh[1]
            var r = len(sh)
            var sc = self.x(n, 1)
            var bias = self.x(n, 2)
            var mean = self.x(n, 3)
            var var_ = self.x(n, 4)
            # the prologue: scale / sqrt(var + eps) and bias - mean * that, once at import
            var ep = self.const_scalar(_af(self.g[], n, "epsilon", 1e-5))
            var fs = self.op2(OPS_DIV, sc, self.op1(OPS_SQRT, self.op2(OPS_ADD, var_, ep)))
            var ft = self.op2(OPS_SUB, bias, self.op2(OPS_MUL, mean, fs))
            fs = self.k(self._channel_param(fs, ch, r))
            ft = self.k(self._channel_param(ft, ch, r))
            var hx = self.x(n, 0)
            res.append(self.op2(OPS_ADD, self.op2(OPS_MUL, hx, fs), ft))
            for _ in range(rest):  # cpu-route: model graph lowered on the host (node outputs)
                res.append(_NO_HANDLE)
            return res^
        if op == "InstanceNormalization":
            var sh = self.c.shape(_in(n, 0))
            var ch = sh[1]
            var r = len(sh)
            var y = self._normalize_last(self.x(n, 0), _range(2, r), _af(self.g[], n, "epsilon", 1e-5))
            y = self.op2(OPS_MUL, y, self._const_channel(_in(n, 1), ch, r))
            res.append(self.op2(OPS_ADD, y, self._const_channel(_in(n, 2), ch, r)))
            return res^
        if op == "GroupNormalization":
            var sh = self.c.shape(_in(n, 0))
            var nb = sh[0]
            var ch = sh[1]
            var r = len(sh)
            var gr = _ai(self.g[], n, "num_groups", 0)
            var xr = self.reshape(self.x(n, 0), [nb, gr, _numel(sh) // (nb * gr)])
            var y = self._normalize_last(xr, [2], _af(self.g[], n, "epsilon", 1e-5))
            var per_group = _numel(self.c.shape(_in(n, 1))) == gr and gr != ch
            if per_group:
                y = self.op2(OPS_MUL, y, self._const_channel(_in(n, 1), gr, 3))
                y = self.op2(OPS_ADD, y, self._const_channel(_in(n, 2), gr, 3))
                res.append(self.reshape(y, sh))
                return res^
            y = self.reshape(y, sh)
            y = self.op2(OPS_MUL, y, self._const_channel(_in(n, 1), ch, r))
            res.append(self.op2(OPS_ADD, y, self._const_channel(_in(n, 2), ch, r)))
            return res^
        if op == "Transpose":
            var xn = _in(n, 0)
            var r = self.rank(xn)
            if r < 2:
                res.append(self.get(xn))
                return res^
            var perm = List[Int]()
            var given = List[Int]()
            if _ail(self.g[], n, "perm", given):
                for j in range(len(given)):  # cpu-route: model graph lowered on the host (one per axis)
                    perm.append(_norm(given[j], r))
            else:
                var j = r - 1
                while j >= 0:  # cpu-route: model graph lowered on the host (one per axis)
                    perm.append(j)
                    j -= 1
            if _same(perm, _range(0, r)):
                res.append(self.get(xn))
                return res^
            res.append(self.opk(OPS_PERMUTE, self.get(xn), _cat1(0, perm)))
            return res^
        if op == "Expand":
            var h = self.x(n, 0)
            var want = _session_shape(self.c.shape(n.outputs[0]))
            if _same(self.shape_h(h), want):
                res.append(h)
            else:
                res.append(self.opk(OPS_EXPAND, h, want))
            return res^
        if op == "Concat":
            var ins = List[Int]()
            for j in range(len(n.inputs)):  # cpu-route: model graph lowered on the host (the operands)
                if n.inputs[j] != S_EMPTY:
                    ins.append(n.inputs[j])
            if len(ins) == 1:
                res.append(self.get(ins[0]))
                return res^
            var r = self.rank(ins[0])
            var hs = List[Int]()
            for j in range(len(ins)):  # driver-loop: over the operands of one node
                hs.append(self.get(ins[j]))
            res.append(self.concat(hs, _sax(_ai(self.g[], n, "axis", 0), r)))
            return res^
        if op == "Split":
            var xn = _in(n, 0)
            var sh = self.c.shape(xn)
            var r = len(sh)
            var ax = _norm(_ai(self.g[], n, "axis", 0), r)
            var sizes = List[Int]()
            for j in range(len(n.outputs)):  # cpu-route: model graph lowered on the host (one per piece)
                sizes.append(self.c.shape(n.outputs[j])[ax])
            if len(sizes) == 1:
                res.append(self.get(xn))
                return res^
            var hx = self.get(xn)
            return self.split(hx, sizes, _sax(ax, r))
        if op == "Slice":
            var sh = self.c.shape(_in(n, 0))
            var r = len(sh)
            var starts = _ints(self.c, self.g[], _in(n, 1))
            var ends = _ints(self.c, self.g[], _in(n, 2))
            if not starts.ok or not ends.ok:
                raise _fatal("the starts or ends of " + self.g[].st.text(n.name) + " are not static")
            var axes: List[Int]
            if _has_in(n, 3):
                var av = _ints(self.c, self.g[], n.inputs[3])
                if not av.ok:
                    raise _fatal("the axes of " + self.g[].st.text(n.name) + " are not static")
                axes = av.v.copy()
            else:
                axes = _range(0, len(starts.v))
            var h = self.x(n, 0)
            var m = min(len(starts.v), min(len(ends.v), len(axes)))
            for j in range(m):  # driver-loop: over the sliced axes of one node
                var a_ = _norm(axes[j], r)
                var ln = 0
                var b = _slice_range(starts.v[j], ends.v[j], 1, sh[a_], ln)
                h = self._narrow(h, b, ln, sh[a_], _sax(a_, r))
                sh[a_] = ln
            res.append(self.fit(h, n.outputs[0]))
            return res^
        if op == "Gather":
            res.append(self._gather(n))
            return res^
        if op == "ArgMax":
            var xn = _in(n, 0)
            var o = n.outputs[0]
            # positions a later node reads stay float values; a returned output is int32 words
            var as_int = o in self.a.outs and o not in self.a.consumers
            if as_int:
                self.int32_out[o] = True
            var hx = self.get(xn)
            var ax = _sax(_ai(self.g[], n, "axis", 0), self.rank(xn))
            var y = self.opk(OPS_ARGMAX, hx, [ax, 1, 0 if as_int else 1])  # argmax on the executor through the session
            res.append(self.fit(y, o))
            return res^
        if op == "GlobalAveragePool" or op == "GlobalMaxPool":
            var r = self.rank(_in(n, 0))
            var hx = self.x(n, 0)
            res.append(self.reduce(OPS_MEAN_N if op == "GlobalAveragePool" else OPS_MAX, hx, _range(2, r)))
            return res^
        if op == "Conv":
            res.append(self._conv(n))
            return res^
        if op == "MaxPool" or op == "AveragePool":
            res.append(self._pool(n, op))
            for _ in range(rest):  # cpu-route: model graph lowered on the host (node outputs)
                res.append(_NO_HANDLE)
            return res^
        if op == "Pad":
            res.append(self._pad(n))
            return res^
        if op == "Trilu":
            res.append(self._trilu(n))
            return res^
        if op == "ConstantOfShape":
            var want = _session_shape(self.c.shape(n.outputs[0]))
            var vk = _last_attr(self.g[].st, n.attrs, "value")
            if vk < 0:
                res.append(self.full(want, 0.0))
                return res^
            if n.attrs[vk].kind != AK_TENSOR:
                raise _fatal("attribute 'value' of " + self.g[].st.text(n.name) + " is not a tensor")
            var ht = self.upload(n.attrs[vk].tensors[0].copy(), False)
            res.append(self.opk(OPS_EXPAND, ht, want))
            return res^
        if op == "Constant":
            var vk = _last_attr(self.g[].st, n.attrs, "value")
            if vk < 0 or n.attrs[vk].kind != AK_TENSOR:
                raise _fatal("Constant " + self.g[].st.text(n.name) + " has no tensor 'value'")
            res.append(self.upload(n.attrs[vk].tensors[0].copy(), False))
            return res^
        raise Error("no lowering for " + op)

    # ---------------------------------------------------------- op helpers
    def _row_param(mut self, name: Int, d: Int) raises -> Int:
        """A per-feature constant as a (1, d) row."""
        var h = self.get(name)
        if not _same(self.shape_h(h), [1, d]):
            h = self.reshape(h, [1, d])
            h = self.kc(h, name)
        return h

    def _channel_param(mut self, h: Int, ch: Int, rank: Int) raises -> Int:
        """A per-channel constant h as (1, c, 1, ...) for a rank-`rank` input."""
        var want = [1, ch]
        for _ in range(rank - 2):  # cpu-route: model graph lowered on the host (one per axis)
            want.append(1)
        if not _same(self.shape_h(h), want):
            return self.reshape(h, want)
        return h

    def _const_channel(mut self, name: Int, ch: Int, rank: Int) raises -> Int:
        var h = self._channel_param(self.get(name), ch, rank)
        return self.kc(h, name)

    def _normalize_last(mut self, h: Int, axes: List[Int], eps: Float64) raises -> Int:
        """(h - mean) / sqrt(var + eps) over `axes` (session axes), the
        statistics kept with length-1 axes."""
        var m = self.reduce(OPS_MEAN_N, h, axes)  # a mean on the executor through the session
        var d = self.op2(OPS_SUB, h, m)
        var v = self.reduce(OPS_MEAN_N, self.op2(OPS_MUL, d, d), axes)  # a mean on the executor through the session
        var ep = self.const_scalar(eps)
        return self.op2(OPS_DIV, d, self.op1(OPS_SQRT, self.op2(OPS_ADD, v, ep)))

    def _narrow(mut self, h: Int, start: Int, length: Int, dim: Int, axis: Int) raises -> Int:
        """Positions start .. start + length of session axis `axis` (a view)."""
        if start == 0 and length == dim:
            return h
        var sizes = List[Int]()
        if start != 0:
            sizes.append(start)
        if length != 0:
            sizes.append(length)
        if dim - start - length != 0:
            sizes.append(dim - start - length)
        var pieces = self.split(h, sizes, axis)
        return pieces[1 if start != 0 else 0]

    def _reduce(mut self, n: GNode, op: String) raises -> Int:
        var xn = _in(n, 0)
        var r = self.rank(xn)
        var axes = List[Int]()
        var have = False
        if len(n.inputs) > 1 and n.inputs[1] != S_EMPTY:
            var v = _ints(self.c, self.g[], n.inputs[1])
            if not v.ok:
                raise _fatal("the axes of " + self.g[].st.text(n.name) + " are not static")
            axes = v.v.copy()
            have = True
        else:
            have = _ail(self.g[], n, "axes", axes)
        var none = False
        if not have or len(axes) == 0:
            if _ai(self.g[], n, "noop_with_empty_axes", 0) != 0:
                none = True
            else:
                axes = _range(0, r)
        if r == 0:
            none = True
        var h = self.x(n, 0)
        if none:
            return self.fit(h, n.outputs[0])
        var sx = List[Int]()
        for j in range(len(axes)):  # cpu-route: model graph lowered on the host (one per axis)
            sx.append(_sax(axes[j], r))
        var y: Int
        if op == "ReduceSum":
            y = self.reduce(OPS_SUM_N, h, sx)  # a sum on the executor through the session
        elif op == "ReduceMean":
            y = self.reduce(OPS_MEAN_N, h, sx)  # a mean on the executor through the session
        elif op == "ReduceMax":
            y = self.reduce(OPS_MAX, h, sx)
        elif op == "ReduceMin":
            y = self.scale(self.reduce(OPS_MAX, self.scale(h, -1.0), sx), -1.0)
        else:  # ReduceL2
            y = self.op1(OPS_SQRT, self.reduce(OPS_SUM_N, self.op2(OPS_MUL, h, h), sx))  # a sum on the executor
        return self.fit(y, n.outputs[0])

    def _gemm(mut self, n: GNode) raises -> Int:
        var a_ = _in(n, 0)
        var b_ = _in(n, 1)
        var c_ = n.inputs[2] if len(n.inputs) > 2 else S_EMPTY
        var alpha = _af(self.g[], n, "alpha", 1.0)
        var beta = _af(self.g[], n, "beta", 1.0)
        var ha = self.get(a_)
        if _ai(self.g[], n, "transA", 0) != 0:
            ha = self.op1(OPS_TRANSPOSE, ha)
        var b_const = self.is_const(b_) and not self.cur_const
        var hb: Int
        if _ai(self.g[], n, "transB", 0) != 0:
            var t = self.g[].init_of(b_)
            if t >= 0 and len(self.g[].inits[t].dims) == 2 and _is_float(self.g[].inits[t].dtype):
                # torch's [out, in] linear weight, transposed once on the executor by the weight import
                var got = self.ht.get(b_)
                if got:
                    hb = got.value()
                else:
                    hb = self.upload(self.g[].inits[t].copy(), True)
                    self.ht[b_] = hb
            else:
                hb = self.op1(OPS_TRANSPOSE, self.get(b_))
                if b_const:
                    hb = self.k(hb)
        else:
            hb = self.get(b_)
        if alpha != 1.0:
            if b_const:
                hb = self.k(self.scale(hb, alpha))
            else:
                ha = self.scale(ha, alpha)
        var hc = _NO_HANDLE
        if c_ != S_EMPTY:
            hc = self.get(c_)
            if beta != 1.0:
                hc = self.scale(hc, beta)
                hc = self.kc(hc, c_)
        var nn = self.c.shape(n.outputs[0])[1]
        if hc != _NO_HANDLE:
            var cs = self.c.shape(c_)
            if _numel(cs) == nn and len(cs) <= 2 and (len(cs) < 2 or cs[0] == 1):
                if not _same(self.shape_h(hc), [1, nn]):
                    hc = self.reshape(hc, [1, nn])
                    hc = self.kc(hc, c_)
                return self.op(OPS_LINEAR, [ha, hb, hc], [0], List[Float32]())
        var y = self.op2(OPS_MATMUL, ha, hb)
        if hc != _NO_HANDLE:
            y = self.op2(OPS_ADD, y, hc)
        return y

    def _gather(mut self, n: GNode) raises -> Int:
        var xn = _in(n, 0)
        var ids = _in(n, 1)
        var sh = self.c.shape(xn)
        var r = len(sh)
        var ax = _norm(_ai(self.g[], n, "axis", 0), r)
        var dim = sh[ax]
        if ids in self.a.index_inputs:
            # ids from a model input: an embedding row lookup for a 2-D table on axis 0, else a take
            var hx = self.get(xn)
            var hi = self.get(ids)
            var y: Int
            if r == 2 and ax == 0:
                y = self.op(OPS_EMBEDDING, [hx, hi], [-1], List[Float32]())
            else:
                y = self.op(OPS_TAKE, [hx, hi], [_sax(ax, r)], List[Float32]())
            return self.fit(y, n.outputs[0])
        var iv = _ints(self.c, self.g[], ids)
        if not iv.ok:
            raise _fatal("the indices of " + self.g[].st.text(n.name) + " are not static")
        var idx = List[Int]()
        for j in range(len(iv.v)):  # cpu-route: model graph lowered on the host (constant index integers)
            idx.append(iv.v[j] + dim if iv.v[j] < 0 else iv.v[j])
        if self.cur_const:
            # a constant table: one take at import (the program reads the result as a parameter)
            var hx = self.get(xn)
            var cnt = len(idx)
            # Session.take with an int64 index array: its 7-word input descriptor, then the axis
            var ints = [Int(idx.unsafe_ptr()), _NPY_INT64, 8, cnt, 1, 8, 8, _sax(ax, r)]
            var y = self.opk(OPS_TAKE, hx, ints)
            _ = idx^
            return self.fit(y, n.outputs[0])
        # runs of consecutive positions, each one view of the input, joined in order
        var rb = List[Int]()
        var rl = List[Int]()
        for j in range(len(idx)):  # cpu-route: model graph lowered on the host (constant index integers)
            var last = len(rb) - 1
            if last >= 0 and rb[last] + rl[last] == idx[j]:
                rl[last] += 1
            else:
                rb.append(idx[j])
                rl.append(1)
        var h = self.get(xn)
        var pieces = List[Int]()
        for j in range(len(rb)):  # driver-loop: over the index runs of one node
            pieces.append(self._narrow(h, rb[j], rl[j], dim, _sax(ax, r)))
        var out: Int
        if len(pieces) == 1:
            out = pieces[0]
        else:
            out = self.concat(pieces, _sax(ax, r))
        return self.fit(out, n.outputs[0])

    def _conv(mut self, n: GNode) raises -> Int:
        var xn = _in(n, 0)
        var wn = _in(n, 1)
        var bn = n.inputs[2] if _has_in(n, 2) else S_EMPTY
        var sh = self.c.shape(xn)
        var w = _window(self.g[], self.c, n)
        var stride = _at(w.st, 0)
        var group = _ai(self.g[], n, "group", 1)
        var hb = _NO_HANDLE
        if bn != S_EMPTY:
            hb = self._row_param(bn, self.c.shape(wn)[0])
        if len(sh) == 4:
            if _all_equal(w.pads):
                var hx = self.get(xn)
                var hw = self.get(wn)
                return self.conv2d(hx, hw, hb, stride, w.pads[0], group)
            # unequal pads (explicit, or SAME with an odd total): the zeros
            # placed first, then the unpadded convolution reads the same words
            var hx = self.pad(self.get(xn), _pad_counts(w.pads), 0.0)
            var hw = self.get(wn)
            return self.conv2d(hx, hw, hb, stride, 0, group)
        # 1-D: (N, C, L) as (N, C, 1, L), padded on the length axis only
        var nb = sh[0]
        var ch = sh[1]
        var ln = sh[2]
        var ws = self.c.shape(wn)
        var hx = self.reshape(self.get(xn), [nb, ch, 1, ln])
        var p0 = _at(w.pads, 0)
        var p1 = _at(w.pads, 1)
        if p0 != 0 or p1 != 0:
            hx = self.pad(hx, [0, 0, 0, p0, 0, 0, 0, p1], 0.0)
        var hw = self.reshape(self.get(wn), [ws[0], ws[1], 1, ws[2]])
        hw = self.kc(hw, wn)
        return self.fit(self.conv2d(hx, hw, hb, stride, 0, group), n.outputs[0])

    def _pool(mut self, n: GNode, op: String) raises -> Int:
        var w = _window(self.g[], self.c, n)
        var kk = _at(w.k, 0)
        var stride = _at(w.st, 0)
        var any_pad = False
        for j in range(len(w.pads)):  # cpu-route: model graph lowered on the host (four pads)
            if w.pads[j] != 0:
                any_pad = True
        var code = OPS_MAXPOOL2D if op == "MaxPool" else OPS_AVGPOOL2D
        if not any_pad:
            return self.opk(code, self.x(n, 0), [kk, stride])
        var counts = _pad_counts(w.pads)
        if op == "MaxPool":
            # ONNX pads a max pool with values that never win: -inf, and every
            # window holds an input value (each pad below the kernel size)
            var ninf = Float64(bitcast[DType.float32](UInt32(0xFF800000)))
            return self.opk(OPS_MAXPOOL2D, self.pad(self.x(n, 0), counts, ninf), [kk, stride])
        if _ai(self.g[], n, "count_include_pad", 0) != 0:
            # the padding counts in every window: zeros placed, then the mean of kernel_shape^2 values
            return self.opk(OPS_AVGPOOL2D, self.pad(self.x(n, 0), counts, 0.0), [kk, stride])
        # count_include_pad=0: each window's sum over the zero-padded input (a
        # depthwise convolution with ones) divided by the window's number of
        # input values, a count map from the same convolution over a ones
        # image padded with zeros (built on the executor once, a parameter)
        var sh = self.c.shape(_in(n, 0))
        var ch = sh[1]
        var hh = sh[2]
        var ww = sh[3]
        var ones_w = self.full([ch, 1, kk, kk], 1.0)
        if not self.cur_const:
            ones_w = self.persist(ones_w)
        var xp = self.pad(self.x(n, 0), counts, 0.0)
        var total = self.conv2d(xp, ones_w, _NO_HANDLE, stride, 0, ch)
        var image = self.pad(self.full([1, 1, hh, ww], 1.0), counts, 0.0)
        var ones_k = self.full([1, 1, kk, kk], 1.0)
        var count = self.conv2d(image, ones_k, _NO_HANDLE, stride, 0, 1)
        if not self.cur_const:
            count = self.persist(count)
        return self.op2(OPS_DIV, total, count)

    def _pad(mut self, n: GNode) raises -> Int:
        var xn = _in(n, 0)
        var sh = self.c.shape(xn)
        var r = len(sh)
        var pads: List[Int]
        if _has_in(n, 1):
            var pv = _ints(self.c, self.g[], n.inputs[1])
            if not pv.ok:
                raise _fatal("the pads of " + self.g[].st.text(n.name) + " are not static")
            pads = pv.v.copy()
        else:
            pads = List[Int]()
            _ = _ail(self.g[], n, "pads", pads)
        var axes: List[Int]
        if _has_in(n, 3):
            var av = _ints(self.c, self.g[], n.inputs[3])
            if not av.ok:
                raise _fatal("the axes of " + self.g[].st.text(n.name) + " are not static")
            axes = av.v.copy()
        else:
            axes = _range(0, r)
        var sshape = _session_shape(sh)
        var sr = len(sshape)
        var before = List[Int](length=sr, fill=0)
        var after = List[Int](length=sr, fill=0)
        for j in range(len(axes)):  # cpu-route: model graph lowered on the host (one per padded axis)
            var sa = _sax(axes[j], r)
            before[sa] = _at(pads, j)
            after[sa] = _at(pads, len(axes) + j)
        var counts = before^
        for j in range(sr):  # cpu-route: model graph lowered on the host (one per axis)
            counts.append(after[j])
        var y = self.pad(self.x(n, 0), counts, 0.0)
        if _has_in(n, 2):
            # a fill value from the model: 1.0 marks the padded positions, where() puts the value there
            var mark = self.pad(self.full(sshape, 0.0), counts, 1.0)
            if not self.cur_const:
                mark = self.persist(mark)
            var fill = self.x(n, 2)
            y = self.where(mark, fill, y)
        return self.fit(y, n.outputs[0])

    def _trilu(mut self, n: GNode) raises -> Int:
        var sh = self.c.shape(_in(n, 0))
        var rows = sh[len(sh) - 2]
        var cols = sh[len(sh) - 1]
        var kd = 0
        if _has_in(n, 1):
            var kv = _ints(self.c, self.g[], n.inputs[1])
            if not kv.ok:
                raise _fatal("k of " + self.g[].st.text(n.name) + " is not static")
            kd = _at(kv.v, 0)
        var upper = _ai(self.g[], n, "upper", 1) != 0
        # the kept positions (column - row >= k above the diagonal, <= k below), a
        # description of positions, never of element values
        var keep = List[UInt8](length=rows * cols, fill=UInt8(0))
        for i in range(rows):  # cpu-route: model graph lowered on the host (matrix positions)
            for j in range(cols):  # cpu-route: model graph lowered on the host (matrix positions)
                var on = (j - i >= kd) if upper else (j - i <= kd)
                if on:
                    keep[i * cols + j] = UInt8(1)
        var mask = self.own(self._upload_any(Int(keep.unsafe_ptr()), CAST_KIND_UNSIGNED, 1, [rows, cols]))
        _ = keep^
        var zero = self.const_scalar(0.0)
        var hx = self.x(n, 0)
        return self.where(mask, hx, zero)


# ------------------------------------------------------------ the entry --

def _problems_out(found: List[_Found]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(found)):  # cpu-route: model graph lowered on the host (one per problem)
        var p = Python.list()
        p.append(PythonObject(found[i].name))
        if found[i].has_op:
            p.append(PythonObject(found[i].op))
        else:
            p.append(PythonObject(None))
        p.append(PythonObject(found[i].why))
        out.append(p)
    return out


def _dims_py(v: List[Int]) raises -> PythonObject:
    var d = Python.list()
    for i in range(len(v)):  # cpu-route: model graph lowered on the host (one per axis)
        d.append(PythonObject(v[i]))
    return d


def onnx_import_py[E: Exec & Movable & Deinitable, KIND: Int](
    h: PythonObject, sid: PythonObject, names: PythonObject, flat: PythonObject, addrs: PythonObject,
    source: PythonObject,
) raises -> PythonObject:
    """Import graph `h` into inference session `sid` (this module's
    docstring): resolve, refuse, lower, capture. Returns [status, ...]."""
    var gp = onnx_graph(Int(py=h))
    if gp[].status != ONNX_OK:
        raise Error("onnx_import: model graph " + String(Int(py=h)) + " was not read")
    var gnames = List[Int]()
    var gdims = List[List[Int]]()
    var pos = 0
    for i in range(len(names)):  # cpu-route: model graph lowered on the host (the shapes argument)
        gnames.append(gp[].st.intern(String(py=names[i])))
        var rank = Int(py=flat[pos])
        pos += 1
        var dims = List[Int]()
        for _ in range(rank):  # cpu-route: model graph lowered on the host (one per axis)
            dims.append(Int(py=flat[pos]))
            pos += 1
        gdims.append(dims^)
    var bufs = _pyint_list(addrs)
    var src = String(py=source)
    var res = Python.list()
    var c = RCtx()
    var no_rule = List[Int]()
    var rprobs = List[_RProblem]()
    try:
        _resolve(gp[], gnames, gdims, c, no_rule, rprobs)
    except e:
        var msg = String(e)
        res.append(PythonObject(IMPORT_BAD_VALUE))
        res.append(PythonObject(fatal_text(msg) if is_fatal(msg) else msg))
        return res
    var a = _analyse(gp[], c)
    var found = _check(gp[], c, a, rprobs, src)
    if len(found) > 0:
        res.append(PythonObject(IMPORT_PROBLEMS))
        res.append(_problems_out(found))
        return res
    # every problem was listed before anything reached the executor; now the session
    var in_meta = Python.list()
    for q in range(len(a.inputs)):  # cpu-route: model graph lowered on the host (graph inputs)
        ref v = gp[].inputs[a.inputs[q]]
        var m = Python.list()
        m.append(PythonObject(gp[].st.text(v.name)))
        m.append(_dims_py(c.shape(v.name)))
        m.append(PythonObject(dtype_name(v.dtype)))
        in_meta.append(m)
    var out_names = List[Int]()
    var out_shapes = List[List[Int]]()
    for i in range(len(gp[].outputs)):  # cpu-route: model graph lowered on the host (graph outputs)
        out_names.append(gp[].outputs[i].name)
        out_shapes.append(c.shape(gp[].outputs[i].name))
    var p = ops_session[E, KIND](Int(py=sid))
    var low = _Low[E](p, gp, c^, a^, bufs^)
    var ins = List[Int]()
    var outs = List[Int]()
    low.run(ins, outs)
    # Session.capture's call, kept as the imported program
    var pid = p[].capture(ins, outs)
    var prog = p[].program(pid)
    var info = program_info(pid, prog)
    # the computed tensors (placeholders, prologue intermediates) go; the
    # uploads the program does not read go too
    var keep = Dict[Int, Bool]()
    for j in range(len(prog.params)):  # cpu-route: model graph lowered on the host (program parameters)
        keep[prog.params[j]] = True
    p[].reset()
    for j in range(len(low.owned)):  # driver-loop: over the import's uploaded tensors
        if low.owned[j] not in keep:
            p[].free(low.owned[j])
    var out_meta = Python.list()
    for i in range(len(out_names)):  # cpu-route: model graph lowered on the host (graph outputs)
        var m = Python.list()
        m.append(PythonObject(gp[].st.text(out_names[i])))
        m.append(_dims_py(out_shapes[i]))
        m.append(PythonObject(1 if out_names[i] in low.int32_out else 0))
        out_meta.append(m)
    res.append(PythonObject(IMPORT_OK))
    res.append(info)
    res.append(in_meta)
    res.append(out_meta)
    return res


# ------------------------------------------------- a graph from words --

struct _Cur(Movable):
    var w: List[Int]
    var i: Int

    def __init__(out self, var w: List[Int]):
        self.w = w^
        self.i = 0

    def take(mut self) raises -> Int:
        if self.i >= len(self.w):
            raise Error("onnx_from_words: the words end early")
        var v = self.w[self.i]
        self.i += 1
        return v


def _le_int(addr: Int, size: Int, dt: Int) -> Int:
    """The little-endian integer of `size` bytes at `addr` (the reader's decode)."""
    var p = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    var u = UInt64(0)
    for b in range(size):  # cpu-route: model graph description (the bytes of one integer)
        u |= UInt64(p[b]) << UInt64(8 * b)
    if dt == DT_I64:
        return Int(bitcast[DType.int64](u))
    if dt == DT_I32:
        return Int(bitcast[DType.int32](u.cast[DType.uint32]()))
    return Int(u)


def _w_tensor_rec(mut c: _Cur, bufs: List[Int]) raises -> TRec:
    var name = c.take()
    var dt = c.take()
    var rank = c.take()
    var dims = List[Int]()
    for _ in range(rank):  # cpu-route: model graph description (one per axis)
        dims.append(c.take())
    var buf = c.take()
    var start = c.take()
    var end = c.take()
    if buf < 0 or buf >= len(bufs) or end < start:
        raise Error("onnx_from_words: a tensor names buffer " + String(buf))
    var t = TRec(name, dt, dims^, buf, start, end - start, List[Int]())
    if is_int_dtype(dt):
        var size = itemsize(dt)
        var count = t.numel()
        if count * size > end - start:
            raise Error("onnx_from_words: a tensor holds fewer bytes than its shape")
        var base = bufs[buf] + start
        for k in range(count):  # cpu-route: model graph description (integer description values)
            t.ints.append(_le_int(base + k * size, size, dt))
    return t^


def _w_gvalue(mut c: _Cur) raises -> GValue:
    var name = c.take()
    var dt = c.take()
    var rank = c.take()
    var dims = List[Int]()
    var dstr = List[Int]()
    for _ in range(rank):  # cpu-route: model graph description (one per axis)
        var is_str = c.take()
        var v = c.take()
        if is_str != 0:
            dims.append(0)
            dstr.append(v)
        else:
            dims.append(v)
            dstr.append(-1)
    return GValue(name, dt, dims^, dstr^)


def onnx_from_words_py(words: PythonObject, strings: PythonObject, floats: PythonObject,
                       addrs: PythonObject) raises -> PythonObject:
    """A graph handle from the `onnx_info` word layout (header, then the
    status-0 body), its strings (index 0 the empty string), its floats and
    the address of each buffer. Returns the handle (>= 1)."""
    var g = OnnxGraph()
    for i in range(len(strings)):  # cpu-route: model graph description (the graph's strings)
        var k = g.st.intern(String(py=strings[i]))
        if k != i:
            raise Error("onnx_from_words: string " + String(i) + " repeats an earlier one")
    var fl = List[Float64]()
    for i in range(len(floats)):  # cpu-route: model graph description (attribute numbers)
        fl.append(Float64(py=floats[i]))
    var bufs = _pyint_list(addrs)
    var c = _Cur(_pyint_list(words))
    if c.take() != 0x4F4E5852 or c.take() != 1 or c.take() != ONNX_OK:
        raise Error("onnx_from_words: not a status-0 graph of word layout 1")
    g.opset = c.take()
    var nb = c.take()
    if nb != len(bufs):
        raise Error("onnx_from_words: " + String(nb) + " buffers, " + String(len(bufs)) + " addresses")
    for _ in range(nb):  # cpu-route: model graph description (the buffers)
        g.buf_kind.append(c.take())
        g.buf_path.append(c.take())
        g.buf_size.append(c.take())
    for _ in range(c.take()):  # cpu-route: model graph description (graph inputs)
        g.inputs.append(_w_gvalue(c))
    for _ in range(c.take()):  # cpu-route: model graph description (graph outputs)
        g.outputs.append(_w_gvalue(c))
    for _ in range(c.take()):  # cpu-route: model graph description (graph nodes)
        var op = c.take()
        var dom = c.take()
        var name = c.take()
        var ins = List[Int]()
        for _ in range(c.take()):  # cpu-route: model graph description (node inputs)
            ins.append(c.take())
        var outs = List[Int]()
        for _ in range(c.take()):  # cpu-route: model graph description (node outputs)
            outs.append(c.take())
        var attrs = List[AttrVal]()
        for _ in range(c.take()):  # cpu-route: model graph description (node attributes)
            var an = c.take()
            var kind = c.take()
            var av = AttrVal.make(an, kind)
            if kind == AK_FLOAT:
                var fi = c.take()
                if fi < 0 or fi >= len(fl):
                    raise Error("onnx_from_words: a float index is out of range")
                av.f = fl[fi]
            elif kind == AK_INT:
                av.i = c.take()
            elif kind == AK_STRING:
                av.s = c.take()
            elif kind == AK_TENSOR:
                av.tensors.append(_w_tensor_rec(c, bufs))
            elif kind == AK_FLOATS:
                var first = c.take()
                var end = c.take()
                if first < 0 or end > len(fl) or end < first:
                    raise Error("onnx_from_words: a float range is out of range")
                for j in range(first, end):  # cpu-route: model graph description (attribute list)
                    av.floats.append(fl[j])
            elif kind == AK_INTS:
                for _ in range(c.take()):  # cpu-route: model graph description (attribute list)
                    av.ints.append(c.take())
            elif kind == AK_STRINGS:
                for _ in range(c.take()):  # cpu-route: model graph description (attribute list)
                    av.strs.append(c.take())
            elif kind == AK_TENSORS:
                for _ in range(c.take()):  # cpu-route: model graph description (attribute list)
                    av.tensors.append(_w_tensor_rec(c, bufs))
            else:
                raise Error("onnx_from_words: attribute kind " + String(kind))
            attrs.append(av^)
        g.nodes.append(GNode(op, dom, name, ins^, outs^, attrs^))
    for _ in range(c.take()):  # cpu-route: model graph description (initializers)
        g.set_init(_w_tensor_rec(c, bufs))
    var p = alloc[OnnxGraph](1)
    p.init_pointee_move(g^)
    var s = _ONNX_GRAPHS.get_or_create_ptr()
    var hid = s[].next_id
    s[].next_id += 1
    s[].ids.append(hid)
    s[].addr.append(Int(p))
    return PythonObject(hid)
