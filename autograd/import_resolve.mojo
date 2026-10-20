# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Static shape resolution of an ONNX import graph read by
autograd/import_reader.mojo: every value's shape and dtype from the inputs'
resolved shapes, with the integer shape subgraphs (`Shape`, `Gather` on
shapes, `Concat` of ints, `Range`, `ConstantOfShape`, integer
`Reshape`/`Slice`/`Unsqueeze`/`Cast`, integer arithmetic) folded into
description integers. The rules, their order of evaluation and the refusal
texts are importers/_graph.py `resolve`'s, so both readers give the same
`Resolved`.

`onnx_resolve(handle, names, flat)`: `names` the inputs given a shape,
`flat` their dims as (rank, d0 .. d(rank-1)) runs. Returns [words, strings]:
  MAGIC, VERSION, status (0, or 4 with a message string),
  n_shapes, (name, dtype, rank, dims ...) ...,
  n_folded, (name, count, values ...) ...,
  n_no_rule, names ...,
  n_problems, (name, op or -1, reason) ...
Never refuses a graph it cannot resolve: the problems (an input axis left
open, a data-dependent shape, inconsistent shapes) are listed and the nodes
with no shape rule named, for the lowering to report with its own refusals."""
# cpu-route: model file parsing. Resolving shapes is part of the explicit
# input step before the runtime: it walks nodes, dims and integer shape
# constants on the host, never tensor element values of floating type.
from std.python import Python, PythonObject

from autograd.import_reader import (
    AK_FLOAT, AK_FLOATS, AK_INT, AK_INTS, AK_STRING, AK_TENSOR, DT_BOOL, DT_F32, DT_I64, FATAL_PREFIX,
    GNode, GValue, ONNX_BAD_VALUE, ONNX_READ_MAGIC, ONNX_READ_VERSION, OnnxGraph, S_EMPTY, StrTab,
    _last_attr, dtype_name, fatal_text, fmt_list, fmt_tuple, ir_dtype, is_fatal, is_int_dtype,
    onnx_dtype_name, onnx_graph, py_repr,
)


@fieldwise_init
struct ROut(Copyable, Movable):
    """One output of a shape rule: static shape, dtype, folded integers."""
    var shape: List[Int]
    var dtype: Int
    var has_f: Bool
    var f: List[Int]


@fieldwise_init
struct MaybeInts(Copyable, Movable):
    var ok: Bool
    var v: List[Int]


def _out(shape: List[Int], dtype: Int) -> ROut:
    return ROut(shape.copy(), dtype, False, List[Int]())


def _outf(shape: List[Int], dtype: Int, f: MaybeInts) -> ROut:
    return ROut(shape.copy(), dtype, f.ok, f.v.copy())


def _fatal(msg: String) -> Error:
    return Error(String(FATAL_PREFIX) + msg)


struct RCtx(Movable):
    """What resolve knows: the shapes and dtypes (S, D) and the folded
    integers (F), each in Python dict order (first insertion keeps its place)."""
    var s_slot: Dict[Int, Int]
    var s_name: List[Int]
    var s_shape: List[List[Int]]
    var s_dtype: List[Int]
    var f_slot: Dict[Int, Int]
    var f_name: List[Int]
    var f_vals: List[List[Int]]

    def __init__(out self):
        self.s_slot = Dict[Int, Int]()
        self.s_name = List[Int]()
        self.s_shape = List[List[Int]]()
        self.s_dtype = List[Int]()
        self.f_slot = Dict[Int, Int]()
        self.f_name = List[Int]()
        self.f_vals = List[List[Int]]()

    def has(self, name: Int) -> Bool:
        return name in self.s_slot

    def set(mut self, name: Int, shape: List[Int], dtype: Int):
        var got = self.s_slot.get(name)
        if got:
            self.s_shape[got.value()] = shape.copy()
            self.s_dtype[got.value()] = dtype
        else:
            self.s_slot[name] = len(self.s_name)
            self.s_name.append(name)
            self.s_shape.append(shape.copy())
            self.s_dtype.append(dtype)

    def shape(self, name: Int) raises -> List[Int]:
        var got = self.s_slot.get(name)
        if not got:
            raise _fatal("no shape for value " + String(name))
        return self.s_shape[got.value()].copy()

    def dtype(self, name: Int) raises -> Int:
        var got = self.s_slot.get(name)
        if not got:
            raise _fatal("no dtype for value " + String(name))
        return self.s_dtype[got.value()]

    def set_f(mut self, name: Int, vals: List[Int]):
        var got = self.f_slot.get(name)
        if got:
            self.f_vals[got.value()] = vals.copy()
        else:
            self.f_slot[name] = len(self.f_name)
            self.f_name.append(name)
            self.f_vals.append(vals.copy())

    def maybe_ints(mut self, g: OnnxGraph, name: Int) -> MaybeInts:
        """The folded integers of `name` (an integer initializer's are
        recorded as folded on first use), or none."""
        var got = self.f_slot.get(name)
        if got:
            return MaybeInts(True, self.f_vals[got.value()].copy())
        var k = g.init_of(name)
        if k >= 0 and is_int_dtype(g.inits[k].dtype):
            self.set_f(name, g.inits[k].ints)
            return MaybeInts(True, g.inits[k].ints.copy())
        return MaybeInts(False, List[Int]())

    def ints(mut self, g: OnnxGraph, name: Int, what: String) raises -> List[Int]:
        var v = self.maybe_ints(g, name)
        if not v.ok:
            raise Error(what + " '" + g.st.text(name) + "' is computed from tensor values (a data-dependent shape)")
        return v.v.copy()


# ------------------------------------------------------------- attributes --

def _attr_ix(g: OnnxGraph, n: GNode, name: String) -> Int:
    return _last_attr(g.st, n.attrs, name)


def _has(g: OnnxGraph, n: GNode, name: String) -> Bool:
    return _attr_ix(g, n, name) >= 0


def _ai(g: OnnxGraph, n: GNode, name: String, default: Int) raises -> Int:
    var k = _attr_ix(g, n, name)
    if k < 0:
        return default
    if n.attrs[k].kind != AK_INT:
        raise _fatal("attribute '" + name + "' of " + g.st.text(n.name) + " is not an int")
    return n.attrs[k].i


def _ail(g: OnnxGraph, n: GNode, name: String, mut dst: List[Int]) raises -> Bool:
    """The ints attribute `name` into dst; False when absent."""
    var k = _attr_ix(g, n, name)
    if k < 0:
        return False
    if n.attrs[k].kind != AK_INTS:
        raise _fatal("attribute '" + name + "' of " + g.st.text(n.name) + " is not a list of ints")
    dst = n.attrs[k].ints.copy()
    return True


def _as(g: OnnxGraph, n: GNode, name: String, default: String) raises -> String:
    var k = _attr_ix(g, n, name)
    if k < 0:
        return default
    if n.attrs[k].kind != AK_STRING:
        raise _fatal("attribute '" + name + "' of " + g.st.text(n.name) + " is not a string")
    return g.st.text(n.attrs[k].s)


def _in(n: GNode, i: Int) raises -> Int:
    """Input i of n (Python's n.inputs[i])."""
    if i >= len(n.inputs):
        raise _fatal("node " + String(n.name) + " has no input " + String(i))
    return n.inputs[i]


def _has_in(n: GNode, i: Int) -> Bool:
    """`len(n.inputs) > i and n.inputs[i]`."""
    return len(n.inputs) > i and n.inputs[i] != S_EMPTY


# ----------------------------------------------------------------- shapes --

def _numel(s: List[Int]) -> Int:
    var n = 1
    for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
        n *= s[i]
    return n


def _slice(s: List[Int], a: Int, b: Int) -> List[Int]:
    """s[a:b] for 0 <= a, b clamped to len(s)."""
    var out = List[Int]()
    var e = b if b < len(s) else len(s)
    for i in range(a, e):  # cpu-route: model file parsing (one per axis)
        out.append(s[i])
    return out^


def _cat(a: List[Int], b: List[Int]) -> List[Int]:
    var out = a.copy()
    for i in range(len(b)):  # cpu-route: model file parsing (one per axis)
        out.append(b[i])
    return out^


def _norm_axis(a: Int, rank: Int, what: String = "axis") raises -> Int:
    var hi = rank if rank > 1 else 1
    if not (-rank <= a and a < hi):
        raise Error(what + " " + String(a) + " is out of range for rank " + String(rank))
    return a + rank if a < 0 else a


def _bcast(shapes: List[List[Int]]) raises -> List[Int]:
    var rank = 0
    for j in range(len(shapes)):  # cpu-route: model file parsing (the operands)
        if len(shapes[j]) > rank:
            rank = len(shapes[j])
    var out = List[Int]()
    for i in range(rank):  # cpu-route: model file parsing (one step per axis)
        var d = 1
        for j in range(len(shapes)):  # cpu-route: model file parsing (the operands)
            var k = i - (rank - len(shapes[j]))
            if k < 0 or shapes[j][k] == 1:
                continue
            if d != 1 and shapes[j][k] != d:
                var msg = String("shapes ")
                for q in range(len(shapes)):  # cpu-route: model file parsing (the operands)
                    if q > 0:
                        msg += ", "
                    msg += fmt_tuple(shapes[q])
                raise Error(msg + " do not broadcast")
            d = shapes[j][k]
        out.append(d)
    return out^


def _bcast2(a: List[Int], b: List[Int]) raises -> List[Int]:
    var s = List[List[Int]]()
    s.append(a.copy())
    s.append(b.copy())
    return _bcast(s)


def _strides(shape: List[Int]) -> List[Int]:
    var st = List[Int](length=len(shape), fill=1)
    var i = len(shape) - 2
    while i >= 0:  # cpu-route: model file parsing (one step per axis)
        st[i] = st[i + 1] * shape[i + 1]
        i -= 1
    return st^


def _bcast_values(vals: List[Int], shape: List[Int], out_shape: List[Int]) raises -> List[Int]:
    """Row-major folded integers of `vals` (shape `shape`) broadcast to `out_shape`."""
    if shape == out_shape:
        return vals.copy()
    var pad = List[Int]()
    for _ in range(len(out_shape) - len(shape)):  # cpu-route: model file parsing (one per axis)
        pad.append(1)
    for i in range(len(shape)):  # cpu-route: model file parsing (one per axis)
        pad.append(shape[i])
    var st = _strides(pad)
    for i in range(len(pad)):  # cpu-route: model file parsing (one per axis)
        if pad[i] == 1:
            st[i] = 0
    var ost = _strides(out_shape)
    var out = List[Int]()
    for flat in range(_numel(out_shape)):  # cpu-route: model file parsing (folded shape integers)
        var src = 0
        for i in range(len(ost)):  # cpu-route: model file parsing (one per axis)
            src += ((flat // ost[i]) % out_shape[i]) * st[i]
        if src >= len(vals):
            raise _fatal("folded values do not cover their shape")
        out.append(vals[src])
    return out^


def _int_div(a: Int, b: Int) raises -> Int:
    if b == 0:
        raise Error("integer division by zero in a shape computation")
    var q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _is_fold_bin(op: String) -> Bool:
    return (op == "Add" or op == "Sub" or op == "Mul" or op == "Div" or op == "Equal" or op == "Less"
            or op == "Greater" or op == "LessOrEqual" or op == "GreaterOrEqual" or op == "And" or op == "Or"
            or op == "Max" or op == "Min")


def _fold_bin(op: String, a: Int, b: Int) raises -> Int:
    if op == "Add":
        return a + b
    if op == "Sub":
        return a - b
    if op == "Mul":
        return a * b
    if op == "Div":
        return _int_div(a, b)
    if op == "Equal":
        return 1 if a == b else 0
    if op == "Less":
        return 1 if a < b else 0
    if op == "Greater":
        return 1 if a > b else 0
    if op == "LessOrEqual":
        return 1 if a <= b else 0
    if op == "GreaterOrEqual":
        return 1 if a >= b else 0
    if op == "And":
        return 1 if (a != 0 and b != 0) else 0
    if op == "Or":
        return 1 if (a != 0 or b != 0) else 0
    if op == "Max":
        return a if a >= b else b
    return a if a <= b else b


def _bool_result(op: String) -> Bool:
    return (op == "Equal" or op == "Less" or op == "Greater" or op == "LessOrEqual" or op == "GreaterOrEqual"
            or op == "And" or op == "Or" or op == "Not")


def _ceil_div(a: Int, b: Int) -> Int:
    """Python's -(-a // b)."""
    return -((-a) // b)


# ------------------------------------------------------------------ rules --

def _r_unary(mut c: RCtx, g: OnnxGraph, n: GNode, op: String) raises -> List[ROut]:
    var x = _in(n, 0)
    var dt = DT_BOOL if op == "Not" else c.dtype(x)
    var res = List[ROut]()
    var folded = MaybeInts(False, List[Int]())
    if op == "Identity" or op == "Neg" or op == "Abs" or op == "Not":
        var v = c.maybe_ints(g, x)
        if v.ok:
            folded.ok = True
            for i in range(len(v.v)):  # cpu-route: model file parsing (folded shape integers)
                var a = v.v[i]
                if op == "Identity":
                    folded.v.append(a)
                elif op == "Neg":
                    folded.v.append(-a)
                elif op == "Abs":
                    folded.v.append(abs(a))
                else:
                    folded.v.append(1 if a == 0 else 0)
    res.append(_outf(c.shape(x), dt, folded))
    return res^


def _r_dropout(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var res = List[ROut]()
    res.append(_out(s, c.dtype(_in(n, 0))))
    res.append(_out(s, DT_BOOL))
    while len(res) > len(n.outputs):  # cpu-route: model file parsing (node outputs)
        _ = res.pop()
    return res^


def _r_binary(mut c: RCtx, g: OnnxGraph, n: GNode, op: String) raises -> List[ROut]:
    var ins = List[Int]()
    for i in range(len(n.inputs)):  # cpu-route: model file parsing (the operands)
        if n.inputs[i] != S_EMPTY:
            ins.append(n.inputs[i])
    if len(ins) == 0:
        raise Error("no inputs")
    var shapes = List[List[Int]]()
    for i in range(len(ins)):  # cpu-route: model file parsing (the operands)
        shapes.append(c.shape(ins[i]))
    var out_shape = _bcast(shapes)
    var dt = DT_BOOL if _bool_result(op) else c.dtype(ins[0])
    var vals = List[MaybeInts]()
    var all_ok = True
    for i in range(len(ins)):  # cpu-route: model file parsing (the operands)
        var v = c.maybe_ints(g, ins[i])
        all_ok = all_ok and v.ok
        vals.append(v^)
    var folded = MaybeInts(False, List[Int]())
    if _is_fold_bin(op) and all_ok:
        var acc = _bcast_values(vals[0].v, shapes[0], out_shape)
        for j in range(1, len(ins)):  # cpu-route: model file parsing (the operands)
            var other = _bcast_values(vals[j].v, shapes[j], out_shape)
            var m = len(acc) if len(acc) < len(other) else len(other)
            var nxt = List[Int]()
            for k in range(m):  # cpu-route: model file parsing (folded shape integers)
                nxt.append(_fold_bin(op, acc[k], other[k]))
            acc = nxt^
        folded = MaybeInts(True, acc^)
    var res = List[ROut]()
    res.append(_outf(out_shape, dt, folded))
    return res^


def _r_where(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    if len(n.inputs) < 3:
        raise _fatal("Where needs three inputs")
    var cond = n.inputs[0]
    var a = n.inputs[1]
    var b = n.inputs[2]
    var shapes = List[List[Int]]()
    shapes.append(c.shape(cond))
    shapes.append(c.shape(a))
    shapes.append(c.shape(b))
    var out_shape = _bcast(shapes)
    var vc = c.maybe_ints(g, cond)
    var va = c.maybe_ints(g, a)
    var vb = c.maybe_ints(g, b)
    var folded = MaybeInts(False, List[Int]())
    if vc.ok and va.ok and vb.ok:
        var xc = _bcast_values(vc.v, shapes[0], out_shape)
        var xa = _bcast_values(va.v, shapes[1], out_shape)
        var xb = _bcast_values(vb.v, shapes[2], out_shape)
        folded.ok = True
        var m = len(xc)
        if len(xa) < m:
            m = len(xa)
        if len(xb) < m:
            m = len(xb)
        for k in range(m):  # cpu-route: model file parsing (folded shape integers)
            folded.v.append(xa[k] if xc[k] != 0 else xb[k])
    var res = List[ROut]()
    res.append(_outf(out_shape, c.dtype(a), folded))
    return res^


def _r_same(mut c: RCtx, n: GNode) raises -> List[ROut]:
    var res = List[ROut]()
    res.append(_out(c.shape(_in(n, 0)), c.dtype(_in(n, 0))))
    return res^


def _r_matmul(mut c: RCtx, n: GNode) raises -> List[ROut]:
    var a = c.shape(_in(n, 0))
    var b = c.shape(_in(n, 1))
    if len(a) == 0 or len(b) == 0:
        raise Error("MatMul of a scalar")
    var sq_a = len(a) == 1
    var sq_b = len(b) == 1
    var a2 = a.copy()
    if sq_a:
        a2 = List[Int]()
        a2.append(1)
        a2.append(a[0])
    var b2 = b.copy()
    if sq_b:
        b2.append(1)
    if a2[len(a2) - 1] != b2[len(b2) - 2]:
        raise Error("inner dimensions differ: " + fmt_tuple(a) + " @ " + fmt_tuple(b))
    var out = _bcast2(_slice(a2, 0, len(a2) - 2), _slice(b2, 0, len(b2) - 2))
    out.append(a2[len(a2) - 2])
    out.append(b2[len(b2) - 1])
    if sq_a:
        var last = out[len(out) - 1]
        _ = out.pop()
        _ = out.pop()
        out.append(last)
    if sq_b:
        _ = out.pop()
    var res = List[ROut]()
    res.append(_out(out, c.dtype(_in(n, 0))))
    return res^


def _r_gemm(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var a = c.shape(_in(n, 0))
    var b = c.shape(_in(n, 1))
    if len(a) != 2 or len(b) != 2:
        raise Error("Gemm needs 2-D operands, got " + fmt_tuple(a) + " and " + fmt_tuple(b))
    var ta = _ai(g, n, "transA", 0)
    var tb = _ai(g, n, "transB", 0)
    var m = a[1] if ta != 0 else a[0]
    var k = a[0] if ta != 0 else a[1]
    var k2 = b[1] if tb != 0 else b[0]
    var nn = b[0] if tb != 0 else b[1]
    if k != k2:
        raise Error("inner dimensions differ: " + fmt_tuple(a) + " and " + fmt_tuple(b) + " (transA="
                    + String(ta) + ", transB=" + String(tb) + ")")
    var mn = List[Int]()
    mn.append(m)
    mn.append(nn)
    if len(n.inputs) > 2 and n.inputs[2] != S_EMPTY:
        var bs = c.shape(n.inputs[2])
        if len(_bcast2(bs, mn)) != 2:
            raise Error("bias " + fmt_tuple(bs) + " does not broadcast to " + fmt_tuple(mn))
    var res = List[ROut]()
    res.append(_out(mn, c.dtype(_in(n, 0))))
    return res^


def _r_norm_last(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var res = List[ROut]()
    res.append(_out(s, c.dtype(_in(n, 0))))
    if len(n.outputs) > 1:
        var ax = _norm_axis(_ai(g, n, "axis", -1), len(s))
        var stat = _slice(s, 0, ax)
        for _ in range(len(s) - ax):  # cpu-route: model file parsing (one per axis)
            stat.append(1)
        for _ in range(len(n.outputs) - 1):  # cpu-route: model file parsing (node outputs)
            res.append(_out(stat, DT_F32))
    return res^


def _r_batchnorm(mut c: RCtx, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var res = List[ROut]()
    res.append(_out(s, c.dtype(_in(n, 0))))
    if len(n.outputs) > 1:
        if len(s) < 2:
            raise Error("BatchNormalization needs a channel axis")
        var ch = List[Int]()
        ch.append(s[1])
        for _ in range(len(n.outputs) - 1):  # cpu-route: model file parsing (node outputs)
            res.append(_out(ch, c.dtype(_in(n, 0))))
    return res^


def _r_transpose(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var r = len(s)
    var perm = List[Int]()
    if not _ail(g, n, "perm", perm):
        var i = r - 1
        while i >= 0:  # cpu-route: model file parsing (one per axis)
            perm.append(i)
            i -= 1
    var seen = List[Int](length=r, fill=0)
    var bad = len(perm) != r
    for i in range(len(perm)):  # cpu-route: model file parsing (one per axis)
        var p = _norm_axis(perm[i], r, "perm entry")
        if not bad:
            seen[p] += 1
    for i in range(r):  # cpu-route: model file parsing (one per axis)
        if seen[i] != 1:
            bad = True
    if bad:
        raise Error("perm " + fmt_tuple(perm) + " is not a permutation of rank " + String(r))
    var out = List[Int]()
    for i in range(len(perm)):  # cpu-route: model file parsing (one per axis)
        out.append(s[_norm_axis(perm[i], r)])
    var v = c.maybe_ints(g, _in(n, 0))
    var folded = MaybeInts(False, List[Int]())
    if v.ok and r <= 1:
        folded = v^
    var res = List[ROut]()
    res.append(_outf(out, c.dtype(_in(n, 0)), folded))
    return res^


def _r_reshape(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var x = _in(n, 0)
    var s = c.shape(x)
    var target = c.ints(g, _in(n, 1), "the target shape")
    var allowzero = _ai(g, n, "allowzero", 0)
    var neg = List[Int]()
    for i in range(len(target)):  # cpu-route: model file parsing (one per axis)
        if target[i] == -1:
            neg.append(i)
    if len(neg) > 1:
        raise Error("target shape " + fmt_tuple(target) + " has more than one -1")
    for i in range(len(target)):  # cpu-route: model file parsing (one per axis)
        var d = target[i]
        if d == 0 and allowzero == 0:
            if i >= len(s):
                raise Error("target shape " + fmt_tuple(target) + ": 0 at axis " + String(i)
                            + " has no input axis to copy")
            target[i] = s[i]
        elif d < -1:
            raise Error("target shape " + fmt_tuple(target) + " has a negative size")
    var total = _numel(s)
    if len(neg) > 0:
        var rest = 1
        for i in range(len(target)):  # cpu-route: model file parsing (one per axis)
            if i != neg[0]:
                rest *= target[i]
        if rest == 0 or total % rest != 0:
            raise Error("cannot reshape " + fmt_tuple(s) + " to " + fmt_tuple(target))
        target[neg[0]] = total // rest
    if _numel(target) != total:
        raise Error("cannot reshape " + fmt_tuple(s) + " to " + fmt_tuple(target))
    var res = List[ROut]()
    res.append(_outf(target, c.dtype(x), c.maybe_ints(g, x)))
    return res^


def _r_flatten(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var ax0 = _ai(g, n, "axis", 1)
    var ax = ax0 + len(s) if ax0 < 0 else ax0
    if not (0 <= ax and ax <= len(s)):
        raise Error("axis " + String(ax0) + " is out of range for rank " + String(len(s)))
    var out = List[Int]()
    out.append(_numel(_slice(s, 0, ax)))
    out.append(_numel(_slice(s, ax, len(s))))
    var res = List[ROut]()
    res.append(_outf(out, c.dtype(_in(n, 0)), c.maybe_ints(g, _in(n, 0))))
    return res^


def _axes_of(mut c: RCtx, g: OnnxGraph, n: GNode, mut axes: List[Int]) raises -> Bool:
    """Axes from the second input (opset 13+) or the attribute; False when absent."""
    if _has_in(n, 1):
        axes = c.ints(g, n.inputs[1], "axes")
        return True
    return _ail(g, n, "axes", axes)


def _r_squeeze(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var axes = List[Int]()
    var keep = List[Int]()
    if not _axes_of(c, g, n, axes):
        for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
            if s[i] != 1:
                keep.append(s[i])
    else:
        var mark = List[Bool](length=len(s), fill=False)
        for i in range(len(axes)):  # cpu-route: model file parsing (one per axis)
            mark[_norm_axis(axes[i], len(s))] = True
        var bad = List[Int]()
        for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
            if mark[i] and s[i] != 1:
                bad.append(i)
        if len(bad) > 0:
            raise Error("cannot squeeze axes " + fmt_list(bad) + " of " + fmt_tuple(s) + " (not size 1)")
        for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
            if not mark[i]:
                keep.append(s[i])
    var res = List[ROut]()
    res.append(_outf(keep, c.dtype(_in(n, 0)), c.maybe_ints(g, _in(n, 0))))
    return res^


def _r_unsqueeze(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var axes = List[Int]()
    if not _axes_of(c, g, n, axes):
        raise Error("no axes")
    var r = len(s) + len(axes)
    var mark = List[Int](length=r if r > 0 else 0, fill=0)
    for i in range(len(axes)):  # cpu-route: model file parsing (one per axis)
        mark[_norm_axis(axes[i], r)] += 1
    for i in range(r):  # cpu-route: model file parsing (one per axis)
        if mark[i] > 1:
            raise Error("repeated axes " + fmt_tuple(axes))
    var out = List[Int]()
    var j = 0
    for i in range(r):  # cpu-route: model file parsing (one per axis)
        if mark[i] > 0:
            out.append(1)
        else:
            out.append(s[j])
            j += 1
    var res = List[ROut]()
    res.append(_outf(out, c.dtype(_in(n, 0)), c.maybe_ints(g, _in(n, 0))))
    return res^


def _r_expand(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var x = _in(n, 0)
    var target = c.ints(g, _in(n, 1), "the target shape")
    var sx = c.shape(x)
    var out = _bcast2(sx, target)
    var v = c.maybe_ints(g, x)
    var folded = MaybeInts(False, List[Int]())
    if v.ok:
        folded = MaybeInts(True, _bcast_values(v.v, sx, out))
    var res = List[ROut]()
    res.append(_outf(out, c.dtype(x), folded))
    return res^


def _r_concat(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var ins = List[Int]()
    for i in range(len(n.inputs)):  # cpu-route: model file parsing (the operands)
        if n.inputs[i] != S_EMPTY:
            ins.append(n.inputs[i])
    if len(ins) == 0:
        raise _fatal("Concat without inputs")
    var shapes = List[List[Int]]()
    for i in range(len(ins)):  # cpu-route: model file parsing (the operands)
        shapes.append(c.shape(ins[i]))
    var r = len(shapes[0])
    var ax = _norm_axis(_ai(g, n, "axis", 0), r)
    if ax >= r:
        raise _fatal("Concat of scalars")
    for j in range(1, len(shapes)):  # cpu-route: model file parsing (the operands)
        var bad = len(shapes[j]) != r
        if not bad:
            for i in range(r):  # cpu-route: model file parsing (one per axis)
                if i != ax and shapes[j][i] != shapes[0][i]:
                    bad = True
        if bad:
            var msg = String("inputs of shapes ")
            for q in range(len(shapes)):  # cpu-route: model file parsing (the operands)
                if q > 0:
                    msg += ", "
                msg += fmt_tuple(shapes[q])
            raise Error(msg + " do not concatenate on axis " + String(ax))
    var out = shapes[0].copy()
    var tot = 0
    for j in range(len(shapes)):  # cpu-route: model file parsing (the operands)
        tot += shapes[j][ax]
    out[ax] = tot
    var all_ok = True
    var vals = List[MaybeInts]()
    for i in range(len(ins)):  # cpu-route: model file parsing (the operands)
        var v = c.maybe_ints(g, ins[i])
        all_ok = all_ok and v.ok
        vals.append(v^)
    var folded = MaybeInts(False, List[Int]())
    if r == 1 and all_ok:
        folded.ok = True
        for j in range(len(vals)):  # cpu-route: model file parsing (the operands)
            for k in range(len(vals[j].v)):  # cpu-route: model file parsing (folded shape integers)
                folded.v.append(vals[j].v[k])
    var res = List[ROut]()
    res.append(_outf(out, c.dtype(ins[0]), folded))
    return res^


def _r_split(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var ax = _norm_axis(_ai(g, n, "axis", 0), len(s))
    if ax >= len(s):
        raise _fatal("Split of a scalar")
    var k = len(n.outputs)
    var sizes = List[Int]()
    if _has_in(n, 1):
        sizes = c.ints(g, n.inputs[1], "the split sizes")
    elif _ail(g, n, "split", sizes):
        pass
    else:
        var d = s[ax]
        if k == 0:
            raise _fatal("Split without outputs")
        if _has(g, n, "num_outputs"):
            var each = _ceil_div(d, k)
            for _ in range(k - 1):  # cpu-route: model file parsing (one per piece)
                sizes.append(each)
            sizes.append(d - each * (k - 1))
        elif d % k == 0:
            for _ in range(k):  # cpu-route: model file parsing (one per piece)
                sizes.append(d // k)
        else:
            raise Error("axis " + String(ax) + " of size " + String(d) + " does not split into " + String(k)
                        + " equal pieces")
    var tot = 0
    var mn = 0
    for i in range(len(sizes)):  # cpu-route: model file parsing (one per piece)
        tot += sizes[i]
        if i == 0 or sizes[i] < mn:
            mn = sizes[i]
    if len(sizes) != k or tot != s[ax] or mn < 0:
        raise Error("split sizes " + fmt_tuple(sizes) + " do not cover axis " + String(ax) + " of " + fmt_tuple(s)
                    + " in " + String(k) + " outputs")
    var res = List[ROut]()
    for i in range(len(sizes)):  # cpu-route: model file parsing (one per piece)
        var o = s.copy()
        o[ax] = sizes[i]
        res.append(_out(o, c.dtype(_in(n, 0))))
    return res^


def _slice_range(start0: Int, end0: Int, step: Int, dim: Int, mut length: Int) raises -> Int:
    """ONNX Slice clamping of one axis: returns start, sets length."""
    if step == 0:
        raise Error("Slice step 0")
    var start = start0
    var end = end0
    if start < 0:
        start += dim
    if end < 0:
        end += dim
    if step > 0:
        start = min(max(start, 0), dim)
        end = min(max(end, 0), dim)
        length = max(0, _ceil_div(end - start, step))
    else:
        start = min(max(start, 0), dim - 1)
        end = min(max(end, -1), dim - 1)
        length = max(0, _ceil_div(start - end, -step))
    return start


def _r_slice(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var x = _in(n, 0)
    var s = c.shape(x)
    var starts = c.ints(g, _in(n, 1), "the slice starts")
    var ends = c.ints(g, _in(n, 2), "the slice ends")
    var axes = List[Int]()
    if _has_in(n, 3):
        axes = c.ints(g, n.inputs[3], "the slice axes")
    else:
        for i in range(len(starts)):  # cpu-route: model file parsing (one per sliced axis)
            axes.append(i)
    var steps = List[Int]()
    if _has_in(n, 4):
        steps = c.ints(g, n.inputs[4], "the slice steps")
    else:
        for _ in range(len(starts)):  # cpu-route: model file parsing (one per sliced axis)
            steps.append(1)
    if not (len(starts) == len(ends) and len(ends) == len(axes) and len(axes) == len(steps)):
        raise Error("starts, ends, axes and steps differ in length")
    var out = s.copy()
    var pick_b = 0
    var pick_sp = 1
    var pick_ln = s[0] if len(s) > 0 else 0
    for i in range(len(starts)):  # cpu-route: model file parsing (one per sliced axis)
        var a = _norm_axis(axes[i], len(s))
        if a >= len(s):
            raise _fatal("Slice of a scalar")
        var ln = 0
        var b = _slice_range(starts[i], ends[i], steps[i], s[a], ln)
        out[a] = ln
        if a == 0:
            pick_b = b
            pick_sp = steps[i]
            pick_ln = ln
    var v = c.maybe_ints(g, x)
    var folded = MaybeInts(False, List[Int]())
    if v.ok and len(s) == 1:
        folded.ok = True
        for i in range(pick_ln):  # cpu-route: model file parsing (folded shape integers)
            var at = pick_b + i * pick_sp
            if at < 0 or at >= len(v.v):
                raise _fatal("folded values do not cover their shape")
            folded.v.append(v.v[at])
    var res = List[ROut]()
    res.append(_outf(out, c.dtype(x), folded))
    return res^


def _r_gather(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var data = _in(n, 0)
    var idx = _in(n, 1)
    var s = c.shape(data)
    var si = c.shape(idx)
    var ax = _norm_axis(_ai(g, n, "axis", 0), len(s))
    var out = _cat(_cat(_slice(s, 0, ax), si), _slice(s, ax + 1, len(s)))
    var iv = c.maybe_ints(g, idx)
    if iv.ok:
        if ax >= len(s):
            raise _fatal("Gather of a scalar")
        var bad = List[Int]()
        for i in range(len(iv.v)):  # cpu-route: model file parsing (folded shape integers)
            if not (-s[ax] <= iv.v[i] and iv.v[i] < s[ax]):
                bad.append(iv.v[i])
        if len(bad) > 0:
            raise Error("constant indices " + fmt_list(_slice(bad, 0, 4)) + " out of range for axis " + String(ax)
                        + " of size " + String(s[ax]))
    var dv = c.maybe_ints(g, data)
    var folded = MaybeInts(False, List[Int]())
    if dv.ok and iv.ok and len(s) == 1:
        folded.ok = True
        for k in range(len(iv.v)):  # cpu-route: model file parsing (folded shape integers)
            var i = iv.v[k]
            var at = i + s[0] if i < 0 else i
            if at < 0 or at >= len(dv.v):
                raise _fatal("folded values do not cover their shape")
            folded.v.append(dv.v[at])
    var res = List[ROut]()
    res.append(_outf(out, c.dtype(data), folded))
    return res^


def _r_gather_nd(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var si = c.shape(_in(n, 1))
    if _ai(g, n, "batch_dims", 0) != 0:
        raise Error("GatherND with batch_dims (only 0 is supported)")
    if len(si) == 0:
        raise Error("GatherND indices of rank 0")
    var k = si[len(si) - 1]
    if k < 1 or k > len(s):
        raise Error("GatherND index depth " + String(k) + " for a tensor of rank " + String(len(s)))
    var res = List[ROut]()
    res.append(_out(_cat(_slice(si, 0, len(si) - 1), _slice(s, k, len(s))), c.dtype(_in(n, 0))))
    return res^


def _r_reduce(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var axes = List[Int]()
    var have = _axes_of(c, g, n, axes)
    var keep = _ai(g, n, "keepdims", 1)
    var res = List[ROut]()
    if not have or len(axes) == 0:
        if _ai(g, n, "noop_with_empty_axes", 0) != 0:
            res.append(_out(s, c.dtype(_in(n, 0))))
            return res^
        axes = List[Int]()
        for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
            axes.append(i)
    var mark = List[Bool](length=len(s), fill=False)
    for i in range(len(axes)):  # cpu-route: model file parsing (one per axis)
        mark[_norm_axis(axes[i], len(s))] = True
    var out = List[Int]()
    for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
        if keep != 0:
            out.append(1 if mark[i] else s[i])
        elif not mark[i]:
            out.append(s[i])
    res.append(_out(out, c.dtype(_in(n, 0))))
    return res^


def _r_argmax(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var ax = _norm_axis(_ai(g, n, "axis", 0), len(s))
    var keep = _ai(g, n, "keepdims", 1)
    var out = List[Int]()
    for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
        if keep != 0:
            out.append(1 if i == ax else s[i])
        elif i != ax:
            out.append(s[i])
    var res = List[ROut]()
    res.append(_out(out, DT_I64))
    return res^


def _r_global_pool(mut c: RCtx, n: GNode, op: String) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    if len(s) < 3:
        raise Error(op + " needs a rank 3 or more input, got " + fmt_tuple(s))
    var out = _slice(s, 0, 2)
    for _ in range(len(s) - 2):  # cpu-route: model file parsing (one per spatial axis)
        out.append(1)
    var res = List[ROut]()
    res.append(_out(out, c.dtype(_in(n, 0))))
    return res^


def _window_out(g: OnnxGraph, n: GNode, s: List[Int], kernel: List[Int]) raises -> List[Int]:
    """Output spatial sizes of a Conv or pool window over s (N, C, spatial...)."""
    var sp = len(s) - 2
    var strides = List[Int]()
    if not _ail(g, n, "strides", strides):
        for _ in range(sp):  # cpu-route: model file parsing (one per spatial axis)
            strides.append(1)
    var dil = List[Int]()
    if not _ail(g, n, "dilations", dil):
        for _ in range(sp):  # cpu-route: model file parsing (one per spatial axis)
            dil.append(1)
    var pads = List[Int]()
    if not _ail(g, n, "pads", pads):
        for _ in range(2 * sp):  # cpu-route: model file parsing (one per spatial axis)
            pads.append(0)
    var auto = _as(g, n, "auto_pad", "NOTSET")
    var ceil = _ai(g, n, "ceil_mode", 0)
    if len(kernel) != sp or len(strides) != sp or len(dil) != sp or len(pads) != 2 * sp:
        raise Error("kernel " + fmt_tuple(kernel) + ", strides " + fmt_tuple(strides) + ", dilations "
                    + fmt_tuple(dil) + " or pads " + fmt_tuple(pads) + " do not match " + String(sp)
                    + " spatial axes")
    var out = List[Int]()
    for i in range(sp):  # cpu-route: model file parsing (one per spatial axis)
        if strides[i] == 0:
            raise _fatal("stride 0")
        var d = s[2 + i]
        var k = (kernel[i] - 1) * dil[i] + 1
        var o: Int
        if auto == "SAME_UPPER" or auto == "SAME_LOWER":
            o = _ceil_div(d, strides[i])
        elif auto == "VALID":
            o = (d - k) // strides[i] + 1
        else:
            var span = d + pads[i] + pads[sp + i] - k
            o = (_ceil_div(span, strides[i]) if ceil != 0 else span // strides[i]) + 1
            if ceil != 0 and (o - 1) * strides[i] >= d + pads[i]:
                o -= 1
        if o <= 0:
            raise Error("the window leaves no output along spatial axis " + String(i) + " of " + fmt_tuple(s))
        out.append(o)
    return out^


def _r_conv(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var w = c.shape(_in(n, 1))
    if len(s) < 3 or len(w) != len(s):
        raise Error("input " + fmt_tuple(s) + " and weight " + fmt_tuple(w) + " ranks do not match")
    var group = _ai(g, n, "group", 1)
    if s[1] != w[1] * group:
        raise Error("input channels " + String(s[1]) + " do not match weight " + fmt_tuple(w) + " with group="
                    + String(group))
    var wk = _slice(w, 2, len(w))
    var kernel = List[Int]()
    if not _ail(g, n, "kernel_shape", kernel):
        kernel = wk.copy()
    if kernel != wk:
        raise Error("kernel_shape " + fmt_tuple(kernel) + " does not match the weight " + fmt_tuple(w))
    if _has_in(n, 2):
        var bs = c.shape(n.inputs[2])
        if len(bs) != 1 or bs[0] != w[0]:
            raise Error("bias " + fmt_tuple(bs) + " does not match " + String(w[0]) + " output channels")
    var out = List[Int]()
    out.append(s[0])
    out.append(w[0])
    var res = List[ROut]()
    res.append(_out(_cat(out, _window_out(g, n, s, kernel)), c.dtype(_in(n, 0))))
    return res^


def _r_pool(mut c: RCtx, g: OnnxGraph, n: GNode, op: String) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    if len(s) < 3:
        raise Error(op + " needs a rank 3 or more input, got " + fmt_tuple(s))
    var kernel = List[Int]()
    if not _ail(g, n, "kernel_shape", kernel):
        raise Error("no kernel_shape")
    var out = _cat(_slice(s, 0, 2), _window_out(g, n, s, kernel))
    var res = List[ROut]()
    res.append(_out(out, c.dtype(_in(n, 0))))
    if len(n.outputs) > 1:
        res.append(_out(out, DT_I64))
    return res^


def _r_shape(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var r = len(s)
    var a = _ai(g, n, "start", 0)
    var b = _ai(g, n, "end", r)
    a = min(max(a + r if a < 0 else a, 0), r)
    b = min(max(b + r if b < 0 else b, 0), r)
    var v = _slice(s, a, b) if a < b else List[Int]()
    var dims = List[Int]()
    dims.append(len(v))
    var res = List[ROut]()
    res.append(_outf(dims, DT_I64, MaybeInts(True, v^)))
    return res^


def _r_size(mut c: RCtx, n: GNode) raises -> List[ROut]:
    var v = List[Int]()
    v.append(_numel(c.shape(_in(n, 0))))
    var res = List[ROut]()
    res.append(_outf(List[Int](), DT_I64, MaybeInts(True, v^)))
    return res^


def _r_constant_of_shape(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var shape = c.ints(g, _in(n, 0), "the shape")
    for i in range(len(shape)):  # cpu-route: model file parsing (one per axis)
        if shape[i] < 0:
            raise Error("negative size in " + fmt_tuple(shape))
    var k = _attr_ix(g, n, "value")
    var is_t = k >= 0 and n.attrs[k].kind == AK_TENSOR
    var dt = n.attrs[k].tensors[0].dtype if is_t else DT_F32
    var folded = MaybeInts(False, List[Int]())
    if is_t and is_int_dtype(dt):
        folded.ok = True
        if len(n.attrs[k].tensors[0].ints) > 0:
            var first = n.attrs[k].tensors[0].ints[0]
            for _ in range(_numel(shape)):  # cpu-route: model file parsing (folded shape integers)
                folded.v.append(first)
    var res = List[ROut]()
    res.append(_outf(shape, dt, folded))
    return res^


def _r_range(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var a = c.maybe_ints(g, _in(n, 0))
    var b = c.maybe_ints(g, _in(n, 1))
    var d = c.maybe_ints(g, _in(n, 2))
    if not a.ok or not b.ok or not d.ok:
        raise Error("Range bounds are not integer constants or folded shape values")
    if len(a.v) == 0 or len(b.v) == 0 or len(d.v) == 0:
        raise _fatal("Range bound without a value")
    var a0 = a.v[0]
    var b0 = b.v[0]
    var d0 = d.v[0]
    if d0 == 0:
        raise Error("Range delta 0")
    var count = max(_ceil_div(b0 - a0, d0), 0)
    var vals = List[Int]()
    for i in range(count):  # cpu-route: model file parsing (folded shape integers)
        vals.append(a0 + i * d0)
    var dims = List[Int]()
    dims.append(count)
    var res = List[ROut]()
    res.append(_outf(dims, c.dtype(_in(n, 0)), MaybeInts(True, vals^)))
    return res^


def _r_cast(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var k = _attr_ix(g, n, "to")
    if k < 0:
        raise Error("cast to data type None")
    if n.attrs[k].kind != AK_INT:
        raise _fatal("attribute 'to' is not an int")
    var to = n.attrs[k].i
    var dt = ir_dtype(to)
    if dt < 0:
        raise Error("cast to " + onnx_dtype_name(to))
    var v = c.maybe_ints(g, _in(n, 0))
    var folded = MaybeInts(False, List[Int]())
    if v.ok and is_int_dtype(dt):
        folded.ok = True
        for i in range(len(v.v)):  # cpu-route: model file parsing (folded shape integers)
            if dt == DT_BOOL:
                folded.v.append(1 if v.v[i] != 0 else 0)
            else:
                folded.v.append(v.v[i])
    var res = List[ROut]()
    res.append(_outf(c.shape(_in(n, 0)), dt, folded))
    return res^


def _r_pad(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var s = c.shape(_in(n, 0))
    var pads = List[Int]()
    if _has_in(n, 1):
        pads = c.ints(g, n.inputs[1], "the pads")
    else:
        _ = _ail(g, n, "pads", pads)
    var axes = List[Int]()
    if _has_in(n, 3):
        axes = c.ints(g, n.inputs[3], "the pad axes")
    else:
        for i in range(len(s)):  # cpu-route: model file parsing (one per axis)
            axes.append(i)
    if len(pads) != 2 * len(axes):
        raise Error("pads " + fmt_tuple(pads) + " do not match " + String(len(axes)) + " axes")
    var out = s.copy()
    for j in range(len(axes)):  # cpu-route: model file parsing (one per axis)
        var a = _norm_axis(axes[j], len(s))
        out[a] = s[a] + pads[j] + pads[len(axes) + j]
        if out[a] < 0:
            raise Error("pads " + fmt_tuple(pads) + " leave a negative size on axis " + String(a))
    var res = List[ROut]()
    res.append(_out(out, c.dtype(_in(n, 0))))
    return res^


def _r_constant(mut c: RCtx, g: OnnxGraph, n: GNode) raises -> List[ROut]:
    var k = _attr_ix(g, n, "value")
    if k < 0 or n.attrs[k].kind != AK_TENSOR:
        raise Error("a Constant without a tensor value")
    var t = n.attrs[k].tensors[0].copy()
    var res = List[ROut]()
    res.append(_outf(t.dims, t.dtype, MaybeInts(is_int_dtype(t.dtype), t.ints.copy())))
    return res^


def _is_unary(op: String) -> Bool:
    return (op == "Relu" or op == "Sigmoid" or op == "Tanh" or op == "Exp" or op == "Log" or op == "Sqrt"
            or op == "Neg" or op == "Reciprocal" or op == "Erf" or op == "Gelu" or op == "Softplus"
            or op == "LeakyRelu" or op == "Elu" or op == "HardSigmoid" or op == "HardSwish" or op == "Abs"
            or op == "Floor" or op == "Ceil" or op == "Round" or op == "Sin" or op == "Cos" or op == "Sign"
            or op == "Not" or op == "Identity" or op == "Softmax" or op == "LogSoftmax" or op == "Trilu")


def _is_binary(op: String) -> Bool:
    return (op == "Add" or op == "Sub" or op == "Mul" or op == "Div" or op == "Pow" or op == "Equal"
            or op == "Greater" or op == "Less" or op == "GreaterOrEqual" or op == "LessOrEqual" or op == "And"
            or op == "Or" or op == "Max" or op == "Min")


def _is_reduce(op: String) -> Bool:
    return (op == "ReduceSum" or op == "ReduceMean" or op == "ReduceMax" or op == "ReduceMin" or op == "ReduceL2"
            or op == "ReduceProd")


def _rule(mut c: RCtx, g: OnnxGraph, n: GNode, op: String, mut found: Bool) raises -> List[ROut]:
    """The shape rule of op (importers/_graph.py SHAPE_RULES); found False when it has none."""
    found = True
    if _is_unary(op):
        return _r_unary(c, g, n, op)
    if _is_binary(op):
        return _r_binary(c, g, n, op)
    if _is_reduce(op):
        return _r_reduce(c, g, n)
    if op == "Dropout":
        return _r_dropout(c, g, n)
    if op == "Where":
        return _r_where(c, g, n)
    if op == "Clip" or op == "InstanceNormalization" or op == "GroupNormalization" or op == "CumSum":
        return _r_same(c, n)
    if op == "GatherND":
        return _r_gather_nd(c, g, n)
    if op == "MatMul":
        return _r_matmul(c, n)
    if op == "Gemm":
        return _r_gemm(c, g, n)
    if op == "LayerNormalization" or op == "RMSNormalization":
        return _r_norm_last(c, g, n)
    if op == "BatchNormalization":
        return _r_batchnorm(c, n)
    if op == "Transpose":
        return _r_transpose(c, g, n)
    if op == "Reshape":
        return _r_reshape(c, g, n)
    if op == "Flatten":
        return _r_flatten(c, g, n)
    if op == "Squeeze":
        return _r_squeeze(c, g, n)
    if op == "Unsqueeze":
        return _r_unsqueeze(c, g, n)
    if op == "Expand":
        return _r_expand(c, g, n)
    if op == "Concat":
        return _r_concat(c, g, n)
    if op == "Split":
        return _r_split(c, g, n)
    if op == "Slice":
        return _r_slice(c, g, n)
    if op == "Gather":
        return _r_gather(c, g, n)
    if op == "ArgMax":
        return _r_argmax(c, g, n)
    if op == "GlobalAveragePool" or op == "GlobalMaxPool":
        return _r_global_pool(c, n, op)
    if op == "Conv":
        return _r_conv(c, g, n)
    if op == "MaxPool" or op == "AveragePool":
        return _r_pool(c, g, n, op)
    if op == "Shape":
        return _r_shape(c, g, n)
    if op == "Size":
        return _r_size(c, n)
    if op == "ConstantOfShape":
        return _r_constant_of_shape(c, g, n)
    if op == "Range":
        return _r_range(c, g, n)
    if op == "Cast":
        return _r_cast(c, g, n)
    if op == "Pad":
        return _r_pad(c, g, n)
    if op == "Constant":
        return _r_constant(c, g, n)
    found = False
    return List[ROut]()


# ---------------------------------------------------------------- resolve --

def _fmt_gdims(g: OnnxGraph, v: GValue) -> String:
    """str(tuple(dims)) of a graph value, its symbolic axes quoted."""
    if len(v.dims) == 0:
        return "()"
    var s = String("(")
    for i in range(len(v.dims)):  # cpu-route: model file parsing (one per axis)
        if i > 0:
            s += ", "
        if v.dim_str[i] >= 0:
            s += py_repr(g.st.text(v.dim_str[i]))
        else:
            s += String(v.dims[i])
    if len(v.dims) == 1:
        s += ","
    return s + ")"


@fieldwise_init
struct _RProblem(Copyable, Movable):
    var name: Int
    var op: Int
    var reason: String


def _resolve(g: OnnxGraph, gnames: List[Int], gdims: List[List[Int]], mut c: RCtx,
             mut no_rule: List[Int], mut problems: List[_RProblem]) raises:
    """importers/_graph.py resolve over the handle's graph."""
    var given = Dict[Int, Int]()
    for i in range(len(gnames)):  # cpu-route: model file parsing (the shapes argument)
        given[gnames[i]] = i
    for i in range(len(g.inits)):  # cpu-route: model file parsing (initializer names)
        c.set(g.inits[i].name, g.inits[i].dims, g.inits[i].dtype)
    var known = Dict[Int, Bool]()
    for i in range(len(g.inputs)):  # cpu-route: model file parsing (graph inputs)
        var name = g.inputs[i].name
        known[name] = True
        var gi = given.get(name)
        if gi:
            var want = gdims[gi.value()].copy()
            var bad = len(want) != len(g.inputs[i].dims)
            if not bad:
                for k in range(len(want)):  # cpu-route: model file parsing (one per axis)
                    if g.inputs[i].dim_str[k] < 0 and want[k] != g.inputs[i].dims[k]:
                        bad = True
            if bad:
                problems.append(_RProblem(name, -1, "shape " + fmt_tuple(want) + " does not match the model's "
                                          + _fmt_gdims(g, g.inputs[i])))
                continue
            var empty = False
            for k in range(len(want)):  # cpu-route: model file parsing (one per axis)
                if want[k] <= 0:
                    empty = True
            if empty:
                problems.append(_RProblem(name, -1, "shape " + fmt_tuple(want) + " has an empty or negative axis"))
                continue
            c.set(name, want, g.inputs[i].dtype)
        else:
            var open_axes = List[Int]()
            for k in range(len(g.inputs[i].dims)):  # cpu-route: model file parsing (one per axis)
                if g.inputs[i].dim_str[k] >= 0 or g.inputs[i].dims[k] <= 0:
                    open_axes.append(k)
            if len(open_axes) > 0:
                problems.append(_RProblem(name, -1, "axes " + fmt_list(open_axes) + " of " + _fmt_gdims(g, g.inputs[i])
                                          + " are open; pass shapes={" + py_repr(g.st.text(name)) + ": (...)}"))
                continue
            c.set(name, g.inputs[i].dims, g.inputs[i].dtype)
    for i in range(len(gnames)):  # cpu-route: model file parsing (the shapes argument)
        if gnames[i] not in known:
            var names = String("")
            for k in range(len(g.inputs)):  # cpu-route: model file parsing (graph inputs)
                if k > 0:
                    names += ", "
                names += g.st.text(g.inputs[k].name)
            problems.append(_RProblem(gnames[i], -1, "is not an input of the model (inputs: " + names + ")"))
    for i in range(len(g.nodes)):  # cpu-route: model file parsing (graph nodes)
        var skip = False
        for k in range(len(g.nodes[i].inputs)):  # cpu-route: model file parsing (node inputs)
            var x = g.nodes[i].inputs[k]
            if x != S_EMPTY and not c.has(x):
                skip = True
        if skip:
            continue  # an upstream node is already reported
        var dom = g.st.text(g.nodes[i].domain)
        var found = False
        var outs = List[ROut]()
        if dom.byte_length() == 0 or dom == "ai.onnx":
            try:
                outs = _rule(c, g, g.nodes[i], g.st.text(g.nodes[i].op), found)
            except e:
                var msg = String(e)
                if is_fatal(msg):
                    raise Error(msg)
                problems.append(_RProblem(g.nodes[i].name, g.nodes[i].op, msg))
                continue
        if not found:
            no_rule.append(g.nodes[i].name)
            continue
        var m = len(outs) if len(outs) < len(g.nodes[i].outputs) else len(g.nodes[i].outputs)
        for k in range(m):  # cpu-route: model file parsing (node outputs)
            var name = g.nodes[i].outputs[k]
            if name == S_EMPTY:
                continue
            c.set(name, outs[k].shape, outs[k].dtype)
            if outs[k].has_f:
                c.set_f(name, outs[k].f)
    for i in range(len(g.outputs)):  # cpu-route: model file parsing (graph outputs)
        var name = g.outputs[i].name
        if not c.has(name):
            continue
        var got = c.shape(name)
        if len(got) == len(g.outputs[i].dims):
            var differs = False
            for k in range(len(got)):  # cpu-route: model file parsing (one per axis)
                var d = g.outputs[i].dims[k]
                if g.outputs[i].dim_str[k] < 0 and d > 0 and d != got[k]:
                    differs = True
            if differs:
                problems.append(_RProblem(name, -1, "the model declares shape " + _fmt_gdims(g, g.outputs[i])
                                          + ", the graph computes " + fmt_tuple(got)))


def onnx_resolve_py(h: PythonObject, names: PythonObject, flat: PythonObject) raises -> PythonObject:
    """[words, strings]: the resolved shapes of handle h (layout in this module's docstring)."""
    var gp = onnx_graph(Int(py=h))
    var gnames = List[Int]()
    var gdims = List[List[Int]]()
    var pos = 0
    for i in range(len(names)):  # cpu-route: model file parsing (the shapes argument)
        gnames.append(gp[].st.intern(String(py=names[i])))
        var rank = Int(py=flat[pos])
        pos += 1
        var dims = List[Int]()
        for _ in range(rank):  # cpu-route: model file parsing (one per axis)
            dims.append(Int(py=flat[pos]))
            pos += 1
        gdims.append(dims^)
    var c = RCtx()
    var no_rule = List[Int]()
    var problems = List[_RProblem]()
    var w = Python.list()
    w.append(PythonObject(ONNX_READ_MAGIC))
    w.append(PythonObject(ONNX_READ_VERSION))
    try:
        _resolve(gp[], gnames, gdims, c, no_rule, problems)
        w.append(PythonObject(0))
        w.append(PythonObject(len(c.s_name)))
        for i in range(len(c.s_name)):  # cpu-route: model file parsing (resolved values)
            w.append(PythonObject(c.s_name[i]))
            w.append(PythonObject(c.s_dtype[i]))
            w.append(PythonObject(len(c.s_shape[i])))
            for k in range(len(c.s_shape[i])):  # cpu-route: model file parsing (one per axis)
                w.append(PythonObject(c.s_shape[i][k]))
        w.append(PythonObject(len(c.f_name)))
        for i in range(len(c.f_name)):  # cpu-route: model file parsing (folded values)
            w.append(PythonObject(c.f_name[i]))
            w.append(PythonObject(len(c.f_vals[i])))
            for k in range(len(c.f_vals[i])):  # cpu-route: model file parsing (folded shape integers)
                w.append(PythonObject(c.f_vals[i][k]))
        w.append(PythonObject(len(no_rule)))
        for i in range(len(no_rule)):  # cpu-route: model file parsing (nodes without a rule)
            w.append(PythonObject(no_rule[i]))
        w.append(PythonObject(len(problems)))
        for i in range(len(problems)):  # cpu-route: model file parsing (the problems found)
            w.append(PythonObject(problems[i].name))
            w.append(PythonObject(problems[i].op))
            w.append(PythonObject(gp[].st.intern(problems[i].reason)))
    except e:
        var msg = String(e)
        w = Python.list()
        w.append(PythonObject(ONNX_READ_MAGIC))
        w.append(PythonObject(ONNX_READ_VERSION))
        w.append(PythonObject(ONNX_BAD_VALUE))
        w.append(PythonObject(gp[].st.intern(fatal_text(msg) if is_fatal(msg) else msg)))
    var strs = Python.list()
    for i in range(len(gp[].st.items)):  # cpu-route: model file parsing (the graph's strings)
        strs.append(PythonObject(gp[].st.items[i]))
    var out = Python.list()
    out.append(w)
    out.append(strs)
    return out
