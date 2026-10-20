# SPDX-License-Identifier: Apache-2.0
"""Read a ``torch.export`` program into the import graph.

:func:`read` takes the :class:`torch.export.ExportedProgram` you get from
``torch.export.export(model.eval(), example_inputs)``, lowers it to core ATen
operations and describes it in the same graph the ONNX reader fills, so both
kinds of source import through one table. The weights are not copied: each
parameter, buffer and constant is handed over as a view of the tensor's
memory until it is uploaded to the device.

torch is imported only when :func:`read` is called; importing this module
never imports torch.
"""
# cpu-route: model graph description on the host. This module is the explicit
# input step before the runtime: it walks the exported graph's nodes, names,
# shapes and integer arguments and hands tensor memory over untouched; it
# never computes on tensor element values.
from __future__ import annotations

import operator
import struct

from ._graph import GraphIR, Node, TensorInfo, UnsupportedModel

# The IR speaks ONNX's vocabulary at this opset: reductions, Squeeze,
# Unsqueeze and Split take their axes and sizes as inputs, Gelu and
# GroupNormalization exist.
OPSET = 20
SOURCE = "torch.export"
_INT64_MAX = 9223372036854775807

# ONNX TensorProto.DataType codes by IR dtype (for Cast)
_ONNX_CODE = {"f32": 1, "f16": 10, "bf16": 16, "f64": 11, "i64": 7, "i32": 6, "bool": 9}

# side-effect-only nodes (shape assertions export emits); never live
_SKIP = frozenset((
    "aten._assert_async", "aten._assert_scalar", "aten._functional_assert_async",
    "aten._functional_assert_scalar", "aten.sym_constrain_range", "aten.sym_constrain_range_for_size",
    "aten._functional_sym_constrain_range", "aten._functional_sym_constrain_range_for_size",
    "aten._assert_tensor_metadata",
))

# one input, one ONNX op, no attributes
_UNARY = {
    "aten.relu": "Relu", "aten.sigmoid": "Sigmoid", "aten.tanh": "Tanh", "aten.exp": "Exp", "aten.log": "Log",
    "aten.sqrt": "Sqrt", "aten.neg": "Neg", "aten.reciprocal": "Reciprocal", "aten.erf": "Erf",
    "aten.abs": "Abs", "aten.floor": "Floor", "aten.ceil": "Ceil", "aten.sin": "Sin", "aten.cos": "Cos",
    "aten.sign": "Sign", "aten.hardswish": "HardSwish", "aten.logical_not": "Not",
}
# two operands (tensors or numbers), one ONNX op
_BINARY = {
    "aten.mul": "Mul", "aten.div": "Div", "aten.maximum": "Max", "aten.minimum": "Min", "aten.eq": "Equal",
    "aten.gt": "Greater", "aten.lt": "Less", "aten.ge": "GreaterOrEqual", "aten.le": "LessOrEqual",
    "aten.logical_and": "And", "aten.logical_or": "Or",
}
_MATMUL = frozenset(("aten.mm", "aten.bmm", "aten.matmul"))
_IDENTITY = frozenset(("aten.clone", "aten.alias", "aten.detach", "aten.lift_fresh_copy", "aten.contiguous",
                       "aten.lift_fresh"))
_REDUCE = {"aten.sum": "ReduceSum", "aten.mean": "ReduceMean", "aten.amax": "ReduceMax", "aten.amin": "ReduceMin"}
# integer arithmetic on symbolic sizes (dynamic exports)
_SYM_BIN = {operator.add: "Add", operator.sub: "Sub", operator.mul: "Mul", operator.floordiv: "Div"}


class _Refuse(Exception):
    pass


def _target_name(target):
    """("aten.add", "Tensor") for an operator overload; (its name, "") otherwise."""
    full = str(target)
    if hasattr(target, "_schema") and hasattr(target, "_overloadname"):
        packet, _, overload = full.rpartition(".")
        return packet, overload
    return getattr(target, "__name__", None) or full, ""


def _is_higher_order(target):
    return any(c.__name__ == "HigherOrderOperator" for c in type(target).__mro__)  # cpu-route: model graph description on the host (class names)


class _Reader:
    def __init__(self, torch, ep):
        self.torch = torch
        self.ep = ep
        self.nodes = []
        self.inits = {}
        self.alias = {}   # FX node name -> IR value name
        self.multi = {}   # FX node name -> {output index: IR value name}
        self.problems = []
        self.failed = set()  # FX nodes already reported, so their users are not reported again
        self.count = 0
        self.dtypes = {
            torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16", torch.float64: "f64",
            torch.int64: "i64", torch.int32: "i32", torch.bool: "bool",
        }

    # ------------------------------------------------------------ values --

    def fresh(self, base):
        self.count += 1
        return f"{base}::{self.count}"

    def emit(self, op, inputs, outputs, name, **attrs):
        self.nodes.append(Node(op, "", name, tuple(inputs), tuple(outputs), attrs))
        return outputs[0] if outputs else None

    def val(self, arg):
        """The IR value name of an FX argument (a node) or "" for None."""
        if arg is None:
            return ""
        if not hasattr(arg, "op") or not hasattr(arg, "name"):
            raise _Refuse(f"expected a tensor argument, got {arg!r}")
        if arg.name in self.multi:
            raise _Refuse(f"uses the whole result of '{arg.name}' (only its single outputs are supported)")
        return self.alias.get(arg.name, arg.name)

    def is_node(self, arg):
        return hasattr(arg, "op") and hasattr(arg, "name")

    def meta(self, arg):
        v = arg.meta.get("val") if self.is_node(arg) else None
        if v is None:
            raise _Refuse(f"'{getattr(arg, 'name', arg)}' has no shape information")
        return v

    def shape(self, arg):
        return tuple(self.meta(arg).shape)

    def rank(self, arg):
        return len(self.meta(arg).shape)

    def is_float(self, arg):
        if not self.is_node(arg):
            return isinstance(arg, float)
        v = arg.meta.get("val")
        return getattr(v, "dtype", None) is not None and v.dtype.is_floating_point

    def const_f(self, v):
        """A float64 scalar constant (rounded to float32 on the device at upload)."""
        name = self.fresh("const")
        self.inits[name] = TensorInfo(name, "f64", (), memoryview(struct.pack("<d", float(v))))
        return name

    def const_i(self, vals, shape=None):
        """An int64 constant: a description integer list (shape, axes, sizes, indices)."""
        vals = [int(v) for v in vals]  # cpu-route: model graph description on the host (integer arguments)
        name = self.fresh("ints")
        shp = (len(vals),) if shape is None else tuple(shape)
        self.inits[name] = TensorInfo(name, "i64", shp, memoryview(struct.pack(f"<{len(vals)}q", *vals)))
        return name

    def number(self, arg, like):
        """A number operand as a constant of the other operand's kind, or a node's value."""
        if self.is_node(arg):
            return self.val(arg)
        if isinstance(arg, bool):
            arg = int(arg)
        if not isinstance(arg, (int, float)):
            raise _Refuse(f"unsupported operand {arg!r}")
        if isinstance(arg, float) or self.is_float(like):
            return self.const_f(arg)
        return self.const_i([arg], ())

    def int_scalar(self, arg):
        """An int argument or a symbolic size node, as a scalar int64 value."""
        if self.is_node(arg):
            return self.val(arg)
        return self.const_i([int(arg)], ())

    def int_list(self, items, name):
        """A list of ints and symbolic sizes as a 1-D int64 value: a constant
        when every entry is an int, else a Concat the shape folding resolves."""
        items = list(items)
        if not any(self.is_node(v) for v in items):  # cpu-route: model graph description on the host (integer arguments)
            return self.const_i(items)
        parts = []
        for v in items:  # cpu-route: model graph description on the host (integer arguments)
            if self.is_node(v):
                parts.append(self.emit("Unsqueeze", [self.val(v), self.const_i([0])], [self.fresh(name)],
                                       self.fresh(name)))
            else:
                parts.append(self.const_i([v]))
        return self.emit("Concat", parts, [self.fresh(name)], self.fresh(name), axis=0)

    # ----------------------------------------------------------- tensors --

    def tensor_info(self, name, t):
        """A torch tensor as an initializer: its memory as little-endian bytes
        (bfloat16 and float16 as their bit patterns), never converted."""
        dt = self.dtypes.get(t.dtype)
        if dt is None:
            raise _Refuse(f"tensor type {t.dtype} (supported: float32, float16, bfloat16, float64, int64, "
                          f"int32 and bool)")
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()  # cpu-route: model graph description on the host (weights handed to the upload)
        if not t.is_contiguous():
            t = t.contiguous()  # cpu-route: model graph description on the host (row-major weight memory)
        raw = t.reshape(-1).view(self.torch.uint8).numpy()
        return TensorInfo(name, dt, tuple(int(d) for d in t.shape), memoryview(raw))  # cpu-route: model graph description on the host (one per axis)


def _bind(node):
    """The node's arguments by schema name, defaults filled in."""
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return {i: a for i, a in enumerate(node.args)}  # cpu-route: model graph description on the host (node arguments)
    out = {}
    for i, a in enumerate(schema.arguments):  # cpu-route: model graph description on the host (node arguments)
        if i < len(node.args):
            out[a.name] = node.args[i]
        elif a.name in node.kwargs:
            out[a.name] = node.kwargs[a.name]
        elif a.has_default_value():
            out[a.name] = a.default_value
        else:
            out[a.name] = None
    return out


def _first(a, *names):
    for n in names:  # cpu-route: model graph description on the host (argument names)
        if n in a:
            return a[n]
    return None


def _pair(v, n, what):
    """A pool or conv size argument as n ints (one int, or a list of one, means all axes)."""
    if isinstance(v, int):
        return (v,) * n
    v = tuple(int(x) for x in v)  # cpu-route: model graph description on the host (one per axis)
    if len(v) == 1:
        return v * n
    if len(v) != n:
        raise _Refuse(f"{what} {v} does not match {n} spatial axes")
    return v


# --------------------------------------------------------------- handlers --
# Each handler takes (reader, node, bound args, overload) and returns the IR
# value of the node's single output, or a dict {output index: value} for a
# node whose result is a tuple. Unsupported arguments raise _Refuse.

def _h_unary(op):
    def h(r, n, a, ov):
        return r.emit(op, [r.val(_first(a, "self", "input"))], [n.name], n.name)
    return h


def _h_binary(op):
    def h(r, n, a, ov):
        x, y = a["self"], a["other"]
        return r.emit(op, [r.number(x, y), r.number(y, x)], [n.name], n.name)
    return h


def _h_ne(r, n, a, ov):
    x, y = a["self"], a["other"]
    eq = r.emit("Equal", [r.number(x, y), r.number(y, x)], [r.fresh(n.name)], r.fresh(n.name))
    return r.emit("Not", [eq], [n.name], n.name)


def _h_add_sub(op):
    def h(r, n, a, ov):
        x, y, alpha = a["self"], a["other"], a.get("alpha", 1)
        yv = r.number(y, x)
        if alpha is not None and alpha != 1:
            yv = r.emit("Mul", [yv, r.number(alpha, x)], [r.fresh(n.name)], r.fresh(n.name))
        return r.emit(op, [r.number(x, y), yv], [n.name], n.name)
    return h


def _h_rsub(r, n, a, ov):
    x, y, alpha = a["self"], a["other"], a.get("alpha", 1)
    xv = r.val(x)
    if alpha is not None and alpha != 1:
        xv = r.emit("Mul", [xv, r.number(alpha, x)], [r.fresh(n.name)], r.fresh(n.name))
    return r.emit("Sub", [r.number(y, x), xv], [n.name], n.name)


def _h_div(r, n, a, ov):
    if a.get("rounding_mode") is not None:
        raise _Refuse(f"rounding_mode={a['rounding_mode']!r} (only true division is supported)")
    x, y = a["self"], a["other"]
    return r.emit("Div", [r.number(x, y), r.number(y, x)], [n.name], n.name)


def _h_pow(r, n, a, ov):
    x, e = a["self"], a["exponent"]
    return r.emit("Pow", [r.number(x, e), r.number(e, x)], [n.name], n.name)


def _h_rsqrt(r, n, a, ov):
    s = r.emit("Sqrt", [r.val(a["self"])], [r.fresh(n.name)], r.fresh(n.name))
    return r.emit("Reciprocal", [s], [n.name], n.name)


def _h_silu(r, n, a, ov):
    x = r.val(a["self"])
    s = r.emit("Sigmoid", [x], [r.fresh(n.name)], r.fresh(n.name))
    return r.emit("Mul", [x, s], [n.name], n.name)


def _h_round(r, n, a, ov):
    if a.get("decimals") not in (None, 0):
        raise _Refuse(f"decimals={a['decimals']} (only rounding to an integer is supported)")
    return r.emit("Round", [r.val(a["self"])], [n.name], n.name)


def _h_gelu(r, n, a, ov):
    return r.emit("Gelu", [r.val(a["self"])], [n.name], n.name, approximate=str(a.get("approximate") or "none"))


def _h_leaky_relu(r, n, a, ov):
    return r.emit("LeakyRelu", [r.val(a["self"])], [n.name], n.name, alpha=float(a.get("negative_slope", 0.01)))


def _h_elu(r, n, a, ov):
    if a.get("scale", 1) != 1 or a.get("input_scale", 1) != 1:
        raise _Refuse("elu with scale or input_scale other than 1 (selu, celu) is not supported")
    return r.emit("Elu", [r.val(a["self"])], [n.name], n.name, alpha=float(a.get("alpha", 1.0)))


def _h_hardsigmoid(r, n, a, ov):
    # torch's hardsigmoid is relu6(x + 3) / 6: ONNX HardSigmoid with alpha 1/6, beta 1/2
    return r.emit("HardSigmoid", [r.val(a["self"])], [n.name], n.name, alpha=0.16666666666666666, beta=0.5)


def _h_softplus(r, n, a, ov):
    if a.get("beta", 1) != 1:
        raise _Refuse(f"beta={a['beta']} (only beta=1 is supported)")
    return r.emit("Softplus", [r.val(a["self"])], [n.name], n.name)


def _h_clamp(r, n, a, ov, lo_name="min", hi_name="max"):
    x = a["self"]
    lo, hi = a.get(lo_name), a.get(hi_name)
    if r.is_node(lo) or r.is_node(hi):
        raise _Refuse("clamp bounds given as tensors (only number bounds are supported)")
    ins = [r.val(x), "" if lo is None else r.const_f(lo), "" if hi is None else r.const_f(hi)]
    return r.emit("Clip", ins, [n.name], n.name)


def _h_clamp_min(r, n, a, ov):
    return r.emit("Clip", [r.val(a["self"]), r.number(a["min"], a["self"]), ""], [n.name], n.name)


def _h_clamp_max(r, n, a, ov):
    return r.emit("Clip", [r.val(a["self"]), "", r.number(a["max"], a["self"])], [n.name], n.name)


def _h_hardtanh(r, n, a, ov):
    return _h_clamp(r, n, a, ov, "min_val", "max_val")


def _h_where(r, n, a, ov):
    c, x, y = a["condition"], a["self"], a["other"]
    return r.emit("Where", [r.val(c), r.number(x, y), r.number(y, x)], [n.name], n.name)


def _h_masked_fill(r, n, a, ov):
    x, m, v = a["self"], a["mask"], a["value"]
    return r.emit("Where", [r.val(m), r.number(v, x), r.val(x)], [n.name], n.name)


def _h_matmul(r, n, a, ov):
    return r.emit("MatMul", [r.val(a["self"]), r.val(_first(a, "mat2", "other"))], [n.name], n.name)


def _h_addmm(r, n, a, ov):
    return r.emit("Gemm", [r.val(a["mat1"]), r.val(a["mat2"]), r.val(a["self"])], [n.name], n.name,
                  alpha=float(a.get("alpha", 1)), beta=float(a.get("beta", 1)))


def _h_linear(r, n, a, ov):
    x, w, b = a["input"], a["weight"], a.get("bias")
    if r.rank(x) == 2:
        ins = [r.val(x), r.val(w)] + ([r.val(b)] if b is not None else [])
        return r.emit("Gemm", ins, [n.name], n.name, transB=1)
    wt = r.emit("Transpose", [r.val(w)], [r.fresh(n.name)], r.fresh(n.name), perm=(1, 0))
    if b is None:
        return r.emit("MatMul", [r.val(x), wt], [n.name], n.name)
    y = r.emit("MatMul", [r.val(x), wt], [r.fresh(n.name)], r.fresh(n.name))
    return r.emit("Add", [y, r.val(b)], [n.name], n.name)


def _conv(r, n, x, w, b, stride, padding, dilation, groups):
    k = tuple(int(d) for d in r.shape(w)[2:])  # cpu-route: model graph description on the host (one per axis)
    sp = len(k)
    pads = _pair(padding, sp, "padding")
    return r.emit("Conv", [r.val(x), r.val(w), r.val(b)] if b is not None else [r.val(x), r.val(w)], [n.name],
                  n.name, kernel_shape=k, strides=_pair(stride, sp, "stride"), pads=pads + pads,
                  dilations=_pair(dilation, sp, "dilation"), group=int(groups))


def _h_convolution(r, n, a, ov):
    if a.get("transposed"):
        raise _Refuse("transposed convolution (ConvTranspose) is not supported")
    return _conv(r, n, a["input"], a["weight"], a.get("bias"), a["stride"], a["padding"], a["dilation"],
                 a.get("groups", 1))


def _h_conv_nd(r, n, a, ov):
    if isinstance(a.get("padding"), str):
        raise _Refuse(f"padding={a['padding']!r} (only explicit padding is supported)")
    return _conv(r, n, a["input"], a["weight"], a.get("bias"), a.get("stride", 1), a.get("padding", 0),
                 a.get("dilation", 1), a.get("groups", 1))


def _h_native_layer_norm(r, n, a, ov):
    x, w, b = a["input"], a.get("weight"), a.get("bias")
    if w is None:
        raise _Refuse("layer norm without a weight (elementwise_affine=False) is not supported")
    k = len(a["normalized_shape"])
    y = r.emit("LayerNormalization", [r.val(x), r.val(w), r.val(b)], [f"{n.name}.0"], n.name,
               axis=-k, epsilon=float(a.get("eps", 1e-5)))
    return {0: y}


def _h_layer_norm(r, n, a, ov):
    return _h_native_layer_norm(r, n, a, ov)[0]


def _h_batch_norm_eval(r, n, a, ov):
    x, w, b = a["input"], a.get("weight"), a.get("bias")
    if w is None or b is None:
        raise _Refuse("batch norm without a weight and bias (affine=False) is not supported")
    if a.get("running_mean") is None or a.get("running_var") is None:
        raise _Refuse("batch norm without running statistics is not supported")
    y = r.emit("BatchNormalization", [r.val(x), r.val(w), r.val(b), r.val(a["running_mean"]),
                                      r.val(a["running_var"])], [f"{n.name}.0"], n.name,
               epsilon=float(a.get("eps", 1e-5)))
    return {0: y}


def _h_batch_norm_train(r, n, a, ov):
    raise _Refuse("batch norm in training mode; call model.eval() before torch.export.export")


def _h_group_norm(r, n, a, ov):
    x, w, b = a["input"], a.get("weight"), a.get("bias")
    if w is None or b is None:
        raise _Refuse("group norm without a weight and bias (affine=False) is not supported")
    y = r.emit("GroupNormalization", [r.val(x), r.val(w), r.val(b)], [f"{n.name}.0"], n.name,
               num_groups=int(a["group"]), epsilon=float(a.get("eps", 1e-5)))
    return {0: y}


def _h_softmax(op):
    def h(r, n, a, ov):
        return r.emit(op, [r.val(a["self"])], [n.name], n.name, axis=int(a["dim"]))
    return h


def _h_reshape(r, n, a, ov):
    size = list(_first(a, "size", "shape"))
    if any(not r.is_node(d) and int(d) == 0 for d in size):  # cpu-route: model graph description on the host (one per axis)
        raise _Refuse("a reshape to an empty axis is not supported")
    return r.emit("Reshape", [r.val(a["self"]), r.int_list(size, n.name)], [n.name], n.name)


def _h_permute(r, n, a, ov):
    rank = r.rank(a["self"])
    perm = tuple(int(d) % rank if rank else 0 for d in a["dims"])  # cpu-route: model graph description on the host (one per axis)
    return r.emit("Transpose", [r.val(a["self"])], [n.name], n.name, perm=perm)


def _h_transpose(r, n, a, ov):
    rank = r.rank(a["self"])
    if rank < 2:
        r.alias[n.name] = r.val(a["self"])
        return None
    d0, d1 = int(a["dim0"]) % rank, int(a["dim1"]) % rank
    perm = list(range(rank))
    perm[d0], perm[d1] = perm[d1], perm[d0]
    return r.emit("Transpose", [r.val(a["self"])], [n.name], n.name, perm=tuple(perm))


def _h_t(r, n, a, ov):
    if r.rank(a["self"]) < 2:
        r.alias[n.name] = r.val(a["self"])
        return None
    return r.emit("Transpose", [r.val(a["self"])], [n.name], n.name, perm=(1, 0))


def _h_squeeze(r, n, a, ov):
    x = a["self"]
    shp = r.shape(x)
    if ov == "default" or _first(a, "dim") is None:
        dims = [i for i, d in enumerate(shp) if isinstance(d, int) and d == 1]  # cpu-route: model graph description on the host (one per axis)
    else:
        dim = a["dim"]
        dims = [dim] if isinstance(dim, int) else list(dim)
        # torch leaves an axis that is not of size 1 in place; ONNX refuses it
        dims = [int(d) % len(shp) for d in dims if isinstance(shp[int(d) % len(shp)], int) and shp[int(d) % len(shp)] == 1]  # cpu-route: model graph description on the host (one per axis)
    if not dims:
        r.alias[n.name] = r.val(x)
        return None
    return r.emit("Squeeze", [r.val(x), r.const_i(dims)], [n.name], n.name)


def _h_unsqueeze(r, n, a, ov):
    return r.emit("Unsqueeze", [r.val(a["self"]), r.const_i([int(a["dim"])])], [n.name], n.name)


def _h_expand(r, n, a, ov):
    # ONNX Expand broadcasts the input against the shape, so torch's -1
    # ("keep this axis") is a 1
    size = [d if r.is_node(d) or int(d) != -1 else 1 for d in a["size"]]  # cpu-route: model graph description on the host (one per axis)
    return r.emit("Expand", [r.val(a["self"]), r.int_list(size, n.name)], [n.name], n.name)


def _h_cat(r, n, a, ov):
    xs = [r.val(t) for t in a["tensors"]]  # cpu-route: model graph description on the host (the operands)
    return r.emit("Concat", xs, [n.name], n.name, axis=int(a.get("dim", 0)))


def _h_split(r, n, a, ov):
    x, dim = a["self"], int(a.get("dim", 0))
    shp = r.shape(x)
    sizes = a.get("split_sizes")
    if sizes is None and isinstance(a.get("split_size"), (list, tuple)):
        sizes = a["split_size"]
    if sizes is None:
        size = a["split_size"]
        total = shp[dim % len(shp)]
        if not isinstance(total, int) or r.is_node(size):
            raise _Refuse("a split of a dynamic axis is not supported")
        size = int(size)
        # the piece sizes of an equal split: description integers of the axis length
        sizes = [size] * (total // size) + ([total % size] if total % size else [])
    sizes = list(sizes)
    if any(r.is_node(s) for s in sizes):  # cpu-route: model graph description on the host (the pieces)
        raise _Refuse("split sizes computed from the input's shape are not supported")
    outs = [f"{n.name}.{i}" for i in range(len(sizes))]  # cpu-route: model graph description on the host (the pieces)
    r.emit("Split", [r.val(x), r.const_i(sizes)], outs, n.name, axis=dim)
    return dict(enumerate(outs))


def _h_slice(r, n, a, ov):
    step = a.get("step", 1)
    if r.is_node(step) or int(step) < 1:
        raise _Refuse(f"step {step} (only constant positive steps are supported)")
    start = a.get("start")
    end = a.get("end")
    starts = r.int_list([0 if start is None else start], n.name)
    ends = r.int_list([_INT64_MAX if end is None else end], n.name)
    return r.emit("Slice", [r.val(a["self"]), starts, ends, r.const_i([int(a.get("dim", 0))]),
                            r.const_i([int(step)])], [n.name], n.name)


def _h_select(r, n, a, ov):
    return r.emit("Gather", [r.val(a["self"]), r.int_scalar(a["index"])], [n.name], n.name, axis=int(a["dim"]))


def _h_embedding(r, n, a, ov):
    return r.emit("Gather", [r.val(a["weight"]), r.val(a["indices"])], [n.name], n.name, axis=0)


def _h_index_select(r, n, a, ov):
    return r.emit("Gather", [r.val(a["self"]), r.val(a["index"])], [n.name], n.name, axis=int(a["dim"]))


def _h_index(r, n, a, ov):
    idx = list(a["indices"])
    used = [i for i, t in enumerate(idx) if t is not None]  # cpu-route: model graph description on the host (index positions)
    if not used:
        raise _Refuse("indexing without an index tensor is not supported")
    for i in used:  # cpu-route: model graph description on the host (index positions)
        if r.meta(idx[i]).dtype == r.torch.bool:
            raise _Refuse("indexing with a boolean mask (a data-dependent shape) is not supported")
    if len(used) == 1:
        return r.emit("Gather", [r.val(a["self"]), r.val(idx[used[0]])], [n.name], n.name, axis=used[0])
    return _index_nd(r, n, a["self"], [idx[i] for i in used], used)  # cpu-route: model graph description on the host (index positions)


def _index_nd(r, n, x, tensors, axes):
    """x[i0, i1, ...] with several index tensors on axes ``axes`` as ONNX
    GatherND: the indexed axes moved to the front (Transpose), the index
    tensors broadcast and stacked on a new last axis, one GatherND, and the
    broadcast axes moved to where the first indexed axis stood when the
    indexed axes are consecutive (torch's rule)."""
    rank = r.rank(x)
    shapes = [r.shape(t) for t in tensors]  # cpu-route: model graph description on the host (the index tensors)
    if any(not isinstance(d, int) for sh in shapes for d in sh) or any(  # cpu-route: model graph description on the host (one per axis)
            not isinstance(d, int) for d in r.shape(x)):  # cpu-route: model graph description on the host (one per axis)
        raise _Refuse("indexing with several index tensors of dynamic shapes is not supported")
    bshape = tuple(int(d) for d in r.torch.broadcast_shapes(*shapes))  # cpu-route: model graph description on the host (one per axis)
    others = [d for d in range(rank) if d not in axes]  # cpu-route: model graph description on the host (one per axis)
    xv = r.val(x)
    if list(axes) != list(range(len(axes))):  # cpu-route: model graph description on the host (one per axis)
        xv = r.emit("Transpose", [xv], [r.fresh(n.name)], r.fresh(n.name), perm=tuple(list(axes) + others))
    parts = []
    for t, sh in zip(tensors, shapes):  # cpu-route: model graph description on the host (the index tensors)
        v = r.val(t)
        if tuple(sh) != bshape:
            v = r.emit("Expand", [v, r.const_i(bshape)], [r.fresh(n.name)], r.fresh(n.name))
        parts.append(r.emit("Unsqueeze", [v, r.const_i([-1])], [r.fresh(n.name)], r.fresh(n.name)))
    packed = r.emit("Concat", parts, [r.fresh(n.name)], r.fresh(n.name), axis=-1)
    nb = len(bshape)
    consecutive = list(axes) == list(range(axes[0], axes[0] + len(axes)))  # cpu-route: model graph description on the host (one per axis)
    if not (consecutive and axes[0] > 0 and nb > 0):
        return r.emit("GatherND", [xv, packed], [n.name], n.name, batch_dims=0)
    y = r.emit("GatherND", [xv, packed], [r.fresh(n.name)], r.fresh(n.name), batch_dims=0)
    lead = axes[0]
    perm = [nb + d for d in range(lead)] + list(range(nb)) + [nb + d for d in range(lead, len(others))]  # cpu-route: model graph description on the host (one per axis)
    return r.emit("Transpose", [y], [n.name], n.name, perm=tuple(perm))


def _is_bool(r, arg):
    if isinstance(arg, bool):
        return True
    return r.is_node(arg) and r.meta(arg).dtype == r.torch.bool


def _h_bitwise(op):
    def h(r, n, a, ov):
        x, y = a["self"], a["other"]
        if not (_is_bool(r, x) and _is_bool(r, y)):
            raise _Refuse(f"bitwise operations on integers (only on booleans, as logical {op}) are not supported")
        return r.emit(op, [r.number(x, y), r.number(y, x)], [n.name], n.name)
    return h


def _h_bitwise_not(r, n, a, ov):
    x = a["self"]
    if not _is_bool(r, x):
        raise _Refuse("bitwise not on integers (only on booleans, as logical not) is not supported")
    return r.emit("Not", [r.val(x)], [n.name], n.name)


def _h_any_all(op):
    # any / all along axes: ReduceMax / ReduceMin over a boolean (0 or 1)
    # tensor; a non-boolean input is first tested against zero
    def h(r, n, a, ov):
        x = a["self"]
        v = r.val(x)
        if not _is_bool(r, x):
            eq = r.emit("Equal", [v, r.number(0, x)], [r.fresh(n.name)], r.fresh(n.name))
            v = r.emit("Not", [eq], [r.fresh(n.name)], r.fresh(n.name))
        dims = _first(a, "dim")
        if dims is not None and not isinstance(dims, int) and len(dims) == 0:
            dims = None
        ins = [v]
        if dims is not None:
            ins.append(r.const_i([dims] if isinstance(dims, int) else list(dims)))
        return r.emit(op, ins, [n.name], n.name, keepdims=int(bool(a.get("keepdim", False))))
    return h


def _h_cumsum(r, n, a, ov):
    return r.emit("CumSum", [r.val(a["self"]), r.const_i([int(a["dim"])], ())], [n.name], n.name,
                  exclusive=0, reverse=0)


def _h_reduce(op):
    def h(r, n, a, ov):
        dims = _first(a, "dim")
        ins = [r.val(a["self"])]
        if dims is not None and not isinstance(dims, int) and len(dims) == 0:
            dims = None
        if dims is not None:
            ins.append(r.const_i([dims] if isinstance(dims, int) else list(dims)))
        return r.emit(op, ins, [n.name], n.name, keepdims=int(bool(a.get("keepdim", False))))
    return h


def _h_max_min_dim(op):
    def h(r, n, a, ov):
        x = a["self"]
        if ov == "other":
            return r.emit("Max" if op == "ReduceMax" else "Min", [r.val(x), r.val(a["other"])], [n.name], n.name)
        if ov == "default":
            return r.emit(op, [r.val(x)], [n.name], n.name, keepdims=0)
        if ov != "dim":
            raise _Refuse(f"overload {ov!r} is not supported")
        dim, keep = int(a["dim"]), int(bool(a.get("keepdim", False)))
        v = r.emit(op, [r.val(x), r.const_i([dim])], [f"{n.name}.0"], n.name, keepdims=keep)
        if op != "ReduceMax":
            return {0: v}
        i = r.emit("ArgMax", [r.val(x)], [f"{n.name}.1"], f"{n.name}.indices", axis=dim, keepdims=keep)
        return {0: v, 1: i}
    return h


def _h_argmax(r, n, a, ov):
    x, dim, keep = a["self"], a.get("dim"), int(bool(a.get("keepdim", False)))
    if dim is None:
        if keep:
            raise _Refuse("argmax over all axes with keepdim=True is not supported")
        flat = r.emit("Reshape", [r.val(x), r.const_i([-1])], [r.fresh(n.name)], r.fresh(n.name))
        return r.emit("ArgMax", [flat], [n.name], n.name, axis=0, keepdims=0)
    return r.emit("ArgMax", [r.val(x)], [n.name], n.name, axis=int(dim), keepdims=keep)


def _pool_attrs(r, a, x):
    sp = r.rank(x) - 2
    k = _pair(a["kernel_size"], sp, "kernel_size")
    stride = a.get("stride")
    s = k if not stride else _pair(stride, sp, "stride")
    p = _pair(a.get("padding", 0), sp, "padding")
    return {"kernel_shape": k, "strides": s, "pads": p + p, "ceil_mode": int(bool(a.get("ceil_mode", False)))}


def _h_max_pool(r, n, a, ov):
    x = a["self"]
    attrs = _pool_attrs(r, a, x)
    attrs["dilations"] = _pair(a.get("dilation", 1), len(attrs["kernel_shape"]), "dilation")
    y = r.emit("MaxPool", [r.val(x)], [f"{n.name}.0"], n.name, **attrs)
    return {0: y}


def _h_max_pool_single(r, n, a, ov):
    return _h_max_pool(r, n, a, ov)[0]


def _h_avg_pool(r, n, a, ov):
    if a.get("divisor_override") is not None:
        raise _Refuse("divisor_override is not supported")
    x = a["self"]
    attrs = _pool_attrs(r, a, x)
    return r.emit("AveragePool", [r.val(x)], [n.name], n.name,
                  count_include_pad=int(bool(a.get("count_include_pad", True))), **attrs)


def _h_adaptive_avg(r, n, a, ov):
    out = tuple(int(d) for d in a["output_size"])  # cpu-route: model graph description on the host (one per axis)
    if any(d != 1 for d in out):  # cpu-route: model graph description on the host (one per axis)
        raise _Refuse(f"output_size {out} (only a global pool, output size 1, is supported)")
    return r.emit("GlobalAveragePool", [r.val(a["self"])], [n.name], n.name)


def _h_identity(r, n, a, ov):
    r.alias[n.name] = r.val(_first(a, "self", "input"))
    return None


def _h_dropout(r, n, a, ov):
    if a.get("train"):
        raise _Refuse("dropout in training mode; call model.eval() before torch.export.export")
    x = r.val(a["input"])
    if str(n.target).startswith("aten.native_dropout"):
        return {0: x}
    r.alias[n.name] = x
    return None


def _h_to_copy(r, n, a, ov):
    x = a["self"]
    dt = a.get("dtype")
    if dt is None or getattr(dt, "is_floating_point", False):
        if r.is_float(x) or dt is None:
            r.alias[n.name] = r.val(x)  # storage is float32: a float-to-float cast is the same tensor
            return None
    code = _ONNX_CODE.get(r.dtypes.get(dt))
    if code is None:
        raise _Refuse(f"cast to {dt}")
    return r.emit("Cast", [r.val(x)], [n.name], n.name, to=code)


def _h_full(r, n, a, ov):
    size = list(a["size"])
    return _fill(r, n, r.int_list(size, n.name), a["fill_value"], r.meta(n).dtype)


def _h_full_like(r, n, a, ov, value=None):
    x = a["self"]
    shp = r.emit("Shape", [r.val(x)], [r.fresh(n.name)], r.fresh(n.name))
    return _fill(r, n, shp, a["fill_value"] if value is None else value, r.meta(n).dtype)


def _fill(r, n, shape_value, value, dtype):
    torch = r.torch
    if r.is_node(value):
        raise _Refuse("a fill value computed from tensors is not supported")
    if dtype.is_floating_point:
        return r.emit("Expand", [r.const_f(value), shape_value], [n.name], n.name)
    dt = r.dtypes.get(dtype)
    if dt not in ("i64", "i32", "bool"):
        raise _Refuse(f"fill of type {dtype}")
    v = TensorInfo(r.fresh("value"), "bool" if dtype == torch.bool else "i64", (1,),
                   memoryview(struct.pack("<B", int(bool(value)))) if dtype == torch.bool
                   else memoryview(struct.pack("<q", int(value))))
    return r.emit("ConstantOfShape", [shape_value], [n.name], n.name, value=v)


def _h_scalar_tensor(r, n, a, ov):
    s, dt = a["s"], r.meta(n).dtype
    if dt.is_floating_point:
        r.alias[n.name] = r.const_f(s)
    else:
        r.alias[n.name] = r.const_i([int(s)], ())
    return None


def _h_trilu(upper):
    def h(r, n, a, ov):
        return r.emit("Trilu", [r.val(a["self"]), r.const_i([int(a.get("diagonal", 0))], ())], [n.name], n.name,
                      upper=upper)
    return h


def _h_pad(r, n, a, ov):
    x = a["self"]
    rank = r.rank(x)
    pad = [int(p) for p in a["pad"]]  # cpu-route: model graph description on the host (one per axis)
    begin, end = [0] * rank, [0] * rank
    for i in range(len(pad) // 2):  # cpu-route: model graph description on the host (one per axis)
        axis = rank - 1 - i
        begin[axis], end[axis] = pad[2 * i], pad[2 * i + 1]
    return r.emit("Pad", [r.val(x), r.const_i(begin + end), r.const_f(a.get("value", 0) or 0)], [n.name], n.name,
                  mode="constant")


def _h_arange(r, n, a, ov):
    dt = a.get("dtype")
    start, end, step = a.get("start", 0), a.get("end"), a.get("step", 1)
    if ov == "default":
        start, end = 0, a["end"]
    vals = (start, end, step)
    if (dt is not None and dt.is_floating_point) or any(isinstance(v, float) for v in vals):  # cpu-route: model graph description on the host (the three bounds)
        raise _Refuse("a floating-point arange is not supported (register the positions as a buffer)")
    return r.emit("Range", [r.int_scalar(start), r.int_scalar(end), r.int_scalar(step)], [n.name], n.name)


def _h_sym_size(r, n, a, ov):
    shp = r.emit("Shape", [r.val(a["self"])], [r.fresh(n.name)], r.fresh(n.name))
    return r.emit("Gather", [shp, r.const_i([int(a["dim"])], ())], [n.name], n.name, axis=0)


def _h_sym_numel(r, n, a, ov):
    return r.emit("Size", [r.val(a["self"])], [n.name], n.name)


_HANDLERS = {packet: _h_unary(op) for packet, op in _UNARY.items()}  # cpu-route: model graph description on the host (op names)
_HANDLERS.update({packet: _h_binary(op) for packet, op in _BINARY.items()})  # cpu-route: model graph description on the host (op names)
_HANDLERS.update({packet: _h_matmul for packet in _MATMUL})  # cpu-route: model graph description on the host (op names)
_HANDLERS.update({packet: _h_identity for packet in _IDENTITY})  # cpu-route: model graph description on the host (op names)
_HANDLERS.update({packet: _h_reduce(op) for packet, op in _REDUCE.items()})  # cpu-route: model graph description on the host (op names)
_HANDLERS.update({
    "aten.add": _h_add_sub("Add"), "aten.sub": _h_add_sub("Sub"), "aten.rsub": _h_rsub, "aten.div": _h_div,
    "aten.true_divide": _h_div, "aten.ne": _h_ne, "aten.pow": _h_pow, "aten.rsqrt": _h_rsqrt,
    "aten.silu": _h_silu, "aten.round": _h_round, "aten.gelu": _h_gelu, "aten.leaky_relu": _h_leaky_relu,
    "aten.elu": _h_elu, "aten.hardsigmoid": _h_hardsigmoid, "aten.softplus": _h_softplus, "aten.clamp": _h_clamp,
    "aten.clamp_min": _h_clamp_min, "aten.clamp_max": _h_clamp_max, "aten.hardtanh": _h_hardtanh,
    "aten.where": _h_where, "aten.masked_fill": _h_masked_fill,
    "aten.addmm": _h_addmm, "aten.linear": _h_linear,
    "aten.convolution": _h_convolution, "aten.conv1d": _h_conv_nd, "aten.conv2d": _h_conv_nd,
    "aten.native_layer_norm": _h_native_layer_norm, "aten.layer_norm": _h_layer_norm,
    "aten._native_batch_norm_legit_no_training": _h_batch_norm_eval,
    "aten._native_batch_norm_legit": _h_batch_norm_train, "aten.native_batch_norm": _h_batch_norm_train,
    "aten._native_batch_norm_legit_functional": _h_batch_norm_train,
    "aten.native_group_norm": _h_group_norm,
    "aten._softmax": _h_softmax("Softmax"), "aten.softmax": _h_softmax("Softmax"),
    "aten._log_softmax": _h_softmax("LogSoftmax"), "aten.log_softmax": _h_softmax("LogSoftmax"),
    "aten.view": _h_reshape, "aten.reshape": _h_reshape, "aten._unsafe_view": _h_reshape,
    "aten.permute": _h_permute, "aten.transpose": _h_transpose, "aten.t": _h_t,
    "aten.squeeze": _h_squeeze, "aten.unsqueeze": _h_unsqueeze, "aten.expand": _h_expand, "aten.cat": _h_cat,
    "aten.split_with_sizes": _h_split, "aten.split": _h_split, "aten.slice": _h_slice, "aten.select": _h_select,
    "aten.embedding": _h_embedding, "aten.index_select": _h_index_select, "aten.index": _h_index,
    "aten.max": _h_max_min_dim("ReduceMax"), "aten.min": _h_max_min_dim("ReduceMin"), "aten.argmax": _h_argmax,
    "aten.max_pool2d_with_indices": _h_max_pool, "aten.max_pool2d": _h_max_pool_single,
    "aten.avg_pool2d": _h_avg_pool, "aten._adaptive_avg_pool2d": _h_adaptive_avg,
    "aten.adaptive_avg_pool2d": _h_adaptive_avg,
    "aten.dropout": _h_dropout, "aten.native_dropout": _h_dropout,
    "aten._to_copy": _h_to_copy, "aten.to": _h_to_copy,
    "aten.full": _h_full, "aten.full_like": _h_full_like,
    "aten.zeros_like": lambda r, n, a, ov: _h_full_like(r, n, a, ov, 0.0),
    "aten.ones_like": lambda r, n, a, ov: _h_full_like(r, n, a, ov, 1.0),
    "aten.scalar_tensor": _h_scalar_tensor,
    "aten.tril": _h_trilu(0), "aten.triu": _h_trilu(1), "aten.constant_pad_nd": _h_pad,
    "aten.arange": _h_arange, "aten.sym_size": _h_sym_size, "aten.sym_numel": _h_sym_numel,
    # attention-mask construction of transformer exports (lane ag-import-lm-ops)
    "aten.any": _h_any_all("ReduceMax"), "aten.all": _h_any_all("ReduceMin"),
    "aten.bitwise_and": _h_bitwise("And"), "aten.bitwise_or": _h_bitwise("Or"),
    "aten.bitwise_not": _h_bitwise_not, "aten.cumsum": _h_cumsum,
})


def supported_targets():
    """The ATen operations (core ATen, after ``run_decompositions``) the reader maps, by name."""
    return tuple(sorted(_HANDLERS))  # cpu-route: model graph description on the host (op names)


def _live(graph):
    """The FX nodes the graph's output depends on (assertion nodes have no users and drop out)."""
    out = [n for n in graph.nodes if n.op == "output"]  # cpu-route: model graph description on the host (graph nodes)
    live, stack = set(), list(out)
    while stack:  # cpu-route: model graph description on the host (graph nodes)
        n = stack.pop()
        if n in live:
            continue
        live.add(n)
        stack.extend(n.all_input_nodes)
    return live


def _prune(nodes, outputs):
    """Drop IR nodes none of whose outputs reach a graph output (the unused
    second result of a node that has two, such as ``max.dim``'s indices)."""
    need = set(outputs)
    keep = []
    for nd in reversed(nodes):  # cpu-route: model graph description on the host (graph nodes)
        if any(o in need for o in nd.outputs):  # cpu-route: model graph description on the host (node outputs)
            keep.append(nd)
            need.update(i for i in nd.inputs if i)  # cpu-route: model graph description on the host (node inputs)
    keep.reverse()
    return tuple(keep), need


def read(ep):
    """Read a ``torch.export`` program into the import graph.

    ``ep`` is a :class:`torch.export.ExportedProgram`, for example
    ``torch.export.export(model.eval(), (x,))``. Its operations are first
    lowered to core ATen (``ep.run_decompositions()``); the model's
    parameters, buffers and constants become the graph's weights, and its
    user inputs and outputs keep their names. An input axis exported as
    dynamic (``torch.export.Dim``) stays open and is fixed at import by
    ``shapes``.

    Raises :class:`UnsupportedModel` listing every operation, argument or
    tensor type the import does not support, by node name.

    >>> import torch, mojolearn
    >>> model = torch.nn.Sequential(torch.nn.Linear(6, 3), torch.nn.ReLU()).eval()
    >>> ep = torch.export.export(model, (torch.zeros(4, 6),))
    >>> prog = mojolearn.import_model(ep)
    """
    import torch  # torch is needed only here; importing mojolearn never imports it

    if not isinstance(ep, torch.export.ExportedProgram):
        raise TypeError(f"expected a torch.export.ExportedProgram, got {type(ep).__name__}")
    ep = ep.run_decompositions()
    r = _Reader(torch, ep)
    sig = ep.graph_signature
    graph = ep.graph_module.graph
    live = _live(graph)
    specs = {s.arg.name: s for s in sig.input_specs if hasattr(s.arg, "name")}  # cpu-route: model graph description on the host (graph inputs)
    state, consts = ep.state_dict, getattr(ep, "constants", {}) or {}
    inputs = []
    for n in graph.nodes:  # cpu-route: model graph description on the host (graph nodes)
        if n.op == "output":
            continue
        spec = specs.get(n.name) if n.op == "placeholder" else None
        kind = (spec.kind.name if spec is not None else "USER_INPUT") if n.op == "placeholder" else None
        # every user input stays an input of the import (even one the model
        # ignores), so the imported model takes the arrays the module does
        if n not in live and kind != "USER_INPUT":
            continue
        try:
            if n.op == "placeholder":
                if kind == "USER_INPUT":
                    v = n.meta.get("val")
                    if not hasattr(v, "shape"):
                        raise _Refuse("a non-tensor input (export the size as part of a tensor's shape)")
                    dt = r.dtypes.get(v.dtype)
                    if dt is None:
                        raise _Refuse(f"input type {v.dtype} is not supported")
                    dims = tuple(int(d) if isinstance(d, int) else str(d) for d in v.shape)  # cpu-route: model graph description on the host (one per axis)
                    inputs.append((n.name, dt, dims))
                elif kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                    t = state.get(spec.target) if spec.target in state else consts.get(spec.target)
                    if t is None:
                        raise _Refuse(f"{kind.lower()} '{spec.target}' has no tensor in the program")
                    r.inits[n.name] = r.tensor_info(n.name, t)
                else:
                    raise _Refuse(f"input kind {kind} is not supported")
            elif n.op == "get_attr":
                t = getattr(ep.graph_module, n.target)
                if not isinstance(t, torch.Tensor):
                    raise _Refuse(f"attribute '{n.target}' is not a tensor")
                r.inits[n.name] = r.tensor_info(n.name, t)
            elif n.op == "call_function":
                _call(r, n)
            else:
                raise _Refuse(f"FX node kind {n.op!r} is not supported")
        except _Refuse as e:
            r.failed.add(n.name)
            r.problems.append((n.name, _target_name(n.target)[0] if n.op == "call_function" else None, str(e)))
    outputs = []
    out_node = next(n for n in graph.nodes if n.op == "output")  # cpu-route: model graph description on the host (graph nodes)
    out_kinds = [s.kind.name for s in sig.output_specs]  # cpu-route: model graph description on the host (graph outputs)
    flat = out_node.args[0] if isinstance(out_node.args[0], (list, tuple)) else (out_node.args[0],)
    for i, o in enumerate(flat):  # cpu-route: model graph description on the host (graph outputs)
        kind = out_kinds[i] if i < len(out_kinds) else "USER_OUTPUT"
        if kind != "USER_OUTPUT":
            r.problems.append((getattr(o, "name", f"output {i}"), None,
                               f"output kind {kind} (the model changes its own state; call model.eval() "
                               f"before torch.export.export)"))
            continue
        if not r.is_node(o):
            r.problems.append((f"output {i}", None, f"a constant output {o!r} is not supported"))
            continue
        try:
            name = r.val(o)
            v = r.meta(o)
            dt = r.dtypes.get(v.dtype)
            if dt is None:
                raise _Refuse(f"output type {v.dtype} is not supported")
            dims = tuple(int(d) if isinstance(d, int) else str(d) for d in v.shape)  # cpu-route: model graph description on the host (one per axis)
            outputs.append((name, dt, dims))
        except _Refuse as e:
            r.problems.append((o.name, None, str(e)))
    if r.problems:
        raise UnsupportedModel(SOURCE, r.problems)
    nodes, need = _prune(r.nodes, [o[0] for o in outputs])  # cpu-route: model graph description on the host (graph outputs)
    inits = {k: t for k, t in r.inits.items() if k in need}  # cpu-route: model graph description on the host (initializer names)
    return GraphIR(opset=OPSET, inputs=tuple(inputs), outputs=tuple(outputs), nodes=nodes, initializers=inits,
                   source=SOURCE)


def _call(r, n):
    """Translate one call_function node, or record why it cannot be."""
    target = n.target
    if target is operator.getitem:
        src, idx = n.args[0], int(n.args[1])
        if src.name in r.failed:
            r.failed.add(n.name)  # the source node is already reported
            return
        outs = r.multi.get(src.name)
        if outs is None:
            raise _Refuse(f"indexes '{src.name}', which has no separate outputs")
        if idx not in outs:
            raise _Refuse(f"uses output {idx} of '{src.name}' (only {sorted(outs)} are supported)")  # cpu-route: model graph description on the host (output indices)
        r.alias[n.name] = outs[idx]
        return
    if target in _SYM_BIN:
        x, y = n.args[:2]
        r.emit(_SYM_BIN[target], [r.int_scalar(x), r.int_scalar(y)], [n.name], n.name)
        return
    packet, overload = _target_name(target)
    if packet in _SKIP:
        return
    if _is_higher_order(target):
        raise _Refuse("control flow and wrapped subgraphs are not supported")
    h = _HANDLERS.get(packet)
    if h is None:
        raise _Refuse("no import rule; see mojolearn.importers.supported_ops()")
    out = h(r, n, _bind(n), overload)
    if isinstance(out, dict):
        r.multi[n.name] = out
