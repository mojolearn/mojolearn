# SPDX-License-Identifier: Apache-2.0
"""The Python static shape resolution the import used before it moved to
Mojo (autograd/import_resolve.mojo): the test oracle the native resolution
is compared with (tests/test_import_onnx_native.py), and the integer decode
of shape constants the Python lowering oracle (tests/_lower_oracle.py) reads.
Never imported by the library."""
from __future__ import annotations

import struct

from mojolearn.importers._graph import (INT_DTYPES, ONNX_DTYPE, Resolved, TensorInfo, onnx_dtype_name)

_STRUCT = {"i64": "q", "i32": "i", "bool": "B"}

def _numel(dims):
    """The element count of a shape."""
    n = 1
    for d in dims:
        n *= d
    return n


def int_values(t):
    """The integers of an integer or bool TensorInfo, row-major (a shape
    constant, axes, split sizes, constant indices). Description only: the
    lowering never uses this on floating-point data."""
    if t.dtype not in _STRUCT:
        raise TypeError(f"{t.name}: {t.dtype} is not an integer tensor")
    n = _numel(t.shape)
    return struct.unpack(f"<{n}{_STRUCT[t.dtype]}", t.data) if n else ()


class _Refuse(Exception):
    pass


class _Ctx:
    def __init__(self, g, shapes, dtypes, folded):
        self.g, self.S, self.D, self.F = g, shapes, dtypes, folded

    def shape(self, name):
        return self.S[name]

    def dtype(self, name):
        return self.D[name]

    def maybe_ints(self, name):
        """The folded integers of ``name`` or None."""
        if name in self.F:
            return self.F[name]
        t = self.g.initializers.get(name)
        if t is not None and t.dtype in INT_DTYPES:
            v = int_values(t)
            self.F[name] = v
            return v
        return None

    def ints(self, name, what):
        v = self.maybe_ints(name)
        if v is None:
            raise _Refuse(f"{what} '{name}' is computed from tensor values (a data-dependent shape)")
        return v


def _norm_axis(a, rank, what="axis"):
    if not -rank <= a < max(rank, 1):
        raise _Refuse(f"{what} {a} is out of range for rank {rank}")
    return a + rank if a < 0 else a


def _bcast(*shapes):
    rank = max(len(s) for s in shapes)
    out = []
    for i in range(rank):
        d = 1
        for s in shapes:
            k = i - (rank - len(s))
            if k < 0 or s[k] == 1:
                continue
            if d != 1 and s[k] != d:
                raise _Refuse("shapes " + ", ".join(str(tuple(x)) for x in shapes)
                              + " do not broadcast")
            d = s[k]
        out.append(d)
    return tuple(out)


def _strides(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


def _bcast_values(vals, shape, out_shape):
    """Row-major folded integers of ``vals`` (shape ``shape``) broadcast to ``out_shape``."""
    if tuple(shape) == tuple(out_shape):
        return tuple(vals)
    pad = (1,) * (len(out_shape) - len(shape)) + tuple(shape)
    st = _strides(pad)
    st = [0 if pad[i] == 1 else st[i] for i in range(len(pad))]
    ost = _strides(out_shape)
    out = []
    for flat in range(_numel(out_shape)):
        src = 0
        for i, s in enumerate(ost):
            src += ((flat // s) % out_shape[i]) * st[i]
        out.append(vals[src])
    return tuple(out)


def _int_div(a, b):
    if b == 0:
        raise _Refuse("integer division by zero in a shape computation")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


_FOLD_BIN = {
    "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b, "Mul": lambda a, b: a * b, "Div": _int_div,
    "Equal": lambda a, b: int(a == b), "Less": lambda a, b: int(a < b), "Greater": lambda a, b: int(a > b),
    "LessOrEqual": lambda a, b: int(a <= b), "GreaterOrEqual": lambda a, b: int(a >= b),
    "And": lambda a, b: int(bool(a) and bool(b)), "Or": lambda a, b: int(bool(a) or bool(b)),
    "Max": lambda a, b: a if a >= b else b, "Min": lambda a, b: a if a <= b else b,
}
_BOOL_RESULT = frozenset(("Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or", "Not"))


def _out(shape, dtype, folded=None):
    return (tuple(shape), dtype, None if folded is None else tuple(folded))


def _r_unary(c, n):
    x = n.inputs[0]
    dt = "bool" if n.op == "Not" else c.dtype(x)
    folded = None
    v = c.maybe_ints(x) if n.op in ("Identity", "Neg", "Abs", "Not") else None
    if v is not None:
        f = {"Identity": lambda a: a, "Neg": lambda a: -a, "Abs": abs, "Not": lambda a: int(not a)}[n.op]
        folded = [f(a) for a in v]
    return [_out(c.shape(x), dt, folded)]


def _r_dropout(c, n):
    s = c.shape(n.inputs[0])
    return [_out(s, c.dtype(n.inputs[0])), _out(s, "bool")][:len(n.outputs)]


def _r_binary(c, n):
    ins = [i for i in n.inputs if i]
    if not ins:
        raise _Refuse("no inputs")
    out_shape = _bcast(*[c.shape(i) for i in ins])
    dt = "bool" if n.op in _BOOL_RESULT else c.dtype(ins[0])
    folded = None
    vals = [c.maybe_ints(i) for i in ins]
    if n.op in _FOLD_BIN and all(v is not None for v in vals):
        bv = [_bcast_values(v, c.shape(i), out_shape) for v, i in zip(vals, ins)]
        f = _FOLD_BIN[n.op]
        acc = list(bv[0])
        for other in bv[1:]:
            acc = [f(a, b) for a, b in zip(acc, other)]
        folded = acc
    return [_out(out_shape, dt, folded)]


def _r_where(c, n):
    cond, a, b = n.inputs[:3]
    out_shape = _bcast(c.shape(cond), c.shape(a), c.shape(b))
    folded = None
    vc, va, vb = c.maybe_ints(cond), c.maybe_ints(a), c.maybe_ints(b)
    if vc is not None and va is not None and vb is not None:
        vc = _bcast_values(vc, c.shape(cond), out_shape)
        va = _bcast_values(va, c.shape(a), out_shape)
        vb = _bcast_values(vb, c.shape(b), out_shape)
        folded = [x if k else y for k, x, y in zip(vc, va, vb)]
    return [_out(out_shape, c.dtype(a), folded)]


def _r_clip(c, n):
    return [_out(c.shape(n.inputs[0]), c.dtype(n.inputs[0]))]


def _r_matmul(c, n):
    a, b = c.shape(n.inputs[0]), c.shape(n.inputs[1])
    if not a or not b:
        raise _Refuse("MatMul of a scalar")
    sq_a, sq_b = len(a) == 1, len(b) == 1
    a2 = (1,) + tuple(a) if sq_a else tuple(a)
    b2 = tuple(b) + (1,) if sq_b else tuple(b)
    if a2[-1] != b2[-2]:
        raise _Refuse(f"inner dimensions differ: {tuple(a)} @ {tuple(b)}")
    out = _bcast(a2[:-2], b2[:-2]) + (a2[-2], b2[-1])
    if sq_a:
        out = out[:-2] + out[-1:]
    if sq_b:
        out = out[:-1]
    return [_out(out, c.dtype(n.inputs[0]))]


def _r_gemm(c, n):
    a, b = c.shape(n.inputs[0]), c.shape(n.inputs[1])
    if len(a) != 2 or len(b) != 2:
        raise _Refuse(f"Gemm needs 2-D operands, got {tuple(a)} and {tuple(b)}")
    m, k = (a[1], a[0]) if n.attrs.get("transA", 0) else (a[0], a[1])
    k2, nn = (b[1], b[0]) if n.attrs.get("transB", 0) else (b[0], b[1])
    if k != k2:
        raise _Refuse(f"inner dimensions differ: {tuple(a)} and {tuple(b)} (transA={n.attrs.get('transA', 0)}, "
                      f"transB={n.attrs.get('transB', 0)})")
    if len(n.inputs) > 2 and n.inputs[2]:
        if len(_bcast(c.shape(n.inputs[2]), (m, nn))) != 2:
            raise _Refuse(f"bias {tuple(c.shape(n.inputs[2]))} does not broadcast to {(m, nn)}")
    return [_out((m, nn), c.dtype(n.inputs[0]))]


def _r_norm_last(c, n):
    s = c.shape(n.inputs[0])
    outs = [_out(s, c.dtype(n.inputs[0]))]
    if len(n.outputs) > 1:
        ax = _norm_axis(n.attrs.get("axis", -1), len(s))
        stat = tuple(s[:ax]) + (1,) * (len(s) - ax)
        outs += [_out(stat, "f32")] * (len(n.outputs) - 1)
    return outs


def _r_batchnorm(c, n):
    s = c.shape(n.inputs[0])
    outs = [_out(s, c.dtype(n.inputs[0]))]
    if len(n.outputs) > 1:
        if len(s) < 2:
            raise _Refuse("BatchNormalization needs a channel axis")
        outs += [_out((s[1],), c.dtype(n.inputs[0]))] * (len(n.outputs) - 1)
    return outs


def _r_same(c, n):
    return [_out(c.shape(n.inputs[0]), c.dtype(n.inputs[0]))]


def _r_transpose(c, n):
    s = c.shape(n.inputs[0])
    perm = n.attrs.get("perm")
    perm = tuple(range(len(s) - 1, -1, -1)) if perm is None else tuple(perm)
    if sorted(_norm_axis(p, len(s), "perm entry") for p in perm) != list(range(len(s))):
        raise _Refuse(f"perm {perm} is not a permutation of rank {len(s)}")
    perm = tuple(_norm_axis(p, len(s)) for p in perm)
    out = tuple(s[p] for p in perm)
    folded = None
    v = c.maybe_ints(n.inputs[0])
    if v is not None and len(s) <= 1:
        folded = v
    return [_out(out, c.dtype(n.inputs[0]), folded)]


def _r_reshape(c, n):
    x = n.inputs[0]
    s = c.shape(x)
    target = list(c.ints(n.inputs[1], "the target shape"))
    allowzero = n.attrs.get("allowzero", 0)
    neg = [i for i, d in enumerate(target) if d == -1]
    if len(neg) > 1:
        raise _Refuse(f"target shape {tuple(target)} has more than one -1")
    for i, d in enumerate(target):
        if d == 0 and not allowzero:
            if i >= len(s):
                raise _Refuse(f"target shape {tuple(target)}: 0 at axis {i} has no input axis to copy")
            target[i] = s[i]
        elif d < -1:
            raise _Refuse(f"target shape {tuple(target)} has a negative size")
    total = _numel(s)
    if neg:
        rest = _numel(d for i, d in enumerate(target) if i != neg[0])
        if rest == 0 or total % rest:
            raise _Refuse(f"cannot reshape {tuple(s)} to {tuple(target)}")
        target[neg[0]] = total // rest
    if _numel(target) != total:
        raise _Refuse(f"cannot reshape {tuple(s)} to {tuple(target)}")
    return [_out(target, c.dtype(x), c.maybe_ints(x))]


def _r_flatten(c, n):
    s = c.shape(n.inputs[0])
    ax = n.attrs.get("axis", 1)
    ax = ax + len(s) if ax < 0 else ax
    if not 0 <= ax <= len(s):
        raise _Refuse(f"axis {n.attrs.get('axis', 1)} is out of range for rank {len(s)}")
    return [_out((_numel(s[:ax]), _numel(s[ax:])), c.dtype(n.inputs[0]), c.maybe_ints(n.inputs[0]))]


def _axes_of(c, n, what="axes"):
    """Axes from the second input (opset 13+) or the attribute; None when absent."""
    if len(n.inputs) > 1 and n.inputs[1]:
        return tuple(c.ints(n.inputs[1], what))
    a = n.attrs.get("axes")
    return None if a is None else tuple(a)


def _r_squeeze(c, n):
    s = c.shape(n.inputs[0])
    axes = _axes_of(c, n)
    if axes is None:
        keep = [d for d in s if d != 1]
    else:
        ax = {_norm_axis(a, len(s)) for a in axes}
        bad = [a for a in sorted(ax) if s[a] != 1]
        if bad:
            raise _Refuse(f"cannot squeeze axes {bad} of {tuple(s)} (not size 1)")
        keep = [d for i, d in enumerate(s) if i not in ax]
    return [_out(keep, c.dtype(n.inputs[0]), c.maybe_ints(n.inputs[0]))]


def _r_unsqueeze(c, n):
    s = c.shape(n.inputs[0])
    axes = _axes_of(c, n)
    if axes is None:
        raise _Refuse("no axes")
    r = len(s) + len(axes)
    ax = sorted(_norm_axis(a, r) for a in axes)
    if len(set(ax)) != len(ax):
        raise _Refuse(f"repeated axes {tuple(axes)}")
    out, it = [], iter(s)
    for i in range(r):
        out.append(1 if i in ax else next(it))
    return [_out(out, c.dtype(n.inputs[0]), c.maybe_ints(n.inputs[0]))]


def _r_expand(c, n):
    x = n.inputs[0]
    target = c.ints(n.inputs[1], "the target shape")
    out = _bcast(c.shape(x), tuple(target))
    v = c.maybe_ints(x)
    return [_out(out, c.dtype(x), None if v is None else _bcast_values(v, c.shape(x), out))]


def _r_concat(c, n):
    ins = [i for i in n.inputs if i]
    shapes = [c.shape(i) for i in ins]
    r = len(shapes[0])
    ax = _norm_axis(n.attrs.get("axis", 0), r)
    for s in shapes[1:]:
        if len(s) != r or any(s[i] != shapes[0][i] for i in range(r) if i != ax):
            raise _Refuse("inputs of shapes " + ", ".join(str(tuple(x)) for x in shapes)
                          + f" do not concatenate on axis {ax}")
    out = list(shapes[0])
    out[ax] = sum(s[ax] for s in shapes)
    folded = None
    vals = [c.maybe_ints(i) for i in ins]
    if r == 1 and all(v is not None for v in vals):
        folded = [a for v in vals for a in v]
    return [_out(out, c.dtype(ins[0]), folded)]


def _r_split(c, n):
    s = c.shape(n.inputs[0])
    ax = _norm_axis(n.attrs.get("axis", 0), len(s))
    k = len(n.outputs)
    if len(n.inputs) > 1 and n.inputs[1]:
        sizes = list(c.ints(n.inputs[1], "the split sizes"))
    elif "split" in n.attrs:
        sizes = list(n.attrs["split"])
    else:
        d = s[ax]
        if "num_outputs" in n.attrs:
            each = -(-d // k)
            sizes = [each] * (k - 1) + [d - each * (k - 1)]
        elif d % k == 0:
            sizes = [d // k] * k
        else:
            raise _Refuse(f"axis {ax} of size {d} does not split into {k} equal pieces")
    if len(sizes) != k or sum(sizes) != s[ax] or min(sizes) < 0:
        raise _Refuse(f"split sizes {tuple(sizes)} do not cover axis {ax} of {tuple(s)} in {k} outputs")
    outs = []
    for sz in sizes:
        o = list(s)
        o[ax] = sz
        outs.append(_out(o, c.dtype(n.inputs[0])))
    return outs


def _slice_range(start, end, step, dim):
    """ONNX Slice clamping: (start, length) of one axis."""
    if step == 0:
        raise _Refuse("Slice step 0")
    if start < 0:
        start += dim
    if end < 0:
        end += dim
    if step > 0:
        start = min(max(start, 0), dim)
        end = min(max(end, 0), dim)
        length = max(0, -(-(end - start) // step))
    else:
        start = min(max(start, 0), dim - 1)
        end = min(max(end, -1), dim - 1)
        length = max(0, -(-(start - end) // -step))
    return start, length


def _r_slice(c, n):
    x = n.inputs[0]
    s = list(c.shape(x))
    starts = c.ints(n.inputs[1], "the slice starts")
    ends = c.ints(n.inputs[2], "the slice ends")
    axes = c.ints(n.inputs[3], "the slice axes") if len(n.inputs) > 3 and n.inputs[3] else tuple(range(len(starts)))
    steps = c.ints(n.inputs[4], "the slice steps") if len(n.inputs) > 4 and n.inputs[4] else (1,) * len(starts)
    if not len(starts) == len(ends) == len(axes) == len(steps):
        raise _Refuse("starts, ends, axes and steps differ in length")
    out = list(s)
    picks = {}
    for st, en, ax, sp in zip(starts, ends, axes, steps):
        a = _norm_axis(ax, len(s))
        b, ln = _slice_range(st, en, sp, s[a])
        out[a] = ln
        picks[a] = (b, sp, ln)
    folded = None
    v = c.maybe_ints(x)
    if v is not None and len(s) == 1:
        b, sp, ln = picks.get(0, (0, 1, s[0]))
        folded = [v[b + i * sp] for i in range(ln)]
    return [_out(out, c.dtype(x), folded)]


def _r_gather(c, n):
    data, idx = n.inputs[0], n.inputs[1]
    s, si = c.shape(data), c.shape(idx)
    ax = _norm_axis(n.attrs.get("axis", 0), len(s))
    out = tuple(s[:ax]) + tuple(si) + tuple(s[ax + 1:])
    iv = c.maybe_ints(idx)
    if iv is not None:
        bad = [i for i in iv if not -s[ax] <= i < s[ax]]
        if bad:
            raise _Refuse(f"constant indices {bad[:4]} out of range for axis {ax} of size {s[ax]}")
    folded = None
    dv = c.maybe_ints(data)
    if dv is not None and iv is not None and len(s) == 1:
        folded = [dv[i + s[0] if i < 0 else i] for i in iv]
    return [_out(out, c.dtype(data), folded)]


def _r_gather_nd(c, n):
    s, si = c.shape(n.inputs[0]), c.shape(n.inputs[1])
    if n.attrs.get("batch_dims", 0):
        raise _Refuse("GatherND with batch_dims (only 0 is supported)")
    if not si:
        raise _Refuse("GatherND indices of rank 0")
    k = si[-1]
    if not isinstance(k, int) or k < 1 or k > len(s):
        raise _Refuse(f"GatherND index depth {k} for a tensor of rank {len(s)}")
    return [_out(tuple(si[:-1]) + tuple(s[k:]), c.dtype(n.inputs[0]))]


def _r_reduce(c, n):
    s = c.shape(n.inputs[0])
    axes = _axes_of(c, n)
    keep = n.attrs.get("keepdims", 1)
    if not axes:
        if n.attrs.get("noop_with_empty_axes", 0):
            return [_out(s, c.dtype(n.inputs[0]))]
        axes = tuple(range(len(s)))
    ax = {_norm_axis(a, len(s)) for a in axes}
    if keep:
        out = [1 if i in ax else d for i, d in enumerate(s)]
    else:
        out = [d for i, d in enumerate(s) if i not in ax]
    return [_out(out, c.dtype(n.inputs[0]))]


def _r_argmax(c, n):
    s = c.shape(n.inputs[0])
    ax = _norm_axis(n.attrs.get("axis", 0), len(s))
    if n.attrs.get("keepdims", 1):
        out = [1 if i == ax else d for i, d in enumerate(s)]
    else:
        out = [d for i, d in enumerate(s) if i != ax]
    return [_out(out, "i64")]


def _r_global_pool(c, n):
    s = c.shape(n.inputs[0])
    if len(s) < 3:
        raise _Refuse(f"{n.op} needs a rank 3 or more input, got {tuple(s)}")
    return [_out(tuple(s[:2]) + (1,) * (len(s) - 2), c.dtype(n.inputs[0]))]


def _window_out(n, s, kernel):
    """Output spatial sizes of a Conv or pool window over ``s`` (N, C, spatial...)."""
    sp = len(s) - 2
    strides = tuple(n.attrs.get("strides", (1,) * sp))
    dil = tuple(n.attrs.get("dilations", (1,) * sp))
    pads = tuple(n.attrs.get("pads", (0,) * (2 * sp)))
    auto = n.attrs.get("auto_pad", "NOTSET")
    ceil = n.attrs.get("ceil_mode", 0)
    if len(kernel) != sp or len(strides) != sp or len(dil) != sp or len(pads) != 2 * sp:
        raise _Refuse(f"kernel {kernel}, strides {strides}, dilations {dil} or pads {pads} do not match "
                      f"{sp} spatial axes")
    out = []
    for i in range(sp):
        d, k = s[2 + i], (kernel[i] - 1) * dil[i] + 1
        if auto in ("SAME_UPPER", "SAME_LOWER"):
            o = -(-d // strides[i])
        elif auto == "VALID":
            o = (d - k) // strides[i] + 1
        else:
            span = d + pads[i] + pads[sp + i] - k
            o = (-(-span // strides[i]) if ceil else span // strides[i]) + 1
            if ceil and (o - 1) * strides[i] >= d + pads[i]:
                o -= 1
        if o <= 0:
            raise _Refuse(f"the window leaves no output along spatial axis {i} of {tuple(s)}")
        out.append(o)
    return tuple(out)


def _r_conv(c, n):
    s, w = c.shape(n.inputs[0]), c.shape(n.inputs[1])
    if len(s) < 3 or len(w) != len(s):
        raise _Refuse(f"input {tuple(s)} and weight {tuple(w)} ranks do not match")
    group = n.attrs.get("group", 1)
    if s[1] != w[1] * group:
        raise _Refuse(f"input channels {s[1]} do not match weight {tuple(w)} with group={group}")
    kernel = tuple(n.attrs.get("kernel_shape", w[2:]))
    if kernel != tuple(w[2:]):
        raise _Refuse(f"kernel_shape {kernel} does not match the weight {tuple(w)}")
    if len(n.inputs) > 2 and n.inputs[2] and tuple(c.shape(n.inputs[2])) != (w[0],):
        raise _Refuse(f"bias {tuple(c.shape(n.inputs[2]))} does not match {w[0]} output channels")
    return [_out((s[0], w[0]) + _window_out(n, s, kernel), c.dtype(n.inputs[0]))]


def _r_pool(c, n):
    s = c.shape(n.inputs[0])
    if len(s) < 3:
        raise _Refuse(f"{n.op} needs a rank 3 or more input, got {tuple(s)}")
    if "kernel_shape" not in n.attrs:
        raise _Refuse("no kernel_shape")
    out = tuple(s[:2]) + _window_out(n, s, tuple(n.attrs["kernel_shape"]))
    outs = [_out(out, c.dtype(n.inputs[0]))]
    if len(n.outputs) > 1:
        outs.append(_out(out, "i64"))
    return outs


def _r_shape(c, n):
    s = c.shape(n.inputs[0])
    r = len(s)
    a, b = n.attrs.get("start", 0), n.attrs.get("end", r)
    a = min(max(a + r if a < 0 else a, 0), r)
    b = min(max(b + r if b < 0 else b, 0), r)
    v = tuple(s[a:b])
    return [_out((len(v),), "i64", v)]


def _r_size(c, n):
    return [_out((), "i64", (_numel(c.shape(n.inputs[0])),))]


def _r_constant_of_shape(c, n):
    shape = tuple(c.ints(n.inputs[0], "the shape"))
    if any(d < 0 for d in shape):
        raise _Refuse(f"negative size in {shape}")
    t = n.attrs.get("value")
    dt = t.dtype if isinstance(t, TensorInfo) else "f32"
    folded = None
    if isinstance(t, TensorInfo) and dt in INT_DTYPES:
        folded = int_values(t)[:1] * _numel(shape)
    return [_out(shape, dt, folded)]


def _r_range(c, n):
    a, b, d = (c.maybe_ints(i) for i in n.inputs[:3])
    if a is None or b is None or d is None:
        raise _Refuse("Range bounds are not integer constants or folded shape values")
    a, b, d = a[0], b[0], d[0]
    if d == 0:
        raise _Refuse("Range delta 0")
    count = max(-(-(b - a) // d), 0)
    return [_out((count,), c.dtype(n.inputs[0]), [a + i * d for i in range(count)])]


def _r_cast(c, n):
    to = n.attrs.get("to")
    dt = ONNX_DTYPE.get(to)
    if dt is None:
        raise _Refuse(f"cast to {onnx_dtype_name(to)}")
    v = c.maybe_ints(n.inputs[0])
    folded = None
    if v is not None and dt in INT_DTYPES:
        folded = [int(bool(a)) for a in v] if dt == "bool" else v
    return [_out(c.shape(n.inputs[0]), dt, folded)]


def _r_pad(c, n):
    s = c.shape(n.inputs[0])
    pads = tuple(c.ints(n.inputs[1], "the pads")) if len(n.inputs) > 1 and n.inputs[1] else tuple(n.attrs.get("pads", ()))
    axes = (tuple(c.ints(n.inputs[3], "the pad axes")) if len(n.inputs) > 3 and n.inputs[3]
            else tuple(range(len(s))))
    if len(pads) != 2 * len(axes):
        raise _Refuse(f"pads {pads} do not match {len(axes)} axes")
    out = list(s)
    for j, a in enumerate(axes):
        a = _norm_axis(a, len(s))
        out[a] = s[a] + pads[j] + pads[len(axes) + j]
        if out[a] < 0:
            raise _Refuse(f"pads {pads} leave a negative size on axis {a}")
    return [_out(out, c.dtype(n.inputs[0]))]


def _r_constant(c, n):
    t = n.attrs.get("value")
    if not isinstance(t, TensorInfo):
        raise _Refuse("a Constant without a tensor value")
    return [_out(t.shape, t.dtype, int_values(t) if t.dtype in INT_DTYPES else None)]


_UNARY = ("Relu", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Neg", "Reciprocal", "Erf", "Gelu", "Softplus",
          "LeakyRelu", "Elu", "HardSigmoid", "HardSwish", "Abs", "Floor", "Ceil", "Round", "Sin", "Cos", "Sign",
          "Not", "Identity", "Softmax", "LogSoftmax", "Trilu")
_BINARY = ("Add", "Sub", "Mul", "Div", "Pow", "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "And",
           "Or", "Max", "Min")
_REDUCE = ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceL2", "ReduceProd")

# shape rules by op (default domain): each returns one (shape, dtype, folded) per output
SHAPE_RULES = {op: _r_unary for op in _UNARY}
SHAPE_RULES.update({op: _r_binary for op in _BINARY})
SHAPE_RULES.update({op: _r_reduce for op in _REDUCE})
SHAPE_RULES.update({
    "Dropout": _r_dropout, "Where": _r_where, "Clip": _r_clip, "MatMul": _r_matmul, "Gemm": _r_gemm,
    "LayerNormalization": _r_norm_last, "RMSNormalization": _r_norm_last,
    "BatchNormalization": _r_batchnorm, "InstanceNormalization": _r_same, "GroupNormalization": _r_same,
    "Transpose": _r_transpose, "Reshape": _r_reshape, "Flatten": _r_flatten, "Squeeze": _r_squeeze,
    "Unsqueeze": _r_unsqueeze, "Expand": _r_expand, "Concat": _r_concat, "Split": _r_split, "Slice": _r_slice,
    "Gather": _r_gather, "ArgMax": _r_argmax, "GlobalAveragePool": _r_global_pool,
    "GlobalMaxPool": _r_global_pool, "Conv": _r_conv, "MaxPool": _r_pool, "AveragePool": _r_pool,
    "Shape": _r_shape, "Size": _r_size, "ConstantOfShape": _r_constant_of_shape, "Range": _r_range,
    "Cast": _r_cast, "Pad": _r_pad, "Constant": _r_constant,
    "CumSum": _r_same, "GatherND": _r_gather_nd,
})


def resolve(g, shapes=None):
    """Every value's static shape and dtype from the inputs' resolved shapes,
    with the integer shape subgraphs folded. Never raises for a graph it
    cannot resolve: the problems (an input axis left open, a shape that
    depends on tensor values, inconsistent shapes) are in ``problems`` and
    the nodes with no shape rule in ``no_rule``, so the lowering can report
    them together with its own refusals in one error."""
    given = dict(shapes or {})
    problems = []
    S, D = {}, {}
    for name, t in g.initializers.items():
        S[name], D[name] = tuple(t.shape), t.dtype
    known = set()
    for name, dt, dims in g.inputs:
        known.add(name)
        if name in given:
            want = tuple(int(d) for d in given[name])
            fixed = [(i, d) for i, d in enumerate(dims) if isinstance(d, int)]
            if len(want) != len(dims) or any(want[i] != d for i, d in fixed):
                problems.append((name, None, f"shape {want} does not match the model's {tuple(dims)}"))
                continue
            if any(d <= 0 for d in want):
                problems.append((name, None, f"shape {want} has an empty or negative axis"))
                continue
            S[name] = want
        else:
            open_axes = [i for i, d in enumerate(dims) if not isinstance(d, int) or d <= 0]
            if open_axes:
                problems.append((name, None, f"axes {open_axes} of {tuple(dims)} are open; pass "
                                             f"shapes={{{name!r}: (...)}}"))
                continue
            S[name] = tuple(dims)
        D[name] = dt
    for name in given:
        if name not in known:
            problems.append((name, None, "is not an input of the model (inputs: "
                             + ", ".join(i[0] for i in g.inputs) + ")"))
    F = {}
    ctx = _Ctx(g, S, D, F)
    no_rule = []
    for n in g.nodes:
        if any(i and i not in S for i in n.inputs):
            continue  # an upstream node is already reported
        rule = SHAPE_RULES.get(n.op) if n.domain in ("", "ai.onnx") else None
        if rule is None:
            no_rule.append(n.name)
            continue
        try:
            outs = rule(ctx, n)
        except _Refuse as e:
            problems.append((n.name, n.op, str(e)))
            continue
        for name, (shp, dt, folded) in zip(n.outputs, outs):
            if not name:
                continue
            S[name], D[name] = shp, dt
            if folded is not None:
                F[name] = folded
    for name, dt, dims in g.outputs:
        if name not in S:
            continue
        got = S[name]
        if len(dims) == len(got) and any(isinstance(d, int) and d > 0 and d != e for d, e in zip(dims, got)):
            problems.append((name, None, f"the model declares shape {tuple(dims)}, the graph computes {got}"))
    return Resolved(S, D, F, tuple(no_rule), tuple(problems))



def resolve_shapes(g, shapes=None):
    """``resolve``'s shapes; raises UnsupportedModel on any problem."""
    from mojolearn.importers._graph import UnsupportedModel
    r = resolve(g, shapes)
    if r.problems:
        raise UnsupportedModel(g.source, r.problems)
    return r.shapes
