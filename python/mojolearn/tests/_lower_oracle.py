# SPDX-License-Identifier: Apache-2.0
"""The Python lowering of an import graph onto session operations: the TEST
ORACLE of autograd/import_lower.mojo (docs/plans/MODEL_IMPORT.md section 7,
step 2). It is the lowering importers/_lower.py ran before the lowering moved
to Mojo, kept unchanged so tests/test_import_lower_native.py can assert the
Mojo-lowered program equals it word for word; ``import_model`` below is the
former importers/_program.py import_model over it.

Original notes (section 3.1, lane ag-import-lower):

``check`` is the refusal pass: every node the outputs need is matched
against the table and its supported subset, and every problem (with the
shape resolution's own) comes back at once, before anything is uploaded.
``lower`` then replays the graph once through an inference ``Session``:

* initializers are uploaded through the cast path (float16, bfloat16 and
  float64 widened or rounded on the device; a ``Gemm`` weight stored
  ``[out, in]`` transposed on the device by the weight import's layout code);
* the parameter-only subgraphs (BatchNorm folding, constant arithmetic, a
  constant ``Gather``) run as session operations once, at import, and their
  results become uploaded parameters, so the captured program holds none of
  that work;
* the graph inputs become placeholder tensors and every remaining node is
  one or a few session calls, the operations ``Session.capture`` records;
  an integer input read only as ``Gather`` indices (a language model's
  token ids) is an ids tensor (``Session.ids``), and its ``Gather`` is an
  ``embedding`` (a 2-D table on axis 0) or a ``take``, so the program takes
  new ids at each run.

Python here maps names, shapes, axes and attribute numbers; it never reads
or computes a floating-point element of the model. Every value computation,
one-element constants included, is a session operation on the device.
"""
# cpu-route: model graph lowered on the host. The lowering is the import
# step before the runtime: it walks nodes, axes and integer shape constants
# (description) and emits one session call per operation; the only element
# bytes it touches are handed to the device upload unchanged.
from __future__ import annotations

from dataclasses import dataclass

from mojolearn.importers._graph import FLOAT_DTYPES, UnsupportedModel, _slice_range, int_values, resolve

# the largest tensor rank of a session (autograd/rankn.mojo RN_MAX_RANK)
MAX_RANK = 8

# the op table mojolearn.importers.supported_ops() returns (importers/_table.py)
from mojolearn.importers._table import SUPPORTED_OPS  # noqa: F401

# the inputs of an op that are description (static integers folded at import), not data
_DESC_INPUTS = {
    "Reshape": (1,), "Expand": (1,), "Squeeze": (1,), "Unsqueeze": (1,), "Slice": (1, 2, 3, 4),
    "Split": (1,), "ReduceSum": (1,), "ReduceMean": (1,), "ReduceMax": (1,), "ReduceMin": (1,),
    "ReduceL2": (1,), "Pad": (1, 3), "ConstantOfShape": (0,), "Trilu": (1,), "Gather": (1,),
    "Dropout": (1, 2), "Shape": (0,), "Size": (0,), "Range": (0, 1, 2),
}

# the session methods of the one-input elementwise ops
_UNARY = {
    "Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Exp": "exp", "Log": "log", "Sqrt": "sqrt",
    "Reciprocal": "reciprocal", "Erf": "erf", "Softplus": "softplus", "Abs": "abs", "Floor": "floor",
    "Ceil": "ceil", "Round": "round", "Sin": "sin", "Cos": "cos", "Sign": "sign", "HardSwish": "hard_swish",
    "Not": "logical_not",
}
_BINARY = {
    "Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Equal": "equal", "Greater": "greater",
    "Less": "less", "GreaterOrEqual": "greater_equal", "LessOrEqual": "less_equal", "And": "logical_and",
    "Or": "logical_or",
}

# autograd/weight_import.mojo WI_* (python/mojolearn/_weight_import.py): the record of one source tensor
_WI_DTYPE = {"f32": (11, 4), "f16": (23, 2), "f64": (12, 8), "bf16": (4096, 2)}
_WI_MAX_RANK = 4
_WI_LAYOUT_AS_IS, _WI_LAYOUT_TRANSPOSE = 2, 3
_WI_STRIDE_C = 0
# NumPy views of an initializer's raw little-endian bytes (no copy, no conversion)
_NP_VIEW = {"f32": "<f4", "f16": "<f2", "f64": "<f8", "i64": "<i8", "i32": "<i4", "bool": "?"}


class _Refuse(Exception):
    pass


def _prod(dims):
    n = 1
    for d in dims:  # cpu-route: model graph lowered on the host (one per axis)
        n *= d
    return n


def session_shape(shape):
    """The shape a session holds for a model value: a session tensor has at
    least two axes, so a scalar is ``(1, 1)`` and a ``[n]`` vector ``(1, n)``
    (NumPy broadcasting pads on the left the same way)."""
    shape = tuple(shape)
    return shape if len(shape) >= 2 else (1,) * (2 - len(shape)) + shape


def _norm(axis, rank):
    if not -max(rank, 1) <= axis < max(rank, 1):
        raise _Refuse(f"axis {axis} is out of range for rank {rank}")
    return axis + rank if axis < 0 else axis


def _sax(axis, rank):
    """The session axis of model axis ``axis`` of a rank-``rank`` value."""
    return _norm(axis, rank) + (2 - rank if rank < 2 else 0)


@dataclass
class Lowered:
    """The session handles of a lowered graph: the placeholder of each graph
    input, the handle of each graph output, the uploaded parameters, and the
    names of the outputs held as int32 words (ArgMax positions)."""
    input_handles: tuple
    output_handles: tuple
    parameter_handles: tuple
    int32_outputs: frozenset


# ------------------------------------------------------------ the analysis --

class _Graph:
    """What the check and the lowering know about a resolved graph: which
    values are constant (computable from initializers alone), which nodes
    the outputs need, and who consumes each value."""

    def __init__(self, g, res):
        self.g, self.res = g, res
        self.inputs = tuple(i for i in g.inputs if i[0] not in g.initializers)  # cpu-route: model graph lowered on the host (graph inputs)
        const = set(g.initializers) | set(res.folded)
        for n in g.nodes:  # cpu-route: model graph lowered on the host (graph nodes)
            if all(not i or i in const for i in n.inputs):  # cpu-route: model graph lowered on the host (node inputs)
                const.update(o for o in n.outputs if o)  # cpu-route: model graph lowered on the host (node outputs)
        self.const = const
        # the nodes the outputs need, back from the outputs; a folded value needs no node
        want = {o[0] for o in g.outputs}  # cpu-route: model graph lowered on the host (graph outputs)
        needed = []
        for n in reversed(g.nodes):  # cpu-route: model graph lowered on the host (graph nodes, last to first)
            if not any(o and o in want and o not in res.folded for o in n.outputs):  # cpu-route: model graph lowered on the host (node outputs)
                continue
            needed.append(n)
            desc = _DESC_INPUTS.get(n.op, ())
            for k, i in enumerate(n.inputs):  # cpu-route: model graph lowered on the host (node inputs)
                if i and k not in desc:
                    want.add(i)
        self.needed = tuple(reversed(needed))
        self.want = want
        self.consumers = {}
        for n in self.needed:  # cpu-route: model graph lowered on the host (needed nodes)
            for i in n.inputs:  # cpu-route: model graph lowered on the host (node inputs)
                if i:
                    self.consumers.setdefault(i, []).append(n)
        self.output_names = frozenset(o[0] for o in g.outputs)  # cpu-route: model graph lowered on the host (graph outputs)
        # integer model inputs that needed nodes read only as Gather indices: held as
        # ids tensors (int32 words), the program's ids inputs
        index_inputs = set()
        for name, dt, _ in self.inputs:  # cpu-route: model graph lowered on the host (graph inputs)
            readers = self.consumers.get(name, ())
            if dt in ("i64", "i32") and readers and name not in self.output_names and all(
                    c.op == "Gather" and len(c.inputs) > 1 and c.inputs[1] == name and c.inputs[0] != name
                    for c in readers):  # cpu-route: model graph lowered on the host (the input's readers)
                index_inputs.add(name)
        self.index_inputs = frozenset(index_inputs)

    def shape(self, name):
        return self.res.shapes[name]

    def has_shape(self, name):
        return name in self.res.shapes

    def ints(self, name):
        """The folded integers of ``name`` (a shape, axes, sizes, indices) or None."""
        if not name:
            return None
        v = self.res.folded.get(name)
        if v is not None:
            return tuple(v)
        t = self.g.initializers.get(name)
        if t is not None and t.dtype in ("i64", "i32", "bool"):
            return tuple(int_values(t))
        return None

    def used(self, name):
        """True when a needed node or a graph output reads value ``name``."""
        return bool(name) and (name in self.consumers or name in self.output_names)


# ------------------------------------------------------------ subset checks --

def _c_none(G, n):
    return []


def _c_extra_outputs(G, n, what):
    return [f"the {what} output is not supported"] if any(G.used(o) for o in n.outputs[1:]) else []  # cpu-route: model graph lowered on the host (node outputs)


def _c_div(G, n):
    if n.op == "Div" and G.res.dtypes.get(n.outputs[0]) in ("i64", "i32"):
        return ["integer division of tensor values is not supported"]
    return []


def _c_cast(G, n):
    to = G.res.dtypes.get(n.outputs[0])
    src = G.res.dtypes.get(n.inputs[0])
    if src in FLOAT_DTYPES and to not in FLOAT_DTYPES:
        return [f"a cast of floating-point values to {to} is not supported"]
    if src in ("i64", "i32") and to == "bool":
        return ["a cast of integer values to bool is not supported"]
    return []


def _c_gelu(G, n):
    a = n.attrs.get("approximate", "none")
    return [] if a in ("none", "tanh") else [f"approximate={a!r} (only 'none' and 'tanh' are supported)"]


def _c_layer_norm(G, n):
    out = _c_extra_outputs(G, n, "Mean or InvStdDev")
    s = G.shape(n.inputs[0])
    r = len(s)
    ax = n.attrs.get("axis", -1)
    if r == 0 or _norm(ax, r) != r - 1:
        out.append(f"axis={ax} normalizes more than the last axis of {tuple(s)} (only the last axis is supported)")
        return out
    for i in n.inputs[1:3]:  # cpu-route: model graph lowered on the host (scale and bias)
        if i and _prod(G.shape(i)) != s[-1]:
            out.append(f"'{i}' of shape {tuple(G.shape(i))} is not one value per feature ({s[-1]})")
    if not n.attrs.get("epsilon", 1e-5) > 0:
        out.append("epsilon must be positive")
    return out


def _c_rms_norm(G, n):
    out = _c_extra_outputs(G, n, "InvStdDev")
    s = G.shape(n.inputs[0])
    r = len(s)
    ax = n.attrs.get("axis", -1)
    if r == 0 or _norm(ax, r) != r - 1:
        out.append(f"axis={ax} normalizes more than the last axis of {tuple(s)} (only the last axis is supported)")
    elif _prod(G.shape(n.inputs[1])) != s[-1]:
        out.append(f"scale of shape {tuple(G.shape(n.inputs[1]))} is not one value per feature ({s[-1]})")
    return out


def _c_batch_norm(G, n):
    out = _c_extra_outputs(G, n, "running statistics")
    if n.attrs.get("training_mode", 0):
        out.append("training_mode=1 (only inference is supported)")
    s = G.shape(n.inputs[0])
    if len(s) < 2:
        out.append(f"input {tuple(s)} has no channel axis")
        return out
    for i in n.inputs[1:5]:  # cpu-route: model graph lowered on the host (the four statistics)
        if not i or i not in G.const:
            out.append(f"'{i}' is computed from the model's inputs (constant statistics are supported)")
        elif _prod(G.shape(i)) != s[1]:
            out.append(f"'{i}' of shape {tuple(G.shape(i))} is not one value per channel ({s[1]})")
    return out


def _c_instance_norm(G, n):
    s = G.shape(n.inputs[0])
    if len(s) < 3:
        return [f"input {tuple(s)} needs rank 3 or more"]
    return [f"'{i}' of shape {tuple(G.shape(i))} is not one value per channel ({s[1]})"
            for i in n.inputs[1:3] if i and _prod(G.shape(i)) != s[1]]  # cpu-route: model graph lowered on the host (scale and bias)


def _c_group_norm(G, n):
    s = G.shape(n.inputs[0])
    if len(s) < 3:
        return [f"input {tuple(s)} needs rank 3 or more"]
    gr = n.attrs.get("num_groups")
    if not isinstance(gr, int) or gr < 1 or s[1] % gr:
        return [f"num_groups={gr} does not divide {s[1]} channels"]
    return [f"'{i}' of shape {tuple(G.shape(i))} is neither per channel ({s[1]}) nor per group ({gr})"
            for i in n.inputs[1:3] if i and _prod(G.shape(i)) not in (s[1], gr)]  # cpu-route: model graph lowered on the host (scale and bias)


def _c_split(G, n):
    s = G.shape(n.inputs[0])
    ax = _norm(n.attrs.get("axis", 0), len(s))
    bad = [o for o in n.outputs if o and G.has_shape(o) and G.shape(o)[ax] == 0]  # cpu-route: model graph lowered on the host (node outputs)
    return ["an empty piece is not supported"] if bad else []


def _c_slice(G, n):
    steps = G.ints(n.inputs[4]) if len(n.inputs) > 4 and n.inputs[4] else None
    out = []
    if steps is not None and any(st != 1 for st in steps):  # cpu-route: model graph lowered on the host (one per sliced axis)
        out.append(f"steps {tuple(steps)} (only steps of 1 are supported)")
    if 0 in G.shape(n.outputs[0]):
        out.append("an empty slice is not supported")
    return out


def _c_gather(G, n):
    if G.ints(n.inputs[1]) is None and n.inputs[1] not in G.index_inputs:
        return ["the indices are computed from the model's inputs (constant indices, or an integer "
                "model input read only as indices, are supported)"]
    return []


def _c_reduce(G, n):
    axes = n.inputs[1] if len(n.inputs) > 1 else ""
    if axes and G.ints(axes) is None:
        return ["the axes are computed from the model's inputs (static axes are supported)"]
    return []


def _c_argmax(G, n):
    return [] if not n.attrs.get("select_last_index", 0) else ["select_last_index=1 is not supported"]


def _window(G, n):
    """(kernel, strides, pads, dilations) of a Conv or pool node, the pads
    explicit: ONNX order (every spatial axis's count before, then every
    axis's count after). ``auto_pad`` resolves from the static shapes as
    ONNX defines it: VALID is no padding; SAME_UPPER and SAME_LOWER pad each
    axis to ``ceil(size / stride)`` outputs, the odd count at the end
    (UPPER) or at the start (LOWER). Shape arithmetic only."""
    s = tuple(G.shape(n.inputs[0]))
    spatial = len(s) - 2
    if "kernel_shape" in n.attrs:
        k = tuple(n.attrs["kernel_shape"])
    else:
        k = tuple(G.shape(n.inputs[1])[2:]) if n.op == "Conv" else ()
    st = tuple(n.attrs.get("strides", (1,) * spatial))
    pads = tuple(n.attrs.get("pads", (0,) * (2 * spatial)))
    dil = tuple(n.attrs.get("dilations", (1,) * spatial))
    auto = n.attrs.get("auto_pad", "NOTSET")
    if auto == "VALID":
        pads = (0,) * (2 * spatial)
    elif auto in ("SAME_UPPER", "SAME_LOWER") and len(k) == spatial and len(st) == spatial:
        before, after = [], []
        for i in range(spatial):  # cpu-route: model graph lowered on the host (one per spatial axis)
            d = s[2 + i]
            total = max((-(-d // st[i]) - 1) * st[i] + (k[i] - 1) * dil[i] + 1 - d, 0)
            first = total // 2 if auto == "SAME_UPPER" else total - total // 2
            before.append(first)
            after.append(total - first)
        pads = tuple(before + after)
    return k, st, pads, dil


def _pad_counts(pads):
    """Session.pad counts of a rank-4 (N, C, H, W) value from 2-D ONNX
    window pads (top, left, bottom, right)."""
    return [0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]]


def _c_conv(G, n):
    s = G.shape(n.inputs[0])
    out = []
    if len(s) not in (3, 4):
        return [f"input {tuple(s)}: only 1-D and 2-D convolutions are supported"]
    _, st, pads, dil = _window(G, n)
    group = n.attrs.get("group", 1)
    oc = G.shape(n.inputs[1])[0]
    if group < 1 or s[1] % group or oc % group:
        out.append(f"group={group} does not divide {s[1]} input and {oc} output channels")
    if any(d != 1 for d in dil):  # cpu-route: model graph lowered on the host (one per spatial axis)
        out.append(f"dilations={dil} (only 1 is supported)")
    if len(set(st)) != 1:
        out.append(f"strides={st} (only equal strides are supported)")
    if any(p < 0 for p in pads):  # cpu-route: model graph lowered on the host (one per side)
        out.append(f"pads={pads} (negative pads are not supported)")
    return out


def _c_pool(G, n):
    out = _c_extra_outputs(G, n, "Indices")
    s = G.shape(n.inputs[0])
    if len(s) != 4:
        return out + [f"input {tuple(s)}: only 2-D pooling is supported"]
    k, st, pads, dil = _window(G, n)
    if len(set(k)) != 1:
        out.append(f"kernel_shape={k} (only a square kernel is supported)")
    if len(set(st)) != 1:
        out.append(f"strides={st} (only equal strides are supported)")
    if k and any(p < 0 or p >= k[0] for p in pads):  # cpu-route: model graph lowered on the host (one per side)
        # a pad of the kernel's size or more makes windows of padding only
        out.append(f"pads={pads} (each pad must be between 0 and kernel_shape - 1)")
    if any(d != 1 for d in dil):  # cpu-route: model graph lowered on the host (one per spatial axis)
        out.append(f"dilations={dil} (only 1 is supported)")
    if n.attrs.get("storage_order", 0):
        out.append("storage_order=1 is not supported")
    if n.attrs.get("ceil_mode", 0) and k and st:
        # ceil_mode changes nothing when every window fits exactly
        floor = tuple((s[2 + i] + pads[i] + pads[2 + i] - k[i]) // st[i] + 1 for i in range(2))  # cpu-route: model graph lowered on the host (two spatial axes)
        if tuple(G.shape(n.outputs[0])[2:]) != floor:
            out.append("ceil_mode=1 with a partial last window is not supported")
    return out


def _c_pad(G, n):
    out = []
    mode = n.attrs.get("mode", "constant")
    if mode != "constant":
        out.append(f"mode={mode!r} (only 'constant' is supported)")
    if len(n.inputs) > 1 and n.inputs[1] and G.ints(n.inputs[1]) is None:
        out.append("the pads are computed from the model's inputs (static pads are supported)")
    return out


def _c_trilu(G, n):
    s = G.shape(n.inputs[0])
    if len(s) < 2:
        return [f"input {tuple(s)} needs rank 2 or more"]
    if len(n.inputs) > 1 and n.inputs[1] and G.ints(n.inputs[1]) is None:
        return ["k is computed from the model's inputs (a static k is supported)"]
    return []


def _c_dropout(G, n):
    out = _c_extra_outputs(G, n, "mask")
    if len(n.inputs) > 2 and n.inputs[2]:
        tm = G.ints(n.inputs[2])
        if tm is None or any(tm):
            out.append("training_mode is not false (only inference is supported)")
    return out


# ------------------------------------------------------------ the lowering --

class _Lowerer:
    """One pass over the needed nodes, in order, through session ``s``."""

    def __init__(self, G, s):
        self.G, self.s = G, s
        self.h = {}            # value name -> session handle
        self.owned = []        # handles made with Session.tensor (uploads and persisted constants)
        self._owned = set()
        self.cur_const = False
        self.int32_outputs = set()

    # ---------------------------------------------------------- values
    def _own(self, h):
        self.owned.append(h)
        self._owned.add(h)
        return h

    def persist(self, h):
        """Constant ``h`` as an uploaded tensor: a computed constant's words
        are read back and uploaded unchanged (a byte copy at import), so the
        captured program reads it as a parameter instead of computing it at
        every run."""
        if h in self._owned:
            return h
        return self._own(self.s.tensor(self.s.numpy(h)))

    def k(self, h):
        """``h``, persisted when it was computed from constants only."""
        return self.persist(h)

    def get(self, name):
        """The handle of value ``name``; a constant read by a node that depends
        on the inputs is a parameter (persisted when it was computed)."""
        h = self.h.get(name)
        if h is None:
            h = self._upload_value(name)
            self.h[name] = h
        if not self.cur_const and name in self.G.const and h not in self._owned:
            h = self.persist(h)
            self.h[name] = h
        return h

    def _upload_value(self, name):
        t = self.G.g.initializers.get(name)
        if t is not None:
            return self.upload(t)
        v = self.G.ints(name)
        if v is not None:
            return self.upload_ints(v, self.G.shape(name))
        raise RuntimeError(f"the lowering reached value '{name}' before the node that computes it")

    def upload(self, t, transpose=False):
        """Initializer ``t`` as a session tensor of its session shape, cast on
        the device; ``transpose=True`` loads a 2-D ``(a, b)`` initializer as
        ``(b, a)`` through the weight import's layout code."""
        from mojolearn._optional_numpy import require_numpy
        np = require_numpy("mojolearn.importers")
        shape = session_shape(t.shape)
        if t.dtype == "bf16" or transpose:
            # the weight import reads the raw words and widens or transposes them on the device
            target = (t.shape[1], t.shape[0]) if transpose else shape
            h = self._own(self.s.tensor(np.zeros(target, dtype=np.float32)))
            src = tuple(t.shape) if t.shape else (1,)
            code, size = _WI_DTYPE[t.dtype]
            buf = np.frombuffer(t.data, dtype=np.uint8)
            rec = ([int(h), 0, _WI_LAYOUT_TRANSPOSE if transpose else _WI_LAYOUT_AS_IS, int(buf.ctypes.data), 0,
                    code, size, len(src)] + list(src) + [0] * (_WI_MAX_RANK - len(src))
                   + [_WI_STRIDE_C] + [0] * _WI_MAX_RANK)
            self.s._b.ops_load_weights(self.s._sid(), rec, [t.name])
            del buf
            return h
        # a view of the raw bytes, cast to float32 on the device by the upload
        a = np.frombuffer(t.data, dtype=np.dtype(_NP_VIEW[t.dtype]), count=t.numel).reshape(shape)
        return self._own(self.s.tensor(a))

    def upload_ints(self, values, shape):
        """Folded description integers (a shape read as data) as a tensor."""
        from mojolearn._optional_numpy import require_numpy
        np = require_numpy("mojolearn.importers")
        return self._own(self.s.tensor(np.asarray(values, dtype=np.int64).reshape(session_shape(shape))))

    def const_scalar(self, value):
        """A ``(1, 1)`` constant of an attribute number (epsilon, a fill), as a parameter."""
        return self.persist(self.s.full((1, 1), float(value)))

    def fit(self, h, name):
        """``h`` reshaped to value ``name``'s session shape when the call left
        it in another shape of the same size (a kept reduced axis, a rank-1
        value held as a row)."""
        want = session_shape(self.G.shape(name))
        have = tuple(self.s.shape(h))
        if have == want:
            return h
        if _prod(have) != _prod(want):
            raise RuntimeError(f"the lowering made {have} for '{name}', the graph says {want}")
        return self.s.reshape(h, want)

    def rank(self, name):
        return len(self.G.shape(name))

    # ---------------------------------------------------------- the pass
    def run(self):
        G, s = self.G, self.s
        inputs = []
        for name, _, _ in G.inputs:  # cpu-route: model graph lowered on the host (graph inputs)
            if name in G.index_inputs:
                # an ids placeholder (id 0 is in range of every table); capture names it an
                # ids input, and it is freed after the capture with the other unread uploads
                from mojolearn._optional_numpy import require_numpy
                np = require_numpy("mojolearn.importers")
                h = self._own(s.ids(np.zeros(session_shape(G.shape(name)), dtype=np.int32)))
            else:
                h = s.full(session_shape(G.shape(name)), 0.0)  # a placeholder; capture names it an input
            self.h[name] = h
            inputs.append(h)
        for n in G.needed:  # cpu-route: model graph lowered on the host (graph nodes)
            self.cur_const = all(not i or i in G.const for i in n.inputs)  # cpu-route: model graph lowered on the host (node inputs)
            outs = _LOWER[n.op](self, n)
            for o, h in zip(n.outputs, outs):  # cpu-route: model graph lowered on the host (node outputs)
                if o and h is not None:
                    self.h[o] = h
        self.cur_const = False
        outputs = [self.get(name) for name, _, _ in G.g.outputs]  # cpu-route: model graph lowered on the host (graph outputs)
        return inputs, outputs


def _x(L, n, k=0):
    return L.get(n.inputs[k])


def _l_unary(L, n):
    return [getattr(L.s, _UNARY[n.op])(_x(L, n))]


def _l_identity(L, n):
    return [_x(L, n)] + [None] * (len(n.outputs) - 1)


def _l_neg(L, n):
    return [L.s.scale(_x(L, n), -1.0)]


def _l_leaky(L, n):
    return [L.s.leaky_relu(_x(L, n), alpha=n.attrs.get("alpha", 0.01))]


def _l_elu(L, n):
    return [L.s.elu(_x(L, n), alpha=n.attrs.get("alpha", 1.0))]


def _l_hard_sigmoid(L, n):
    return [L.s.hard_sigmoid(_x(L, n), alpha=n.attrs.get("alpha", 0.2), beta=n.attrs.get("beta", 0.5))]


def _l_gelu(L, n):
    return [L.s.gelu(_x(L, n), approximate=n.attrs.get("approximate", "none"))]


def _l_binary(L, n):
    return [getattr(L.s, _BINARY[n.op])(_x(L, n, 0), _x(L, n, 1))]


def _l_minmax(L, n):
    f = L.s.maximum if n.op == "Max" else L.s.minimum
    ins = [i for i in n.inputs if i]  # cpu-route: model graph lowered on the host (the operands)
    h = L.get(ins[0])
    for i in ins[1:]:  # cpu-route: model graph lowered on the host (the operands of one node)
        h = f(h, L.get(i))
    return [L.fit(h, n.outputs[0])]


def _l_pow(L, n):
    e = L.G.ints(n.inputs[1])
    if e is not None and len(e) == 1 and L.rank(n.inputs[1]) <= L.rank(n.inputs[0]):
        # an integer exponent is description: the number goes in the call
        return [L.s.pow_scalar(_x(L, n), float(e[0]))]
    return [L.s.pow(_x(L, n, 0), _x(L, n, 1))]


def _l_where(L, n):
    return [L.s.where(_x(L, n, 0), _x(L, n, 1), _x(L, n, 2))]


def _l_clip(L, n):
    h = _x(L, n)
    if len(n.inputs) > 1 and n.inputs[1]:
        h = L.s.maximum(h, _x(L, n, 1))
    if len(n.inputs) > 2 and n.inputs[2]:
        h = L.s.minimum(h, _x(L, n, 2))
    return [L.fit(h, n.outputs[0])]


def _l_matmul(L, n):
    a, b = n.inputs[0], n.inputs[1]
    ha, hb = L.get(a), L.get(b)
    if L.rank(b) == 1:
        # a [k] right operand is a (k, 1) column (its session row is (1, k))
        col = L.s.reshape(hb, (L.G.shape(b)[0], 1))
        hb = L.k(col) if b in L.G.const and not L.cur_const else col
    return [L.fit(L.s.matmul(ha, hb), n.outputs[0])]


def _l_gemm(L, n):
    G, s = L.G, L.s
    a, b = n.inputs[0], n.inputs[1]
    c = n.inputs[2] if len(n.inputs) > 2 else ""
    alpha, beta = n.attrs.get("alpha", 1.0), n.attrs.get("beta", 1.0)
    ha = L.get(a)
    if n.attrs.get("transA", 0):
        ha = s.transpose(ha)
    b_const = b in G.const and not L.cur_const
    if n.attrs.get("transB", 0):
        t = G.g.initializers.get(b)
        if t is not None and len(t.shape) == 2 and t.dtype in FLOAT_DTYPES:
            # torch's [out, in] linear weight, transposed once on the device by the weight import
            key = (b, "T")
            hb = L.h.get(key)
            if hb is None:
                hb = L.upload(t, transpose=True)
                L.h[key] = hb
        else:
            hb = s.transpose(L.get(b))
            hb = L.k(hb) if b_const else hb
    else:
        hb = L.get(b)
    if alpha != 1.0:
        if b_const:
            hb = L.k(s.scale(hb, alpha))
        else:
            ha = s.scale(ha, alpha)
    hc = None
    if c:
        hc = L.get(c)
        if beta != 1.0:
            hc = s.scale(hc, beta)
            hc = L.k(hc) if c in G.const and not L.cur_const else hc
    nn = G.shape(n.outputs[0])[1]
    if hc is not None and _prod(G.shape(c)) == nn and len(G.shape(c)) <= 2 and (len(G.shape(c)) < 2 or G.shape(c)[0] == 1):
        if tuple(s.shape(hc)) != (1, nn):
            hc = s.reshape(hc, (1, nn))
            hc = L.k(hc) if c in G.const and not L.cur_const else hc
        return [s.linear(ha, hb, hc)]
    y = s.matmul(ha, hb)
    if hc is not None:
        y = s.add(y, hc)
    return [y]


def _l_softmax(L, n):
    x = n.inputs[0]
    r = L.rank(x)
    ax = _sax(n.attrs.get("axis", -1), r)
    srank = len(session_shape(L.G.shape(x)))
    f = L.s.softmax if n.op == "Softmax" else L.s.log_softmax
    return [f(L.get(x), axis=-1 if ax == srank - 1 else ax)]


def _row_param(L, name, d):
    """A per-feature constant as a ``(1, d)`` row."""
    h = L.get(name)
    if tuple(L.s.shape(h)) != (1, d):
        h = L.s.reshape(h, (1, d))
        h = L.k(h) if name in L.G.const and not L.cur_const else h
    return h


def _l_layer_norm(L, n):
    x = n.inputs[0]
    d = L.G.shape(x)[-1]
    w = _row_param(L, n.inputs[1], d)
    if len(n.inputs) > 2 and n.inputs[2]:
        b = _row_param(L, n.inputs[2], d)
    else:
        b = L.persist(L.s.full((1, d), 0.0))
    y = L.s.layer_norm(L.get(x), w, b, eps=n.attrs.get("epsilon", 1e-5))
    return [y] + [None] * (len(n.outputs) - 1)


def _l_rms_norm(L, n):
    s = L.s
    x = n.inputs[0]
    sh = session_shape(L.G.shape(x))
    hx = L.get(x)
    ms = s.mean(s.mul(hx, hx), axis=len(sh) - 1, keepdims=True)  # glue: a mean on the device through the session
    den = s.sqrt(s.add(ms, L.const_scalar(n.attrs.get("epsilon", 1e-5))))
    y = s.mul(s.div(hx, den), _row_param(L, n.inputs[1], sh[-1]))
    return [y] + [None] * (len(n.outputs) - 1)


def _channel_param(L, h, c, rank):
    """A per-channel constant ``h`` as ``(1, c, 1, ...)`` for a rank-``rank`` input."""
    want = (1, c) + (1,) * (rank - 2)
    if tuple(L.s.shape(h)) != want:
        h = L.s.reshape(h, want)
    return h


def _l_batch_norm(L, n):
    s, G = L.s, L.G
    sh = G.shape(n.inputs[0])
    c, r = sh[1], len(sh)
    sc, bias, mean, var = (L.get(i) for i in n.inputs[1:5])  # cpu-route: model graph lowered on the host (the four statistics)
    # the prologue: scale / sqrt(var + eps) and bias - mean * that, once at import
    fs = s.div(sc, s.sqrt(s.add(var, L.const_scalar(n.attrs.get("epsilon", 1e-5)))))
    ft = s.sub(bias, s.mul(mean, fs))
    fs = L.k(_channel_param(L, fs, c, r))
    ft = L.k(_channel_param(L, ft, c, r))
    return [s.add(s.mul(L.get(n.inputs[0]), fs), ft)] + [None] * (len(n.outputs) - 1)


def _normalize_last(L, h, axes, eps):
    """``(h - mean) / sqrt(var + eps)`` over ``axes`` (session axes), the
    statistics kept with length-1 axes."""
    s = L.s
    m = s.mean(h, axis=axes, keepdims=True)  # glue: a mean on the device through the session
    d = s.sub(h, m)
    v = s.mean(s.mul(d, d), axis=axes, keepdims=True)  # glue: a mean on the device through the session
    return s.div(d, s.sqrt(s.add(v, L.const_scalar(eps))))


def _const_channel(L, name, c, rank):
    h = _channel_param(L, L.get(name), c, rank)
    return L.k(h) if name in L.G.const and not L.cur_const else h


def _l_instance_norm(L, n):
    s = L.s
    sh = L.G.shape(n.inputs[0])
    c, r = sh[1], len(sh)
    y = _normalize_last(L, L.get(n.inputs[0]), tuple(range(2, r)), n.attrs.get("epsilon", 1e-5))
    y = s.mul(y, _const_channel(L, n.inputs[1], c, r))
    return [s.add(y, _const_channel(L, n.inputs[2], c, r))]


def _l_group_norm(L, n):
    s = L.s
    sh = L.G.shape(n.inputs[0])
    nb, c, r = sh[0], sh[1], len(sh)
    gr = n.attrs["num_groups"]
    xr = s.reshape(L.get(n.inputs[0]), (nb, gr, _prod(sh) // (nb * gr)))
    y = _normalize_last(L, xr, 2, n.attrs.get("epsilon", 1e-5))
    per_group = _prod(L.G.shape(n.inputs[1])) == gr and gr != c
    if per_group:
        y = s.mul(y, _const_channel(L, n.inputs[1], gr, 3))
        y = s.add(y, _const_channel(L, n.inputs[2], gr, 3))
        return [s.reshape(y, sh)]
    y = s.reshape(y, sh)
    y = s.mul(y, _const_channel(L, n.inputs[1], c, r))
    return [s.add(y, _const_channel(L, n.inputs[2], c, r))]


def _l_transpose(L, n):
    x = n.inputs[0]
    r = L.rank(x)
    if r < 2:
        return [_x(L, n)]
    perm = n.attrs.get("perm")
    perm = tuple(range(r - 1, -1, -1)) if perm is None else tuple(_norm(p, r) for p in perm)  # cpu-route: model graph lowered on the host (one per axis)
    if perm == tuple(range(r)):
        return [_x(L, n)]
    return [L.s.permute(_x(L, n), perm)]


def _l_reshape(L, n):
    # Reshape, Flatten, Squeeze, Unsqueeze: the resolved shape, row-major
    return [L.fit(_x(L, n), n.outputs[0])]


def _l_expand(L, n):
    h = _x(L, n)
    want = session_shape(L.G.shape(n.outputs[0]))
    if tuple(L.s.shape(h)) == want:
        return [h]
    return [L.s.expand(h, want)]


def _l_concat(L, n):
    ins = [i for i in n.inputs if i]  # cpu-route: model graph lowered on the host (the operands)
    if len(ins) == 1:
        return [L.get(ins[0])]
    r = L.rank(ins[0])
    return [L.s.concat([L.get(i) for i in ins], axis=_sax(n.attrs.get("axis", 0), r))]  # cpu-route: model graph lowered on the host (the operands)


def _narrow(L, h, start, length, dim, axis):
    """Positions ``start .. start + length`` of session axis ``axis`` (a view)."""
    if start == 0 and length == dim:
        return h
    sizes = [v for v in (start, length, dim - start - length) if v]  # cpu-route: model graph lowered on the host (three piece lengths)
    return L.s.split(h, sizes, axis=axis)[1 if start else 0]


def _l_split(L, n):
    x = n.inputs[0]
    sh = L.G.shape(x)
    r = len(sh)
    ax = _norm(n.attrs.get("axis", 0), r)
    sizes = [L.G.shape(o)[ax] for o in n.outputs]  # cpu-route: model graph lowered on the host (one per piece)
    if len(sizes) == 1:
        return [_x(L, n)]
    return L.s.split(_x(L, n), sizes, axis=_sax(ax, r))


def _l_slice(L, n):
    G = L.G
    x = n.inputs[0]
    sh = list(G.shape(x))
    r = len(sh)
    starts, ends = G.ints(n.inputs[1]), G.ints(n.inputs[2])
    axes = G.ints(n.inputs[3]) if len(n.inputs) > 3 and n.inputs[3] else tuple(range(len(starts)))
    h = _x(L, n)
    for st, en, ax in zip(starts, ends, axes):  # cpu-route: model graph lowered on the host (the sliced axes)
        a = _norm(ax, r)
        b, ln = _slice_range(st, en, 1, sh[a])
        h = _narrow(L, h, b, ln, sh[a], _sax(a, r))
        sh[a] = ln
    return [L.fit(h, n.outputs[0])]


def _l_gather(L, n):
    from mojolearn._optional_numpy import require_numpy
    G, s = L.G, L.s
    x = n.inputs[0]
    sh = G.shape(x)
    r = len(sh)
    ax = _norm(n.attrs.get("axis", 0), r)
    dim = sh[ax]
    if n.inputs[1] in G.index_inputs:
        # ids from a model input: an embedding row lookup for a 2-D table on axis 0, else a take
        hx, hi = L.get(x), L.get(n.inputs[1])
        y = s.embedding(hx, hi) if r == 2 and ax == 0 else s.take(hx, hi, axis=_sax(ax, r))
        return [L.fit(y, n.outputs[0])]
    idx = [i + dim if i < 0 else i for i in G.ints(n.inputs[1])]  # cpu-route: model graph lowered on the host (constant index integers)
    if L.cur_const:
        # a constant table: one take at import (the program reads the result as a parameter)
        np = require_numpy("mojolearn.importers")
        return [L.fit(s.take(L.get(x), np.asarray(idx, dtype=np.int64), axis=_sax(ax, r)), n.outputs[0])]
    # runs of consecutive positions, each one view of the input, joined in order
    runs = []
    for i in idx:  # cpu-route: model graph lowered on the host (constant index integers)
        if runs and runs[-1][0] + runs[-1][1] == i:
            runs[-1][1] += 1
        else:
            runs.append([i, 1])
    h = L.get(x)
    pieces = [_narrow(L, h, b, ln, dim, _sax(ax, r)) for b, ln in runs]  # cpu-route: model graph lowered on the host (index runs)
    out = pieces[0] if len(pieces) == 1 else s.concat(pieces, axis=_sax(ax, r))
    return [L.fit(out, n.outputs[0])]


def _reduce_axes(L, n):
    x = n.inputs[0]
    r = L.rank(x)
    if len(n.inputs) > 1 and n.inputs[1]:
        axes = L.G.ints(n.inputs[1])
    else:
        axes = n.attrs.get("axes")
    if not axes:
        if n.attrs.get("noop_with_empty_axes", 0):
            return None
        axes = tuple(range(r))
    if r == 0:
        return None
    return tuple(_sax(a, r) for a in axes)  # cpu-route: model graph lowered on the host (one per axis)


def _l_reduce(L, n):
    s = L.s
    axes = _reduce_axes(L, n)
    h = _x(L, n)
    if axes is None:
        return [L.fit(h, n.outputs[0])]
    if n.op == "ReduceSum":
        y = s.sum(h, axis=axes, keepdims=True)  # glue: a sum on the device through the session
    elif n.op == "ReduceMean":
        y = s.mean(h, axis=axes, keepdims=True)  # glue: a mean on the device through the session
    elif n.op == "ReduceMax":
        y = s.max(h, axis=axes, keepdims=True)
    elif n.op == "ReduceMin":
        y = s.scale(s.max(s.scale(h, -1.0), axis=axes, keepdims=True), -1.0)
    else:  # ReduceL2
        y = s.sqrt(s.sum(s.mul(h, h), axis=axes, keepdims=True))  # glue: a sum on the device through the session
    return [L.fit(y, n.outputs[0])]


def _l_argmax(L, n):
    x = n.inputs[0]
    o = n.outputs[0]
    # positions a later node reads stay float values; a returned output is int32 words
    as_int = o in L.G.output_names and o not in L.G.consumers
    if as_int:
        L.int32_outputs.add(o)
    y = L.s.argmax(_x(L, n), axis=_sax(n.attrs.get("axis", 0), L.rank(x)), keepdims=True, int32=as_int)  # glue: argmax on the device through the session
    return [L.fit(y, o)]


def _l_global_pool(L, n):
    s = L.s
    r = L.rank(n.inputs[0])
    axes = tuple(range(2, r))
    if n.op == "GlobalAveragePool":
        return [s.mean(_x(L, n), axis=axes, keepdims=True)]  # glue: a mean on the device through the session
    return [s.max(_x(L, n), axis=axes, keepdims=True)]


def _l_conv(L, n):
    s, G = L.s, L.G
    x, w = n.inputs[0], n.inputs[1]
    b = n.inputs[2] if len(n.inputs) > 2 and n.inputs[2] else ""
    sh = G.shape(x)
    _, st, pads, _ = _window(G, n)
    stride = st[0]
    group = n.attrs.get("group", 1)
    hb = None
    if b:
        hb = _row_param(L, b, G.shape(w)[0])
    if len(sh) == 4:
        if len(set(pads)) == 1:
            return [s.conv2d(_x(L, n), L.get(w), hb, stride=stride, padding=pads[0], groups=group)]
        # unequal pads (explicit, or SAME with an odd total): the zeros
        # placed first, then the unpadded convolution reads the same words
        hx = s.pad(_x(L, n), _pad_counts(pads))
        return [s.conv2d(hx, L.get(w), hb, stride=stride, padding=0, groups=group)]
    # 1-D: (N, C, L) as (N, C, 1, L), padded on the length axis only
    nb, c, ln = sh
    ws = G.shape(w)
    hx = s.reshape(_x(L, n), (nb, c, 1, ln))
    if pads[0] or pads[1]:
        hx = s.pad(hx, [0, 0, 0, pads[0], 0, 0, 0, pads[1]])
    hw = s.reshape(L.get(w), (ws[0], ws[1], 1, ws[2]))
    hw = L.k(hw) if w in G.const and not L.cur_const else hw
    return [L.fit(s.conv2d(hx, hw, hb, stride=stride, padding=0, groups=group), n.outputs[0])]


#: the fill of a padded MaxPool's border: the lowest finite float32, below every real value
_MAXPOOL_PAD = -3.4028234663852886e38


def _l_pool(L, n):
    s = L.s
    k, st, pads, _ = _window(L.G, n)
    kk, stride = k[0], st[0]
    rest = [None] * (len(n.outputs) - 1)
    if not any(pads):  # cpu-route: model graph lowered on the host (four pads)
        f = s.max_pool2d if n.op == "MaxPool" else s.avg_pool2d
        return [f(_x(L, n), kk, stride=stride)] + rest
    counts = _pad_counts(pads)
    if n.op == "MaxPool":
        # ONNX pads a max pool with values that never win: -inf, and every
        # window holds an input value (each pad below the kernel size)
        return [s.max_pool2d(s.pad(_x(L, n), counts, float("-inf")), kk, stride=stride)] + rest
    if n.attrs.get("count_include_pad", 0):
        # the padding counts in every window: zeros placed, then the mean of
        # kernel_shape^2 values per window
        return [s.avg_pool2d(s.pad(_x(L, n), counts, 0.0), kk, stride=stride)] + rest
    # count_include_pad=0: each window's sum over the zero-padded input (a
    # depthwise convolution with ones: the taps added in (kh, kw) order)
    # divided by the window's number of input values, a count map from the
    # same convolution over a ones image padded with zeros (exact small
    # integers, built on the device once, a parameter of the program)
    sh = L.G.shape(n.inputs[0])
    c, h, w = sh[1], sh[2], sh[3]
    ones_w = s.full((c, 1, kk, kk), 1.0)
    ones_w = ones_w if L.cur_const else L.persist(ones_w)
    total = s.conv2d(s.pad(_x(L, n), counts, 0.0), ones_w, None, stride=stride, padding=0, groups=c)
    image = s.pad(s.full((1, 1, h, w), 1.0), counts, 0.0)
    count = s.conv2d(image, s.full((1, 1, kk, kk), 1.0), None, stride=stride, padding=0)
    count = count if L.cur_const else L.persist(count)
    return [s.div(total, count)] + rest


def _l_pad(L, n):
    s, G = L.s, L.G
    x = n.inputs[0]
    sh = G.shape(x)
    r = len(sh)
    pads = G.ints(n.inputs[1]) if len(n.inputs) > 1 and n.inputs[1] else tuple(n.attrs.get("pads", ()))
    axes = (G.ints(n.inputs[3]) if len(n.inputs) > 3 and n.inputs[3] else tuple(range(r)))
    sr = len(session_shape(sh))
    before, after = [0] * sr, [0] * sr
    for j, a in enumerate(axes):  # cpu-route: model graph lowered on the host (one per padded axis)
        before[_sax(a, r)] = pads[j]
        after[_sax(a, r)] = pads[len(axes) + j]
    counts = before + after
    y = s.pad(_x(L, n), counts, 0.0)
    if len(n.inputs) > 2 and n.inputs[2]:
        # a fill value from the model: 1.0 marks the padded positions, where() puts the value there
        mark = s.pad(s.full(session_shape(sh), 0.0), counts, 1.0)
        mark = mark if L.cur_const else L.persist(mark)
        y = s.where(mark, _x(L, n, 2), y)
    return [L.fit(y, n.outputs[0])]


def _l_trilu(L, n):
    from mojolearn._optional_numpy import require_numpy
    np = require_numpy("mojolearn.importers")
    s = L.s
    sh = L.G.shape(n.inputs[0])
    rows, cols = sh[-2], sh[-1]
    k = L.G.ints(n.inputs[1])[0] if len(n.inputs) > 1 and n.inputs[1] else 0
    upper = n.attrs.get("upper", 1)
    # the kept positions (column - row >= k above the diagonal, <= k below), a
    # description of positions, never of element values
    keep = [[1 if ((j - i >= k) if upper else (j - i <= k)) else 0 for j in range(cols)]  # cpu-route: model graph lowered on the host (matrix positions)
            for i in range(rows)]  # cpu-route: model graph lowered on the host (matrix positions)
    mask = L._own(s.tensor(np.asarray(keep, dtype=np.uint8)))
    zero = L.const_scalar(0.0)
    return [s.where(mask, _x(L, n), zero)]


def _l_constant_of_shape(L, n):
    s = L.s
    want = session_shape(L.G.shape(n.outputs[0]))
    t = n.attrs.get("value")
    if t is None:
        return [s.full(want, 0.0)]
    return [s.expand(L.upload(t), want)]


def _l_constant(L, n):
    return [L.upload(n.attrs["value"])]


def _l_folded(L, n):
    # Shape, Size, Range: every output is folded at import
    return [None] * len(n.outputs)


# op -> (subset check, lowering)
_TABLE = {}
for _op in _UNARY:  # cpu-route: model graph lowered on the host (op names)
    _TABLE[_op] = (_c_none, _l_unary)
for _op in _BINARY:  # cpu-route: model graph lowered on the host (op names)
    _TABLE[_op] = (_c_div, _l_binary)
for _op in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceL2"):  # cpu-route: model graph lowered on the host (op names)
    _TABLE[_op] = (_c_reduce, _l_reduce)
for _op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze"):  # cpu-route: model graph lowered on the host (op names)
    _TABLE[_op] = (_c_none, _l_reshape)
for _op in ("Shape", "Size"):  # cpu-route: model graph lowered on the host (op names)
    _TABLE[_op] = (_c_none, _l_folded)
_TABLE.update({
    "Identity": (_c_none, _l_identity), "Dropout": (_c_dropout, _l_identity), "Cast": (_c_cast, _l_identity),
    "Neg": (_c_none, _l_neg), "LeakyRelu": (_c_none, _l_leaky), "Elu": (_c_none, _l_elu),
    "HardSigmoid": (_c_none, _l_hard_sigmoid), "Gelu": (_c_gelu, _l_gelu), "Pow": (_c_none, _l_pow),
    "Max": (_c_none, _l_minmax), "Min": (_c_none, _l_minmax), "Where": (_c_none, _l_where),
    "Clip": (_c_none, _l_clip), "MatMul": (_c_none, _l_matmul), "Gemm": (_c_none, _l_gemm),
    "Softmax": (_c_none, _l_softmax), "LogSoftmax": (_c_none, _l_softmax),
    "LayerNormalization": (_c_layer_norm, _l_layer_norm), "RMSNormalization": (_c_rms_norm, _l_rms_norm),
    "BatchNormalization": (_c_batch_norm, _l_batch_norm),
    "InstanceNormalization": (_c_instance_norm, _l_instance_norm),
    "GroupNormalization": (_c_group_norm, _l_group_norm), "Transpose": (_c_none, _l_transpose),
    "Expand": (_c_none, _l_expand), "Concat": (_c_none, _l_concat), "Split": (_c_split, _l_split),
    "Slice": (_c_slice, _l_slice), "Gather": (_c_gather, _l_gather), "ArgMax": (_c_argmax, _l_argmax),
    "GlobalAveragePool": (_c_none, _l_global_pool), "GlobalMaxPool": (_c_none, _l_global_pool),
    "Conv": (_c_conv, _l_conv), "MaxPool": (_c_pool, _l_pool), "AveragePool": (_c_pool, _l_pool),
    "Pad": (_c_pad, _l_pad), "Trilu": (_c_trilu, _l_trilu), "ConstantOfShape": (_c_none, _l_constant_of_shape),
    "Constant": (_c_none, _l_constant),
})
_LOWER = {op: low for op, (_, low) in _TABLE.items()}  # cpu-route: model graph lowered on the host (op names)


# ------------------------------------------------------------ entry points --

def check(g, shapes=None):
    """Resolve ``g``'s shapes and match every node the outputs need against
    the table. Returns the analysis; raises :class:`UnsupportedModel` with
    every problem, in graph order, before anything is uploaded."""
    res = resolve(g, shapes)
    G = _Graph(g, res)
    order = {n.name: k for k, n in enumerate(g.nodes)}  # cpu-route: model graph lowered on the host (graph nodes)
    input_names = {i[0] for i in g.inputs}  # cpu-route: model graph lowered on the host (graph inputs)
    found = []
    for name, op, why in res.problems:  # cpu-route: model graph lowered on the host (shape problems)
        if op is not None:
            key = order.get(name, len(order))
        else:
            key = -1 if name in input_names or name not in G.output_names else len(order)
        found.append((key, name, op, why))
    flagged = {p[0] for p in res.problems if p[1] is not None}  # cpu-route: model graph lowered on the host (shape problems)
    if not G.inputs:
        found.append((-1, g.source, None, "the model has no inputs"))
    for name, dt, dims in G.inputs:  # cpu-route: model graph lowered on the host (graph inputs)
        if dt == "bf16":
            found.append((-1, name, None, "bfloat16 inputs are not supported (pass the input as float32)"))
        if len(dims) > MAX_RANK:
            found.append((-1, name, None, f"rank {len(dims)} (at most {MAX_RANK} is supported)"))
    for n in G.needed:  # cpu-route: model graph lowered on the host (needed nodes)
        k = order[n.name]
        if n.name in flagged:
            continue
        entry = _TABLE.get(n.op) if n.domain in ("", "ai.onnx") else None
        if entry is None and n.domain not in ("", "ai.onnx"):
            found.append((k, n.name, n.op, f"operations of domain '{n.domain}' are not supported"))
            continue
        if entry is None:
            found.append((k, n.name, n.op, "no kernel; see mojolearn.importers.supported_ops()"))
            continue
        if any(i and not G.has_shape(i) for i in n.inputs) or any(o and not G.has_shape(o) for o in n.outputs):  # cpu-route: model graph lowered on the host (node values)
            continue  # an upstream problem is already listed
        why = []
        for o in n.outputs:  # cpu-route: model graph lowered on the host (node outputs)
            if o and G.used(o):
                if len(G.shape(o)) > MAX_RANK:
                    why.append(f"'{o}' has rank {len(G.shape(o))} (at most {MAX_RANK} is supported)")
                elif 0 in G.shape(o):
                    why.append(f"'{o}' is empty")
        for i in n.inputs:  # cpu-route: model graph lowered on the host (node inputs)
            t = g.initializers.get(i)
            if t is not None and t.numel == 0 and i in G.want:
                why.append(f"'{i}' is an empty tensor")
            elif t is not None and t.dtype == "bf16" and len(t.shape) > _WI_MAX_RANK:
                why.append(f"'{i}' is a bfloat16 tensor of more than {_WI_MAX_RANK} dimensions")
        try:
            why += entry[0](G, n)
        except _Refuse as e:
            why.append(str(e))
        found += [(k, n.name, n.op, w) for w in why]  # cpu-route: model graph lowered on the host (one per problem)
    if found:
        found.sort(key=lambda p: p[0])
        raise UnsupportedModel(g.source, [(name, op, why) for _, name, op, why in found])  # cpu-route: model graph lowered on the host (one per problem)
    return G


def lower(g, value_shapes, session):
    """Replay graph ``g`` through inference session ``session``: uploads,
    the import-time prologue and one session call per remaining operation.
    ``value_shapes`` is the analysis ``check`` returned (or the ``shapes``
    argument of the import, checked here). Returns :class:`Lowered`."""
    G = value_shapes if isinstance(value_shapes, _Graph) else check(g, value_shapes)
    L = _Lowerer(G, session)
    inputs, outputs = L.run()
    return Lowered(tuple(inputs), tuple(outputs), tuple(L.owned), frozenset(L.int32_outputs))


# ------------------------------------------------------------ the oracle import --

def import_model(source, *, shapes=None, numeric_mode=None):
    """The former importers/_program.py import_model: read, check, one
    inference Session, lower (one Session call per operation), capture."""
    from mojolearn.importers._graph import GraphIR
    from mojolearn.importers._program import ImportedProgram, _dtype_name
    from mojolearn.ops import Session
    if isinstance(source, GraphIR):
        g = source
    elif hasattr(source, "graph_signature") and hasattr(source, "graph_module"):
        from mojolearn.importers import torch_export
        g = torch_export.read(source)
    else:
        from mojolearn.importers import onnx as onnx_reader
        g = onnx_reader.read(source)
    G = check(g, shapes)
    s = Session(record=False, numeric_mode=numeric_mode)
    try:
        low = lower(g, G, s)
        info = s._b.ops_capture(s._sid(), Session._handles(low.input_handles), Session._handles(low.output_handles))
        inputs = [(name, G.shape(name), _dtype_name(dt)) for name, dt, _ in G.inputs]
        outputs = [(name, G.shape(name), "int32" if name in low.int32_outputs else "float32")
                   for name, _, _ in g.outputs]
        prog = ImportedProgram(s, info, inputs, outputs)
        keep = set(prog.params)
        s.reset()
        for h in low.parameter_handles:
            if h not in keep:
                s.free(h)
    except BaseException:
        s.close()
        raise
    return prog
