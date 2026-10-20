# SPDX-License-Identifier: Apache-2.0
"""``import_model``: a model file or object as a captured program
(docs/plans/MODEL_IMPORT.md section 6)."""
# Engineering note: read (onnx.py, or torch_export.py whose GraphIR goes to
# Mojo as words through onnx_from_words) -> one onnx_import call into an
# inference Session (autograd/import_lower.mojo: shape resolution, the
# refusal pass before any upload, the uploads, the import-time prologue, one
# session call per operation, the capture, the release of unread uploads).
# Python passes names, shapes and buffer addresses only. The program keeps
# its session; run() only reshapes the caller's arrays to the session's
# shapes (a rank-1 model value is held as a (1, n) row) and back, as views.
from __future__ import annotations

import os

from ..ops import Program, Session
from ._graph import GraphIR, TensorInfo, UnsupportedModel
from ._table import session_shape

__all__ = ["import_model", "ImportedProgram"]


def _read(source, binding):
    if isinstance(source, GraphIR):
        return source
    if isinstance(source, (str, bytes, os.PathLike)):
        from . import onnx as onnx_reader
        return onnx_reader.read(os.fsdecode(source), binding=binding)  # cpu-route: model file parsed in Mojo
    if hasattr(source, "graph_signature") and hasattr(source, "graph_module"):
        try:
            from . import torch_export
        except ImportError:
            raise TypeError("importing a torch.export.ExportedProgram is not available in this install") from None
        return torch_export.read(source)
    raise TypeError("source must be the path of an ONNX file or a torch.export.ExportedProgram, "
                    f"got {type(source).__name__}")


# autograd/import_reader.mojo's word layout (importers/onnx.py reads it back)
_MAGIC, _VERSION = 0x4F4E5852, 1
_DTYPE_CODE = {"f32": 0, "f16": 1, "bf16": 2, "f64": 3, "i64": 4, "i32": 5, "bool": 6}
_A_FLOAT, _A_INT, _A_STRING, _A_TENSOR, _A_FLOATS, _A_INTS, _A_STRINGS, _A_TENSORS = 1, 2, 3, 4, 6, 7, 8, 9
_BUF_EXTERNAL = 2
# autograd/import_lower.mojo onnx_import statuses
_IMPORT_OK, _IMPORT_BAD_VALUE, _IMPORT_PROBLEMS = 0, 4, 5


def _bytes_of(data):
    """A NumPy uint8 view of a buffer (no copy), or None when it is empty."""
    from .._optional_numpy import require_numpy
    np = require_numpy("mojolearn.importers")
    raw = memoryview(data).cast("B") if data is not None else memoryview(b"")
    return np.frombuffer(raw, dtype=np.uint8) if raw.nbytes else None


def _addr(a):
    return 0 if a is None else int(a.ctypes.data)


def _graph_words(g):
    """GraphIR ``g`` as onnx_info's words, strings and floats, plus one
    uint8 view per tensor's bytes (each its own buffer): the graph handed to
    Mojo's onnx_from_words. Description only."""
    # cpu-route: model graph description handed to Mojo (names, dims, attributes; tensor bytes pass by address)
    strings, index, floats, bufs, body = [""], {"": 0}, [], [], []

    def sid(x):
        k = index.get(x)
        if k is None:
            k = index[x] = len(strings)
            strings.append(x)
        return k

    def tensor(t):
        a = _bytes_of(t.data)
        bufs.append(a)
        body.extend([sid(t.name), _DTYPE_CODE[t.dtype], len(t.shape)])
        body.extend(int(d) for d in t.shape)  # cpu-route: model graph description handed to Mojo (one per axis)
        body.extend([len(bufs) - 1, 0, 0 if a is None else a.size])

    def value(v):
        name, dtype, dims = v
        body.extend([sid(name), _DTYPE_CODE[dtype], len(dims)])
        for d in dims:  # cpu-route: model graph description handed to Mojo (one per axis)
            body.extend([1, sid(d)] if isinstance(d, str) else [0, int(d)])

    def attr(v):
        if isinstance(v, TensorInfo):
            body.append(_A_TENSOR)
            tensor(v)
        elif isinstance(v, str):
            body.extend([_A_STRING, sid(v)])
        elif isinstance(v, float):
            body.extend([_A_FLOAT, len(floats)])
            floats.append(v)
        elif isinstance(v, int):
            body.extend([_A_INT, int(v)])
        else:
            items = tuple(v)
            if items and all(isinstance(x, TensorInfo) for x in items):  # cpu-route: model graph description handed to Mojo (attribute list)
                body.extend([_A_TENSORS, len(items)])
                for x in items:  # cpu-route: model graph description handed to Mojo (attribute list)
                    tensor(x)
            elif items and all(isinstance(x, str) for x in items):  # cpu-route: model graph description handed to Mojo (attribute list)
                body.extend([_A_STRINGS, len(items)] + [sid(x) for x in items])  # cpu-route: model graph description handed to Mojo (attribute list)
            elif any(isinstance(x, float) for x in items):  # cpu-route: model graph description handed to Mojo (attribute list)
                body.extend([_A_FLOATS, len(floats), len(floats) + len(items)])
                floats.extend(float(x) for x in items)  # cpu-route: model graph description handed to Mojo (attribute list)
            else:
                body.extend([_A_INTS, len(items)] + [int(x) for x in items])  # cpu-route: model graph description handed to Mojo (attribute list)

    body.append(len(g.inputs))
    for v in g.inputs:  # cpu-route: model graph description handed to Mojo (graph inputs)
        value(v)
    body.append(len(g.outputs))
    for v in g.outputs:  # cpu-route: model graph description handed to Mojo (graph outputs)
        value(v)
    body.append(len(g.nodes))
    for n in g.nodes:  # cpu-route: model graph description handed to Mojo (graph nodes)
        body.extend([sid(n.op), sid(n.domain), sid(n.name), len(n.inputs)])
        body.extend(sid(i) for i in n.inputs)  # cpu-route: model graph description handed to Mojo (node inputs)
        body.append(len(n.outputs))
        body.extend(sid(o) for o in n.outputs)  # cpu-route: model graph description handed to Mojo (node outputs)
        body.append(len(n.attrs))
        for key, v in n.attrs.items():  # cpu-route: model graph description handed to Mojo (node attributes)
            body.append(sid(key))
            attr(v)
    body.append(len(g.initializers))
    for t in g.initializers.values():  # cpu-route: model graph description handed to Mojo (initializers)
        tensor(t)
    head = [_MAGIC, _VERSION, 0, int(g.opset), len(bufs)]
    for a in bufs:  # cpu-route: model graph description handed to Mojo (one per tensor buffer)
        head.extend([_BUF_EXTERNAL, -1, 0 if a is None else a.size])
    return head + body, strings, floats, bufs


def _shape_words(shapes):
    """``shapes`` as onnx_resolve takes it: the names and (rank, dims...) runs."""
    names, flat = [], []
    for name, dims in dict(shapes or {}).items():  # glue: the shapes argument, one entry per named input
        dims = tuple(int(d) for d in dims)  # glue: at most eight dimensions
        names.append(str(name))
        flat.append(len(dims))
        flat.extend(dims)
    return names, flat


class ImportedProgram(Program):
    """An imported model, run in one call.

    Made by :func:`mojolearn.import_model`. :meth:`run` takes one array per
    model input, of the shape listed in :attr:`inputs`, and returns the
    model's outputs.

    Attributes
    ----------
    inputs : list of tuple
        ``(name, shape, dtype)`` of each model input, in the order
        :meth:`run` takes them.
    outputs : list of tuple
        ``(name, shape, dtype)`` of each output, in the order :meth:`run`
        returns them. An ``argmax`` output has dtype ``'int32'``; the others
        are ``'float32'``.
    """

    def __init__(self, session, info, inputs, outputs):
        super().__init__(session, info)
        self.inputs = list(inputs)
        self.outputs = list(outputs)

    def run(self, *arrays, **named):
        """Run the model on new input arrays.

        Pass one array per input, in the order of :attr:`inputs`, or by
        input name as keywords. Each array must have that input's shape; any
        bool, integer or float dtype is used as float32, except that an
        integer input the model reads as indices (such as a language model's
        token ids) takes an integer array of ids. Returns one array
        for a model with one output, else a tuple in the order of
        :attr:`outputs`.
        """
        from .._optional_numpy import require_numpy
        np = require_numpy("mojolearn.importers")
        if named:
            if arrays:
                raise ValueError("pass the inputs either in order or by name, not both")
            names = [name for name, _, _ in self.inputs]  # glue: the model's input names
            unknown = [k for k in named if k not in names]  # glue: the keyword argument names
            if unknown:
                raise ValueError(f"the model has no input {unknown[0]!r} (inputs: {', '.join(names)})")
            missing = [k for k in names if k not in named]  # glue: the model's input names
            if missing:
                raise ValueError(f"input {missing[0]!r} is missing")
            arrays = tuple(named[k] for k in names)  # glue: the model's input names
        if len(arrays) != len(self.inputs):
            raise ValueError(f"the model takes {len(self.inputs)} inputs, got {len(arrays)}")
        views = []
        for (name, shape, _), x in zip(self.inputs, arrays):  # glue: one view per model input
            a = np.asarray(x)
            if tuple(a.shape) != tuple(shape):
                raise ValueError(f"input {name!r} must have the shape {tuple(shape)}, got {tuple(a.shape)}")
            # a rank-0 or rank-1 input is held as a row: a view, no copy
            views.append(a.reshape(session_shape(shape)) if a.ndim < 2 else a)
        got = self._s.run(self, *views)
        got = got if isinstance(got, tuple) else (got,)
        outs = []
        for (_, shape, dtype), y in zip(self.outputs, got):  # glue: one view per model output
            y = y.view(np.int32) if dtype == "int32" else y
            outs.append(y.reshape(shape))
        return outs[0] if len(outs) == 1 else tuple(outs)


def _dtype_name(ir_dtype):
    return {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "f64": "float64", "i64": "int64",
            "i32": "int32", "bool": "bool"}.get(ir_dtype, ir_dtype)


def import_model(source, *, shapes=None, numeric_mode=None):
    """Import a trained model for inference.

    ``source`` is the path of an ONNX file (opset 13 or later) or a
    ``torch.export.ExportedProgram``. The model's weights are copied to the
    device once and its operations run as one call per
    :meth:`ImportedProgram.run`, with the same results on Apple, NVIDIA and
    AMD GPUs and on the CPU. In `identical` mode, mojolearn returns the same
    bits on Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    source : str, path-like or torch.export.ExportedProgram
        The model.
    shapes : dict, optional
        The shape of each input whose size the model leaves open (such as
        the batch), by input name: ``{"input": (8, 3, 32, 32)}``. The
        imported model then accepts arrays of exactly those shapes.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default (``'identical'``);
        ``'fast'`` is available on Apple GPUs only.

    Returns
    -------
    ImportedProgram
        The model, with :meth:`~ImportedProgram.run`,
        :attr:`~ImportedProgram.inputs`, :attr:`~ImportedProgram.outputs`
        and :meth:`~ImportedProgram.save`. Its float64 weights are rounded
        once to float32; float16 and bfloat16 weights are widened exactly.

    Raises
    ------
    UnsupportedModel
        The model uses operations, attributes or shapes the import does not
        support; the message lists every one, by node name.

    Examples
    --------
    >>> import numpy as np
    >>> import mojolearn
    >>> prog = mojolearn.import_model("classifier.onnx", shapes={"input": (8, 3, 32, 32)})  # doctest: +SKIP
    >>> logits = prog.run(np.zeros((8, 3, 32, 32), dtype=np.float32))  # doctest: +SKIP
    >>> prog.save("classifier.mlprog")  # doctest: +SKIP
    """
    s = Session(record=False, numeric_mode=numeric_mode)
    b = s._b
    words_handle = None
    keep = ()
    try:
        if not hasattr(b, "onnx_import"):
            raise RuntimeError("mojolearn: this install's x_sequence binding has no model import (onnx_import); "
                               "rebuild the x_sequence bindings")
        g = _read(source, b)
        native = getattr(g, "_native", None)
        if native is not None and native.b is b:
            # a graph the Mojo reader holds in this session's module: its buffers by address
            keep = [_bytes_of(v) for v in native.views]  # glue: one view per model file or data buffer
            handle = native.h
        else:
            # a GraphIR built in Python (torch.export), or read by another module: to Mojo as words
            words, strings, floats, keep = _graph_words(g)
            handle = words_handle = int(b.onnx_from_words(words, strings, floats, [_addr(a) for a in keep]))  # glue: buffer addresses
        names, flat = _shape_words(shapes)
        # one call: every problem is listed before anything reaches the device
        res = b.onnx_import(handle, s._sid(), names, flat, [_addr(a) for a in keep], str(g.source))  # glue: buffer addresses
        status = int(res[0])
        if status == _IMPORT_PROBLEMS:
            raise UnsupportedModel(g.source, [(str(name), None if op is None else str(op), str(why))
                                              for name, op, why in res[1]])  # glue: one entry per problem
        if status == _IMPORT_BAD_VALUE:
            raise ValueError(f"the model graph: {res[1]}")
        if status != _IMPORT_OK:
            raise RuntimeError("mojolearn: the x_sequence binding's model import has another result layout; "
                               "rebuild the x_sequence bindings")
        inputs = [(str(name), tuple(int(d) for d in dims), _dtype_name(str(dt)))  # glue: at most eight dimensions
                  for name, dims, dt in res[2]]  # glue: one entry per model input
        outputs = [(str(name), tuple(int(d) for d in dims), "int32" if int(flag) else "float32")  # glue: at most eight dimensions
                   for name, dims, flag in res[3]]  # glue: one entry per model output
        prog = ImportedProgram(s, res[1], inputs, outputs)
    except BaseException:
        s.close()
        raise
    finally:
        if words_handle is not None:
            b.onnx_close(words_handle)
        del keep
    return prog
