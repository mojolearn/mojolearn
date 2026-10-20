# SPDX-License-Identifier: Apache-2.0
"""Read an ONNX model file into the import graph.

:func:`read` reads the file with a reader of mojolearn's own (no ``onnx``
package needed), checks that the model is opset 13 or later in the default
ONNX domain with tensor types the import supports, and returns the graph
with its nodes in topological order. The weights are not copied: each
initializer's bytes are a view into the file (or into its external data
file) until they are uploaded to the device.
"""
# Engineering note: the file is read, decoded and checked in Mojo
# (autograd/import_reader.mojo, autograd/import_resolve.mojo) through the
# x_sequence binding; this module only maps the files for the weight views
# and wraps the reader's tables in the GraphIR classes. The former Python
# reader is the test oracle in tests/_onnx_reader_oracle.py.
from __future__ import annotations

import ctypes
import mmap
import os
import weakref

from ._graph import GraphIR, Node, Resolved, TensorInfo, UnsupportedModel

OPSET_MIN = 13

# the reader's word layout (autograd/import_reader.mojo docstring)
_MAGIC, _VERSION = 0x4F4E5852, 1
_OK, _NOT_A_MODEL, _CANNOT_OPEN, _BAD_TEXT, _BAD_VALUE, _PROBLEMS = 0, 1, 2, 3, 4, 5
_DTYPES = ("f32", "f16", "bf16", "f64", "i64", "i32", "bool")
_BUF_MODEL, _BUF_BLOB, _BUF_EXTERNAL = 0, 1, 2
_A_FLOAT, _A_INT, _A_STRING, _A_TENSOR, _A_FLOATS, _A_INTS, _A_STRINGS, _A_TENSORS = 1, 2, 3, 4, 6, 7, 8, 9


def _binding():
    from .. import _backend
    b = _backend.binding("_mojolearn_x_sequence")
    if not hasattr(b, "onnx_read"):  # cpu-route: model file parsing in Mojo (the reader entry)
        raise RuntimeError("mojolearn: this install's x_sequence binding has no ONNX reader (onnx_read); "
                           "rebuild the x_sequence bindings")
    return b


def _map_file(path):
    """A read-only memoryview of the whole file (memory-mapped; the view keeps the map alive)."""
    with open(path, "rb") as f:
        try:
            return memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        except ValueError:  # an empty file cannot be mapped
            return memoryview(f.read())


class _Words:
    """A cursor over the reader's words."""

    __slots__ = ("w", "i")

    def __init__(self, words):
        self.w, self.i = words, 0

    def take(self):
        v = self.w[self.i]
        self.i += 1
        return v


def _header(words, strings, path):
    """The status word after the magic and version; raises for a file the reader refused outright."""
    if len(words) < 3 or words[0] != _MAGIC or words[1] != _VERSION:
        raise RuntimeError("mojolearn: the x_sequence binding's ONNX reader has another word layout; "
                           "rebuild the x_sequence bindings")
    status = words[2]
    if status == _NOT_A_MODEL:
        raise ValueError(f"{path} is not an ONNX model: {strings[words[3]]}")
    if status == _CANNOT_OPEN:
        if not os.path.exists(path):
            raise FileNotFoundError(2, "No such file or directory", path)
        raise OSError(f"{path}: {strings[words[3]]}")
    if status in (_BAD_TEXT, _BAD_VALUE):
        raise ValueError(f"{path}: {strings[words[3]]}")
    return status


def _problems(c, strings):
    out = []
    for _ in range(c.take()):  # glue: one record per problem the reader found
        name, op, reason = c.take(), c.take(), c.take()
        out.append((strings[name], None if op < 0 else strings[op], strings[reason]))
    return out


def _value(c, strings):
    name, dtype, rank = strings[c.take()], _DTYPES[c.take()], c.take()
    dims = []
    for _ in range(rank):  # glue: one record per axis of a graph value
        is_str, v = c.take(), c.take()
        dims.append(strings[v] if is_str else v)
    return name, dtype, tuple(dims)


def _tensor(c, strings, views):
    name, dtype, rank = strings[c.take()], _DTYPES[c.take()], c.take()
    shape = tuple([c.take() for _ in range(rank)])  # glue: one word per axis of a tensor record
    buf, start, end = c.take(), c.take(), c.take()
    return TensorInfo(name, dtype, shape, views[buf][start:end])


def _attr(c, strings, floats, views):
    kind = c.take()
    if kind == _A_FLOAT:
        return floats[c.take()]
    if kind == _A_INT:
        return c.take()
    if kind == _A_STRING:
        return strings[c.take()]
    if kind == _A_TENSOR:
        return _tensor(c, strings, views)
    if kind == _A_FLOATS:
        first, end = c.take(), c.take()
        return tuple(floats[first:end])
    count = c.take()
    if kind == _A_INTS:
        return tuple([c.take() for _ in range(count)])  # glue: one word per entry of an ints attribute record
    if kind == _A_STRINGS:
        return tuple([strings[c.take()] for _ in range(count)])  # glue: one word per entry of a strings attribute record
    return tuple([_tensor(c, strings, views) for _ in range(count)])  # glue: one record per tensors attribute entry


def _views(b, h, c, strings):
    """One memoryview per buffer the tables point into: the model file and its
    external data files mapped here, the reader's own bytes copied once."""
    views = []
    for _ in range(c.take()):  # glue: one record per file of the model
        kind, path, nbytes = c.take(), c.take(), c.take()
        if kind == _BUF_BLOB:
            blob = bytearray(nbytes)
            if nbytes:
                b.onnx_copy_blob(h, ctypes.addressof((ctypes.c_char * nbytes).from_buffer(blob)))
            views.append(memoryview(blob))
            continue
        v = _map_file(strings[path])
        if len(v) != nbytes:
            raise ValueError(f"{strings[path]} changed while it was read")
        views.append(v)
    return views


class _Native:
    """The Mojo reader's handle of one graph, released with its GraphIR."""

    def __init__(self, b, h):
        self.b, self.h = b, h

    def resolve(self, shapes=None):
        """importers._graph.resolve of this graph, by the Mojo rules."""
        given = dict(shapes or {})
        names, flat = [], []
        for name, dims in given.items():  # glue: the shapes argument, one entry per named input
            dims = tuple([int(d) for d in dims])  # glue: one argument check per axis of a given shape
            names.append(str(name))
            flat.append(len(dims))
            flat.extend(dims)
        words, strings = self.b.onnx_resolve(self.h, names, flat)  # cpu-route: model file parsing in Mojo
        _header(words, strings, "the model graph")
        c = _Words(words)
        c.i = 3
        shapes_out, dtypes = {}, {}
        for _ in range(c.take()):  # glue: one record per resolved value
            name, dt, rank = strings[c.take()], _DTYPES[c.take()], c.take()
            shapes_out[name] = tuple([c.take() for _ in range(rank)])  # glue: one word per axis of a resolved shape
            dtypes[name] = dt
        folded = {}
        for _ in range(c.take()):  # glue: one record per folded value
            name, count = strings[c.take()], c.take()
            folded[name] = tuple([c.take() for _ in range(count)])  # glue: the folded words the reader returned
        no_rule = tuple([strings[c.take()] for _ in range(c.take())])  # glue: one word per node without a rule
        return Resolved(shapes_out, dtypes, folded, no_rule, tuple(_problems(c, strings)))


def read(path):
    """Read the ONNX model at ``path`` into an import graph.

    Returns the model's graph: its inputs and outputs with their types and
    shapes, its nodes in topological order and whose initializers are views of the file's
    bytes (``Constant`` nodes become initializers). Raises
    :class:`UnsupportedModel` listing every problem found at once: an opset
    older than 13, a node outside the default ONNX domain, an initializer,
    input or output of an unsupported type, a subgraph attribute; raises
    ``ValueError`` when the file is not an ONNX model.
    """
    path = os.fspath(path)
    b = _binding()
    h = int(b.onnx_read(os.fsdecode(path)))  # cpu-route: model file parsing in Mojo
    try:
        words, strings, floats = b.onnx_info(h)  # cpu-route: model file parsing in Mojo
        status = _header(words, strings, path)
        c = _Words(words)
        c.i = 3
        if status == _PROBLEMS:
            raise UnsupportedModel(path, _problems(c, strings))
        opset = c.take()
        views = _views(b, h, c, strings)
        inputs = tuple([_value(c, strings) for _ in range(c.take())])  # glue: one record per graph input
        outputs = tuple([_value(c, strings) for _ in range(c.take())])  # glue: one record per graph output
        nodes = []
        for _ in range(c.take()):  # glue: one record per graph node
            op, domain, name = strings[c.take()], strings[c.take()], strings[c.take()]
            ins = tuple([strings[c.take()] for _ in range(c.take())])  # glue: one word per node input name
            outs = tuple([strings[c.take()] for _ in range(c.take())])  # glue: one word per node output name
            attrs = {}
            for _ in range(c.take()):  # glue: one record per node attribute
                key = strings[c.take()]
                attrs[key] = _attr(c, strings, floats, views)
            nodes.append(Node(op, domain, name, ins, outs, attrs))
        inits = {}
        for _ in range(c.take()):  # glue: one record per initializer
            t = _tensor(c, strings, views)
            inits[t.name] = t
    except BaseException:
        b.onnx_close(h)
        raise
    g = GraphIR(opset=opset, inputs=inputs, outputs=outputs, nodes=tuple(nodes), initializers=inits, source=path)
    g._native = _Native(b, h)
    weakref.finalize(g, b.onnx_close, h)
    return g
