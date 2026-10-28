# SPDX-License-Identifier: Apache-2.0
"""Read an ONNX model file into the import graph.

:func:`read` reads the file with a reader of mojolearn's own (no ``onnx``
package needed), checks that the model is opset 13 or later in the default
ONNX domain with tensor types the import supports, and returns the graph
with its nodes in topological order. The weights are not copied: each
initializer's bytes are a view into the file's bytes (or its external data
file's) until they are uploaded to the device.
"""
# Engineering note: the file is opened, read, decoded and checked in Mojo
# (autograd/import_reader.mojo, autograd/import_resolve.mojo) through the
# x_sequence binding, and the reader keeps the files' bytes under its handle
# (onnx_buffers gives their addresses); this module opens no file and only
# connects the native reader, which wraps GraphIR metadata and views of
# the reader's bytes. import_model reads a path in one native call
# (onnx_import_file) and never builds a GraphIR. The former Python reader is
# the test oracle in tests/_onnx_reader_oracle.py.
from __future__ import annotations

from ._graph import GraphIR, Node, Resolved, TensorInfo, UnsupportedModel, _Native, _READER_NATIVE, _READER_CONTEXT

OPSET_MIN = 13


def reader_error(status, message, missing, path):
    """Raise the error of a file the reader refused outright (statuses 1–4)."""
    return _READER_NATIVE("reader_error", status, (message, missing, path), _READER_CONTEXT)


def read(path, binding=None):
    """Read the ONNX model at ``path`` into an import graph.

    Returns the model's graph: its inputs and outputs with their types and
    shapes, its nodes in topological order and whose initializers are views of the file's
    bytes (``Constant`` nodes become initializers). Raises
    :class:`UnsupportedModel` listing every problem found at once: an opset
    older than 13, a node outside the default ONNX domain, an initializer,
    input or output of an unsupported type, a subgraph attribute; raises
    ``ValueError`` when the file is not an ONNX model.
    """
    return _READER_NATIVE("reader_read", path, binding, _READER_CONTEXT)
