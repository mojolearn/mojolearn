# SPDX-License-Identifier: Apache-2.0
"""``import_model``: a model file or object as a captured program
(docs/plans/MODEL_IMPORT.md section 6)."""
# Engineering note: an ONNX path is one onnx_import_file call (the Mojo
# reader opens and reads the file, keeps its bytes, and the lowering runs on
# them; no GraphIR is built in Python). A graph onnx.read returned (its
# handle in the Mojo reader) goes through one onnx_import call into an
# inference Session (autograd/import_lower.mojo: shape resolution, the
# refusal pass before any upload, the uploads, the import-time prologue, one
# session call per operation, the capture, the release of unread uploads).
# Python passes names, shapes and buffer addresses only. The program keeps
# its session; run() only reshapes the caller's arrays to the session's
# shapes (a rank-1 model value is held as a (1, n) row) and back, as views.
from __future__ import annotations

import os

from ..ops import Program, Session
from ._graph import GraphIR, UnsupportedModel, buffer_address, bytes_view
from ._table import session_shape

__all__ = ["import_model", "ImportedProgram"]


_TORCH_TO_ONNX = ('export the torch model to ONNX first, then import the file: '
                  'torch.onnx.export(model, args, "model.onnx"); mojolearn.import_model("model.onnx")')


def _read(source):
    # a graph mojolearn.importers.onnx.read returned: its handle in the Mojo reader
    if isinstance(source, GraphIR) and getattr(source, "_native", None) is not None:
        return source
    # a torch object (an nn.Module, a user subclass of one, or a torch.export program)
    if any(str(c.__module__).split(".", 1)[0] == "torch" for c in type(source).__mro__):  # glue: the argument's class chain
        raise ValueError(f"mojolearn.import_model takes an ONNX file, not a {type(source).__name__}: "
                         + _TORCH_TO_ONNX)
    raise TypeError(f"source must be the path of an ONNX file, got {type(source).__name__}")


# autograd/import_lower.mojo onnx_import statuses
_IMPORT_OK, _IMPORT_BAD_VALUE, _IMPORT_PROBLEMS, _IMPORT_READ_FAILED = 0, 4, 5, 6


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

    ``source`` is the path of an ONNX file (opset 13 or later). The model's weights are copied to the
    device once and its operations run as one call per
    :meth:`ImportedProgram.run`, with the same results on Apple, NVIDIA and
    AMD GPUs and on the CPU. In `identical` mode, mojolearn returns the same
    bits on Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    source : str or path-like
        The model's ONNX file. To import a PyTorch model, export it to ONNX
        first with ``torch.onnx.export(model, args, "model.onnx")`` and pass
        the file.
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
    ValueError
        ``source`` is a PyTorch object; export it to ONNX first.

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
    keep = ()
    try:
        if not hasattr(b, "onnx_import_file"):
            raise RuntimeError("mojolearn: this install's x_sequence binding has no model import (onnx_import_file); "
                               "rebuild the x_sequence bindings")
        names, flat = _shape_words(shapes)
        if isinstance(source, (str, bytes, os.PathLike)):
            # an ONNX file: read, resolved, refused and lowered in one Mojo call
            src = os.fsdecode(os.fspath(source))
            res = b.onnx_import_file(src, s._sid(), names, flat, src)
        else:
            g = _read(source)
            src = g.source
            native = g._native
            if native.b is not b:
                raise ValueError("the graph was read by another binding than this session's; "
                                 "pass the ONNX file's path instead")
            # a graph the Mojo reader holds in this session's module: its buffers by address
            keep = [bytes_view(v) for v in native.views]  # glue: one view per model file or data buffer
            handle = native.h
            # one call: every problem is listed before anything reaches the device
            res = b.onnx_import(handle, s._sid(), names, flat, [buffer_address(a) for a in keep], str(src))  # glue: one address per tensor buffer
        status = int(res[0])
        if status == _IMPORT_READ_FAILED:
            from .onnx import reader_error
            reader_error(int(res[1]), str(res[2]), bool(int(res[3])), src)
        if status == _IMPORT_PROBLEMS:
            raise UnsupportedModel(src, [(str(name), None if op is None else str(op), str(why))
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
        del keep
    return prog
