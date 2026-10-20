# SPDX-License-Identifier: Apache-2.0
"""``import_model``: a model file or object as a captured program
(docs/plans/MODEL_IMPORT.md section 6)."""
# Engineering note: read (onnx.py or torch_export.py) -> check (the refusal
# pass, before any upload) -> one inference Session -> lower (uploads, the
# import-time prologue, one session call per operation) -> Session.capture.
# The program keeps its session; run() only reshapes the caller's arrays to
# the session's shapes (a rank-1 model value is held as a (1, n) row) and
# back, as views.
from __future__ import annotations

import os

from ..ops import Program, Session
from ._graph import GraphIR
from ._lower import check, lower, session_shape

__all__ = ["import_model", "ImportedProgram"]


def _read(source):
    if isinstance(source, GraphIR):
        return source
    if isinstance(source, (str, bytes, os.PathLike)):
        from . import onnx as onnx_reader
        return onnx_reader.read(os.fsdecode(source))  # cpu-route: model file parsed on the host
    if hasattr(source, "graph_signature") and hasattr(source, "graph_module"):
        try:
            from . import torch_export
        except ImportError:
            raise TypeError("importing a torch.export.ExportedProgram is not available in this install") from None
        return torch_export.read(source)
    raise TypeError("source must be the path of an ONNX file or a torch.export.ExportedProgram, "
                    f"got {type(source).__name__}")


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
        bool, integer or float dtype is used as float32. Returns one array
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
    g = _read(source)
    # every problem is listed before anything reaches the device
    G = check(g, shapes)
    s = Session(record=False, numeric_mode=numeric_mode)
    try:
        low = lower(g, G, s)
        # Session.capture's call, kept as an ImportedProgram
        info = s._b.ops_capture(s._sid(), Session._handles(low.input_handles), Session._handles(low.output_handles))
        inputs = [(name, G.shape(name), _dtype_name(dt)) for name, dt, _ in G.inputs]  # glue: one entry per model input
        outputs = [(name, G.shape(name), "int32" if name in low.int32_outputs else "float32")
                   for name, _, _ in g.outputs]  # glue: one entry per model output
        prog = ImportedProgram(s, info, inputs, outputs)
        # the computed tensors (placeholders, prologue intermediates) go; the
        # uploads the program does not read go too
        keep = set(prog.params)
        s.reset()
        for h in low.parameter_handles:  # glue: the import's uploaded tensors
            if h not in keep:
                s.free(h)
    except BaseException:
        s.close()
        raise
    return prog
