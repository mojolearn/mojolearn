# SPDX-License-Identifier: Apache-2.0
"""Import an exported model into a native program."""
# Owner clarification 2026-10-27: an API break is allowed to keep graph capture
# outside product execution. Import accepts exported artifacts, never a Python
# model, exporter, serialization callback or implicit changed-weight refresh.
from __future__ import annotations

from ..ops import Program, Session
from ._graph import GraphIR, UnsupportedModel

__all__ = ["import_model", "ImportedProgram"]


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
        _IMPORT_NATIVE(self, session, info, inputs, outputs, _IMPORT_CONTEXT)

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
        return self._run_native(self, arrays, named)

    def run_torch(self, *tensors):
        """Run this imported artifact with PyTorch input and output tensors.

        Parameters
        ----------
        tensors : torch.Tensor
            One tensor per declared input, in input order and with its fixed
            shape. Tensor metadata is refreshed once per execution call.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Outputs on the first input's device. CUDA and ROCm tensors on
            GPU zero use device buffers; other inputs use host buffers.

        Examples
        --------
        >>> result = program.run_torch(torch.ones(8, 3))  # doctest: +SKIP
        """
        return self._run_torch_native(self, tensors, None, False)


def import_model(source, *, shapes=None, numeric_mode=None):
    """Import an exported ONNX model for native inference.

    In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    source : str, bytes, pathlib.Path or GraphIR
        An ONNX file (opset 13 or later), or a graph returned by
        ``mojolearn.importers.onnx.read``. Export Python models before calling
        this function. The imported program owns a snapshot of the artifact's
        weights; import an updated artifact to replace them.
    shapes : dict, optional
        Fixed dimensions for inputs whose sizes the artifact leaves open,
        by input name, for example ``{"input": (8, 3)}``.
    numeric_mode : {'identical', 'fast'} or None, default None
        None uses the library default. ``'fast'`` is Apple only.

    Returns
    -------
    ImportedProgram
        Native program with input/output descriptors and ``run``/``save``
        methods. Float64 weights are rounded to float32; float16 and bfloat16
        weights are widened exactly.

    Raises
    ------
    UnsupportedModel
        The artifact contains unsupported operations, attributes or shapes.
    TypeError
        The source is not an exported artifact or a native reader graph.

    Examples
    --------
    >>> import mojolearn
    >>> prog = mojolearn.import_model("classifier.onnx", shapes={"input": (8, 3)})  # doctest: +SKIP
    >>> logits = prog.run([[1.0, 2.0, 3.0]] * 8)  # doctest: +SKIP
    """
    return _IMPORT_NATIVE(source, None, shapes, numeric_mode, _IMPORT_CONTEXT)


from . import _native_import_bridge  # noqa: E402
_IMPORT_NATIVE, _IMPORT_CONTEXT = _native_import_bridge.connect(
    Session, Program, ImportedProgram, GraphIR, UnsupportedModel)
