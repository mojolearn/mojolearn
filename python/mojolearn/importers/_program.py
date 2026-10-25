# SPDX-License-Identifier: Apache-2.0
"""``import_model``: a model file or object as a captured program
(docs/plans/MODEL_IMPORT.md section 6)."""
# Artifact admission, Session setup, file/GraphIR import, status translation,
# output descriptors and program allocation live in import_frontend_pyapi.mojo.
# The public source-kind dispatcher below is still Python runtime debt.
# Torch export executes module.eval()/forward via torch.onnx.export; ModelProto
# serialization also remains here. Neither is an allowed runtime exception.
# ImportedProgram.run retains the existing one-call native executor.
from __future__ import annotations

import os

from ..ops import Program, Session
from ._graph import GraphIR, UnsupportedModel

__all__ = ["import_model", "ImportedProgram"]


# the ONNX opset a PyTorch module is exported at (the import reads opset 13 or later)
_TORCH_OPSET = 17


def _is_torch(source):
    # a torch object (an nn.Module, a user subclass of one, or a torch.export program), found by its
    # class chain so that torch is imported only when the caller passes one of its objects
    return any(str(c.__module__).split(".", 1)[0] == "torch" for c in type(source).__mro__)  # glue: the argument's class chain


def _is_onnx_proto(source):
    # an in-memory onnx.ModelProto: written to a file as it is, then read like any ONNX file
    return (type(source).__name__ == "ModelProto" and str(type(source).__module__).split(".", 1)[0] == "onnx"
            and hasattr(source, "SerializeToString"))


def _read(source):
    # a graph mojolearn.importers.onnx.read returned: its handle in the Mojo reader
    if isinstance(source, GraphIR) and getattr(source, "_native", None) is not None:
        return source
    if _is_torch(source):
        raise ValueError(f"mojolearn.import_model takes a torch.nn.Module or an ONNX file, not a "
                         f"{type(source).__name__}: pass the module itself with example_inputs=, or export it "
                         'with torch.onnx.export(model, args, "model.onnx") and pass the file')
    raise TypeError("source must be the path of an ONNX file, an onnx.ModelProto or a torch.nn.Module, "
                    f"got {type(source).__name__}")


def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("mojolearn.import_model needs PyTorch to import a torch.nn.Module: "
                          "pip install torch") from e
    return torch


def _example_args(torch, shapes, example_inputs):
    """The example inputs torch.onnx.export traces, as a tuple, and their input names."""
    if example_inputs is None:
        if not shapes:
            raise ValueError("importing a torch.nn.Module needs example inputs of the shape you will run: pass "
                             "example_inputs=torch.zeros(8, 3, 32, 32) (a tuple for several inputs), or "
                             "shapes={\"input\": (8, 3, 32, 32)} for float32 inputs")
        names = [str(k) for k in dict(shapes)]  # glue: the shapes argument, one entry per named input
        # float32 zeros of each shape, made by torch: the export reads their shapes only
        args = tuple(torch.zeros(tuple(int(d) for d in dims)) for dims in dict(shapes).values())  # glue: one tensor per named input
        return args, names
    args = tuple(example_inputs) if isinstance(example_inputs, (tuple, list)) else (example_inputs,)
    args = tuple(a if isinstance(a, torch.Tensor) else torch.as_tensor(a) for a in args)  # glue: one tensor per model input
    if shapes:
        names = [str(k) for k in dict(shapes)]  # glue: the shapes argument, one entry per named input
        if len(names) != len(args):
            raise ValueError(f"shapes names {len(names)} inputs but example_inputs holds {len(args)}")
    else:
        names = ["input"] if len(args) == 1 else [f"input_{i}" for i in range(len(args))]  # glue: one name per model input
    return args, names


def _export_torch(module, path, shapes, example_inputs):
    """Export a torch.nn.Module to the ONNX file ``path`` in this process."""
    torch = _require_torch()
    if not isinstance(module, torch.nn.Module):
        _read(module)  # a torch object that is not a module: refused with the file path
    args, names = _example_args(torch, shapes, example_inputs)
    kwargs = {"opset_version": _TORCH_OPSET, "input_names": names, "do_constant_folding": True}
    import inspect
    try:
        params = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
        params = {}
    if "dynamo" in params:
        # the TorchScript exporter: no onnxscript needed, fixed shapes from the example inputs
        kwargs["dynamo"] = False
    label = type(module).__name__
    try:
        with torch.no_grad():
            torch.onnx.export(module.eval(), args, path, **kwargs)
    except Exception as e:
        raise UnsupportedModel(label, [("torch.onnx.export", None, f"{type(e).__name__}: {e}")]) from e
    return label


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
        # the session binding's ImportedProgram.run entry (autograd/program_pyapi.mojo
        # ops_imported_run_py), retained once here: run is that one call
        self._run_native = session._b.ops_imported_run
        # the same run on caller CUDA/ROCm device buffers (autograd/program_device_pyapi.mojo
        # ops_imported_run_device_py; the torch.compile backend's device route). None on a binding
        # without it (the CPU-only install's host binding): callers then pass host arrays to run.
        self._run_device = getattr(session._b, "ops_imported_run_device", None)

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


def import_model(source, *, shapes=None, example_inputs=None, numeric_mode=None):
    """Import a trained model for inference.

    ``source`` is the path of an ONNX file (opset 13 or later), an
    ``onnx.ModelProto``, or a PyTorch ``torch.nn.Module``; a PyTorch module is
    exported to ONNX in this process and imported from that. The model's weights are copied to the
    device once and its operations run as one call per
    :meth:`ImportedProgram.run`, with the same results on Apple, NVIDIA and
    AMD GPUs and on the CPU. In `identical` mode, mojolearn returns the same
    bits on Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    source : str, path-like, onnx.ModelProto or torch.nn.Module
        The model: its ONNX file, an ONNX model in memory, or a PyTorch
        module. A PyTorch module is put in evaluation mode and exported with
        ``torch.onnx.export`` at opset 17; a model that export cannot write
        can still be exported by hand and imported from the file.
    shapes : dict, optional
        The shape of each input whose size the model leaves open (such as
        the batch), by input name: ``{"input": (8, 3, 32, 32)}``. The
        imported model then accepts arrays of exactly those shapes. For a
        PyTorch module without ``example_inputs``, the module is exported
        with float32 zeros of these shapes, and the keys name its inputs.
    example_inputs : tensor or tuple of tensors, optional
        For a PyTorch module: example inputs of the shapes and dtypes you
        will run, one per argument of ``forward`` (a tuple for several).
        Their values are not used. The imported model takes inputs of
        exactly these shapes. Pass these or ``shapes``.
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
        For a PyTorch module, the export's error is the one problem listed.
    ValueError
        A PyTorch module was passed without ``example_inputs`` or ``shapes``.
    ImportError
        A PyTorch module was passed and PyTorch cannot be imported.

    Examples
    --------
    >>> import numpy as np
    >>> import mojolearn
    >>> prog = mojolearn.import_model("classifier.onnx", shapes={"input": (8, 3, 32, 32)})  # doctest: +SKIP
    >>> logits = prog.run(np.zeros((8, 3, 32, 32), dtype=np.float32))  # doctest: +SKIP
    >>> prog.save("classifier.mlprog")  # doctest: +SKIP

    A PyTorch module, imported directly:

    >>> import torch  # doctest: +SKIP
    >>> prog = mojolearn.import_model(model, example_inputs=torch.zeros(8, 3, 32, 32))  # doctest: +SKIP
    """
    if _is_torch(source) or _is_onnx_proto(source):
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="mojolearn-import-")
        try:
            path = os.path.join(tmp, "model.onnx")
            if _is_onnx_proto(source):
                with open(path, "wb") as f:
                    f.write(source.SerializeToString())
                label = "ModelProto"
            else:
                label = _export_torch(source, path, shapes, example_inputs)
            return _import(path, label, shapes, numeric_mode)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if example_inputs is not None:
        raise ValueError("example_inputs is for a torch.nn.Module; an ONNX model takes shapes=")
    return _import(source, None, shapes, numeric_mode)


def _import(source, label, shapes, numeric_mode):
    # Artifact setup is one compiled call. The public dispatcher/export above
    # remains Python runtime debt; this does not qualify generic Torch import.
    return _IMPORT_NATIVE(source, label, shapes, numeric_mode, _IMPORT_CONTEXT)


from . import _native_import_bridge  # noqa: E402
_IMPORT_NATIVE, _IMPORT_CONTEXT = _native_import_bridge.connect(
    Session, Program, ImportedProgram, GraphIR, UnsupportedModel)
