# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Run mojolearn operations and whole-model training from PyTorch.

Importing this module registers mojolearn's operators with the PyTorch
dispatcher:

* ``torch.ops.mojolearn.operation(inputs, code, output_shape, options=[],
  scalars=[], integer_output=False, device=None)`` runs one mojolearn
  operation and returns a new tensor of ``output_shape``.
* ``torch.ops.mojolearn.train(inputs, model, options, scalars)`` fits a whole
  mojolearn Sequential model and returns its parameters, optimizer state and
  loss curve.

Both call the compiled mojolearn library that ships with your mojolearn GPU
install. Tensors are contiguous float32 or int32, on the CPU or on GPU 0
(CUDA, ROCm or MPS).

Requires PyTorch: ``pip install "mojolearn[torch]"`` (on Linux,
``pip install mojolearn "mojolearn-core[torch]"``; on AMD, install PyTorch's
ROCm build first).

Examples
--------
>>> import torch
>>> from mojolearn.adapters import torch as mojolearn_torch
>>> x = torch.ones(2, 3, device="cuda")
>>> torch.ops.mojolearn.operation([x, x], 2, [2, 3]).sum().item()  # code 2: add
12.0
"""
import platform as _platform

from . import _shipped

__all__ = ["ops", "library_path", "native_library_path"]

try:
    import torch as _torch
except ImportError as _exc:
    raise ImportError(
        "mojolearn.adapters.torch needs PyTorch. Install it with: pip install \"mojolearn[torch]\" "
        "(Linux: pip install mojolearn \"mojolearn-core[torch]\"; AMD: the ROCm build from "
        "https://download.pytorch.org/whl/rocm7.2)") from _exc


def _refusal(record):
    """Why this torch cannot load the shipped registration, or None. The
    registration is a C++ extension built against one Torch minor series, one
    backend and one C++ ABI (recorded in adapters.json)."""
    want = record.get("torch") or {}
    have = _shipped.series(_torch.__version__)
    if have != want.get("series"):
        return (f"it was built for torch {want.get('series')}.x and torch {_torch.__version__} is installed; "
                f"install {record.get('requirements', {}).get('torch', 'torch ' + str(want.get('series')))}")
    backend = want.get("backend")
    if backend == "rocm" and not getattr(_torch.version, "hip", None):
        return f"this mojolearn runs on AMD and torch {_torch.__version__} is not a ROCm build"
    if backend == "cuda" and (getattr(_torch.version, "hip", None) or not getattr(_torch.version, "cuda", None)):
        return f"this mojolearn runs on NVIDIA and torch {_torch.__version__} is not a CUDA build"
    if backend == "mps" and _platform.system() != "Darwin":
        return "the MPS build of the adapter loads on macOS only"
    abi = want.get("cxx11_abi")
    if abi is not None and int(bool(_torch.compiled_with_cxx11_abi())) != abi:
        return f"torch {_torch.__version__} uses a different C++ ABI than the one the adapter was built with"
    return None


def _load():
    record = _shipped.manifest()
    why = _refusal(record)
    if why is not None:
        raise ImportError("mojolearn.adapters.torch cannot load: " + why)
    if not hasattr(_torch.ops, "mojolearn") or not hasattr(_torch.ops.mojolearn, "operation"):
        _torch.ops.load_library(_shipped.path("torch"))
    return _torch.ops.mojolearn


#: ``torch.ops.mojolearn``: the registered ``operation`` and ``train`` operators.
ops = _load()


def library_path():
    """The path of the PyTorch registration this module loaded.

    Returns
    -------
    str
        The installed ``mojolearn_torch.so``, usable with
        ``torch.ops.load_library`` in another process.
    """
    return _shipped.path("torch")


def native_library_path():
    """The path of the native mojolearn adapter library the operators call.

    Returns
    -------
    str
        The installed ``libmojolearn_adapter.so``.
    """
    return _shipped.path("native")
