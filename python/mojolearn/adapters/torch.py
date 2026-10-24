# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Run a PyTorch model as one mojolearn program, and train mojolearn models from PyTorch.

Importing this module registers:

* the ``torch.compile`` backend ``"mojolearn"`` (also :data:`backend`):
  ``torch.compile(model, backend="mojolearn")`` runs every operation of the
  model inside one mojolearn program call, so PyTorch only holds the input
  and output tensors. A model with an operation the import does not support
  is refused with the names of those nodes; nothing runs in PyTorch instead.
* ``torch.ops.mojolearn.train(inputs, model, options, scalars)``, which fits a
  whole mojolearn Sequential model and returns its parameters, optimizer
  state and loss curve.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD
GPUs.

Requires PyTorch: ``pip install "mojolearn[torch]"`` (on Linux,
``pip install mojolearn "mojolearn-core[torch]"``; on AMD, install PyTorch's
ROCm build first).

Examples
--------
>>> import torch
>>> import mojolearn.adapters.torch
>>> model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
>>> compiled = torch.compile(model, backend="mojolearn", fullgraph=True, dynamic=False)
>>> compiled(torch.ones(8, 3)).shape
torch.Size([8, 2])
"""
import platform as _platform

from . import _shipped
from ._torch_compile import backend, register as _register_backend

__all__ = ["backend", "ops", "library_path", "native_library_path"]


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "mojolearn.adapters.torch needs PyTorch. Install it with: pip install \"mojolearn[torch]\" "
            "(Linux: pip install mojolearn \"mojolearn-core[torch]\"; AMD: the ROCm build from "
            "https://download.pytorch.org/whl/rocm7.2)") from exc
    return torch


_torch = _import_torch()

# torch.compile(model, backend="mojolearn"): registered before the operator library loads, since the
# backend runs through the mojolearn bindings and needs no registration library.
_register_backend()


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


# torch.ops.mojolearn.operation (one mojolearn operation per call) stays registered for the trainer
# and the shipped adapter smoke, but is not documented (docs-site/api_exclude.txt): it is a building
# block; a whole model runs as one program (backend="mojolearn" or mojolearn.import_model).
#: ``torch.ops.mojolearn``: the registered ``train`` operator.
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
