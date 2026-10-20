# SPDX-License-Identifier: Apache-2.0
"""Import trained models for inference.

An imported model runs on mojolearn's own operations, so it returns the same
results on Apple, NVIDIA and AMD GPUs and on the CPU. The model file is read
on the host; its weights are copied to the device once.

:func:`supported_ops` lists the operations an import supports and the
attribute values each accepts. A model that uses anything else raises
:class:`UnsupportedModel`, which names every node the import cannot run.
"""
from __future__ import annotations

from ._graph import UnsupportedModel

UnsupportedModel.__module__ = __name__  # documented as mojolearn.importers.UnsupportedModel

__all__ = ["UnsupportedModel", "supported_ops", "import_model"]


def supported_ops():
    """The operations a model import supports.

    Returns a list of ``(op, supported_subset)`` pairs: the ONNX operation
    name and a short description of the attribute values and input shapes it
    accepts. Use it to check a model's operations before exporting it.

    >>> from mojolearn import importers
    >>> ops = dict(importers.supported_ops())
    """
    # The table is the lowering's (importers/_lower.py, `SUPPORTED_OPS`: a
    # sequence of (op, subset) string pairs); read lazily so this package
    # imports before the lowering module exists.
    try:
        from . import _lower
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}._lower":
            raise
        return []
    return list(getattr(_lower, "SUPPORTED_OPS", ()))


def __getattr__(name):
    # `import_model` lives with the Program glue (importers/_program.py)
    if name == "import_model":
        from ._program import import_model
        return import_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
