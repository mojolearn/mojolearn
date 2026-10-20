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
    # the table of autograd/import_lower.mojo's ops and checks (importers/_table.py)
    from ._table import SUPPORTED_OPS
    return list(SUPPORTED_OPS)


def __getattr__(name):
    # `import_model` lives with the Program glue (importers/_program.py)
    if name == "import_model":
        from ._program import import_model
        return import_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---- lane ag-import-torch: the reader for a source, by its type ----------
# `import_model` (importers/_program.py) reads its `source` through
# `_read_source`, so a path and a torch.export program take one route.
# torch is never imported here: a torch.export program can only exist once
# the caller has imported torch, so the check looks in sys.modules.
def _read_source(source):
    """The import graph of ``source``: an ONNX file path or a
    ``torch.export.ExportedProgram``."""
    import os
    import sys

    if isinstance(source, (str, bytes, os.PathLike)):
        from .onnx import read
        return read(source)
    texport = sys.modules.get("torch.export")
    exported = getattr(texport, "ExportedProgram", None)
    if exported is not None and isinstance(source, exported):
        from .torch_export import read
        return read(source)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(source, torch.nn.Module):
        raise TypeError("import a torch module by exporting it first: "
                        "mojolearn.import_model(torch.export.export(model.eval(), example_inputs))")
    raise TypeError(f"source must be the path of an ONNX file or a torch.export.ExportedProgram, "
                    f"got {type(source).__name__}")
# ---- end lane ag-import-torch ---------------------------------------------
