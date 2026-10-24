# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Call mojolearn operations and whole-model training from a MAX graph.

mojolearn ships a MAX custom-operation package whose operations call the
compiled mojolearn library: ``mojolearn_matmul``, ``mojolearn_relu``,
``mojolearn_operation0`` to ``mojolearn_operation3`` (one generic operation of
zero to three inputs), ``mojolearn_train`` and ``mojolearn_train_resume``.
Give the graph the package with :func:`custom_extensions` and every
operation the library with :func:`parameters`.

Requires MAX: ``pip install "mojolearn[max]"`` (on Linux,
``pip install mojolearn "mojolearn-core[max]"``).

Examples
--------
>>> from max.driver import Accelerator
>>> from max.dtype import DType
>>> from max.engine import InferenceSession
>>> from max.graph import DeviceRef, Graph, TensorType, ops
>>> from mojolearn.adapters import max as mojolearn_max
>>> gpu = DeviceRef.GPU(0)
>>> t = TensorType(DType.float32, [2, 2], device=gpu)
>>> with Graph("relu", input_types=[t], custom_extensions=mojolearn_max.custom_extensions()) as g:
...     y = ops.custom("mojolearn_relu", gpu, [g.inputs[0].tensor], out_types=[t],
...                    parameters=mojolearn_max.parameters())[0].tensor
...     g.output(y)
>>> model = InferenceSession(devices=[Accelerator(0)]).load(g)
"""
import importlib.metadata as _metadata

from . import _shipped

__all__ = ["custom_extensions", "native_library_path", "parameters"]


def _import_max():
    try:
        import max
    except ImportError as exc:
        raise ImportError(
            "mojolearn.adapters.max needs MAX. Install it with: pip install \"mojolearn[max]\" "
            "(Linux: pip install mojolearn \"mojolearn-core[max]\")") from exc
    return max


_max = _import_max()


def _installed_max():
    try:
        return _metadata.version("max")
    except _metadata.PackageNotFoundError:
        return getattr(_max, "__version__", None)


def _check():
    record = _shipped.manifest()
    want = (record.get("max") or {}).get("version")
    have = _installed_max()
    if have != want:
        raise ImportError(
            f"mojolearn.adapters.max cannot load: the operations were built for MAX {want} and MAX {have} "
            f"is installed; install {record.get('requirements', {}).get('max', 'max==' + str(want))}")
    _shipped.path("max")
    _shipped.path("native")


_check()


def custom_extensions():
    """The MAX custom-operation package, for ``Graph(..., custom_extensions=...)``.

    Returns
    -------
    list of pathlib.Path
        The installed ``mojolearn_ops.mojoc``.
    """
    import pathlib
    return [pathlib.Path(_shipped.path("max"))]


def native_library_path():
    """The path of the compiled mojolearn adapter library the operations call.

    Returns
    -------
    str
        The installed ``libmojolearn_adapter.so``.
    """
    return _shipped.path("native")


def parameters(**extra):
    """The ``parameters`` of a mojolearn custom operation.

    Parameters
    ----------
    **extra
        The operation's own parameters, for example ``code``, ``options`` and
        ``scalars`` of ``mojolearn_operation1``.

    Returns
    -------
    dict
        ``{"library": <the native library path>, **extra}``.
    """
    return dict(extra, library=native_library_path())
