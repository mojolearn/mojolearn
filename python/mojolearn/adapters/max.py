# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Train a mojolearn model from a MAX graph.

mojolearn ships a MAX custom-operation package with two whole-training
operations that call the compiled mojolearn library: ``mojolearn_train`` fits
a mojolearn Sequential model in one call, and ``mojolearn_train_resume``
continues a fit from its saved state. Give the graph the package with
:func:`custom_extensions` and each operation its parameters with
:func:`parameters`.

Requires MAX: ``pip install "mojolearn[max]"`` (on Linux,
``pip install mojolearn "mojolearn-core[max]"``).

Examples
--------
>>> from max.dtype import DType
>>> from max.graph import DeviceRef, Graph, TensorType, ops
>>> from mojolearn.adapters import max as mojolearn_max
>>> gpu = DeviceRef.GPU(0)
>>> x_t = TensorType(DType.float32, [8, 1], device=gpu)
>>> p_t = TensorType(DType.float32, [2], device=gpu)
>>> outs = [p_t, p_t, p_t, TensorType(DType.int32, [24], device=gpu),
...         TensorType(DType.float32, [1], device=gpu)]
>>> params = mojolearn_max.parameters(model="1,1,1,1,1,1,0",
...                                   options="0,0,0,0,0,1,2,1,0,1,0,0,0",
...                                   scalars="0.125,0,0,1e-8,0,0,0,1,1,0")
>>> with Graph("fit", input_types=[x_t, x_t, p_t],
...            custom_extensions=mojolearn_max.custom_extensions()) as g:
...     fitted = ops.custom("mojolearn_train", gpu, [v.tensor for v in g.inputs],
...                         out_types=outs, parameters=params)
...     g.output(*[t.tensor for t in fitted])
"""
# The package also holds per-operation building blocks (mojolearn_matmul, mojolearn_relu,
# mojolearn_operation0..3). They stay for the trainer and the shipped adapter smoke but are not
# documented (docs-site/api_exclude.txt): a building block; a whole model runs as one program
# (mojolearn.import_model, or torch.compile(model, backend="mojolearn")).
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
