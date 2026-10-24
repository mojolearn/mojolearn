# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Locate the framework adapter files this install ships, and read their manifest.

Glue only: paths and the build manifest's metadata. The adapter files sit in
`_adapters/` beside the GPU set mojolearn selected (Linux: inside the vendor
wheel's architecture set; macOS: in the package directory), written by
packaging/adapters/build.sh and described by `adapters.json`
(packaging/adapters/manifest.py, schema mojolearn.adapters.v1).
"""
import json
import os

# The same names as packaging/adapters/manifest.py (test_adapters.py holds them equal).
SCHEMA = "mojolearn.adapters.v1"
ABI = 1
DIRECTORY = "_adapters"
MANIFEST = "adapters.json"
FILES = {"native": "libmojolearn_adapter.so", "torch": "mojolearn_torch.so", "max": "mojolearn_ops.mojoc"}

_CACHE = {}


def directory():
    """The `_adapters/` directory of the GPU set this process uses."""
    if "directory" in _CACHE:
        return _CACHE["directory"]
    from .. import _backend
    try:
        _kind, base = _backend._layout()
    except ImportError as exc:
        raise ImportError(
            "mojolearn: the framework adapters run on a GPU, and this install selected no GPU set "
            f"({exc})") from exc
    path = os.path.join(base, DIRECTORY)
    if not os.path.isfile(os.path.join(path, MANIFEST)):
        raise ImportError(
            "mojolearn: this install carries no framework adapters for its GPU set "
            f"(no {MANIFEST} in {path}). They ship in the mojolearn GPU wheels: "
            "reinstall mojolearn (on Linux: mojolearn-nvidia or mojolearn-amd).")
    _CACHE["directory"] = path
    return path


def manifest():
    """The parsed `adapters.json` of this install's adapters."""
    if "manifest" in _CACHE:
        return _CACHE["manifest"]
    path = os.path.join(directory(), MANIFEST)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("abi") != ABI:
        raise ImportError(f"mojolearn: {path} is not a {SCHEMA} manifest of adapter ABI {ABI}; reinstall mojolearn")
    _CACHE["manifest"] = doc
    return doc


def path(role):
    """The installed path of one adapter file: 'native', 'torch' or 'max'."""
    full = os.path.join(directory(), FILES[role])
    if not os.path.isfile(full):
        raise ImportError(f"mojolearn: the framework adapter file {full} is missing; reinstall mojolearn")
    return full


def series(version):
    """'2.13' from '2.13.0+rocm7.2' (the Torch C++ ABI is per minor series)."""
    return ".".join(version.split("+", 1)[0].split(".")[:2])
