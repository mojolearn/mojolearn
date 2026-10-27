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
    return _NATIVE("directory", (), _CONTEXT)


def manifest():
    """The parsed `adapters.json` of this install's adapters."""
    return _NATIVE("manifest", (), _CONTEXT)


def path(role):
    """The installed path of one adapter file: 'native', 'torch' or 'max'."""
    return _NATIVE("path", (role,), _CONTEXT)


def series(version):
    """'2.13' from '2.13.0+rocm7.2' (the Torch C++ ABI is per minor series)."""
    return _NATIVE("series", (version,), _CONTEXT)


def _connect():
    # glue: framework installation and exact path storage are connected once.
    # No model or public metadata helper calls this setup function.
    from .. import _backend
    from .._native_family_bridge import _unavailable
    from pathlib import PosixPath
    from types import MemberDescriptorType
    try:
        _kind, base = _backend._layout()
        directory = os.path.join(base, DIRECTORY)
        error = None
    except ImportError as exc:
        directory, error = None, str(exc)
    fields = {}
    for cls in PosixPath.__mro__:  # glue: fixed standard-library slot descriptors
        fields.update({name: value for name, value in vars(cls).items()  # glue: import-time native slot metadata only
                       if type(value) is MemberDescriptorType})
    raw = '_raw_paths' in fields
    layout = {'class': PosixPath, 'raw_paths': raw}
    if raw:
        layout['set_raw'] = fields['_raw_paths'].__set__
    else:
        layout.update(set_drive=fields['_drv'].__set__, set_root=fields['_root'].__set__,
                      set_parts=fields['_parts'].__set__)
    try:
        entry = _backend.binding("_mojolearn_x_sequence").ops_import_model
    except (ImportError, AttributeError) as exc:
        entry = _unavailable(exc)
    return entry, dict(cache=_CACHE, directory=directory, layout_error=error,
                               schema=SCHEMA, abi=ABI, files=FILES, path_layout=layout)


_NATIVE, _CONTEXT = _connect()
