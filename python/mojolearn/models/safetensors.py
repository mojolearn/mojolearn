# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Read single-file and sharded safetensors checkpoints into owned arrays.

F32, I32 and I64 preserve their stored bytes. F16 and BF16 widen exactly to
float32; `bf16="bits"` retains uint16 bits and returns a `BF16Weight` for a
matrix. Other storage types are refused by name. Files are mapped lazily and
may be closed explicitly or used with a context manager.
"""
import json
import mmap
import os
import struct

from .. import lowbit as _lowbit
from .._array import Array
from ..tokenizer import _binding

_NATIVE = _binding()._module
from .._buffer import frombytes

__all__ = ["SafetensorsFile", "Checkpoint", "TensorInfo", "DTYPES", "INDEX_NAME", "SINGLE_NAME"]

#: safetensors dtype name -> (Array typestr, memoryview cast code, itemsize)
DTYPES = {
    "F32": ("<f4", "f", 4),
    "F16": ("<f2", "e", 2),
    "BF16": ("<u2", "H", 2),
    "I32": ("<i4", "i", 4),
    "I64": ("<i8", "q", 8),
}

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"

_HEADER_LIMIT = 100 * 1024 * 1024  # a header above this is not a checkpoint


class TensorInfo:
    """One header entry: `name`, `dtype` (the safetensors spelling), `shape`
    (a tuple), `begin`/`end` (offsets into the data section) and `file`."""

    __slots__ = ("name", "dtype", "shape", "begin", "end", "file")

    def __init__(self, name, dtype, shape, begin, end, file):
        return _NATIVE.tensor_info_init(self, (name, dtype, shape, begin, end, file))

    @property
    def nbytes(self):
        return _NATIVE.tensor_info_size(self, True)

    @property
    def size(self):
        return _NATIVE.tensor_info_size(self, False)

    def __repr__(self):
        return f"TensorInfo({self.name!r}, {self.dtype}, shape={self.shape})"


def widen_f16(bits):
    """A float16 Array of any rank (`'<f2'`, or its bits as `'<u2'`) to
    float32, by bit construction: a normal value keeps its mantissa shifted
    up by 13 and rebiases the exponent by 112, a subnormal is renormalized
    into a normal float32, zero keeps its sign, and the infinities and
    every NaN keep sign and payload. The construction is
    `gemm/contract.mojo::f16_bits_to_f32`, run by the linalg binding's
    `from_f16` (a device kernel on a GPU install, the linalg host binding
    on a CPU-only one); the per-element Python loop is gone
    (pyglue-text-io, Oct 3). Exact, so every column has the same bits."""
    from .. import _linalg_impl
    shape = tuple(bits.shape)
    if bits.dtype != "<u2":
        bits = frombytes(bits.tobytes(), "<u2", shape if shape else (1,))
    if bits.size == 0:
        return frombytes(b"", "<f4", shape)
    out = _linalg_impl.from_f16(bits)
    return out.reshape(shape) if shape != tuple(out.shape) else out


def _parse_header(raw, path):
    try:
        header = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"mojolearn.models.safetensors: {path}: the header is not JSON: {exc}") from None
    if not isinstance(header, dict):
        raise ValueError(f"mojolearn.models.safetensors: {path}: the header is not a JSON object")
    return header


class SafetensorsFile:
    """One `.safetensors` file. `names()`, `info(name)`, `read(name)`; a
    context manager that closes the map. The map is opened on the first
    read and released by `close()`."""

    def __init__(self, path):
        return _NATIVE.safetensors_init(self, path, _NATIVE_CONTEXT)

    # ------------------------------------------------------------- listing
    def names(self):
        return _NATIVE.safetensors_names(self)

    def info(self, name):
        return _NATIVE.safetensors_info(self, name)

    def __contains__(self, name):
        return _NATIVE.safetensors_contains(self, name)

    # ------------------------------------------------------------- reading
    def _map(self):
        return _NATIVE.safetensors_map(self)

    def read(self, name, *, bf16="widen"):
        """The tensor `name` as an owned `Array` (see the module header for
        what each dtype becomes). `bf16` is "widen" (float32) or "bits" (the
        uint16 bits; a 2-D tensor as a `lowbit.BF16Weight`)."""
        return _NATIVE.safetensors_read(self, name, bf16, _NATIVE_CONTEXT)

    def close(self):
        return _NATIVE.safetensors_close(self)

    def __enter__(self):
        return _NATIVE.checkpoint_enter(self)

    def __exit__(self, *exc):
        return _NATIVE.safetensors_close(self)


    def __repr__(self):
        return f"SafetensorsFile({self.path!r}, tensors={len(self._infos)})"


class Checkpoint:
    """A single `model.safetensors`, an explicit `.safetensors` path, or a
    directory holding `model.safetensors.index.json` and its shards. Shards
    are opened on first use and closed by `close()`."""

    def __init__(self, files, weight_map, root):
        return _NATIVE.checkpoint_init(self, files, weight_map, root)

    @classmethod
    def open(cls, path):
        return _NATIVE.checkpoint_open(cls, path, _NATIVE_CONTEXT)

    def _file(self, path):
        return _NATIVE.checkpoint_file(self, path, _NATIVE_CONTEXT)

    def _index(self):
        return _NATIVE.checkpoint_index(self, _NATIVE_CONTEXT)

    def names(self):
        return _NATIVE.checkpoint_access(self, 0, None, _NATIVE_CONTEXT)

    def info(self, name):
        return _NATIVE.checkpoint_access(self, 1, name, _NATIVE_CONTEXT)

    def __contains__(self, name):
        return _NATIVE.checkpoint_access(self, 2, name, _NATIVE_CONTEXT)

    def read(self, name, *, bf16="widen"):
        return _NATIVE.checkpoint_read(self, name, bf16, _NATIVE_CONTEXT)

    def close(self):
        return _NATIVE.checkpoint_close(self)

    def __enter__(self):
        return _NATIVE.checkpoint_enter(self)

    def __exit__(self, *exc):
        return _NATIVE.checkpoint_close(self)

    def __repr__(self):
        return f"Checkpoint({self.root!r}, shards={len(self._paths)})"


def write_safetensors(path, tensors, metadata=None):
    """WRITE a `.safetensors` file from `{name: (dtype, shape, bytes)}`, the
    tests' synthetic-checkpoint writer. `dtype` is the safetensors spelling,
    `bytes` the row-major little-endian payload. Kept beside the reader so
    the two agree on the one format; not a checkpoint saver for models."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    offset = 0
    payload = []
    for name in sorted(tensors):  # cpu-route: writes the safetensors header in name order (file output)
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(encoded) % 8) % 8
    encoded += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for raw in payload:  # cpu-route: writes the tensor payloads (file output)
            fh.write(raw)
    return path

# Native file/mmap owners release their resources with normal reference lifetime;
# no Python __del__ callback is needed when native load owns temporary readers.
_NATIVE_CONTEXT = {"array_class": Array, "bf16_class": _lowbit.BF16Weight,
                   "info_class": TensorInfo, "file_class": SafetensorsFile,
                   "checkpoint_class": Checkpoint}
