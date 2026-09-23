# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A pure-Python reader of the `.safetensors` format (lane/model-loader,
2026-09-17), returning `mojolearn.Array` tensors.

THE FORMAT. Eight bytes, a little-endian uint64 `N`; then `N` bytes of JSON,
an object whose keys are tensor names (plus an optional `__metadata__`) and
whose values are `{"dtype": ..., "shape": [...], "data_offsets": [begin,
end]}` with the offsets relative to the byte after the header; then the
tensor bytes, row-major, little-endian, back to back. A sharded checkpoint
adds `model.safetensors.index.json`, `{"weight_map": {name: file}}`, one
file per shard beside it.

WHAT MOVES AND WHAT DOES NOT. Every tensor is memory-mapped and copied ONCE
into an owned `Array` (a `memoryview` of the map, cast to the element type,
then one `Array.copy()`); a shard is never read whole into a Python `bytes`,
and never twice. The dtypes and what each becomes:

    F32    float32 as stored, bit for bit
    BF16   the uint16 bits; widened to float32 EXACTLY by the shift through
           `mojolearn.lowbit` (contract L-1, the one spelling the inference
           classes materialize with), or kept as bits when the caller asks
           for `bf16="bits"` (a 2-D tensor then comes back as a
           `lowbit.BF16Weight`, the form a block accepts directly)
    F16    widened to float32 EXACTLY by bit construction here (`_f16_bits_to_f32`:
           sign, exponent rebias by 112, mantissa shift by 13, subnormals
           renormalized, infinities and NaNs carried), no library cast
    I32    int32 as stored
    I64    int64 as stored

Every other dtype (F64, F8_*, U8, I8, I16, U16, BOOL, ...) is refused BY
NAME with the tensor's name; nothing is downcast. This reader moves no
arithmetic bit, so it takes no DEVIATION number.
"""
import array
import json
import mmap
import os
import struct

from .. import lowbit as _lowbit
from .._array import Array
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
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.begin = begin
        self.end = end
        self.file = file

    @property
    def nbytes(self):
        return self.end - self.begin

    @property
    def size(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __repr__(self):
        return f"TensorInfo({self.name!r}, {self.dtype}, shape={self.shape})"


def _f16_bits_to_f32(h):
    """The float32 bit pattern of the float16 bit pattern `h`, exact.

    A float16 is `sign:1 exponent:5 mantissa:10` with bias 15; a float32 is
    `1:8:23` with bias 127. A normal value keeps its mantissa (shifted up
    by 13) and rebiases the exponent by 112. A subnormal float16
    (`exponent == 0`, mantissa nonzero) is `mantissa * 2^-24`, which is a
    NORMAL float32: its leading one is found by shifting, the exponent
    adjusted by the same count. Zero keeps its sign; the infinities and
    every NaN keep theirs and their payload."""
    s = (h & 0x8000) << 16
    e = (h >> 10) & 0x1F
    m = h & 0x03FF
    if e == 0:
        if m == 0:
            return s
        # renormalize: shift until the leading one sits at bit 10
        shift = 0
        while (m & 0x0400) == 0:
            m <<= 1
            shift += 1
        m &= 0x03FF
        return s | ((113 - shift) << 23) | (m << 13)
    if e == 31:
        return s | 0x7F800000 | (m << 13)
    return s | ((e + 112) << 23) | (m << 13)


def widen_f16(bits):
    """A `'<f2'` Array of any rank to float32, by bit construction."""
    h = array.array("H")
    h.frombytes(bits.tobytes())
    u = array.array("I", bytes(4 * len(h)))
    for k in range(len(h)):
        u[k] = _f16_bits_to_f32(h[k])
    f = array.array("f")
    f.frombytes(u.tobytes())
    return Array._owned(f, tuple(bits.shape), "<f4", "C")


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
        self.path = os.fspath(path)
        self._fh = None
        self._mm = None
        with open(self.path, "rb") as fh:
            head = fh.read(8)
            if len(head) != 8:
                raise ValueError(f"mojolearn.models.safetensors: {self.path} is shorter than its 8-byte header length")
            (n,) = struct.unpack("<Q", head)
            if n > _HEADER_LIMIT:
                raise ValueError(f"mojolearn.models.safetensors: {self.path}: header length {n} exceeds {_HEADER_LIMIT}")
            raw = fh.read(n)
            if len(raw) != n:
                raise ValueError(f"mojolearn.models.safetensors: {self.path}: header is truncated ({len(raw)} of {n} bytes)")
            fh.seek(0, os.SEEK_END)
            total = fh.tell()
        self.data_start = 8 + n
        self.data_size = total - self.data_start
        header = _parse_header(raw, self.path)
        self.metadata = header.pop("__metadata__", None)
        self._infos = {}
        for name, entry in header.items():
            if not isinstance(entry, dict) or not {"dtype", "shape", "data_offsets"} <= set(entry):
                raise ValueError(f"mojolearn.models.safetensors: {self.path}: tensor {name!r} lacks dtype/shape/data_offsets")
            dtype = entry["dtype"]
            shape = tuple(int(s) for s in entry["shape"])
            begin, end = (int(v) for v in entry["data_offsets"])
            if begin < 0 or end < begin or end > self.data_size:
                raise ValueError(f"mojolearn.models.safetensors: {self.path}: tensor {name!r} offsets [{begin}, {end}) fall outside the {self.data_size}-byte data section")
            if dtype in DTYPES:
                size = 1
                for s in shape:
                    size *= s
                if size * DTYPES[dtype][2] != end - begin:
                    raise ValueError(f"mojolearn.models.safetensors: {self.path}: tensor {name!r} of {dtype} shape {shape} needs {size * DTYPES[dtype][2]} bytes, the offsets give {end - begin}")
            self._infos[name] = TensorInfo(name, dtype, shape, begin, end, self.path)

    # ------------------------------------------------------------- listing
    def names(self):
        return list(self._infos)

    def info(self, name):
        try:
            return self._infos[name]
        except KeyError:
            raise KeyError(f"mojolearn.models.safetensors: {self.path} holds no tensor {name!r}") from None

    def __contains__(self, name):
        return name in self._infos

    # ------------------------------------------------------------- reading
    def _map(self):
        if self._mm is None:
            self._fh = open(self.path, "rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def read(self, name, *, bf16="widen"):
        """The tensor `name` as an owned `Array` (see the module header for
        what each dtype becomes). `bf16` is "widen" (float32) or "bits" (the
        uint16 bits; a 2-D tensor as a `lowbit.BF16Weight`)."""
        if bf16 not in ("widen", "bits"):
            raise ValueError(f"mojolearn.models.safetensors: bf16 must be 'widen' or 'bits', got {bf16!r}")
        t = self.info(name)
        if t.dtype not in DTYPES:
            raise TypeError(
                f"mojolearn.models.safetensors: tensor {name!r} in {self.path} has dtype {t.dtype}, "
                f"which this reader refuses by name; it reads {sorted(DTYPES)} and downcasts nothing")
        typestr, code, itemsize = DTYPES[t.dtype]
        mm = self._map()
        view = memoryview(mm)
        try:
            raw = view[self.data_start + t.begin:self.data_start + t.end]
            try:
                if t.size == 0:
                    owned = frombytes(b"", typestr, t.shape)
                elif code == "e":
                    # memoryview.cast("e") exists only from Python 3.12, and the
                    # wheel supports 3.10 up; float16 is widened from its bits
                    # below, so its bytes are copied as they are.
                    owned = frombytes(raw.tobytes(), typestr, t.shape if t.shape else (1,))
                    if not t.shape:
                        owned = owned.reshape(())
                else:
                    shape = t.shape if t.shape else (1,)
                    typed = raw.cast("B").cast(code, shape)
                    try:
                        owned = Array.from_buffer(typed).copy()  # the one copy
                    finally:
                        typed.release()
                    if not t.shape:
                        owned = owned.reshape(())
            finally:
                raw.release()
        finally:
            view.release()
        if t.dtype == "F16":
            return widen_f16(owned)
        if t.dtype == "BF16":
            if bf16 == "bits":
                return _lowbit.BF16Weight(owned) if owned.ndim == 2 else owned
            return _lowbit.widen_bf16(owned)
        return owned

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __repr__(self):
        return f"SafetensorsFile({self.path!r}, tensors={len(self._infos)})"


class Checkpoint:
    """A single `model.safetensors`, an explicit `.safetensors` path, or a
    directory holding `model.safetensors.index.json` and its shards. Shards
    are opened on first use and closed by `close()`."""

    def __init__(self, files, weight_map, root):
        self.root = root
        self._paths = list(files)
        self._weight_map = dict(weight_map)
        self._open = {}
        self._infos = None

    @classmethod
    def open(cls, path):
        path = os.fspath(path)
        if os.path.isdir(path):
            index = os.path.join(path, INDEX_NAME)
            single = os.path.join(path, SINGLE_NAME)
            if os.path.isfile(index):
                with open(index, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                wm = data.get("weight_map") if isinstance(data, dict) else None
                if not isinstance(wm, dict) or not wm:
                    raise ValueError(f"mojolearn.models.safetensors: {index} has no weight_map")
                files = sorted(set(os.path.join(path, f) for f in wm.values()))
                missing = [f for f in files if not os.path.isfile(f)]
                if missing:
                    raise FileNotFoundError(f"mojolearn.models.safetensors: {index} names shards that do not exist: {missing}")
                return cls(files, {n: os.path.join(path, f) for n, f in wm.items()}, path)
            if os.path.isfile(single):
                return cls([single], {}, path)
            others = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
            if len(others) == 1:
                one = os.path.join(path, others[0])
                return cls([one], {}, path)
            raise FileNotFoundError(
                f"mojolearn.models.safetensors: {path} holds neither {SINGLE_NAME} nor {INDEX_NAME}"
                + (f" (found {others}; one unnamed shard set is not an index)" if others else ""))
        if os.path.isfile(path):
            return cls([path], {}, os.path.dirname(path) or ".")
        raise FileNotFoundError(f"mojolearn.models.safetensors: {path} does not exist")

    def _file(self, path):
        f = self._open.get(path)
        if f is None:
            f = SafetensorsFile(path)
            self._open[path] = f
        return f

    def _index(self):
        if self._infos is None:
            infos = {}
            for p in self._paths:
                f = self._file(p)
                for n in f.names():
                    if n in infos:
                        raise ValueError(f"mojolearn.models.safetensors: tensor {n!r} appears in both {infos[n].file} and {p}")
                    infos[n] = f.info(n)
            for n, p in self._weight_map.items():
                if n not in infos:
                    raise ValueError(f"mojolearn.models.safetensors: the index maps {n!r} to {p} but that shard does not hold it")
            self._infos = infos
        return self._infos

    def names(self):
        return list(self._index())

    def info(self, name):
        try:
            return self._index()[name]
        except KeyError:
            raise KeyError(f"mojolearn.models.safetensors: no tensor {name!r} under {self.root}") from None

    def __contains__(self, name):
        return name in self._index()

    def read(self, name, *, bf16="widen"):
        t = self.info(name)
        return self._file(t.file).read(name, bf16=bf16)

    def close(self):
        for f in self._open.values():
            f.close()
        self._open = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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
    for name in sorted(tensors):
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
        for raw in payload:
            fh.write(raw)
    return path
