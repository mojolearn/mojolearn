# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Deterministic npz save and load for fitted models, without NumPy
(DEVIATION 2302).

Model files exist for the E1 train-here-infer-there leg, so two rules are
load-bearing rather than stylistic.

FLOATS TRAVEL AS RAW BYTES, NEVER AS DECIMAL TEXT. The npy member format
stores the array's exact bytes, so every float round-trips bit-exactly by
construction. Nothing in this module formats or parses a number.

THE FILE BYTES ARE A PURE FUNCTION OF THE ARRAYS. `np.savez` stamps each
zip member with the current time, so two saves of the SAME model would
hash differently and a cross-machine file comparison would be voided
before it starts. `write_npz` pins the member timestamp to the zip epoch
and the member order to sorted names, so bit-identical arrays give a
bit-identical file. `np.load` reads the result unchanged.

THE SAME BYTES AS 0.6.x. The NPY member is written by a hand-written v1.0
codec that reproduces `numpy.lib.format.write_array` byte for byte:

    \\x93NUMPY                 6 bytes, magic
    \\x01\\x00                   2 bytes, format version 1.0
    <H                         2 bytes, little-endian header length H
    {'descr': '<f4', 'fortran_order': False, 'shape': (3, 2), }
                               ASCII dict, keys in that (sorted) order
    ' ' * pad + '\\n'           spaces so that 10 + H is a multiple of 64
    <data>                     the raw element bytes, C order (or F order
                               when fortran_order is True)

`write_npz` linearizes every member to C order and promotes a 0-d value to
shape `(1,)`, exactly what the 0.6.x writer's `np.ascontiguousarray` did,
so a 0.6.x file and a 0.7 file of the same model are the same bytes. The
lower-level `encode_npy` keeps F order (`fortran_order: True`) like
`write_array` itself.

WHAT COMES BACK. `read_npz` returns a dict whose numeric members are
`Array`s. A string member (`'<U..'`) comes back as a `str` when it holds
exactly one element and as a (nested) list of `str` otherwise; a bytes
member (`'|S..'`) likewise as `bytes` / list of `bytes`; a bool member
(`'|b1'`) as a `'<u1'` Array of 0/1 (DEVIATION 2308). `exact()` keeps
refusing dtype mismatches, and `scalar_str()` reads a one-element string
field as before.
"""

from __future__ import annotations

import ast
import struct
import zipfile

from ._array import Array, normalize_dtype, _ITEMSIZE, _prod
from . import _buffer

MAGIC = b"\x93NUMPY"
_ARRAY_ALIGN = 64
_MAGIC_LEN = len(MAGIC) + 2

# Array typestr -> NPY descr (one-byte types carry '|' in numpy's writer)
_DESCR_OF = {"<u1": "|u1"}
# NPY descr -> Array typestr for the numeric dtypes the codec restores
_TYPESTR_OF = {
    "<f4": "<f4", "<f8": "<f8", "<i4": "<i4", "<i8": "<i8", "<u4": "<u4",
    "|u1": "<u1", "<u1": "<u1", "<f2": "<f2", "|b1": "<u1",
    "=f4": "<f4", "=f8": "<f8", "=i4": "<i4", "=i8": "<i8", "=u4": "<u4",
}


# ------------------------------------------------------------------ header


def _header(descr, fortran_order, shape):
    """The complete NPY v1.0 prefix (magic, version, length, padded dict)."""
    text = "{'descr': %s, 'fortran_order': %s, 'shape': %s, }" % (
        repr(descr), repr(bool(fortran_order)), repr(tuple(int(s) for s in shape))
    )
    body = text.encode("latin1")
    hlen = len(body) + 1  # the trailing newline
    padlen = _ARRAY_ALIGN - ((_MAGIC_LEN + 2 + hlen) % _ARRAY_ALIGN)
    total = hlen + padlen
    if total >= 1 << 16:
        # numpy would switch to version 2.0 here; a model header never
        # comes near this and the writer refuses rather than diverge.
        raise ValueError("mojolearn: npy header too long for format 1.0")
    return MAGIC + b"\x01\x00" + struct.pack("<H", total) + body + b" " * padlen + b"\n"


def _parse_header(data):
    """`(descr, fortran_order, shape, data_offset)` from the start of an
    npy byte string. Versions 1.0, 2.0 and 3.0 are read."""
    if data[:6] != MAGIC:
        raise ValueError("mojolearn: not an npy member (bad magic)")
    major, minor = data[6], data[7]
    if (major, minor) == (1, 0):
        (hlen,) = struct.unpack("<H", data[8:10])
        start = 10
        enc = "latin1"
    elif (major, minor) in ((2, 0), (3, 0)):
        (hlen,) = struct.unpack("<I", data[8:12])
        start = 12
        enc = "latin1" if major == 2 else "utf8"
    else:
        raise ValueError(f"mojolearn: unsupported npy version {major}.{minor}")
    text = data[start:start + hlen].decode(enc)
    try:
        d = ast.literal_eval(text)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"mojolearn: cannot parse npy header {text!r}: {e}") from None
    if not isinstance(d, dict) or set(d) != {"descr", "fortran_order", "shape"}:
        raise ValueError(f"mojolearn: malformed npy header {text!r}")
    descr, fortran, shape = d["descr"], d["fortran_order"], d["shape"]
    if not isinstance(descr, str):
        raise ValueError("mojolearn: structured npy dtypes are not supported")
    if not isinstance(fortran, bool) or not isinstance(shape, tuple):
        raise ValueError(f"mojolearn: malformed npy header {text!r}")
    return descr, fortran, tuple(int(s) for s in shape), start + hlen


# ----------------------------------------------------------------- encode


def _utf32(strings, width):
    out = bytearray()
    for s in strings:
        b = s.encode("utf-32-le")
        out += b + b"\x00" * (4 * width - len(b))
    return bytes(out)


def _describe(value, *, c_order):
    """`(descr, fortran_order, shape, raw)` for anything `write_npz`
    accepts: an Array, a buffer-protocol object (a NumPy array when the
    caller has NumPy), a str / bytes / int / float / bool scalar, or a
    (nested) list of them. With `c_order` the bytes are linearized to C
    order and a 0-d value becomes shape `(1,)`."""
    if isinstance(value, Array):
        a = value._as_c() if c_order else value
        shape = a.shape if (a.ndim or not c_order) else (1,)
        descr = _DESCR_OF.get(a.dtype, a.dtype)
        return descr, a.order == "F" and a.ndim > 1, shape, a.tobytes()
    if isinstance(value, bool):
        return "|b1", False, (1,) if c_order else (), bytes([1 if value else 0])
    if isinstance(value, int):
        return "<i8", False, (1,) if c_order else (), struct.pack("<q", value)
    if isinstance(value, float):
        return "<f8", False, (1,) if c_order else (), struct.pack("<d", value)
    if isinstance(value, str):
        n = max(len(value), 1)
        return f"<U{n}", False, (1,) if c_order else (), _utf32([value], n)
    if isinstance(value, bytes):
        n = max(len(value), 1)
        return f"|S{n}", False, (1,) if c_order else (), value.ljust(n, b"\x00")
    if isinstance(value, (list, tuple)):
        return _describe_list(value)
    with _buffer.view(value, name="array") as b:
        descr = _buffer.typestr_of(b)
        descr = _DESCR_OF.get(descr, descr)
        shape = b.shape if b.ndim else ()
        if c_order:
            fortran = False
            raw = memoryview(value).tobytes()  # C order for any strides
            if not shape:
                shape = (1,)
        else:
            if b.c_contiguous:
                fortran = False
                raw = memoryview(value).tobytes()
            elif b.f_contiguous:
                fortran = True
                raw = memoryview(value).tobytes(order="F")
            else:
                fortran = False
                raw = memoryview(value).tobytes()
        return descr, fortran, shape, raw


def _describe_list(value):
    from ._array import _flatten
    shape, flat = _flatten(value)
    if not flat:
        return "<f8", False, shape, b""
    if all(isinstance(v, str) for v in flat):
        n = max(max(len(s) for s in flat), 1)
        return f"<U{n}", False, shape, _utf32(flat, n)
    if all(isinstance(v, bytes) for v in flat):
        n = max(max(len(s) for s in flat), 1)
        return f"|S{n}", False, shape, b"".join(s.ljust(n, b"\x00") for s in flat)
    if all(isinstance(v, bool) for v in flat):
        return "|b1", False, shape, bytes(1 if v else 0 for v in flat)
    if any(isinstance(v, float) for v in flat):
        return "<f8", False, shape, struct.pack("<%dd" % len(flat), *flat)
    if all(isinstance(v, int) for v in flat):
        return "<i8", False, shape, struct.pack("<%dq" % len(flat), *flat)
    raise TypeError("mojolearn: cannot serialize a list of mixed kinds")


def encode_npy(value, *, c_order=False):
    """The complete npy bytes for `value`: what `numpy.lib.format.write_array`
    would write, including `fortran_order: True` for an F-order block
    unless `c_order` asks for the 0.6.x `write_npz` linearization."""
    descr, fortran, shape, raw = _describe(value, c_order=c_order)
    return _header(descr, fortran, shape) + raw


def write_npy(fp, value, *, c_order=False):
    fp.write(encode_npy(value, c_order=c_order))


# ----------------------------------------------------------------- decode


def _itemsize_of(descr):
    if descr in _TYPESTR_OF:
        return _ITEMSIZE[_TYPESTR_OF[descr]]
    if descr[:2] in ("<U", "=U") and descr[2:].isdigit():
        return 4 * int(descr[2:])
    if descr[:2] == "|S" and descr[2:].isdigit():
        return int(descr[2:])
    raise ValueError(f"mojolearn: unsupported npy dtype {descr!r}")


def decode_npy(data):
    """An Array (numeric), str / list of str (`'<U'`), or bytes / list of
    bytes (`'|S'`) from complete npy bytes."""
    descr, fortran, shape, offset = _parse_header(data)
    itemsize = _itemsize_of(descr)
    n = _prod(shape)
    raw = data[offset:offset + n * itemsize]
    if len(raw) != n * itemsize:
        raise ValueError(
            f"mojolearn: npy member holds {len(raw)} data bytes, shape "
            f"{shape} of {descr!r} needs {n * itemsize}"
        )
    if descr in _TYPESTR_OF:
        a = _buffer.frombytes(raw, _TYPESTR_OF[descr], shape)
        if fortran and a.ndim > 1:
            a._set_meta(shape, a.dtype, "F")
        return a
    if descr[1] == "U":
        width = itemsize
        items = [raw[i * width:(i + 1) * width].decode("utf-32-le").rstrip("\x00")
                 for i in range(n)]
    else:
        width = itemsize
        items = [raw[i * width:(i + 1) * width].rstrip(b"\x00") for i in range(n)]
    if fortran and len(shape) > 1:
        # column-major text blocks never occur in model files; linearize
        # through an index map rather than mis-shape them
        idx = list(range(n))
        tmp = Array.from_list(idx, "<i8")
        tmp._set_meta(shape, "<i8", "F")
        items = [items[i] for i in tmp._as_c()._values()]
    if n == 1:
        return items[0]
    from ._array import _nest
    return _nest(items, shape) if shape else items[0]


def read_npy(fp):
    return decode_npy(fp.read())


# --------------------------------------------------------------------- npz


def write_npz(path, arrays):
    """Write `arrays` (a dict of name to array-like) to `path` as an
    uncompressed npz whose bytes depend only on the array contents."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            payload = encode_npy(arrays[name], c_order=True)
            info = zipfile.ZipInfo(
                name + ".npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload)
    return path


def read_npz(path, expected_format):
    """Load an npz written by `write_npz` (or `np.savez`) into a dict,
    checking its `format` tag against `expected_format`. Pickle never
    enters: every member is decoded by this module's own codec. See the
    module docstring for what each member kind comes back as."""
    out = {}
    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".npy"):
                continue
            out[member[:-4]] = decode_npy(zf.read(member))
    tag = scalar_str(out, "format") if "format" in out else ""
    if tag != expected_format:
        raise ValueError(
            f"mojolearn: {path!r} holds model format {tag!r}, this loader "
            f"reads {expected_format!r}"
        )
    return out


def _first(value, name):
    """The first element of a member in flat order, as the old
    `np.asarray(v).reshape(-1)[0]` gave it."""
    if isinstance(value, Array):
        if value.size == 0:
            raise ValueError(f"mojolearn: model field {name!r} is empty")
        return value.ravel()[0]
    while isinstance(value, list):
        if not value:
            raise ValueError(f"mojolearn: model field {name!r} is empty")
        value = value[0]
    return value


def scalar_str(arrays, name):
    """A string field as a plain str. Scalars are stored as one-element
    arrays because the npy writer promotes 0-d to 1-d."""
    if name not in arrays:
        raise ValueError(f"mojolearn: model file is missing field {name!r}")
    return str(_first(arrays[name], name))


def _dtype_name(value):
    if isinstance(value, Array):
        return value.dtype
    if isinstance(value, str):
        return "<U"
    if isinstance(value, bytes):
        return "|S"
    if isinstance(value, list):
        inner = value
        while isinstance(inner, list) and inner:
            inner = inner[0]
        return "<U" if isinstance(inner, str) else "|S" if isinstance(inner, bytes) else "list"
    return type(value).__name__


def exact(arrays, name, dtype):
    """`arrays[name]` with its dtype REQUIRED to match, never cast. A cast
    on load could silently change bits, which is the one failure a model
    file must not have. `dtype` is a typestr (`'<i4'`), a NumPy dtype or
    scalar type when the caller has NumPy, or `float` / `int`. Returns a
    C-order Array."""
    if name not in arrays:
        raise ValueError(f"mojolearn: model file is missing field {name!r}")
    want = normalize_dtype(dtype)
    a = arrays[name]
    if not isinstance(a, Array) or a.dtype != want:
        raise ValueError(
            f"mojolearn: model field {name!r} has dtype {_dtype_name(a)}, the "
            f"format stores {want}; refusing to cast a model file"
        )
    return a._as_c()
