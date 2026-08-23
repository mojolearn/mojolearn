"""Deterministic npz save and load for fitted models.

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
"""

import io
import zipfile

import numpy as np
from numpy.lib import format as _npy_format


def write_npz(path, arrays):
    """Write `arrays` (a dict of name to array-like) to `path` as an
    uncompressed npz whose bytes depend only on the array contents."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            _npy_format.write_array(
                buf,
                np.ascontiguousarray(np.asarray(arrays[name])),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                name + ".npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())
    return path


def read_npz(path, expected_format):
    """Load an npz written by `write_npz` into a dict of arrays, checking
    its `format` tag against `expected_format`. Pickle stays off; every
    member is a plain array."""
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for name in z.files:
            out[name] = z[name]
    tag = scalar_str(out, "format") if "format" in out else ""
    if tag != expected_format:
        raise ValueError(
            f"mojolearn: {path!r} holds model format {tag!r}, this loader "
            f"reads {expected_format!r}"
        )
    return out


def scalar_str(arrays, name):
    """A string field as a plain str. Scalars are stored as one-element
    arrays because the npy writer promotes 0-d to 1-d."""
    if name not in arrays:
        raise ValueError(f"mojolearn: model file is missing field {name!r}")
    return str(np.asarray(arrays[name]).reshape(-1)[0])


def exact(arrays, name, dtype):
    """`arrays[name]` with its dtype REQUIRED to match, never cast. A cast
    on load could silently change bits, which is the one failure a model
    file must not have."""
    if name not in arrays:
        raise ValueError(f"mojolearn: model file is missing field {name!r}")
    a = arrays[name]
    if a.dtype != np.dtype(dtype):
        raise ValueError(
            f"mojolearn: model field {name!r} has dtype {a.dtype}, the "
            f"format stores {np.dtype(dtype)}; refusing to cast a model file"
        )
    return np.ascontiguousarray(a)
