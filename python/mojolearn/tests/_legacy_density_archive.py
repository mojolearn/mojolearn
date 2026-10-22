# SPDX-License-Identifier: Apache-2.0
"""Frozen 30841c221 deterministic NPZ writer, used only as an independent test oracle."""
import struct
import zipfile
from mojolearn._array import Array
from mojolearn import _buffer

MAGIC = b"\x93NUMPY"
_ARRAY_ALIGN = 64
_MAGIC_LEN = len(MAGIC) + 2
_DESCR_OF = {"<u1": "|u1"}

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
        raw = a._mv.cast("B") if a.nbytes else b""
        return descr, a.order == "F" and a.ndim > 1, shape, raw
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
            view = memoryview(value)
            # A byte view borrows an already C-contiguous payload. Strided or
            # Fortran inputs still need the established C-order linearization.
            raw = (view.cast("B") if b.nbytes else b"") if view.c_contiguous \
                else view.tobytes()
            if not shape:
                shape = (1,)
        else:
            if b.c_contiguous:
                fortran = False
                raw = memoryview(value).cast("B") if b.nbytes else b""
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

def write_npy(fp, value, *, c_order=False):
    # Keep the header and the model-sized payload as separate writes.  Joining
    # them would allocate and copy a second complete numeric payload merely to
    # hand it to a file object.
    descr, fortran, shape, raw = _describe(value, c_order=c_order)
    fp.write(_header(descr, fortran, shape))
    fp.write(raw)

def write_npz(path, arrays):
    """Write `arrays` (a dict of name to array-like) to `path` as an
    uncompressed npz whose bytes depend only on the array contents."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(
                name + ".npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            # ZipFile.open() produces the same seekable ZIP_STORED bytes as
            # writestr(), while letting write_npy avoid a header+payload join.
            with zf.open(info, "w") as fp:
                write_npy(fp, arrays[name], c_order=True)
    return path


# Frozen 30841c221 public density save schemas. Canonical names are the
# intentional roundtrip repair; canonical=False preserves the original bytes.
from mojolearn import _backend
from mojolearn.tests._legacy_host_iforest_fit import parameters as _iforest_parameters

def _saved_mode(est):
    """The tier `predict` would run on now, persisted as GradientBoosting
    persists it, so a loaded model never silently changes tier."""
    mode = getattr(est, "numeric_mode", None) or _backend.default_mode()
    if not isinstance(mode, str) or mode.strip().lower() not in (
        "fast", "identical"
    ):
        raise ValueError(f"mojolearn: cannot save invalid numeric_mode {mode!r}")
    return mode.strip().lower()

def save_kde(self, path, *, canonical=False):
    """Save the fitted model to ``path`` as an ``.npz`` file.

    Stores the float32 training data (the whole fitted state of a
    brute-force KDE), the weights if any, ``bandwidth``, ``kernel``,
    ``metric`` and the fitted sizes. ``mojolearn.host_model(path)``
    scores from the file on a machine without a GPU. An unfitted model
    raises ``RuntimeError``.

    Parameters
    ----------
    path : str or path-like
    """
    if not hasattr(self, "_x"):
        raise RuntimeError("this estimator is not fitted yet")
    arrays = {
        "format": 'mojolearn-kde-1',
        "estimator": ('KernelDensity' if canonical else type(self).__name__),
        "numeric_mode": _saved_mode(self),
        "kernel": str(self.kernel),
        "metric": str(self.metric),
        "x": self._x,
        "bandwidth": Array.from_list([float(self.bandwidth)], "<f8"),
        "meta": Array.from_list(
            [int(self.n_features_in_), int(self.n_samples_fit_),
             1 if self._w is not None else 0],
            "<i8",
        ),
    }
    if self._w is not None:
        arrays["weights"] = self._w
    return write_npz(path, arrays)

def save_iforest(self, path, *, canonical=False):
    """Write the fitted model to `path` as an npz file.

    The file holds the training matrix and the resolved parameters the
    forest is built from, so it is as large as the training data.
    """
    if not hasattr(self, "_x"):
        raise RuntimeError("this estimator is not fitted yet")
    if "_seed" not in self.__dict__:  # an instance assembled without fit or load
        _iforest_parameters(self)
    return write_npz(path, {
        "format": 'mojolearn-iforest-1',
        "estimator": ('IsolationForest' if canonical else type(self).__name__),
        "numeric_mode": _saved_mode(self),
        "x": self._x,
        "ints": Array.from_list([
            int(self._n_estimators), self._max_samples_mode, self._max_samples_int,
            self._max_depth, self._max_features_mode, self._max_features_int,
            1 if self._bootstrap else 0, self._seed, 1 if self._contamination_auto else 0,
            int(self.n_features_in_), int(self.max_samples_),
        ], "<i8"),
        "reals": Array.from_list([
            float(self._max_samples_frac), float(self._max_features_frac),
            float(self._contamination), float(self.offset_),
        ], "<f8"),
    })
