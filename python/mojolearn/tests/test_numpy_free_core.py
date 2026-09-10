# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gates for the NumPy-free core: `_array.Array`, `_buffer`, `_serialize`.

NumPy is used HERE, from the `test` extra, as the oracle: the library must
not import it, and this file proves the library's conversions and file
bytes agree with it bit for bit (DEVIATION 2310). Runs under pytest or as
a plain program:

    cd python && python3 -m mojolearn.tests.test_numpy_free_core

Nothing in this file touches a GPU binding: `all_finite` falls back to its
host path when the base extension is not built, and that path is the one
gated.
"""

import array
import io
import os
import struct
import sys
import tempfile
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from mojolearn import _arrays, _buffer, _serialize
from mojolearn._array import Array


# ----------------------------------------------------------------- helpers


def _assert_same_bytes(got, want, what):
    assert bytes(got) == bytes(want), what


def _f32_boundary_block(rng, n):
    """float64 values that stress a float64 -> float32 cast: random
    doubles across the float32 range, exact float32 midpoints (ties, which
    round-to-nearest-EVEN decides), the midpoints nudged by one float64
    ulp either way, subnormal float32 values and their halves, the largest
    finite float32 and just above it (overflow to inf), NaN with a payload,
    both infinities and both zeros."""
    out = []
    out.append(rng.standard_normal(n) * np.exp(rng.uniform(-40, 40, n)))
    f32 = rng.standard_normal(n).astype(np.float32)
    nxt = np.nextafter(f32, np.float32(np.inf)).astype(np.float64)
    mid = (f32.astype(np.float64) + nxt) / 2.0            # exact ties
    out.append(mid)
    out.append(np.nextafter(mid, np.inf))                   # just above
    out.append(np.nextafter(mid, -np.inf))                  # just below
    sub = rng.integers(1, 1 << 23, n).astype(np.uint32).view(np.float32).astype(np.float64)
    out.append(sub)                                        # subnormals
    out.append(sub / 2.0)                                  # sub-subnormal ties
    out.append(sub * 1.5)
    big = np.float64(np.finfo(np.float32).max)
    out.append(np.array([big, np.nextafter(big, np.inf), big * 1.0000001, -big * 2,
                         np.float64(np.finfo(np.float32).tiny) / 3, 0.0, -0.0,
                         np.inf, -np.inf, np.nan, -np.nan,
                         np.array([0x7FF4_0000_DEAD_BEEF], np.uint64).view(np.float64)[0]]))
    return np.concatenate(out)


# --------------------------------------------------------------- zero-copy


def test_array_interface_is_zero_copy():
    a = _buffer.zeros((3, 2), "<f4")
    v = np.asarray(a)
    assert v.dtype == np.float32 and v.shape == (3, 2)
    assert v.__array_interface__["data"][0] == a.__array_interface__["data"][0]
    assert np.shares_memory(v, np.asarray(a))
    v[1, 1] = 7.5
    assert a[1, 1] == 7.5, "a write through the NumPy view lands in the Array"
    assert a.tolist() == [[0.0, 0.0], [0.0, 7.5], [0.0, 0.0]]
    # F-order block: NumPy views it with its real strides, still no copy
    f, copied = _buffer.as_f32_colmajor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name="X")
    assert copied and f.order == "F"
    vf = np.asarray(f)
    assert vf.flags["F_CONTIGUOUS"] and not vf.flags["C_CONTIGUOUS"]
    assert vf.__array_interface__["data"][0] == f.__array_interface__["data"][0]
    assert vf.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    # the address does not move over the lifetime
    addr0 = _buffer.addr(a, name="a")
    for _ in range(3):
        _ = a.astype("<f8")
        _ = a.reshape((6,))
    assert _buffer.addr(a, name="a") == addr0


def test_from_buffer_is_a_view_over_numpy():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = Array.from_buffer(x)
    assert a.shape == (3, 4) and a.dtype == "<f4" and a.order == "C"
    assert _buffer.addr_ro(a, name="a") == x.ctypes.data
    x[2, 3] = -1.0
    assert a[2, 3] == -1.0
    xf = np.asfortranarray(x)
    af = Array.from_buffer(xf)
    assert af.order == "F" and af.tolist() == x.tolist()
    assert af._flat().tolist() == xf.T.reshape(-1).tolist()
    try:
        Array.from_buffer(x[:, ::2])
        raise AssertionError("a non-contiguous view must be refused, not copied")
    except ValueError as e:
        assert "contiguous" in str(e)


# ------------------------------------------------------ f64 -> f32 casting


def test_f64_to_f32_matches_numpy_bit_for_bit():
    rng = np.random.default_rng(2300)
    x = _f32_boundary_block(rng, 4096)
    x = x[: x.size - x.size % 16]
    assert x.size > 20000
    with np.errstate(over="ignore", invalid="ignore"):  # inf/nan are the point
        want = x.astype(np.float32)
    for shape in ((x.size,), (x.size // 8, 8), (-1, 16)):
        src = x.reshape(shape)
        got, copied = _buffer.as_f32_c(src, ndim=src.ndim, name="X")
        assert copied and got.dtype == "<f4" and got.shape == src.shape
        _assert_same_bytes(got.tobytes(), want.reshape(shape).tobytes(),
                           f"f64->f32 cast, shape {src.shape}")
    # a Python list goes through the same double, so the same single rounding
    got, _ = _buffer.as_f32_c(x[:1000].tolist(), ndim=1, name="X")
    _assert_same_bytes(got.tobytes(), want[:1000].tobytes(), "list -> f32")
    # Array.astype is the same cast
    a64 = Array.from_buffer(x)
    _assert_same_bytes(a64.astype("<f4").tobytes(), want.tobytes(), "Array.astype")
    # widening back is exact
    _assert_same_bytes(Array.from_buffer(want).astype("<f8").tobytes(),
                       want.astype(np.float64).tobytes(), "f32->f64 widening")


def test_int_to_f32_matches_numpy():
    rng = np.random.default_rng(2301)
    i32 = rng.integers(-(1 << 31), 1 << 31, 5000, dtype=np.int32)
    got, _ = _buffer.as_f32_c(i32, ndim=1, name="y")
    _assert_same_bytes(got.tobytes(), i32.astype(np.float32).tobytes(), "i32->f32")
    i64 = rng.integers(-(1 << 53), 1 << 53, 5000, dtype=np.int64)
    got, _ = _buffer.as_f32_c(i64, ndim=1, name="y")
    _assert_same_bytes(got.tobytes(), i64.astype(np.float32).tobytes(), "i64->f32 within 2**53")
    got, _ = _buffer.as_f64_c(i64, ndim=1, name="y")
    _assert_same_bytes(got.tobytes(), i64.astype(np.float64).tobytes(), "i64->f64")
    got, _ = _buffer.as_i32_c([1, 2, 3], ndim=1, name="y")
    assert got.dtype == "<i4" and got.tolist() == [1, 2, 3]
    got, _ = _buffer.as_i64_c(i32, ndim=1, name="y")
    _assert_same_bytes(got.tobytes(), i32.astype(np.int64).tobytes(), "i32->i64")
    # float labels -> int truncate toward zero as NumPy does
    f = np.array([1.9, -1.9, 0.0, 2.0], np.float32)
    got, _ = _buffer.as_i32_c(f, ndim=1, name="y")
    assert got.tolist() == f.astype(np.int32).tolist()


def test_colmajor_matches_asfortranarray():
    rng = np.random.default_rng(2302)
    for rows, cols in ((7, 3), (1, 5), (64, 1), (33, 17), (2, 2)):
        x = _f32_boundary_block(rng, 128)[: rows * cols].reshape(rows, cols)
        want = np.asfortranarray(x, dtype=np.float32)
        for src, label in ((x, "C f64"), (np.asfortranarray(x), "F f64"),
                           (x.astype(np.float32), "C f32"),
                           (x[:, ::-1][:, ::-1], "view f64"),
                           (x.tolist(), "list")):
            got, copied = _buffer.as_f32_colmajor(src, name="X")
            # a float32 block that is already F-contiguous (which a (1, n)
            # or (n, 1) C-order block also is) is borrowed, not copied
            expect_copy = not (isinstance(src, np.ndarray) and src.dtype == np.float32
                               and src.flags["F_CONTIGUOUS"])
            assert copied == expect_copy, (label, rows, cols)
            assert got.order == "F" and got.shape == (rows, cols)
            _assert_same_bytes(got.tobytes(), want.tobytes(order="F"),
                               f"colmajor bytes, {label}, {rows}x{cols}")
            assert np.array_equal(np.asarray(got), want, equal_nan=True), label
        # an F-order float32 input is a zero-copy borrow
        got, copied = _buffer.as_f32_colmajor(want, name="X")
        assert not copied and _buffer.addr_ro(got, name="X") == want.ctypes.data
        # the shim's flat is the column-major flat over the same buffer
        a, flat, copied = _arrays.as_f32_colmajor(x, "X")
        assert flat.shape == (rows * cols,)
        assert _buffer.addr_ro(flat, name="f") == _buffer.addr_ro(a, name="a")
        for f_ in range(cols):
            for r in range(rows):
                assert flat[f_ * rows + r] == a[r, f_]


def test_c_order_from_any_layout_matches_ascontiguousarray():
    rng = np.random.default_rng(2303)
    x = rng.standard_normal((9, 5))
    cases = {
        "F f64": np.asfortranarray(x),
        "strided f64": np.ascontiguousarray(np.repeat(x, 2, axis=1))[:, ::2],
        "reversed rows": x[::-1],
        "F f32": np.asfortranarray(x, dtype=np.float32),
        "C f32": x.astype(np.float32),
    }
    for label, src in cases.items():
        want = np.ascontiguousarray(src, dtype=np.float32)
        got, copied = _buffer.as_f32_c(src, ndim=2, name="X")
        assert got.order == "C" and got.shape == (9, 5)
        _assert_same_bytes(got.tobytes(), want.tobytes(), label)
        assert copied == (label != "C f32"), label
    a3 = rng.standard_normal((2, 3, 4))
    got, _ = _buffer.as_f32_c(np.asfortranarray(a3), ndim=3, name="X")
    _assert_same_bytes(got.tobytes(), a3.astype(np.float32).tobytes(), "3-D F -> C")


# ------------------------------------------------------------- NPY codec


_CODEC_DTYPES = (np.float32, np.int32, np.int64, np.float64)
_CODEC_SHAPES = ((3,), (1,), (2, 3), (5, 7), (4, 1, 2), (0, 3), (3, 0), (2, 2, 2, 2))


def _numpy_npy(a):
    buf = io.BytesIO()
    npy_format.write_array(buf, a, allow_pickle=False)
    return buf.getvalue()


def test_npy_bytes_identical_to_numpy():
    rng = np.random.default_rng(2304)
    for dt in _CODEC_DTYPES:
        for shape in _CODEC_SHAPES:
            base = rng.standard_normal(shape) * 1000
            for order in ("C", "F"):
                x = np.array(base, dtype=dt, order=order)
                want = _numpy_npy(x)
                got_np = _serialize.encode_npy(x)
                _assert_same_bytes(got_np, want, f"encode_npy(ndarray {dt.__name__} {shape} {order})")
                got_arr = _serialize.encode_npy(Array.from_buffer(x))
                _assert_same_bytes(got_arr, want, f"encode_npy(Array {dt.__name__} {shape} {order})")
                back = np.load(io.BytesIO(got_arr))
                assert back.dtype == dt and back.shape == shape
                assert np.array_equal(back, x)
                mine = _serialize.decode_npy(got_arr)
                assert mine.shape == shape and np.asarray(mine).dtype == dt
                assert np.array_equal(np.asarray(mine), x), "decode_npy round trip"
                if x.ndim > 1 and x.size:
                    assert mine.order == order
    # header layout: 64-byte aligned, ends in newline, v1.0
    h = _serialize.encode_npy(np.zeros((2, 3), np.float32))
    assert h[:8] == b"\x93NUMPY\x01\x00"
    (hlen,) = struct.unpack("<H", h[8:10])
    assert (10 + hlen) % 64 == 0 and h[10 + hlen - 1:10 + hlen] == b"\n"
    assert h[10:10 + hlen].rstrip() == b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"


def test_npy_scalars_and_strings_match_numpy():
    for value, npval in ((3, np.asarray(3)), (2.5, np.asarray(2.5)),
                         (True, np.asarray(True)), ("identical", np.asarray("identical")),
                         ("", np.asarray("")), (b"identical", np.asarray(b"identical")),
                         (["a", "bcd"], np.asarray(["a", "bcd"])),
                         ([1, 2, 3], np.asarray([1, 2, 3])), ([1.0, 2], np.asarray([1.0, 2])),
                         ([[1, 2], [3, 4]], np.asarray([[1, 2], [3, 4]])),
                         ([], np.asarray([]))):
        want = _numpy_npy(np.ascontiguousarray(npval))
        got = _serialize.encode_npy(value, c_order=True)
        _assert_same_bytes(got, want, f"scalar/list {value!r}")
        _assert_same_bytes(_serialize.encode_npy(npval, c_order=True), want, f"ndarray {value!r}")
    assert _serialize.decode_npy(_serialize.encode_npy("héllo€")) == "héllo€"
    assert _serialize.decode_npy(_serialize.encode_npy(["x", "yz"])) == ["x", "yz"]
    assert _serialize.decode_npy(_serialize.encode_npy(b"abc")) == b"abc"
    assert _serialize.decode_npy(_serialize.encode_npy(np.asarray([], dtype="U1"))) == []
    b = _serialize.decode_npy(_serialize.encode_npy(np.asarray([True, False])))
    assert b.dtype == "<u1" and b.tolist() == [1, 0]


def _old_write_npz(path, arrays):
    """The 0.6.x writer, verbatim, as the oracle for file bytes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            npy_format.write_array(buf, np.ascontiguousarray(np.asarray(arrays[name])),
                                   allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())
    return path


def test_npz_files_identical_to_0_6_and_load_both_ways():
    rng = np.random.default_rng(2305)
    model = {
        "format": np.asarray("mojolearn-extratrees-1"),
        "estimator": np.asarray("ExtraTreesClassifier"),
        "device": np.asarray("metal"),
        "offsets": rng.integers(0, 100, 17, dtype=np.int32),
        "colid": rng.integers(0, 8, 40, dtype=np.int32),
        "quesval": rng.standard_normal(40).astype(np.float32),
        "left_child": rng.integers(-1, 40, 40, dtype=np.int32),
        "leaves": rng.standard_normal((40, 3)).astype(np.float32),
        "meta": np.asarray([8, 4, 3, 1, 12, 2], dtype=np.int64),
        "bias": np.asarray(0.125, dtype=np.float64),
        "model": np.frombuffer(b"trees 1\nleaf 0 0 x/3f800000\n", dtype=np.uint8),
        "flag": np.asarray(True),
    }
    with tempfile.TemporaryDirectory() as d:
        old = os.path.join(d, "old.npz")
        new = os.path.join(d, "new.npz")
        new2 = os.path.join(d, "new2.npz")
        _old_write_npz(old, model)
        _serialize.write_npz(new, model)  # ndarrays through the buffer protocol
        with open(old, "rb") as f1, open(new, "rb") as f2:
            _assert_same_bytes(f2.read(), f1.read(), "0.7 writer == 0.6 writer, ndarray members")
        # 0.6 file -> 0.7 reader
        arrays = _serialize.read_npz(old, "mojolearn-extratrees-1")
        assert _serialize.scalar_str(arrays, "estimator") == "ExtraTreesClassifier"
        assert arrays["format"] == "mojolearn-extratrees-1"
        for name in ("offsets", "colid", "quesval", "left_child", "leaves", "meta", "model"):
            a = _serialize.exact(arrays, name, model[name].dtype)
            assert isinstance(a, Array) and a.shape == model[name].shape
            _assert_same_bytes(a.tobytes(), model[name].tobytes(), name)
        bias = _serialize.exact(arrays, "bias", np.float64)
        assert bias.shape == (1,) and bias[0] == 0.125
        assert arrays["flag"].dtype == "<u1" and arrays["flag"].tolist() == [1]
        # 0.7 Arrays -> file -> np.load (the 0.6 reader) and back to identical bytes
        _serialize.write_npz(new2, arrays)
        with open(old, "rb") as f1, open(new2, "rb") as f2:
            old_bytes, new_bytes = f1.read(), f2.read()
        # the bool member is the one documented difference: it loads as uint8
        assert len(old_bytes) == len(new_bytes)
        with np.load(new2, allow_pickle=False) as z:
            for name in model:
                back = z[name]
                if name == "flag":
                    assert back.dtype == np.uint8 and back.tolist() == [1]
                    continue
                assert back.dtype == np.ascontiguousarray(model[name]).dtype, name
                assert np.array_equal(back, np.ascontiguousarray(model[name]).reshape(back.shape)), name
        # two saves of the same arrays are the same bytes
        _serialize.write_npz(new, arrays)
        with open(new, "rb") as f1, open(new2, "rb") as f2:
            _assert_same_bytes(f1.read(), f2.read(), "deterministic file bytes")
        # refusals keep their messages
        try:
            _serialize.exact(arrays, "quesval", np.int32)
            raise AssertionError("dtype mismatch must be refused")
        except ValueError as e:
            assert "refusing to cast a model file" in str(e)
        try:
            _serialize.exact(arrays, "nope", np.int32)
            raise AssertionError("missing field must be refused")
        except ValueError as e:
            assert "missing field 'nope'" in str(e)
        try:
            _serialize.read_npz(old, "other-format")
            raise AssertionError("format mismatch must be refused")
        except ValueError as e:
            assert "holds model format" in str(e)
        # the corrupt-mode probes test_gbdt_mode_serialization writes
        for bad in (np.asarray("unknown"), np.asarray([]), np.asarray(["identical", "fast"]),
                    np.asarray([], dtype="U1"), np.asarray(1), np.asarray(b"identical")):
            _serialize.write_npz(new, dict(arrays, numeric_mode=bad))
            got = _serialize.read_npz(new, "mojolearn-extratrees-1")["numeric_mode"]
            with np.load(new, allow_pickle=False) as z:
                theirs = z["numeric_mode"]
            assert theirs.dtype == np.ascontiguousarray(bad).dtype
            if bad.dtype.kind == "U" and bad.size == 1:
                assert got == str(bad.reshape(-1)[0])
            elif bad.dtype.kind in "US":
                assert isinstance(got, (list, bytes))
            else:
                assert isinstance(got, Array)


# ------------------------------------------------------ PyObject_GetBuffer


def test_getbuffer_paths():
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    # bytes: read-only; addr() refuses, addr_ro() works
    with _buffer.view(raw) as b:
        assert b.readonly and b.format == "B" and b.shape == (16,) and b.c_contiguous
        assert b.addr != 0 and b.nbytes == 16 and b.itemsize == 1
    assert _buffer.addr_ro(raw, name="raw") != 0
    try:
        _buffer.addr(raw, name="out")
        raise AssertionError("a read-only buffer must be refused")
    except ValueError as e:
        assert "read-only" in str(e)
    # the shim keeps the old message word for word
    try:
        _arrays._addr(raw)
        raise AssertionError("shim must refuse read-only")
    except ValueError as e:
        assert str(e) == "mojolearn: output buffer is read-only, refusing to write to it"
    # bytearray: writable
    ba = bytearray(raw)
    assert _buffer.addr(ba, name="out") != 0
    with _buffer.view(ba, writable=True) as b:
        assert not b.readonly
    # array.array: format and itemsize
    aa = array.array("f", [1.0, 2.0, 3.0, 4.0])
    with _buffer.view(aa) as b:
        assert b.format == "f" and b.itemsize == 4 and b.shape == (4,)
        assert b.addr == aa.buffer_info()[0]
    got, copied = _buffer.as_f32_c(aa, ndim=1, name="v")
    assert not copied and _buffer.addr_ro(got, name="v") == aa.buffer_info()[0]
    got, copied = _buffer.as_f32_c(array.array("d", [1.5, 2.5]), ndim=1, name="v")
    assert copied and got.tolist() == [1.5, 2.5]
    # memoryview, including a strided one
    mv = memoryview(aa)
    with _buffer.view(mv) as b:
        assert b.addr == aa.buffer_info()[0] and b.c_contiguous
    got, copied = _buffer.as_f32_c(mv[::2], ndim=1, name="v")
    assert copied and got.tolist() == [1.0, 3.0]
    # numpy: C, F, non-contiguous (copied), read-only
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    with _buffer.view(x) as b:
        assert b.c_contiguous and not b.f_contiguous and b.strides == (12, 4)
    got, copied = _buffer.as_f32_c(x, ndim=2, name="X")
    assert not copied and _buffer.addr_ro(got, name="X") == x.ctypes.data
    xf = np.asfortranarray(x)
    with _buffer.view(xf) as b:
        assert b.f_contiguous and not b.c_contiguous and b.strides == (4, 8)
    got, copied = _buffer.as_f32_c(xf, ndim=2, name="X")
    assert copied and got.tolist() == x.tolist()
    xs = x[:, ::2]
    with _buffer.view(xs) as b:
        assert not b.c_contiguous and not b.f_contiguous
    got, copied = _buffer.as_f32_c(xs, ndim=2, name="X")
    assert copied and got.tolist() == xs.tolist()
    ro = np.zeros(3, np.float32)
    ro.flags.writeable = False
    assert _buffer.addr_ro(ro, name="X") == ro.ctypes.data
    try:
        _buffer.addr(ro, name="out")
        raise AssertionError("read-only ndarray must be refused")
    except ValueError as e:
        assert "read-only" in str(e)
    # dtype width from itemsize, not the letter: int64 exports 'l' on LP64
    with _buffer.view(np.zeros(2, np.int64)) as b:
        assert _buffer.typestr_of(b) == "<i8"
    with _buffer.view(np.zeros(2, np.uint8)) as b:
        assert _buffer.typestr_of(b) == "<u1"
    with _buffer.view(np.asarray(["ab"])) as b:
        assert _buffer.typestr_of(b) == "<U2"
    # no buffer protocol at all
    for bad in ([1, 2], 3, None, object()):
        try:
            _buffer.view(bad, name="X")
            raise AssertionError(f"{bad!r} must be refused")
        except TypeError as e:
            assert str(e) == "mojolearn: X does not support the buffer protocol"
    # a list is converted, not refused, by the converters
    got, copied = _buffer.as_f32_c([[1, 2], [3, 4]], ndim=2, name="X")
    assert copied and got.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    # ndim and emptiness keep the old messages, positional shim included
    for call, msg in ((lambda: _arrays.as_f32_c(np.zeros(3, np.float32), "X"), "X must be 2-D, got 1-D shape (3,)"),
                      (lambda: _arrays.as_f32_c(np.zeros((0, 3), np.float32), "X"), "X is empty, shape (0, 3)"),
                      (lambda: _buffer.as_f32_c([1.0], ndim=2, name="y"), "y must be 2-D, got 1-D shape (1,)"),
                      (lambda: _buffer.as_f64_c(np.zeros((2, 2)), ndim=1, name="y"), "y must be 1-D, got 2-D shape (2, 2)")):
        try:
            call()
            raise AssertionError(msg)
        except ValueError as e:
            assert str(e) == "mojolearn: " + msg, str(e)


# ---------------------------------------------------------------- Array API


def test_array_surface():
    a = Array.from_list([[1, 2, 3], [4, 5, 6]], "<i4")
    assert repr(a) == "Array(shape=(2, 3), dtype='<i4')"
    assert a.ndim == 2 and a.size == 6 and a.nbytes == 24 and a.itemsize == 4
    assert a.strides == (12, 4) and a.order == "C"
    assert a.flags == {"C_CONTIGUOUS": True, "F_CONTIGUOUS": False, "WRITEABLE": True}
    assert len(a) == 2 and [r.tolist() for r in a] == [[1, 2, 3], [4, 5, 6]]
    assert a[1, 2] == 6 and a[-1, -1] == 6 and a[0].tolist() == [1, 2, 3]
    assert a[:, 1].tolist() == [2, 5] and a[1:, ::2].tolist() == [[4, 6]]
    assert a[:, ::-1].tolist() == [[3, 2, 1], [6, 5, 4]]
    assert a.reshape((3, 2)).tolist() == [[1, 2], [3, 4], [5, 6]]
    assert a.reshape(-1).tolist() == [1, 2, 3, 4, 5, 6] and a.ravel().shape == (6,)
    assert a.reshape((3, -1)).__array_interface__["data"][0] == a.__array_interface__["data"][0]
    assert a.tobytes() == struct.pack("<6i", 1, 2, 3, 4, 5, 6)
    assert a.min() == 1 and a.max() == 6 and a.sum() == 21 and a.argmax() == 5
    assert (a == 3).tolist() == [[0, 0, 1], [0, 0, 0]] and (a == 3).dtype == "<u1"
    assert (a == a.copy()).sum() == 6 and (a != a).sum() == 0
    c = a.copy()
    assert c.tobytes() == a.tobytes() and _buffer.addr_ro(c, name="c") != _buffer.addr_ro(a, name="a")
    assert a.astype("<f8").tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert a.astype(np.float32).dtype == "<f4"
    assert _buffer.full((2, 2), 2.5, "<f4").tolist() == [[2.5, 2.5], [2.5, 2.5]]
    assert _buffer.zeros(3, "<i8").tolist() == [0, 0, 0] and _buffer.empty((0, 2), "<f4").size == 0
    fb = _buffer.frombytes(struct.pack("<3f", 1, 2, 3), "<f4", (3,))
    assert fb.tolist() == [1.0, 2.0, 3.0]
    try:
        _buffer.frombytes(b"\x00" * 5, "<f4", (2,))
        raise AssertionError("byte count must match")
    except ValueError:
        pass
    f = Array.from_list([1.0, -2.0, 0.5], "<f2")
    assert f.tolist() == [1.0, -2.0, 0.5] and f.tobytes() == struct.pack("<3e", 1.0, -2.0, 0.5)
    assert np.asarray(f).dtype == np.float16 and np.asarray(f).tolist() == [1.0, -2.0, 0.5]
    # argmax: first max wins; reductions on an F-order block walk the C view
    assert Array.from_list([1.0, 3.0, 3.0, 2.0], "<f4").argmax() == 1
    fo, _ = _buffer.as_f32_colmajor([[1.0, 9.0], [9.0, 2.0]], name="X")
    assert fo.argmax() == 1 and fo[1, 0] == 9.0 and fo.reshape(-1).tolist() == [1.0, 9.0, 9.0, 2.0]
    assert fo.tobytes() == struct.pack("<4f", 1.0, 9.0, 9.0, 2.0)
    # memoryview on 3.12+ (a pure-Python class cannot export earlier)
    if sys.version_info >= (3, 12):
        m = memoryview(a)
        assert m.shape == (2, 3) and m.format == "i" and m.tolist() == a.tolist()
    # scalar and bad indexing
    try:
        a[0, 0, 0]
        raise AssertionError("too many indices")
    except IndexError:
        pass
    try:
        a[5]
        raise AssertionError("out of range")
    except IndexError:
        pass
    try:
        bool(a)
        raise AssertionError("ambiguous truth value")
    except ValueError:
        pass


def test_all_finite_host_path():
    ok = Array.from_list([1.0, 1e-45, -0.0, 3.4e38], "<f4")
    assert _buffer.all_finite(ok)
    assert _buffer.all_finite(ok.astype("<f8"))
    assert _buffer.all_finite(_buffer.empty((0,), "<f4"))
    for bad in (float("nan"), float("inf"), float("-inf")):
        arr = Array.from_list([1.0, bad, 2.0], "<f4")
        assert not _buffer.all_finite(arr), bad
        assert not _buffer.all_finite(arr.astype("<f8")), bad
    # all finite but the sum overflows: the confirmation loop must answer
    huge = Array.from_list([1.7e308, 1.7e308, -1.7e308], "<f8")
    assert _buffer.all_finite(huge)
    both = Array.from_list([float("inf"), float("-inf")], "<f8")
    assert not _buffer.all_finite(both)
    try:
        _buffer.all_finite(Array.from_list([1], "<i4"))
        raise AssertionError("int arrays are refused")
    except TypeError:
        pass


# ---------------------------------------------------------------- runner


def main():
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"{len(names) - failed}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
