# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gates for the NumPy-free core: `_array.Array`, `_buffer`, `_serialize`.

NumPy is used HERE, from the `test` extra, as the oracle: the library must
not import it, and this file proves the library's conversions and file
bytes agree with it bit for bit (DEVIATION 2310). Runs under pytest or as
a plain program:

    cd python && python3 -m mojolearn.tests.test_numpy_free_core

The native finiteness verdicts behind `_buffer.all_finite` are
checks/native_tests/core_all_finite.mojo; only its Python refusals are here.
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

from mojolearn import _buffer, _serialize
from mojolearn._array import Array


# ----------------------------------------------------------------- helpers


def _assert_same_bytes(got, want, what):
    assert bytes(got) == bytes(want), what


# --------------------------------------------------------------- zero-copy


def test_empty_array_interface_numpy_conversion():
    # Force the array-interface path even on Python 3.12+, where a buffer
    # export would otherwise conceal the older-NumPy null-pointer failure.
    class InterfaceOnly:
        def __init__(self, value):
            self.value = value

        @property
        def __array_interface__(self):
            return self.value.__array_interface__

    nonempty = Array((2,), "<i8")
    exported = np.asarray(InterfaceOnly(nonempty))
    assert exported.__array_interface__["data"][0] == nonempty._addr
    exported[0] = 7
    assert nonempty[0] == 7

    for dtype in ("<i8", "<i4", "<f4", "<f8", "<f2"):
        for shape in ((0,), (0, 3), (2, 0)):
            value = Array(shape, dtype)
            interface = value.__array_interface__
            assert interface["data"][0] != 0
            for wrapped in (value, InterfaceOnly(value)):
                got = np.asarray(wrapped)
                assert got.shape == shape and got.dtype == np.dtype(dtype)
                assert got.size == 0 and got.tobytes() == b""


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
    f = Array.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "<f4")._as_order("F")
    assert f.order == "F"
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


# TOMBSTONE: test_f64_to_f32_matches_numpy_bit_for_bit, test_int_to_f32_matches_numpy,
# test_colmajor_matches_asfortranarray, test_c_order_from_any_layout_matches_ascontiguousarray (they tested
# the deleted `_buffer.as_*_c` converters; casts are native, checks/native_tests/core_convert.mojo) deleted
# 2026-11-15 by lane py-tail-2; code recoverable at 1a9d8c01bd.


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
                streamed = io.BytesIO()
                _serialize.write_npy(streamed, x)
                _assert_same_bytes(streamed.getvalue(), want,
                                   f"write_npy(ndarray {dt.__name__} {shape} {order})")
                got_arr = _serialize.encode_npy(Array.from_buffer(x))
                _assert_same_bytes(got_arr, want, f"encode_npy(Array {dt.__name__} {shape} {order})")
                if x.nbytes and order == "C":
                    writes = []

                    class RecordingBytesIO(io.BytesIO):
                        def write(self, value):
                            writes.append(type(value))
                            return super().write(value)

                    # A Python `write` override is a file protocol the native archive writer refuses
                    # before any write (bindings/classical_archive_door.mojo _archive_methods, no calls
                    # back into Python); an exact BytesIO streams the Array natively.
                    try:
                        _serialize.write_npy(RecordingBytesIO(), Array.from_buffer(x))
                    except TypeError as refusal:
                        assert "Python file protocols cannot execute" in str(refusal), refusal
                    else:
                        raise AssertionError("a Python write override was accepted")
                    assert writes == []
                    streamed_array = io.BytesIO()
                    _serialize.write_npy(streamed_array, Array.from_buffer(x))
                    _assert_same_bytes(streamed_array.getvalue(), want,
                                       f"write_npy(Array {dt.__name__} {shape} {order})")
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


def test_numeric_npy_decode_borrows_payload_until_owned_copy():
    """A large numeric member must not be sliced into a second `bytes`.

    `_buffer.frombytes` is the ownership boundary: it copies into the
    returned Array.  The codec should hand that boundary a borrowed view of
    the ZIP payload, while text members keep their established decoding.
    """
    payload = _serialize.encode_npy(np.arange(1024, dtype=np.float32))
    original = _buffer.frombytes
    seen = []

    def counted(raw, dtype, shape):
        seen.append(type(raw))
        return original(raw, dtype, shape)

    _buffer.frombytes = counted
    try:
        got = _serialize.decode_npy(payload)
    finally:
        _buffer.frombytes = original
    # decode is native: the payload is copied into the returned Array without the Python
    # `_buffer.frombytes` boundary (so it is never sliced into a second `bytes` there either)
    assert seen == []
    assert got.tobytes() == np.arange(1024, dtype=np.float32).tobytes()
    assert _serialize.decode_npy(_serialize.encode_npy("identity")) == "identity"


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
        routing = _serialize.peek_npz(old, ("format", "estimator"))
        assert routing == {
            "format": "mojolearn-extratrees-1",
            "estimator": "ExtraTreesClassifier",
        }
        assert _serialize.peek_npz(old, ("missing",)) == {}
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
    # the old message word for word (the `_arrays` shim is deleted)
    try:
        _buffer.addr(raw, name="output buffer")
        raise AssertionError("a read-only output buffer must be refused")
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
    # memoryview, including a strided one
    mv = memoryview(aa)
    with _buffer.view(mv) as b:
        assert b.addr == aa.buffer_info()[0] and b.c_contiguous
    # numpy: C, F, non-contiguous (copied), read-only
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    with _buffer.view(x) as b:
        assert b.c_contiguous and not b.f_contiguous and b.strides == (12, 4)
    assert _buffer.addr_ro(x, name="X") == x.ctypes.data
    xf = np.asfortranarray(x)
    with _buffer.view(xf) as b:
        assert b.f_contiguous and not b.c_contiguous and b.strides == (4, 8)
    xs = x[:, ::2]
    with _buffer.view(xs) as b:
        assert not b.c_contiguous and not b.f_contiguous
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
    fo = Array.from_list([[1.0, 9.0], [9.0, 2.0]], "<f4")._as_order("F")
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


def test_all_finite_python_boundary():
    # The native verdicts on finite, NaN/Inf and overflowing-sum vectors are
    # checks/native_tests/core_all_finite.mojo; the empty shortcut and the
    # dtype refusal are this wrapper's own.
    assert _buffer.all_finite(_buffer.empty((0,), "<f4"))
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
