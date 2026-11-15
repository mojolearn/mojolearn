# SPDX-License-Identifier: Apache-2.0
"""`mojolearn._serialize` is one native codec call per helper
(bindings/archive_npz_door.mojo): the refusals keep the Python codec's words
and exception types, and the encoded bytes are numpy's `write_array` bytes."""

import io

import pytest

from mojolearn import Array, _serialize


def test_round_trip_scalars_and_lists(tmp_path):
    path = str(tmp_path / "m.npz")
    arrays = {"format": "fmt-1", "estimator": "X", "i": 7, "f": 0.5, "b": True,
              "raw": b"ab", "names": ["a", "bc"], "grid": [[1, 2], [3, 4]],
              "w": Array.from_list([1.0, 2.0], "<f4")}
    assert _serialize.write_npz(path, arrays) == path
    out = _serialize.read_npz(path, "fmt-1")
    assert out["format"] == "fmt-1" and out["names"] == ["a", "bc"] and out["raw"] == b"ab"
    assert _serialize.exact(out, "i", "<i8").tolist() == [7]
    assert _serialize.exact(out, "f", "<f8").tolist() == [0.5]
    assert _serialize.exact(out, "grid", "<i8").shape == (2, 2)
    assert _serialize.exact(out, "w", "<f4").tolist() == [1.0, 2.0]
    assert _serialize.scalar_str(out, "estimator") == "X"
    assert _serialize.peek_npz(path, ("format",)) == {"format": "fmt-1"}
    assert _serialize.peek_npz(path, ("missing",)) == {}


def test_refusals_keep_their_words(tmp_path):
    path = str(tmp_path / "m.npz")
    _serialize.write_npz(path, {"format": "fmt-1", "meta": Array.from_list([1.0], "<f8")})
    with pytest.raises(ValueError, match=r"holds model format 'fmt-1', this loader reads 'fmt-2'"):
        _serialize.read_npz(path, "fmt-2")
    out = _serialize.read_npz(path, ("fmt-0", "fmt-1"))
    with pytest.raises(ValueError, match=r"model file is missing field 'coef'"):
        _serialize.exact(out, "coef", "<f4")
    with pytest.raises(ValueError, match=r"field 'meta' has dtype <f8, the format stores <i8; refusing to cast"):
        _serialize.exact(out, "meta", "<i8")
    with pytest.raises(ValueError, match=r"model file is missing field 'name'"):
        _serialize.scalar_str(out, "name")
    with pytest.raises(TypeError, match="cannot serialize a list of mixed kinds"):
        _serialize.encode_npy(["a", 1])
    with pytest.raises(ValueError, match="not rectangular"):
        _serialize.encode_npy([[1, 2], [3]])
    with pytest.raises(ValueError, match="bad magic"):
        _serialize.decode_npy(b"not an npy member at all")


def test_npy_bytes_and_header():
    blob = _serialize.encode_npy("héllo€")
    assert blob[:10] == b"\x93NUMPY\x01\x00" + blob[8:10] and len(blob) % 4 == 0
    assert _serialize.decode_npy(blob) == "héllo€"
    descr, fortran, shape, offset = _serialize._parse_header(_serialize.encode_npy(2.5))
    assert (descr, fortran, shape) == ("<f8", False, ())
    assert offset % 64 == 0
    stream = io.BytesIO()
    _serialize.write_npy(stream, [1.0, 2.0])
    assert stream.getvalue() == _serialize.encode_npy([1.0, 2.0])
