"""The native host converters of DEVIATION 2470 (`cast_f64_to_f32`), 2471
(`cast_colmajor_f64_to_f32`) and 2472 (`transpose_f32`) are BYTE-IDENTICAL
to NumPy, and the
pure-Python fallback in `_buffer` produces the same bytes with the binding
forced absent.

NumPy is the ORACLE here and belongs in this file; the shipped module under
test imports none of it. Every comparison is on raw bytes, never a
tolerance: the float64 -> float32 cast is one round-to-nearest-even per
element and there is nothing to be approximately right about.

The direct-binding tests skip when the base binding is not built; the
fallback tests run everywhere.
"""
import array
import math
import struct

import numpy as np
import pytest

# The inputs overflow float32 ON PURPOSE; NumPy warns when the oracle casts
# them, and that warning is the oracle's, not a finding.
pytestmark = pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")

from mojolearn import _buffer
from mojolearn._array import Array

# -------------------------------------------------------------- the inputs

_F32_MAX = float(np.finfo(np.float32).max)
_F32_TINY = float(np.finfo(np.float32).tiny)       # smallest normal
_F32_SUBNORMAL = float(np.float32(1e-40))            # a float32 subnormal
_F64_SUBNORMAL = 5e-324                              # rounds to +0.0


def _ties():
    """float64 values exactly halfway between two adjacent float32s, chosen
    so round-to-nearest-EVEN and round-half-away-from-zero DISAGREE.

    1 + 2**-24 sits between 1 (mantissa ...000, even) and 1 + 2**-23
    (...001, odd): RNE answers 1.0, half-away answers 1 + 2**-23.
    1 + 3 * 2**-24 sits between 1 + 2**-23 (odd) and 1 + 2**-22 (even):
    RNE rounds UP here, so a round-half-DOWN rule fails on this one while
    passing the first. Same pairs negated, and the same pattern at another
    binade and in the float32 subnormal range.
    """
    ulp = 2.0 ** -23
    half = ulp / 2
    vals = [
        1.0 + half, 1.0 + 3 * half, -(1.0 + half), -(1.0 + 3 * half),
        # another binade: ulp of 1000.0 in float32 is 2**-14
        1000.0 + 2.0 ** -15, 1000.0 + 3 * 2.0 ** -15,
        # subnormal float32 range: spacing is 2**-149
        2.0 ** -149 * 2.5, 2.0 ** -149 * 3.5, -(2.0 ** -149 * 2.5),
    ]
    # prove the ties are real ties under both rules before using them
    for v in vals:
        lo = np.nextafter(np.float32(v), np.float32(-np.inf))
        hi = np.nextafter(np.float32(v), np.float32(np.inf))
        f = np.float32(v)
        # v must be strictly between the two float32 neighbors of its
        # rounded value and equidistant from the two candidates
        cands = sorted({float(lo), float(f), float(hi)})
        below = max(c for c in cands if c <= v)
        above = min(c for c in cands if c >= v)
        assert below < v < above or v in cands, v
        if v not in cands:
            assert math.isclose(v - below, above - v, rel_tol=0, abs_tol=0), v
    return vals


def _edge_values():
    return [
        0.0, -0.0, 1.0, -1.0, 0.1, 1 / 3, math.pi, -math.e,
        _F32_MAX, -_F32_MAX,
        _F32_MAX * (1 + 2.0 ** -25),            # rounds back to FLT_MAX
        _F32_MAX * (1 + 2.0 ** -24),            # the tie: rounds to inf
        _F32_MAX * 2, -_F32_MAX * 2, 1e39, -1e39, 1e308,
        _F32_TINY, -_F32_TINY, _F32_SUBNORMAL, -_F32_SUBNORMAL,
        _F64_SUBNORMAL, -_F64_SUBNORMAL, 1e-46, 2.0 ** -150,
        math.inf, -math.inf, math.nan, -math.nan,
        struct.unpack("<d", struct.pack("<Q", 0x7FF8DEADBEEF0001))[0],
    ] + _ties()


def _shapes():
    return [
        (1, 1), (1, 7), (7, 1), (3, 5), (5, 3), (2, 64), (64, 2),
        (128, 64), (129, 65), (130, 20), (1000, 10), (257, 3), (3, 257),
    ]


def _matrix(shape, seed):
    """A float64 C-order matrix of `shape` seeded with the edge values,
    ties and random data, so the rounding cases land in every tile
    position across the shapes."""
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    flat = rng.standard_normal(n) * rng.choice([1e-30, 1e-3, 1.0, 1e3, 1e30], n)
    edges = np.array(_edge_values(), dtype=np.float64)
    idx = rng.permutation(n)[: min(n, len(edges))]
    flat[idx] = edges[: len(idx)]
    return np.ascontiguousarray(flat.reshape(shape))


def _bytes_equal(a, b):
    assert len(a) == len(b)
    if a != b:
        # locate the first differing float32 for the report
        fa = np.frombuffer(a, dtype=np.float32)
        fb = np.frombuffer(b, dtype=np.float32)
        bad = np.nonzero(fa.view(np.uint32) != fb.view(np.uint32))[0]
        # NaN payloads are not promised: accept NaN == NaN at the bit level
        # only if both are NaN
        real = [i for i in bad if not (np.isnan(fa[i]) and np.isnan(fb[i]))]
        assert not real, (
            f"first mismatch at {real[0]}: ours={fa[real[0]]!r} "
            f"({fa.view(np.uint32)[real[0]]:#010x}) numpy={fb[real[0]]!r} "
            f"({fb.view(np.uint32)[real[0]]:#010x}); {len(real)} differ"
        )


# ---------------------------------------------------------- the binding

def _fn(name):
    fn = _buffer._native(name)
    if fn is None:
        pytest.skip(f"base binding not built or lacks {name}")
    return fn


def _f64_store(x):
    s = array.array("d")
    s.frombytes(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    return s


@pytest.mark.parametrize("shape", _shapes())
def test_flat_cast_matches_numpy(shape):
    fn = _fn("cast_f64_to_f32")
    x = _matrix(shape, seed=shape[0] * 1000 + shape[1])
    src = _f64_store(x)
    dst = array.array("f", bytes(4 * x.size))
    assert fn(src.buffer_info()[0], dst.buffer_info()[0], x.size) == 0
    _bytes_equal(dst.tobytes(), np.ascontiguousarray(x, dtype=np.float32).tobytes())


@pytest.mark.parametrize("shape", _shapes())
def test_colmajor_cast_matches_numpy(shape):
    fn = _fn("cast_colmajor_f64_to_f32")
    x = _matrix(shape, seed=shape[0] * 7 + shape[1] * 13)
    src = _f64_store(x)
    dst = array.array("f", bytes(4 * x.size))
    assert fn(src.buffer_info()[0], dst.buffer_info()[0], *shape) == 0
    _bytes_equal(dst.tobytes(), np.asfortranarray(x, dtype=np.float32).tobytes(order="F"))


def test_every_edge_value_alone():
    """Each edge value as a one-element buffer, through both helpers, so a
    failure names the value rather than a position."""
    flat = _fn("cast_f64_to_f32")
    col = _fn("cast_colmajor_f64_to_f32")
    for v in _edge_values():
        src = array.array("d", [v])
        want = np.float32(v).tobytes()
        d1 = array.array("f", [0.0])
        flat(src.buffer_info()[0], d1.buffer_info()[0], 1)
        d2 = array.array("f", [0.0])
        col(src.buffer_info()[0], d2.buffer_info()[0], 1, 1)
        for got in (d1.tobytes(), d2.tobytes()):
            if math.isnan(v):
                assert math.isnan(struct.unpack("<f", got)[0]), v
            else:
                assert got == want, (v, got.hex(), want.hex())


def test_ties_round_to_even_not_away():
    """The tie values must come out as RNE says, and NOT as
    round-half-away-from-zero says; the two answers differ by one ulp."""
    fn = _fn("cast_f64_to_f32")
    disagreements = 0
    for v in _ties():
        src = array.array("d", [v])
        dst = array.array("f", [0.0])
        fn(src.buffer_info()[0], dst.buffer_info()[0], 1)
        got = dst[0]
        rne = float(np.float32(v))
        # the two float32 candidates bracketing v, then the half-away pick
        f = np.float32(v)
        lo = float(np.nextafter(f, np.float32(-np.inf)))
        hi = float(np.nextafter(f, np.float32(np.inf)))
        below = max(c for c in (lo, rne, hi) if c <= v)
        above = min(c for c in (lo, rne, hi) if c >= v)
        assert v - below == above - v, ("not a tie", v)
        away = above if v > 0 else below
        assert got == rne, (v, got, rne)
        if away != rne:
            disagreements += 1
            assert got != away, (v, "helper rounded half away from zero")
    assert disagreements >= 4, "tie set no longer separates the two rules"


def test_empty_and_negative():
    flat = _fn("cast_f64_to_f32")
    col = _fn("cast_colmajor_f64_to_f32")
    # n == 0 / a zero dimension: returns 0 and reads neither address, so
    # null addresses are fine here and MUST NOT raise
    assert flat(0, 0, 0) == 0
    assert col(0, 0, 0, 5) == 0
    assert col(0, 0, 5, 0) == 0
    with pytest.raises(Exception, match="cast_f64_to_f32: n must be non-negative"):
        flat(0, 0, -1)
    with pytest.raises(Exception, match="rows must be non-negative"):
        col(0, 0, -1, 3)
    with pytest.raises(Exception, match="cols must be non-negative"):
        col(0, 0, 3, -1)
    # a null address with a positive count is refused, not read
    with pytest.raises(Exception, match="null buffer address"):
        flat(0, 0, 4)


def test_tail_lengths_every_residue():
    """Every remainder modulo the SIMD width, so the scalar tail is
    exercised at each length it can have."""
    fn = _fn("cast_f64_to_f32")
    rng = np.random.default_rng(2470)
    for n in range(1, 40):
        x = rng.standard_normal(n)
        x[0] = 1.0 + 2.0 ** -24  # a tie in the head
        x[-1] = 1000.0 + 3 * 2.0 ** -15  # and one in the tail
        src = _f64_store(x)
        dst = array.array("f", bytes(4 * n))
        fn(src.buffer_info()[0], dst.buffer_info()[0], n)
        _bytes_equal(dst.tobytes(), x.astype(np.float32).tobytes())


@pytest.mark.parametrize("shape", _shapes())
def test_transpose_f32_matches_numpy_both_directions(shape):
    fn = _fn("transpose_f32")
    x = _matrix(shape, seed=shape[0] * 3 + shape[1] * 31).astype(np.float32)
    # plant NaN payloads and both signed zeros: a pure move keeps every bit
    flat = x.reshape(-1)
    if flat.size >= 3:
        flat[0] = np.frombuffer(struct.pack("<I", 0x7FC0BEEF), np.float32)[0]
        flat[1] = np.float32(-0.0)
        flat[2] = np.frombuffer(struct.pack("<I", 0xFFA00001), np.float32)[0]
    src = array.array("f", x.tobytes())
    dst = array.array("f", bytes(4 * x.size))
    assert fn(src.buffer_info()[0], dst.buffer_info()[0], *shape) == 0
    assert dst.tobytes() == np.asfortranarray(x).tobytes(order="F")
    # the other direction: the column-major block as a C [cols, rows] source
    back = array.array("f", bytes(4 * x.size))
    assert fn(dst.buffer_info()[0], back.buffer_info()[0], shape[1], shape[0]) == 0
    assert back.tobytes() == x.tobytes()


def test_transpose_f32_empty_and_negative():
    fn = _fn("transpose_f32")
    assert fn(0, 0, 0, 3) == 0 and fn(0, 0, 3, 0) == 0
    with pytest.raises(Exception, match="transpose_f32: rows must be non-negative"):
        fn(0, 0, -2, 3)
    with pytest.raises(Exception, match="transpose_f32: cols must be non-negative"):
        fn(0, 0, 3, -2)
    with pytest.raises(Exception, match="null buffer address"):
        fn(0, 0, 2, 2)


# --------------------------------------------- through _buffer, both arms

_KEYS = ("cast_f64_to_f32", "cast_colmajor_f64_to_f32", "transpose_f32")


def _forced_absent(monkeypatch):
    for key in _KEYS:
        monkeypatch.setitem(_buffer._NATIVE, key, None)


def _forced_present(monkeypatch, *, required=True):
    """Restore the native lookups. With `required` (the default) a missing
    binding skips the test; with `required=False` it returns False so the
    test can assert the pure-Python arm on its own and still PASS on a
    checkout whose binary predates these helpers."""
    for key in _KEYS:
        monkeypatch.delitem(_buffer._NATIVE, key, raising=False)
    if any(_buffer._native(key) is None for key in _KEYS):
        if required:
            pytest.skip("base binding not built or predates DEVIATION 2470-2472")
        return False
    return True


@pytest.mark.parametrize("shape", _shapes())
@pytest.mark.parametrize("src_order", ["C", "F"])
@pytest.mark.parametrize("src_dtype", [np.float64, np.float32])
def test_as_f32_c_two_arms(monkeypatch, shape, src_order, src_dtype):
    x = _matrix(shape, seed=11).astype(src_dtype)
    if src_order == "F":
        x = np.asfortranarray(x)
    if src_dtype is np.float32 and (src_order == "C" or 1 in shape):
        pytest.skip("already float32 in the target layout: a borrow, tested below")
    want = np.ascontiguousarray(x, dtype=np.float32)

    _forced_absent(monkeypatch)
    py_arr, copied = _buffer.as_f32_c(x, name="X")
    assert copied and py_arr.dtype == "<f4" and py_arr.order == "C"
    _bytes_equal(py_arr.tobytes(), want.tobytes())

    if not _forced_present(monkeypatch, required=False):
        return  # the fallback arm above is the whole test on this checkout
    nat_arr, copied = _buffer.as_f32_c(x, name="X")
    assert copied and nat_arr.dtype == "<f4" and nat_arr.order == "C"
    assert nat_arr.tobytes() == py_arr.tobytes()
    assert nat_arr.shape == py_arr.shape == tuple(shape)


@pytest.mark.parametrize("shape", _shapes())
@pytest.mark.parametrize("src_order", ["C", "F"])
@pytest.mark.parametrize("src_dtype", [np.float64, np.float32])
def test_as_f32_colmajor_two_arms(monkeypatch, shape, src_order, src_dtype):
    x = _matrix(shape, seed=17).astype(src_dtype)
    if src_order == "F":
        x = np.asfortranarray(x)
    if src_dtype is np.float32 and (src_order == "F" or 1 in shape):
        pytest.skip("already float32 in the target layout: a borrow, tested below")
    want = np.asfortranarray(x, dtype=np.float32)

    _forced_absent(monkeypatch)
    py_arr, copied = _buffer.as_f32_colmajor(x, name="X")
    assert copied and py_arr.dtype == "<f4"
    assert py_arr._has_order("F")
    # NumPy's tobytes() serializes in C order WHATEVER the memory layout;
    # the column-major bytes need order="F"
    _bytes_equal(py_arr.tobytes(), want.tobytes(order="F"))
    # the storage IS the column-major flat
    assert py_arr._flat().tobytes() == want.T.reshape(-1).tobytes()

    if not _forced_present(monkeypatch, required=False):
        return  # the fallback arm above is the whole test on this checkout
    nat_arr, copied = _buffer.as_f32_colmajor(x, name="X")
    assert copied and nat_arr.dtype == "<f4"
    assert nat_arr._has_order("F")
    assert nat_arr.tobytes() == py_arr.tobytes()
    assert np.asarray(nat_arr).tobytes() == np.asarray(py_arr).tobytes()
    assert np.array_equal(np.asarray(nat_arr), want, equal_nan=True)


def test_float32_in_target_order_is_still_a_borrow(monkeypatch):
    """The native path is for float64 input only; a float32 block already
    in the target order stays a zero-copy borrow of the caller's memory."""
    _forced_present(monkeypatch)
    xc = np.ascontiguousarray(np.arange(60, dtype=np.float32).reshape(6, 10))
    a, copied = _buffer.as_f32_c(xc, name="X")
    assert not copied and a._addr == xc.ctypes.data
    xf = np.asfortranarray(xc)
    a, copied = _buffer.as_f32_colmajor(xf, name="X")
    assert not copied and a._addr == xf.ctypes.data


def test_native_result_owns_fresh_storage(monkeypatch):
    """The converted Array must not alias the caller's float64 buffer."""
    _forced_present(monkeypatch)
    x = np.ones((4, 3), dtype=np.float64)
    a, _ = _buffer.as_f32_colmajor(x, name="X")
    x[:] = 5.0
    assert np.asarray(a).tolist() == np.ones((4, 3)).tolist()
    assert isinstance(a, Array) and a._base is None


def test_anon_store_behaves_like_an_array_store(monkeypatch):
    """On 3.12+ the native converters write into an anonymous mapping. The
    resulting Array must do everything an array.array-backed one does."""
    import sys
    _forced_present(monkeypatch)
    x = _matrix((300, 7), seed=2471)
    a, copied = _buffer.as_f32_colmajor(x, name="X")
    assert copied
    if sys.version_info >= (3, 12):
        assert isinstance(a._store, _buffer._AnonStore)
    want = np.asfortranarray(x, dtype=np.float32)
    assert a.tobytes() == want.tobytes(order="F")
    assert np.array_equal(np.asarray(a), want, equal_nan=True)
    assert np.asarray(a).ctypes.data == a._addr  # zero-copy view
    c = a.copy()
    assert c.tobytes() == a.tobytes() and c._addr != a._addr
    r = a.reshape(-1)  # F -> C copy then flat view
    assert r.tobytes() == np.ascontiguousarray(want).reshape(-1).tobytes()
    assert a._flat().tobytes() == want.T.reshape(-1).tobytes()
    # feeding the Array back in is a zero-copy borrow of the mapping
    b, copied = _buffer.as_f32_colmajor(a, name="X")
    assert not copied and b._addr == a._addr
    # and a C-order request from it goes through the pure reorder
    d, copied = _buffer.as_f32_c(a, name="X")
    assert copied and d.tobytes() == np.ascontiguousarray(want).tobytes()
    # element access and reductions read the right values
    assert a[2, 3] == float(want[2, 3]) or (math.isnan(a[2, 3]) and math.isnan(want[2, 3]))
    finite = want[np.isfinite(want)]
    assert _buffer.all_finite(a) is False  # the edge values include inf/nan
