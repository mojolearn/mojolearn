"""DEVIATION 2489: the Mojo nonzero scan behind `_spectral_impl._coo_triples`
against the pure-Python loop it replaced, on raw bytes, never on values.

The oracle is the Python loop that used to live in `_coo_triples`
(DEVIATION 2373), kept HERE and nowhere else: the package has no Python
fallback for a native helper (removed 2026-09-10). Both must produce
identical int32 row bytes, int32 column bytes and float32 value bytes,
including the scan order. Values are planted so that every branch of the
nonzero test is exercised: exact zeros, -0.0 (a zero), NaN (NOT a zero),
subnormals, values that are nonzero in float64 but narrow to 0.0f (kept as
an explicit zero, as np.nonzero on the source dtype kept them), float32
rounding ties chosen so round-to-nearest-even and round-half-away
disagree, and overflow to inf.
"""
import array
import math
import random
import struct

import pytest

from mojolearn import _buffer
from mojolearn._array import Array
from mojolearn._buffer import as_f64_c
from mojolearn._spectral_impl import _coo_triples


def _oracle(A):
    """The retired DEVIATION 2373 loop, verbatim: the definition of the
    answer, only slow."""
    dense, _ = as_f64_c(A, ndim=2, name="X")
    r_idx, c_idx, values = [], [], []
    for r, row in enumerate(dense.tolist()):
        for c, v in enumerate(row):
            if v != 0.0:
                r_idx.append(r)
                c_idx.append(c)
                values.append(v)
    return (Array.from_list(r_idx, "<i4"), Array.from_list(c_idx, "<i4"),
            Array.from_list(values, "<f4"), dense.shape[0])


def _f32_ties():
    """float64 values exactly halfway between two adjacent float32s, whose
    two candidate roundings differ in the last bit's parity."""
    out = []
    for m in (0x800001, 0x800003, 0x800005, 0xFFFFFD, 0xFFFFFF):
        lo = struct.unpack("<f", struct.pack("<I", m))[0]
        hi = struct.unpack("<f", struct.pack("<I", m + 1))[0]
        out.append((lo + hi) / 2.0)  # exact in float64
        out.append(-(lo + hi) / 2.0)
    return out


def _planted_values():
    vals = [
        0.0, -0.0, 1.0, -1.0, 2.5, 1e-300, -1e-300,  # 1e-300 -> 0.0f, kept
        5e-324, 1e-45, 1.4e-45, 1e-38, 1.1754943508222875e-38,  # subnormal f32
        3.4028234663852886e38, 3.5e38, -3.5e38, 1e300,  # FLT_MAX, overflow
        float("inf"), -float("inf"), float("nan"), -float("nan"),
        0.1, 0.2, 0.3, 1.0 / 3.0, math.pi, math.e,
    ]
    vals += _f32_ties()
    return vals


def _matrix(n, seed, zero_fraction=0.6):
    rng = random.Random(seed)
    planted = _planted_values()
    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            u = rng.random()
            if u < zero_fraction:
                row.append(0.0 if rng.random() < 0.9 else -0.0)
            elif u < zero_fraction + 0.2:
                row.append(rng.choice(planted))
            else:
                row.append(rng.uniform(-10.0, 10.0))
        rows.append(row)
    return rows


def _as_bytes(triple):
    rows, cols, vals, n = triple
    assert rows.dtype == "<i4" and cols.dtype == "<i4" and vals.dtype == "<f4"
    assert rows.shape == cols.shape == vals.shape
    return bytes(rows), bytes(cols), bytes(vals), n


def _run_both(monkeypatch, A):
    return _as_bytes(_oracle(A)), _as_bytes(_coo_triples(A)), True


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 33, 64, 129])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dense_scan_bytes_match_python_loop(monkeypatch, n, seed):
    A = _matrix(n, seed * 1000 + n)
    ref, got, _ = _run_both(monkeypatch, A)
    assert got == ref
    # The Python arm on its own is a real assertion too: the planted NaNs
    # must be present and the -0.0s absent.
    rows, cols, vals, size = ref
    assert size == n
    flat = [v for row in A for v in row]
    expect_nnz = sum(1 for v in flat if v != 0.0)
    assert len(rows) == 4 * expect_nnz


def test_every_planted_value_in_one_row(monkeypatch):
    planted = _planted_values()
    n = len(planted)
    A = [[0.0] * n for _ in range(n)]
    for i, v in enumerate(planted):
        A[i][i] = v          # diagonal carries each planted value once
        A[0][i] = v          # row 0 carries them all in order
    ref, got, _ = _run_both(monkeypatch, A)
    assert got == ref
    rows, cols, vals, _ = ref
    r = array.array("i"); r.frombytes(rows)
    c = array.array("i"); c.frombytes(cols)
    v = array.array("f"); v.frombytes(vals)
    # Scan order is row-major: row 0 first, then the diagonal from row 1.
    first_row_count = sum(1 for x in planted if x != 0.0)
    assert list(r[:first_row_count]) == [0] * first_row_count
    assert all(rr <= r[i + 1] for i, rr in enumerate(r[:-1]))
    # 1e-300 narrowed to +0.0f is KEPT; -0.0 and 0.0 are dropped.
    kept = [struct.pack("<f", x) for x in v]
    assert struct.pack("<f", 0.0) in kept
    assert struct.pack("<f", -0.0) not in kept or any(x == -1e-300 for x in planted)
    # NaN kept with a NaN payload (any), infinities kept.
    assert any(math.isnan(x) for x in v)
    assert any(math.isinf(x) and x > 0 for x in v)


def test_ties_round_to_even_not_away(monkeypatch):
    ties = _f32_ties()
    n = len(ties)
    A = [[0.0] * n for _ in range(n)]
    for i, t in enumerate(ties):
        A[i][i] = t
    ref, got, _ = _run_both(monkeypatch, A)
    assert got == ref
    v = array.array("f"); v.frombytes(ref[2])
    for t, x in zip(ties, v):
        # round-half-away would land on the odd neighbor; RNE on the even
        assert struct.unpack("<I", struct.pack("<f", x))[0] & 1 == 0
        assert abs(x - t) <= abs(struct.unpack("<f", struct.pack("<f", t))[0] - t)


def test_all_zero_and_all_nonzero(monkeypatch):
    for A in ([[0.0] * 5 for _ in range(5)], [[-0.0] * 3 for _ in range(3)],
              [[1.0] * 4 for _ in range(4)]):
        ref, got, _ = _run_both(monkeypatch, A)
        assert got == ref


def test_sparse_input_is_duck_typed_not_imported(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("numpy", "scipy"):
            raise AssertionError("unexpected runtime dependency: " + name)
        return real_import(name, *args, **kwargs)

    class COO:
        shape = (3, 3)
        row = array.array("i", [0, 1, 2])
        col = array.array("i", [2, 0, 1])
        data = array.array("d", [1.5, -2.0, 1e-300])  # float64 in, f32 out

        def tocoo(self):
            return self

    monkeypatch.setattr(builtins, "__import__", guarded)
    rows, cols, vals, n = _coo_triples(COO())
    assert n == 3
    assert rows.tolist() == [0, 1, 2] and cols.tolist() == [2, 0, 1]
    assert bytes(vals) == array.array("f", [1.5, -2.0, 1e-300]).tobytes()

    class NotSquare(COO):
        shape = (3, 2)

    with pytest.raises(ValueError, match="must be square"):
        _coo_triples(NotSquare())


def test_direct_binding_refusals():
    count_fn = _buffer._native("nonzero_f64_count")
    fill_fn = _buffer._native("nonzero_f64_fill")
    src = Array.from_list([1.0, 0.0, 2.0, 3.0], "<f8")
    a = _buffer.addr_ro(src, name="src")
    assert count_fn(a, 0) == 0
    assert count_fn(a, 4) == 3
    with pytest.raises(Exception, match="non-negative"):
        count_fn(a, -1)
    out = [_buffer.empty((3,), "<i4"), _buffer.empty((3,), "<i4"),
           _buffer.empty((3,), "<f4")]
    addrs = [_buffer.addr(o, name="o") for o in out]
    assert fill_fn(a, 2, 2, addrs, 3) == 3
    assert out[0].tolist() == [0, 1, 1] and out[1].tolist() == [0, 0, 1]
    assert out[2].tolist() == [1.0, 2.0, 3.0]
    # Stale, too-small capacity: refused, and nothing written past it.
    small = [_buffer.empty((1,), "<i4"), _buffer.empty((1,), "<i4"),
             _buffer.empty((1,), "<f4")]
    saddrs = [_buffer.addr(o, name="o") for o in small]
    with pytest.raises(Exception, match="more than 1 nonzero"):
        fill_fn(a, 2, 2, saddrs, 1)
    assert fill_fn(a, 0, 5, addrs, 3) == 0
    with pytest.raises(Exception, match="non-negative"):
        fill_fn(a, -1, 2, addrs, 3)
    with pytest.raises(Exception, match="3 addresses"):
        fill_fn(a, 2, 2, addrs[:2], 3)
