# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The three host helpers of DEVIATION 2303 in `bindings/_mojolearn.mojo`:
`all_finite_f32`, `all_finite_f64` and `column_mean_f64`.

DEVIATIONS 2326 through 2329. Written 2026-09-07 on branch numpy-free-0.7.

Every buffer here is an `array.array`, never a NumPy array, because the
helpers exist so the Python layer can stop importing NumPy
(`python/mojolearn/NUMPY_FREE_CONTRACT.md`) and a test of that layer that
needed NumPy to build its inputs would be checking the wrong thing.

WHAT THE column_mean CHECK IS, AND WHAT IT IS NOT
--------------------------------------------------
`column_mean_f64` is defined as a sequential row-order float64
accumulation, and its docstring writes that order out. The oracle below
IS that order, in Python floats (IEEE binary64 on every CPython this
project supports), and the comparison is bit equality through
`struct.pack('<d', ...)`, not `math.isclose`. A tolerance would hide
exactly the defect being hunted: a reassociated sum that lands within an
ulp of the defined one and moves every IDENTICAL OLS centering bit.

The check against `math.fsum` is the OPPOSITE assertion. `fsum` is the
correctly rounded sum; the helper is deliberately NOT that, and column 0
of the fixture is planted so the two cannot agree. Someone who rewrites
the helper to be "more accurate" fails that test on purpose.
"""

import array
import math
import random
import struct

import pytest

try:
    from mojolearn import _mojolearn as _ext
except ImportError:  # the base extension is not built in this checkout
    _ext = None

_NEEDED = ("all_finite_f32", "all_finite_f64", "column_mean_f64")

if _ext is None:
    pytestmark = pytest.mark.skip(
        reason="python/mojolearn/_mojolearn.so is not built; "
        "bash bindings/build.sh"
    )
elif not all(callable(getattr(_ext, name, None)) for name in _NEEDED):
    pytestmark = pytest.mark.skip(
        reason="python/mojolearn/_mojolearn.so predates DEVIATION 2303 "
        "(no all_finite_f32/all_finite_f64/column_mean_f64); rebuild with "
        "bash bindings/build.sh"
    )


ROWS, COLS = 1000, 7
SEED = 2303
NAN, PINF, NINF = float("nan"), float("inf"), float("-inf")
# Subnormal in each width: both are strictly inside (0, min_normal).
F32_SUBNORMAL = 1e-40      # min normal float32 is ~1.18e-38
F64_SUBNORMAL = 5e-324     # the smallest float64 subnormal


def _addr(buf):
    """The address of an `array.array`'s first element."""
    return buf.buffer_info()[0]


def _bits(x):
    return struct.pack("<d", x)


# ---------------------------------------------------------------------------
# all_finite_f32 / all_finite_f64 (DEVIATION 2326)
# ---------------------------------------------------------------------------

def _finite_cases(typecode, subnormal):
    base = [1.0, -2.5, 0.0, -0.0, 3.0e5, subnormal, -subnormal]
    yield "all finite incl. subnormals", array.array(typecode, base), 1
    for label, bad in (("nan", NAN), ("+inf", PINF), ("-inf", NINF)):
        for where in ("first", "middle", "last"):
            vals = list(base)
            i = {"first": 0, "middle": len(vals) // 2, "last": len(vals) - 1}[where]
            vals[i] = bad
            yield f"{label} at {where}", array.array(typecode, vals), 0


@pytest.mark.parametrize("width", ["f32", "f64"])
def test_all_finite_flags_nan_and_both_infinities(width):
    typecode, subnormal = ("f", F32_SUBNORMAL) if width == "f32" else ("d", F64_SUBNORMAL)
    fn = getattr(_ext, f"all_finite_{width}")
    for label, buf, expected in _finite_cases(typecode, subnormal):
        # The subnormal really is stored as a subnormal, not flushed to 0.
        assert buf[5] != 0.0 and abs(buf[5]) < 1.2e-38, label
        got = fn(_addr(buf), len(buf))
        assert got == expected, f"{width}: {label}: got {got!r}"
        assert isinstance(got, int)


@pytest.mark.parametrize("width", ["f32", "f64"])
def test_all_finite_reads_exactly_n_elements(width):
    """A NaN just PAST `n` is not read, and a NaN AT index n-1 is: the scan
    covers exactly the `n` elements it was told about."""
    typecode = "f" if width == "f32" else "d"
    fn = getattr(_ext, f"all_finite_{width}")
    buf = array.array(typecode, [1.0, 2.0, 3.0, NAN])
    assert fn(_addr(buf), 3) == 1
    assert fn(_addr(buf), 4) == 0
    assert fn(_addr(buf), 0) == 1


@pytest.mark.parametrize("width", ["f32", "f64"])
def test_all_finite_refuses_null_and_negative(width):
    typecode = "f" if width == "f32" else "d"
    fn = getattr(_ext, f"all_finite_{width}")
    buf = array.array(typecode, [1.0])
    with pytest.raises(Exception, match="null buffer address"):
        fn(0, 1)
    with pytest.raises(Exception, match="non-negative"):
        fn(_addr(buf), -1)


# ---------------------------------------------------------------------------
# column_mean_f64 (DEVIATIONS 2327-2329)
# ---------------------------------------------------------------------------

def _fixture():
    """A seeded ROWS x COLS float32 block as an `array.array('f')`.

    Columns 1..6 are seeded uniform values spanning several decades so the
    accumulation order matters. Column 0 is PLANTED: `2**60, 1.0, -2**60`
    then zeros. In binary64 the ulp at 2**60 is 256, so `2**60 + 1.0`
    rounds back to `2**60` and the sequential total is exactly 0.0, while
    the correctly rounded total (`math.fsum`) is exactly 1.0. Both planted
    magnitudes are exact in float32, so the widening loses nothing and the
    disagreement is entirely the accumulation order's.
    """
    rng = random.Random(SEED)
    vals = []
    for r in range(ROWS):
        for c in range(COLS):
            if c == 0:
                vals.append({0: 2.0 ** 60, 1: 1.0, 2: -(2.0 ** 60)}.get(r, 0.0))
            else:
                mag = 10.0 ** rng.uniform(-3, 6)
                vals.append(rng.choice((-1.0, 1.0)) * mag)
    buf = array.array("f", vals)      # rounds each value to float32 once
    assert len(buf) == ROWS * COLS
    return buf


def _sequential_oracle(buf, rows, cols):
    """THE definition, verbatim from the helper's docstring: row-major,
    one binary64 addition per element, then one division per column."""
    x = buf.tolist()                  # the exact float32 values, as floats
    acc = [0.0] * cols
    for r in range(rows):
        for c in range(cols):
            acc[c] = acc[c] + x[r * cols + c]
    return [acc[c] / float(rows) for c in range(cols)]


def _run_column_mean(buf, rows, cols):
    out = array.array("d", [NAN] * cols)   # NaN-filled so an unwritten slot shows
    ret = _ext.column_mean_f64(_addr(buf), rows, cols, _addr(out))
    assert ret == 0
    return out.tolist()


def test_column_mean_f64_is_bit_identical_to_the_sequential_definition():
    """DEVIATION 2327: bit equality, per column, against the written-out
    order. This is not a tolerance test and must never become one."""
    buf = _fixture()
    got = _run_column_mean(buf, ROWS, COLS)
    want = _sequential_oracle(buf, ROWS, COLS)
    assert len(got) == COLS
    for c in range(COLS):
        assert _bits(got[c]) == _bits(want[c]), (
            f"column {c}: helper {got[c]!r} != sequential {want[c]!r}"
        )
    # The planted column proves the accumulation really was sequential:
    # a tree or SIMD reduction that paired 1.0 with a zero first would
    # keep it.
    assert _bits(got[0]) == _bits(0.0)


def test_column_mean_f64_is_not_fsum_and_that_is_intentional():
    """DEVIATION 2328: the helper is NOT the correctly rounded mean.

    Column 0 is planted so the two disagree by construction (see
    `_fixture`). This test exists to fail loudly if someone "improves" the
    helper into `fsum`, pairwise, or Kahan summation: that would be a
    silent re-definition of the OLS/ridge centering that every IDENTICAL
    reference card was baselined against.
    """
    buf = _fixture()
    got = _run_column_mean(buf, ROWS, COLS)
    x = buf.tolist()
    fsum_mean = [
        math.fsum(x[r * COLS + c] for r in range(ROWS)) / float(ROWS)
        for c in range(COLS)
    ]
    assert fsum_mean[0] == 1.0 / ROWS
    assert got[0] == 0.0
    assert _bits(got[0]) != _bits(fsum_mean[0])


def test_column_mean_f64_small_shapes_and_widening():
    """DEVIATION 2329: rows == 1 is the identity (widened), cols == 1 is a
    plain sequential mean, and the float32 to float64 widening is exact
    for a value float32 cannot hold as typed."""
    one_row = array.array("f", [0.1, -7.0, 1e-40])
    got = _run_column_mean(one_row, 1, 3)
    assert [_bits(v) for v in got] == [_bits(v) for v in one_row.tolist()]
    # 0.1 as float32 is not 0.1 as float64; the helper reports the float32.
    assert got[0] != 0.1 and got[0] == one_row[0]

    col = array.array("f", [1.0, 2.0, 4.0, 8.0])
    assert _run_column_mean(col, 4, 1) == [_sequential_oracle(col, 4, 1)[0]]
    assert _run_column_mean(col, 4, 1) == [3.75]


def test_column_mean_f64_refuses_bad_shapes_and_null():
    buf = array.array("f", [1.0, 2.0])
    out = array.array("d", [0.0])
    with pytest.raises(Exception, match="rows must be positive"):
        _ext.column_mean_f64(_addr(buf), 0, 1, _addr(out))
    with pytest.raises(Exception, match="cols must be positive"):
        _ext.column_mean_f64(_addr(buf), 2, 0, _addr(out))
    with pytest.raises(Exception, match="null buffer address"):
        _ext.column_mean_f64(0, 2, 1, _addr(out))
    with pytest.raises(Exception, match="null buffer address"):
        _ext.column_mean_f64(_addr(buf), 2, 1, 0)


# ---------------------------------------------------------------------------
# center_columns_f32 / scale_rows_f32 (DEVIATIONS 2444-2449): bit identity
# against the Python loops they replace, `linear_model.py::_center` and
# `::_scale_rows` (DEVIATION 2362), on a seeded 2000 x 9 float32 block.
#
# The Python loops are imported and RUN here as the oracle. They are the
# definition (float64 operation, one narrowing through `array.array('f')`);
# the helpers are written in that same order, so equality is by
# construction and the comparison is on the raw bytes of the whole block.
# ---------------------------------------------------------------------------

try:
    from mojolearn import linear_model as _lm
    from mojolearn._array import Array as _Array
    from mojolearn._buffer import addr as _w_addr, addr_ro as _ro_addr, empty as _empty
except ImportError:  # the NumPy-free Python layer is not on this checkout
    _lm = None

ROWS2, COLS2 = 2000, 9
SEED2 = 2440


def _need(name):
    if _lm is None:
        pytest.skip("mojolearn.linear_model / _array / _buffer not importable")
    fn = getattr(_ext, name, None)
    if not callable(fn):
        pytest.skip(f"python/mojolearn/_mojolearn.so predates DEVIATION 2440 "
                    f"(no {name}); rebuild with bash bindings/build.sh")
    return fn


def _block2():
    """Seeded 2000 x 9 float32 block spanning nine decades, both signs,
    with four planted elements: a float32 max, a subnormal, a zero and a
    negative zero, so the narrowing sees an overflow-to-inf candidate and
    the signed-zero cases along with the ordinary ones."""
    rng = random.Random(SEED2)
    rows = []
    for r in range(ROWS2):
        row = [rng.choice((-1.0, 1.0)) * 10.0 ** rng.uniform(-4, 5)
               for _ in range(COLS2)]
        rows.append(row)
    rows[0][0] = 3.4028234663852886e38      # float32 max, exact
    rows[1][1] = 1e-40                      # float32 subnormal
    rows[2][2] = 0.0
    rows[3][3] = -0.0
    return _Array.from_list(rows, "<f4")


def _mu32():
    """Float32-valued means, as `_column_means` hands `_center` (it ends in
    `_round_f32`). Seeded independently of the block so they are not the
    block's means: the helper subtracts whatever it is given."""
    rng = random.Random(SEED2 + 1)
    return [_lm._round_f32(rng.uniform(-1e3, 1e3)) for _ in range(COLS2)]


def _root32():
    """`fl32(sqrt(w))` per row, the value `LinearRegression.fit` builds
    (`root = [_round_f32(math.sqrt(v)) for v in weights.tolist()]`), with
    row 0's weight planted at 4.0 so `root[0] == 2.0` multiplies the
    float32 max into an overflow, which the Python's C `(float)` cast
    makes +inf and the helper's `Float32(...)` must make +inf too."""
    rng = random.Random(SEED2 + 2)
    w = [rng.uniform(0.01, 50.0) for _ in range(ROWS2)]
    w[0] = 4.0
    return [_lm._round_f32(math.sqrt(v)) for v in w]


def _assert_same_bytes(got, want, label):
    gb, wb = got.tobytes(), want.tobytes()
    if gb != wb:
        g, w = got.tolist(), want.tolist()
        bad = [(r, c, g[r][c], w[r][c]) for r in range(ROWS2)
               for c in range(COLS2) if _bits(g[r][c]) != _bits(w[r][c])]
        pytest.fail(f"{label}: {len(bad)} element(s) differ; first: {bad[:3]}")


def test_center_columns_f32_is_bit_identical_to_the_python_center():
    """DEVIATION 2444."""
    fn = _need("center_columns_f32")
    x, mu32 = _block2(), _mu32()
    want = _lm._center(x, mu32)
    assert want.dtype == "<f4" and want.shape == (ROWS2, COLS2)
    mean = array.array("d", mu32)
    out = _empty((ROWS2, COLS2), "<f4")
    ret = fn(_ro_addr(x, name="X"), ROWS2, COLS2, _addr(mean),
             _w_addr(out, name="centered"))
    assert ret == 0
    _assert_same_bytes(out, want, "center_columns_f32 vs _center")
    # The subtraction really happened (not a copy): column 0 moved by mu[0].
    assert out.tolist()[5][0] != x.tolist()[5][0]


def test_center_columns_f32_may_alias_its_input():
    """DEVIATION 2445: in-place is the same bits, since no element depends
    on another. `fit` can overwrite its float32 copy instead of allocating
    a second block."""
    fn = _need("center_columns_f32")
    x, mu32 = _block2(), _mu32()
    want = _lm._center(x, mu32)
    work = x.copy()
    mean = array.array("d", mu32)
    fn(_ro_addr(work, name="X"), ROWS2, COLS2, _addr(mean),
       _w_addr(work, name="X"))
    _assert_same_bytes(work, want, "center_columns_f32 in place vs _center")


def test_scale_rows_f32_is_bit_identical_to_the_python_scale_rows():
    """DEVIATION 2446."""
    fn = _need("scale_rows_f32")
    x, root = _block2(), _root32()
    want = _lm._scale_rows(x, root)
    assert want.dtype == "<f4" and want.shape == (ROWS2, COLS2)
    w = array.array("f", root)              # exact: already float32 values
    assert w.tolist() == root
    out = _empty((ROWS2, COLS2), "<f4")
    ret = fn(_ro_addr(x, name="X"), ROWS2, COLS2, _addr(w),
             _w_addr(out, name="scaled"))
    assert ret == 0
    _assert_same_bytes(out, want, "scale_rows_f32 vs _scale_rows")
    # The planted overflow: float32 max times 2.0 is +inf on BOTH sides.
    assert math.isinf(want.tolist()[0][0]) and math.isinf(out.tolist()[0][0])
    # Row 0 is scaled by root[0] == 2.0 exactly; every other element of it
    # is the input doubled (exact in float32 short of overflow).
    for c in range(1, COLS2):
        assert out.tolist()[0][c] == 2.0 * x.tolist()[0][c]


def test_scale_rows_f32_may_alias_its_input():
    """DEVIATION 2447."""
    fn = _need("scale_rows_f32")
    x, root = _block2(), _root32()
    want = _lm._scale_rows(x, root)
    work = x.copy()
    w = array.array("f", root)
    fn(_ro_addr(work, name="X"), ROWS2, COLS2, _addr(w),
       _w_addr(work, name="X"))
    _assert_same_bytes(work, want, "scale_rows_f32 in place vs _scale_rows")


def test_center_then_scale_matches_the_fit_path_composition():
    """DEVIATION 2448: `fit` centers THEN scales (linear_model.py, the
    weighted branch after `_center`). The two helpers composed in that
    order equal the two Python loops composed in that order, byte for
    byte, on the same block."""
    cen, sca = _need("center_columns_f32"), _need("scale_rows_f32")
    x, mu32, root = _block2(), _mu32(), _root32()
    want = _lm._scale_rows(_lm._center(x, mu32), root)
    mean, w = array.array("d", mu32), array.array("f", root)
    work = x.copy()
    cen(_ro_addr(work, name="X"), ROWS2, COLS2, _addr(mean), _w_addr(work, name="X"))
    sca(_ro_addr(work, name="X"), ROWS2, COLS2, _addr(w), _w_addr(work, name="X"))
    _assert_same_bytes(work, want, "center then scale vs the Python pair")


def test_elementwise_helpers_refuse_null_and_negative_and_accept_empty():
    """DEVIATION 2449."""
    cen, sca = _need("center_columns_f32"), _need("scale_rows_f32")
    x = array.array("f", [1.0, 2.0])
    mean = array.array("d", [0.5])
    w = array.array("f", [2.0])
    out = array.array("f", [0.0, 0.0])
    with pytest.raises(Exception, match="null buffer address"):
        cen(0, 2, 1, _addr(mean), _addr(out))
    with pytest.raises(Exception, match="null buffer address"):
        cen(_addr(x), 2, 1, 0, _addr(out))
    with pytest.raises(Exception, match="null buffer address"):
        sca(_addr(x), 2, 1, _addr(w), 0)
    with pytest.raises(Exception, match="non-negative"):
        cen(_addr(x), -1, 1, _addr(mean), _addr(out))
    with pytest.raises(Exception, match="non-negative"):
        sca(_addr(x), 2, -1, _addr(w), _addr(out))
    # Zero rows or zero cols writes nothing and returns 0.
    assert cen(_addr(x), 0, 1, _addr(mean), _addr(out)) == 0
    assert sca(_addr(x), 2, 0, _addr(w), _addr(out)) == 0
    assert out.tolist() == [0.0, 0.0]
