# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host helpers of DEVIATION 2303 in `bindings/_mojolearn.mojo`:
`all_finite_f32` and `all_finite_f64` (DEVIATION 2326), plus the
probability and gather helpers. Written 2026-09-07 on branch numpy-free-0.7.
`column_mean_f64`, `center_columns_f32` and `scale_rows_f32` were deleted
by lane hr-small-passes (2026-10-02): the linear models center through the
estimators binding (`lm_col_sums`, `lm_center`, `lm_scale_rows`).

Every buffer here is an `array.array` or a raw buffer, because the helpers
exist so the Python layer can stop importing NumPy
(`python/mojolearn/NUMPY_FREE_CONTRACT.md`).
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

_NEEDED = ("all_finite_f32", "all_finite_f64")

if _ext is None:
    pytestmark = pytest.mark.skip(
        reason="python/mojolearn/_mojolearn.so is not built; "
        "bash bindings/build.sh"
    )
elif not all(callable(getattr(_ext, name, None)) for name in _NEEDED):
    pytestmark = pytest.mark.skip(
        reason="python/mojolearn/_mojolearn.so predates DEVIATION 2303 "
        "(no all_finite_f32/all_finite_f64); rebuild with "
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
    with pytest.raises(Exception, match="null (float(32|64) )?buffer address"):
        fn(0, 1)
    with pytest.raises(Exception, match="non-negative"):
        fn(_addr(buf), -1)


def test_probability_validation_and_binary_packing():
    import numpy as np  # oracle only; implementation uses raw buffers
    p = np.array([0, 1, .125, .7, np.nextafter(np.float32(0), np.float32(1))], dtype=np.float32)
    out = np.full((len(p), 2), -99, dtype=np.float32)
    assert _ext.probability_rows_f32(p.ctypes.data, out.ctypes.data, len(p), 1, 1) == 0
    expected = np.column_stack((np.float32(1) - p, p))
    assert out.tobytes() == expected.tobytes()
    assert _ext.probability_rows_f32(out.ctypes.data, 0, len(p), 2, 0) == 0
    for bad, code in ((float('nan'), 1), (float('inf'), 1), (-.1, 2), (1.1, 2)):
        p[2] = bad
        assert _ext.probability_rows_f32(p.ctypes.data, out.ctypes.data, len(p), 1, 1) == code
    wrong = np.array([[.1, .2], [.3, .7]], dtype=np.float32)
    assert _ext.probability_rows_f32(wrong.ctypes.data, 0, 2, 2, 0) == 3
    # Keep the documented error-category precedence when different worker
    # shards observe different failures.
    mixed = np.full((20000, 2), np.float32(.5), dtype=np.float32)
    mixed[1, 0] = np.nan
    mixed[-1] = (-.25, .25)
    assert _ext.probability_rows_f32(mixed.ctypes.data, 0, len(mixed), 2, 0) == 1
    mixed[1] = (.5, .5)
    assert _ext.probability_rows_f32(mixed.ctypes.data, 0, len(mixed), 2, 0) == 2
    assert _ext.probability_rows_f32(0, 0, 0, 1, 1) == 0


def test_gather_rows_bytes_checks_and_copies_exactly():
    import numpy as np
    source = np.arange(35, dtype=np.uint8).reshape(5, 7)
    indices = np.array([4, 0, 4, 2], dtype=np.int64)
    out = np.full((4, 7), 255, dtype=np.uint8)
    _ext.gather_rows_bytes(source.ctypes.data, out.ctypes.data,
                           indices.ctypes.data, 5, 4, 7)
    assert out.tobytes() == source[indices].tobytes()
    for bad in (-1, 5):
        indices[-1] = bad
        before = out.tobytes()
        with pytest.raises(Exception, match='index out of bounds'):
            _ext.gather_rows_bytes(source.ctypes.data, out.ctypes.data,
                                  indices.ctypes.data, 5, 4, 7)
        assert out.tobytes() == before
    assert _ext.gather_rows_bytes(0, 0, 0, 0, 0, 7) == 0
