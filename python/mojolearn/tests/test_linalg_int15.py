# SPDX-License-Identifier: Apache-2.0
"""The Python surface of the fifteen-bit profile,
`mojolearn.identical.gemm.int15i64.v1`, gated BITWISE against the CPU host
binding (the oracle the CPU identity gate runs) and against the rule written
out in NumPy integers. Lane lane/lowbit-int15, 2026-09-29;
gemm/IDENTICAL_LOWBIT_CONTRACT.md section 6.

Runs only when the identical linalg extension AND the linalg host binding
are built in this checkout; skips by name otherwise.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn  # noqa: E402,F401
from mojolearn import _backend  # noqa: E402
from mojolearn import linalg  # noqa: E402


def _gpu():
    try:
        linalg.require_identical()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"identical linalg extension not loaded: {e}")
    return linalg


def _host():
    try:
        h = _backend.load_host_module("_mojolearn_linalg_host")
    except ImportError as e:
        pytest.skip(f"linalg host binding not built: {e}")
    if not hasattr(h, "gemm_int15"):
        pytest.skip("the built linalg host binding predates the fifteen-bit profile")
    return h


def _addr(a):
    return a.__array_interface__["data"][0]


def _vals(shape, seed):
    """20-bit significands over eight binades: a fifteen-bit code keeps
    fourteen magnitude bits, so every value is rounded."""
    rng = np.random.default_rng(seed)
    mant = 1.0 + rng.integers(0, 1 << 20, size=shape) / float(1 << 20)
    e = rng.integers(-4, 4, size=shape)
    sign = np.where(rng.integers(0, 2, size=shape) == 1, -1.0, 1.0)
    return (sign * mant * np.ldexp(1.0, e)).astype(np.float32)


def _rule(x):
    """Contract W-1 to W-3 in NumPy: the exponent from the row's absmax, the
    code by round to nearest even (np.rint), the clamp, the two planes."""
    absmax = np.abs(x).max(axis=1)
    e = np.where(absmax == 0, 0, np.floor(np.log2(np.where(absmax == 0, 1, absmax))).astype(np.int64) - 13)
    scaled = x.astype(np.float64) * np.ldexp(1.0, -e)[:, None]
    q = np.clip(np.rint(scaled), -16383, 16383).astype(np.int64)
    return (q >> 7).astype(np.int8), (q & 127).astype(np.int8), e.astype(np.int32), q


def test_planes_are_the_rule_and_the_host_agrees():
    g = _gpu()
    x = _vals((37, 300), 1)
    hi, lo, e = (np.asarray(v) for v in g.quantize_int15(x))
    assert hi.dtype == np.int8 and lo.dtype == np.int8 and e.dtype == np.int32
    wh, wl, we, q = _rule(x)
    assert np.array_equal(e, we)
    assert np.array_equal(hi, wh) and np.array_equal(lo, wl)
    assert lo.min() >= 0 and np.array_equal(hi.astype(np.int64) * 128 + lo, q)
    h = _host()
    hh, hl, he = np.empty(x.shape, np.int8), np.empty(x.shape, np.int8), np.empty(x.shape[0], np.int32)
    h.quantize_int15([_addr(hh), _addr(hl), _addr(he), _addr(np.ascontiguousarray(x))], list(x.shape))
    assert np.array_equal(hh, hi) and np.array_equal(hl, lo) and np.array_equal(he, e)
    y = np.asarray(g.dequantize_int15(hi, lo, e))
    assert np.array_equal(y, (q * np.ldexp(1.0, e.astype(np.int64))[:, None]).astype(np.float32))


def test_product_is_the_exact_integer_sum_and_the_host_agrees():
    g = _gpu()
    for m, n, k in ((1, 40, 33), (17, 33, 1000), (5, 9, 4097)):
        a, b = _vals((m, k), 10 + m), _vals((n, k), 20 + n)
        ah, al, ea, qa = _rule(a)
        bh, bl, eb, qb = _rule(b)
        c = np.asarray(g.matmul_int15(a, b))
        c2 = np.asarray(g.matmul_int15(tuple(g.quantize_int15(a)), tuple(g.quantize_int15(b))))
        assert c.dtype == np.float32 and c.shape == (m, n)
        assert c.tobytes() == c2.tobytes()
        # the exact sums are below 2^44, so Python integers and one
        # correctly rounded conversion are the reference
        sums = qa @ qb.T
        want = (sums.astype(np.float64).astype(np.float32)
                * np.ldexp(np.float32(1.0), ea[:, None].astype(np.int64) + eb[None, :])).astype(np.float32)
        assert c.tobytes() == want.tobytes(), (m, n, k)
        h = _host()
        hc = np.empty((m, n), np.float32)
        h.gemm_int15([_addr(hc), _addr(ah), _addr(al), _addr(ea), _addr(bh), _addr(bl), _addr(eb)], [m, n, k])
        assert hc.tobytes() == c.tobytes(), (m, n, k)


def test_planted_codes_reach_the_high_piece_of_minus_128():
    g = _gpu()
    k = 4096
    hi = np.full((2, k), -128, np.int8)
    lo = np.full((2, k), 1, np.int8)           # the code -16383
    e = np.zeros(2, np.int32)
    c = np.asarray(g.matmul_int15((hi, lo, e), (hi, lo, e)))
    assert np.all(c == np.float32(16383 * 16383 * k))


def test_a_contracted_extent_above_the_bound_is_refused_by_name():
    g = _gpu()
    k = linalg.INT15_MAX_K + 1
    z = np.zeros((1, k), np.int8)
    with pytest.raises(Exception, match="65536"):
        g.matmul_int15((z, z, np.zeros(1, np.int32)), (z, z, np.zeros(1, np.int32)))


def test_operands_of_the_wrong_kind_are_refused():
    g = _gpu()
    x = _vals((3, 8), 5)
    with pytest.raises(TypeError):
        g.matmul_int15((x, x), x)
    hi, lo, e = g.quantize_int15(x)
    with pytest.raises(ValueError):
        g.matmul_int15((hi, lo, np.zeros(4, np.int32)), x)
