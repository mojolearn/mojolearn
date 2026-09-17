# SPDX-License-Identifier: Apache-2.0
"""The Python surface of the two low-bit profiles, gated BITWISE against the
CPU host binding (the same oracles the CPU identity gate runs) and against
NumPy for the one seam NumPy can spell exactly (the bf16 widening is a
shift). Lane lane/identical-lowbit-inference, 2026-09-17;
gemm/IDENTICAL_LOWBIT_CONTRACT.md.

Runs only when the identical linalg extension AND the linalg host binding
are built in this checkout; skips by name otherwise.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn  # noqa: E402
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
        return _backend.load_host_module("_mojolearn_linalg_host")
    except ImportError as e:
        pytest.skip(f"linalg host binding not built: {e}")


def _addr(a):
    return a.__array_interface__["data"][0]


def _vals(shape, seed):
    """13-bit significands over eight binades, the device check's generator:
    every product inexact, so a wrong seam moves bits."""
    rng = np.random.default_rng(seed)
    mant = 1.0 + rng.integers(0, 4096, size=shape) / 4096.0
    e = rng.integers(-4, 4, size=shape)
    sign = np.where(rng.integers(0, 2, size=shape) == 1, -1.0, 1.0)
    return (sign * mant * np.ldexp(1.0, e)).astype(np.float32)


def test_widening_is_the_shift_and_narrowing_is_rne():
    g = _gpu()
    x = _vals((37, 53), 1)
    bits = np.asarray(g.to_bf16(x))
    assert bits.dtype == np.uint16 and bits.shape == x.shape
    # host binding, the same seam
    h = _host()
    hb = np.empty(x.shape, np.uint16)
    h.to_bf16(_addr(hb), _addr(np.ascontiguousarray(x)), [x.size])
    assert np.array_equal(bits, hb)
    # NumPy spelling of round-to-nearest-even on the low 16 bits
    u = x.view(np.uint32)
    ref = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    assert np.array_equal(bits, ref)
    # widening is exactly the shift, on the GPU and on the host
    wide = np.asarray(g.from_bf16(bits))
    assert np.array_equal(wide.view(np.uint32), bits.astype(np.uint32) << 16)
    hw = np.empty(x.shape, np.float32)
    h.from_bf16(_addr(hw), _addr(bits), [bits.size])
    assert np.array_equal(hw.view(np.uint32), wide.view(np.uint32))


@pytest.mark.parametrize("m,n,k,ta,tb", [
    (1, 32, 32, False, True), (1, 64, 128, False, True), (2, 96, 127, False, True),
    (7, 33, 129, False, True), (5, 9, 256, False, False), (9, 5, 300, True, False),
    (129, 129, 256, False, True), (1, 1024, 512, False, True),
])
def test_matmul_bf16_matches_host_and_widened_fp32(m, n, k, ta, tb):
    g = _gpu()
    h = _host()
    a = _vals((k, m) if ta else (m, k), 10 + k)
    b32 = _vals((n, k) if tb else (k, n), 20 + k)
    b = np.asarray(g.to_bf16(b32))
    c = np.asarray(g.matmul_bf16(a, b, transpose_a=ta, transpose_b=tb))
    assert c.shape == (m, n) and c.dtype == np.float32
    # the host oracle, bit for bit
    op = 2 if ta else (1 if tb else 0)
    hc = np.empty((m, n), np.float32)
    h.gemm_bf16(_addr(hc), _addr(np.ascontiguousarray(a)), _addr(b), [m, n, k, op, 0])
    assert np.array_equal(c.view(np.uint32), hc.view(np.uint32))
    # the fp32 profile on the exactly widened operand, bit for bit: the
    # bf16 profile IS fp32.v1 on widened operands (contract section 0)
    wide = np.asarray(g.from_bf16(b))
    c32 = np.asarray(g.matmul(a, wide, transpose_a=ta, transpose_b=tb))
    assert np.array_equal(c.view(np.uint32), c32.view(np.uint32))


def test_matmul_bf16_both_operands_bf16():
    g = _gpu()
    h = _host()
    m, n, k = 6, 40, 200
    a = np.asarray(g.to_bf16(_vals((m, k), 3)))
    b = np.asarray(g.to_bf16(_vals((n, k), 4)))
    c = np.asarray(g.matmul_bf16(a, b, transpose_b=True))
    hc = np.empty((m, n), np.float32)
    h.gemm_bf16(_addr(hc), _addr(a), _addr(b), [m, n, k, 1, 1])
    assert np.array_equal(c.view(np.uint32), hc.view(np.uint32))


@pytest.mark.parametrize("m,n,k", [(1, 32, 32), (2, 96, 127), (7, 33, 129), (16, 64, 1000), (1, 1024, 512)])
def test_matmul_int8_matches_host(m, n, k):
    g = _gpu()
    h = _host()
    a = _vals((m, k), 30 + k)
    b = _vals((n, k), 40 + k)
    qa, ea = g.quantize_int8(a)
    qb, eb = g.quantize_int8(b)
    qa, ea, qb, eb = (np.asarray(v) for v in (qa, ea, qb, eb))
    assert qa.dtype == np.int8 and ea.dtype == np.int32 and ea.shape == (m,)
    assert np.all(np.abs(qa.astype(np.int32)) <= 127)
    # the host quantizer produces the same codes and exponents
    hqa = np.empty((m, k), np.int8)
    hea = np.empty((m,), np.int32)
    h.quantize_int8(_addr(hqa), _addr(hea), _addr(a), [m, k])
    assert np.array_equal(qa, hqa) and np.array_equal(ea, hea)
    # the product: float32 in, (codes, exponents) in, host oracle, all equal
    c1 = np.asarray(g.matmul_int8(a, b))
    c2 = np.asarray(g.matmul_int8((qa, ea), (qb, eb)))
    hc = np.empty((m, n), np.float32)
    h.gemm_int8(_addr(hc), _addr(qa), _addr(ea), _addr(qb), _addr(eb), [m, n, k])
    assert np.array_equal(c1.view(np.uint32), c2.view(np.uint32))
    assert np.array_equal(c1.view(np.uint32), hc.view(np.uint32))
    # an exact integer spelling of the same contract, in NumPy
    acc = qa.astype(np.int64) @ qb.astype(np.int64).T
    ref = (acc.astype(np.float32) * np.ldexp(np.float32(1.0), ea[:, None] + eb[None, :]).astype(np.float32)).astype(np.float32)
    ref[np.abs(ref) < np.float32(1.1754943508222875e-38)] = 0.0
    assert np.array_equal(c1.view(np.uint32), ref.view(np.uint32))
    # dequantization is exact and equal on both sides
    y = np.asarray(g.dequantize_int8(qb, eb))
    hy = np.empty((n, k), np.float32)
    h.dequantize_int8(_addr(hy), _addr(qb), _addr(eb), [n, k])
    assert np.array_equal(y.view(np.uint32), hy.view(np.uint32))
    assert np.array_equal(y, qb.astype(np.float32) * np.ldexp(np.float32(1.0), eb)[:, None].astype(np.float32))


def test_refusals_are_by_name():
    g = _gpu()
    a = _vals((4, 8), 5)
    with pytest.raises(TypeError, match="bf16 BITS"):
        g.matmul_bf16(a, a, transpose_b=True)
    b = np.asarray(g.to_bf16(a))
    with pytest.raises(ValueError, match="three operations"):
        g.matmul_bf16(a, b, transpose_a=True, transpose_b=True)
    with pytest.raises(ValueError, match="contracted extents"):
        g.matmul_int8(a, _vals((3, 9), 6))
