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


def test_bf16_conversion_packs_expected_shapes_and_dtypes():
    # Exhaustive RNE/shift word checks now run directly in Mojo through the
    # typed host entries and GPU kernels (checks/native_tests/bf16_conversion_*).
    # This test retains the public ndarray and raw host-binding buffer seams.
    g = _gpu()
    x = np.array([[1.25, -2.5], [0.0, 1.0]], np.float32)
    bits = np.asarray(g.to_bf16(x))
    assert bits.dtype == np.uint16 and bits.shape == x.shape
    assert bits[0, 0] == 0x3FA0  # Public route and result-packing sentinel.
    h = _host()
    hb = np.empty(x.shape, np.uint16)
    assert h.to_bf16(_addr(hb), _addr(x), [x.size]) == x.size
    assert hb[0, 0] == 0x3FA0
    wide = np.asarray(g.from_bf16(bits))
    assert wide.dtype == np.float32 and wide.shape == x.shape
    assert wide[0, 0] == 1.25
    hw = np.empty(x.shape, np.float32)
    assert h.from_bf16(_addr(hw), _addr(bits), [bits.size]) == bits.size
    assert hw[0, 0] == 1.25


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


def test_int8_input_forms_and_raw_bindings_pack_native_buffers():
    # Quantizer, integer dot-product and dequantizer numerical oracles now
    # run directly in checks/native_tests/int8_linalg_* for all five shapes.
    g = _gpu()
    h = _host()
    a = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], np.float32)
    qa, ea = (np.asarray(v) for v in g.quantize_int8(a))
    qb, eb = (np.asarray(v) for v in g.quantize_int8(b))
    assert qa.dtype == np.int8 and qa.shape == (2, 2)
    assert ea.dtype == np.int32 and ea.shape == (2,)
    assert qb.dtype == np.int8 and qb.shape == (3, 2)
    assert eb.dtype == np.int32 and eb.shape == (3,)
    hqa, hea = np.empty((2, 2), np.int8), np.empty((2,), np.int32)
    assert h.quantize_int8(_addr(hqa), _addr(hea), _addr(a), [2, 2]) == 4
    assert hqa[0, 0] == 64 and hea[0] == -6
    c1 = np.asarray(g.matmul_int8(a, b))
    c2 = np.asarray(g.matmul_int8((qa, ea), (qb, eb)))
    assert c1.dtype == c2.dtype == np.float32 and c1.shape == c2.shape == (2, 3)
    assert np.array_equal(c1, c2)  # Both admitted Python input forms route identically.
    assert c1[0, 0] == 1.0
    hc = np.empty((2, 3), np.float32)
    assert h.gemm_int8(_addr(hc), _addr(qa), _addr(ea), _addr(qb), _addr(eb), [2, 3, 2]) == 6
    assert hc[0, 0] == 1.0
    y = np.asarray(g.dequantize_int8(qb, eb))
    assert y.dtype == np.float32 and y.shape == (3, 2) and y[0, 0] == 1.0
    hy = np.empty((3, 2), np.float32)
    assert h.dequantize_int8(_addr(hy), _addr(qb), _addr(eb), [3, 2]) == 6
    assert hy[0, 0] == 1.0


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
