# SPDX-License-Identifier: Apache-2.0
"""Python buffer, dtype, transpose and refusal contracts for low-bit linalg.

Numerical BF16 conversions/GEMM and INT8 quantize/GEMM/dequantize cases run
directly in Mojo (checks/native_tests/{bf16_conversion,bf16_linalg,int8_linalg}_*).
These small cases retain public input-form dispatch, packed outputs, raw host
binding pointer/count ABI and named errors. They require the corresponding
installed identical linalg bindings.
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


@pytest.mark.parametrize("ta,tb,lowbit_a", [
    (False, True, False), (False, False, False),
    (True, False, False), (False, True, True),
])
def test_bf16_operand_forms_and_transpose_flags_pack_native_buffers(ta, tb, lowbit_a):
    g = _gpu()
    h = _host()
    a = np.array([[1.0, 2.0], [0.0, 1.0]], np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], np.float32)
    if ta:
        a = np.ascontiguousarray(a.T)
    if not tb:
        b = np.ascontiguousarray(b.T)
    if lowbit_a:
        a = np.asarray(g.to_bf16(a))
    b = np.asarray(g.to_bf16(b))
    result = np.asarray(g.matmul_bf16(a, b, transpose_a=ta, transpose_b=tb))
    assert result.dtype == np.float32 and result.shape == (2, 3)
    assert result[0, 1] == 2.0 and result[0, 2] == 3.0  # Transpose/input-form routing sentinels.
    host_result = np.empty((2, 3), np.float32)
    op = 2 if ta else (1 if tb else 0)
    h.gemm_bf16(_addr(host_result), _addr(a), _addr(b), [2, 3, 2, op, int(lowbit_a)])
    assert host_result[0, 1] == 2.0 and host_result[0, 2] == 3.0


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
