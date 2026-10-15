# SPDX-License-Identifier: Apache-2.0
"""Native Apple fast BF16 candidate correctness, ownership and boundary tests.

Run after the orchestrator builds the opt-in Apple fast linalg binding, with
MOJOLEARN_REQUIRE_FAST_BF16=1 so a missing candidate is a failure, not a skip.
These are correctness tests, not full-workload promotion evidence. NumPy and
Python tracing are independent test oracles, never part of product execution.
"""
import os
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def fast():
    from mojolearn import linalg_fast

    if not linalg_fast.has_bf16():
        if os.environ.get("MOJOLEARN_REQUIRE_FAST_BF16") == "1":
            pytest.fail("requested Apple fast BF16 candidate is unavailable in the loaded binding")
        pytest.skip("Apple fast BF16 candidate was not enabled in this binding")
    return linalg_fast


def _native_call(function, *args, **kwargs):
    frames = []
    previous = sys.getprofile()

    def trace(frame, event, arg):
        if event == "call" and "/mojolearn/" in frame.f_code.co_filename.replace("\\", "/"):
            frames.append((frame.f_code.co_filename, frame.f_code.co_name))

    try:
        sys.setprofile(trace)
        result = function(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    assert not frames, f"product Python bytecode executed: {frames[:8]}"
    return result


def _rne(values):
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    # Match the conversion's specified flush of float32 subnormals.
    words = np.where((words & 0x7f800000) == 0, words & 0x80000000, words).astype(np.uint32)
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def _wide(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def test_conversion_rounding_native_boundary_and_owned_lifetime(fast):
    words = np.array([0, 0x80000000, 1, 0x80000001, 0x3f808000, 0x3f818000,
                      0x3f807fff, 0xbf818000, 0x7f800000, 0xff800000], dtype=np.uint32)
    values = words.view(np.float32).reshape(2, 5)
    bits = _native_call(fast.to_bf16, values)
    np.testing.assert_array_equal(np.asarray(bits), _rne(values))
    restored = _native_call(fast.from_bf16, bits)
    del bits
    np.testing.assert_array_equal(np.asarray(restored).view(np.uint32),
                                  _wide(_rne(values)).view(np.uint32))


@pytest.mark.parametrize("m,n,k", [(1, 9, 7), (31, 33, 17), (65, 63, 129)])
@pytest.mark.parametrize("ta,tb", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("a_bits", [False, True])
def test_products_tail_shapes_transposes_and_float32_a(fast, m, n, k, ta, tb, a_bits):
    rng = np.random.default_rng(8237 + m + n + k)
    a = rng.uniform(-1, 1, (k, m) if ta else (m, k)).astype(np.float32)
    b = rng.uniform(-1, 1, (n, k) if tb else (k, n)).astype(np.float32)
    ab = _native_call(fast.to_bf16, a) if a_bits else a
    bb = _native_call(fast.to_bf16, b)
    got = _native_call(fast.matmul_bf16, ab, bb, transpose_a=ta, transpose_b=tb)
    ar = _wide(_rne(a)) if a_bits else a
    br = _wide(_rne(b))
    expected = (ar.T if ta else ar).astype(np.float64) @ (br.T if tb else br).astype(np.float64)
    # Bound FP32 accumulation error using the absolute product sum. This
    # remains meaningful near cancellation, unlike relative error alone.
    scale = np.abs(ar.T if ta else ar).astype(np.float64) @ np.abs(br.T if tb else br).astype(np.float64)
    error = np.abs(np.asarray(got, dtype=np.float64) - expected)
    assert np.all(error <= (4 * k * np.finfo(np.float32).eps) * scale + 1e-6)
    assert got.shape == (m, n)


def test_strided_readonly_inputs_and_output_ownership(fast):
    a = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    b = _rne(np.arange(15, dtype=np.float32).reshape(5, 3)).T
    a.flags.writeable = False
    b.flags.writeable = False
    out = np.full((4, 5), np.nan, dtype=np.float32)
    got = _native_call(fast.matmul_bf16, a, b, out=out)
    assert got is out
    np.testing.assert_array_equal(out, a @ _wide(b))
    owned = _native_call(fast.matmul_bf16, a, b)
    got = _native_call(fast.matmul_bf16, a, b, out=owned)
    assert got is owned
    del a, b
    np.testing.assert_array_equal(np.asarray(owned), out)


def test_invalid_parameters_refuse_before_dispatch(fast):
    a = np.ones((3, 4), dtype=np.float32)
    b = _rne(np.ones((4, 2), dtype=np.float32))
    invalid = [
        ((a, b), {"transpose_a": True, "transpose_b": True}),
        ((a, b), {"unknown": 1}),
        ((a, b.astype(np.float16)), {}),
        ((a, b[:3]), {}),
        ((a, b), {"out": np.empty((3, 2), dtype=np.float64)}),
        ((a, b), {"out": np.empty((3, 4), dtype=np.float32)[:, ::2]}),
    ]
    for args, kwargs in invalid:
        with pytest.raises(Exception):
            _native_call(fast.matmul_bf16, *args, **kwargs)
