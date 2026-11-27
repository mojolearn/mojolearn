# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public `linalg` doors at the Python boundary: returned shapes and dtypes,
caller storage left untouched, and the refusals by type and words. Their
numerical comparisons against NumPy run in the native Mojo suites; their bits on each
side are the verify lanes."""
import os
import sys

import numpy as np
import pytest

from mojolearn import linalg


def _host_built():
    from pathlib import Path
    from mojolearn import _backend
    return Path(_backend.host_module_path("_mojolearn_linalg_host")).exists()


needs_host = pytest.mark.skipif(
    not _host_built(), reason="_mojolearn_linalg_host.so is not built here"
)


def _matrix(rows, cols, seed=5):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))


@needs_host
@pytest.mark.parametrize("rows,cols", [(48, 5), (16, 4)])
def test_qr_r_public_shape_and_dtype_match_numpy(rows, cols):
    """Keep the returned-array and caller-storage boundary against NumPy.

    The original two shapes' triangularity and Gram assertions now run over
    the actual CPU/GPU entries in checks/native_tests/linalg_numerical_*.
    """
    a = _matrix(rows, cols)
    original = a.copy()
    r = np.asarray(linalg.qr(a, mode="r"))
    assert r.shape == (cols, cols)
    assert r.dtype == np.float32
    np.testing.assert_array_equal(a, original)


@needs_host
def test_qr_default_preserves_published_r_only_shape_and_bytes():
    a = _matrix(48, 5)
    result = linalg.qr(a)
    assert result.shape == (5, 5)
    assert np.asarray(result).tobytes() == np.asarray(linalg.qr(a, mode="r")).tobytes()


@needs_host
def test_eigh_refuses_a_non_square_matrix():
    with pytest.raises(ValueError, match="must be square"):
        linalg.eigh(_matrix(16, 4))


@needs_host
@pytest.mark.parametrize("fn", ["qr", "eigh", "svdvals"])
def test_float64_is_refused_and_never_cast(fn):
    """`_operand`'s rule, which these doors inherit: a silent downcast loses
    29 mantissa bits before the profile sees them."""
    a = _matrix(16, 4).astype(np.float64)
    with pytest.raises(TypeError, match="float32"):
        getattr(linalg, fn)(a)


# ---------------------------------------------------------------------------------------------------------------------
# Apple fast BF16 (moved from test_linalg_fast_bf16.py): an invalid operand or keyword is refused before dispatch.
@pytest.fixture(scope="module")
def fast():
    import mojolearn

    # The BF16 entries exist only in the Apple (Metal) linalg binding: a CUDA or
    # HIP set does not carry them, and a CPU-only install (vendor "cpu") has no
    # host entry for them by design, so `linalg_fast` refuses them by name at
    # import there. Decide from the vendor the loaded binaries report.
    loaded = mojolearn.vendor()
    if loaded != "metal":
        if os.environ.get("MOJOLEARN_REQUIRE_FAST_BF16") == "1":
            pytest.fail(f"Apple fast BF16 was required but the loaded binaries are {loaded!r}, not 'metal'")
        pytest.skip(f"Apple fast BF16 is Apple-only; the loaded binaries are {loaded!r}")
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
