# SPDX-License-Identifier: Apache-2.0
"""Tree input packing preserves native float bits, layout, and borrowing."""
import numpy as np
import pytest

from mojolearn._arrays import as_f32_colmajor


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_large_row_major_conversion_matches_numpy_bits(dtype):
    # Cross both the tiling threshold and a partial final tile. Include NaN
    # payloads, signed zeros, infinities, and subnormals in the float32 arm.
    bits = np.array([0, 0x80000000, 1, 0x007fffff, 0x3f800001,
                     0x7f800000, 0xff800000, 0x7fc12345], np.uint32)
    source = np.resize(bits.view(np.float32), 80003 * 28).reshape(80003, 28)
    source = source.astype(dtype)
    before = source.copy()
    source.flags.writeable = False
    expected = np.asfortranarray(source, dtype=np.float32)
    actual, flat, copied = as_f32_colmajor(source, "X")
    actual, flat = np.asarray(actual), np.asarray(flat)
    assert copied and actual.flags.f_contiguous
    assert np.shares_memory(actual, flat)
    assert not np.shares_memory(actual, source)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(flat.view(np.uint32),
                                  expected.ravel(order="F").view(np.uint32))
    assert source.tobytes() == before.tobytes()


def test_fortran_float32_is_borrowed():
    source = np.ones((80003, 28), dtype=np.float32, order="F")
    source.flags.writeable = False
    actual, flat, copied = as_f32_colmajor(source, "X")
    assert np.shares_memory(actual, source) and not copied
    assert not np.asarray(actual).flags.writeable
    assert np.shares_memory(source, flat)


@pytest.mark.parametrize("variant", ["strided", "reversed", "integer", "bigendian", "narrow"])
def test_fallback_layouts_match_numpy(variant):
    source = np.arange(100 * 28, dtype=np.float32).reshape(100, 28)
    source = {"strided": source[::2, ::2], "reversed": source[::-1],
              "integer": source.astype(np.int32),
              "bigendian": source.astype(">f4"), "narrow": source[:, :1]}[variant]
    actual, flat, _ = as_f32_colmajor(source, "X")
    expected = np.asfortranarray(source, dtype=np.float32)
    actual, flat = np.asarray(actual), np.asarray(flat)
    assert actual.flags.f_contiguous
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(flat, expected.ravel(order="F"))
