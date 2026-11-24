# SPDX-License-Identifier: Apache-2.0
"""`mojolearn.lowbit` at the Python boundary: the packed containers' returned types and dtypes. The stored bytes and
the packed models' answers are the lowbit-conversions and *-bf16w / *-int8w verify lanes."""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

from mojolearn import lowbit  # noqa: E402


def _hw(shape, seed, lo=-0.125, hi=0.125):
    rng = np.random.default_rng(seed)
    return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)


# ---------------------------------------------------------------- moved from test_lowbit_weights.py

def test_packed_containers_are_arrays():
    """`pack`/`unpack` hand back mojolearn Arrays, never NumPy (the
    conversion itself runs in the linalg binding)."""
    from mojolearn import Array
    # the packed containers are Arrays, never NumPy (the former `_linalg()`
    # probe left lowbit; an install without the binding refuses pack itself)
    try:
        packed = lowbit.pack({"a": _hw((8, 4), 3), "n": _hw((4,), 4)}, "int8")
    except ImportError as e:
        pytest.skip(f"no linalg binding: {e}")
    assert isinstance(packed["a"].codes, Array) and isinstance(packed["a"].exponents, Array)
    assert lowbit.is_packed(packed["a"]) and not lowbit.is_packed(packed["n"])
    packed = lowbit.pack({"a": _hw((8, 4), 3)}, "bfloat16")
    assert isinstance(packed["a"].bits, Array) and packed["a"].bits.dtype == "<u2"
    f32, fmt = lowbit.unpack(packed)
    assert fmt == "bfloat16" and isinstance(f32["a"], Array) and f32["a"].dtype == "<f4"
