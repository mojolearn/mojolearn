# SPDX-License-Identifier: Apache-2.0
"""The Python surface of the fifteen-bit profile,
`mojolearn.identical.gemm.int15i64.v1`: the named refusals of the public
door. Lane lane/lowbit-int15, 2026-09-29; gemm/IDENTICAL_LOWBIT_CONTRACT.md
section 6. The quantizer rule, the exact integer product, host agreement and
the planted -128 worst case are gated in Mojo by gemm/checks/gemm_int15_check.mojo
(pixi run check-gemm-int15).

Runs only when the identical linalg extension is built in this checkout;
skips by name otherwise.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn  # noqa: E402,F401
from mojolearn import linalg  # noqa: E402


def _gpu():
    try:
        linalg.require_identical()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"identical linalg extension not loaded: {e}")
    return linalg


def _vals(shape, seed):
    """20-bit significands over eight binades: a fifteen-bit code keeps
    fourteen magnitude bits, so every value is rounded."""
    rng = np.random.default_rng(seed)
    mant = 1.0 + rng.integers(0, 1 << 20, size=shape) / float(1 << 20)
    e = rng.integers(-4, 4, size=shape)
    sign = np.where(rng.integers(0, 2, size=shape) == 1, -1.0, 1.0)
    return (sign * mant * np.ldexp(1.0, e)).astype(np.float32)


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
