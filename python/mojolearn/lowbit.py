# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Low-bit WEIGHT STORAGE for every inference class, with fp32 arithmetic.

Lane lane/identical-lowbit-inference, 2026-09-17. Contract
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`; profiles
`mojolearn.identical.gemm.bf16f32.v1` and `mojolearn.identical.gemm.int8i32.v1`.

WHAT THIS IS. A model's matrix weights held as bf16 bits or as int8 codes
with one power-of-two exponent per output row, half or a quarter of the
float32 bytes, on disk and in memory. When an inference class is built from
packed weights it MATERIALIZES them: bf16 widens by the shift (exact,
contract L-1) and int8 dequantizes by `codes * 2^e` (exact, L-6), and the
float32 result runs through the block's certified fp32 path unchanged. So a
block built from packed weights computes, bit for bit, what the fp32 block
computes from the materialized weights, on every certified backend; the
`*-bf16w` and `*-int8w` lanes of `tools/identity_break.py` measure exactly
that on each column, and `python/mojolearn/tests/test_lowbit_weights.py`
asserts the equality on one box.

WHAT IT IS NOT. Not the fp32 model: a bf16 weight is a rounded weight and an
int8 weight a coarser one. Not a lower-precision matrix unit: the arithmetic
is the fp32 profile's. Not a speed claim.

WHERE THE MATERIALIZATION RUNS. Through the linalg extension's conversion
kernels on a GPU column, through `_mojolearn_linalg_host` on a CPU column,
and through the NumPy spelling when neither is loaded (an install with no
binding at all). All three are the same exact integer construction, and the
test asserts that where two are available they agree.

    packed = mojolearn.lowbit.pack(weights, "bfloat16")     # or "int8"
    blk = mojolearn.TransformerBlock(packed, n_heads=8)      # weight_format "bfloat16"
    f32 = mojolearn.lowbit.unpack(packed)                    # the materialized dict

Only 2-D tensors are packed: the projection matrices. Norm weights, biases,
convolution taps, `A_log`, `D` and every other 1-D or 3-D tensor stay
float32, because they never enter a GEMM and the profiles are about the
GEMM's operands. `pack` and `unpack` are inverses on the packed tensors up to
the rounding `pack` performs, and `unpack(pack(unpack(p)))` is `unpack(p)`.
"""
import numpy as np

from . import _backend
from ._bufcheck import dtype_name, is_native_f32, probe

FORMATS = ("float32", "bfloat16", "int8")

__all__ = ["FORMATS", "BF16Weight", "Int8Weight", "pack", "unpack", "materialize",
           "format_of", "is_packed", "pack_one", "materialize_one"]


class BF16Weight:
    """A 2-D float32 weight stored as bf16 bits (uint16), contract L-2."""

    format = "bfloat16"

    def __init__(self, bits):
        b = np.ascontiguousarray(bits)
        if b.dtype != np.uint16 or b.ndim != 2:
            raise TypeError("mojolearn.lowbit.BF16Weight: bits must be a 2-D uint16 array")
        self.bits = b
        self.shape = b.shape

    def __repr__(self):
        return f"BF16Weight(shape={self.shape})"


class Int8Weight:
    """A 2-D float32 weight stored as int8 codes with one int32 exponent per
    row, contract L-3 and L-4: `row = codes * 2^exponent`."""

    format = "int8"

    def __init__(self, codes, exponents):
        q = np.ascontiguousarray(codes)
        e = np.ascontiguousarray(exponents)
        if q.dtype != np.int8 or q.ndim != 2:
            raise TypeError("mojolearn.lowbit.Int8Weight: codes must be a 2-D int8 array")
        if e.dtype != np.int32 or e.shape != (q.shape[0],):
            raise TypeError("mojolearn.lowbit.Int8Weight: exponents must be int32 of shape (rows,)")
        self.codes = q
        self.exponents = e
        self.shape = q.shape

    def __repr__(self):
        return f"Int8Weight(shape={self.shape})"


def is_packed(value):
    return isinstance(value, (BF16Weight, Int8Weight))


def format_of(weights):
    """The one format the packed tensors of `weights` share, or "float32"
    when none is packed. Two formats in one dict are refused: a model is
    stored one way."""
    fmts = set()
    values = weights.values() if hasattr(weights, "values") else weights
    for v in values:
        if is_packed(v):
            fmts.add(v.format)
    if not fmts:
        return "float32"
    if len(fmts) > 1:
        raise ValueError(f"mojolearn.lowbit: weights mix formats {sorted(fmts)}; store a model one way")
    return fmts.pop()


# ----------------------------------------------------------------- packing


def _f32_2d(value, name):
    try:
        pb = probe(value)
    except TypeError:
        raise TypeError(f"mojolearn.lowbit: {name} does not support the buffer protocol") from None
    if not is_native_f32(pb.format):
        raise TypeError(f"mojolearn.lowbit: {name} has dtype {dtype_name(value, pb)}; pack takes float32")
    a = np.ascontiguousarray(np.asarray(value))
    if a.dtype != np.float32:
        a = a.view(np.float32) if a.dtype.itemsize == 4 else a.astype(np.float32)
    return a


def _conversion_backend():
    """Which exact spelling materializes: "gpu" (the linalg extension's
    kernels), "host" (the linalg host binding) or "numpy"."""
    try:
        from . import linalg
        if _backend.vendor() != "cpu":
            linalg.require_identical()
            return "gpu"
    except Exception:  # noqa: BLE001
        pass
    try:
        _backend.load_host_module("_mojolearn_linalg_host")
        return "host"
    except Exception:  # noqa: BLE001
        return "numpy"


def _addr(a):
    return a.__array_interface__["data"][0]


def _to_bf16_numpy(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # contract L-2: flush, then round to nearest even on the low 16 bits
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    u = np.where(sub, u & 0x80000000, u).astype(np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    rne = ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    quiet = ((u >> 16) | 0x0040).astype(np.uint16)
    return np.where(nan, quiet, rne).astype(np.uint16)


def _from_bf16_numpy(bits):
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _quantize_int8_numpy(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    xf = np.where(sub, (u & 0x80000000).astype(np.uint32), u).view(np.float32)
    absmax = np.max(np.abs(xf), axis=1)
    ub = absmax.view(np.uint32)
    exp = (((ub >> 23) & 0xFF).astype(np.int32) - 127) - 6
    exp = np.where(absmax == 0, np.int32(0), exp).astype(np.int32)
    scaled = (xf * np.ldexp(np.float32(1.0), -exp)[:, None].astype(np.float32)).astype(np.float32)
    us = scaled.view(np.uint32)
    sub2 = ((us & 0x7F800000) == 0) & ((us & 0x007FFFFF) != 0)
    scaled = np.where(sub2, (us & 0x80000000).astype(np.uint32), us).view(np.float32)
    # round half to even by the magic constant, |scaled| <= 128 << 2^22
    magic = np.float32(12582912.0)
    r = np.where(scaled >= 0, (scaled + magic) - magic, (scaled - magic) + magic).astype(np.float32)
    r = np.where(np.isnan(scaled), np.float32(0.0), r)
    r = np.clip(r, -127.0, 127.0)
    return r.astype(np.int8), exp


def _dequantize_int8_numpy(codes, exponents):
    y = (codes.astype(np.float32) * np.ldexp(np.float32(1.0), exponents)[:, None].astype(np.float32)).astype(np.float32)
    u = y.view(np.uint32)
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    return np.where(sub, (u & 0x80000000).astype(np.uint32), u).view(np.float32)


def pack_one(value, fmt, name="weight"):
    """One 2-D float32 tensor to `fmt`. "float32" returns a float32 copy."""
    if fmt not in FORMATS:
        raise ValueError(f"mojolearn.lowbit: unknown format {fmt!r}; one of {FORMATS}")
    a = _f32_2d(value, name)
    if a.ndim != 2:
        raise ValueError(f"mojolearn.lowbit: {name} must be 2-D to pack, got shape {a.shape}")
    if fmt == "float32":
        return a.copy()
    where = _conversion_backend()
    if fmt == "bfloat16":
        if where == "gpu":
            from . import linalg
            return BF16Weight(np.asarray(linalg.to_bf16(a)))
        if where == "host":
            h = _backend.load_host_module("_mojolearn_linalg_host")
            out = np.empty(a.shape, np.uint16)
            h.to_bf16(_addr(out), _addr(a), [int(a.size)])
            return BF16Weight(out)
        return BF16Weight(_to_bf16_numpy(a))
    if where == "gpu":
        from . import linalg
        q, e = linalg.quantize_int8(a)
        return Int8Weight(np.asarray(q), np.asarray(e))
    if where == "host":
        h = _backend.load_host_module("_mojolearn_linalg_host")
        q = np.empty(a.shape, np.int8)
        e = np.empty((a.shape[0],), np.int32)
        h.quantize_int8(_addr(q), _addr(e), _addr(a), [int(a.shape[0]), int(a.shape[1])])
        return Int8Weight(q, e)
    q, e = _quantize_int8_numpy(a)
    return Int8Weight(q, e)


def materialize_one(value, name="weight"):
    """A packed tensor as float32 (exact), or a float32 tensor as itself."""
    if isinstance(value, BF16Weight):
        where = _conversion_backend()
        if where == "gpu":
            from . import linalg
            return np.asarray(linalg.from_bf16(value.bits)).astype(np.float32, copy=False)
        if where == "host":
            h = _backend.load_host_module("_mojolearn_linalg_host")
            out = np.empty(value.shape, np.float32)
            h.from_bf16(_addr(out), _addr(value.bits), [int(value.bits.size)])
            return out
        return _from_bf16_numpy(value.bits)
    if isinstance(value, Int8Weight):
        where = _conversion_backend()
        if where == "gpu":
            from . import linalg
            return np.asarray(linalg.dequantize_int8(value.codes, value.exponents)).astype(np.float32, copy=False)
        if where == "host":
            h = _backend.load_host_module("_mojolearn_linalg_host")
            out = np.empty(value.shape, np.float32)
            h.dequantize_int8(_addr(out), _addr(value.codes), _addr(value.exponents),
                              [int(value.shape[0]), int(value.shape[1])])
            return out
        return _dequantize_int8_numpy(value.codes, value.exponents)
    return value


def pack(weights, fmt):
    """Every 2-D float32 tensor of a weight dict to `fmt`; everything else
    is passed through as float32. Returns a new dict."""
    if fmt not in FORMATS:
        raise ValueError(f"mojolearn.lowbit: unknown format {fmt!r}; one of {FORMATS}")
    if not hasattr(weights, "keys"):
        raise TypeError("mojolearn.lowbit.pack: weights must be a dict keyed by name")
    out = {}
    for name, value in weights.items():
        if is_packed(value):
            value = materialize_one(value, name)
        a = np.asarray(value)
        if fmt != "float32" and a.ndim == 2:
            out[name] = pack_one(a, fmt, name)
        else:
            out[name] = a
    return out


def unpack(weights, what="weights"):
    """`(float32 dict, format)`: every packed tensor materialized exactly,
    every other tensor as given. The hook the inference classes call."""
    if not hasattr(weights, "keys"):
        return weights, "float32"
    fmt = format_of(weights)
    if fmt == "float32":
        return weights, fmt
    return {name: materialize_one(v, name) for name, v in weights.items()}, fmt


materialize = unpack
