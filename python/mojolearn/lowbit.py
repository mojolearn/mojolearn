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
and through the pure-Python spelling below when neither is loaded (an
install with no binding at all). All three are the same exact integer
construction -- the pure-Python one is `checks/numerics.mojo`'s six seams
(`ftz`, `f32_to_bf16_bits_rne`, `bf16_bits_to_f32`, `int8_row_exponent`,
`quantize_int8_value`, `dequant_int8_pinned`) written over Python ints and
`struct`, NOT a NumPy vectorization of them -- and the test asserts that
where two are available they agree. The pure-Python spelling is a per-element
Python loop and is the spelling of last resort: an install with no binding
cannot run a block anyway, so nothing that matters is slow because of it.

NUMPY-FREE (lane/model-loader, 2026-09-17). This module reaches no NumPy:
packed tensors are `mojolearn.Array`s (`'<u2'` bits, `'<i1'` codes, `'<i4'`
exponents), inputs are anything with the buffer protocol, and the NumPy
oracle spellings that used to live here now live in the test file, where
NumPy is allowed. No bit moved: the three spellings are unchanged
constructions, so this rewrite takes no DEVIATION number.

    packed = mojolearn.lowbit.pack(weights, "bfloat16")     # or "int8"
    blk = mojolearn.TransformerBlock(packed, n_heads=8)      # weight_format "bfloat16"
    f32, fmt = mojolearn.lowbit.unpack(packed)               # (materialized dict, format)

Only 2-D tensors are packed. `pack` selects BY RANK, not by name: every 2-D
tensor of the dict is packed, which is the projection matrices and ALSO a
Mamba-1 `A_log` of shape (d_inner, 16) (the block materializes it exactly
like the rest; `test_lowbit_weights.py`'s Mamba-1 case packs it). Norm
weights, biases, convolution taps, `D` and every other 1-D or 3-D tensor
stay float32, because they never enter a GEMM and the profiles are about
the GEMM's operands. A caller that wants `A_log` kept float32 packs by name
with `pack_one` (the model loader does). `pack` and `unpack` are inverses on
the packed tensors up to the rounding `pack` performs, and
`unpack(pack(unpack(p)[0], fmt))` is `unpack(p)` (`unpack` returns the
materialized dict and its format).
"""
import array
from . import _portable_math as math
import struct

from . import _backend
from ._array import Array
from ._buffer import (
    addr as _addr_w, addr_ro as _addr_r, as_f32_c, as_i8_c, as_i32_c, as_u16_c, empty,
)
from ._bufcheck import base_format, dtype_name, is_native_f32, probe

FORMATS = ("float32", "bfloat16", "int8")

__all__ = ["FORMATS", "BF16Weight", "Int8Weight", "pack", "unpack", "materialize",
           "format_of", "is_packed", "pack_one", "materialize_one", "widen_bf16"]

if array.array("I").itemsize != 4 or array.array("H").itemsize != 2:  # pragma: no cover
    raise ImportError("mojolearn.lowbit: this platform's array('I')/('H') are not 32/16 bits")


class BF16Weight:
    """A 2-D float32 weight stored as bf16 bits (uint16), contract L-2."""

    format = "bfloat16"

    def __init__(self, bits):
        try:
            pb = probe(bits)
        except TypeError:
            raise TypeError("mojolearn.lowbit.BF16Weight: bits must be a 2-D uint16 array") from None
        if base_format(pb.format) != "H" or pb.itemsize != 2 or pb.ndim != 2:
            raise TypeError("mojolearn.lowbit.BF16Weight: bits must be a 2-D uint16 array")
        b, _ = as_u16_c(bits, ndim=2, name="bits")  # layout only; zero-copy when C-contiguous
        self.bits = b
        self.shape = tuple(b.shape)

    def __repr__(self):
        return f"BF16Weight(shape={self.shape})"


class Int8Weight:
    """A 2-D float32 weight stored as int8 codes with one int32 exponent per
    row, contract L-3 and L-4: `row = codes * 2^exponent`."""

    format = "int8"

    def __init__(self, codes, exponents):
        try:
            pq = probe(codes)
        except TypeError:
            raise TypeError("mojolearn.lowbit.Int8Weight: codes must be a 2-D int8 array") from None
        if base_format(pq.format) != "b" or pq.itemsize != 1 or pq.ndim != 2:
            raise TypeError("mojolearn.lowbit.Int8Weight: codes must be a 2-D int8 array")
        try:
            pe = probe(exponents)
        except TypeError:
            raise TypeError("mojolearn.lowbit.Int8Weight: exponents must be int32 of shape (rows,)") from None
        if (base_format(pe.format) not in ("i", "l") or pe.itemsize != 4
                or pe.shape != (pq.shape[0],)):
            raise TypeError("mojolearn.lowbit.Int8Weight: exponents must be int32 of shape (rows,)")
        q, _ = as_i8_c(codes, ndim=2, name="codes")
        e, _ = as_i32_c(exponents, ndim=1, name="exponents")
        self.codes = q
        self.exponents = e
        self.shape = tuple(q.shape)

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
    a, _ = as_f32_c(value, ndim=None, name=name)
    return a


def _conversion_backend():
    """Which exact spelling materializes: "gpu" (the linalg extension's
    kernels), "host" (the linalg host binding) or "python"."""
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
        return "python"


# ------------------------------------------------- the pure-Python spelling
#
# Each function below is one seam of `checks/numerics.mojo`, written over
# Python ints and `struct` so that it is the SAME integer construction and
# not a library's. `_f32` is the one rounding primitive: a Python float (an
# exact float64) to the nearest float32, ties to even, which is what the C
# `(float)` cast inside `struct.pack('<f')` does; a magnitude the cast would
# take to infinity is the infinity.


def _f32(v):
    try:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def _u32_of(a):
    """The float32 bits of a C-order float32 Array, as a uint32 array."""
    u = array.array("I")
    u.frombytes(a.tobytes())
    return u


def _ftz_bits(x):
    """`ftz` on the bit pattern: a subnormal becomes its signed zero."""
    if (x & 0x7F800000) == 0 and (x & 0x007FFFFF) != 0:
        return x & 0x80000000
    return x


def _ftz(v):
    """`ftz` on a float value: below the smallest normal is a signed zero."""
    if v != 0.0 and abs(v) < 1.1754943508222875e-38 and not math.isinf(v) and v == v:
        return math.copysign(0.0, v)
    return v


def _pow2_f32(e):
    """`pow2_f32` (DEVIATION 2903): the infinity above 127, `+0.0` below -126."""
    if e > 127:
        return math.inf
    if e < -126:
        return 0.0
    return math.ldexp(1.0, e)


def _to_bf16_py(a):
    """`f32_to_bf16_bits_rne` per element (DEVIATION 2901): flush, then a
    NaN keeps its sign and top payload and is forced quiet, else round to
    nearest even on the discarded 16 bits. Returns a `'<u2'` Array."""
    u = _u32_of(a)
    out = array.array("H", bytes(2 * len(u)))
    for k in range(len(u)):
        b = _ftz_bits(u[k])
        if (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0:
            out[k] = ((b >> 16) | 0x0040) & 0xFFFF
        else:
            out[k] = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return Array._owned(out, tuple(a.shape), "<u2", "C")


def _from_bf16_py(bits):
    """`bf16_bits_to_f32` per element (DEVIATION 2900): the shift, exact."""
    h = array.array("H")
    h.frombytes(bits.tobytes())
    u = array.array("I", bytes(4 * len(h)))
    for k in range(len(h)):
        u[k] = h[k] << 16
    f = array.array("f")
    f.frombytes(u.tobytes())
    return Array._owned(f, tuple(bits.shape), "<f4", "C")


def _quantize_int8_py(a):
    """`quantize_rows_int8` (DEVIATIONS 2902, 2903, 2905): per row the
    absmax after the flush (a maximum, order-free; a NaN never wins it, an
    infinity does), `e = floor(log2 absmax) - 6` (an all-zero row takes 0),
    then `clamp(rne(ftz(ftz(x) * 2^-e)), -127, 127)` with a NaN quantizing
    to zero. The rounding is ties-to-even on the exact value, which is what
    the magic-constant spelling computes for `|s| <= 2^22`."""
    rows, cols = a.shape
    u = _u32_of(a)
    for k in range(len(u)):
        u[k] = _ftz_bits(u[k])
    xf = array.array("f")
    xf.frombytes(u.tobytes())
    codes = array.array("b", bytes(rows * cols))
    exps = array.array("i", bytes(4 * rows))
    for r in range(rows):
        base = r * cols
        best = 0.0
        for c in range(cols):
            v = abs(xf[base + c])
            if v > best:  # a NaN compares false, exactly as `row_absmax`'s `>`
                best = v
        if best == 0.0 or best != best:
            e = 0
        else:
            # `f32_exponent(absmax) - INT8_TARGET_EXPONENT`, read off the
            # exponent field of the float32 bits (the infinity reads 128)
            bits = struct.unpack("<I", struct.pack("<f", best))[0] if not math.isinf(best) else 0x7F800000
            e = (((bits >> 23) & 0xFF) - 127) - 6
        exps[r] = e
        scale = _pow2_f32(-e)
        for c in range(cols):
            x = xf[base + c]
            if x != x:
                codes[base + c] = 0
                continue
            s = _ftz(_f32(x * scale))
            if s != s:  # 0 * inf on a row scaled by the infinity: the Mojo cast of a NaN is not defined; zero here
                codes[base + c] = 0
                continue
            if math.isinf(s):
                rr = 127.0 if s > 0 else -127.0
            else:
                rr = float(round(s))  # ties to even on the exact value
            if rr > 127.0:
                rr = 127.0
            if rr < -127.0:
                rr = -127.0
            codes[base + c] = int(rr)
    return (Array._owned(codes, (rows, cols), "<i1", "C"),
            Array._owned(exps, (rows,), "<i4", "C"))


def _dequantize_int8_py(codes, exponents):
    """`dequant_int8_pinned` per element (DEVIATION 2904): `ftz(q * 2^e)`,
    one multiply by a power of two, exact unless it lands below the smallest
    normal, where the flush makes it a signed zero."""
    rows, cols = codes.shape
    q = array.array("b")
    q.frombytes(codes.tobytes())
    e = array.array("i")
    e.frombytes(exponents.tobytes())
    out = array.array("f", bytes(4 * rows * cols))
    for r in range(rows):
        scale = _pow2_f32(e[r])
        base = r * cols
        for c in range(cols):
            out[base + c] = _ftz(_f32(float(q[base + c]) * scale))
    return Array._owned(out, (rows, cols), "<f4", "C")


def widen_bf16(bits):
    """bf16 bits of ANY rank (a `'<u2'` buffer) to float32, exact, through
    whichever of the three spellings this install has. The model loader
    widens 1-D norm weights and the embedding table through this; the 2-D
    projections go through `BF16Weight` and `materialize_one`."""
    b, _ = as_u16_c(bits, ndim=None, name="bits")
    where = _conversion_backend()
    if where == "gpu":
        from . import linalg
        return linalg.from_bf16(b)
    if where == "host":
        h = _backend.load_host_module("_mojolearn_linalg_host")
        out = empty(tuple(b.shape), "<f4")
        h.from_bf16(_addr_w(out, name="out"), _addr_r(b, name="bits"), [int(b.size)])
        return out
    return _from_bf16_py(b)


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
            return BF16Weight(linalg.to_bf16(a))
        if where == "host":
            h = _backend.load_host_module("_mojolearn_linalg_host")
            out = empty(tuple(a.shape), "<u2")
            h.to_bf16(_addr_w(out, name="out"), _addr_r(a, name=name), [int(a.size)])
            return BF16Weight(out)
        return BF16Weight(_to_bf16_py(a))
    if where == "gpu":
        from . import linalg
        q, e = linalg.quantize_int8(a)
        return Int8Weight(q, e)
    if where == "host":
        h = _backend.load_host_module("_mojolearn_linalg_host")
        q = empty(tuple(a.shape), "<i1")
        e = empty((a.shape[0],), "<i4")
        h.quantize_int8(_addr_w(q, name="codes"), _addr_w(e, name="exponents"),
                        _addr_r(a, name=name), [int(a.shape[0]), int(a.shape[1])])
        return Int8Weight(q, e)
    q, e = _quantize_int8_py(a)
    return Int8Weight(q, e)


def materialize_one(value, name="weight"):
    """A packed tensor as float32 (exact), or a float32 tensor as itself."""
    if isinstance(value, BF16Weight):
        where = _conversion_backend()
        if where == "gpu":
            from . import linalg
            return linalg.from_bf16(value.bits)
        if where == "host":
            h = _backend.load_host_module("_mojolearn_linalg_host")
            out = empty(value.shape, "<f4")
            h.from_bf16(_addr_w(out, name="out"), _addr_r(value.bits, name=name), [int(value.bits.size)])
            return out
        return _from_bf16_py(value.bits)
    if isinstance(value, Int8Weight):
        where = _conversion_backend()
        if where == "gpu":
            from . import linalg
            return linalg.dequantize_int8(value.codes, value.exponents)
        if where == "host":
            h = _backend.load_host_module("_mojolearn_linalg_host")
            out = empty(value.shape, "<f4")
            h.dequantize_int8(_addr_w(out, name="out"), _addr_r(value.codes, name=name),
                              _addr_r(value.exponents, name=name),
                              [int(value.shape[0]), int(value.shape[1])])
            return out
        return _dequantize_int8_py(value.codes, value.exponents)
    return value


def _as_given(value, name):
    """A non-packed tensor as it was given when it has a buffer, else (a
    nested list) as a float32 Array; `(array, ndim)`."""
    try:
        pb = probe(value)
        return value, pb.ndim
    except TypeError:
        a = Array.from_list(value, "<f4")
        return a, a.ndim


def pack(weights, fmt):
    """Every 2-D float32 tensor of a weight dict to `fmt`; everything else
    is passed through as given. Returns a new dict."""
    if fmt not in FORMATS:
        raise ValueError(f"mojolearn.lowbit: unknown format {fmt!r}; one of {FORMATS}")
    if not hasattr(weights, "keys"):
        raise TypeError("mojolearn.lowbit.pack: weights must be a dict keyed by name")
    out = {}
    for name, value in weights.items():
        if is_packed(value):
            # Repacking in the format it already has needs independent
            # storage (as the float32 arm below provides), but not a full
            # float32 expansion followed by the same quantizer. Besides
            # wasting bandwidth, that old route briefly needed 4 bytes per
            # weight for a model whose packed representation needs 1 or 2.
            if value.format == fmt:
                if isinstance(value, BF16Weight):
                    out[name] = BF16Weight(value.bits.copy())
                else:
                    out[name] = Int8Weight(value.codes.copy(), value.exponents.copy())
                continue
            value = materialize_one(value, name)
        a, ndim = _as_given(value, name)
        if fmt != "float32" and ndim == 2:
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
