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
kernels on a GPU install (only there: a GPU install never converts on the
host), through `_mojolearn_linalg_host` on a CPU-only install (the selector
installs it under the linalg binding's canonical name, so `mojolearn.linalg`
reaches it the same way). An install with no linalg binding at all refuses
by name (pyglue-sweep, 2026-10-03: Python is glue only, so the per-element
pure-Python spelling that used to run there is gone; the NumPy oracle in
the test file still holds the kernels and the host binding to the
`checks/numerics.mojo` seams).

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
from ._array import Array
from ._buffer import (
    as_f32_c, as_i8_c, as_i32_c, as_u16_c,
)
from ._bufcheck import base_format, dtype_name, is_native_f32, probe

FORMATS = ("float32", "bfloat16", "int8")

__all__ = ["FORMATS", "BF16Weight", "Int8Weight", "pack", "unpack", "materialize",
           "format_of", "is_packed", "pack_one", "materialize_one", "widen_bf16"]

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
    for v in values:  # glue: format tag per named tensor
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


def _linalg():
    """The linalg binding this install SELECTED: the GPU extension's kernels
    on a GPU install, the linalg host binding installed under the same
    canonical name on a CPU-only install. A GPU install never converts on
    the host (cpu-gpu-cleanup n-pyneural, 2026-10-02): a GPU binding that
    refuses, for instance a FAST tier, raises by name from `linalg`. An
    install with no linalg binding refuses here: the conversion is Mojo
    only (Python is glue only, pyglue-sweep 2026-10-03)."""
    from . import _linalg_impl
    try:
        _linalg_impl._load()
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "mojolearn.lowbit: packing and materializing need the linalg binding "
            f"(GPU extension or _mojolearn_linalg_host); none is loaded: {e}") from None
    from . import linalg
    return linalg


def widen_bf16(bits):
    """bf16 bits of ANY rank (a `'<u2'` buffer) to float32, exact, through
    whichever of the three spellings this install has. The model loader
    widens 1-D norm weights and the embedding table through this; the 2-D
    projections go through `BF16Weight` and `materialize_one`."""
    b, _ = as_u16_c(bits, ndim=None, name="bits")
    return _linalg().from_bf16(b)


def pack_one(value, fmt, name="weight"):
    """One 2-D float32 tensor to `fmt`. "float32" returns a float32 copy."""
    if fmt not in FORMATS:
        raise ValueError(f"mojolearn.lowbit: unknown format {fmt!r}; one of {FORMATS}")
    a = _f32_2d(value, name)
    if a.ndim != 2:
        raise ValueError(f"mojolearn.lowbit: {name} must be 2-D to pack, got shape {a.shape}")
    if fmt == "float32":
        return a.copy()
    linalg = _linalg()
    if fmt == "bfloat16":
        return BF16Weight(linalg.to_bf16(a))
    q, e = linalg.quantize_int8(a)
    return Int8Weight(q, e)


def materialize_one(value, name="weight"):
    """A packed tensor as float32 (exact), or a float32 tensor as itself."""
    if isinstance(value, BF16Weight):
        return _linalg().from_bf16(value.bits)
    if isinstance(value, Int8Weight):
        return _linalg().dequantize_int8(value.codes, value.exponents)
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
    for name, value in weights.items():  # glue: dispatches each named weight tensor
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
    return {name: materialize_one(v, name) for name, v in weights.items()}, fmt  # glue: dispatches each named weight tensor


materialize = unpack
