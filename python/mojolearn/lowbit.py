# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Low-bit weight storage for the inference classes, with float32 arithmetic.

A model's 2-D weight matrices can be stored as bfloat16 bits or as int8 codes with
one power-of-two exponent per row, half or a quarter of the float32 size on disk and
in memory. An inference class built from packed weights first converts them back to
float32 exactly (bfloat16 by widening, int8 as ``codes * 2**exponent``) and then runs
its normal float32 path, so it computes the same bits as the float32 block given the
converted weights. Packed weights are rounded weights: the model is not the original
float32 model, and the arithmetic is not lower precision.

`pack` converts every 2-D tensor of a weight dict, selected by rank, not by name
(so a Mamba-1 ``A_log`` of shape (d_inner, 16) is packed too). 1-D and 3-D tensors
(norm weights, biases, convolution taps, ``D``) stay float32. Use `pack_one` to pack
selected tensors by name. `unpack` (also `materialize`) returns the float32 dict and
its format; ``unpack(pack(unpack(p)[0], fmt))`` equals ``unpack(p)``.

Conversion runs on the GPU on a GPU install and on the host binding on a CPU-only
install; an install without the linalg extension raises an error. NumPy is not
required: packed tensors are `mojolearn.Array` objects and inputs can be any buffer.

Examples
--------
>>> import mojolearn
>>> packed = mojolearn.lowbit.pack(weights, "bfloat16")        # or "int8"
>>> blk = mojolearn.TransformerBlock(packed, n_heads=8)
>>> f32, fmt = mojolearn.lowbit.unpack(packed)
"""
from ._array import Array
from ._buffer import (
    as_f32_c, as_i8_c, as_i32_c, as_u16_c,
)
from ._bufcheck import base_format, dtype_name, is_native_f32, probe

FORMATS = ("float32", "bfloat16", "int8")

__all__ = ["FORMATS", "BF16Weight", "Int8Weight", "pack", "unpack", "materialize",
           "format_of", "is_packed", "pack_one", "materialize_one", "widen_bf16"]

# Contract gemm/IDENTICAL_LOWBIT_CONTRACT.md (profiles bf16f32.v1, int8i32.v1): bf16 is L-2
# (widen L-1), int8 is L-3/L-4 (dequantize L-6). The *-bf16w / *-int8w lanes of
# tools/identity_break.py and tests/test_lowbit_weights.py check packed == materialized fp32.
class BF16Weight:
    """A 2-D weight stored as bfloat16 bits.

    Parameters
    ----------
    bits : 2-D uint16 buffer
        bfloat16 bits, for example from `mojolearn.linalg.to_bf16`. Anything else
        raises `TypeError`.

    Attributes
    ----------
    bits : mojolearn.Array of uint16
    shape : tuple of int
    format : str
        ``'bfloat16'``.
    """

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
    """A 2-D weight stored as int8 codes with one int32 exponent per row: ``row = codes * 2**exponent``.

    Parameters
    ----------
    codes : 2-D int8 buffer
    exponents : int32 buffer of shape (rows,)
        Wrong dtypes or shapes raise `TypeError`.

    Attributes
    ----------
    codes : mojolearn.Array of int8
    exponents : mojolearn.Array of int32
    shape : tuple of int
    format : str
        ``'int8'``.
    """

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
    """Whether `value` is a packed weight (`BF16Weight` or `Int8Weight`)."""
    return isinstance(value, (BF16Weight, Int8Weight))


def format_of(weights):
    """The format shared by the packed tensors of `weights`, or ``'float32'`` when none is packed.

    Raises `ValueError` when the dict mixes formats.
    """
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
    # `_linalg_impl` itself, not the public `linalg` door (lane/lazy-imports-2,
    # 2026-10-11): the four functions used here (to_bf16, from_bf16,
    # quantize_int8, dequantize_int8) are the same objects `linalg` re-exports,
    # and the door also imports `Cholesky` (`_cholesky_impl`, the GP binding),
    # which put the GP binding in every lowbit user's closure (mamba1 among them).
    return _linalg_impl


def widen_bf16(bits):
    """Convert bfloat16 bits of any rank (a uint16 buffer) to float32, exactly."""
    b, _ = as_u16_c(bits, ndim=None, name="bits")
    return _linalg().from_bf16(b)


def pack_one(value, fmt, name="weight"):
    """Pack one 2-D float32 tensor to `fmt`.

    Parameters
    ----------
    value : 2-D float32 buffer
    fmt : {'float32', 'bfloat16', 'int8'}
        ``'float32'`` returns a float32 copy. Unknown formats and non-2-D tensors raise
        `ValueError`.
    name : str, default 'weight'
        Name used in error messages.

    Returns
    -------
    BF16Weight, Int8Weight or mojolearn.Array
    """
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
    """A packed tensor converted to float32 exactly, or a float32 tensor returned as is."""
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
    """Pack every 2-D tensor of a weight dict to `fmt`; other tensors pass through unchanged.

    Parameters
    ----------
    weights : dict
        Tensors by name. Anything but a dict raises `TypeError`.
    fmt : {'float32', 'bfloat16', 'int8'}
        Target format; unknown formats raise `ValueError`.

    Returns
    -------
    dict
        A new dict.
    """
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
    """Convert every packed tensor of a weight dict to float32, exactly.

    Parameters
    ----------
    weights : dict
        Tensors by name; other tensors pass through unchanged.
    what : str, default 'weights'
        Name used in error messages.

    Returns
    -------
    (dict, str)
        The float32 dict and the format the weights were stored in.
    """
    if not hasattr(weights, "keys"):
        return weights, "float32"
    fmt = format_of(weights)
    if fmt == "float32":
        return weights, fmt
    return {name: materialize_one(v, name) for name, v in weights.items()}, fmt  # glue: dispatches each named weight tensor


materialize = unpack
