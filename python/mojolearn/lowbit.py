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
        return _NATIVE(self, "family_low_bf16_init", (bits,), _NATIVE_CONTEXT)

    def __repr__(self):
        return _NATIVE(self, "family_low_repr", (), _NATIVE_CONTEXT)


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
        return _NATIVE(self, "family_low_int8_init", (codes, exponents), _NATIVE_CONTEXT)

    def __repr__(self):
        return _NATIVE(self, "family_low_repr", (), _NATIVE_CONTEXT)


def is_packed(value):
    """Whether `value` is a packed weight (`BF16Weight` or `Int8Weight`)."""
    return _NATIVE(None, "family_low_is_packed", (value,), _NATIVE_CONTEXT)


def format_of(weights):
    """The format shared by the packed tensors of `weights`, or ``'float32'`` when none is packed.

    Raises `ValueError` when the dict mixes formats.
    """
    return _NATIVE(None, "family_low_format_of", (weights,), _NATIVE_CONTEXT)


# ----------------------------------------------------------------- packing






def widen_bf16(bits):
    """Convert bfloat16 bits of any rank (a uint16 buffer) to float32, exactly."""
    return _NATIVE(None, "family_low_widen_bf16", (bits,), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_low_pack_one", (value, fmt, name), _NATIVE_CONTEXT)


def materialize_one(value, name="weight"):
    """A packed tensor converted to float32 exactly, or a float32 tensor returned as is."""
    return _NATIVE(None, "family_low_materialize_one", (value, name), _NATIVE_CONTEXT)




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
    return _NATIVE(None, "family_low_pack", (weights, fmt), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_low_unpack", (weights, what), _NATIVE_CONTEXT)


materialize = unpack


# Import-time connection only; every operation above enters the shared native
# family owner and uses the existing linalg provider for numerical conversion.
from . import _linalg_impl as _connected_linalg
from . import _native_family_bridge
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(family_low_bf16=BF16Weight, family_low_int8=Int8Weight)
