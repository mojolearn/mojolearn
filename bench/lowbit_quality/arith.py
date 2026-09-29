# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Reference arithmetic of the low-bit candidates, in PyTorch.

Lane lane/lowbit-quality, 2026-09-29. Plan `docs/lanes/LOWBIT_UNITS_PLAN.md`,
contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md` (clauses L-1 to L-7), seams
`checks/numerics.mojo` (the block headed LOW-BIT STORAGE SEAMS).

WHAT THIS IS. A simulation of what a matrix product computes under each
candidate, so that the loss from rounding values into codes can be measured
on a real model before any kernel is written. It is not a kernel and it
carries no speed meaning.

EVERY PRODUCT IS OP_NT: `C = A @ B^T`, `A` of shape `[..., m, k]` and `B` of
shape `[..., n, k]`, each operand's rows running along the contracted extent
`k`. That is the only orientation in which a per-row scale on both operands
is a per-cell scale of the output (contract section 0). A caller whose
product is written another way passes the transposed operand, and the rows
the quantizer sees are then the rows the GEMM would see.

OPERAND KINDS.
  fp32    the value as given.
  bf16    L-2 then L-1: flush, round to nearest even on the low sixteen
          bits, widen by the shift. The product accumulates in fp32.
  int8    L-3, L-4: per row `e = floor(log2 absmax) - 6`, so the row's
          largest magnitude lands in [64, 128); code
          `clamp(rne(ftz(ftz(x) * 2^-e)), -127, 127)`.
  int15   THE SAME RULE ONE STEP WIDER, defined by this lane because no
          contract clause exists for it yet: `e = floor(log2 absmax) - 13`,
          so the largest magnitude lands in [8192, 16384); code clamped to
          [-16383, 16383]. A signed fifteen-bit integer, which is what two
          int8 pieces of seven magnitude bits each can carry.

INTEGER PRODUCTS ARE EXACT HERE. Codes are held as float64 integers and the
matrix product runs in float64. Every partial sum is an integer of magnitude
at most `qmax_a * qmax_b * k`, which `_check_exact` holds below 2^53, so
each is exactly representable and the sum is the same integer under every
order of addition, as it is in an Int32 (or, for int15, a wider integer)
accumulator. Then:
  L-5  the sum to float32, round to nearest even. `i32_to_f32_pinned` is a
       20-bit and a 12-bit part, both exact, and ONE IEEE addition, which
       is the correctly rounded float32 of the integer; the float64 to
       float32 cast is the same rounding. `quantizer_check.py` compares the
       two spellings value by value.
  L-6  one multiply by `2^(ea + eb)` (the infinity above 127, `+0.0` below
       -126), then the flush.

WHAT IS NOT PINNED HERE. The fp32 accumulation of the `fp32` and `bf16`
kinds is torch's float32 matmul, not fp32.v1's fold tree. The two differ by
the order of additions. `infer_eval.py` measures how much that order can
move the metric with the arm `fp32-acc64` and reports it as the numerical
floor of the inference table.
"""
import torch

F32_MIN_NORMAL = 1.1754943508222875e-38

#: kind -> (target exponent, largest code magnitude)
INT_SPECS = {"int8": (6, 127), "int15": (13, 16383)}
FLOAT_KINDS = ("fp32", "bf16")
KINDS = FLOAT_KINDS + tuple(INT_SPECS)

#: Sabotage arms of the quantizer, for `quantizer_check.py` only. A check
#: that cannot fail is not a check: each of these must be SEEN to disagree
#: with `mojolearn.lowbit.pack` on real tensors.
ROUNDINGS = ("rne", "half_away", "truncate")


def ftz(x):
    """`ftz`: a float32 below the smallest normal becomes its signed zero."""
    assert x.dtype == torch.float32, x.dtype
    return torch.where(x.abs() < F32_MIN_NORMAL, torch.copysign(torch.zeros_like(x), x), x)


def pow2_f32(e):
    """`pow2_f32` (DEVIATION 2903): `2^e` from its exponent field; the
    infinity above 127 and `+0.0` below -126."""
    e = e.to(torch.int64)
    bits = ((e.clamp(-126, 127) + 127) << 23).to(torch.int32).contiguous()
    v = bits.view(torch.float32)
    v = torch.where(e > 127, torch.full_like(v, float("inf")), v)
    return torch.where(e < -126, torch.zeros_like(v), v)


def row_exponent(xf, target):
    """`int8_row_exponent` (L-3) with the target exponent as a parameter.
    `xf` is already flushed. The absmax is a maximum: exact and order-free.
    A NaN never wins it, an infinity does (its exponent field reads 128)."""
    a = xf.abs()
    a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    absmax = a.amax(dim=-1).contiguous()
    field = (absmax.view(torch.int32) >> 23) & 0xFF
    e = field - 127 - target
    return torch.where(absmax == 0, torch.zeros_like(e), e)


def quantize_rows(x, kind, rounding="rne", target_shift=0):
    """`(codes, exponents)`: codes as float64 integers, one int32 exponent
    per row. `rounding` and `target_shift` other than the defaults are the
    sabotage arms."""
    target, qmax = INT_SPECS[kind]
    xf = ftz(x)
    e = row_exponent(xf, target + target_shift)
    s = ftz(xf * pow2_f32(-e).unsqueeze(-1))
    if rounding == "rne":
        r = torch.round(s)  # ties to even
    elif rounding == "half_away":
        r = torch.sign(s) * torch.floor(s.abs() + 0.5)
    elif rounding == "truncate":
        r = torch.trunc(s)
    else:
        raise ValueError(rounding)
    r = torch.nan_to_num(r, nan=0.0, posinf=float(qmax), neginf=-float(qmax))
    return r.clamp(-qmax, qmax).to(torch.float64), e


def round_bf16(x, return_bits=False):
    """L-2 then L-1: the float32 a bf16 operand widens to."""
    xf = ftz(x).contiguous()
    b = xf.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    isnan = ((b & 0x7F800000) == 0x7F800000) & ((b & 0x007FFFFF) != 0)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    r = torch.where(isnan, ((b >> 16) | 0x0040) & 0xFFFF, r)
    if return_bits:
        return r
    w = r << 16
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)
    return w.view(torch.float32)


def dequantize_rows(codes, e):
    """L-6 on a stored operand: `ftz(code * 2^e)`, float32, exact unless it
    leaves the normal range. What `mojolearn.lowbit.materialize_one` does."""
    return ftz(codes.to(torch.float32) * pow2_f32(e).unsqueeze(-1))


class Operand:
    """One prepared operand: integer codes with exponents, or a float32."""

    __slots__ = ("codes", "e", "f", "qmax", "k")

    def __init__(self, codes=None, e=None, f=None, qmax=None):
        self.codes, self.e, self.f, self.qmax = codes, e, f, qmax
        self.k = (codes if codes is not None else f).shape[-1]

    @property
    def is_int(self):
        return self.codes is not None

    def as_f32(self):
        return dequantize_rows(self.codes, self.e) if self.is_int else self.f

    def rows(self, lo, hi):
        """Rows `lo:hi` of a 2-D operand. Row scales are per row, so a
        product over a slice of rows is the same cells of the whole."""
        if self.is_int:
            return Operand(codes=self.codes[lo:hi], e=self.e[lo:hi], qmax=self.qmax)
        return Operand(f=self.f[lo:hi])


def prepare(x, kind):
    if kind in INT_SPECS:
        codes, e = quantize_rows(x, kind)
        return Operand(codes=codes, e=e, qmax=INT_SPECS[kind][1])
    if kind == "bf16":
        return Operand(f=round_bf16(x))
    if kind == "fp32":
        return Operand(f=x)
    raise ValueError(f"unknown operand kind {kind!r}; one of {KINDS}")


def _check_exact(a, b):
    bound = a.qmax * b.qmax * a.k
    if bound >= 2 ** 53:
        raise ValueError(f"integer product bound {bound} is not exact in float64")


def product_prepared(a, b, acc64=False):
    """`A @ B^T` of two prepared operands, float32 out.

    Two integer operands: the exact integer sum, L-5, L-6. Anything else:
    the float32 product of the operands' float32 values, an integer operand
    entering as its exact dequantization. `acc64` accumulates the float
    product in float64 and rounds once, for the numerical floor arm."""
    if a.k != b.k:
        raise ValueError(f"contracted extents differ: {a.k} and {b.k}")
    if a.is_int and b.is_int:
        _check_exact(a, b)
        acc = a.codes @ b.codes.transpose(-1, -2)
        scale = pow2_f32(a.e.unsqueeze(-1).to(torch.int64) + b.e.unsqueeze(-2).to(torch.int64))
        return ftz(acc.to(torch.float32) * scale)
    fa, fb = a.as_f32(), b.as_f32()
    if acc64:
        return (fa.to(torch.float64) @ fb.to(torch.float64).transpose(-1, -2)).to(torch.float32)
    return fa @ fb.transpose(-1, -2)


def product_nt(A, B, kind_a, kind_b, acc64=False):
    """`A @ B^T` with `A` of kind `kind_a` and `B` of kind `kind_b`."""
    return product_prepared(prepare(A, kind_a), prepare(B, kind_b), acc64=acc64)


class Spec:
    """One arm: which kind each operand of each product takes.

    `w`     the kind of a WEIGHT operand (the right operand of a projection)
    `a`     the kind of an ACTIVATION operand (the left operand of a
            projection, and both operands of an attention product)
    `attn`  whether the attention products (QK and PV) are replaced; when
            False they stay fp32 whatever `a` says
    `acc64` the numerical floor arm: float products accumulate in float64
    `overrides`  product name prefix or suffix -> (kind_a, kind_b), for the
            follow-up arms that keep named products in fp32
    """

    def __init__(self, name, w="fp32", a="fp32", attn=False, acc64=False, overrides=None, note=""):
        for kind in (w, a):
            if kind not in KINDS:
                raise ValueError(f"unknown operand kind {kind!r}")
        self.name, self.w, self.a, self.attn, self.acc64 = name, w, a, attn, acc64
        self.overrides = dict(overrides or {})
        self.note = note

    def kinds(self, product):
        """`(kind_a, kind_b)` of the product named `product`. Names:
        `layers.<i>.<q|k|v|o|gate|up|down>_proj`, `lm_head`,
        `layers.<i>.attn_qk`, `layers.<i>.attn_pv`."""
        for key, kinds in self.overrides.items():
            if product == key or product.endswith("." + key) or product.startswith(key + "."):
                return kinds
        if product.endswith("attn_qk") or product.endswith("attn_pv"):
            return (self.a, self.a) if self.attn else ("fp32", "fp32")
        return (self.a, self.w)

    def describe(self):
        return dict(name=self.name, weight_kind=self.w, activation_kind=self.a,
                    attention_products=self.attn, acc64=self.acc64,
                    overrides={k: list(v) for k, v in self.overrides.items()}, note=self.note)
