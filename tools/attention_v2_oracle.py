#!/usr/bin/env python3
"""Normative host oracle for opt-in tiled-attention v2.

This is deliberately a different floating-point function from transformer v1.
Tiles and folds are logical contract quantities, never device tuning knobs.
"""
import math, struct
import numpy as np

TILE = 32

def f32(x):
    return np.float32(x)

try:
    _fma64 = math.fma
except AttributeError:  # math.fma is Python 3.13+; the wheel supports 3.10 up
    from fractions import Fraction

    def _fma64(a, b, c):
        """a*b+c with ONE rounding, as math.fma: exact rational arithmetic,
        then Fraction.__float__, which CPython rounds correctly (int/int true
        division). Non-finite operands follow math.fma: NaN propagates,
        inf*0 and inf-inf raise ValueError, an infinite addend wins over a
        finite product even when that product overflows a double; an exact
        zero keeps its sign by the IEEE 754 sum rule (-0 only when both
        addends are -0)."""
        if math.isnan(a) or math.isnan(b) or math.isnan(c):
            return math.nan
        if math.isinf(a) or math.isinf(b):
            if a == 0.0 or b == 0.0:
                raise ValueError("invalid operation in fma")
            product = math.copysign(math.inf, math.copysign(1.0, a) * math.copysign(1.0, b))
            if math.isinf(c) and c != product:
                raise ValueError("invalid operation in fma")
            return product
        if math.isinf(c):
            return c
        exact = Fraction(a) * Fraction(b) + Fraction(c)
        if exact == 0:
            product_negative_zero = a * b == 0.0 and math.copysign(1.0, a) != math.copysign(1.0, b)
            return -0.0 if product_negative_zero and c == 0.0 and math.copysign(1.0, c) < 0 else 0.0
        return float(exact)  # OverflowError past the float range, as math.fma raises

def _fma32(a,b,c):
    return f32(_fma64(float(f32(a)),float(f32(b)),float(f32(c))))

def _pow2(k):
    return np.frombuffer(struct.pack('<I',(k+127)<<23),dtype='<f4')[0]

def exp32(x):
    """Bit transcription of checks/numerics.mojo::portable_expf."""
    x=f32(x)
    if np.isnan(x): return x
    if x > f32(88.722835): return f32(np.inf)
    if x < f32(-87.33655): return f32(0)
    t=f32(f32(x*f32(1.4426950408889634))+f32(.5)); zf=math.floor(float(t)); k=int(zf)
    r=_fma32(f32(zf),f32(-.693359375),x); r=_fma32(f32(zf),f32(2.12194440e-4),r)
    q=f32(1.9875691500e-4)
    for c in (1.3981999507e-3,8.3334519073e-3,4.1665795894e-2,1.6666665459e-1,5.0000001201e-1): q=_fma32(q,r,f32(c))
    y=f32(_fma32(q,f32(r*r),r)+f32(1)); k1=k>>1; k2=k-k1; y=f32(f32(y*_pow2(k1))*_pow2(k2))
    return f32(0) if y < f32(1.1754943508222875e-38) else y

def div32(a,b):
    """Host spelling of checks/numerics.mojo::portable_divf."""
    a=f32(a); b=f32(b); tiny=f32(1.1754943508222875e-38)
    if abs(a)<tiny: a=f32(-0.0 if np.signbit(a) else 0.0)
    if abs(b)<tiny: b=f32(-0.0 if np.signbit(b) else 0.0)
    y=f32(a/b)
    return f32(-0.0 if np.signbit(y) else 0.0) if abs(y)<tiny else y

def online_row(scores, values, tile=TILE):
    """Fixed-tile, serial-tile online softmax and weighted-value row."""
    s = np.asarray(scores, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    if s.ndim != 1 or v.ndim != 2 or v.shape[0] != s.size or s.size == 0:
        raise ValueError("scores [S] and values [S,D], S > 0 required")
    if tile != TILE:
        raise ValueError(f"v2 tile is fixed at {TILE}")
    m = f32(-np.inf); z = f32(0.0); out = np.zeros(v.shape[1], np.float32)
    for lo in range(0, s.size, TILE):
        hi = min(lo + TILE, s.size)
        tm = f32(-np.inf)
        for j in range(lo, hi):
            # identical_fmax spelling: equal signed zeros choose +0.
            x = f32(s[j])
            if x > tm or (x == tm and not np.signbit(x)):
                tm = x
        nm = tm if tm > m else m
        corr = f32(0.0) if np.isneginf(m) else exp32(f32(m - nm))
        z = f32(z * corr)
        out = np.asarray([f32(x * corr) for x in out], np.float32)
        for j in range(lo, hi):
            w = exp32(f32(s[j] - nm))
            z = f32(z + w)
            for d in range(v.shape[1]):
                out[d] = _fma32(w, f32(v[j, d]), out[d])
        m = nm
    return np.asarray([div32(x,z) for x in out], np.float32), m, z

def eager_row(scores, values):
    """Small host witness for v1-style materialized max/sum; not its oracle."""
    s=np.asarray(scores,np.float32); v=np.asarray(values,np.float32)
    m=f32(-np.inf)
    for x in s:
        x=f32(x)
        if x > m or (x == m and not np.signbit(x)): m=x
    ws=[]; z=f32(0)
    for x in s:
        w=exp32(f32(x-m)); ws.append(w); z=f32(z+w)
    out=np.zeros(v.shape[1],np.float32)
    for j,w in enumerate(ws):
        p=f32(w/z)
        for d in range(v.shape[1]): out[d]=f32(out[d]+f32(p*f32(v[j,d])))
    return out

def workspace_bytes(batch, heads, q_rows, value_width):
    """Persistent v2 forward metadata: max, denominator, output only."""
    return 4 * batch * heads * q_rows * (2 + value_width)

def materialized_bytes(batch, heads, q_rows, kv_rows):
    """v1 score/probability pair, excluding Q/K/V and other stages."""
    return 8 * batch * heads * q_rows * kv_rows
