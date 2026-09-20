#!/usr/bin/env python3
"""Normative host oracle for opt-in tiled-attention v2.

This is deliberately a different floating-point function from transformer v1.
Tiles and folds are logical contract quantities, never device tuning knobs.
"""
import math
import numpy as np

TILE = 32

def f32(x):
    return np.float32(x)

def exp32(x):
    # GPU work must replace this leaf with the repository's portable exp32
    # and prove its bits against the device columns.
    return f32(math.exp(float(f32(x))))

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
                out[d] = f32(out[d] + f32(w * f32(v[j, d])))
        m = nm
    return np.asarray([f32(x / z) for x in out], np.float32), m, z

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
