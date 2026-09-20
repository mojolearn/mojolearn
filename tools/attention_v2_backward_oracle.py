#!/usr/bin/env python3
"""Executable float32 reference for deterministic tiled-attention-v2 backward.

This is an oracle, not a fast implementation.  It deliberately recomputes
the forward row from Q and K, then spells dQ in ascending key order and dK/dV
in ascending query order.  Device scheduling may not change those logical
folds.  TILE is part of the arithmetic contract.
"""
import numpy as np
from tools.attention_v2_oracle import exp32, _fma32

TILE = 32


def f32(x):
    return np.float32(x)


def _dot(a, b):
    acc = f32(0.0)
    for i in range(a.size):
        acc = _fma32(a[i], b[i], acc)
    return acc


def _scores(qrow, keys, scale, visible):
    out = np.full(keys.shape[0], f32(-np.inf), np.float32)
    for j in range(keys.shape[0]):
        if visible[j]:
            out[j] = f32(_dot(qrow, keys[j]) * f32(scale))
    return out


def _normalizer(scores, visible):
    m = f32(-np.inf)
    z = f32(0.0)
    for lo in range(0, scores.size, TILE):
        hi = min(lo + TILE, scores.size)
        tm = f32(-np.inf)
        for j in range(lo, hi):
            if visible[j]:
                x = f32(scores[j])
                if x > tm or (x == tm and not np.signbit(x)):
                    tm = x
        nm = tm if tm > m else m
        corr = f32(0.0) if np.isneginf(m) else exp32(f32(m - nm))
        z = f32(z * corr)
        for j in range(lo, hi):
            if visible[j]:
                z = f32(z + exp32(f32(scores[j] - nm)))
        m = nm
    return m, z


def _row_backward(qrow, keys, values, dyrow, scale, visible):
    scores = _scores(qrow, keys, scale, visible)
    m, z = _normalizer(scores, visible)
    if not np.isfinite(z) or not (z > f32(0.0)):
        raise ValueError("each query row must have at least one finite visible key")
    probs = np.zeros(keys.shape[0], np.float32)
    dyv = np.zeros(keys.shape[0], np.float32)
    zdot = f32(0.0)
    for j in range(keys.shape[0]):
        if visible[j]:
            probs[j] = f32(exp32(f32(scores[j] - m)) / z)
            dyv[j] = _dot(dyrow, values[j])
            zdot = _fma32(probs[j], dyv[j], zdot)
    ds = np.zeros(keys.shape[0], np.float32)
    dq = np.zeros(keys.shape[1], np.float32)
    for j in range(keys.shape[0]):
        if visible[j]:
            ds[j] = f32(f32(probs[j] * f32(dyv[j] - zdot)) * f32(scale))
            for d in range(keys.shape[1]):
                dq[d] = _fma32(ds[j], keys[j, d], dq[d])
    return probs, ds, dq


def attention_v2_backward(q, k, v, dy, scale=1.0, mask=None,
                          reverse_query_fold=False):
    """Return ``(dQ,dK,dV)`` for one head, all arrays float32.

    ``reverse_query_fold`` is a negative-control arm.  It changes only the
    dK/dV query fold and must move a non-associative fixture.
    """
    q = np.asarray(q, np.float32); k = np.asarray(k, np.float32)
    v = np.asarray(v, np.float32); dy = np.asarray(dy, np.float32)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or dy.ndim != 2:
        raise ValueError("q, k, v and dy must be rank-2")
    if q.shape[1] != k.shape[1] or q.shape[0] != dy.shape[0] or v.shape != (k.shape[0], dy.shape[1]):
        raise ValueError("attention dimensions do not agree")
    if q.shape[0] == 0 or k.shape[0] == 0:
        raise ValueError("query and key row counts must be positive")
    visible = np.ones((q.shape[0], k.shape[0]), bool) if mask is None else np.asarray(mask, bool)
    if visible.shape != (q.shape[0], k.shape[0]):
        raise ValueError("mask must have shape [query_rows,key_rows]")

    probs = np.zeros((q.shape[0], k.shape[0]), np.float32)
    ds = np.zeros_like(probs); dq = np.zeros_like(q)
    for i in range(q.shape[0]):
        probs[i], ds[i], dq[i] = _row_backward(q[i], k, v, dy[i], scale, visible[i])

    dk = np.zeros_like(k); dv = np.zeros_like(v)
    order = range(q.shape[0] - 1, -1, -1) if reverse_query_fold else range(q.shape[0])
    for j in range(k.shape[0]):
        for i in order:
            if visible[i, j]:
                for d in range(k.shape[1]):
                    dk[j, d] = _fma32(ds[i, j], q[i, d], dk[j, d])
                for d in range(v.shape[1]):
                    dv[j, d] = _fma32(probs[i, j], dy[i, d], dv[j, d])
    return dq, dk, dv
