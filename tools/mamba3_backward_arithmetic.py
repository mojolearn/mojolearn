#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned Mamba3 reduction/join arithmetic, separate from operand semantics.

This module cannot certify gradients by itself. The caller must independently
check every external gradient and forward operand, then require exact output
bits for every operation here. In particular a matching staged backward is
never a substitute for the whole-forward public-gradient oracle.
"""
import ctypes
import ctypes.util
import hashlib
from pathlib import Path

import numpy as np

from mamba3_join_diagnostics import ftz, pinned_join


FORWARD_OPERANDS = ("rot.k", "bcnorm.B", "bcnorm.C", "B_bias", "C_bias", "dt.out", "trap.sigma", "angle.theta")
GRADIENT_OPERANDS = (
    "stage.qkdot.out", "partial.qkdot.dt", "partial.s16.kscale",
    "partial.s17.recur.kscale", "partial.angle.dt", "partial.dt.from_seg",
    "partial.join.dt.from_adt", "partial.join.angle.dt",
)
OUTPUTS = (
    "partial.qkdot.gamma", "partial.s15.scale", "partial.scale.gamma",
    "partial.scale.beta", "partial.gamma.total", "partial.dt.current_total",
    "partial.dt.available_total", "partial.dt.with_seg", "partial.join.kscale",
    "partial.join.s15.scale", "partial.join.gamma.total",
    "partial.join.dt.current_total", "partial.join.dt.available_total",
)


def mul(left, right):
    return ftz(ftz(left) * ftz(right))


def serial_fma_dot(left, right):
    """Use C's correctly rounded float32 fma, including cancellation ties.

    A float64 multiply/add followed by float32 conversion can double-round;
    it is deliberately not used to emulate the pinned instruction.
    """
    left, right = ftz(left), ftz(right)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("FMA dot operands must have matching [rows, width] shapes")
    lib = ctypes.CDLL(ctypes.util.find_library("m") or None)
    fma = lib.fmaf
    fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fma.restype = ctypes.c_float
    result = np.zeros(left.shape[0], dtype=np.float32)
    for row in range(left.shape[0]):
        acc = 0.0
        for column in range(left.shape[1]):
            acc = fma(float(left[row, column]), float(right[row, column]), acc)
            if 0 < abs(acc) < float(np.finfo(np.float32).tiny):
                acc = -0.0 if acc < 0 else 0.0
        result[row] = acc
    return result


def evaluate(gradients, forward):
    """Reconstruct all thirteen outputs from external captured operands.

    Dependent operations consume reconstructed values, not captured outputs:
    an error in an earlier result cannot be hidden by a consistent downstream
    error. Every reconstructed value is subsequently compared to its dump.
    """
    def grad(name):
        return ftz(gradients[name]).reshape(-1)

    dt = ftz(forward["dt.out"])
    sigma = ftz(forward["trap.sigma"])
    if dt.ndim == 2 and sigma.shape == dt.shape:
        # Forward stages flatten batch/token, whereas S16 gradients retain
        # [B,L,H,N]. Recover boundaries explicitly; never shift beta between
        # batches merely because adjacent flattened rows exist.
        s16_shape = np.shape(gradients["partial.s16.kscale"])
        if len(s16_shape) != 4 or s16_shape[-1] != 128:
            raise ValueError("flattened dt requires an explicit [B,L,H,128] S16 operand")
        dt = dt.reshape(s16_shape[:3])
        sigma = sigma.reshape(s16_shape[:3])
    if dt.ndim != 3 or sigma.shape != dt.shape:
        raise ValueError("dt/sigma must have matching [batch, length, heads] shapes")
    batch, length, heads = dt.shape
    rows = batch * length * heads
    krot = ftz(forward["rot.k"]).reshape(rows, 128)
    qb = ftz(forward["bcnorm.C"]).reshape(batch * length, 1, 128)
    kb = ftz(forward["bcnorm.B"]).reshape(batch * length, 1, 128)
    q = ftz(qb + ftz(forward["C_bias"]).reshape(1, heads, 128)).reshape(rows, 128)
    k = ftz(kb + ftz(forward["B_bias"]).reshape(1, heads, 128)).reshape(rows, 128)
    out = {}
    out["partial.qkdot.gamma"] = mul(grad("stage.qkdot.out"), serial_fma_dot(q, k))
    scale = serial_fma_dot(grad("partial.s16.kscale").reshape(rows, 128), krot)
    out["partial.s15.scale"] = scale
    out["partial.scale.gamma"] = scale.copy()
    out["partial.scale.beta"] = scale.copy()
    out["partial.gamma.total"] = pinned_join((out["partial.qkdot.gamma"], scale))

    def current_dt(scale_gradient):
        sg = scale_gradient.reshape(batch, length, heads)
        current = ftz(grad("partial.qkdot.dt").reshape(dt.shape) + mul(sg, sigma))
        shifted = mul(sg[:, :-1], ftz(np.float32(1.0) - sigma[:, 1:]))
        current[:, 1:] = ftz(current[:, 1:] + shifted)
        return current.reshape(-1)

    out["partial.dt.current_total"] = current_dt(scale)
    out["partial.dt.available_total"] = pinned_join((out["partial.dt.current_total"], grad("partial.angle.dt")))
    out["partial.dt.with_seg"] = pinned_join((out["partial.dt.available_total"], grad("partial.dt.from_seg")))
    joined_k = pinned_join((grad("partial.s16.kscale"), grad("partial.s17.recur.kscale")))
    out["partial.join.kscale"] = joined_k
    joined_scale = serial_fma_dot(joined_k.reshape(rows, 128), krot)
    out["partial.join.s15.scale"] = joined_scale
    out["partial.join.gamma.total"] = pinned_join((out["partial.qkdot.gamma"], joined_scale))
    out["partial.join.dt.current_total"] = current_dt(joined_scale)
    out["partial.join.dt.available_total"] = pinned_join((
        out["partial.join.dt.current_total"], grad("partial.join.dt.from_adt"), grad("partial.join.angle.dt"),
    ))
    return out


def audit(oracle_dir, actual_dir, manifest, dump, rtol, atol):
    """Check independent external operands and all reconstructed output bits."""
    oracle_dir, actual_dir = Path(oracle_dir), Path(actual_dir)
    if dump.get("numeric_mode") != "IDENTICAL":
        raise ValueError("Mamba3 arithmetic contract requires explicit IDENTICAL provenance")
    if (set(manifest.get("forward_operands", {})) != set(FORWARD_OPERANDS)
            or dump.get("forward_operands") != list(FORWARD_OPERANDS)):
        raise ValueError("Mamba3 arithmetic contract requires all eight forward operands")
    failures, reports = [], []

    def reference(entry, key="file", dtype="<f8", sha="sha256"):
        raw = (oracle_dir / entry[key]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry[sha]:
            raise ValueError(f"reference hash mismatch: {entry[key]}")
        values = np.frombuffer(raw, dtype=dtype)
        if values.size != np.prod(entry["shape"]) or not np.isfinite(values).all():
            raise ValueError(f"invalid reference shape/values: {entry[key]}")
        return values.reshape(entry["shape"])

    def actual(name, entry, prefix):
        values = np.fromfile(actual_dir / f"{prefix}.{name}.f32", dtype="<f4")
        if values.size != np.prod(entry["shape"]) or not np.isfinite(values).all():
            raise ValueError(f"invalid native shape/values: {name}")
        return values.reshape(entry["shape"])

    def semantic(name, values, expected, kind):
        if not np.allclose(values.astype(np.float64), expected.astype(np.float64),
                           rtol=rtol, atol=atol, equal_nan=False):
            failures.append(f"{name}: independent {kind} operand semantics failed")

    forward, gradients = {}, {}
    for name in FORWARD_OPERANDS:
        entry = manifest["forward_operands"][name]
        forward[name] = actual(name, entry, "operand")
        if name != "rot.k":
            semantic(name, forward[name], reference(entry), "forward float64")
    # Rotation has cancellation around zero, after an accumulated angle.
    # Require independently correct base/bias/angle operands above, then
    # independently evaluate the rotation in float64 at those exact inputs.
    # This is a local forward semantic gate, not a second staged backward.
    heads = forward["B_bias"].shape[0]
    biased = forward["bcnorm.B"].astype(np.float64).reshape(-1, 1, 128) + forward["B_bias"].astype(np.float64).reshape(1, heads, 128)
    pair = biased.reshape(-1, heads, 64, 2)
    theta = forward["angle.theta"].astype(np.float64).reshape(-1, heads, 32)
    cosine = np.pad(np.cos(theta), ((0, 0), (0, 0), (0, 32)), constant_values=1)
    sine = np.pad(np.sin(theta), ((0, 0), (0, 0), (0, 32)))
    rotated = np.stack((pair[..., 0]*cosine - pair[..., 1]*sine,
                        pair[..., 0]*sine + pair[..., 1]*cosine), axis=-1).reshape(forward["rot.k"].shape)
    semantic("rot.k", forward["rot.k"], rotated, "rotation at exact native operands float64")
    direct_rot = reference(manifest["forward_operands"]["rot.k"])
    delta = np.abs(forward["rot.k"].astype(np.float64) - direct_rot)
    reports.append(f"rot.k: compositional forward semantics; direct-float64 maxabs={float(delta.max()):.3e} bad={int((delta > atol + rtol * np.abs(direct_rot)).sum())}")
    for name in GRADIENT_OPERANDS:
        if name not in dump.get("tensors", []):
            raise ValueError(f"missing declared external gradient operand: {name}")
        entry = manifest["gradients"][name]
        gradients[name] = actual(name, entry, "grad")
        semantic(name, gradients[name], reference(entry), "float64")
        semantic(name, gradients[name], reference(entry, "ref32_file", "<f4", "ref32_sha256"), "float32")
    reconstructed = evaluate(gradients, forward)
    for name in OUTPUTS:
        if name not in dump.get("tensors", []):
            raise ValueError(f"missing declared arithmetic output: {name}")
        entry = manifest["gradients"][name]
        values = actual(name, entry, "grad").reshape(-1)
        computed = reconstructed[name].reshape(-1)
        equal = values.shape == computed.shape and np.array_equal(
            values.view(np.uint32), computed.view(np.uint32))
        report = f"{name}: reconstructed DAG bitwise={equal}"
        for kind, key, dtype, sha in (("float64", "file", "<f8", "sha256"),
                                      ("float32", "ref32_file", "<f4", "ref32_sha256")):
            expected = reference(entry, key, dtype, sha).reshape(-1).astype(np.float64)
            delta = np.abs(values.astype(np.float64) - expected)
            report += f"; direct-{kind} maxabs={float(delta.max()):.3e} bad={int((delta > atol + rtol * np.abs(expected)).sum())}"
        reports.append(report)
        if not equal:
            failures.append(f"{name}: exact-operand arithmetic failed")
    return failures, reports
