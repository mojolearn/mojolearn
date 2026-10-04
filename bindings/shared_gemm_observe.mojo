# SPDX-License-Identifier: Apache-2.0
"""Diagnostic exports linked into the actual estimator extension, not a probe."""
from std.python import PythonObject
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from experiments.apple_fast.gemm.shared_dispatch import COUNTERS, AUDIT, G1, G5, audit_count, reset_static_counts


def shared_gemm_count_binding(route: PythonObject, column: PythonObject) raises -> PythonObject:
    return PythonObject(audit_count(Int(py=route), Int(py=column)))


def shared_gemm_reset_binding() raises -> PythonObject:
    reset_static_counts()
    return PythonObject(1)


def shared_gemm_variant_binding() raises -> PythonObject:
    comptime if AUDIT:
        raise Error("runtime audit selector is not a static downstream arm")
    return PythonObject(5 if G5 else (1 if G1 else 0))


def shared_gemm_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def shared_gemm_vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))
