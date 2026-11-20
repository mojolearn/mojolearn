# SPDX-License-Identifier: Apache-2.0
"""The host metrics image's R^2 bodies over plain addresses and their native
table entry (lane calltable-xseq, 2026-11-11). `_mojolearn_metrics_host`
publishes `metrics_r2_host_v1` (core/metrics_entry_table.mojo) at module init
and its Python `r2_score` / `r2_score_weighted` entries run the same bodies.
"""
from core.hostptr import read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.metrics_entry_table import MetricsEntryKey, MetricsR2Owner, MR_R2, metrics_r2_check_spans, metrics_r2_publish
from metrics.host.metrics_oracle import host_r2_score, host_weighted_r2


def _check_float_pair(a: List[Float32], b: List[Float32], n: Int) raises:
    if n <= 0:
        raise Error("metrics: n must be positive, got " + String(n))
    if len(a) < n or len(b) < n:
        raise Error(
            "metrics: the two float arrays hold " + String(len(a)) + " and "
            + String(len(b)) + " entries, both must hold at least n = "
            + String(n)
        )


def _r2_value(y_addr: Int, yh_addr: Int, n: Int) raises -> Float32:
    """`r2_score`'s host body over plain addresses: the Python entry of `bindings/_mojolearn_metrics_host.mojo`
    and the native table entry `metrics_r2_host_v1`."""
    var y = read_f32(y_addr, n)
    var yh = read_f32(yh_addr, n)
    _check_float_pair(y, yh, n)
    return host_r2_score(y, yh, n)


def _r2_weighted_value(y_addr: Int, yh_addr: Int, w_addr: Int, n: Int) raises -> Float32:
    """`r2_score_weighted`'s host body over plain addresses (both entries)."""
    var y = read_f32(y_addr, n)
    var yh = read_f32(yh_addr, n)
    var w = read_f32(w_addr, n)
    return host_weighted_r2(y, yh, w, n)


def metrics_r2_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The R^2 entry of the host image's metrics table (`METRICS_R2_ABI`)."""
    try:
        var n = metrics_r2_check_spans(operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var value: Float32
        if operation == MR_R2:
            value = _r2_value(a[0], a[1], n)
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[2])[] = Float64(value)
        else:
            value = _r2_weighted_value(a[0], a[1], a[2], n)
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[3])[] = Float64(value)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


comptime METRICS_R2_HOST_OWNER = SharedSlot[MetricsR2Owner, "MojolearnMetricsR2HostOwnerV1"]


def publish_metrics_r2_host() raises:
    """Publish the host R^2 entry at module init (vendor word 0)."""
    var owner = METRICS_R2_HOST_OWNER.get_or_create_ptr()
    metrics_r2_publish(MetricsEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(owner), metrics_r2_host_v1)
