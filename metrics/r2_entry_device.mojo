# SPDX-License-Identifier: Apache-2.0
"""The device metrics image's R^2 bodies over plain addresses and their
native table entry (lane calltable-xseq, 2026-11-11). `_mojolearn_metrics`
publishes `metrics_r2_device_v1` (core/metrics_entry_table.mojo) at module
init and its Python `r2_score` / `r2_score_weighted` entries run the same
bodies (`r2_score_host`, `r2_score_weighted_host`: host reductions, as
before); `read_f32(addr, max(0, n))` is the binding's `_load_f32`.
"""
from core.hostptr import read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.metrics_entry_table import (
    MetricsEntryKey, MetricsR2Owner, MR_R2, metrics_r2_check_spans, metrics_r2_publish, metrics_vendor_word,
)
from metrics.estimator import r2_score_host, r2_score_weighted_host


def _r2_value(y_addr: Int, yh_addr: Int, n: Int) raises -> Float32:
    """`r2_score`'s body over plain addresses: the Python entry of `bindings/_mojolearn_metrics.mojo` and the
    native table entry `metrics_r2_device_v1` (core/metrics_entry_table.mojo)."""
    var y = read_f32(y_addr, max(0, n))
    var yh = read_f32(yh_addr, max(0, n))
    return r2_score_host(y, yh, n)


def _r2_weighted_value(y_addr: Int, yh_addr: Int, w_addr: Int, n: Int) raises -> Float32:
    """`r2_score_weighted`'s body over plain addresses (both entries)."""
    var y = read_f32(y_addr, max(0, n))
    var yh = read_f32(yh_addr, max(0, n))
    var w = read_f32(w_addr, max(0, n))
    return r2_score_weighted_host(y, yh, w, n)


def metrics_r2_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The R^2 entry of this image's metrics table (`METRICS_R2_ABI`), for
    native callers in other modules; the caller releases the GIL."""
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


comptime METRICS_R2_DEVICE_OWNER = SharedSlot[MetricsR2Owner, "MojolearnMetricsR2DeviceOwnerV1"]


def publish_metrics_r2_device() raises:
    """Publish the R^2 entry at module init; a build with no GPU vendor
    publishes nothing (its callers refuse: "provider is not published")."""
    var word = metrics_vendor_word(COMPILED_VENDOR)
    if word < 0:
        return
    var owner = METRICS_R2_DEVICE_OWNER.get_or_create_ptr()
    metrics_r2_publish(MetricsEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(owner), metrics_r2_device_v1)
