# SPDX-License-Identifier: Apache-2.0
"""The metrics GPU image's resident store and its scope transport entry (lane vdf-scope-cluster4, 2026-11-24).

ONE store for the two members the `_mojolearn_metrics` image runs (SpectralClustering, core/spectral_member.mojo;
UMAP, core/manifold_member.mojo): a `DeviceStore` (core/device_store.mojo) on the image's one process context
(`MojolearnMetricsContext*`, the context of spectral/estimator.mojo and umap/ops_entry_device.mojo), so a slot is a
buffer those kernels read directly. `metrics_store_v1` is the image's transport entry with the x_prep numbering
(core/family_store_transport.mojo): the `resident()` scope keeps a model-selection X here ONCE
(`arena_scope_store_put`, prefix `METRICS_STORE_PREFIX`) and each fold's rows are taken on the device
(`arena_store_take_rows`). The binding exports it as the capsule `metrics_store_v1` and publishes it in the
image's store table (core/metrics_store_ref.mojo `metrics_store_publish`), where a member resolves it by its
route's (mode, vendor): no route slot carries it (the spectral member uses all five). GPU image only.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.family_store_transport import family_store_entry
from x_prep.transport_abi import XP_TRANSPORT_ABI

# Exactly the metrics image's owning context (spectral/estimator.mojo, umap/ops_entry_device.mojo).
comptime _DEVCTX_SLOT = ("MojolearnMetricsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                         else "MojolearnMetricsContextFastV3")

comptime METRICS_STORE = SharedSlot[DeviceStore, ("MojolearnMetricsStoreIdenticalV1"
                                                  if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                                  else "MojolearnMetricsStoreFastV1")]


def metrics_store_ctx() raises -> DeviceContext:
    """The context every slot of the store is made on: the metrics image's kernels' own."""
    return process_ctx[_DEVCTX_SLOT]()


def metrics_store_buffer(slot: Int, words: Int) raises -> DeviceBuffer[DType.float32]:
    """A reference to slot `slot`'s buffer (at least `words` words), copied out under the store's lock: it keeps
    the device memory alive past a concurrent free while the caller's kernels read it."""
    return METRICS_STORE.get_or_create_ptr()[]._held(slot, words)


def metrics_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                     context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The store's transport entry (ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE)."""
    try:
        var ctx = metrics_store_ctx()
        return family_store_entry(METRICS_STORE.get_or_create_ptr()[], ctx, "metrics", operation, aa, na,
                                  pp, np, nf, result, context, sink)
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MetricsStoreAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(XP_TRANSPORT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(metrics_store_v1)


comptime METRICS_STORE_API = SharedSlot[MetricsStoreAPI, ("MojolearnMetricsStoreAPIIdenticalV1"
                                                          if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                                                          else "MojolearnMetricsStoreAPIFastV1")]
