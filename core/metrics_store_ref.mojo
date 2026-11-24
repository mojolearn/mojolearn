# SPDX-License-Identifier: Apache-2.0
"""Where a metrics-image member finds the image's resident store (lane vdf-scope-cluster4, 2026-11-24).

The spectral member's route uses all five entry slots and UMAP's member runs on the image's operations table, so
neither carries the store (core/metrics_store.mojo `metrics_store_v1`) in its route. The GPU metrics image
PUBLISHES the store's transport entry once at module initialization in a process-wide entry table per tier
(`metrics_store_publish`, core/entry_table.mojo), and a member resolves it from its route's (mode, vendor)
(`metrics_store_find`): the same entry the scope's cache then records and frees into. A host image publishes
nothing, so its members find no store and keep their host path. Host-safe: no device import.

Serial by nature: one table publication per image, one locked read per member call.
"""
from core.entry_table import EntryRef, entry_ref_word, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall
from x_prep.transport_abi import XP_TRANSPORT_ABI, XP_OP_COUNT

#: The scope cache's family prefix of the metrics image's store (SpectralClustering and UMAP).
comptime METRICS_STORE_PREFIX = "metrics"

comptime _TABLE_IDENTICAL = "MojolearnEntriesMetricsStoreIdenticalV1"
comptime _TABLE_FAST = "MojolearnEntriesMetricsStoreFastV1"


def metrics_store_publish(mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish the GPU metrics image's store transport entry for tier `mode` on `vendor` (`vendor` < 1: a build
    without a GPU vendor publishes nothing)."""
    if vendor < 1:
        return
    var schemas: List[String] = [String(XP_TRANSPORT_ABI)]
    var operations: List[Int] = [XP_OP_COUNT]
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    if mode == 1:
        entry_table_publish[_TABLE_IDENTICAL](mode, vendor, publisher, schemas, operations, entries)
    else:
        entry_table_publish[_TABLE_FAST](mode, vendor, publisher, schemas, operations, entries)


def metrics_store_find(mode: Int, vendor: Int) -> Optional[EntryRef]:
    """The published store entry of tier `mode` on `vendor`, or none (a host route, or no GPU metrics image of
    that tier loaded): the member then keeps its host-gather path."""
    if vendor < 1 or mode < 0 or mode > 1:
        return Optional[EntryRef]()
    try:
        var found: EntryRef
        if mode == 1:
            found = entry_table_resolve[_TABLE_IDENTICAL](mode, vendor, 0, String(XP_TRANSPORT_ABI), XP_OP_COUNT)
        else:
            found = entry_table_resolve[_TABLE_FAST](mode, vendor, 0, String(XP_TRANSPORT_ABI), XP_OP_COUNT)
        return Optional[EntryRef](EntryRef(found.call, entry_ref_word(mode, vendor, XP_OP_COUNT)))
    except:
        return Optional[EntryRef]()
