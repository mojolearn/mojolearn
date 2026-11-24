# SPDX-License-Identifier: Apache-2.0
"""THE FOREST IMAGES' STORE TRANSPORT (lane vdf-scope-trees, 2026-11-24).

A forest GPU image that keeps a device store (core/device_store.mojo; the
image answers XF_CAP_DEV_ROWS) publishes ONE transport entry over it with the
x_prep transport numbering and signature (x_prep/transport_abi.mojo:
XP_DEV_PUT a [src] i [n_words] -> slot id; XP_DEV_FREE i [id];
XP_DEV_TAKE_ROWS a [idx int64] i [src_id, row_words, n_idx] -> slot id;
XP_DEV_LIVE -> live slots). A resident scope's cache puts and frees through
it (bindings/arena_cache_run.mojo `arena_scope_store_put`, the per-family
store API of lane vdf-scope-run), and the image's XF_FIT_ROWS reads the slot
ids it handed out. One table per family and tier (a tier's images of one
family are one publication per process).

`forest_store_dispatch` holds the span admission and op words shared by
the images; the image passes its store and context. Serial by nature: each
op is one put, free or gather on one device context.
"""
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall
from core.forest_entry_table import FOREST_TABLE_RF, FOREST_TABLE_ET
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, XP_OP_COUNT

#: The schema of a forest store transport entry (the x_prep op numbering).
comptime FOREST_STORE_ABI = "mojolearn.trees.forest_store.v1"
#: The binding-family prefix a forest store's scope cache is kept under.
comptime FOREST_STORE_PREFIX_RF = "forest_rf"
comptime FOREST_STORE_PREFIX_ET = "forest_et"

comptime _RF_IDENTICAL = "MojolearnForestStoreTransportRfIdenticalV1"
comptime _RF_FAST = "MojolearnForestStoreTransportRfFastV1"
comptime _ET_IDENTICAL = "MojolearnForestStoreTransportEtIdenticalV1"
comptime _ET_FAST = "MojolearnForestStoreTransportEtFastV1"


def forest_store_publish(family: Int, mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish an image's store transport entry (module initialization,
    beside its forest entry). `family` FOREST_TABLE_RF or FOREST_TABLE_ET."""
    var schemas: List[String] = [String(FOREST_STORE_ABI)]
    var operations: List[Int] = [XP_OP_COUNT]
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    if family == FOREST_TABLE_RF:
        if mode == 1:
            entry_table_publish[_RF_IDENTICAL](mode, vendor, publisher, schemas, operations, entries)
        else:
            entry_table_publish[_RF_FAST](mode, vendor, publisher, schemas, operations, entries)
    elif family == FOREST_TABLE_ET:
        if mode == 1:
            entry_table_publish[_ET_IDENTICAL](mode, vendor, publisher, schemas, operations, entries)
        else:
            entry_table_publish[_ET_FAST](mode, vendor, publisher, schemas, operations, entries)
    else:
        raise Error("forest store transport: unknown family " + String(family))


def forest_store_transport_ref(family: Int, mode: Int, vendor: Int) raises -> EntryRef:
    """The store transport entry of the image (`family`, `mode`, `vendor`),
    resolved once per member state (its fit's UNPACK of the route). Raises
    the table's words when the image publishes none."""
    if family == FOREST_TABLE_RF:
        if mode == 1:
            return entry_table_resolve[_RF_IDENTICAL](mode, vendor, 0, String(FOREST_STORE_ABI), XP_OP_COUNT)
        return entry_table_resolve[_RF_FAST](mode, vendor, 0, String(FOREST_STORE_ABI), XP_OP_COUNT)
    if family == FOREST_TABLE_ET:
        if mode == 1:
            return entry_table_resolve[_ET_IDENTICAL](mode, vendor, 0, String(FOREST_STORE_ABI), XP_OP_COUNT)
        return entry_table_resolve[_ET_FAST](mode, vendor, 0, String(FOREST_STORE_ABI), XP_OP_COUNT)
    raise Error("forest store transport: unknown family " + String(family))


def forest_store_words(aa: Int, na: Int) -> List[Int]:
    var out = List[Int](capacity=na)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    for k in range(na):  # small-loop(request words): at most one address or three integers
        out.append(p[k])
    return out^


def forest_store_admit(op: Int, na: Int, np: Int, nf: Int, result: Int) raises:
    """The x_prep transport's span rule for the four store ops."""
    var want_a = 0
    var want_i = 0
    if op == XP_DEV_PUT:
        want_a = 1
        want_i = 1
    elif op == XP_DEV_FREE:
        want_i = 1
    elif op == XP_DEV_TAKE_ROWS:
        want_a = 1
        want_i = 3
    elif op != XP_DEV_LIVE:
        raise Error("forest store transport: unknown operation " + String(op))
    if na != want_a or np != want_i or nf != 0:
        raise Error("forest store transport: operation " + String(op) + " takes " + String(want_a)
                    + " addresses, " + String(want_i) + " integers and no floats")
    if result == 0:
        raise Error("forest store transport: null result address")
