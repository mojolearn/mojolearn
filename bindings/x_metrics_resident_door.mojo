# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the device-resident inputs of an x_metrics program (lane
vdf-xm-transport, 2026-11-24; ONE store, lane vdf-xm-wire, 2026-11-24): the
resident transport of the shared program runner
(`bindings/arena_prog_native_door.mojo` `metrics_run_plan_entries`).

THE ONE STORE. Resident metric inputs live in the x_prep image's store
(x_prep/store_slot.mojo `X_PREP_STORE` on `x_prep_ctx`), the store the
`resident()` scope, `DeviceRows`, the estimators, the x_prep programs and the
`_mojolearn_metrics` kernel entries (metrics/slot_inputs.mojo) use. The
x_metrics GPU image runs on that store's context (x_metrics/device.mojo
`metrics_ctx`), so a program metric and a kernel metric read the SAME slot in
place, and one `resident()` block uploads an X once whichever metric family
reads it. Put, gather and free go through the x_prep transport entry of the
program image's tier and vendor (`metric_store_ref`); on an image with the
store, an x_prep image that is not connected is refused in words (lane
vdf-xm-wire2), never read as "no store".

On an image with the store (`MetricRoute.store` and `metric_store_ref`), a
program input is read from it in place:

  * a `DeviceRows` fold view of a float32 base: the base is put ONCE (into the
    active `resident()` scope's x_prep cache at any size, identity-keyed, so
    every fold, candidate and metric of the block shares the upload; else
    into a slot this run owns) and the fold's rows are gathered on the device
    (`XP_DEV_TAKE_ROWS`, core/device_store.mojo `take_rows`); the rows never
    come to the host;
  * a host buffer of at least the scope's floor inside `resident()`: put once
    and found again by identity (`XprepResident`, the x_prep runner's form).

The slot reaches the program through the range route's input triple
(core/arena_io.mojo `upload_ranges`, device to device on the store's context).

The host column, a CPU-only install and `MOJOLEARN_ARENA_RANGES=0` have no
store (`MetricRoute.store` false, the store is never asked for): their routes read host words exactly as before (a DeviceRows is its
host base's rows, the parent decision on addendum 3 item 4).

Where a word travels moves no bit: a slot holds exactly the host words the
host path would have copied into the arena (the same float32 rows in index
order), and `upload_ranges` copies them byte for byte.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_arrays_door import classical_address, classical_is_array, classical_c_order
from bindings.sequential_admission_door import sequential_admit_value
from bindings.x_prep_native_inputs_door import _rows_fields, _dense_values
from bindings.x_prep_native_owner_door import XprepResident, xprep_resident_none
from bindings.arena_scope_door import arena_scope_active
from bindings.arena_cache_run import arena_scope_floor
from core.entry_table import EntryRef
from bindings.metric_store_refusal import metric_store_refusal
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.prep_entry_table import PrepEntryKey, xp_transport_ref, xp_transport_call
from x_metrics.descriptor_schema import MP_MODE, MP_VENDOR
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS

comptime _INT32_MAX = 2147483647


def metric_store_ref(program: NativeDescriptorCall) raises -> EntryRef:
    """The x_prep transport entry of the program image's tier and vendor (two
    entry calls, then the process table read): the one resident store's
    door. Asked only on an image that has the store (`MetricRoute.store`),
    once per run, at its first input that could be resident. A table refusal
    is raised in words (`metric_store_refusal`): the selector refuses, it
    never widens to the host route (lane vdf-xm-wire2; before, it returned
    None and the run silently copied the input from the host). The x_prep
    image of each tier is connected with the x_metrics image at import
    (python/mojolearn/_native_arena_prog.py `connect`), so a refusal here
    means mismatched or missing binaries."""
    var mode = native_descriptor_call(program, MP_MODE, [], [], [])
    var vendor = native_descriptor_call(program, MP_VENDOR, [], [], [])
    try:
        return xp_transport_ref(PrepEntryKey(mode, vendor))
    except error:
        raise Error(metric_store_refusal("x_metrics", mode, vendor, String(error)))


def metric_scope_word(context: PythonObject) raises -> Int:
    """The calling thread's active `resident()` scope word, -1 outside one."""
    # The thread's active scope (bindings/arena_scope_door.mojo): the
    # thread-local's dictionary, one slot read, one downcast. A program
    # context connected without the scope keys (none today: every x_metrics
    # runner context comes from python/mojolearn/_native_arena_prog.py
    # `connect`, or a copy of it) keeps nothing resident.
    var local = py_get(context, "arena_local")
    if local is None:
        return -1
    return arena_scope_active(local, native_loaded_module("_thread")._local,
                              native_loaded_module("builtins").str, context["arena_scope_owner"],
                              context["arena_scope_class"])


def metric_resident_of(scope: Int, store: EntryRef) raises -> XprepResident:
    """The run's `resident()` scope (`metric_scope_word`) on the one store
    (`store`: its x_prep transport entry), or none outside a scope."""
    if scope < 0:
        return xprep_resident_none()
    return XprepResident(scope, arena_scope_floor(scope), store)


def metric_store_put(store: EntryRef, address: Int, words: Int) raises -> Int:
    """A slot this run owns holding the `words` host words at `address`."""
    var src: List[Int] = [address]
    var count: List[Int] = [words]
    var slot = -1
    with GILReleased(Python()):
        slot = xp_transport_call(store, XP_DEV_PUT, src, count)
    return slot


def metric_rows_gather(value: PythonObject, store: EntryRef, mut owned: List[Int],
                       mut resident: XprepResident, context: PythonObject) raises -> Int:
    """The store slot of a DeviceRows' rows gathered on the device out of its
    base (`_rows_gather` of the x_prep runner): the base put once through the
    active `resident()` scope (any size) or into a slot this run owns, then
    `XP_DEV_TAKE_ROWS` into a slot this run owns (appended to `owned`).
    -1 when the fold cannot be gathered as the program's float32 words (a
    base that is not float32, an empty fold): the caller reads its host
    rows, as before."""
    var fold = _rows_fields(value, context)
    if fold.dtype != "<f4":
        return -1
    var n = fold.rows
    var d = fold.row_words
    if n < 1 or d < 1 or n > _INT32_MAX // d:
        return -1
    var indices = fold.indices
    sequential_admit_value(indices, context)
    if (not classical_is_array(indices, context) or String(indices.dtype) != "<i8" or Int(py=indices.ndim) != 1
            or not classical_c_order(indices, context)):
        raise Error("TypeError: native row selection requires a contiguous int64 index buffer")
    var base = _dense_values(fold.base, context)
    var base_words = Int(py=base.size)
    var base_address = Int(py=classical_address(base))
    var base_slot = -1
    if resident.active():
        # the run's resident scope keeps a DeviceRows base at any size
        base_slot = resident.slot(base, base_address, base_words)
    else:
        base_slot = metric_store_put(store, base_address, base_words)
        owned.append(base_slot)
    var idx: List[Int] = [Int(py=classical_address(indices))]
    var shape: List[Int] = [base_slot, d, n]
    var slot = -1
    with GILReleased(Python()):
        slot = xp_transport_call(store, XP_DEV_TAKE_ROWS, idx, shape)
    owned.append(slot)
    # read by address inside the calls above: named again so they outlive them
    _ = indices^
    _ = base^
    return slot


def metric_store_release(store: EntryRef, owned: List[Int]) -> String:
    """Free every slot a run acquired, newest first, even when one fails.
    Returns the failures' text (empty: none); never raises."""
    var failures = String()
    for i in range(len(owned)):  # driver-loop: over the device slots one program run acquired
        var slot = owned[len(owned) - 1 - i]
        try:
            var ids: List[Int] = [slot]
            with GILReleased(Python()):
                _ = xp_transport_call(store, XP_DEV_FREE, List[Int](), ids)
        except error:
            failures += ("" if failures.byte_length() == 0 else "; ") + "slot " + String(slot) + ": " + String(error)
    return failures
