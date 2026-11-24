# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the device-resident inputs of an x_metrics program (lane
vdf-xm-transport, 2026-11-24): the x_metrics resident transport of the shared
program runner (`bindings/arena_prog_native_door.mojo`
`metrics_run_plan_entries`).

On an image with a resident store (`MetricRoute.store`: the GPU images, the
range route on), a program input is read from the image's store in place:

  * a `DeviceRows` fold view of a float32 base: the base is put ONCE (into the
    active `resident()` scope's x_metrics cache at any size, identity-keyed,
    so every fold, candidate and metric of the block shares the upload; else
    into a slot this run owns) and the fold's rows are gathered on the device
    (`MP_DEV_TAKE_ROWS`, core/device_store.mojo `take_rows`); the rows never
    come to the host (before: `_values(value, context, True)`, a host gather
    of every fold, then the arena upload);
  * a host buffer of at least the scope's floor inside `resident()`: put once
    and found again by identity (`DeviceCache`'s contract, restored for
    x_metrics: the `x_metrics_dev_*` entries were deleted as dead on
    2026-10-26 by c209716bf).

The slot reaches the program through the range route's input triple
(core/arena_io.mojo `upload_ranges`, device to device). The store
operations are the image's program entry (x_metrics/descriptor_schema.mojo
`MP_DEV_*`); the scope's cache uses the same entry through
bindings/arena_cache_run.mojo (route bit `ARENA_ROUTE_METRICS`). This is the
estimators' convention (python/mojolearn/_arena_io.py), the x_prep runner's
form (`_rows_gather`, `XprepResident`), not a second one.

The host column, a CPU-only install and `MOJOLEARN_ARENA_RANGES=0` have no
store: their routes read host words exactly as before (a DeviceRows is its
host base's rows, the parent decision on addendum 3 item 4).

Where a word travels moves no bit: a slot holds exactly the host words the
host path would have copied into the arena (the same float32 rows in index
order), and `upload_ranges` copies them byte for byte.
"""
from std.memory import ArcPointer
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get, py_type
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_arrays_door import classical_address, classical_is_array, classical_c_order
from bindings.sequential_admission_door import sequential_admit_value
from bindings.x_prep_native_inputs_door import _rows_fields, _dense_values
from bindings.arena_scope_door import arena_scope_active, arena_cache_slot
from bindings.arena_cache_run import arena_scope_cache, arena_scope_floor, arena_metrics_store
from core.arena_state import ARENA_CACHES, ARENA_HANDLE_NONE, DeviceCacheState, arena_handle
from core.entry_table import EntryRef, entry_ref_unresolved
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_metrics.descriptor_schema import MP_MODE, MP_VENDOR, MP_DEV_PUT, MP_DEV_FREE, MP_DEV_TAKE_ROWS

comptime _INT32_MAX = 2147483647


struct MetricResident(Movable):
    """The calling thread's `resident()` scope resolved for ONE x_metrics
    program run: the scope's handle word (`ARENA_HANDLE_NONE` outside a
    scope), its size floor and the image's store entry. The scope's x_metrics
    cache is created and leased at the first input the run keeps resident.
    (`XprepResident` of bindings/x_prep_native_owner_door.mojo, on the
    x_metrics store.)"""
    var scope: Int
    var min_words: Int
    var store: EntryRef
    var cache: Int
    var lease: Optional[ArcPointer[DeviceCacheState]]

    def __init__(out self, scope: Int, min_words: Int, store: EntryRef):
        self.scope = scope
        self.min_words = min_words
        self.store = store
        self.cache = ARENA_HANDLE_NONE
        self.lease = Optional[ArcPointer[DeviceCacheState]]()

    def active(self) -> Bool:
        return self.scope >= 0

    def keeps(self, rows: Bool, words: Int) -> Bool:
        """Whether the run keeps an input of `words` words resident (a
        DeviceRows base at any size)."""
        return self.scope >= 0 and (rows or words >= self.min_words)

    def slot(mut self, source: PythonObject, address: Int, words: Int) raises -> Int:
        """The store id of `source`'s resident copy: found (a hit), or put
        once and published with `source` pinned by the cache."""
        if not self.lease:
            self.cache = arena_scope_cache(self.scope, self.store, "x_metrics")
            self.lease = Optional[ArcPointer[DeviceCacheState]](ARENA_CACHES.acquire(arena_handle(self.cache)))
        return arena_cache_slot(self.lease.value(), self.cache, self.store, source, address, words)


def metric_resident_none() -> MetricResident:
    """The value of a run that keeps nothing resident."""
    return MetricResident(ARENA_HANDLE_NONE, 0, entry_ref_unresolved())


def _metric_scope_word(context: PythonObject) raises -> Int:
    # The thread's active scope (bindings/arena_scope_door.mojo): the
    # thread-local's dictionary, one slot read, one downcast. A program
    # context connected without the scope keys (none today: every x_metrics
    # runner context comes from python/mojolearn/_native_arena_prog.py
    # `connect`, or a copy of it) keeps nothing resident.
    var local = py_get(context, "arena_local")
    if local is None:
        return ARENA_HANDLE_NONE
    return arena_scope_active(local, native_loaded_module("_thread")._local,
                              native_loaded_module("builtins").str, context["arena_scope_owner"],
                              context["arena_scope_class"])


def metric_resident_of(program: NativeDescriptorCall, context: PythonObject) raises -> MetricResident:
    """The run's resident scope on the store of the image whose program
    entry is `program` (a route with `store`). The image's tier and vendor
    words (two entry calls) and its store entry are resolved only inside a
    scope. Once per run, at its first nonempty input."""
    var scope = _metric_scope_word(context)
    if scope < 0:
        return metric_resident_none()
    var mode = native_descriptor_call(program, MP_MODE, [], [], [])
    var vendor = native_descriptor_call(program, MP_VENDOR, [], [], [])
    return MetricResident(scope, arena_scope_floor(scope), arena_metrics_store(mode, vendor))


def metric_store_put(program: NativeDescriptorCall, address: Int, words: Int) raises -> Int:
    """A slot this run owns holding the `words` host words at `address`."""
    var slot = -1
    with GILReleased(Python()):
        slot = native_descriptor_call(program, MP_DEV_PUT, [address], [words], [])
    return slot


def metric_rows_gather(value: PythonObject, program: NativeDescriptorCall, mut owned: List[Int],
                       mut resident: MetricResident, context: PythonObject) raises -> Int:
    """The store slot of a DeviceRows' rows gathered on the device out of its
    base (`_rows_gather` of the x_prep runner): the base put once through the
    active `resident()` scope (any size) or into a slot this run owns, then
    `MP_DEV_TAKE_ROWS` into a slot this run owns (appended to `owned`).
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
        base_slot = metric_store_put(program, base_address, base_words)
        owned.append(base_slot)
    var index_address = Int(py=classical_address(indices))
    var slot = -1
    with GILReleased(Python()):
        slot = native_descriptor_call(program, MP_DEV_TAKE_ROWS, [index_address], [base_slot, d, n], [])
    owned.append(slot)
    # read by address inside the calls above: named again so they outlive them
    _ = indices^
    _ = base^
    return slot


def metric_store_release(program: NativeDescriptorCall, owned: List[Int]) -> String:
    """Free every slot a run acquired, newest first, even when one fails.
    Returns the failures' text (empty: none); never raises."""
    var failures = String()
    for i in range(len(owned)):  # driver-loop: over the device slots one program run acquired
        var slot = owned[len(owned) - 1 - i]
        try:
            with GILReleased(Python()):
                _ = native_descriptor_call(program, MP_DEV_FREE, [], [slot], [])
        except error:
            failures += ("" if failures.byte_length() == 0 else "; ") + "slot " + String(slot) + ": " + String(error)
    return failures
