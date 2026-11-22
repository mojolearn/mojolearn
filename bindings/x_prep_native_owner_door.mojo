# SPDX-License-Identifier: Apache-2.0
# Native owner for the established _Prog transport contract. Arithmetic and
# stream synchronization remain in the same x_prep exported implementations.
from bindings.native_array_storage_door import native_array_empty, native_array_zeros
from bindings.native_loaded_module_door import native_loaded_module
from std.os import getenv
from bindings.x_prep_native_owner_run import xprep_r3, XprepInput, XprepRunInput, XprepRoute, XprepCaps, xprep_execute, xprep_release
from std.time import perf_counter_ns
from std.memory import ArcPointer
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setattr
from bindings.classical_arrays_door import classical_address, classical_is_array, classical_c_order, classical_wrap_array
from bindings.sequential_admission_door import sequential_admit_value
from bindings.x_prep_native_inputs_door import _dense_values, _values, _rows_fields
from bindings.py_cpython_table import cpython_table
from bindings.arena_scope_door import arena_scope_active, arena_cache_slot, arena_cache_drop_inout
from bindings.arena_cache_run import arena_scope_cache, arena_scope_cache_found, arena_scope_floor
from core.arena_state import ARENA_CACHES, ARENA_HANDLE_NONE, DeviceCacheState, arena_handle

from std.python._cpython import GILReleased
from bindings.prep_entry_cache_unpack import prep_entry_word, prep_entry_text
from core.entry_table import EntryRef, entry_ref_unresolved
from core.prep_entry_table import PrepEntryKey, xp_transport_ref, xp_transport_call
from x_prep.transport_abi import xp_vendor_word


def xprep_key(entries: PythonObject) raises -> EntryRef:
    """The x_prep transport entry of a connected entries dict (the image's
    compiled mode and vendor words, read once per binding by
    prep_entry_cache), resolved once per operation into an `EntryRef`."""
    var mode = prep_entry_word(entries, "x_prep_numeric_mode", -1)
    var vendor = xp_vendor_word(prep_entry_text(entries, "x_prep_vendor"))
    if mode < 0 or mode > 1 or vendor < 0:
        raise Error("ImportError: mojolearn: the x_prep binding publishes no native entry table")
    return xp_transport_ref(PrepEntryKey(mode, vendor))


# TOMBSTONE: the x_metrics Python-entry arms of `_store_put`/`_store_free` (unreachable; lane
# pyout-metrics-cnn) deleted 2026-11-11; code recoverable at 6f82d0255d.


def xprep_address(value: PythonObject) raises -> Int:
    # Only retained native array.array/mmap/memoryview storage reaches here.
    # The two buffer calls go through the process-wide CPython table
    # (bindings/py_cpython_table.mojo), not a dlsym per call (perf2-prep).
    ref cpy = Python().cpython()
    var view = List[Int](length=10,fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpython_table()[].PyObject_GetBuffer(value._obj_ptr,pointer,Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var address = view[0]
    cpython_table()[].PyBuffer_Release(pointer)
    _ = view^
    return address


# TOMBSTONE: MOJOLEARN_XPREP_FAST_ARENA_HUGEPAGE (NOISE, A/B round 3 NVIDIA L40S, wheel 28a780604c, job 2 of the round (ab3b), one run per arm, digests equal: 20 cells 0.952-1.092, geomean 1.012; normalizer istella fit 1.091, infer 1.092) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
# Restore: git apply experiments/removed/MOJOLEARN_XPREP_FAST_ARENA_HUGEPAGE.patch; record in docs/TOMBSTONES.md.


def xprep_storage(words: Int, context: PythonObject) raises -> PythonObject:
    var profile = String(getenv("MOJOLEARN_XPREP_PROFILE","0")) == "1"
    var start = Int(perf_counter_ns()) if profile else 0
    var n = max(words,1)
    var b = native_loaded_module("builtins")
    var storage = PythonObject(None)
    var mapped = n >= 262144 and xprep_r3("mapped")
    if mapped:
        storage = py_call(py_call(b.memoryview,py_call(context["arena_mmap"],-1,4*n)).cast,"f")
    else:
        storage = native_array_zeros(String("f"), n)
    var address = xprep_address(storage)
    return Python.tuple(storage,address,Int(perf_counter_ns())-start if profile else 0)


def xprep_take(storage: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var shape = Python.tuple(rows) if cols == 0 else Python.tuple(rows,cols)
    var count = rows*(1 if cols == 0 else cols)
    var part = py_call(b.memoryview,storage)[py_call(b.slice,offset,offset+count)]
    if py_type(storage) is b.memoryview and count >= 1048576 and xprep_r3("view"):
        # A large mapped output retains its slice and mapping after the
        # operation returns, exactly as _Prog._take's view path does.
        var result = py_call(b.object.__new__,context["array_class"])
        py_setattr(result,"_store",PythonObject(None))
        py_setattr(result,"_base",storage)
        py_setattr(result,"_pin",PythonObject(None))
        py_setattr(result,"_mv",part)
        py_setattr(result,"_addr",PythonObject(xprep_address(part)))
        py_setattr(result,"_readonly",PythonObject(False))
        py_setattr(result,"shape",shape)
        py_setattr(result,"dtype",PythonObject("<f4"))
        py_setattr(result,"ndim",PythonObject(1 if cols == 0 else 2))
        py_setattr(result,"size",PythonObject(count))
        py_setattr(result,"nbytes",PythonObject(count*4))
        py_setattr(result,"itemsize",PythonObject(4))
        py_setattr(result,"strides",Python.tuple(4) if cols == 0 else Python.tuple(cols*4,4))
        py_setattr(result,"order",PythonObject("C"))
        return result
    var store = storage
    if py_type(storage) is b.memoryview:
        store = native_array_empty(String("f"))
        _ = py_call(store.frombytes,py_call(part.cast,"B"))
    elif offset != 0 or len(storage) != count:
        store = storage[py_call(b.slice,offset,offset+count)]
    return classical_wrap_array(store,shape,"<f4","C",context)


def xprep_ranges(entries: PythonObject, scratch: Int) raises -> Bool:
    var device_scratch = scratch == 0 or py_get(entries,"x_prep_run_out") is not None or py_get(entries,"x_prep_run_scratch") is not None
    return device_scratch and String(getenv("MOJOLEARN_ARENA_RANGES","1").strip()) != "0" and py_get(entries,"x_prep_run_ranges") is not None


def xprep_values(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # (kind, native buffer/base, indices, rows, columns). Row handles remain
    # on device when the existing ranges/store ABI is available.
    if py_type(source) is context["maxabs_rows_class"]:
        var fold = _rows_fields(source,context)
        if fold.dtype == "":
            raise Error("TypeError: native row selection requires native dtype metadata")
        if fold.dtype == "<f4" and xprep_ranges(entries,1) and py_get(entries,"x_prep_dev_take_rows") is not None and py_get(entries,"x_prep_dev_put") is not None and py_get(entries,"x_prep_dev_free") is not None:
            var base = _dense_values(fold.base,context)
            var indices = fold.indices
            sequential_admit_value(indices,context)
            if not classical_is_array(indices,context) or String(indices.dtype) != "<i8" or Int(py=indices.ndim) != 1 or not classical_c_order(indices,context):
                raise Error("TypeError: native row selection requires a contiguous int64 index buffer")
            var n = Int(py=indices.size)
            var d = Int(py=base.shape[1])
            if n < 1 or n > 2147483647//d:
                raise Error("native row selection must be nonempty and within native indexing bounds")
            return Python.tuple(1,base,indices,n,d)
        # A missing GPU transport capability has the same host snapshot
        # semantics as CPU, without invoking DeviceRows._take.
        var values = _values(source,context,True)
        return Python.tuple(0,values,PythonObject(None),values.shape[0],values.shape[1])
    var values = _values(source,context)
    return Python.tuple(0,values,PythonObject(None),values.shape[0],values.shape[1])


struct XprepResident(Movable):
    """The calling thread's `resident()` scope resolved for ONE program run
    (lane arena, 2026-11-21): the scope's handle word (`ARENA_HANDLE_NONE`
    outside a scope, or when the image has no device store), its size floor
    and the image's transport entry. The scope's cache on that entry is
    created and leased at the first input the run keeps resident."""
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
            self.cache = arena_scope_cache(self.scope, self.store, "x_prep")
            self.lease = Optional[ArcPointer[DeviceCacheState]](ARENA_CACHES.acquire(arena_handle(self.cache)))
        return arena_cache_slot(self.lease.value(), self.cache, self.store, source, address, words)


def _scope_word(context: PythonObject) raises -> Int:
    # The thread's active scope: the thread-local's dictionary, one slot
    # read, one downcast (bindings/arena_scope_door.mojo).
    return arena_scope_active(context["arena_local"], native_loaded_module("_thread")._local,
                              native_loaded_module("builtins").str, context["arena_scope_owner"],
                              context["arena_scope_class"])


def xprep_resident_none() -> XprepResident:
    """The value of a run that keeps nothing resident."""
    return XprepResident(ARENA_HANDLE_NONE, 0, entry_ref_unresolved())


def xprep_resident(context: PythonObject, key: EntryRef, dev_put: Bool) raises -> XprepResident:
    """The run's resident scope for a caller that already holds the image's
    transport entry and its device-store capability. Once per run."""
    var scope = _scope_word(context)
    if scope < 0 or not dev_put:
        return xprep_resident_none()
    return XprepResident(scope, arena_scope_floor(scope), key)


def xprep_resident_of(entries: PythonObject, context: PythonObject) raises -> XprepResident:
    """The run's resident scope for a caller that holds only the entries
    dictionary: the transport entry is resolved only inside a scope."""
    var scope = _scope_word(context)
    if scope < 0 or py_get(entries, "x_prep_dev_put") is None or py_get(entries, "x_prep_dev_free") is None:
        return xprep_resident_none()
    return XprepResident(scope, arena_scope_floor(scope), xprep_key(entries))


def _drop_resident_impl(values: List[PythonObject], entries: PythonObject, context: PythonObject) raises -> String:
    var resident = xprep_resident_of(entries, context)
    if not resident.active():
        return String()
    var cache = arena_scope_cache_found(resident.scope, resident.store, "x_prep")
    if cache < 0:
        return String()
    var addresses = List[Int](capacity=len(values))
    var words = List[Int](capacity=len(values))
    for i in range(len(values)):  # driver-loop: over the run's in-out inputs
        addresses.append(Int(py=classical_address(values[i])))
        words.append(Int(py=values[i].size))
    return arena_cache_drop_inout(cache, resident.store, addresses, words)


def xprep_drop_resident(values: List[PythonObject], entries: PythonObject, context: PythonObject) -> String:
    """Drop the resident device copies of a run's in-out inputs (p3-array-arena,
    hazard D). A stage rewrote their host words, so a device copy an earlier
    run of the same `resident()` scope took for the same (address, size) is
    stale: it is unpublished and freed, and the next plain use uploads the
    new words. Returns failure text (empty on success); never raises, so the
    caller's own error or cleanup report stays in charge."""
    if len(values) == 0:
        return String()
    try:
        return _drop_resident_impl(values, entries, context)
    except error:
        return "resident in-out drop: "+String(error)


def xprep_caps(entries: PythonObject) raises -> XprepCaps:
    """The run capabilities of an entry dictionary, read by name and through
    the table (the per-run reads of a caller that holds no device route)."""
    return XprepCaps(True, xprep_key(entries), py_get(entries,"x_prep_run_out") is not None,
                     py_get(entries,"x_prep_run_scratch") is not None, py_get(entries,"x_prep_run_ranges") is not None,
                     py_get(entries,"x_prep_dev_put") is not None and py_get(entries,"x_prep_dev_free") is not None)


def xprep_run(storage: PythonObject, host: Int, scratch: Int, output: PythonObject, out_words: Int,
              var program: List[Int32], inputs: PythonObject, entries: PythonObject, context: PythonObject,
              logical_output: Int = 0, inout_ranges: List[Int] = List[Int]()) raises:
    """`xprep_run_records` of a Python input list: each item is
    `(offset, (kind, value, indices, rows, cols))` (the stateless owners)."""
    xprep_run_caps(storage, host, scratch, output, out_words, program^, inputs, entries, context,
                   XprepCaps(False, entry_ref_unresolved(), False, False, False, False), logical_output, inout_ranges)


def xprep_run_caps(storage: PythonObject, host: Int, scratch: Int, output: PythonObject, out_words: Int,
                   var program: List[Int32], inputs: PythonObject, entries: PythonObject, context: PythonObject,
                   caps: XprepCaps, logical_output: Int = 0, inout_ranges: List[Int] = List[Int]()) raises:
    """`xprep_run` with the image's capabilities already resolved by the
    caller's device route (`caps.held`), or read here once (not held)."""
    var records = List[XprepInput](capacity=len(inputs))
    var owners = List[PythonObject](capacity=len(inputs))
    for k in range(len(inputs)):  # driver-loop: over the program's input records
        var record = inputs[k][1]
        records.append(XprepInput(Int(py=inputs[k][0]), Int(py=record[3]) * Int(py=record[4]), 0, k))
        owners.append(record)
    xprep_run_records(storage, host, scratch, output, out_words, program^, records, owners, entries, context,
                      logical_output, inout_ranges, caps if caps.held else xprep_caps(entries))


def xprep_run_records(storage: PythonObject, host: Int, scratch: Int, output: PythonObject, out_words: Int,
                      var program: List[Int32], records: List[XprepInput], owners: List[PythonObject],
                      entries: PythonObject, context: PythonObject,
                      logical_output: Int, inout_ranges: List[Int]) raises:
    """The run of a caller that holds no device route: the image's
    capabilities are read from `entries` here, once per run."""
    xprep_run_records(storage, host, scratch, output, out_words, program^, records, owners, entries, context,
                      logical_output, inout_ranges, xprep_caps(entries))


def xprep_run_records(storage: PythonObject, host: Int, scratch: Int, output: PythonObject, out_words: Int,
                      var program: List[Int32], records: List[XprepInput], owners: List[PythonObject],
                      entries: PythonObject, context: PythonObject,
                      logical_output: Int, inout_ranges: List[Int], caps: XprepCaps) raises:
    """One x_prep program run (lane encprog-typed, 2026-11-12: UNPACK, typed
    RUN, then cleanup). UNPACK resolves each input once: its buffer address,
    its resident cache slot, its DeviceRows fields, and the selected image's
    route words. RUN is `xprep_execute` (bindings/x_prep_native_owner_run.mojo)
    with the GIL released: the same puts, gathers, copies, ranges and run call
    the loop here made, so no bit moves.
    `inout_ranges`: [lo, hi) pairs of inputs a stage rewrites and the caller
    reads after the run (`_Prog.put(..., inout=True)`): they are copied up
    from the host (never a resident or direct slot) and come back like
    outputs."""
    var profile = String(getenv("MOJOLEARN_XPREP_PROFILE","0")) == "1"
    var start = Int(perf_counter_ns()) if profile else 0
    var kernel_start = start
    var arena = Int(py=storage[1])
    var op = 0 if out_words == 0 else Int(py=output[1])
    # The image's capabilities and its transport entry: resolved once per
    # estimator by the caller's device route, or once per run by `xprep_caps`
    # (the same reads this function made by name). The run-time switch
    # MOJOLEARN_ARENA_RANGES is still read here, per run.
    var has_out = caps.has_out
    var device_scratch = scratch > 0 and (has_out or caps.has_scratch)
    var ranges = ((scratch == 0 or has_out or caps.has_scratch)
                  and String(getenv("MOJOLEARN_ARENA_RANGES","1").strip()) != "0" and caps.has_ranges)
    var key = caps.key
    # Each input can own at most a direct base and a gathered selection.
    # Reserve before acquisition so recording a successful put/take cannot
    # allocate and strand a slot on List growth failure.
    var owned = List[Int](capacity=2*len(records))
    var run_inputs = List[XprepRunInput](capacity=len(records))
    # Per run, not per input: the thread's resident scope (thread-local, one
    # slot read, the scope's floor) is resolved once, at the first input
    # that could be kept resident; the scope's cache on this image's store
    # is created and leased once, at the first input that is. Per input a
    # hit is one lock word and one typed lookup.
    var resident = xprep_resident_none()
    var resident_read = False
    var inout_values = List[PythonObject]()
    # TOMBSTONE: MOJOLEARN_XPREP_FAST_DIRECT_OFF (B arm loses: host-arena staging of large inputs; A/B abn9 NVIDIA L40S, 48 cells 0.96-17.38 arm/base, median 1.75, 34 slower) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_XPREP_FAST_DIRECT_OFF.patch; record in docs/TOMBSTONES.md.
    var dev_put_on = caps.dev_put
    try:
        # UNPACK
        for k in range(len(records)):  # driver-loop: over the program's input records
            var item = records[k]
            var offset = item.offset
            var words = item.words
            if item.owner < 0:
                # a program-owned parameter list: copied into the arena
                run_inputs.append(XprepRunInput(offset, words, item.address, -1, False, 0, False, 0, 0, 0))
                continue
            var record = owners[item.owner]
            var rows = Int(py=record[0]) == 1
            var value = record[1]
            var source = -1
            var put = False
            var put_words = 0
            var address = 0
            var indices = 0
            var n_rows = 0
            var cols = 0
            var is_inout = False
            for j in range(len(inout_ranges) // 2):  # driver-loop: over the program's in-out input ranges
                if inout_ranges[2 * j] == offset and inout_ranges[2 * j + 1] == offset + words:
                    is_inout = True
            if is_inout:
                inout_values.append(value)
            if ranges and not is_inout:
                if not resident_read:
                    resident = xprep_resident(context, key, dev_put_on)
                    resident_read = True
                # fb12-rows (2026-11-17): a host record's extent is its record
                # word count (`words`, what the program reads at `offset`),
                # not `value.size`: an 8-byte label view (int64 / float64,
                # `_label_buffer`) holds `size` items but 2 * size words, so
                # `size` uploaded half of it. For every 4-byte record the two
                # are equal (no change); a DeviceRows record keeps its base's
                # size (`words` there is the selection, not the base).
                var extent = Int(py=value.size) if rows else words
                if resident.keeps(rows, extent):
                    source = resident.slot(value, Int(py=classical_address(value)), extent)
                elif dev_put_on and (rows or words >= 1048576):
                    put = True
                    put_words = extent
                if rows:
                    indices = Int(py=classical_address(record[2]))
                    n_rows = Int(py=record[3])
                    cols = Int(py=record[4])
            if put or (source < 0 and not rows):
                address = Int(py=classical_address(value))
            run_inputs.append(XprepRunInput(offset, words, address, source, put, put_words, rows, indices, n_rows, cols))
        kernel_start = Int(perf_counter_ns()) if profile else 0
        # RUN
        var route = XprepRoute(ranges, device_scratch, has_out)
        with GILReleased(Python()):
            xprep_execute(key, arena, host, scratch, op, out_words, program, run_inputs, inout_ranges, route, owned)
    except error:
        var cleanup = String()
        with GILReleased(Python()):
            cleanup = xprep_release(key, owned)
        var dropped = xprep_drop_resident(inout_values,entries,context)
        if dropped.byte_length() > 0:
            cleanup += ("" if cleanup.byte_length() == 0 else "; ")+dropped
        if cleanup.byte_length() > 0:
            raise Error("RuntimeError: x_prep operation failed: "+String(error)+"; cleanup also failed: "+cleanup)
        raise error
    var cleanup = String()
    with GILReleased(Python()):
        cleanup = xprep_release(key, owned)
    var dropped = xprep_drop_resident(inout_values,entries,context)
    if dropped.byte_length() > 0:
        cleanup += ("" if cleanup.byte_length() == 0 else "; ")+dropped
    if cleanup.byte_length() > 0:
        raise Error("RuntimeError: x_prep slot cleanup failed: "+cleanup)
    if profile:
        var done = Int(perf_counter_ns())
        var names = String()
        for i in range(len(program)//16):  # driver-loop: consumed stage metadata for the existing profiling line
            var op = Int(program[i*16])
            var name = String(op)
            if op == 0:
                name = "sort_cols"
            elif op == 1:
                name = "col_stats"
            elif op == 2:
                name = "quantile"
            elif op == 3:
                name = "affine"
            elif op == 4:
                name = "scale_params"
            elif op == 73:
                name = "robust_uv"
            elif op == 124:
                name = "colb_part"
            elif op == 125:
                name = "colb_fold"
            elif op == 128:
                name = "maxabs_fold"
            names += ("" if i == 0 else "+")+name
        print("XPPROG ops="+names+" arena="+String(host)+" scratch="+String(scratch)+" out="+String(logical_output)+" dev_out="+String(Int(out_words > 0))+" ranges="+String(Int(ranges))+" alloc_s="+String(Float64(Int(py=storage[2]))/1.0e9)+" inputs_s="+String(Float64(kernel_start-start)/1.0e9)+" call_s="+String(Float64(done-kernel_start)/1.0e9))
    # Own the caller's command List and retain it through execution even
    # when profiling is disabled. A raw integer address is not an owner.
    _ = program^
