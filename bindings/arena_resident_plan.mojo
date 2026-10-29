# SPDX-License-Identifier: Apache-2.0
"""Private operation-owned resident rows, independent of writable _Resident.

Only native model-selection loops own this plan. Neither entries nor prepared
Arrays are exposed to an estimator or Python wrapper. Each miss admits and
captures one immutable schema and independent storage snapshot; the device
store already has first-upload snapshot semantics. Public _Resident/DeviceRows
remain on their separately checked, mutable lifecycle path.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.arena_cache_release import FirstFailure
from bindings.arena_materialize_native import _native_call, _prepare, _prepared_info, _cast, _slot, _empty
from bindings.msel_inputs import msel_input
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer


struct ResidentRecord(Copyable, Movable):
    # Record copies retain metadata/buffer owners only. Slot release belongs
    # exclusively to the plan's live map, never to a record destructor.
    var original: PythonObject
    var prepared: PythonObject
    var shape: PythonObject
    var dtype: String
    var size: Int
    var nbytes: Int
    var sid: Int

    def __init__(out self, original: PythonObject, prepared: PythonObject,
                 metadata: PythonObject) raises:
        self.original = original
        self.prepared = prepared
        self.shape = metadata["shape"]
        self.dtype = String(metadata["dtype"])
        self.size = Int(py=metadata["size"])
        self.nbytes = Int(py=metadata["nbytes"])
        self.sid = -1


struct ArenaResidentPlan(Movable):
    var context: PythonObject
    var put: PythonObject
    var free: PythonObject
    var gather: PythonObject
    var records: Dict[Int, ResidentRecord]

    def __init__(out self, context: PythonObject) raises:
        # Import-connected native entries are pinned for this operation, just
        # as a public _Resident pins _put/_free/_take at initialization. No
        # _Resident Python instance or mutable class namespace is involved.
        self.context = context
        self.put = context["arena_msel_put"]
        self.free = context["arena_msel_free"]
        self.gather = context["arena_msel_take_rows"]
        self.records = Dict[Int, ResidentRecord]()

    def entry(mut self, values: PythonObject, indices: Bool = False) raises -> Int:
        var key = Int(values._obj_ptr)
        if key in self.records:
            # Strong original ownership prevents address reuse; only this
            # native owner can publish/erase records. No Python equality,
            # dictionary-key scan, public class scan or metadata rebuild.
            return key
        var context = self.context
        var source = values
        if py_type(values) is not context["array_class"] and py_type(values) is not py_get(context, "ndarray"):
            var inputs = py_get(context, "arena_msel_input_context")
            if inputs is None:
                raise Error("RuntimeError: native resident rows require the model-selection input connection")
            source = msel_input(values, "fold data", inputs)
        # Always request a private snapshot, including when the supplied Array
        # is already contiguous. Caller slot replacement or memoryview.release
        # must not change our uploaded schema or invalidate its backing owner.
        var array = _prepare(source, context, True)
        if indices and String(_slot(array, "dtype", context)) != "<i8":
            array = _cast(array, "<i8", context)
        var metadata = _prepared_info(array, context)
        var record = ResidentRecord(values, array, metadata)
        if record.nbytes:
            # _prepared_info admits this native exporter. Pin the actual bytes
            # throughout the GIL-releasing upload, including non-C conversions.
            var pin = AdmittedBorrowedBuffer(_slot(array, "_mv", context))
            try:
                record.sid = Int(py=_native_call(self.put, Python.tuple(pin.layout.value().address, record.nbytes)))
            finally:
                pin.release()
        var sid = record.sid
        try:
            self.records[key] = record^
        except problem:
            # As for the public owner, failed publication frees the new slot
            # without replacing the primary pending-error receipt.
            if sid >= 0:
                try:
                    _ = py_call(self.free, sid)
                except cleanup_error:
                    pass
            raise problem^
        return key

    def store(mut self, values: PythonObject, indices: Bool = False) raises -> Int:
        var key = self.entry(values, indices)
        return self.records[key].sid

    def take(mut self, values: PythonObject, indices_value: PythonObject) raises -> PythonObject:
        var base_key = self.entry(values)
        var index_key = self.entry(indices_value, True)
        # Resolve borrows only after both insertions; a Dict growth cannot
        # invalidate either reference while it is used below.
        ref base = self.records[base_key]
        ref indices = self.records[index_key]
        var shape = Python.list(indices.size)
        for i in range(1, len(base.shape)):  # driver-loop: preserve every trailing dimension
            _ = py_call(shape.append, base.shape[i])
        var output = _empty(py_call(self.context["arena_native_types"]["tuple"], shape), base.dtype, self.context)
        var n = Int(py=base.shape[0])
        var row_bytes = base.nbytes // n if n else 0
        if indices.size and row_bytes:
            _ = _native_call(self.gather, Python.tuple(base.sid, n, row_bytes, indices.sid, indices.size, _slot(output, "_addr", self.context)))
        return output

    def release(mut self, values: PythonObject) raises:
        var key = Int(values._obj_ptr)
        if key not in self.records:
            return
        # Unpublish before the backend can release the GIL. The record's
        # owners outlive the free and any retry publication.
        var record = self.records.pop(key)
        var first = FirstFailure()
        try:
            if record.sid >= 0:
                _ = _native_call(self.free, Python.tuple(record.sid))
        except problem:
            first.keep(problem)
            try:
                self.records[key] = record^
            except publish_error:
                first.keep(publish_error)
            first.finish()
            return
        _ = record^
        first.finish()

    def close(mut self) raises:
        var pending = self.records^
        self.records = Dict[Int, ResidentRecord]()
        var failed = List[Int]()
        var first = FirstFailure()
        for item in pending.items():  # driver-loop: attempt every independent native store release
            try:
                if item.value.sid >= 0:
                    _ = _native_call(self.free, Python.tuple(item.value.sid))
            except problem:
                first.keep(problem)
                failed.append(item.key)
        for i in range(len(failed)):  # driver-loop: failed records stay owned for a retry
            try:
                self.records[failed[i]] = pending.pop(failed[i])^
            except publish_error:
                first.keep(publish_error)
        # Keep successful owners too until all frees and publications finish.
        _ = pending^
        first.finish()
