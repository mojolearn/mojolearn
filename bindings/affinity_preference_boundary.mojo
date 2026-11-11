# SPDX-License-Identifier: Apache-2.0
"""Preference UNPACK: retained input storage and native sequence capture only."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.cpython_table import cpython_table
from bindings.classical_arrays import classical_numpy
from bindings.classical_pandas import native_pandas_exact
from bindings.native_loaded_module import native_loaded_module
from bindings.native_array_snapshot import native_array_snapshot
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.sequential_admission import _one_of
from bindings.sequential_inputs import SequenceBoundaryCapture, sequential_capture_row_items
from bindings.pandas_preference_items import pandas_preference_items
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from core.borrowed_buffer import BorrowedBufferLayout
from core.affinity_preference_plan import AffinityPreferencePlan
from core.shared_slot import SharedSlot


struct PreferenceIteratorTypes(Defaultable, Movable):
    var classes: List[PythonObject]

    def __init__(out self):
        self.classes = List[PythonObject]()


comptime PREFERENCE_ITERATORS = SharedSlot[PreferenceIteratorTypes, "MojolearnPreferenceNativeIteratorsV1"]


def _known_iterator(cls: PythonObject, b: PythonObject) raises -> Bool:
    # Capture only canonical C iterator identities, once while holding the GIL.
    # A rebound context type table cannot turn a Python map/callback into an
    # admitted iterator. All retained types are immutable CPython builtins.
    var state = PREFERENCE_ITERATORS.get_or_create_ptr()
    if len(state[].classes) == 0:
        var table = cpython_table()
        var captured = List[PythonObject]()
        var empty = Python.list()
        var samples: List[PythonObject] = [empty, Python.tuple(), py_call(b.range, 0)]
        for sample in samples:  # fixed metadata: the three connected builtin iterator types
            var raw = table[].PyObject_GetIter(sample._obj_ptr)
            if not raw:
                raise Python().cpython().unsafe_get_error()
            var iterator = PythonObject(from_owned=raw)
            captured.append(py_type(iterator))
        captured.append(py_type(py_call(b.reversed, empty)))
        state[].classes = captured^
    for candidate in state[].classes:  # fixed metadata: four native iterator identities
        if cls is candidate:
            return True
    return False


def _native_iterator(value: PythonObject, context: PythonObject) raises -> List[PythonObject]:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var allowed = (cls is b.list or cls is b.tuple or cls is b.range or cls is b.str
        or cls is b.bytes or cls is b.bytearray or cls is b.dict)
    var flags = cpython_table()[].PyType_GetFlags(cls._obj_ptr)
    if not allowed and _one_of(cls, context["mapping_types"]):
        # Exact admitted native dict subtypes (OrderedDict/defaultdict), with
        # their own C iteration order. Never call a default factory.
        allowed = (flags & UInt64(1 << 8)) != 0 and (flags & UInt64(1 << 29)) != 0
    if not allowed and (cls is b.memoryview or cls is context["array_type"] or cls is context["ndarray"]):
        admit_native_buffer_exporter(value)
        allowed = True
    if not allowed and _one_of(cls, context["scalar_types"]):
        # Native NumPy scalars are non-iterable; let the C iterator door retain
        # its original TypeError, including the extension's qualified tp_name.
        # Static type + canonical numpy.generic ancestry excludes callback
        # types inserted into a mutable context scalar table.
        var sys = native_loaded_module("sys")
        var np_module = py_get(sys.modules, "numpy")
        if np_module is not None and py_type(np_module) is py_type(sys) and (flags & UInt64(512)) == 0:
            var generic = py_get(py_fields(np_module), "generic")
            var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
            for i in range(len(ancestry)):  # metadata: fixed native scalar ancestry
                if ancestry[i] is generic:
                    allowed = True
    if not allowed:
        allowed = _known_iterator(cls, b)
    if not allowed:
        raise Error("TypeError: preference requires an admitted native iterable or numeric buffer; Python iterator callbacks are unsupported")
    ref cpy = Python().cpython()
    var table = cpython_table()
    var raw = table[].PyObject_GetIter(value._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var iterator = PythonObject(from_owned=raw)
    # list(iterable) queried this before consuming items; keep native range
    # overflow and native length-hint errors at their original stage.
    var hint = cpy.lib.get_function[Int]("PyObject_LengthHint")(value._obj_ptr, Int(8))
    if hint < 0:
        raise cpy.unsafe_get_error()
    var result = List[PythonObject](capacity=hint)
    while True:  # admission: ordered CPython native iterator reads under the GIL
        var item = table[].PyIter_Next(iterator._obj_ptr)
        if not item:
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            break
        result.append(PythonObject(from_owned=item))
    _ = iterator
    _ = value
    return result^


def _array_capture(value: PythonObject, context: PythonObject, numpy_connected: Bool) raises -> SequenceBoundaryCapture:
    var snapshot = native_array_snapshot(value, context, True)
    var rank = Int(py=snapshot.ndim)
    if rank == 0:
        raise Error("TypeError: iteration over a 0-d Array")
    var capture = SequenceBoundaryCapture()
    capture.plan.container(0, 1)
    capture.plan.container(1, Int(py=snapshot.shape[0]))
    if rank > 1 and not numpy_connected:
        raise Error("TypeError: inputs must contain numeric scalars")
    var shape = List[Int](capacity=rank)
    var strides = List[Int](capacity=rank)
    for axis in range(rank):  # metadata: admitted Array rank only
        shape.append(Int(py=snapshot.shape[axis]))
        strides.append(Int(py=snapshot.strides[axis]))
    var suboffsets = List[Int](length=rank, fill=-1)
    var pin = AdmittedBorrowedBuffer(snapshot._mv)
    var layout = BorrowedBufferLayout(Int(py=snapshot._addr), Int(py=snapshot.itemsize),
        Bool(py=snapshot._readonly), shape^, strides^, suboffsets^)
    var dtype = String(snapshot.dtype)
    var floating = dtype == "<f2" or dtype == "<f4" or dtype == "<f8"
    var unsigned = dtype == "<u1" or dtype == "<u2" or dtype == "<u4"
    capture.plan.array(1, layout^, "f" if floating else ("u" if unsigned else "i"), True)
    if rank == 1:
        # Array's native iterator boxes floating words as Python doubles and
        # every supported integer dtype (at most uint32/int64) as Python ints.
        # Represent that inferred dtype directly, without a temporary Python
        # list or scalar objects. The existing parallel writer performs the
        # same widening, including half; subsequent Float32 cast is unchanged.
        capture.plan.kind = "f" if floating else "i"
        capture.plan.itemsize = 8
    capture.pins.append(pin^)
    _ = snapshot
    _ = value
    return capture^


def affinity_preference_prepare(value: PythonObject, context: PythonObject) raises -> AffinityPreferencePlan:
    var numpy_connected = classical_numpy(context)
    var capture = SequenceBoundaryCapture()
    var items = List[PythonObject]()
    if py_type(value) is context["array_class"]:
        capture = _array_capture(value, context, numpy_connected)
    else:
        if native_pandas_exact(value):
            items = pandas_preference_items(value, context)
        else:
            items = _native_iterator(value, context)
        capture = sequential_capture_row_items(items, context)
    var result = AffinityPreferencePlan()
    var array_scalars = py_type(value) is context["array_class"] and len(capture.plan.shape) == 2
    if array_scalars:
        # Array.tolist previously captured scalar values while holding the
        # GIL. Preserve that snapshot stage; native independent copy units
        # still run in parallel. Later casting reads only Mojo-owned storage.
        result.capture(capture.plan)
    try:
        with GILReleased(Python()):
            if array_scalars:
                result.finish(capture.plan, numpy_connected)
            else:
                result.prepare(capture.plan, numpy_connected)
    finally:
        _ = capture
        _ = items
        _ = value
    return result^
