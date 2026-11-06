# SPDX-License-Identifier: Apache-2.0
"""Process records for the Array / arena owners (lane p3-array-arena).

Two records the Array, arena cache, resident and DeviceRows owners used to
rebuild on every call:

1. CLASS LAYOUT ADMISSIONS. Each owner checked its connected class on every
   call: an MRO read, a `list()` copy of the class dict keys with a type check
   per key, hook absence and slot-descriptor identities. Those results depend
   only on the class rows (plus one connected context object, `extra`, such as
   the retained slot table). `p3_layout_admitted(kind, cls, extra)` is True
   when the same check passed for the same class and `extra` and the class (or
   an ancestor) has not been modified since: validity comes from the CPython
   type watcher in `bindings/native_class_admission.mojo` (Python >= 3.12;
   `PyType_Modified` runs on every class attribute assignment or deletion and
   on `__bases__` assignment). Where type watchers are unavailable
   (Python 3.10 / 3.11, or a refused watch), nothing is admitted and every
   call runs the full check, exactly as before. Each admitted class and
   `extra` object is retained (one reference, never released) so a recycled
   address can never match a record.

2. NUMPY CLASS IDENTITIES. The NumPy scalar types and dtype classes the
   materializer compares against were rebuilt per call by constructing 14 to
   17 dtype objects through Python. They are process constants of one
   connected `numpy.dtype` factory: resolved once per factory (retained) and
   compared by address.

Everything here runs with the GIL held; the records are native and never
call back into Python on a hit.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_class_admission import _class_arm, _class_epoch, class_watch_ready


# Admission kinds: one per distinct check, so one class can carry several.
comptime P3_ARRAY_LAYOUT = 1          # arena_materialize_native._array_layout
comptime P3_RESIDENT_LAYOUT = 2       # arena_materialize_native._resident_fields (class half)
comptime P3_CACHE_LAYOUT = 3          # x_prep_native_owner._cache_layout
comptime P3_ROWS_LAYOUT = 4           # DeviceRows class (x_prep_native_inputs / arena_rows_native)
comptime P3_RESIDENT_NATIVE = 5       # arena_resident_native class admission
comptime P3_CACHE_KEY_LAYOUT = 6      # arena_cache_get.arena_cache_key model class
comptime P3_FROM_BUFFER_LAYOUT = 7    # array_from_buffer_native class slots
comptime P3_ARRAY_DOOR_LAYOUT = 8     # array_native._admit
comptime P3_STATS_LAYOUT = 9          # arena_cache_stats class walks


struct P3LayoutRecords(Defaultable, Movable):
    # Quads: kind, class address, extra address, class epoch at admission.
    var words: List[Int]

    def __init__(out self):
        self.words = List[Int]()


comptime P3_LAYOUT_RECORDS = _Global[
    StorageType=P3LayoutRecords, name="MojolearnP3ArrayArenaLayoutsV1",
    init_fn=P3LayoutRecords.__init__,
]


def p3_layout_admitted(kind: Int, cls: PythonObject, extra: PythonObject) raises -> Bool:
    """True when the `kind` check already passed for this exact class and
    `extra` and the class has not been modified since."""
    var g = P3_LAYOUT_RECORDS.get_or_create_ptr()
    var n = len(g[].words) // 4
    if n == 0:
        return False
    var address = Int(cls._obj_ptr)
    var other = Int(extra._obj_ptr)
    for i in range(n):  # driver-loop: over the few admitted (kind, class) records
        if g[].words[4 * i] == kind and g[].words[4 * i + 1] == address and g[].words[4 * i + 2] == other:
            return g[].words[4 * i + 3] == _class_epoch(cls)
    return False


def p3_layout_admit(kind: Int, cls: PythonObject, extra: PythonObject) raises:
    """Record that the full `kind` check just passed for `cls` and `extra`.
    No-op where type watchers are unavailable."""
    if not class_watch_ready():
        return
    if not _class_arm(cls):
        return
    var epoch = _class_epoch(cls)
    var address = Int(cls._obj_ptr)
    var other = Int(extra._obj_ptr)
    var g = P3_LAYOUT_RECORDS.get_or_create_ptr()
    var n = len(g[].words) // 4
    for i in range(n):  # driver-loop: over the few admitted (kind, class) records
        if g[].words[4 * i] == kind and g[].words[4 * i + 1] == address and g[].words[4 * i + 2] == other:
            g[].words[4 * i + 3] = epoch
            return
    # Retain both owners for the process so their addresses never recycle.
    ref cpy = Python().cpython()
    cpy.Py_IncRef(cls._obj_ptr)
    cpy.Py_IncRef(extra._obj_ptr)
    g[].words.append(kind)
    g[].words.append(address)
    g[].words.append(other)
    g[].words.append(epoch)


struct P3NumpyClasses(Defaultable, Movable):
    var factory: Int
    # Exact NumPy real scalar types an object-dtype element may have.
    var scalars: List[Int]
    # dtype classes `_prepare` accepts before any dtype access.
    var dtypes: List[Int]

    def __init__(out self):
        self.factory = 0
        self.scalars = List[Int]()
        self.dtypes = List[Int]()


comptime P3_NUMPY_CLASSES = _Global[
    StorageType=P3NumpyClasses, name="MojolearnP3ArrayArenaNumpyClassesV1",
    init_fn=P3NumpyClasses.__init__,
]


def _p3_numpy_fill(factory: PythonObject) raises:
    var g = P3_NUMPY_CLASSES.get_or_create_ptr()
    if g[].factory == Int(factory._obj_ptr):
        return
    ref cpy = Python().cpython()
    var scalars = List[Int]()
    for code in ["i1","u1","i2","u2","i4","u4","i8","u8","f2","f4","f8","g","q","Q"]:  # driver-loop: once per connected factory; the exact real scalar interfaces
        var scalar = py_call(factory,code).type
        cpy.Py_IncRef(scalar._obj_ptr)
        scalars.append(Int(scalar._obj_ptr))
    var dtypes = List[Int]()
    for code in ["?","i1","u1","i2","u2","i4","u4","i8","u8","f2","f4","f8","O","c8","c16","q","Q"]:  # driver-loop: once per connected factory; the exact dtype classes
        var cls = py_type(py_call(factory,code))
        cpy.Py_IncRef(cls._obj_ptr)
        dtypes.append(Int(cls._obj_ptr))
    cpy.Py_IncRef(factory._obj_ptr)
    g[].scalars = scalars^
    g[].dtypes = dtypes^
    g[].factory = Int(factory._obj_ptr)


def p3_numpy_scalar_class(factory: PythonObject, cls: PythonObject) raises -> Bool:
    """True when `cls` is one of the exact NumPy real scalar types."""
    _p3_numpy_fill(factory)
    var g = P3_NUMPY_CLASSES.get_or_create_ptr()
    var address = Int(cls._obj_ptr)
    for i in range(len(g[].scalars)):  # driver-loop: fixed set of 14 class addresses
        if g[].scalars[i] == address:
            return True
    return False


def p3_numpy_scalar_classes(factory: PythonObject) raises -> List[Int]:
    """The exact NumPy real scalar type addresses (copy, for a native loop)."""
    _p3_numpy_fill(factory)
    var g = P3_NUMPY_CLASSES.get_or_create_ptr()
    return g[].scalars.copy()


def p3_numpy_dtype_class(factory: PythonObject, cls: PythonObject) raises -> Bool:
    """True when `cls` is one of the exact supported dtype classes."""
    _p3_numpy_fill(factory)
    var g = P3_NUMPY_CLASSES.get_or_create_ptr()
    var address = Int(cls._obj_ptr)
    for i in range(len(g[].dtypes)):  # driver-loop: fixed set of 17 class addresses
        if g[].dtypes[i] == address:
            return True
    return False
