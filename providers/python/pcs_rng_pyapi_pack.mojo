# SPDX-License-Identifier: Apache-2.0
# PolynomialCountSketch's single RandomState seed draw, through NumPy's public
# BitGenerator capsule ABI, never a RandomState method or Python RNG callback.
# Public ABI: https://numpy.org/doc/stable/reference/random/extending.html
# Source pinned during this repair: NumPy v2.4.4 numpy/random/bitgen.h,
# mtrand.pyx:174,216-226,779-801 and src/distributions/distributions.c:1159,
# 1538-1570. randint(0, 2**31, size=1) uses the masked uint32 path for both
# int32 and int64 outputs: one next_uint32(state) & 0x7fffffff, under the
# BitGenerator lock. All other BitGenerator state is opaque and untouched.
# Retained source: ~/mojolearn-evidence/peer1-radius-native-20261027/numpy-source/.
# Unsupported: assigning RandomState._bit_generator directly can desynchronize
# NumPy's private copied _bitgen/lock from that attribute. Only normally created
# RandomStates and NumPy's supported state/reseed/set_bit_generator APIs are in
# this contract. We neither inspect private object layouts nor repair them.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.sys import size_of
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.mode_metadata_native import _class_fields

comptime PCSStatePointer = MutPointer[UInt8, MutUntrackedOrigin]
comptime PCSNextUInt32 = def(PCSStatePointer) thin abi("C") -> UInt32


comptime _PCS_NO_NUMPY = "ImportError: PolynomialCountSketch RandomState/None requires optional NumPy support; pass an integer seed or install mojolearn[numpy]"


def _pcs_class(fields: PythonObject, name: String) raises -> PythonObject:
    var cls = py_get(fields, PythonObject(name))
    if cls is None:
        raise Error("TypeError: PCS RNG requires NumPy's native " + name + " BitGenerator")
    return cls


def _pcs_connect(context: PythonObject) raises -> PythonObject:
    """The NumPy RNG metadata record (exact module, class, descriptor and lock
    identities), connected natively at the first non-integer seed and kept in
    the shared family context. Importing the package never loads NumPy
    (NUMPY_FREE_CONTRACT.md): numpy.random is read from the loaded-module
    registry, or imported here, inside the call that needs it. The record
    holds the same identities the former import-time hook pinned
    (_expansion_neighbors._connect_pcs_rng), so the draw is unchanged."""
    var sys = native_loaded_module("sys")
    var random = py_get(sys.modules, "numpy.random")
    if random is None:
        try:
            # import-time: one-time cold connection of optional NumPy RNG metadata; the record is kept in the context
            random = Python.import_module("numpy.random")
        except error:
            raise Error(String(_PCS_NO_NUMPY) + " (" + String(error) + ")")
    var mtrand = py_get(sys.modules, "numpy.random.mtrand")
    var module_type = py_type(sys)
    if py_type(random) is not module_type or py_type(mtrand) is not module_type:
        raise Error("TypeError: PCS RNG requires the native numpy.random modules")
    # Module dictionaries are read directly: no module attribute hook runs.
    var fields = _loaded_fields(random)
    var mtrand_fields = _loaded_fields(mtrand)
    var randomstate = py_get(mtrand_fields, "RandomState")
    var bitgenerator = py_get(fields, "BitGenerator")
    if randomstate is None or bitgenerator is None:
        raise Error("TypeError: PCS RNG requires NumPy's native RandomState and BitGenerator")
    var bitgenerators = Python.tuple(_pcs_class(fields, "MT19937"), _pcs_class(fields, "PCG64"),
                                     _pcs_class(fields, "PCG64DXSM"), _pcs_class(fields, "Philox"),
                                     _pcs_class(fields, "SFC64"))
    var state_getter = py_get(_class_fields(randomstate), "_bit_generator")
    var base_fields = _class_fields(bitgenerator)
    var capsule_getter = py_get(base_fields, "capsule")
    var lock_getter = py_get(base_fields, "lock")
    var types = native_loaded_module("types")
    var getset_type = types.GetSetDescriptorType
    var method_type = types.MethodDescriptorType
    if (py_type(state_getter) is not getset_type or py_type(capsule_getter) is not getset_type
            or py_type(lock_getter) is not getset_type):
        raise Error("TypeError: PCS RNG requires native NumPy state/capsule/lock getters")
    var thread = native_loaded_module("_thread")
    var lock_type = thread.LockType
    var lock_fields = _class_fields(lock_type)
    var acquire = py_get(lock_fields, "acquire")
    var release = py_get(lock_fields, "release")
    if py_type(acquire) is not method_type or py_type(release) is not method_type:
        raise Error("TypeError: PCS RNG requires ordinary native lock methods")
    # NumPy 2.5 BitGenerators hold a `_thread.RLock` (the C re-entrant lock),
    # older ones a `_thread.lock`; both are native, with C acquire/release.
    var rlock_type = py_get(_loaded_fields(thread), "RLock")
    var rlock_acquire = PythonObject(None)
    var rlock_release = PythonObject(None)
    if rlock_type is not None:
        var rlock_fields = _class_fields(rlock_type)
        rlock_acquire = py_get(rlock_fields, "acquire")
        rlock_release = py_get(rlock_fields, "release")
        if py_type(rlock_acquire) is not method_type or py_type(rlock_release) is not method_type:
            raise Error("TypeError: PCS RNG requires ordinary native lock methods")
    var record = Python.dict()
    py_setitem(record, "mtrand", mtrand)
    py_setitem(record, "randomstate", randomstate)
    py_setitem(record, "bitgenerator", bitgenerator)
    py_setitem(record, "bitgenerators", bitgenerators)
    py_setitem(record, "state_getter", state_getter)
    py_setitem(record, "state_get", state_getter.__get__)
    py_setitem(record, "capsule_getter", capsule_getter)
    py_setitem(record, "capsule_get", capsule_getter.__get__)
    py_setitem(record, "lock_getter", lock_getter)
    py_setitem(record, "lock_get", lock_getter.__get__)
    py_setitem(record, "lock_type", lock_type)
    py_setitem(record, "acquire", acquire)
    py_setitem(record, "release", release)
    py_setitem(record, "rlock_type", rlock_type)
    py_setitem(record, "rlock_acquire", rlock_acquire)
    py_setitem(record, "rlock_release", rlock_release)
    # A second thread that connects concurrently (the import above may run
    # Python and switch threads) stores a record of the same identities.
    py_setitem(context, "family_pcs_rng", record)
    return record


def pcs_seed(seed: PythonObject, context: PythonObject) raises -> PythonObject:
    comptime assert size_of[PCSStatePointer]() == size_of[Int](), "NumPy bitgen_t requires pointer-sized Int"
    comptime assert size_of[PCSNextUInt32]() == size_of[PCSStatePointer](), "NumPy bitgen_t requires pointer-sized native function pointers"
    var b = native_loaded_module("builtins")
    if py_type(seed) is b.int:
        return seed
    var record = py_get(context, "family_pcs_rng")
    if record is None:
        record = _pcs_connect(context)
    if py_type(record) is not b.dict:
        raise Error("TypeError: PCS RNG admission requires its natively connected metadata")
    var state = seed
    var module = py_get(native_loaded_module("sys").modules, "numpy.random.mtrand")
    if module is not record["mtrand"]:
        raise Error("TypeError: NumPy RandomState module changed after native connection")
    if seed is None:
        state = py_get(_loaded_fields(module), "_rand")
    if py_type(state) is not record["randomstate"]:
        raise Error("TypeError: PolynomialCountSketch random_state requires an exact int, native numpy.random.RandomState, or None; Python RNG callbacks are not supported")
    if py_get(_class_fields(record["randomstate"]), "_bit_generator") is not record["state_getter"]:
        raise Error("TypeError: NumPy RandomState native BitGenerator getter changed")
    # Exact Cython getset descriptor; do not dispatch state.__getattribute__.
    var bitgen = py_call(record["state_get"], state, record["randomstate"])
    var allowed = record["bitgenerators"]
    var found = False
    for i in range(len(allowed)):  # small-loop(five approved native NumPy BitGenerator classes)
        found = found or py_type(bitgen) is allowed[i]
    if not found:
        raise Error("TypeError: native PCS RandomState supports NumPy MT19937, PCG64, PCG64DXSM, Philox and SFC64; custom BitGenerator callbacks are not supported")
    var base_fields = _class_fields(record["bitgenerator"])
    if py_get(base_fields, "capsule") is not record["capsule_getter"] or py_get(base_fields, "lock") is not record["lock_getter"]:
        raise Error("TypeError: NumPy native capsule/lock getters changed")
    var capsule = py_call(record["capsule_get"], bitgen, py_type(bitgen))
    var lock = py_call(record["lock_get"], bitgen, py_type(bitgen))
    var held_type = record["lock_type"]
    var acquire = record["acquire"]
    var release = record["release"]
    if py_type(lock) is not held_type:
        if record["rlock_type"] is None or py_type(lock) is not record["rlock_type"]:
            raise Error("TypeError: native PCS RNG requires NumPy's ordinary native lock")
        held_type = record["rlock_type"]
        acquire = record["rlock_acquire"]
        release = record["rlock_release"]
    var lock_fields = _class_fields(held_type)
    if py_get(lock_fields, "acquire") is not acquire or py_get(lock_fields, "release") is not release:
        raise Error("TypeError: native PCS RNG lock methods changed")
    ref cpy = Python().cpython()
    var name = String("BitGenerator")
    var cname = name.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin()
    if cpy.lib.get_function[Int32]("PyCapsule_IsValid")(capsule._obj_ptr, cname) == 0:
        raise Error("TypeError: NumPy RNG requires its named BitGenerator capsule")
    var raw = cpy.lib.get_function[Optional[MutPointer[UInt8, MutUntrackedOrigin]]]("PyCapsule_GetPointer")(capsule._obj_ptr, cname)
    _ = name^
    if not raw:
        raise cpy.unsafe_get_error()
    var address = Int(raw.value())
    # Public bitgen_t: opaque state, uint64 function, uint32 function, double
    # function, raw function. Supported CPython targets use 64-bit pointers.
    var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    if words[0] == 0 or words[2] == 0:
        raise Error("TypeError: NumPy BitGenerator capsule lacks native state/next_uint32")
    var next = MutPointer[PCSNextUInt32, MutUntrackedOrigin](unsafe_from_address=address + 2 * size_of[Int]())[0]
    # Pin exact native lock descriptors. acquire releases the GIL while blocked;
    # the one native draw needs no Python execution and keeps every owner alive.
    # Allocate the call envelope before acquisition. In particular, do not use
    # py_call in finally: a failed tuple allocation there could strand the lock.
    var lock_args = cpy.PyTuple_New(1)
    if not lock_args:
        raise cpy.unsafe_get_error()
    _ = cpy.PyTuple_SetItem(lock_args, 0, cpy.Py_NewRef(lock._obj_ptr))
    var envelope = PythonObject(from_owned=lock_args)
    var acquired = cpy.PyObject_Call(acquire._obj_ptr, envelope._obj_ptr, PyObjectPtr())
    if not acquired:
        raise cpy.unsafe_get_error()
    var acquired_owner = PythonObject(from_owned=acquired)
    var word = UInt32(0)
    try:
        word = next(PCSStatePointer(unsafe_from_address=words[0])) & UInt32(0x7FFFFFFF)
    finally:
        var released = cpy.PyObject_Call(release._obj_ptr, envelope._obj_ptr, PyObjectPtr())
        if not released:
            raise cpy.unsafe_get_error()
        var released_owner = PythonObject(from_owned=released)
    _ = acquired_owner^
    _ = state^
    _ = bitgen^
    _ = capsule^
    _ = lock^
    return PythonObject(Int(word))
