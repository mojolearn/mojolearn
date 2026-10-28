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
from std.sys import size_of
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_loaded_module import native_loaded_module, _loaded_fields
from bindings.mode_metadata_native import _class_fields

comptime PCSNextUInt32 = def(Int) thin abi("C") -> UInt32


def pcs_seed(seed: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(seed) is b.int:
        return seed
    var record = py_get(context, "family_pcs_rng")
    if record is None:
        raise Error("ImportError: PolynomialCountSketch RandomState/None requires optional NumPy support; pass an integer seed or install mojolearn[numpy]")
    if py_type(record) is not b.dict:
        raise Error("TypeError: PCS RNG admission requires its import-connected metadata")
    var state = seed
    var module = native_loaded_module("numpy.random.mtrand")
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
    if py_type(lock) is not record["lock_type"]:
        raise Error("TypeError: native PCS RNG requires NumPy's ordinary native lock")
    var lock_fields = _class_fields(record["lock_type"])
    if py_get(lock_fields, "acquire") is not record["acquire"] or py_get(lock_fields, "release") is not record["release"]:
        raise Error("TypeError: native PCS RNG lock methods changed")
    ref cpy = Python().cpython()
    var name = String("BitGenerator")
    var cname = name.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin()
    if cpy.lib.get_function[Int32]("PyCapsule_IsValid")(capsule._obj_ptr, cname) == 0:
        raise Error("TypeError: NumPy RNG requires its named BitGenerator capsule")
    var raw = cpy.lib.get_function[Optional[MutPointer[UInt8, MutUntrackedOrigin]]]("PyCapsule_GetPointer")(capsule._obj_ptr, cname)
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
    _ = py_call(record["acquire"], lock)
    var word = UInt32(0)
    try:
        word = next(words[0]) & UInt32(0x7FFFFFFF)
    finally:
        _ = py_call(record["release"], lock)
    _ = state
    _ = bitgen
    _ = capsule
    _ = lock
    return PythonObject(Int(word))
