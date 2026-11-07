# SPDX-License-Identifier: Apache-2.0
# Native public model attribute lookup. Only connected function/native
# descriptors may bind; arbitrary Python properties/hooks are never called.
from bindings.native_loaded_module import native_loaded_module, NativeModules, native_namespace_string_keys
from bindings.admitted_call import AdmittedCall
from bindings.cpython_table import cpython_table
from bindings.native_class_admission import class_record, class_admit, class_watch_ready
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_key_same, prepared_class_stamp, prepared_class_matches
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_error_class import error_without_class
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.native_numpy_metadata import native_numpy_type, native_numpy_module_fields


def _type_fields(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    return py_call(b.type.__dict__["__dict__"].__get__, cls)


def _type_mro(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    return py_call(b.type.__dict__["__mro__"].__get__, cls)


def _descriptor_slot(cls: PythonObject, name: String) raises -> Bool:
    var mro = _type_mro(cls)
    for k in range(len(mro)):  # driver-loop: over native descriptor class metadata
        if py_contains(_type_fields(mro[k]), name):
            return True
    return False


def _public_plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = model if py_type(model) is b.type else py_type(model)
    if py_type(cls) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: native public attributes cannot invoke Python metaclass hooks")
    var layouts = py_get(context, "classical_public_layouts")
    if layouts is None:
        raise Error("RuntimeError: native public model metadata was not connected")
    var plan = py_get(layouts, cls)
    if plan is None:
        raise Error("TypeError: native public model requires an exact connected class")
    return plan


def classical_check_layout(model: PythonObject, context: PythonObject) raises:
    # Callers without an operation handle keep their own module view.
    var mods = NativeModules()
    _ = classical_check_layout(mods, model, context)


def classical_check_layout(mut mods: NativeModules, model: PythonObject, context: PythonObject) raises -> PythonObject:
    # Returns the connected public plan it admitted (None when no installer
    # metadata exists), so classical_getattribute does not resolve it again.
    # Existing private-owner callers have no installer metadata. Once public
    # installation is requested, the complete layout is mandatory and checked.
    if py_get(context, "classical_public_layouts") is None:
        if Bool(py=py_get(context, "classical_public_installed", False)):
            raise Error("RuntimeError: native public model metadata was removed")
        return PythonObject(None)
    var b = mods.module("builtins")
    var cls = model if py_type(model) is b.type else py_type(model)
    var plan = _public_plan(model, context)
    # Prepared class admission: the ancestry, hook, finalizer and dictionary
    # descriptor checks run on first use per class and again whenever the
    # class stamp changes; instance keys are admitted on every call.
    var finalizer = py_get(context, "host_archive_finalizer_descriptor")
    var peer_finalizer = py_get(context, "host_archive_peer_finalizer_descriptor")
    var admitted = prepared_get(mods, _LAYOUT_CLASS_SLOT, cls)
    # Python >= 3.12: the type watchers of native_class_admission report any
    # change of the class or an ancestor (every namespace edit goes through
    # type attribute assignment, which notifies them). While the class is
    # unchanged since the stamp last matched and the three admission inputs
    # are the same objects, the stamp still matches: skip its row-by-row
    # comparison. Python 3.10/3.11 (no watcher) keep the comparison.
    var watched = class_record(mods, _LAYOUT_WATCH_SLOT, cls)
    if (admitted is not None and watched is not None and watched[0] is plan
            and watched[1] is finalizer and watched[2] is peer_finalizer):
        _classical_check_instance(model, context, cls, plan)
        return plan
    if admitted is None or not prepared_class_matches(mods, cls, [plan, finalizer, peer_finalizer], admitted):
        _classical_check_class(model, context, cls, plan)
        var stamp = prepared_class_stamp(mods, cls, Python.list(plan, finalizer, peer_finalizer))
        prepared_put(mods, _LAYOUT_CLASS_SLOT, cls, stamp)
    # The payload holds the admission inputs only, never the class. Not built
    # where no record can be kept (Python 3.10/3.11).
    if class_watch_ready():
        class_admit(mods, _LAYOUT_WATCH_SLOT, cls, Python.tuple(plan, finalizer, peer_finalizer))
    _classical_check_instance(model, context, cls, plan)
    return plan


comptime _LAYOUT_CLASS_SLOT = "classical_layout_class_v1"
comptime _LAYOUT_NAMES_SLOT = "classical_layout_names_v1"
comptime _LAYOUT_WATCH_SLOT = "classical_layout_watch_v1"
comptime _DESCRIPTOR_KIND_SLOT = "classical_descriptor_kind_v1"


def _descriptor_kind(mut mods: NativeModules, member_type: PythonObject) raises -> Int:
    # Bit 1: the member's class defines __get__; 3: it also defines __set__ or
    # __delete__ (a data descriptor). An exact Python function (every method
    # lookup) is answered by identity: its static type defines __get__ only.
    # Other member classes are walked once per watched class state (three
    # ancestry walks per read before).
    if member_type is mods.module("types").FunctionType:
        return 1
    var record = class_record(mods, _DESCRIPTOR_KIND_SLOT, member_type)
    if record is not None:
        return Int(py=record)
    var kind = 0
    if _descriptor_slot(member_type, "__get__"):
        kind = 1
        if _descriptor_slot(member_type, "__set__") or _descriptor_slot(member_type, "__delete__"):
            kind = 3
    class_admit(mods, _DESCRIPTOR_KIND_SLOT, member_type, PythonObject(kind))
    return kind


def _layout_class_names(mut mods: NativeModules, cls: PythonObject, plan: PythonObject) raises -> PythonObject:
    """An exact dict of every name the connected ancestry rows of `cls`
    define, computed once per admitted class layout stamp. Call only after
    classical_check_layout admitted `cls` in the same native call: the stamp
    it left in the registry is then the live one, and the names below were
    read while that stamp was current (any later namespace edit, insertion or
    removal on a mutable row changes the stamp, and the names are read
    again). None: no admitted stamp, or a class row holds a non-string key;
    the caller keeps the full member walk."""
    var stamp = prepared_get(mods, _LAYOUT_CLASS_SLOT, cls)
    if stamp is None:
        return PythonObject(None)
    var record = prepared_get(mods, _LAYOUT_NAMES_SLOT, cls)
    if record is not None and record[0] is stamp:
        return record[1]
    var b = mods.module("builtins")
    var names = Python.dict()
    var usable = True
    var mro = plan["mro"]
    for k in range(len(mro)):  # driver-loop: over connected class ancestry metadata
        var keys = py_call(b.list, py_call(_type_fields(mro[k]).keys))
        for j in range(len(keys)):  # small-loop(keys: one class namespace): exact string names only
            if py_type(keys[j]) is not b.str:
                usable = False
                break
            py_setitem(names, keys[j], PythonObject(True))
        if not usable:
            break
    var payload = names if usable else PythonObject(None)
    # The record retains the stamp (as the layout record itself retains the
    # ancestry); identity, not equality, decides reuse.
    prepared_put(mods, _LAYOUT_NAMES_SLOT, cls, Python.tuple(stamp, payload))
    return payload


def _instance_result(name: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    # A public trailing-underscore Array field reads as its ndarray view once
    # NumPy is connected. Native slot getters + ndarray C type wrap the same
    # owner/storage. Never invoke Array.__array__, properties, or output
    # Python code. Without NumPy the Array itself is the fitted attribute.
    # `name` is an exact str (checked by the caller): native string tests.
    var text = String(name)
    if (text.byte_length() > 0 and text.endswith("_") and not text.startswith("_")
            and py_type(value) is context["array_class"] and py_get(context, "ndarray") is not None):
        sequential_admit_value(value, context)
        return sequential_input(value, context)
    return value


def _classical_check_class(model: PythonObject, context: PythonObject, cls: PythonObject, plan: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var mro = _type_mro(cls)
    var connected = plan["mro"]
    if len(mro) != len(connected):
        raise Error("TypeError: native public model class layout changed")
    var found_fields = False
    var found_finalizer = False
    for k in range(len(mro)):  # driver-loop: over exact connected class ancestry
        if mro[k] is not connected[k]:
            raise Error("TypeError: native public model class ancestry changed")
        if py_contains(_type_fields(mro[k]), "__getattr__"):
            # CPython otherwise invokes this hook after AttributeError from
            # our native getter. Refuse while we still control the boundary.
            raise Error("TypeError: Python model __getattr__ hooks are not accepted")
        if not found_finalizer and py_contains(_type_fields(mro[k]), "__del__"):
            # Explicit CPU archive classes replace device finalizers with a
            # connected native descriptor. Only the first resolved slot can
            # run; a shadowed GPU-base Python finalizer is never invoked.
            # A context shared by a public class and its explicit CPU
            # subclass (the neighbor queries) holds both connected native
            # descriptors: the public one, and the host archive's as its
            # peer, exactly as host_archive_check_layout admits them.
            var member = _type_fields(mro[k])["__del__"]
            var native_finalizer = py_get(context, "host_archive_finalizer_descriptor")
            var peer_finalizer = py_get(context, "host_archive_peer_finalizer_descriptor")
            var connected_finalizer = (native_finalizer is not None and member is native_finalizer) or (
                peer_finalizer is not None and member is peer_finalizer)
            if not connected_finalizer:
                raise Error("TypeError: Python model finalizer hooks are not accepted")
            found_finalizer = True
        if not found_fields:
            var descriptor = py_get(_type_fields(mro[k]), "__dict__")
            if descriptor is not None:
                if descriptor is not plan["fields_descriptor"]:
                    raise Error("TypeError: Python model dictionary descriptors are not accepted")
                found_fields = True
    if not found_fields:
        raise Error("TypeError: native public model dictionary descriptor is missing")


def _classical_check_instance(model: PythonObject, context: PythonObject, cls: PythonObject, plan: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(model) is not b.type:
        var fields = py_call(plan["fields_get"], model, cls)
        if py_type(fields) is b.dict:
            # Native key walk of the exact instance dict: no key view, list
            # or per-key Python indexing. Same refusal.
            if not native_namespace_string_keys(fields._obj_ptr, b.str):
                raise Error("TypeError: native model dictionary requires exact string keys")
            return
        var keys = py_call(py_type(Python.list()), fields.keys())
        for k in range(len(keys)):  # driver-loop: over native instance attribute names
            if py_type(keys[k]) is not b.str:
                # Refuse before a dict lookup can compare a malicious key
                # against one of the native implementation's string names.
                raise Error("TypeError: native model dictionary requires exact string keys")


def _classical_descriptor_get(binding: PythonObject, model: PythonObject, cls: PythonObject,
                              plan: PythonObject) raises -> PythonObject:
    # py_call's generic Mojo error keeps the text, not the CPython class.
    # An unset native member must remain AttributeError for getattr/hasattr.
    # Resolve all metadata/functions before the call can set an exception.
    ref cpy = Python().cpython()
    var getter = binding[1]
    var envelope = Python.tuple(model, cls)
    var attribute_error = plan["attribute_error"]
    var table = cpython_table()
    var result = cpy.PyObject_Call(getter._obj_ptr, envelope._obj_ptr, PyObjectPtr())
    if not result:
        var missing = table[].PyErr_ExceptionMatches(attribute_error._obj_ptr) != 0
        var error = cpy.unsafe_get_error()
        if missing:
            raise Error("AttributeError: " + String(error))
        raise error^
    return PythonObject(from_owned=result)


def classical_getattribute(model: PythonObject, name: PythonObject, context: PythonObject,
                           layout_checked: Bool = False) raises -> PythonObject:
    # Callers without an admission record (host density, KDE, neighbours,
    # SVC option reads). `layout_checked`: the caller ran
    # classical_check_layout on this model in the same native call.
    var mods = NativeModules()
    var plan = PythonObject(None)
    if not layout_checked:
        plan = classical_check_layout(mods, model, context)
    return _classical_getattribute_impl(mods, model, name, context, plan)


def classical_check_layout(mut call: AdmittedCall, model: PythonObject, context: PythonObject) raises -> PythonObject:
    # The door's one layout check (ne1-e-admitted-call, row N259): the plan it
    # admitted is kept on the call, so a later read in the same call neither
    # re-checks the class and instance keys nor resolves the plan again.
    if not call.layout_checked:
        call.layout_plan = classical_check_layout(call.modules, model, context)
        call.layout_checked = True
    return call.layout_plan


def classical_getattribute(mut call: AdmittedCall, model: PythonObject, name: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    # The door's admission record: its module view and its admitted plan.
    var plan = classical_check_layout(call, model, context)
    return _classical_getattribute_impl(call.modules, model, name, context, plan)


def _classical_getattribute_impl(mut mods: NativeModules, model: PythonObject, name: PythonObject,
                                 context: PythonObject, admitted_plan: PythonObject) raises -> PythonObject:
    # `admitted_plan`: the plan classical_check_layout returned in this call
    # (None when no installer metadata exists). One module view and one plan
    # resolution for the whole read.
    var b = mods.module("builtins")
    if py_type(name) is not b.str:
        raise Error("TypeError: native model attribute name must be an exact string")
    var plan = admitted_plan
    if plan is None:
        plan = _public_plan(model, context)
    var cls = py_type(model)
    # Instance dictionary first. A name no connected class row defines has
    # no class member, so no descriptor can take precedence: Python's lookup
    # (and the walk below, with `present` False) answers from the instance
    # dictionary or raises AttributeError. One dict probe replaces the member
    # walk and the three descriptor-slot ancestry walks for every fitted
    # field and parameter read.
    var names = _layout_class_names(mods, cls, plan)
    if names is not None and not py_contains(names, name):
        var stored = py_call(plan["fields_get"], model, cls)
        if py_contains(stored, name):
            return _instance_result(name, stored[name], context)
        raise Error("AttributeError: native model has no attribute '" + String(name) + "'")
    var member = PythonObject(None)
    var present = False
    var mro = plan["mro"]
    for k in range(len(mro)):  # driver-loop: over native class attribute lookup order
        var fields = _type_fields(mro[k])
        if py_contains(fields, name):
            member = fields[name]
            present = True
            break
    var binding = py_get(plan["bindings"], name)
    var known = binding is not None and member is binding[0]
    var member_type = py_type(member)
    var kind = _descriptor_kind(mods, member_type) if present else 0
    var descriptor = kind != 0
    var data_descriptor = kind == 3
    if data_descriptor:
        if not known:
            raise Error("TypeError: Python model property/descriptor hooks are not accepted")
        return _classical_descriptor_get(binding, model, cls, plan)
    var fields = py_call(plan["fields_get"], model, cls)
    if py_contains(fields, name):
        return _instance_result(name, fields[name], context)
    if not present:
        raise Error("AttributeError: native model has no attribute '" + String(name) + "'")
    if descriptor:
        if not known:
            raise Error("TypeError: unconnected Python model descriptors are not accepted")
        return _classical_descriptor_get(binding, model, cls, plan)
    return member


def _connect_probe(mapping: PythonObject, key: StaticString) raises -> Int:
    # One native dictionary probe for the connected-NumPy fast path, in place
    # of a bound `mapping.get` lookup plus a Python call. Returns 1 when the
    # exact dict holds a non-None value at `key`, 0 when it holds None or
    # nothing, and -1 when `mapping` is not an exact dict (the caller then
    # keeps the general `py_get` path). The connected contexts are package
    # dictionaries keyed by exact str names; the result is borrowed and only
    # compared against None, so no reference escapes.
    ref cpy = Python().cpython()
    if py_type(mapping) is not py_type(Python.dict()):
        return -1
    var name = PythonObject(String(key))
    var raw = cpy.PyDict_GetItemWithError(mapping._obj_ptr, name._obj_ptr)
    if not raw:
        if cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        return 0
    var none = PythonObject(None)
    var found = 0 if raw == none._obj_ptr else 1
    _ = mapping
    _ = name
    return found


def classical_connect_numpy(context: PythonObject) raises:
    """Connect loaded NumPy's native types without a Python setup callback.

    Idempotent and cheap after the first connection: neural scalar admission
    calls this per value, so the connected case is one native dict probe."""
    var connected = _connect_probe(context, "ndarray")
    if connected == 1:
        return
    if connected == -1 and py_get(context, "ndarray") is not None:
        return
    # NumPy not loaded: the modules table probe is also native when exact.
    var modules = context["modules"]
    if _connect_probe(modules, "numpy") == 0:
        return
    var numpy = py_get(modules, "numpy")
    if numpy is None:
        return
    if py_type(numpy) is not context["native_module_type"]:
        raise Error("TypeError: classical NumPy support requires the native NumPy module")
    # Read the loaded module dictionary directly; module __getattr__ is a
    # Python hook. All dtype construction below uses NumPy's C dtype type.
    var fields = native_numpy_module_fields(numpy)
    # C-created builtin values supply stable constructor/type identities.
    # These names may be replaced in builtins after package import; no
    # replacement becomes a callback or changes scalar admission metadata.
    var list_type = py_type(Python.list())
    var tuple_type = py_type(Python.tuple())
    var int_type = py_type(PythonObject(0))
    var float_type = py_type(PythonObject(0.0))
    var bool_type = py_type(PythonObject(True))
    # Check both factory identities before invoking or publishing either.
    # A late NumPy import must not turn first-call metadata setup into a call
    # to an arbitrary Python replacement of numpy.dtype or numpy.ndarray.
    var dtype_class = native_numpy_type("dtype", context)
    var ndarray = native_numpy_type("ndarray", context)
    var numeric = Python.dict()
    var classes = Python.dict()
    var scalars = py_call(list_type)
    var integers = py_call(list_type, Python.tuple(int_type))
    var kinds = Python.tuple("u", "i", "f", "b")
    var sizes = Python.tuple(1, 2, 4, 8)
    for kind in range(4):  # driver-loop: over fixed native scalar ABI metadata
        for index in range(4):  # driver-loop: over supported scalar widths
            var size = Int(py=sizes[index])
            if (kind == 2 and size == 1) or (kind == 3 and size != 1):
                continue
            var code = "?" if kind == 3 else String(kinds[kind]) + String(size)
            var dtype = py_call(dtype_class, code)
            py_setitem(numeric, Python.tuple(kinds[kind], size), dtype)
            py_setitem(classes, Python.tuple(kind, size), dtype)
            _ = py_call(scalars.append, dtype.type)
            if kind < 2:
                _ = py_call(integers.append, dtype.type)
    var values = py_call(list_type, context["native_value_types"])
    var owners = py_call(list_type, context["buffer_owner_types"])
    _ = py_call(values.append, ndarray)
    # NumPy's text and byte scalars (numpy.str_, numpy.bytes_) are C types
    # with no Python protocol hooks; label lists built from a NumPy string
    # array hold them (scikit-learn accepts such labels).
    var text_names = Python.tuple("str_", "bytes_")
    for i in range(len(text_names)):  # driver-loop: over two fixed NumPy scalar type names
        var text_type = py_get(fields, text_names[i])
        if text_type is not None and py_type(text_type) is py_type(int_type):
            _ = py_call(values.append, text_type)
    _ = py_call(owners.append, ndarray)
    var archive = Python.dict()
    var archive_codes = Python.tuple("<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2",
                                     "=f4", "=f8", "=i4", "=i8", "=u4", "|u1", "|b1")
    for i in range(len(archive_codes)):  # driver-loop: over supported NPY header dtype metadata
        var code = archive_codes[i]
        # The archive's Boolean payload is represented by byte storage.
        var dtype = py_call(dtype_class, "u1" if String(code) == "|b1" else String(code))
        py_setitem(archive, code, dtype)
    var update = Python.dict(
        ndarray=ndarray, scalar_type=fields["generic"], integer=fields["integer"],
        scalar_types=py_call(tuple_type, scalars), integer_types=py_call(tuple_type, integers),
        native_value_types=py_call(tuple_type, values), buffer_owner_types=py_call(tuple_type, owners),
        numeric_dtypes=numeric, class_dtypes=classes, archive_npy_dtypes=archive,
        float_types=Python.tuple(float_type, fields["floating"]),
        int_types=Python.tuple(int_type, fields["integer"]), bool_types=Python.tuple(bool_type, fields["bool_"]),
        float32=py_call(dtype_class, "f4"), float64=py_call(dtype_class, "f8"),
        int32=py_call(dtype_class, "i4"), int64=py_call(dtype_class, "i8"),
        uint64=py_call(dtype_class, "u8"), bool=py_call(dtype_class, "?"))
    # Publish together after setup succeeds. Only admission metadata changes:
    # explicit Host* vendor and binding entries retain their CPU ownership.
    _ = py_call(context.update, update)


def _attribute_context(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var host = py_get(context, "classical_host_context")
    if host is not None and py_contains(host["classical_kinds"], py_type(model)):
        return host
    return context


def classical_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var model = args[1]
        var context = _attribute_context(model, args[0])
        classical_connect_numpy(context)
        var result = classical_getattribute(model, args[2], context)
        return result.steal_data()
    except error:
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)


def classical_attribute_descriptor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        ref cpy = Python().cpython()
        var function = cpy.PyTuple_GetItem(args_ptr, 0)
        if not function:
            raise cpy.unsafe_get_error()
        var value = PythonObject(from_borrowed=function)
        var b = native_loaded_module("builtins")
        var types = native_loaded_module("types")
        if py_type(value) is not types.MethodType or py_type(value.__func__) is not types.BuiltinFunctionType:
            raise Error("native attribute descriptor requires a bound compiled entry")
        # Documented CPython API, present throughout package Python >=3.10.
        # Borrows `function`; owns one new reference returned directly to the
        # C caller. No stdlib patch, raw type-slot edit or refcount adjustment.
        # https://docs.python.org/3.10/c-api/method.html#c.PyInstanceMethod_New
        # Resolve against std.python's already selected interpreter library.
        # OwnedDLHandle.get_function retains the library borrow through the
        # call; an extension must not assume CPython symbols link globally.
        var make_descriptor = cpy.lib.get_function[PyObjectPtr]("PyInstanceMethod_New")
        var descriptor = make_descriptor(function)
        if not descriptor:
            raise cpy.unsafe_get_error()
        return descriptor
    except error:
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
