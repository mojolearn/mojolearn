# SPDX-License-Identifier: Apache-2.0
# Provider-boundary ownership and dispatch for independent captured programs.
# Python objects remain here under the GIL; only the selected image's typed
# distributed/partition implementation receives the execution request.
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.native_loaded_module import native_loaded_module
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from providers.python.autograd_program_arrays_pyapi import program_admission_context
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, NativeDescriptorResource, neural_descriptor_entry,
    neural_descriptor_connect, neural_descriptor_execute, neural_descriptor_adopt,
)

comptime OWNED_PROGRAM_ABI_NAME = "mojolearn.owned_program.public.v1"


struct _OwnedProgramConnection(Movable):
    var classes: PythonObject
    var layouts: PythonObject
    var binding: PythonObject
    var provider: PythonObject
    var executor: Int

    def __init__(out self, classes: PythonObject, layouts: PythonObject,
                 binding: PythonObject, provider: PythonObject, executor: Int):
        self.classes = classes
        self.layouts = layouts
        self.binding = binding
        self.provider = provider
        self.executor = executor


struct _OwnedProgramConnections(Defaultable, Movable):
    var rows: List[_OwnedProgramConnection]

    def __init__(out self):
        self.rows = List[_OwnedProgramConnection]()


comptime _OWNED_PROGRAMS = _Global[StorageType=_OwnedProgramConnections,
    name="MojolearnOwnedProgramPublicV1", init_fn=_OwnedProgramConnections.__init__]


struct NativeOwnedProgramRequest(Movable):
    var owner: PythonObject
    var operation: PythonObject
    var args: PythonObject
    var handle: Int
    var result: PythonObject

    def __init__(out self, owner: PythonObject, operation: PythonObject,
                 args: PythonObject, handle: Int):
        self.owner = owner
        self.operation = operation
        self.args = args
        self.handle = handle
        self.result = PythonObject(None)


struct NativeOwnedProgramCreateRequest(Movable):
    var lifetime: PythonObject
    var binding: PythonObject
    var cls: PythonObject
    var operation: String
    var sid: Int
    var pid: Int
    var args: PythonObject
    var result: PythonObject

    def __init__(out self, lifetime: PythonObject, binding: PythonObject,
                 cls: PythonObject, var operation: String, sid: Int, pid: Int, args: PythonObject):
        self.lifetime = lifetime
        self.binding = binding
        self.cls = cls
        self.operation = operation^
        self.sid = sid
        self.pid = pid
        self.args = args
        self.result = PythonObject(None)


def _owned_program_layout(cls: PythonObject) raises -> PythonObject:
    # Import-time only: pin direct object ancestry and all lookup/storage hooks.
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: captured program class requires the native type metaclass")
    var bases = py_call(b.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 1 or bases[0] is not b.object:
        raise Error("TypeError: captured program class requires direct object ancestry")
    var namespace = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var dictionary = py_get(namespace, "__dict__")
    if (py_type(dictionary) is not py_type(b.type.__dict__["__dict__"])
            or dictionary.__objclass__ is not cls):
        raise Error("TypeError: captured program requires its native instance dictionary descriptor")
    var forbidden: List[String] = ["__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__", "_b", "_id", "_lifetime"]
    for name in forbidden:  # driver-loop: over storage hooks that must remain native before connecting a new owner class
        if py_get(namespace, name) is not None:
            raise Error("TypeError: captured program connection refuses Python storage or finalizer hooks")
    var layout = Python.list()
    var names: List[String] = ["__dict__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__", "_b", "_id", "_lifetime"]
    for name in names:  # driver-loop: over fixed storage and lifecycle hook identities
        _ = py_call(layout.append, Python.tuple(PythonObject(name), py_get(namespace, name)))
    return layout


def ops_owned_program_connect_py(distributed_class: PythonObject, partitioned_class: PythonObject,
                                  binding: PythonObject, module: PythonObject) raises -> PythonObject:
    # Register every installed connection in each connected image at import.
    # Public execution never calls this factory or resolves a Python callable.
    var classes = Python.tuple(distributed_class, partitioned_class)
    var layouts = Python.tuple(_owned_program_layout(distributed_class), _owned_program_layout(partitioned_class))
    var provider = neural_descriptor_connect(neural_descriptor_entry(module, "owned_program_public_api_v1"),
                                             String(OWNED_PROGRAM_ABI_NAME), 4)
    var executor = neural_descriptor_execute(provider, 2, List[Int](), List[Int](), List[Float64](), False)
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    for i in range(len(state[].rows)):  # driver-loop: over fixed import-time provider connections
        if state[].rows[i].binding is binding:
            raise Error("TypeError: captured program connection is already registered")
    state[].rows.append(_OwnedProgramConnection(classes, layouts, binding, provider, executor))
    return PythonObject(None)


def _owned_program_class(cls: PythonObject) raises -> Int:
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    if len(state[].rows) == 0:
        raise Error("TypeError: captured program providers must be connected during package import")
    ref row = state[].rows[0]
    var kind = -1
    for i in range(2):  # small-loop(2: distributed and partitioned native owner classes)
        if cls is row.classes[i]:
            kind = i
    if kind == -1:
        raise Error("TypeError: captured program requires its exact connected owner class")
    var b = native_loaded_module("builtins")
    var bases = py_call(b.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 1 or bases[0] is not b.object:
        raise Error("TypeError: captured program ancestry changed after connection")
    var namespace = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var layout = row.layouts[kind]
    for i in range(len(layout)):  # driver-loop: over connected native owner storage and lifecycle identities
        if py_get(namespace, layout[i][0]) is not layout[i][1]:
            raise Error("TypeError: captured program storage or attribute hook changed after connection")
    return kind


def _owned_program_fields(owner: PythonObject) raises -> Tuple[PythonObject, Int]:
    var kind = _owned_program_class(py_type(owner))
    var b = native_loaded_module("builtins")
    var fields = py_fields(owner)
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: captured program requires an exact instance dictionary")
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: over exact string owner keys before any lookup or mutation
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: captured program fields require exact string keys")
    return (fields, kind)


def _connection(binding: PythonObject) raises -> Int:
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    for i in range(len(state[].rows)):  # driver-loop: over exact retained binding owners, never instance-selected callables
        if state[].rows[i].binding is binding:
            return i
    raise Error("TypeError: captured program requires its connected native binding owner")


def _release_previous(fields: PythonObject, binding: PythonObject, kind: Int, retained: PythonObject) raises:
    var context = program_admission_context()
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    if py_contains(fields, "_b") and fields["_b"] is not binding:
        var pinned = False
        for i in range(len(state[].rows)):  # driver-loop: over independently retained native connection owners
            if fields["_b"] is state[].rows[i].binding:
                pinned = True
        if not pinned:
            native_release_admit(fields["_b"], context)
    if py_contains(fields, "_id"):
        native_release_admit(fields["_id"], context)
    var previous = py_get(fields, "_lifetime")
    if previous is not None and previous is not retained:
        native_release_weakrefs(previous, context)
        var resource = previous.downcast_value_ptr[NativeDescriptorResource]()
        var admitted = False
        for i in range(len(state[].rows)):  # driver-loop: over prior native registry owners before releasing a resource
            if (resource[].provider is state[].rows[i].provider and len(resource[].context) == 2
                    and resource[].context[0] == kind and resource[].context[1] == state[].rows[i].executor
                    and resource[].close_operation == 1):
                admitted = True
        if not admitted:
            raise Error("TypeError: captured program replacement requires its retained native lifetime")


def owned_program_initialize_py(owner: PythonObject, binding: PythonObject, handle: PythonObject) raises -> PythonObject:
    var admitted = _owned_program_fields(owner)
    var fields = admitted[0]
    if py_type(handle) is not py_type(PythonObject(0)) or Int(py=handle) <= 0:
        raise Error("TypeError: captured program handle requires a positive native integer")
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    ref row = state[].rows[_connection(binding)]
    var previous = py_get(fields, "_lifetime")
    var retained = PythonObject(None)
    if previous is not None:
        var held = previous.downcast_value_ptr[NativeDescriptorResource]()
        if (held[].provider is row.provider and held[].handle == Int(py=handle) and len(held[].context) == 2
                and held[].context[0] == admitted[1] and held[].context[1] == row.executor
                and held[].close_operation == 1):
            retained = previous
    _release_previous(fields, binding, admitted[1], retained)
    if retained is not None:
        py_setitem(fields, "_b", binding)
        py_setitem(fields, "_id", handle)
        return PythonObject(None)
    var requests = List[NativeOwnedProgramRequest]()
    requests.append(NativeOwnedProgramRequest(owner, PythonObject("enter"), Python.tuple(), Int(py=handle)))
    _ = neural_descriptor_execute(row.provider, 0, [Int(MutPointer(to=requests[0]))],
                                  [admitted[1], row.executor], List[Float64](), False)
    var resource = neural_descriptor_adopt(row.provider, Int(py=handle), 1, [admitted[1], row.executor])
    py_setitem(fields, "_b", binding)
    py_setitem(fields, "_id", handle)
    py_setitem(fields, "_lifetime", resource)
    return PythonObject(None)


def owned_program_object_py(cls: PythonObject, binding: PythonObject, handle: PythonObject) raises -> PythonObject:
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    var kind = _owned_program_class(cls)
    ref row = state[].rows[_connection(binding)]
    if py_type(handle) is not py_type(PythonObject(0)) or Int(py=handle) <= 0:
        raise Error("TypeError: captured program handle requires a positive native integer")
    # Take ownership before allocation, so a failed native shell allocation
    # closes the newly created registry entry. No Python constructor runs.
    var resource = neural_descriptor_adopt(row.provider, Int(py=handle), 1, [kind, row.executor])
    var b = native_loaded_module("builtins")
    var owner = py_call(b.object.__new__, cls)
    var fields = _owned_program_fields(owner)[0]
    py_setitem(fields, "_b", binding)
    py_setitem(fields, "_id", handle)
    py_setitem(fields, "_lifetime", resource)
    return owner


def ops_owned_program_create_request_py(state: PythonObject, pid: Int, operation: String,
                                        args: PythonObject) raises -> PythonObject:
    # Called only after the Session/Program public owner admitted its storage.
    # Provider selection uses the native Session lifetime, never _b or a callable.
    var lifetime = state["_lifetime"]
    var source = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    var model = source[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if source[].handle == 0 or source[].handle != Int(py=state["_id"]):
        raise Error("ValueError: this Session is closed or its handle changed")
    var connections = _OWNED_PROGRAMS.get_or_create_ptr()
    var kind = 1 if operation == "partition" or operation == "load_partitioned" else 0
    for i in range(len(connections[].rows)):  # driver-loop: over import-time providers for the exact Session image
        ref row = connections[].rows[i]
        var provider = row.provider.downcast_value_ptr[NeuralDescriptorProvider]()
        if provider[].module is model[].module:
            _ = _owned_program_class(row.classes[kind])
            var requests = List[NativeOwnedProgramCreateRequest]()
            requests.append(NativeOwnedProgramCreateRequest(lifetime, row.binding, row.classes[kind],
                            operation, source[].handle, pid, args))
            _ = neural_descriptor_execute(row.provider, 3, [Int(MutPointer(to=requests[0]))],
                                           [kind, row.executor], List[Float64](), False)
            return requests[0].result
    raise Error("TypeError: Session has no connected captured-program provider")


def ops_owned_program_call_py(owner: PythonObject, operation: PythonObject, args: PythonObject) raises -> PythonObject:
    if py_type(operation) is not py_type(PythonObject("")) or py_type(args) is not py_type(Python.tuple()):
        raise Error("TypeError: captured program requires an exact operation string and argument tuple")
    var name = String(py=operation)
    if name == "initialize":
        if len(args) != 2:
            raise Error("TypeError: captured program initialization requires binding and handle")
        return owned_program_initialize_py(owner, args[0], args[1])
    var admitted = _owned_program_fields(owner)
    var lifetime = admitted[0]["_lifetime"]
    var resource = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    var provider = resource[].provider
    var descriptor = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if descriptor[].name != OWNED_PROGRAM_ABI_NAME or len(resource[].context) != 2 or resource[].context[0] != admitted[1]:
        raise Error("TypeError: captured program resource belongs to another native owner")
    var handle = admitted[0]["_id"]
    if py_type(handle) is not py_type(PythonObject(0)) or Int(py=handle) != resource[].handle or resource[].handle == 0:
        raise Error("TypeError: captured program handle is closed or differs from its native owner")
    if name == "close" or name == "exit":
        if len(args) != 0:
            raise Error("TypeError: captured program close accepts no arguments")
        resource[].close()
        if name == "exit":
            return PythonObject(None)
        return PythonObject(0)
    var requests = List[NativeOwnedProgramRequest]()
    requests.append(NativeOwnedProgramRequest(owner, operation, args, resource[].handle))
    _ = neural_descriptor_execute(provider, 0, [Int(MutPointer(to=requests[0]))],
                                   resource[].context, List[Float64](), False)
    return requests[0].result
