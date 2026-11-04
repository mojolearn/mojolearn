# SPDX-License-Identifier: Apache-2.0
# Provider-boundary ownership and dispatch for independent captured programs.
# Python objects remain here under the GIL; only the selected image's typed
# distributed/partition implementation receives the execution request.
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem
from bindings.native_loaded_module import native_loaded_module
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


def _owned_program_layout(cls: PythonObject) raises -> PythonObject:
    # Import-time only: pin direct object ancestry and all lookup/storage hooks.
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: captured program class requires the native type metaclass")
    var bases = py_call(b.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 1 or bases[0] is not b.object:
        raise Error("TypeError: captured program class requires direct object ancestry")
    var namespace = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var layout = Python.list()
    for name in List[String]("__dict__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__"):
        _ = py_call(layout.append, Python.tuple(PythonObject(name), py_get(namespace, name)))
    return layout


def ops_owned_program_connect_py(distributed_class: PythonObject, partitioned_class: PythonObject,
                                  binding: PythonObject, module: PythonObject) raises -> PythonObject:
    # Register every installed connection in each connected image at import.
    # Public execution never calls this factory or resolves a Python callable.
    var classes = Python.tuple(distributed_class, partitioned_class)
    var layouts = Python.tuple(_owned_program_layout(distributed_class), _owned_program_layout(partitioned_class))
    var provider = neural_descriptor_connect(neural_descriptor_entry(module, "owned_program_public_api_v1"),
                                             String(OWNED_PROGRAM_ABI_NAME), 3)
    var executor = neural_descriptor_execute(provider, 2, List[Int](), List[Int](), List[Float64](), False)
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    for row in state[].rows:
        if row.binding is binding:
            raise Error("TypeError: captured program connection is already registered")
    state[].rows.append(_OwnedProgramConnection(classes, layouts, binding, provider, executor))
    return PythonObject(None)


def _owned_program_fields(owner: PythonObject) raises -> Tuple[PythonObject, Int]:
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    if len(state[].rows) == 0:
        raise Error("TypeError: captured program providers must be connected during package import")
    ref row = state[].rows[0]
    var cls = py_type(owner)
    var kind = -1
    for i in range(2):
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
    for i in range(len(layout)):
        if py_get(namespace, layout[i][0]) is not layout[i][1]:
            raise Error("TypeError: captured program storage or attribute hook changed after connection")
    var fields = py_fields(owner)
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: captured program requires an exact instance dictionary")
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: captured program fields require exact string keys")
    return (fields, kind)


def owned_program_initialize_py(owner: PythonObject, binding: PythonObject, handle: PythonObject) raises -> PythonObject:
    var admitted = _owned_program_fields(owner)
    var fields = admitted[0]
    if py_type(handle) is not py_type(PythonObject(0)) or Int(py=handle) <= 0:
        raise Error("TypeError: captured program handle requires a positive native integer")
    if py_get(fields, "_lifetime") is not None:
        raise Error("TypeError: captured program owner is already initialized")
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    for row in state[].rows:
        if row.binding is binding:
            var resource = neural_descriptor_adopt(row.provider, Int(py=handle), 1, [admitted[1], row.executor])
            var requests = List[NativeOwnedProgramRequest]()
            requests.append(NativeOwnedProgramRequest(owner, PythonObject("enter"), Python.tuple(), Int(py=handle)))
            _ = neural_descriptor_execute(row.provider, 0, [Int(MutPointer(to=requests[0]))],
                                           [admitted[1], row.executor], List[Float64](), False)
            py_setitem(fields, "_b", binding)
            py_setitem(fields, "_id", handle)
            py_setitem(fields, "_lifetime", resource)
            return PythonObject(None)
    raise Error("TypeError: captured program requires its connected native binding owner")


def owned_program_object_py(cls: PythonObject, binding: PythonObject, handle: PythonObject) raises -> PythonObject:
    var state = _OWNED_PROGRAMS.get_or_create_ptr()
    if len(state[].rows) == 0 or (cls is not state[].rows[0].classes[0] and cls is not state[].rows[0].classes[1]):
        raise Error("TypeError: captured program shell requires its exact connected class")
    var b = native_loaded_module("builtins")
    var owner = py_call(b.object.__new__, cls)
    _ = owned_program_initialize_py(owner, binding, handle)
    return owner


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
