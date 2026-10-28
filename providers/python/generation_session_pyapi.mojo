# SPDX-License-Identifier: Apache-2.0
# Generation's private provider connection. Only setup uses CPython calls to
# admit the versioned native table and session handle. Repeated execution uses
# native descriptors and the cached C ABI target, not PyObject_Call/vectorcall.
from std.ffi import c_int
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from transformer.native_session_call import generation_session_forward
from bindings.py_balanced import py_call
from transformer.session_abi import GENERATION_SESSION_ABI_NAME, GenerationSessionForward


struct GenerationNativeProvider(Movable, Writable):
    var module: PythonObject
    var capsule: PythonObject
    var factory: PythonObject
    var image: PythonObject
    var admit_session: PythonObject
    var forward: GenerationSessionForward

    def __init__(out self, module: PythonObject, capsule: PythonObject,
                 factory: PythonObject, image: PythonObject, admit_session: PythonObject, forward: GenerationSessionForward):
        self.module = module
        self.capsule = capsule
        self.factory = factory
        self.image = image
        self.admit_session = admit_session
        self.forward = forward

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GenerationNativeProvider(v1)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct GenerationNativeSession(Movable, Writable):
    var provider: PythonObject
    var owner: PythonObject
    var handle: Int

    def __init__(out self, provider: PythonObject, owner: PythonObject, handle: Int):
        self.provider = provider
        self.owner = owner
        self.handle = handle

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GenerationNativeSession(v1)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def generation_connect_native_provider(
    module: PythonObject, factory: PythonObject, admit_session: PythonObject,
) raises -> PythonObject:
    # The caller has checked both setup entries are native functions of its
    # selected transformer module and validated that module's vendor/mode.
    var capsule = py_call(factory)
    var name = String(GENERATION_SESSION_ABI_NAME)
    ref cpy = Python().cpython()
    if cpy.lib.get_function[c_int]("PyCapsule_IsValid")(
        capsule._obj_ptr, name.as_c_string_slice().unsafe_ptr()) == 0:
        raise Error("CausalLM: transformer native ABI capsule is invalid or has a different version")
    var pointer = cpy.lib.get_function[Optional[MutPointer[UInt8, MutUntrackedOrigin]]]("PyCapsule_GetPointer")(
        capsule._obj_ptr, name.as_c_string_slice().unsafe_ptr())
    if not pointer:
        raise cpy.unsafe_get_error()
    # This is our documented native table, not a CPython type/object layout.
    # Copy the sole typed function slot once; retain its capsule and image.
    var table = MutPointer[GenerationSessionForward, MutUntrackedOrigin](unsafe_from_address=Int(pointer.value()))
    return PythonObject(alloc=GenerationNativeProvider(module, capsule, factory, factory.__self__, admit_session, table[0]))


def generation_reuse_native_provider(
    previous: PythonObject, module: PythonObject, factory: PythonObject, admit_session: PythonObject,
) raises -> PythonObject:
    # Admission has already rechecked mode/vendor and both native setup entries.
    # Reuse only the immutable typed table, never block dimensions or buffers.
    if previous is not None:
        var held = previous.downcast_value_ptr[GenerationNativeProvider]()
        if held[].module is module and held[].factory is factory and held[].image is factory.__self__ and held[].admit_session is admit_session:
            return previous
    return generation_connect_native_provider(module, factory, admit_session)


def generation_connect_native_session(provider: PythonObject, session: PythonObject) raises -> PythonObject:
    var api = provider.downcast_value_ptr[GenerationNativeProvider]()
    var handle = Int(py=py_call(api[].admit_session, session))
    if handle == 0:
        raise Error("CausalLM: transformer native ABI returned a null session handle")
    return PythonObject(alloc=GenerationNativeSession(provider, session, handle))


def generation_native_session_matches(link: PythonObject, provider: PythonObject, session: PythonObject) raises -> Bool:
    var held = link.downcast_value_ptr[GenerationNativeSession]()
    return held[].provider is provider and held[].owner is session


def generation_native_forward(link: PythonObject, addresses: List[Int], params: List[Int]) raises -> Int:
    var held = link.downcast_value_ptr[GenerationNativeSession]()
    var provider_owner = held[].provider
    var api = provider_owner.downcast_value_ptr[GenerationNativeProvider]()
    var result = 0
    # This provider owns the Python references and GIL transition. Only native
    # values enter the native transformer helpers; they never import this
    # provider back.
    # Reserve while holding the GIL, exactly as the established Python door
    # does. The second call cannot race another thread's session admission.
    _ = generation_session_forward(0, api[].forward, held[].handle, addresses, params)
    try:
        with GILReleased(Python()):
            result = generation_session_forward(1, api[].forward, held[].handle, addresses, params)
    except problem:
        _ = generation_session_forward(2, api[].forward, held[].handle, addresses, params)
        raise problem
    _ = generation_session_forward(2, api[].forward, held[].handle, addresses, params)
    _ = link
    _ = provider_owner
    return result
