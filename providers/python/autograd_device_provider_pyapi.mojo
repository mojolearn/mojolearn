# SPDX-License-Identifier: Apache-2.0
# One admitted device-program request through the existing whole-model provider.
# Framework capture/registration is separate; this boundary retains the program,
# selected Session owner, descriptors and caller storage through synchronous run.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_fields, py_type, py_get
from bindings.native_loaded_module import native_loaded_module
from providers.python.autograd_program_arrays_pyapi import program_admit_value
from providers.python.autograd_session_frontend_pyapi import ops_session_fields
from providers.python.neural_descriptor_pyapi import NativeDescriptorResource, NeuralDescriptorProvider, neural_descriptor_execute, neural_descriptor_entry
from sequence.neural_model_dispatch import PROGRAM_DEVICE_RUN
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL


def _metadata_sequence(value: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.list and py_type(value) is not b.tuple:
        raise Error("TypeError: program descriptors require native list or tuple metadata")


def _descriptor_kind(dtype: String) raises -> Tuple[Int, Int]:
    if dtype == "float32":
        return (CAST_KIND_FLOAT, 4)
    if dtype == "float64":
        return (CAST_KIND_FLOAT, 8)
    if dtype == "float16":
        return (CAST_KIND_FLOAT, 2)
    if dtype == "int64":
        return (CAST_KIND_SIGNED, 8)
    if dtype == "int32":
        return (CAST_KIND_SIGNED, 4)
    if dtype == "int16":
        return (CAST_KIND_SIGNED, 2)
    if dtype == "int8":
        return (CAST_KIND_SIGNED, 1)
    if dtype == "uint8":
        return (CAST_KIND_UNSIGNED, 1)
    if dtype == "bool":
        return (CAST_KIND_BOOL, 1)
    raise Error("TypeError: unsupported Torch program input dtype " + dtype)


def imported_device_provider_run(program: PythonObject, descriptors: PythonObject,
                                 output_addresses: PythonObject) raises:
    var fields = py_fields(program)
    var session = fields["_s"]
    var session_fields = ops_session_fields(session)
    if session_fields["_id"] is None:
        raise Error("ValueError: this session is closed")
    var b = native_loaded_module("builtins")
    if py_type(session_fields["_id"]) is not b.int or py_type(fields["_id"]) is not b.int:
        raise Error("TypeError: program and Session handles require native integers")
    var lifetime = session_fields["_lifetime"]
    var owner = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    if owner[].handle == 0 or owner[].handle != Int(py=session_fields["_id"]):
        raise Error("TypeError: Session handle differs from its native registry owner")
    var provider = owner[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var binding = session_fields["_b"]
    # _b can be the supported Python module facade. Compare its retained native
    # factory, rather than requiring that facade to be the extension module.
    if py_type(binding) is not py_type(b):
        raise Error("TypeError: Session binding requires exact module storage")
    if py_get(py_fields(binding), "model_native_api_v1") is not provider[].factory:
        raise Error("TypeError: Session binding differs from its native provider")
    if provider[].factory is not neural_descriptor_entry(provider[].module, "model_native_api_v1"):
        raise Error("TypeError: Session provider factory changed; reconnect the compiled binding")
    var inputs = fields["inputs"]
    var outputs = fields["outputs"]
    program_admit_value(inputs)
    program_admit_value(outputs)
    program_admit_value(descriptors)
    program_admit_value(output_addresses)
    _metadata_sequence(inputs)
    _metadata_sequence(outputs)
    _metadata_sequence(descriptors)
    _metadata_sequence(output_addresses)
    if len(inputs) != len(descriptors):
        raise Error("ValueError: the model takes " + String(len(inputs)) + " inputs, got " + String(len(descriptors)))
    if len(outputs) != len(output_addresses):
        raise Error("ValueError: the model returns " + String(len(outputs)) + " outputs, got " + String(len(output_addresses)) + " output buffers")
    var addresses = List[Int]()
    var words = owner[].context.copy()
    words.append(owner[].handle)
    words.append(Int(py=fields["_id"]))
    words.append(len(inputs))
    words.append(len(outputs))
    for k in range(len(inputs)):  # driver-loop: over admitted device input metadata
        var descriptor = descriptors[k]
        _metadata_sequence(descriptor)
        _metadata_sequence(inputs[k])
        if len(descriptor) != 3:
            raise Error("TypeError: device input descriptor requires address, shape and dtype")
        if len(inputs[k]) != 3 or py_type(descriptor[2]) is not b.str:
            raise Error("TypeError: program input metadata requires name, shape and native dtype text")
        _metadata_sequence(descriptor[1])
        _metadata_sequence(inputs[k][1])
        var given = py_call(b.tuple, descriptor[1])
        var wanted = py_call(b.tuple, inputs[k][1])
        if not Bool(py=given == wanted):
            raise Error("ValueError: input " + String(py=py_call(b.repr, inputs[k][0])) + " must have the shape "
                + String(py=py_call(b.repr, wanted)) + ", got " + String(py=py_call(b.repr, given)))
        var kind = _descriptor_kind(String(descriptor[2]))
        if len(given) > 8:
            raise Error("ValueError: program descriptor rank must be at most eight")
        addresses.append(Int(py=descriptor[0]))
        words.append(kind[0])
        words.append(kind[1])
        words.append(len(given))
        for axis in range(len(given)):  # small-loop(given: admitted input rank, at most eight)
            words.append(Int(py=given[axis]))
    for k in range(len(outputs)):  # driver-loop: over admitted device output metadata
        _metadata_sequence(outputs[k])
        if len(outputs[k]) != 3:
            raise Error("TypeError: program output metadata requires name, shape and dtype")
        _metadata_sequence(outputs[k][1])
        var dims = py_call(b.tuple, outputs[k][1])
        if len(dims) > 8:
            raise Error("ValueError: program descriptor rank must be at most eight")
        addresses.append(Int(py=output_addresses[k]))
        words.append(len(dims))
        for axis in range(len(dims)):  # small-loop(dims: admitted output rank, at most eight)
            words.append(Int(py=dims[axis]))
    # Keep the GIL as the original packed device-program door did: the Session
    # registry and tape are not silently given new concurrent mutation semantics.
    _ = neural_descriptor_execute(owner[].provider, PROGRAM_DEVICE_RUN, addresses, words, List[Float64](), False)
    _ = lifetime
    _ = session
    _ = program
    _ = descriptors
    _ = output_addresses
