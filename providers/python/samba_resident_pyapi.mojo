# SPDX-License-Identifier: Apache-2.0
# Resident Samba ownership/admission over the SAME provider used by the other
# neural routes. Python objects end here; all model and optimizer work uses the
# shared native descriptor table and its original native numerical owners.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem
from bindings.causal_lm_public import _native_function
from bindings.generation_buffers import generation_array
from core.native_descriptor_abi import PRIMITIVE_ABI_NAME
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_handle, neural_descriptor_adopt, neural_descriptor_resource_handle, neural_descriptor_reserved_execute


def samba_resident_provider(fields: PythonObject, entries: PythonObject) raises -> PythonObject:
    var provider = py_get(fields, "_native_primitives")
    if provider is None:
        return provider
    provider = neural_descriptor_reuse(provider, _native_function(entries, "neural_primitives_api_v1"), String(PRIMITIVE_ABI_NAME), 29)
    py_setitem(fields, "_native_primitives", provider)
    return provider


def samba_resident_session(fields: PythonObject, config: PythonObject, provider: PythonObject) raises -> PythonObject:
    var session = py_get(fields, "_resident_handle")
    var pinned = py_get(fields, "_resident_native_provider")
    if session is not None:
        if pinned is not None:
            # A resident resource always belongs to its retained image. Reject
            # image replacement instead of reading its owner in another image.
            if pinned.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
                raise Error("Samba resident session belongs to the prior binding; create a new stack after replacing bindings")
            return session
        var module = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
        var handle = neural_descriptor_handle(provider, _native_function(module, "samba_resident_native_handle_v1"), session)
        py_setitem(fields, "_resident_native_address", PythonObject(handle))
        py_setitem(fields, "_resident_native_provider", provider)
        py_setitem(fields, "_resident_native_resource", False)
        return session
    var p: List[Int] = [Int(py=config["vocab"]), Int(py=config["d_model"])]
    for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # small-loop(resident model dimensions)
        p.append(0 if config[name] is None else Int(py=config[name]))
    p.append(Int(Bool(py=config["tie_embeddings"])))
    p.append(Int(py=fields["n_total"]))
    var kinds = config["layers"]
    for i in range(len(kinds)):  # small-loop(resident model layer kinds)
        p.append(0 if String(kinds[i]) == "mamba3" else 1)
    # The old resident constructor holds the GIL. Preserve that reservation and
    # allocation ordering; only forward/step use the common released-GIL door.
    var handle = neural_descriptor_execute(provider, 12, [Int(py=fields["flat"]._addr)], p,
        [Float64(py=config["norm_eps"])], False)
    session = neural_descriptor_adopt(provider, handle, 13, [12])
    py_setitem(fields, "_resident_native_address", PythonObject(handle))
    py_setitem(fields, "_resident_native_provider", provider)
    py_setitem(fields, "_resident_native_resource", True)
    py_setitem(fields, "_resident_handle", session)
    return session


def samba_resident_address(fields: PythonObject, session: PythonObject) raises -> Int:
    if Bool(py=fields["_resident_native_resource"]):
        return neural_descriptor_resource_handle(session, [12])
    return Int(py=fields["_resident_native_address"])


def samba_resident_native_forward(fields: PythonObject, config: PythonObject, entries: PythonObject, ids: PythonObject) raises -> PythonObject:
    var provider = samba_resident_provider(fields, entries)
    if provider is None:
        return PythonObject(None)
    if Float64(py=config["dropout"]) != 0.0 or neural_descriptor_execute(provider, 24, [], [], [], False) == 0:
        return PythonObject(None)
    var session = samba_resident_session(fields, config, provider)
    var handle = samba_resident_address(fields, session)
    var shape = ids.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var output = generation_array([b, l, Int(py=config["vocab"])])
    _ = neural_descriptor_reserved_execute(provider, session, handle, 14, 15, 16,
        [Int(py=fields["flat"]._addr), Int(py=ids._addr), Int(py=output._addr)], [b, l], [])
    return output


def samba_resident_native_step(fields: PythonObject, config: PythonObject, provider: PythonObject,
                               opt: PythonObject, ids: PythonObject, targets: PythonObject,
                               info: PythonObject, loss: PythonObject, plan: PythonObject,
                               accumulation: Int, total_count: Int) raises:
    var session = samba_resident_session(fields, config, provider)
    var handle = samba_resident_address(fields, session)
    # Validation remains GIL-held and BEFORE reserving model.busy, exactly as
    # the existing binding. The numerical operation consumes the validated slot.
    var h = neural_descriptor_execute(provider, 19, [], [Int(py=opt["_res"]), Int(py=fields["n_total"])], [], False)
    var shape = ids.shape
    var p: List[Int] = [h, Int(py=shape[0]), Int(py=shape[1])]
    if accumulation > 1:
        p.append(accumulation)
        p.append(total_count)
    for i in range(4):  # small-loop(optimizer tensor count/kind/step/nesterov)
        p.append(Int(py=plan[i]))
    var f = List[Float64]()
    for i in range(4, 12):  # small-loop(optimizer eight floating settings)
        f.append(Float64(py=plan[i]))
    if len(plan) > 12 and Int(py=plan[12]) != 0:
        raise Error("samba_resident_step: maximize is not carried by the resident step (SambaStack never sets it)")
    _ = neural_descriptor_reserved_execute(provider, session, handle, 14, 15, 18 if accumulation > 1 else 17,
        [Int(py=fields["flat"]._addr), Int(py=ids._addr), Int(py=targets._addr), Int(py=opt["offsets"]._addr),
         Int(py=opt["buf_initialized"]._addr), Int(py=info._addr), Int(py=loss._addr)], p, f)


def samba_optimizer_execute(provider: PythonObject, operation: Int, addresses: List[Int],
                            plan: PythonObject, handle: Int, io: Int) raises -> Int:
    var p = List[Int]()
    if operation != 25:
        p.append(handle)
    if operation == 27:
        p.append(io)
    for i in range(4):  # small-loop(optimizer integer settings)
        p.append(Int(py=plan[i]))
    p.append(Int(py=plan[12]) if len(plan) == 13 else 0)
    var f = List[Float64]()
    for i in range(4, 12):  # driver-loop: over eight optimizer floating settings
        f.append(Float64(py=plan[i]))
    return neural_descriptor_execute(provider, operation, addresses, p, f)
