# SPDX-License-Identifier: Apache-2.0
# Resident Samba ownership/admission over the SAME provider used by the other
# neural routes. Python objects end here; all model and optimizer work uses the
# shared native descriptor table and its original native numerical owners.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem
from bindings.causal_lm_public import _native_function
from bindings.generation_buffers import generation_array
from core.native_descriptor_abi import PRIMITIVE_ABI_NAME
from core.samba_optimizer_plan import SambaOptimizerPlan
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


# TOMBSTONE: SAMBA_THREE_TIER_FORWARD samba_resident_native_forward (descriptor op 16 forward) deleted 2026-11-09 by lane/x6-samba-on-executor; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/SAMBA_THREE_TIER_FORWARD.patch


def samba_resident_native_step(fields: PythonObject, config: PythonObject, provider: PythonObject,
                               opt: PythonObject, ids: PythonObject, targets: PythonObject,
                               info: PythonObject, loss: PythonObject, plan: SambaOptimizerPlan,
                               optimizer_handle: Int, accumulation: Int, total_count: Int) raises:
    var session = samba_resident_session(fields, config, provider)
    var handle = samba_resident_address(fields, session)
    # Validation remains GIL-held and BEFORE reserving model.busy, exactly as
    # the existing binding. The numerical operation consumes the validated slot.
    var h = neural_descriptor_execute(provider, 19, [], [optimizer_handle, Int(py=fields["n_total"])], [], False)
    var shape = ids.shape
    var p: List[Int] = [h, Int(py=shape[0]), Int(py=shape[1])]
    if accumulation > 1:
        p.append(accumulation)
        p.append(total_count)
    p.append(plan.tensors)
    p.append(plan.hyper.kind)
    p.append(plan.step)
    p.append(plan.hyper.nesterov)
    var f = plan.floats()
    if plan.hyper.maximize:
        raise Error("samba_resident_step: maximize is not carried by the resident step (SambaStack never sets it)")
    var parameters = fields["flat"]
    var offsets = opt["offsets"]
    var initialized = opt["buf_initialized"]
    _ = neural_descriptor_reserved_execute(provider, session, handle, 14, 15, 18 if accumulation > 1 else 17,
        [Int(py=parameters._addr), Int(py=ids._addr), Int(py=targets._addr), Int(py=offsets._addr),
         Int(py=initialized._addr), Int(py=info._addr), Int(py=loss._addr)], p, f)
    _ = parameters
    _ = offsets
    _ = initialized
