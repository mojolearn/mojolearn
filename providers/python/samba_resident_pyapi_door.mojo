# SPDX-License-Identifier: Apache-2.0
# Resident Samba ownership/admission over the SAME provider used by the other
# neural routes. Python objects end here; all model and optimizer work uses the
# shared native descriptor table and its original native numerical owners.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem
from bindings.causal_lm_public_door import _native_function
from bindings.generation_buffers_door import generation_array
from core.native_descriptor_abi import PRIMITIVE_ABI_NAME
from core.samba_optimizer_plan import SambaOptimizerPlan
# RUN (lane rs-neural): the resident step words, no Python import.
from training.samba_resident_run import samba_resident_step_words
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_adopt, neural_descriptor_resource_handle, neural_descriptor_reserved_execute


def samba_resident_provider(fields: PythonObject, entries: PythonObject) raises -> PythonObject:
    var provider = py_get(fields, "_native_primitives")
    if provider is None:
        return provider
    provider = neural_descriptor_reuse(provider, _native_function(entries, "neural_primitives_api_v1"), String(PRIMITIVE_ABI_NAME), 29)
    py_setitem(fields, "_native_primitives", provider)
    return provider


def samba_resident_session(fields: PythonObject, config: PythonObject, provider: PythonObject) raises -> PythonObject:
    # Lane samba-sessions-native (2026-11-12): the session is always the
    # primitive entry's native owner (op 12), adopted as a descriptor
    # resource (release op 13). The second, Python-object session
    # (`samba_resident_open` + `samba_resident_native_handle_v1`) and the
    # `_resident_native_address` / `_resident_native_resource` words that
    # told the two apart are deleted.
    var session = py_get(fields, "_resident_handle")
    if session is not None:
        var pinned = fields["_resident_native_provider"]
        # A resident resource always belongs to its retained image. Reject
        # image replacement instead of reading its owner in another image.
        if pinned.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
            raise Error("Samba resident session belongs to the prior binding; create a new stack after replacing bindings")
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
    py_setitem(fields, "_resident_native_provider", provider)
    py_setitem(fields, "_resident_handle", session)
    return session


def samba_resident_address(session: PythonObject) raises -> Int:
    return neural_descriptor_resource_handle(session, [12])


# TOMBSTONE: SAMBA_THREE_TIER_FORWARD samba_resident_native_forward (descriptor op 16 forward) deleted 2026-11-09 by lane/x6-samba-on-executor; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/SAMBA_THREE_TIER_FORWARD.patch
# RESTORED 2026-11-13 (lane restore-samba-resident-forward) from
# `git show 5b5490960d:providers/python/samba_resident_pyapi.mojo`, on today's
# session owner (op 12, adopted as a descriptor resource; the address read
# through `samba_resident_address(session)`). The measured winner of
# samba-forward (training/neural_identical_experiments.mojo
# SAMBA_PROGRAM_FORWARD has the numbers); same words, same kernels.


def samba_resident_native_forward(fields: PythonObject, config: PythonObject, entries: PythonObject,
                                  ids: PythonObject) raises -> PythonObject:
    """`SambaStack.forward(inputs)` without state on the resident session:
    `(B, L)` int32 ids -> a fresh `(B, L, vocab)` logits array, or None when
    this stack's primitive image has no resident session (no primitive
    provider, a dropout stack, or an image whose op 24 reports the resident
    route compiled out: CPU, `fast`, MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or
    MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2). UNPACK/PACK only: one reserved native
    call (op 14 reserve, op 16 forward, op 15 release), the GIL released
    around op 16; the session's registry upload count it returns is written
    to `_samba_resident_uploads`."""
    var provider = samba_resident_provider(fields, entries)
    if provider is None:
        return PythonObject(None)
    if Float64(py=config["dropout"]) != 0.0 or neural_descriptor_execute(provider, 24, [], [], [], False) == 0:
        return PythonObject(None)
    var session = samba_resident_session(fields, config, provider)
    var handle = samba_resident_address(session)
    var shape = ids.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var output = generation_array([b, l, Int(py=config["vocab"])])
    var parameters = fields["flat"]
    # Op 16 returns the session's registry upload count (lane
    # fix-samba-weight-upload: the forward uploads only when the weight key is
    # not current; training/samba_resident.mojo samba_resident_forward). PACK
    # records it on the stack, as the route word beside it, so a boundary
    # test can prove an unchanged registry is not uploaded again.
    var uploads = neural_descriptor_reserved_execute(provider, session, handle, 14, 15, 16,
        [Int(py=parameters._addr), Int(py=ids._addr), Int(py=output._addr)], [b, l], [])
    py_setitem(fields, "_samba_resident_uploads", PythonObject(uploads))
    _ = parameters  # keep the Python objects alive through the native call above
    _ = ids
    _ = session
    _ = provider
    return output


def samba_resident_native_step(fields: PythonObject, config: PythonObject, provider: PythonObject,
                               opt: PythonObject, ids: PythonObject, targets: PythonObject,
                               info: PythonObject, loss: PythonObject, plan: SambaOptimizerPlan,
                               optimizer_handle: Int, accumulation: Int, total_count: Int) raises:
    var session = samba_resident_session(fields, config, provider)
    var handle = samba_resident_address(session)
    # Validation remains GIL-held and BEFORE reserving model.busy, exactly as
    # the existing binding. The numerical operation consumes the validated slot.
    var h = neural_descriptor_execute(provider, 19, [], [optimizer_handle, Int(py=fields["n_total"])], [], False)
    var shape = ids.shape
    var p = samba_resident_step_words(h, Int(py=shape[0]), Int(py=shape[1]), accumulation, total_count, plan)
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
