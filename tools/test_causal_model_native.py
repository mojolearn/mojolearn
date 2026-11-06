"""Causal model source controls; compilation and runtime checks are separate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "bindings/causal_lm_public.mojo").read_text()


def test_compiled_entry_resolution_admits_raw_storage_before_keyed_reads():
    fields = PUBLIC.split("def _native_entry_fields", 1)[1].split("def _native_has_entry", 1)[0]
    assert 'py_type(entries) is types.ModuleType' in fields
    assert 'py_type(entries) is types.SimpleNamespace' in fields
    # CPython calls use the std handle's pointers and the process-wide typed
    # table (bindings/cpython_table.mojo), never a per-call by-name lookup.
    assert 'cpy.PyModule_GetDict(entries._obj_ptr)' in fields
    assert 'cpython_table()[].PyObject_GenericGetDict(entries._obj_ptr, PyObjectPtr())' in fields
    assert 'get_function' not in PUBLIC
    assert 'from bindings.cpython_table import cpython_table' in PUBLIC
    assert 'py_type(keys[i]) is not bi.str' in fields
    function = PUBLIC.split("def _native_function", 1)[1].split("def causal_lm_admit_binding", 1)[0]
    assert 'py_get(_native_entry_fields(mods, module), name)' in function
    assert 'getattr' not in function and '.getattr' not in fields
    optional = PUBLIC.split("def _native_has_entry", 1)[1].split("def _native_function", 1)[0]
    assert 'py_contains(_native_entry_fields(mods, entries), name)' in optional


def test_model_wrappers_never_select_binding_through_mutable_instance_fields():
    import ast
    tree = ast.parse((ROOT / "python/mojolearn/models/causal_lm.py").read_text())
    expected = {
        "CausalLM": ("forward", "allocate_state", "reset_state", "step", "generate", "stream", "generate_text", "stream_text", "resume_generation", "parameters", "blocks", "__repr__"),
        "CausalLMState": ("__init__", "__repr__"),
        "GenerationStream": ("read", "info", "state_dict"),
    }
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in expected:
            continue
        methods = {item.name: item for item in cls.body if isinstance(item, ast.FunctionDef)}
        for name in expected[cls.name]:
            method = methods[name]
            body = method.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            assert len(body) == 1 and isinstance(body[0], ast.Return), (cls.name, name)
            assert isinstance(body[0].value, ast.Call), (cls.name, name)
            assert ast.unparse(body[0].value.func).startswith("_NATIVE.causal_"), (cls.name, name)


def test_state_construction_and_error_classification_admit_before_protocols():
    state = PUBLIC.split("def causal_state_init_binding", 1)[1].split("def causal_state_repr_binding", 1)[0]
    assert state.index("_causal_state_fields(mods, state)") < state.index("py_call(bi.int, values[0])")
    assert state.index("neural_native_scalar(mods, values[i], context)") < state.index("py_call(bi.int, values[0])")
    assert 'py_type(values[2]) is not bi.list' in state
    assert 'py_setattr' not in state
    classification = PUBLIC.split("def _causal_owner_unusable", 1)[1].split("def causal_public_c", 1)[0]
    assert 'neural_record_fields(mods, args[0]' in classification
    assert 'py_call(' not in classification and '.getattr' not in classification
    integer = PUBLIC.split("def _public_int", 1)[1].split("def causal_lm_allocate_binding", 1)[0]
    assert 'neural_native_scalar(mods, value, _causal_context(), True)' in integer
    assert 'operator' not in integer


def test_model_modules_use_loaded_connections_and_safe_optional_entries():
    for path in ("bindings/causal_lm_public.mojo", "bindings/causal_lm_construct.mojo"):
        source = (ROOT / path).read_text()
        assert 'Python.import_module(' not in source
        assert '.hasattr' not in source
        assert '_native_has_entry(' in source


def test_construct_captures_plan_options_and_all_weights_before_numerical_work():
    source = (ROOT / "bindings/causal_lm_construct.mojo").read_text()
    construct = source.split("def causal_model_init_binding", 1)[1].split("def causal_model_read_weights_binding", 1)[0]
    for guard in ("_causal_model_fields(mods, model)", "model_plan_snapshot(mods, input_plan)", "_model_options_snapshot(mods, input_options, context)", "_model_weights_snapshot(mods, input_weights, context)"):
        assert construct.index(guard) < construct.index("_causal_model_prepare_snapshot(")
    block = source.split("def _model_block(", 1)[1].split("def causal_model_prepare", 1)[0]
    assert '_model_block_plan(mods, input_plan, context)' in block
    assert 'neural_block_fields(mods, block, context, False)' in block
    assert 'neural_block_weight_names(mods, block)' in block
    assert 'plan.' not in block and 'cls._W_NAMES' not in block
    assert 'object.__getattribute__' not in source
    assert 'context["backend"]._DEFAULT_MODE' not in source
    read = source.split("def causal_model_read_weights_binding", 1)[1].split("def causal_model_load_binding", 1)[0]
    assert 'model_checkpoint_names_snapshot(mods, plan)' in read and '_model_layer_weights(mods, plan, i)' in read
    assert 'infos[name].' not in read and '_info_snapshot(mods, infos[info_names[i]])' in read


def test_packed_weights_admit_metadata_and_keep_original_quantization_routes():
    source = (ROOT / "bindings/causal_lm_construct.mojo").read_text()
    unpack = source.split("def _model_unpack", 1)[1].split("def _model_quantize15", 1)[0]
    assert '_model_packed_fields(mods, value, context)' in unpack
    assert 'value.shape' not in unpack and 'value.bits' not in unpack and 'value.codes' not in unpack
    packed = source.split("def _model_native_buffer", 1)[1].split("def _model_pack(", 1)[0]
    assert 'neural_array_snapshot(value, context)' in packed
    assert packed.index('admit_native_buffer_exporter(value)') < packed.index('AdmittedBorrowedBuffer(value)')
    assert 'layout.copy_to(Int(py=result._addr))' in packed
    for numerical in ('to_bf16', 'from_bf16', 'quantize_int8', 'dequantize_int8', 'quantize_int15'):
        assert f'_native_function(mods, linalg, "{numerical}")' in source


def test_execution_registry_is_pinned_and_native_handles_do_not_reread_bindings():
    owner = (ROOT / "bindings/causal_operation_owner.mojo").read_text()
    registry = owner.split("def _causal_entries", 1)[1].split("def causal_model_capture", 1)[0]
    assert 'downcast_value_ptr[CausalExecutionEntries]' in registry
    assert 'item is not original[names[i]]' in registry
    assert 'entries["parallel_call"] is not connected["parallel_causal_call"]' in registry
    assert 'causal_pin_entries(mods, _standalone_record_fields(mods, model), entries)' in PUBLIC
    run = PUBLIC.split("def _causal_layer_run_prepared", 1)[1].split("def causal_model_embedding", 1)[0]
    assert 'e["transformer_native_session_handle_v1"]' in run
    assert '_native_function(e["block_binding"]' not in run


def test_operation_owner_preserves_aliases_tied_weights_and_original_parameters():
    owner = (ROOT / "bindings/causal_operation_owner.mojo").read_text()
    # The full capture moved behind the prepared-capture reuse (053c27e4f2).
    model = owner.split("def _causal_model_capture_full", 1)[1].split("\ndef ", 1)[0]
    assert 'identity = py_call(bi.id, original)' in model
    assert 'captured = py_get(buffers, identity)' in model
    assert '"_causal_entries_identity", fields["_native_entries"]' in model
    assert '"_causal_original_parameters", original_parameters' in model
    state = owner.split("def causal_state_capture", 1)[1].split("def causal_layer_publish", 1)[0]
    assert 'identity = Int(layers[i]._obj_ptr)' in state and 'layer = seen.get(identity, PythonObject(None))' in state
    assert '"_native_state_identity", state' in state
    assert '"_causal_admission_failure"' in state
    params = PUBLIC.split("def causal_lm_parameters_binding", 1)[1].split("def causal_lm_blocks_binding", 1)[0]
    assert 'py_setitem(result, name, originals[key])' in params
    assert '"_neural_original_w"' in params and '"_neural_original_wopt"' in params
    assert 'pieces[k][2], keep' in params  # fused outputs consume captured bytes


def test_success_and_failure_publish_only_admitted_live_fields():
    owner = (ROOT / "bindings/causal_operation_owner.mojo").read_text()
    state = owner.split("def causal_state_publish", 1)[1].split("def causal_model_cache_set", 1)[0]
    assert state.index('neural_string_keys(mods, live)') < state.index('py_setitem(live, name')
    assert 'if identity in seen:' in state
    layer = owner.split("def causal_layer_publish", 1)[1].split("def causal_state_publish", 1)[0]
    assert layer.index('neural_string_keys(mods, target)') < layer.index('py_setitem(target, name')
    assert 'layer[name] is initial[name]' in layer  # do not replace original mutable exporters
    door = PUBLIC.split("def _causal_public_execute[", 1)[1].split("def causal_public_c[", 1)[0]
    assert door.count('causal_state_publish(mods, state)') == 2
    assert door.count('causal_model_publish(mods, model)') == 2
    assert 'native state publication also failed:' in door
    model = owner.split("def causal_model_publish", 1)[1].split("def causal_layer_capture", 1)[0]
    # Publication admits live fields but never replays a captured snapshot
    # (f5a41c515f: replay restored busy=True after overlapping reads).
    assert 'neural_string_keys(mods, original)' in model and 'py_setitem(' not in model


def test_parallel_paths_pass_identity_and_distinct_captured_payloads():
    owner = (ROOT / "bindings/causal_operation_owner.mojo").read_text()
    assert '"capture_model", Python.list(model, fields)' in owner
    assert '"capture_state", Python.list(layers[i], model)' in owner
    assert '"run", Python.list(causal_model_origin(mods, model), ids, causal_state_origin(mods, state), remote_step, model, state)' in PUBLIC
    assert '"release_layers", Python.list(causal_model_origin(mods, model), state, model, captured)' in PUBLIC
    assert 'causal_layer_publish(mods, dest)' in PUBLIC
    assert '"state_publish", Python.list(causal_state_origin(mods, s[].state.layers[i]), causal_state_origin(mods, dest), s[].state.layers[i])' in PUBLIC


def test_streams_and_checkpoints_capture_before_execution_and_guard_output_metadata():
    owner = (ROOT / "bindings/causal_operation_owner.mojo").read_text()
    door = PUBLIC.split("def _causal_public_execute[", 1)[1].split("def causal_public_c[", 1)[0]
    assert door.index('causal_model_capture(mods, args[0]') < door.index('causal_lm_forward_binding(mods, model')
    assert door.index('causal_checkpoint_capture(mods, args[1]') < door.index('causal_lm_resume_binding(mods, model')
    assert 'causal_model_capture(mods, causal_model_origin(mods, stream[].model)' in door
    assert 'causal_state_capture(mods, causal_state_origin(mods, stream[].state), model' in door
    assert 'causal_lm_read_binding(mods, args[0], args[1], stream_native)' in door
    assert 'state = stream[].state' in door
    assert 'causal_lm_resident_binding(mods, model, s.prompt, PythonObject(s.max_new)' in PUBLIC
    assert 'generation_array_snapshot(args[3])' in door
    read = PUBLIC.split("def _read(mut", 1)[1].split("def _decode_chunk", 1)[0]
    run = read.index('_run_prepared(')
    # One post-release re-check per forward, against the call's admitted snapshot.
    guard = read.index('admission.value().recheck(generation_array_context())', run)
    assert guard < read.index('logits._addr', run)
    assert read.index('admission = NativeArrayAdmission(generation_array_context())') < run
    checkpoint = owner.split('def causal_checkpoint_capture', 1)[1]
    assert checkpoint.index('neural_string_keys(mods, checkpoint)') < checkpoint.index('checkpoint.keys()')
    assert 'generation_array_snapshot(item)' in checkpoint
