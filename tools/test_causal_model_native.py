"""Causal model source controls; compilation and runtime checks are separate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "bindings/causal_lm_public.mojo").read_text()


def test_compiled_entry_resolution_admits_raw_storage_before_keyed_reads():
    fields = PUBLIC.split("def _native_entry_fields", 1)[1].split("def _native_has_entry", 1)[0]
    assert 'py_type(entries) is types.ModuleType' in fields
    assert 'py_type(entries) is types.SimpleNamespace' in fields
    assert '"PyModule_GetDict"' in fields and '"PyObject_GenericGetDict"' in fields
    assert 'py_type(keys[i]) is not bi.str' in fields
    function = PUBLIC.split("def _native_function", 1)[1].split("def causal_lm_admit_binding", 1)[0]
    assert 'py_get(_native_entry_fields(module), name)' in function
    assert 'getattr' not in function and '.getattr' not in fields
    optional = PUBLIC.split("def _native_has_entry", 1)[1].split("def _native_function", 1)[0]
    assert 'py_contains(_native_entry_fields(entries), name)' in optional


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
    assert state.index("_causal_state_fields(state)") < state.index("py_call(bi.int, values[0])")
    assert state.index("neural_native_scalar(values[i], context)") < state.index("py_call(bi.int, values[0])")
    assert 'py_type(values[2]) is not bi.list' in state
    assert 'py_setattr' not in state
    classification = PUBLIC.split("def _causal_owner_unusable", 1)[1].split("def causal_public_c", 1)[0]
    assert 'neural_record_fields(args[0]' in classification
    assert 'py_call(' not in classification and '.getattr' not in classification
    integer = PUBLIC.split("def _public_int", 1)[1].split("def causal_lm_allocate_binding", 1)[0]
    assert 'neural_native_scalar(value, _causal_context(), True)' in integer
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
    for guard in ("_causal_model_fields(model)", "model_plan_snapshot(input_plan)", "_model_options_snapshot(input_options, context)", "_model_weights_snapshot(input_weights, context)"):
        assert construct.index(guard) < construct.index("_causal_model_prepare_snapshot(")
    block = source.split("def _model_block(", 1)[1].split("def causal_model_prepare", 1)[0]
    assert '_model_block_plan(input_plan, context)' in block
    assert 'neural_block_fields(block, context, False)' in block
    assert 'neural_block_weight_names(block)' in block
    assert 'plan.' not in block and 'cls._W_NAMES' not in block
    assert 'object.__getattribute__' not in source
    assert 'context["backend"]._DEFAULT_MODE' not in source
    read = source.split("def causal_model_read_weights_binding", 1)[1].split("def causal_model_load_binding", 1)[0]
    assert 'model_checkpoint_names_snapshot(plan)' in read and '_model_layer_weights(plan, i)' in read
    assert 'infos[name].' not in read and '_info_snapshot(infos[info_names[i]])' in read


def test_packed_weights_admit_metadata_and_keep_original_quantization_routes():
    source = (ROOT / "bindings/causal_lm_construct.mojo").read_text()
    unpack = source.split("def _model_unpack", 1)[1].split("def _model_quantize15", 1)[0]
    assert '_model_packed_fields(value, context)' in unpack
    assert 'value.shape' not in unpack and 'value.bits' not in unpack and 'value.codes' not in unpack
    packed = source.split("def _model_native_buffer", 1)[1].split("def _model_pack(", 1)[0]
    assert 'neural_array_snapshot(value, context)' in packed
    assert packed.index('admit_native_buffer_exporter(value)') < packed.index('AdmittedBorrowedBuffer(value)')
    assert 'layout.copy_to(Int(py=result._addr))' in packed
    for numerical in ('to_bf16', 'from_bf16', 'quantize_int8', 'dequantize_int8', 'quantize_int15'):
        assert f'_native_function(linalg, "{numerical}")' in source
