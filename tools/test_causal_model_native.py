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
