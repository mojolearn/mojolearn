"""Source boundary invariants; public behavior is tested against built artifacts."""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_state_restore_has_one_native_frontdoor():
    tree = ast.parse((ROOT / 'python/mojolearn/_byte_lm_impl.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmallByteLanguageModelTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'load_state_dict')
    body = method.body[1:]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert isinstance(body[0].value, ast.Call) and body[0].value.func.id == '_LM_TRAINER'


def test_validation_precedes_session_mutation_and_uses_shared_admission():
    source = (ROOT / 'bindings/byte_lm_trainer_native_door.mojo').read_text()
    restore = source.split('def byte_lm_load_state(')[1].split('def byte_lm_train_step(')[0]
    assert restore.index('byte_state_unpack(') < restore.index('_close_forget(') < restore.index('byte_trainer_replace(')
    assert 'byte_trainer_session_close(s)' in source
    admission = source.split('def byte_state_unpack(')[1].split('def byte_lm_load_state(')[0]
    assert admission.index('classical_admit_value(source, context)') < admission.index('_shape_of(source)')
    assert admission.count('training_words32_into(') == 4
    assert 'training_admit_moments_and_flags' in admission
    assert 'py_call_typed' not in admission


def test_buffer_copy_and_state_value_checks_are_native():
    for name in ('core/training_input_copy.mojo', 'core/training_state_validation.mojo'):
        source = (ROOT / name).read_text()
        for forbidden in ('PythonObject', 'std.python', 'bindings.', 'providers.'):
            assert forbidden not in source
    boundary = (ROOT / 'providers/python/training_buffer_pyapi_door.mojo').read_text()
    strict = boundary.split('def training_words32_into(')[1].split('# TOMBSTONE: training_f32_cast_owned')[0]
    assert strict.count('var shape = value.shape') == 1
    assert strict.count('var strides = value.strides') == 1
    assert 'training_copy_words32(' in boundary


def test_binary_checkpoint_public_methods_have_one_native_frontdoor():
    tree = ast.parse((ROOT / 'python/mojolearn/_byte_lm_impl.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmallByteLanguageModelTrainer')
    for name in ('save_checkpoint', 'export_checkpoint', 'from_checkpoint', 'from_checkpoint_bytes',
                 'export_checkpoint_binary', 'from_checkpoint_binary'):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        body = method.body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        assert body[0].value.func.id == '_LM_TRAINER'
    source = (ROOT / 'bindings/byte_lm_trainer_native_door.mojo').read_text()
    checkpoint = source.split('def byte_lm_checkpoint(')[1].split('def byte_lm_train_step(')[0]
    assert 'byte_checkpoint_load(args[0], io_error)' in checkpoint
    assert 'byte_checkpoint_save(args[0], state, io_error)' in checkpoint
    assert checkpoint.index('byte_state_unpack(mods, source, context, tier)') < checkpoint.index('native_storage_allocate(trainer)')
    assert 'return io_error^.restore()' in source
    assert '_initialize_fields(mods, restored, data^,' in checkpoint
    assert 'py_call(trainer' not in checkpoint


def test_attention_report_uses_one_native_status_snapshot():
    source = (ROOT / 'bindings/byte_lm_trainer_native_door.mojo').read_text()
    report = source.split('def _attention_stage_report(')[1].split('def byte_lm_consume(')[0]
    assert 'byte_trainer_stage_info(s)' in report
    run = (ROOT / 'bindings/byte_lm_trainer_native_run.mojo').read_text()
    assert '_session_run(s, 11, a, p,' in run and 'var info = List[Int]()' in run
    owner = (ROOT / 'training/byte_lm_native_status.mojo').read_text()
    assert owner.count('byte_attention_eager_cells(owner.trainer.value())') == 1
    assert 'PythonObject' not in owner and 'std.python' not in owner
    binding = (ROOT / 'bindings/_mojolearn_byte_lm.mojo').read_text()
    legacy = binding.split('def byte_lm_session_info_binding(')[1].split('def _byte_config(')[0]
    assert 'byte_lm_status(owner[])' in legacy
    tree = ast.parse((ROOT / 'python/mojolearn/_byte_lm_impl.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmallByteLanguageModelTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'attention_stage_report')
    assert len(method.body) == 2 and method.body[1].value.func.id == '_LM_TRAINER'


def test_json_checkpoint_retains_bounded_format_and_native_codec():
    source = (ROOT / 'providers/python/byte_lm_json_checkpoint_pyapi_door.mojo').read_text()
    assert 'BYTE_JSON_LIMIT = 2097152' in source
    assert 'native_json_parse_unique(encoded)' in source
    assert 'sha256_hexdigest(_canonical(payload))' in source
    assert 'py_type(encoded) is not bi.bytes' in source
    # The file calls are typed (no Python file object, no `os` call).
    assert 'byte_file_write(target, address, total, file_failure)' in source
    assert 'byte_file_read(target, Int(data.unsafe_ptr()), BYTE_JSON_LIMIT + 1, file_failure)' in source
    assert 'byte_checkpoint_io(' not in source and 'c[].os' not in source
    assert 'generation_array([cells]' in source
    assert 'py_call(context[' not in source
