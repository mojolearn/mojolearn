"""Source invariants; runtime qualification is the separately retained public probe."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_small_mlp_uses_shared_provider_and_native_owners():
    src=(ROOT/'bindings/small_mlp_native.mojo').read_text()
    assert 'neural_descriptor_reuse' in src and 'neural_descriptor_execute' in src
    assert 'struct SmallMlpProvider' not in src
    operation=src.split('def _mode_entries(')[1].split('def _mode_name(')[0]
    assert 'entries["small_mlp_api_v1"]' in operation
    assert 'MlpOperationMetadata(entries, provider, String(py=mode))' in operation
    batch=src.split('def _batch_in(')[1].split('def _owned_rows(')[0]
    assert batch.count('= _view(')==1 and batch.count('= _shape(')==1
    assert '_words_from_view(value, view, shape,' in batch
    table=(ROOT/'training/small_mlp_dispatch.mojo').read_text()
    for forbidden in ('PythonObject', 'std.python', 'bindings.', 'providers.'):
        assert forbidden not in table
    for owner in ('mlp_train_step_host(', 'neural_gemm_native(', 'mlp_bias_activation_host(',
                  'mlp_relu_backward_host(', 'mlp_sum_rows_host(', 'identical_ce_loss_host(', 'per_call_step('):
        assert owner in table


def test_neural_gemm_both_doors_share_native_transport():
    door=(ROOT/'bindings/neural_gemm_boundary.mojo').read_text()
    assert 'neural_gemm_native(aa, ba, ca, shape)' in door
    assert 'identical_gemm_into(' not in door
    owner=(ROOT/'training/neural_gemm_native.mojo').read_text()
    assert owner.count('identical_gemm_into(')==1
    assert 'gemm_up_f32(' in owner and 'gemm_down_f32(' in owner
    assert 'ctx.synchronize()' in owner and 'lease.release()' in owner


def test_small_mlp_rejects_hooks_before_admission():
    src=(ROOT/'bindings/small_mlp_native.mojo').read_text().split('def small_mlp_call(')[1]
    assert src.index('sequential_admit_value(args,') < src.index('return small_mlp_create(args)')


def test_byte_lm_frontdoors_and_native_state_share_provider():
    import ast
    source=(ROOT/'python/mojolearn/_byte_lm_impl.py').read_text()
    cls=next(node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=='SmallByteLanguageModelTrainer')
    for name in ('logits','next_bytes','train_step'):
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
        body=method.body[1:] if isinstance(method.body[0],ast.Expr) and isinstance(method.body[0].value,ast.Constant) else method.body
        assert len(body)==1 and isinstance(body[0],ast.Return)
        assert isinstance(body[0].value.func,ast.Name) and body[0].value.func.id=='_LM_TRAINER'
    assert 'MethodType(module.byte_lm_pooled_finalize_call, context)' in source
    door=(ROOT/'bindings/byte_lm_trainer_native.mojo').read_text()
    mode=door.split('def _mode(')[1].split('def _has(')[0]
    assert 'backend["_DEFAULT_MODE"]' in mode
    assert 'py_call_typed' not in mode
    assert 'neural_descriptor_handle(provider, factory, session)' in door
    assert 'neural_descriptor_reuse' in door
    table=(ROOT/'training/byte_lm_native_dispatch.mojo').read_text()
    for forbidden in ('PythonObject','std.python','bindings.','providers.'):
        assert forbidden not in table
    for owner in ('byte_owner_step(', 'byte_owner_logits(', 'byte_owner_export_state(', 'byte_lm_stateless_logits_native('):
        assert owner in table


def test_byte_lm_input_admission_precedes_buffer_export():
    door=(ROOT/'bindings/byte_lm_trainer_native.mojo').read_text().split('def _ids(')[1].split('def _f32(')[0]
    assert door.index('classical_admit_value(source, context)') < door.index('native_buffer_view(source)')


def test_training_constructors_share_admission_helpers():
    import ast
    source=(ROOT/'python/mojolearn/_byte_lm_impl.py').read_text()
    cls=next(node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=='SmallByteLanguageModelTrainer')
    constructor=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
    assert len(constructor.body)==1
    assert constructor.body[0].value.func.id=='_LM_TRAINER'
    assert 'byte_lm_lock=type(threading.RLock())' in source
    byte=(ROOT/'bindings/byte_lm_trainer_native.mojo').read_text()
    small=(ROOT/'bindings/small_mlp_native.mojo').read_text()
    for src in (byte,small):
        assert 'training_f32_scalar(' in src
        assert 'training_schedule_json(' in src
    construct=byte.split('def byte_lm_construct(')[1].split('def byte_lm_logits(')[0]
    assert construct.index('_load_snapshot(shape, context)') < construct.index('training_f32_owned(')
    assert 'training_f32_owned(' in construct
    copy=(ROOT/'core/training_input_copy.mojo').read_text()
    assert 'std.python' not in copy and 'PythonObject' not in copy


def test_linalg_gemm_uses_shared_native_transport():
    import ast
    source=(ROOT/'python/mojolearn/_linalg_impl.py').read_text()
    funcs={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    for name in ('matmul','matmul_bf16','matmul_int8','to_bf16','from_bf16','quantize_int8','dequantize_int8'):
        calls=[n for n in ast.walk(funcs[name]) if isinstance(n,ast.Call)]
        assert not any(isinstance(n.func,ast.Name) and n.func.id=='_load' for n in calls)
    door=(ROOT/'bindings/linalg_ops_lowbit.mojo').read_text()
    assert 'neural_descriptor_execute(provider, op, addresses, parameters' in door
    assert 'neural_descriptor_reuse' in door
    assert 'family_call(bf["default_mode"])' not in door
    assert '.environ' not in door
    native=(ROOT/'gemm/linalg_native.mojo').read_text()
    assert 'std.python' not in native and 'PythonObject' not in native
    for owner in ('identical_gemm_host(', 'identical_gemm_int8_into(', 'identical_gemm_bf16w_into('):
        assert native.count(owner)==(2 if owner=='identical_gemm_bf16w_into(' else 1)
    binding=(ROOT/'bindings/_mojolearn_linalg.mojo').read_text()
    assert 'result = gemm_native(' in binding
    assert 'result = gemm_bf16_native(' in binding
    assert 'result = gemm_int8_native(' in binding
