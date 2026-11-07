"""Source invariants of the byte LM multi-step session entry (plan section 2C)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    return (ROOT / name).read_text()


def test_train_steps_has_one_native_frontdoor():
    tree = ast.parse(_read('python/mojolearn/_byte_lm_impl.py'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmallByteLanguageModelTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'train_steps')
    body = method.body[1:]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert body[0].value.func.id == '_LM_TRAINER'
    assert body[0].value.args[1].value == 'train_steps'


def test_both_columns_expose_operation_12_from_the_admitted_session():
    device = _read('training/byte_lm_native_dispatch.mojo')
    assert 'BYTE_LM_NATIVE_OPERATION_COUNT = 13' in device
    assert 'byte_owner_steps(owner[], step_addresses, p[0], p[1])' in device
    host = _read('training/byte_lm_host_trainer_dispatch.mojo')
    assert 'BYTE_HOST_TRAINER_COUNT = 13' in host
    assert 'owner[].steps([a[1], a[2], a[3]], p[0], p[1])' in host
    ops = _read('training/byte_lm_session_ops.mojo')
    steps = ops.split('def byte_owner_steps(')[1].split('def byte_owner_export_state(')[0]
    # The same per-step path as train_step, from the session's own config.
    assert 'byte_session_step_run(' in steps
    assert '.config.copy()' in steps and '.optimizer.copy()' in steps
    assert steps.index('byte_validate_tokens(ids') < steps.index('step = byte_session_step_run(')
    assert 'byte_owner_recover(s[], completed + k)' in steps
    session = _read('training/byte_lm_host_session.mojo')
    host_steps = session.split('    def steps(')[1].split('    def evaluate(')[0]
    assert 'self.step(' in host_steps and 'self.optimizer.copy()' in host_steps


def test_binding_admits_once_and_runs_the_remaining_steps_in_one_call():
    native = _read('bindings/byte_lm_trainer_native.mojo')
    assert '"mojolearn.training.byte_lm.v1", 13)' in native
    entry = native.split('def byte_lm_train_steps(')[1].split('def _logits_input(')[0]
    assert entry.count('py_call(lock.acquire)') == 1
    assert entry.count('byte_state_admit(') == 1 and entry.count('_admitted_shape(') == 1
    run = native.split('def _run_steps(')[1].split('def byte_lm_train_steps(')[0]
    assert run.count('neural_descriptor_execute(provider, _BYTE_LM_OP_STEPS,') == 1
    assert '_steps_reconcile(' in run
    assert 'if operation == "train_steps":' in native


def test_step_descriptors_are_native_words():
    native = _read('bindings/byte_lm_trainer_native.mojo')
    resident = native.split('def _run_resident(')[1].split('def _run_stateless(')[0]
    assert '_native_execute_words(fields, binding, provider, 1, a,' in resident
    assert 'var a = List[Int]()' in resident


def test_teardown_switch_is_read_once():
    for name in ('bindings/byte_lm_trainer_native.mojo', 'bindings/_mojolearn_byte_lm.mojo'):
        source = _read(name)
        assert 'getenv("MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL")' not in source
        assert 'byte_lm_teardown_with_gil()' in source


def test_ragged_next_bytes_picks_on_the_device_and_natively_on_the_host():
    logits = _read('training/byte_lm_logits.mojo')
    assert 'def byte_next_byte_rows_kernel(' in logits
    assert 'ctx.enqueue_function[byte_next_byte_rows_kernel](' in logits
    ops = _read('training/byte_lm_session_ops.mojo')
    owner = ops.split('def byte_owner_logits(')[1].split('def byte_file_sha256(')[0]
    assert 'next_bytes and len(addr) == 7' in owner
    host = _read('training/byte_lm_host_trainer_dispatch.mojo')
    assert 'owner[].next_rows(' in host
    native = _read('bindings/byte_lm_trainer_native.mojo')
    body = native.split('def byte_lm_logits(')[1].split('def _registry(')[0]
    assert 'args[1] is None or provider is not None' in body


def test_pool_and_offload_exports_write_into_the_caller():
    binding = _read('bindings/_mojolearn_byte_lm.mojo')
    assert 'export_values(' not in binding
    assert binding.count('.export_into([0, 1, 2],') == 4
    for name in ('training/byte_lm_model_pool.mojo', 'training/byte_lm_offload.mojo'):
        assert 'def export_into(' in _read(name)
