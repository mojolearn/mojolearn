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


def test_both_columns_take_n_steps_through_operation_1():
    # lane x7-byte-lm-on-executor: operation 12 is gone on both columns;
    # operation 1 takes the step count (p[14]). The device runs the steps as
    # k restored hand steps in one native call (default since 2026-11-13) or,
    # under -D MOJOLEARN_BYTE_LM_PROGRAM_STEP, as ONE Program.fit.
    device = _read('training/byte_lm_native_dispatch.mojo')
    assert 'BYTE_LM_NATIVE_OPERATION_COUNT = 13' in device
    assert 'byte_owner_step(owner[], addresses, completed, cfg, shape, action == 1, max(1, p[14]))' in device
    assert 'operation 12 was removed' in device
    host = _read('training/byte_lm_host_trainer_dispatch.mojo')
    assert 'BYTE_HOST_TRAINER_COUNT = 13' in host
    assert 'max(1, p[14])' in host and 'owner.steps([a[0], a[5], a[6]], n_steps, completed)' in host
    ops = _read('training/byte_lm_session_ops.mojo')
    assert 'def byte_owner_steps(' not in ops
    step = ops.split('def byte_owner_step(')[1].split('def byte_owner_export_state(')[0]
    # every batch admitted before any device work, then one multi-step run
    assert step.index('byte_validate_tokens(') < step.index('byte_session_steps_run(')
    assert 'byte_owner_recover(s[], completed)' in step
    run = ops.split('def byte_session_steps_run(')[1].split('def byte_owner_step(')[0]
    assert 'byte_train_steps_resident(' in run
    lm = _read('training/byte_lm.mojo')
    steps = lm.split('def byte_train_steps_resident(')[1].split('def byte_rollback(')[0]
    assert 'comptime if BYTE_LM_PROGRAM_STEP:' in steps and '_byte_program_steps(' in steps
    # the hand arm: one hand step per batch, the call's entry state restored on a refusal
    assert 'byte_train_step_resident(' in steps and '# driver-loop: over the call\'s training steps' in steps
    assert 'trainer.completed_steps = call_step' in steps
    program_steps = lm.split('def _byte_program_steps(')[1].split('def _byte_program_gradient(')[0]
    assert 'byte_program_fit(' in program_steps
    program = _read('training/byte_lm_program.mojo')
    assert 'train_program_control(' in program


def test_binding_admits_once_and_runs_every_step_in_one_call():
    native = _read('bindings/byte_lm_trainer_native_door.mojo')
    # lane byte-lm-sessions-native: the resident session runs through the
    # image's native session entry (core/byte_lm_entry_table_sessions.mojo)
    assert 'byte_lm_resident_run(key, handle, operation,' in native
    entry = native.split('def byte_lm_train_steps(')[1].split('def _logits_input(')[0]
    assert entry.count('py_call(lock.acquire)') == 1
    assert entry.count('byte_state_admit(') == 1 and entry.count('_admitted_shape(') == 1
    run = native.split('def _run_steps(')[1].split('def byte_lm_train_steps(')[0]
    assert 'while k < n_steps' not in run
    assert run.count('_run_resident(mods, fields, stack, shape, binding, context, True, n_steps, True)') == 1
    resident = native.split('def _run_resident(')[1].split('def _run_stateless(')[0]
    assert '_steps_reconcile(' in resident
    assert 'if operation == "train_steps":' in native


def test_step_descriptors_are_native_words():
    native = _read('bindings/byte_lm_trainer_native_door.mojo')
    resident = native.split('def _run_resident(')[1].split('def _run_stateless(')[0]
    assert '_native_execute(fields, 1, a,' in resident
    assert 'var a = List[Int]()' in resident


def test_teardown_switch_is_read_once():
    for name in ('bindings/byte_lm_trainer_native_door.mojo', 'bindings/_mojolearn_byte_lm.mojo'):
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
    native = _read('bindings/byte_lm_trainer_native_door.mojo')
    body = native.split('def byte_lm_logits(')[1].split('def _registry(')[0]
    # every byte LM image carries the trainer table (lane calltable-byte-lm)
    assert 'var direct_next = next_bytes and resident' in body
    assert 'provider is None' not in body and 'py_call_typed' not in body


def test_pool_and_offload_exports_write_into_the_caller():
    binding = _read('bindings/_mojolearn_byte_lm.mojo')
    # lane rs-embedding-bytelm: the export bodies are RUN code beside the door.
    run = _read('training/byte_lm_door_run.mojo')
    assert 'export_values(' not in binding and 'export_values(' not in run
    assert run.count('.export_into([0, 1, 2],') == 4
    for name in ('training/byte_lm_model_pool.mojo', 'training/byte_lm_offload.mojo'):
        assert 'def export_into(' in _read(name)


def test_cpu_column_follows_the_gpu_columns_step_switch():
    # lane restore-byte-lm-host-step (2026-11-13): one switch for every
    # column, defined once in the host-safe config module.
    config = _read('training/byte_lm_config.mojo')
    assert 'comptime BYTE_LM_PROGRAM_STEP = is_defined["MOJOLEARN_BYTE_LM_PROGRAM_STEP"]()' in config
    lm = _read('training/byte_lm.mojo')
    assert 'comptime BYTE_LM_PROGRAM_STEP =' not in lm
    assert 'from training.byte_lm_config import ByteConfig, BYTE_LM_PROGRAM_STEP' in lm
    # the stateless step and loss: composed by default, the host program as the B arm
    execution = _read('training/byte_lm_host_execution.mojo')
    train = execution.split('def byte_host_train_native(')[1].split('def byte_host_loss_native(')[0]
    assert train.index('comptime if not BYTE_LM_PROGRAM_STEP:') < train.index('byte_host_train_step(')
    assert train.index('byte_host_train_step(') < train.index('byte_program_fit(')
    loss = execution.split('def byte_host_loss_native(')[1]
    assert loss.index('comptime if not BYTE_LM_PROGRAM_STEP:') < loss.index('byte_host_loss(')
    assert loss.index('byte_host_loss(') < loss.index('byte_program_eval(')
    # the session: k composed steps on the call's own lists, committed together
    session = _read('training/byte_lm_host_session.mojo')
    fit = session.split('def _fit(')[1].split('def step(')[0]
    assert 'comptime if BYTE_LM_PROGRAM_STEP:' in fit and 'byte_program_fit(' in fit
    assert 'byte_host_train_native(' in fit
    assert "# driver-loop: over the call's training steps (one composed host step each)" in fit
    assert fit.index('host_state_validate(') < fit.index('self.p = p^')
    evaluate = session.split('def evaluate(')[1].split('def rollback(')[0]
    assert 'byte_host_loss_native(' in evaluate and 'byte_program_eval(' in evaluate
    steps = session.split('def steps(')[1].split('def evaluate(')[0]
    assert 'self._fit(a[0], n_steps, a[1], completed)' in steps
    # the composed step itself is back (training/byte_lm_host_backward.mojo)
    backward = _read('training/byte_lm_host_backward.mojo')
    assert 'def byte_host_train_step(' in backward and 'def byte_host_adamw(' not in backward
