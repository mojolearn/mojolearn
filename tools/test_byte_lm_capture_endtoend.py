#!/usr/bin/env python3
"""Run the whole capture harness locally, with a fake trainer and no GPU.

WHY THIS EXISTS, AND WHAT IT COST. The capture harness only ever ran where a GPU
was, so every line of its `main()` was first executed on a rented box. Four
leases were spent on that arrangement. Two of the failures were mine and both
were of a kind that needs no GPU to find:

  * a witness check comparing the BINARY's profile against the RUN's shape,
    which is true at the default shape and false at every other one;
  * two missed call sites after a signature changed, on a continuation line
    inside a dict literal, which Python does not notice until that line runs,
    and that line runs at the very END of a capture, after the device has
    already done all of the expensive work.

The trainer surface the harness touches is small and closed, nine members plus
the constructor and the registry classmethod, so a stand in is cheap. And
`validate_platform_vendor` admits Darwin with Metal and Linux with CUDA, so this
runs on a laptop with nothing built.

WHAT THIS DOES NOT DO. It proves nothing about arithmetic. Every number here
comes from the stand in, so a passing run says the harness assembles a complete,
self consistent tree at a given shape and nothing whatever about whether a GPU
agrees with a CPU. The bytes are qualified by
tools/byte_lm_cpu_train_gate.py against recorded captures. This is the part that
was missing, not a replacement for that part.
"""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f'_{name}', ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shape_module = _load('byte_lm_shape')
capture = _load('byte_lm_real_text_capture')

#: The one vendor each platform's admission rule allows, so the harness's own
#: refusal stays intact rather than being monkeypatched away.
VENDOR = 'metal' if sys.platform == 'darwin' else 'cuda'


def install_fake_trainer(monkeypatch, tmp_path):
    """Put a stand in where `main()` imports its trainer from.

    The imports happen inside `main()`, so the modules are injected into
    `sys.modules` first, which is the same pattern the host binding tests use."""
    import numpy as np

    # Callers hand this a per-run subdirectory so two captures in one test cannot
    # share an output path, and a subdirectory of tmp_path does not exist yet.
    tmp_path.mkdir(parents=True, exist_ok=True)
    binding = tmp_path / '_mojolearn_byte_lm.so'
    binding.write_bytes(b'not a real binding, only its digest is checked')
    digest = hashlib.sha256(binding.read_bytes()).hexdigest()

    class Config:
        """Stands in for ByteLanguageModelConfig, holding the nine dimensions."""

        def __init__(self, **fields):
            missing = set(shape_module.FIELDS) - set(fields)
            if missing:
                raise ValueError(f'fake config needs every dimension, missing {sorted(missing)}')
            for name, value in fields.items():
                setattr(self, name, value)

        @property
        def derived(self):
            return shape_module.Shape([getattr(self, n) for n in shape_module.FIELDS])

    class FakeTrainer:
        """Every member the harness touches, and nothing else.

        The arithmetic is deliberately trivial and exact in FP32. What is NOT
        trivial is the bookkeeping the harness validates: the profile of the
        shape it was handed, the optimizer configuration echoed back in the
        exact form the harness compares against, the step counters advancing,
        and the state arrays at the sizes the registry implies."""

        def __init__(self, parameters, *, data_schedule, lr, betas, eps,
                     weight_decay, shape=None, resident=False, step_result=None):
            if shape is None:
                raise AssertionError('the harness must hand the trainer its shape')
            self.shape = shape.derived
            n = self.shape.n_total
            if len(parameters) != n:
                raise AssertionError(f'initialization is {len(parameters)}, shape implies {n}')
            self.parameters = np.array(parameters, dtype=np.float32)
            self.m = np.zeros(n, dtype=np.float32)
            self.v = np.zeros(n, dtype=np.float32)
            self.flags = np.zeros(20, dtype=np.int32)
            self.completed = 0
            self.schedule = data_schedule
            self.seen = []
            self.config = dict(
                kind=2, lr=float(np.float32(lr)), beta1=float(np.float32(betas[0])),
                beta2=float(np.float32(betas[1])), eps=float(np.float32(eps)),
                weight_decay=float(np.float32(weight_decay)), momentum=0.,
                dampening=0., nesterov=False, max_norm=0.)

        @property
        def step_(self):
            return self.completed

        def state_dict(self):
            return dict(profile=self.shape.profile, data_schedule=self.schedule,
                        config=dict(self.config), completed_steps=self.completed,
                        next_batch_index=self.completed,
                        parameters=self.parameters.copy(), m=self.m.copy(),
                        v=self.v.copy(), flags=self.flags.copy())

        def load_state_dict(self, state):
            self.parameters = np.array(state['parameters'], dtype=np.float32)
            self.m = np.array(state['m'], dtype=np.float32)
            self.v = np.array(state['v'], dtype=np.float32)

        def train_step(self, ids):
            expected = (self.shape.batch, self.shape.length + 1)
            if tuple(ids.shape) != expected:
                raise AssertionError(f'ids {tuple(ids.shape)}, shape implies {expected}')
            self.seen.append(ids.copy())
            n = self.shape.n_total
            self.completed += 1
            gradient = np.full(n, np.float32(0.5), dtype=np.float32)
            self.parameters = (self.parameters - np.float32(0.25)).astype(np.float32)
            self.m = np.full(n, np.float32(0.125), dtype=np.float32)
            self.v = np.full(n, np.float32(0.0625), dtype=np.float32)
            return dict(flat_gradients=gradient, loss=float(np.float32(1.5)))

        def evaluate(self, ids):
            # A FALLING held-out loss, so a full run exercises the learning gate
            # as a gate rather than skipping past it. 2.0 over 2.5 is a ratio of
            # 0.8, inside the harness's predeclared 0.9 threshold.
            expected = (self.shape.batch, self.shape.length + 1)
            if tuple(ids.shape) != expected:
                raise AssertionError(f'heldout ids {tuple(ids.shape)}, shape implies {expected}')
            return float(np.float32(2.5 if self.completed == 0 else 2.0))

        def export_gradients(self):
            return dict(flat_gradients=np.full(self.shape.n_total, np.float32(0.5),
                                               dtype=np.float32))

        def run_metadata(self):
            # native_profile is the BINARY's identity and never varies with the
            # shape; profile is the RUN's. Conflating them is the bug this file
            # was written to catch.
            return dict(native_vendor=VENDOR,
                        native_profile=shape_module.DEFAULT_PROFILE,
                        native_numeric_mode=1, profile=self.shape.profile,
                        binding_file=str(binding), binding_sha256=digest,
                        step_result='full', source_sha256='0' * 64)

        def save_checkpoint(self, path):
            Path(path).write_text(json.dumps(
                dict(profile=self.shape.profile, completed_steps=self.completed)) + '\n')

        @staticmethod
        def parameter_registry(shape=None):
            derived = shape.derived
            return [dict(name=e['name'], shape=e['shape'], offset=e['offset'],
                         size=e['count']) for e in derived.registry()]

    package = types.ModuleType('mojolearn')
    package.ByteLanguageModelConfig = Config
    language_model = types.ModuleType('mojolearn.language_model')
    language_model.SmallByteLanguageModelTrainer = FakeTrainer
    package.language_model = language_model
    monkeypatch.setitem(sys.modules, 'mojolearn', package)
    monkeypatch.setitem(sys.modules, 'mojolearn.language_model', language_model)
    return FakeTrainer


def run_capture(monkeypatch, tmp_path, spec):
    install_fake_trainer(monkeypatch, tmp_path)
    monkeypatch.setenv('MOJOLEARN_NUMERIC_MODE', 'identical')
    out = tmp_path / 'capture'
    argv = ['byte_lm_real_text_capture.py', '--output', str(out),
            '--expected-vendor', VENDOR, '--steps', '1']
    if spec is not None:
        argv += ['--shape', spec]
    monkeypatch.setattr(sys, 'argv', argv)
    assert capture.main() == 0
    return out


@pytest.mark.parametrize('spec', [None, '4,32'])
def test_the_whole_capture_runs_at_both_shapes(tmp_path, monkeypatch, spec):
    shape = shape_module.parse(spec)
    out = run_capture(monkeypatch, tmp_path, spec)

    step = out / 'step000001'
    manifest = json.loads((step / 'capture.json').read_text())
    assert manifest['schema'] == 'mojolearn.byte-lm.gradient-capture.v1'
    assert manifest['config']['profile'] == shape.profile
    assert manifest['registry'] == shape.registry()

    # Every recorded digest must describe the bytes beside it, which is the
    # property every downstream reader relies on.
    for key, descriptor in manifest['arrays'].items():
        raw = (step / descriptor['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == descriptor['sha256'], key
        assert len(raw) == descriptor['count'] * 4, key

    counts = shape.counts()
    assert set(manifest['arrays']) == set(counts)
    for key, count in counts.items():
        assert manifest['arrays'][key]['count'] == count, key

    summary = json.loads((out / 'summary.json').read_text())
    assert summary['completed_steps'] == 1 and summary['expected_steps'] == 1
    assert summary['records'][0]['capture'] == 'step000001'
    assert summary['initial_state']['completed_steps'] == 0
    assert summary['final_state']['completed_steps'] == 1


def test_the_second_shape_records_its_shape_and_the_default_does_not(tmp_path, monkeypatch):
    """Absence means the default, because that is what every retained tree is,
    and writing the key for the default would change bytes those readers check."""
    four = run_capture(monkeypatch, tmp_path / 'four', '4,32')
    config = json.loads((four / 'step000001' / 'capture.json').read_text())['config']
    assert config['model_shape'] == shape_module.parse('4,32').to_json()

    two = run_capture(monkeypatch, tmp_path / 'two', None)
    config = json.loads((two / 'step000001' / 'capture.json').read_text())['config']
    assert 'model_shape' not in config


def test_the_two_shapes_differ_where_they_must_and_agree_where_they_must(tmp_path, monkeypatch):
    two = run_capture(monkeypatch, tmp_path / 'two', None)
    four = run_capture(monkeypatch, tmp_path / 'four', '4,32')

    ids_two = (two / 'step000001' / 'ids.i32').read_bytes()
    ids_four = (four / 'step000001' / 'ids.i32').read_bytes()
    assert len(ids_two) == 66 * 4
    assert len(ids_four) == 132 * 4

    # Same parameter count at both shapes, which is the point of this second
    # shape: the arrays are the same size and the sums behind them are not.
    grad_two = (two / 'step000001' / 'grad.f32').read_bytes()
    grad_four = (four / 'step000001' / 'grad.f32').read_bytes()
    assert len(grad_two) == len(grad_four) == 34944 * 4

    # The held out schedule is the same 512 target bytes read either way.
    assert len((two / 'heldout-schedule.i32').read_bytes()) == 8 * 66 * 4
    assert len((four / 'heldout-schedule.i32').read_bytes()) == 4 * 132 * 4


def test_a_witness_that_names_the_run_shape_as_the_binary_is_refused(tmp_path, monkeypatch):
    """The exact bug that cost a lease, reproduced against the harness itself."""
    trainer_class = install_fake_trainer(monkeypatch, tmp_path)
    original = trainer_class.run_metadata

    def lying(self):
        witness = original(self)
        witness['native_profile'] = self.shape.profile
        return witness

    monkeypatch.setattr(trainer_class, 'run_metadata', lying)
    monkeypatch.setenv('MOJOLEARN_NUMERIC_MODE', 'identical')
    monkeypatch.setattr(sys, 'argv', [
        'byte_lm_real_text_capture.py', '--output', str(tmp_path / 'cap'),
        '--expected-vendor', VENDOR, '--steps', '1', '--shape', '4,32'])
    with pytest.raises(ValueError, match='native runtime witness differs'):
        capture.main()


def test_a_trainer_given_no_shape_fails_loudly(tmp_path, monkeypatch):
    """If the harness ever stops handing the trainer its shape, the run must not
    quietly proceed at the default."""
    trainer_class = install_fake_trainer(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match='must hand the trainer its shape'):
        trainer_class([0.0], data_schedule={}, lr=1e-3, betas=(.9, .999),
                      eps=1e-8, weight_decay=0.0)


def run_full(monkeypatch, tmp_path, spec, patch=None):
    """A complete 128-step run, which is the only path that evaluates held-out
    batches and reaches the learning gate. `--steps 1` skips both."""
    trainer_class = install_fake_trainer(monkeypatch, tmp_path)
    if patch is not None:
        monkeypatch.setattr(trainer_class, 'evaluate', patch)
    monkeypatch.setenv('MOJOLEARN_NUMERIC_MODE', 'identical')
    out = tmp_path / 'capture'
    argv = ['byte_lm_real_text_capture.py', '--output', str(out),
            '--expected-vendor', VENDOR, '--steps', '128']
    if spec is not None:
        argv += ['--shape', spec]
    monkeypatch.setattr(sys, 'argv', argv)
    return out, capture.main()


def test_a_full_run_evaluates_heldout_at_the_second_shape(tmp_path, monkeypatch):
    """The held-out schedule is where the shape threading is least obvious. The
    four-row shape reads four batches of four rows where the default reads eight
    of two, and both must cover the same 512 target bytes."""
    shape = shape_module.parse('4,32')
    out, code = run_full(monkeypatch, tmp_path, '4,32')
    assert code == 0

    summary = json.loads((out / 'summary.json').read_text())
    assert summary['completed_steps'] == 128 and len(summary['records']) == 128
    assert summary['learning']['eligible'] is True
    assert summary['learning']['observed_gate_passed'] is True
    assert summary['learning']['ratio'] == 2.0 / 2.5

    for folder in ('heldout-initial', 'heldout-final'):
        report = json.loads((out / folder / 'evaluation.json').read_text())
        assert len(report['batches']) == shape.validation_batches == 4
        assert report['state_unchanged'] is True
        assert str(shape_module.VALIDATION_TARGETS) in report['aggregation']
        for index, record in enumerate(report['batches']):
            ids = (out / folder / f'batch{index:02d}.ids.i32').read_bytes()
            assert len(ids) == shape.n_ids * 4 == 132 * 4
            assert record['start'] == shape.validation_starts[index]


def test_the_learning_gate_can_fail(tmp_path, monkeypatch):
    """A gate that cannot fail proves nothing. A flat held-out loss is a ratio of
    one, outside the predeclared threshold, and the run must say so and exit
    non-zero while still retaining everything it captured."""
    import numpy as np

    out, code = run_full(monkeypatch, tmp_path, None,
                         patch=lambda self, ids: float(np.float32(2.5)))
    assert code == 1
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['learning']['observed_gate_passed'] is False
    assert summary['learning']['ratio'] == 1.0
    assert summary['learning']['admitted'] is False
    # The refusal is about the learning claim, not about the bytes: the capture
    # is still complete and still self describing.
    assert len(summary['records']) == 128
    assert (out / 'step000128' / 'capture.json').exists()
