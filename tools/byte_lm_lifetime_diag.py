#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded Byte-LM trainer lifetime diagnostic (DEVIATION 2494).

Every lifetime case runs in its OWN subprocess with a per-case deadline. The
parent never imports mojolearn, so a hung case cannot take the driver with
it. On a timeout the parent retains, for that case:

  * the child's Python stacks (faulthandler, periodic and on SIGUSR1),
  * a native stack when `py-spy` or `gdb` is on PATH (best effort),
  * the child's per-thread kernel state and wchan from /proc,
  * whether the child's CPU time is still advancing (spinning vs blocked),
  * an `nvidia-smi` utilization/memory sample (if the tool exists),
  * the child's own progress record, written after every completed event.

Where a case completes, the record carries the sha256 of every returned
state/loss/gradient array, so bit equality across lifetime cases can be
compared without shipping the arrays. The parent writes one JSON record per
case plus `summary.json` with the cross-case equality matrix.

IDENTICAL mode only. NumPy is optional: when importable it reproduces the
exact fixture the WP6/WP7 surface captures used (default_rng(19)); otherwise
a pure-Python deterministic fixture is used and the record says so, in which
case hashes compare across cases of the same run but not against the
retained RTX 4090 captures.

    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \\
        python3 tools/byte_lm_lifetime_diag.py --out /path/to/new/dir

DEVIATION 2513 adds three CONTROL cases that never touch the byte LM
binding (or touch it only after another binding has created and destroyed a
context): two KMeans fits through the base binding, two ExtraTrees fits
through the trees binding, and a KMeans fit followed by one stateless byte
LM step. Each of those bindings creates a DeviceContext per call
(bindings/_mojolearn.mojo:421, bindings/_mojolearn_trees.mojo:338) and lets
it die at the end of its GILReleased block, so "second fit" means "second
context after the first was destroyed", the exact shape of every hung case.
It also adds two variants that set MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1, the
binding's opt-in process-lifetime context keeper, and record whether the
keeper was reached (`byte_lm_context_keeper_active`).

This file makes no speed, learning or cross-vendor claim. It records what a
process did, in which order, and where it stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

FIXTURE = dict(batch=1, length=5, d_model=16, n_heads=2, n_kv=1, head_dim=8,
               intermediate=24, n_layers=3, vocab_size=257)
FIXTURE_SEED = 19
FIXTURE_SCALE = .02
DEFAULT_DEADLINE = 120

# Case order is the order of the RUN OWED; each name is one subprocess.
CASES = [
    'stateless_x1',
    'stateless_x2',
    'stateless_then_resident',
    'resident_x2',
    'resident_then_stateless',
    'resident_close_reopen',
    'restore_then_step',
    'failure_recovery',
    'resident_mismatch_recovery',
    'stateless_x2_default_profile',
    'stateless_x2_gc_pause',
    'stateless_x2_launch_blocking',
    # DEVIATION 2513: controls through OTHER bindings, then the keeper.
    'control_kmeans_x2',
    'control_extratrees_x2',
    'control_kmeans_then_bytelm',
    'stateless_x2_keep_context',
    'resident_close_reopen_keep_context',
]

# Controls that must not import the byte LM binding before their own work.
CONTROL_ONLY = ('control_kmeans_x2', 'control_extratrees_x2')
KEEP_CONTEXT_CASES = ('stateless_x2_keep_context', 'resident_close_reopen_keep_context')
CONTROL_FIXTURE = dict(kmeans=dict(rows=256, features=4, clusters=4, seed=2513),
                       extratrees=dict(rows=512, features=8, classes=2, n_estimators=8,
                                       max_depth=6, seed=2513))

# Cases whose second completed training step must be bit-equal to each other
# (same fixture, same completed_steps, only the lifetime differs).
SECOND_STEP_GROUP = ('stateless_x2', 'resident_x2', 'resident_close_reopen',
                     'restore_then_step', 'resident_mismatch_recovery',
                     'stateless_x2_gc_pause', 'stateless_x2_launch_blocking',
                     'stateless_x2_keep_context', 'resident_close_reopen_keep_context')
FIRST_STEP_GROUP = ('stateless_x1', 'stateless_x2', 'stateless_then_resident',
                    'resident_x2', 'resident_then_stateless', 'resident_close_reopen',
                    'restore_then_step', 'failure_recovery', 'resident_mismatch_recovery',
                    'stateless_x2_gc_pause', 'stateless_x2_launch_blocking',
                    'control_kmeans_then_bytelm', 'stateless_x2_keep_context',
                    'resident_close_reopen_keep_context')


# ----------------------------------------------------------------- fixtures

def _fixture_bytes(shape_kwargs):
    """(parameter bytes '<f4', id bytes '<i4', description). NumPy when
    importable reproduces packaging/language_model_smoke.py and the WP67
    surface driver exactly; the fallback is deterministic but different."""
    from mojolearn import LanguageModelConfig
    shape = LanguageModelConfig(**shape_kwargs)
    n = shape.n_total
    count = shape.batch * (shape.length + 1)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        rng = np.random.default_rng(FIXTURE_SEED)
        weights = (rng.standard_normal(n) * FIXTURE_SCALE).astype(np.float32)
        ids = rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
        return weights.tobytes(), ids.tobytes(), 'numpy.default_rng(%d) standard_normal*%s then integers' % (
            FIXTURE_SEED, FIXTURE_SCALE)
    import random
    rng = random.Random(FIXTURE_SEED)
    weights = struct.pack('<%df' % n, *[rng.gauss(0.0, 1.0) * FIXTURE_SCALE for _ in range(n)])
    ids = struct.pack('<%di' % count, *[rng.randrange(shape.vocab_size) for _ in range(count)])
    return weights, ids, 'random.Random(%d) gauss*%s then randrange (NOT the NumPy fixture)' % (
        FIXTURE_SEED, FIXTURE_SCALE)


def _make_inputs(shape_kwargs):
    from mojolearn import LanguageModelConfig
    from mojolearn._buffer import frombytes
    shape = LanguageModelConfig(**shape_kwargs)
    weights, ids, description = _fixture_bytes(shape_kwargs)
    parameters = frombytes(weights, '<f4', (shape.n_total,))
    tokens = frombytes(ids, '<i4', (shape.batch, shape.length + 1))
    return shape, parameters, tokens, description


def _trainer(shape, parameters, resident, tag='lifetime-diag'):
    from mojolearn import LanguageModelTrainer
    return LanguageModelTrainer(parameters, shape=shape, resident=resident,
                                data_schedule={'dataset': tag, 'deviation': 2494})


# ----------------------------------------------------------------- recording

class Recorder:
    """Progress record rewritten after every event, so a hung case keeps
    what it completed. Hashes are sha256 of the exact little-endian bytes."""

    def __init__(self, case, path):
        self.case = case
        self.path = Path(path)
        self.started = time.time()
        self.record = dict(case=case, schema='byte-lm-lifetime-diag.record.v1',
                           pid=os.getpid(), python=sys.version.split()[0],
                           numeric_mode=os.environ.get('MOJOLEARN_NUMERIC_MODE'),
                           events=[], hashes={}, status='running', error=None)
        self.flush()

    def flush(self):
        self.record['elapsed_s'] = round(time.time() - self.started, 3)
        tmp = self.path.with_suffix(self.path.suffix + '.tmp')
        with tmp.open('w') as stream:
            json.dump(self.record, stream, indent=1, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)

    def event(self, name, **fields):
        entry = dict(name=name, t=round(time.time() - self.started, 3), **fields)
        self.record['events'].append(entry)
        self.flush()
        print('[%s] %7.3fs %s %s' % (self.case, entry['t'], name,
                                     json.dumps(fields, sort_keys=True) if fields else ''), flush=True)

    def _hash_value(self, value):
        if hasattr(value, 'tobytes') and hasattr(value, 'shape'):
            raw = value.tobytes()
            return dict(sha256=hashlib.sha256(raw).hexdigest(), nbytes=len(raw),
                        shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value, float):
            raw = struct.pack('<f', value)
            return dict(sha256=hashlib.sha256(raw).hexdigest(), nbytes=4, shape=[], dtype='<f4',
                        value=value)
        if isinstance(value, int):
            return dict(value=value)
        raise TypeError('unhashable result value %r' % type(value))

    def hash_step(self, label, result, state):
        """Record a train_step result and the committed state after it."""
        entry = dict(loss=self._hash_value(result['loss']),
                     step=self._hash_value(result['step']),
                     flat_gradients=self._hash_value(result['flat_gradients']))
        for key in ('parameters', 'm', 'v', 'flags'):
            entry['state.' + key] = self._hash_value(state[key])
        entry['state.completed_steps'] = self._hash_value(state['completed_steps'])
        self.record['hashes'][label] = entry
        self.event('hashed', label=label, loss=result['loss'], step=result['step'])

    def hash_eval(self, label, loss, state):
        entry = dict(loss=self._hash_value(loss))
        for key in ('parameters', 'm', 'v', 'flags'):
            entry['state.' + key] = self._hash_value(state[key])
        self.record['hashes'][label] = entry
        self.event('hashed', label=label, loss=loss)

    def hash_values(self, label, values):
        """Record a mapping of arrays/scalars (the controls' fit outputs)."""
        self.record['hashes'][label] = {key: self._hash_value(value) for key, value in values.items()}
        self.event('hashed', label=label)

    def finish(self, status, error=None):
        self.record['status'] = status
        self.record['error'] = error
        self.flush()


# ----------------------------------------------------------------- cases

def _keeper_active():
    """DEVIATION 2513 reach check: True once the binding's process-lifetime
    keeper holds a context; None on a binary without the read-back."""
    from mojolearn import _backend
    getter = getattr(_backend.binding('_mojolearn_byte_lm', 'identical'),
                     'byte_lm_context_keeper_active', None)
    return None if getter is None else bool(getter())


def _step(rec, trainer, tokens, label):
    rec.event('native_call_begin', label=label, resident=trainer._resident,
              completed_steps=trainer.step_)
    result = trainer.train_step(tokens)
    rec.event('native_call_end', label=label,
              keep_context_env=os.environ.get('MOJOLEARN_BYTE_LM_KEEP_CONTEXT'),
              keeper_active=_keeper_active())
    rec.hash_step(label, result, trainer.state_dict())
    return result


def _eval(rec, trainer, tokens, label):
    rec.event('native_call_begin', label=label, resident=trainer._resident,
              completed_steps=trainer.step_, action='eval')
    loss = trainer.evaluate(tokens)
    rec.event('native_call_end', label=label)
    rec.hash_eval(label, loss, trainer.state_dict())
    return loss


def case_stateless_x1(rec):
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, False)
    try:
        _step(rec, trainer, tokens, 'step1')
    finally:
        trainer.close()


def case_stateless_x2(rec):
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, False)
    try:
        _step(rec, trainer, tokens, 'step1')
        _step(rec, trainer, tokens, 'step2')
    finally:
        trainer.close()


def case_stateless_then_resident(rec):
    # packaging/language_model_smoke.py's order: the stateless trainer steps,
    # then a SECOND trainer (resident) steps from the same initialization.
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    stateless = _trainer(shape, parameters, False)
    resident = _trainer(shape, parameters, True)
    try:
        _step(rec, stateless, tokens, 'step1')
        _step(rec, resident, tokens, 'step1_resident')
    finally:
        stateless.close()
        resident.close()


def case_resident_x2(rec):
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, True)
    try:
        _step(rec, trainer, tokens, 'step1')
        _step(rec, trainer, tokens, 'step2')
        _eval(rec, trainer, tokens, 'eval_after_step2')
    finally:
        trainer.close()


def case_resident_then_stateless(rec):
    # The first context is still ALIVE when the second is created.
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    resident = _trainer(shape, parameters, True)
    stateless = _trainer(shape, parameters, False)
    try:
        _step(rec, resident, tokens, 'step1')
        _step(rec, stateless, tokens, 'step1_stateless')
    finally:
        resident.close()
        stateless.close()


def case_resident_close_reopen(rec):
    # close() tears the context down with the GIL HELD (byte_lm_session_close);
    # the next call creates a new session and context.
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, True)
    try:
        _step(rec, trainer, tokens, 'step1')
        rec.event('close_begin')
        trainer.close()
        rec.event('close_end', session_released=trainer._native_session is None)
        _step(rec, trainer, tokens, 'step2')
    finally:
        trainer.close()


def case_restore_then_step(rec):
    import tempfile
    from mojolearn import LanguageModelTrainer
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, False)
    try:
        _step(rec, trainer, tokens, 'step1')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step1.checkpoint.json'
            trainer.save_checkpoint(path)
            rec.event('checkpoint_saved', nbytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            trainer.close()
            restored = LanguageModelTrainer.from_checkpoint(path, resident=False)
        rec.event('checkpoint_restored', completed_steps=restored.step_)
        try:
            _step(rec, restored, tokens, 'step2')
        finally:
            restored.close()
    finally:
        trainer.close()


def case_failure_recovery(rec):
    # A deliberate bad-shape call (refused before any native work), then a
    # good call. The ids buffer has the wrong shape for the configured model.
    from mojolearn._buffer import frombytes
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    bad = frombytes(tokens.tobytes()[:4 * shape.batch * shape.length], '<i4',
                    (shape.batch, shape.length))
    trainer = _trainer(shape, parameters, False)
    try:
        _step(rec, trainer, tokens, 'step1')
        rec.event('bad_shape_call_begin', shape=list(bad.shape))
        try:
            trainer.train_step(bad)
        except (ValueError, TypeError) as error:
            rec.event('bad_shape_refused', error=str(error)[:200])
        else:
            raise AssertionError('bad-shape ids were accepted')
        _step(rec, trainer, tokens, 'step2')
    finally:
        trainer.close()


def case_resident_mismatch_recovery(rec):
    # tools/byte_lm_session_check.py's control: the host mirror is altered,
    # the resident admission must refuse INSIDE the native call (a failure
    # after context creation), the failed session is discarded, and a
    # restored state must resume on a NEW context.
    from mojolearn._buffer import frombytes
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, True)
    try:
        _step(rec, trainer, tokens, 'step1')
        saved = trainer.state_dict()
        current = trainer._state['parameters']
        raw = bytearray(current.tobytes())
        struct.pack_into('<f', raw, 0, struct.unpack_from('<f', raw, 0)[0] + .25)
        trainer._state['parameters'] = frombytes(bytes(raw), '<f4', current.shape)
        rec.event('mirror_altered')
        try:
            trainer.train_step(tokens)
        except Exception as error:
            rec.event('mismatch_refused', error=str(error)[:200],
                      session_released=trainer._native_session is None)
        else:
            raise AssertionError('resident mismatch control did not fire')
        trainer.load_state_dict(saved)
        rec.event('state_restored', completed_steps=trainer.step_)
        _step(rec, trainer, tokens, 'step2')
    finally:
        trainer.close()


def case_stateless_x2_default_profile(rec):
    # The exact path the Sep 3 and Sep 7 NVIDIA campaigns took 128 to 640
    # times per process: the default B2/L32/DM32 profile through the v1
    # `byte_lm_run` ABI. Separates "shape/ABI" from "second context".
    shape, parameters, tokens, _ = _make_inputs({})
    rec.event('profile', profile=shape.profile)
    trainer = _trainer(shape, parameters, False)
    try:
        _step(rec, trainer, tokens, 'step1')
        _step(rec, trainer, tokens, 'step2')
    finally:
        trainer.close()


def case_stateless_x2_gc_pause(rec):
    # If a deferred teardown races the next creation, a collection and a
    # pause between the two calls would change the outcome.
    import gc
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, False)
    try:
        _step(rec, trainer, tokens, 'step1')
        gc.collect()
        time.sleep(3.0)
        rec.event('paused', seconds=3.0)
        _step(rec, trainer, tokens, 'step2')
    finally:
        trainer.close()


def case_stateless_x2_launch_blocking(rec):
    # The parent sets CUDA_LAUNCH_BLOCKING=1 for this child (see CHILD_ENV).
    rec.event('env', CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING'))
    case_stateless_x2(rec)


# ----------------------------------------------------------------- controls
# DEVIATION 2513. These never construct a LanguageModelTrainer (except the
# mixed case, after the KMeans fit). Small data on purpose: this is a
# lifetime question, not a timing one (the 1M-row floor is for tree TIMING).


def _control_bytes(count, seed, scale=1.0):
    import random
    rng = random.Random(seed)
    return struct.pack('<%df' % count, *[rng.gauss(0.0, 1.0) * scale for _ in range(count)])


def _binding_identity(name, vendor_fn, mode_fn):
    """File, sha256, vendor and compiled mode of a NON byte LM binding."""
    from mojolearn import _backend
    module = _backend.binding(name, 'identical')
    path = getattr(module, '__file__', None)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    return dict(name=name, binding_file=path, binding_sha256=digest,
                native_vendor=str(getattr(module, vendor_fn)()),
                native_numeric_mode=int(getattr(module, mode_fn)()))


def _kmeans_fit(rec, label):
    """One KMeans fit through the base binding: bindings/_mojolearn.mojo:421
    creates `var ctx = DeviceContext()` inside GILReleased and lets it die at
    the end of that block, so a second fit is a second context after the
    first was destroyed."""
    from mojolearn import KMeans
    from mojolearn._buffer import frombytes
    spec = CONTROL_FIXTURE['kmeans']
    x = frombytes(_control_bytes(spec['rows'] * spec['features'], spec['seed']), '<f4',
                  (spec['rows'], spec['features']))
    model = KMeans(n_clusters=spec['clusters'], n_init=1, max_iter=50,
                   random_state=spec['seed'], numeric_mode='identical')
    rec.event('native_call_begin', label=label, binding='_mojolearn', entry='kmeans_fit',
              rows=spec['rows'], features=spec['features'])
    model.fit(x)
    rec.event('native_call_end', label=label, n_iter=int(model.n_iter_),
              inertia=float(model.inertia_))
    rec.hash_values(label, {'cluster_centers_': model.cluster_centers_,
                            'labels_': model.labels_,
                            'inertia_': float(model.inertia_),
                            'n_iter_': int(model.n_iter_)})
    return model


def _extratrees_fit(rec, label):
    """One ExtraTreesClassifier fit through the trees binding:
    bindings/_mojolearn_trees.mojo:338 creates a local DeviceContext per fit
    (et_classifier_fit_export); the export registry (ET_EXPORTS, a
    std.ffi._Global) keeps only a HOST FitResult afterwards, and the Python
    side copies it out and releases the handle before fit() returns."""
    from mojolearn import ExtraTreesClassifier
    from mojolearn._buffer import frombytes
    spec = CONTROL_FIXTURE['extratrees']
    x = frombytes(_control_bytes(spec['rows'] * spec['features'], spec['seed']), '<f4',
                  (spec['rows'], spec['features']))
    # Labels: the sign of the first feature, so the split is learnable and
    # both classes are present.
    raw = x.tobytes()
    labels = [1 if struct.unpack_from('<f', raw, 4 * spec['features'] * r)[0] > 0.0 else 0
              for r in range(spec['rows'])]
    y = frombytes(struct.pack('<%di' % spec['rows'], *labels), '<i4', (spec['rows'],))
    model = ExtraTreesClassifier(n_estimators=spec['n_estimators'], max_depth=spec['max_depth'],
                                 random_state=spec['seed'], numeric_mode='identical')
    rec.event('native_call_begin', label=label, binding='_mojolearn_trees',
              entry='et_classifier_fit(_export)', rows=spec['rows'], features=spec['features'])
    model.fit(x, y)
    rec.event('native_call_end', label=label, n_trees=int(model._n_trees))
    rec.hash_values(label, {'offsets': model._offsets, 'colid': model._colid,
                            'quesval': model._quesval, 'left_child': model._left_child,
                            'leaves': model._leaves})
    return model


def case_control_kmeans_x2(rec):
    rec.record['control_binding'] = _binding_identity('_mojolearn', 'mojolearn_vendor',
                                                      'mojolearn_numeric_mode')
    _kmeans_fit(rec, 'fit1')
    _kmeans_fit(rec, 'fit2')


def case_control_extratrees_x2(rec):
    rec.record['control_binding'] = _binding_identity('_mojolearn_trees', 'trees_vendor',
                                                      'trees_numeric_mode')
    _extratrees_fit(rec, 'fit1')
    _extratrees_fit(rec, 'fit2')


def case_control_kmeans_then_bytelm(rec):
    # A context created and destroyed by the BASE binding, then the byte
    # LM's first (stateless) context. Hangs here with control_kmeans_x2
    # passing: the byte LM's context CREATION side is the sensitive one.
    rec.record['control_binding'] = _binding_identity('_mojolearn', 'mojolearn_vendor',
                                                      'mojolearn_numeric_mode')
    _kmeans_fit(rec, 'fit1')
    shape, parameters, tokens, _ = _make_inputs(FIXTURE)
    trainer = _trainer(shape, parameters, False)
    try:
        _step(rec, trainer, tokens, 'step1')
    finally:
        trainer.close()


def case_stateless_x2_keep_context(rec):
    # The parent sets MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1 (CHILD_ENV): the
    # first creating call also creates the keeper context, which is never
    # released, so the second call's context is created while one is alive.
    rec.event('env', MOJOLEARN_BYTE_LM_KEEP_CONTEXT=os.environ.get('MOJOLEARN_BYTE_LM_KEEP_CONTEXT'),
              keeper_active_before=_keeper_active())
    case_stateless_x2(rec)


def case_resident_close_reopen_keep_context(rec):
    rec.event('env', MOJOLEARN_BYTE_LM_KEEP_CONTEXT=os.environ.get('MOJOLEARN_BYTE_LM_KEEP_CONTEXT'),
              keeper_active_before=_keeper_active())
    case_resident_close_reopen(rec)


CHILD_ENV = {
    'stateless_x2_launch_blocking': {'CUDA_LAUNCH_BLOCKING': '1'},
    'stateless_x2_keep_context': {'MOJOLEARN_BYTE_LM_KEEP_CONTEXT': '1'},
    'resident_close_reopen_keep_context': {'MOJOLEARN_BYTE_LM_KEEP_CONTEXT': '1'},
}

CASE_FUNCTIONS = {name: globals()['case_' + name] for name in CASES}


# ----------------------------------------------------------------- child

def run_child(case, record_path, deadline):
    import faulthandler
    stack_path = Path(record_path).with_suffix('.pystack.txt')
    stack_file = stack_path.open('w')
    faulthandler.enable(file=stack_file, all_threads=True)
    if hasattr(signal, 'SIGUSR1'):
        faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True, chain=False)
    # Periodic dumps: the last one before the parent's deadline is retained
    # even if the parent's signal never reaches a wedged process.
    faulthandler.dump_traceback_later(max(5.0, min(30.0, deadline / 3.0)), repeat=True, file=stack_file)
    rec = Recorder(case, record_path)
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        rec.finish('refused', 'MOJOLEARN_NUMERIC_MODE must be identical')
        return 3
    try:
        import mojolearn
        from mojolearn import LanguageModelTrainer
        rec.event('imported', mojolearn_file=getattr(mojolearn, '__file__', None),
                  version=getattr(mojolearn, '__version__', None))
        if case in CONTROL_ONLY:
            # DEVIATION 2513: the pure controls never load the byte LM
            # binding; their own binding identity is recorded by the case.
            rec.record['fixture'] = dict(control=dict(CONTROL_FIXTURE))
            rec.record['binding'] = None
        else:
            probe = _make_inputs(FIXTURE)
            rec.record['fixture'] = dict(shape=dict(FIXTURE), profile=probe[0].profile,
                                         n_total=probe[0].n_total, source=probe[3])
            # Binding identity without any model execution.
            witness = _trainer(probe[0], probe[1], False).run_metadata()
            rec.record['binding'] = dict(binding_file=witness['binding_file'],
                                         binding_sha256=witness['binding_sha256'],
                                         native_vendor=witness['native_vendor'],
                                         native_profile=witness['native_profile'])
            rec.event('binding', vendor=witness['native_vendor'], sha256=witness['binding_sha256'][:12])
        CASE_FUNCTIONS[case](rec)
    except BaseException as error:  # noqa: BLE001 -- the record must say what happened
        rec.finish('failed', '%s: %s' % (type(error).__name__, error))
        import traceback
        traceback.print_exc()
        return 1
    rec.finish('passed')
    return 0


# ----------------------------------------------------------------- parent

def _read_text(path):
    try:
        return Path(path).read_text(errors='replace')
    except OSError as error:
        return 'unavailable: %s' % error


def _proc_snapshot(pid):
    """Per-thread state and wchan for a Linux pid; empty elsewhere."""
    root = Path('/proc') / str(pid)
    out = {}
    if not root.is_dir():
        return out
    out['stat'] = _read_text(root / 'stat').strip()
    out['status'] = _read_text(root / 'status')
    out['wchan'] = _read_text(root / 'wchan').strip()
    threads = {}
    task = root / 'task'
    if task.is_dir():
        for tid in sorted(task.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            threads[tid.name] = dict(stat=_read_text(tid / 'stat').strip(),
                                     wchan=_read_text(tid / 'wchan').strip(),
                                     stack=_read_text(tid / 'stack')[:4000])
    out['threads'] = threads
    return out


def _cpu_ticks(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[11]) + int(fields[12])  # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def _run_tool(argv, timeout=60):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(argv=argv, returncode=done.returncode, stdout=done.stdout[-20000:],
                    stderr=done.stderr[-4000:])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=argv, error=str(error))


def _nvidia_sample(rounds=3):
    if shutil.which('nvidia-smi') is None:
        return {'available': False}
    samples = []
    for _ in range(rounds):
        samples.append(_run_tool(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu',
                                  '--format=csv,noheader'], timeout=20).get('stdout', '').strip())
        time.sleep(1.0)
    apps = _run_tool(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], timeout=20)
    return {'available': True, 'samples': samples, 'compute_apps': apps.get('stdout', '').strip()}


def _native_stack(pid):
    result = {}
    if shutil.which('py-spy'):
        result['py-spy'] = _run_tool(['py-spy', 'dump', '--pid', str(pid), '--native'], timeout=90)
    if shutil.which('gdb'):
        result['gdb'] = _run_tool(['gdb', '-p', str(pid), '-batch', '-ex', 'set pagination off',
                                   '-ex', 'thread apply all bt'], timeout=120)
    if not result:
        result['note'] = 'neither py-spy nor gdb on PATH; only Python stacks retained'
    return result


def _hang_diagnostics(pid):
    diag = {}
    ticks0 = _cpu_ticks(pid)
    diag['proc_before'] = _proc_snapshot(pid)
    diag['nvidia'] = _nvidia_sample()
    ticks1 = _cpu_ticks(pid)
    diag['cpu_ticks'] = dict(before=ticks0, after=ticks1,
                             advancing=(None if ticks0 is None or ticks1 is None else ticks1 > ticks0),
                             note='advancing=true means the child is spinning (host loop or driver '
                                  'spin-wait); false means it is blocked (futex/cond wait)')
    diag['native'] = _native_stack(pid)
    diag['proc_after'] = _proc_snapshot(pid)
    return diag


def run_case(case, out, deadline, python):
    record_path = out / (case + '.json')
    log_path = out / (case + '.log')
    env = dict(os.environ)
    env['MOJOLEARN_NUMERIC_MODE'] = 'identical'
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.update(CHILD_ENV.get(case, {}))
    argv = [python, os.path.abspath(__file__), '--case', case, '--record', str(record_path),
            '--deadline', str(deadline)]
    started = time.time()
    with log_path.open('w') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
        timed_out = False
        diagnostics = None
        try:
            child.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            if hasattr(signal, 'SIGUSR1'):
                try:
                    os.kill(child.pid, signal.SIGUSR1)
                except OSError:
                    pass
                time.sleep(2.0)
            diagnostics = _hang_diagnostics(child.pid)
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except OSError:
                child.kill()
            child.wait()
    wall = time.time() - started
    record = None
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as error:
        record = dict(unreadable=str(error))
    result = dict(case=case, argv=argv, env_overrides=CHILD_ENV.get(case, {}),
                  exit_code=child.returncode, wall_s=round(wall, 3), deadline_s=deadline,
                  timed_out=timed_out,
                  status='timeout' if timed_out else (record.get('status') if isinstance(record, dict) else None),
                  record=record, log=str(log_path),
                  python_stacks=_read_text(record_path.with_suffix('.pystack.txt'))[-20000:],
                  hang_diagnostics=diagnostics)
    with (out / (case + '.result.json')).open('w') as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write('\n')
    return result


def _equality(results, group, label_of):
    """For each hashed field, the set of distinct sha256 values across the
    completed cases of `group`; one value means bit-equal."""
    fields = {}
    present = []
    for result in results:
        case = result['case']
        if case not in group:
            continue
        record = result.get('record') or {}
        label = label_of(case)
        hashes = (record.get('hashes') or {}).get(label)
        if not hashes:
            continue
        present.append(case)
        for key, value in hashes.items():
            if isinstance(value, dict) and 'sha256' in value:
                fields.setdefault(key, {})[case] = value['sha256']
    verdict = {}
    for key, per_case in fields.items():
        distinct = sorted(set(per_case.values()))
        verdict[key] = dict(bit_equal=len(distinct) == 1, distinct=len(distinct), per_case=per_case)
    return dict(cases_compared=present, fields=verdict)


def _pair_equality(results, case, label_a, label_b):
    """Per-field bit equality of two hashed labels inside ONE case."""
    for result in results:
        if result['case'] != case:
            continue
        hashes = (result.get('record') or {}).get('hashes') or {}
        a, b = hashes.get(label_a), hashes.get(label_b)
        if not a or not b:
            return dict(compared=False)
        fields = {}
        for key in a:
            if isinstance(a[key], dict) and 'sha256' in a[key] and key in b:
                fields[key] = dict(bit_equal=a[key]['sha256'] == b[key].get('sha256'))
        return dict(compared=True, fields=fields, all_bit_equal=all(v['bit_equal'] for v in fields.values()))
    return dict(compared=False)


def _keeper_reached(results, case):
    """The last `native_call_end` event's keeper read-back for a case: True
    means the keeper held a context (the switch was reached), False means
    the switch was set but never armed (a stale binary or an unreached
    branch), None means the binary has no read-back."""
    for result in results:
        if result['case'] != case:
            continue
        events = (result.get('record') or {}).get('events') or []
        ends = [e for e in events if e.get('name') == 'native_call_end']
        return dict(observed=bool(ends),
                    keeper_active=(ends[-1].get('keeper_active') if ends else None),
                    keep_context_env=(ends[-1].get('keep_context_env') if ends else None))
    return dict(observed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, help='new output directory (parent mode)')
    parser.add_argument('--deadline', type=float, default=DEFAULT_DEADLINE, help='seconds per case')
    parser.add_argument('--cases', default=','.join(CASES), help='comma-separated subset, in order')
    parser.add_argument('--python', default=sys.executable, help='interpreter for the children')
    parser.add_argument('--case', help=argparse.SUPPRESS)
    parser.add_argument('--record', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        if args.case not in CASE_FUNCTIONS or not args.record:
            parser.error('unknown case or missing --record')
        return run_child(args.case, args.record, args.deadline)
    if args.out is None:
        parser.error('--out is required')
    cases = [name.strip() for name in args.cases.split(',') if name.strip()]
    unknown = [name for name in cases if name not in CASE_FUNCTIONS]
    if unknown:
        parser.error('unknown cases: %s (known: %s)' % (unknown, ', '.join(CASES)))
    if not 5 <= args.deadline <= 1800:
        parser.error('--deadline must be in [5, 1800] seconds')
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in cases:
        print('== %s (deadline %.0fs)' % (case, args.deadline), flush=True)
        result = run_case(case, out, args.deadline, args.python)
        results.append(result)
        print('   %s exit=%s wall=%.1fs' % (result['status'], result['exit_code'], result['wall_s']), flush=True)
    summary = dict(
        schema='byte-lm-lifetime-diag.summary.v1',
        deviation=[2494, 2513],
        fixture=dict(shape=FIXTURE, seed=FIXTURE_SEED, scale=FIXTURE_SCALE),
        deadline_s=args.deadline,
        python=args.python,
        cases={r['case']: dict(status=r['status'], exit_code=r['exit_code'], wall_s=r['wall_s'],
                               timed_out=r['timed_out']) for r in results},
        first_step_equality=_equality(results, FIRST_STEP_GROUP, lambda case: 'step1'),
        second_step_equality=_equality(results, SECOND_STEP_GROUP, lambda case: 'step2'),
        stateless_then_resident_equality=_equality(
            results, ('stateless_then_resident',), lambda case: 'step1_resident'),
        resident_then_stateless_equality=_equality(
            results, ('resident_then_stateless',), lambda case: 'step1_stateless'),
        # DEVIATION 2513: each control's two fits must be bit-equal (same
        # seed, IDENTICAL); a difference is a separate finding.
        control_fit_equality={
            case: _pair_equality(results, case, 'fit1', 'fit2')
            for case in ('control_kmeans_x2', 'control_extratrees_x2')},
        keeper_reached={
            case: _keeper_reached(results, case) for case in KEEP_CONTEXT_CASES},
        hung=[r['case'] for r in results if r['timed_out']],
        failed=[r['case'] for r in results if not r['timed_out'] and r['exit_code'] != 0],
        passed=[r['case'] for r in results if r['exit_code'] == 0],
    )
    # The mixed cases' second trainer ran completed_steps=0 -> 1 on a second
    # context, so it must equal every case's step1, not step2.
    mixed = {}
    for result in results:
        record = result.get('record') or {}
        for label in ('step1_resident', 'step1_stateless'):
            hashes = (record.get('hashes') or {}).get(label)
            if hashes:
                mixed[result['case'] + ':' + label] = {k: v['sha256'] for k, v in hashes.items()
                                                       if isinstance(v, dict) and 'sha256' in v}
    reference = None
    for result in results:
        record = result.get('record') or {}
        hashes = (record.get('hashes') or {}).get('step1')
        if hashes and result['case'] == 'stateless_x1':
            reference = {k: v['sha256'] for k, v in hashes.items() if isinstance(v, dict) and 'sha256' in v}
    summary['mixed_second_trainer_vs_stateless_x1_step1'] = {
        key: dict(bit_equal=(reference is not None and value == reference)) for key, value in mixed.items()}
    with (out / 'summary.json').open('w') as stream:
        json.dump(summary, stream, indent=1, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(passed=summary['passed'], failed=summary['failed'], hung=summary['hung']),
                     sort_keys=True), flush=True)
    return 0 if not summary['hung'] and not summary['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())
