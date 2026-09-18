#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact checkpoint/resume at a shape the 2 MiB checkpoint file refuses.

`SmallByteLanguageModelTrainer.export_checkpoint` refuses any model above
87,381 parameters (`_CHECKPOINT_LIMIT = 2 MiB`, 24 bytes of JSON/hex per
parameter), so `from_checkpoint` cannot be reached at the 162,147,840-parameter
target shape at all. Its own docstring names the substitute: "at larger shapes
`export_state()` arrays are the checkpoint". This harness IS that substitute,
and it restores through exactly the calls `from_checkpoint_bytes` uses --
construct with the saved parameters, optimizer config, shape and data
schedule, then `load_state_dict(state)` -- so what it proves is the real
restore path minus the size-bounded envelope.

THREE ARMS, and the BROKEN ONE RUNS FIRST so the comparison is watched to
fail before it is trusted:

  baseline  one process: `--warmup` steps, save state, then `--tail` more
            steps, witnessing sha256 of loss, gradient, parameters, m, v and
            flags after each tail step.
  control   a FRESH process restores the same state with m and v ZEROED and
            runs the same tail. AdamW without its moments must diverge, so
            every tail witness must DIFFER from the baseline. A control that
            matches means the witnesses cannot see the optimizer and the
            resume arm proves nothing; the harness then fails.
  resume    a FRESH process restores the complete state and runs the same
            tail. Every witness must match the baseline hash for hash.

Batches are a pure function of the absolute step index (the byte-corpus
schedule of tools/lm_step_memory_probe.py, or a seeded generator), so the
tail sees the same tokens in every arm without any state being carried in
the harness.

Every arm prints the matching and differing hashes by name. It never prints
only a count.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_step_memory_probe import CONTROL_SHAPE, TARGET_SHAPE, CorpusBatches  # noqa: E402

SCHEMA = 'mojolearn.lm-shakedown-resume.v1'
ARRAY_KEYS = ('parameters', 'm', 'v', 'flags')
WITNESS_KEYS = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _build_trainer(shape_args, seed, parameters=None, config=None, schedule=None, resident=True):
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    shape = Shape(*shape_args)
    if parameters is None:
        rng = np.random.default_rng(seed)
        parameters = rng.normal(0, .02, shape.n_total).astype(np.float32)
        for entry in Trainer.parameter_registry(shape):
            if 'norm' in entry['name']:
                parameters[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    cfg = config or dict(lr=1e-3, beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01)
    trainer = Trainer(parameters, shape=shape,
                      data_schedule=schedule or {'fixture': 'lm shakedown resume', 'seed': seed},
                      lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']),
                      eps=cfg['eps'], weight_decay=cfg['weight_decay'],
                      resident=resident, step_result='lean' if resident else 'full')
    return trainer, shape


def _batches(args, shape):
    import numpy as np
    if args.corpus:
        corpus = CorpusBatches(args.corpus, shape.batch, shape.length)
        return lambda index: corpus.ids(index)

    def ids(index):
        rng = np.random.default_rng(args.seed * 1000003 + index)
        return rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
    return ids


def _witness(trainer, loss):
    import numpy as np
    gradients = trainer.export_gradients(named=False)['flat_gradients']
    state = trainer.export_state()
    return dict(loss=_sha(np.array([loss], np.float32).tobytes()),
                gradients=_sha(gradients.tobytes()),
                parameters=_sha(state['parameters'].tobytes()),
                m=_sha(state['m'].tobytes()), v=_sha(state['v'].tobytes()),
                flags=_sha(state['flags'].tobytes()))


def _save_state(trainer, directory):
    """The arrays raw, everything else canonical JSON. This IS the checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    state = trainer.export_state()
    meta = {}
    digests = {}
    for key, value in state.items():
        if key in ARRAY_KEYS:
            raw = value.tobytes()
            (directory / (key + '.bin')).write_bytes(raw)
            digests[key] = _sha(raw)
            meta[key] = dict(file=key + '.bin', bytes=len(raw),
                             dtype='<i4' if key == 'flags' else '<f4')
        else:
            meta[key] = value
    meta['array_sha256'] = digests
    (directory / 'state.json').write_text(json.dumps(meta, sort_keys=True, indent=1))
    return digests, state['completed_steps']


def _load_state(directory, zero_moments):
    import numpy as np
    meta = json.loads((directory / 'state.json').read_text())
    digests = meta.pop('array_sha256')
    state = dict(meta)
    for key in ARRAY_KEYS:
        entry = meta[key]
        raw = (directory / entry['file']).read_bytes()
        if _sha(raw) != digests[key]:
            raise SystemExit('checkpoint array %s failed its sha256' % key)
        array = np.frombuffer(raw, dtype=entry['dtype'])
        if zero_moments and key in ('m', 'v'):
            array = np.zeros_like(array)
        state[key] = np.array(array)
    return state


def run_arm(args):
    """One arm in ONE process. Prints a JSON line per tail step."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape_args = TARGET_SHAPE if args.target else args.shape
    record = dict(schema=SCHEMA, arm=args.arm, shape=shape_args, corpus=str(args.corpus) if args.corpus else None)

    if args.arm == 'baseline':
        trainer, shape = _build_trainer(shape_args, args.seed)
        batches = _batches(args, shape)
        t0 = time.perf_counter()
        for index in range(args.warmup):
            trainer.train_step(batches(index))
        record['warmup_seconds'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        digests, completed = _save_state(trainer, Path(args.state))
        record['save_seconds'] = time.perf_counter() - t0
        record['saved_at_step'] = completed
        record['state_sha256'] = digests
        first = args.warmup
    else:
        state = _load_state(Path(args.state), zero_moments=(args.arm == 'control'))
        t0 = time.perf_counter()
        trainer, shape = _build_trainer(shape_args, args.seed, parameters=state['parameters'],
                                        config=state['config'], schedule=state['data_schedule'])
        trainer.load_state_dict(state)
        record['restore_seconds'] = time.perf_counter() - t0
        record['restored_at_step'] = state['completed_steps']
        batches = _batches(args, shape)
        first = state['completed_steps']

    witnesses = []
    for offset in range(args.tail):
        index = first + offset
        start = time.perf_counter()
        result = trainer.train_step(batches(index))
        seconds = time.perf_counter() - start
        hashes = _witness(trainer, result['loss'])
        witnesses.append(dict(step_index=index, seconds=seconds, loss=result['loss'], sha256=hashes))
        print(json.dumps(dict(arm=args.arm, step_index=index, loss=result['loss'], sha256=hashes)), flush=True)
    record['witnesses'] = witnesses
    (out / ('%s.json' % args.arm)).write_text(json.dumps(record, indent=1))
    return 0


def compare_resume(baseline, other):
    """Every witness, every tail step, must match. Prints each by name."""
    ok = True
    print('\n--- resume vs baseline (expect EQUAL everywhere) ---')
    for b, o in zip(baseline['witnesses'], other['witnesses']):
        if b['step_index'] != o['step_index']:
            print('  step index mismatch: baseline %d, resume %d' % (b['step_index'], o['step_index']))
            return False
        for key in WITNESS_KEYS:
            same = b['sha256'][key] == o['sha256'][key]
            print('  step %d %-11s %s  baseline=%s  resume=%s'
                  % (b['step_index'], key, 'SAME' if same else 'DIFF',
                     b['sha256'][key], o['sha256'][key]))
            if not same:
                ok = False
    return ok


def compare_control(baseline, other):
    """The missing-moments control, with the expectation the arithmetic
    actually supports.

    Zeroing m and v leaves the PARAMETERS untouched, so the first tail step
    reads the same weights and its loss and gradient MUST match the baseline.
    That match is the control's own sanity check: it proves the parameter
    restore worked and isolates the optimizer state as the only difference.
    The moments and the updated parameters must differ from that first step
    onward, and by the last tail step the loss and the gradient must differ
    too -- a control whose loss never moves means the witnesses cannot see
    the optimizer and the resume arm would prove nothing.

    `flags` is one int32 per parameter tensor (110 at the target shape, 2 + 9
    per layer) and stays zero in a healthy run, so it cannot
    separate the arms and nothing is required of it.
    """
    ok = True
    print('\n--- control (m and v zeroed) vs baseline ---')
    first, last = baseline['witnesses'][0], baseline['witnesses'][-1]
    cfirst, clast = other['witnesses'][0], other['witnesses'][-1]
    for b, o in zip(baseline['witnesses'], other['witnesses']):
        for key in WITNESS_KEYS:
            same = b['sha256'][key] == o['sha256'][key]
            print('  step %d %-11s %s  baseline=%s  control=%s'
                  % (b['step_index'], key, 'SAME' if same else 'DIFF',
                     b['sha256'][key], o['sha256'][key]))

    def require(condition, message):
        nonlocal ok
        print('  CHECK %s: %s' % ('pass' if condition else 'FAIL', message))
        if not condition:
            ok = False

    for key in ('loss', 'gradients'):
        require(first['sha256'][key] == cfirst['sha256'][key],
                'first tail step %s matches (the parameter restore is sound)' % key)
    for key in ('m', 'v', 'parameters'):
        require(first['sha256'][key] != cfirst['sha256'][key],
                'first tail step %s differs (the zeroed moments reached the update)' % key)
    if len(baseline['witnesses']) > 1:
        for key in ('loss', 'gradients', 'parameters'):
            require(last['sha256'][key] != clast['sha256'][key],
                    'last tail step %s differs (the witnesses can see the optimizer)' % key)
    else:
        require(False, 'the control needs at least 2 tail steps to separate the arms')
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--state', required=True, help='checkpoint directory')
    parser.add_argument('--shape', nargs=9, type=int, default=CONTROL_SHAPE)
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--tail', type=int, default=4,
                        help='at least 2: the control cannot separate the arms on one step')
    parser.add_argument('--seed', type=int, default=93261)
    parser.add_argument('--corpus', type=Path, default=None)
    parser.add_argument('--arm', choices=['baseline', 'control', 'resume'], default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('requires MOJOLEARN_NUMERIC_MODE=identical in the environment')
    if args.tail < 2:
        parser.error('--tail must be at least 2 so the missing-moments control can separate the arms')
    if args.arm:
        return run_arm(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # BROKEN ARM FIRST: baseline, then control (must FAIL), then resume.
    for arm in ('baseline', 'control', 'resume'):
        cmd = [sys.executable, __file__, '--out', str(out), '--state', args.state,
               '--warmup', str(args.warmup), '--tail', str(args.tail),
               '--seed', str(args.seed), '--arm', arm]
        cmd += ['--target'] if args.target else ['--shape'] + [str(x) for x in args.shape]
        if args.corpus:
            cmd += ['--corpus', str(args.corpus)]
        print('\n=== ARM %s (fresh process) ===' % arm, flush=True)
        completed = subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            print('arm %s exited %d' % (arm, completed.returncode))
            (out / 'verdict.json').write_text(json.dumps(
                dict(schema=SCHEMA, verdict='ARM FAILED', arm=arm, returncode=completed.returncode), indent=1))
            return 1

    loaded = {arm: json.loads((out / ('%s.json' % arm)).read_text()) for arm in ('baseline', 'control', 'resume')}
    control_differs = compare_control(loaded['baseline'], loaded['control'])
    resume_matches = compare_resume(loaded['baseline'], loaded['resume'])
    verdict = dict(schema=SCHEMA, shape=loaded['baseline']['shape'],
                   warmup=args.warmup, tail=args.tail,
                   saved_at_step=loaded['baseline'].get('saved_at_step'),
                   save_seconds=loaded['baseline'].get('save_seconds'),
                   restore_seconds=loaded['resume'].get('restore_seconds'),
                   control_missing_moments_differs=control_differs,
                   resume_bitwise_equal=resume_matches,
                   verdict='PASS' if (control_differs and resume_matches) else 'FAIL',
                   qualification='one GPU, one shape; not a cross-vendor resume claim')
    (out / 'verdict.json').write_text(json.dumps(verdict, indent=1))
    print('\n' + json.dumps(verdict, indent=1))
    return 0 if verdict['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
