#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does recycling the resident session undo the long-run drift?

Leg 1 of lane/lm-training-shakedown ran 2,000 consecutive steps at the
162,147,840-parameter target shape and found two things a 3-step or 4-step
run cannot see (and every LM run on record in this repository is 3 or 4
steps):

  * device memory grew from 16.949 GB to 34.397 GB between roughly step 210
    and step 476 and then PLATEAUED;
  * step time kept climbing past that plateau, from 0.207 s to over 0.44 s,
    with GPU clocks at maximum, no throttle reason active, 42 C, host RSS
    flat and host CPU pressure `full` at exactly 0.

If the cause lives in one resident session's state, then tearing the session
down and building a new one from the exported state should reset it. That is
also the only mitigation available without a native fix, and it is known to
be lossless: `tools/lm_shakedown_resume.py` proved on the same box that
export_state + load_state_dict continues bit for bit at this shape, with a
missing-moments control that separated.

Two arms, same step count, same batches, same corpus:

  straight   one resident session for the whole run (the leg-1 shape).
  recycle    every --recycle-every steps: export_state(), drop the trainer,
             construct a new one from the exported arrays, load_state_dict,
             continue. The recycle seconds are recorded SEPARATELY and are
             never inside a step's wall time.

Reports per-step wall time, the polled device window and host RSS for both,
so "did it reset" is answered by the curve and not by an average.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_step_memory_probe import (CONTROL_SHAPE, TARGET_SHAPE, CorpusBatches,  # noqa: E402
                                  DeviceMemorySampler, _rss_bytes)

SCHEMA = 'mojolearn.lm-recycle-probe.v1'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _new_trainer(Trainer, shape, parameters, config, schedule):
    return Trainer(parameters, shape=shape, data_schedule=schedule,
                   lr=config['lr'], betas=(config['beta1'], config['beta2']),
                   eps=config['eps'], weight_decay=config['weight_decay'],
                   resident=True, step_result='lean')


def run(args):
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events = out / 'events.jsonl'
    shape_args = TARGET_SHAPE if args.target else args.shape
    shape = Shape(*shape_args)
    tokens_per_step = shape.batch * shape.length

    def emit(record):
        with events.open('a') as stream:
            stream.write(json.dumps(dict(record, t=time.time()), allow_nan=False) + '\n')

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if 'norm' in entry['name']:
            weights[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    schedule = {'fixture': 'lm recycle probe', 'seed': args.seed,
                'arm': 'recycle' if args.recycle_every else 'straight'}
    corpus = CorpusBatches(args.corpus, shape.batch, shape.length) if args.corpus else None

    def batch_ids(index):
        if corpus is not None:
            return corpus.ids(index)
        return rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)

    config = dict(lr=1e-3, beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01)
    trainer = _new_trainer(Trainer, shape, weights, config, schedule)
    sampler = DeviceMemorySampler(args.sample_interval, args.gpu_index)
    sampler.start()
    emit(dict(event='setup', schema=SCHEMA, shape=shape.to_dict(), parameters=shape.n_total,
              steps=args.steps, recycle_every=args.recycle_every,
              corpus=corpus.describe() if corpus is not None else None,
              tokens_per_step=tokens_per_step, seed=args.seed,
              qualification='one GPU, one shape; a drift and mitigation measurement, '
                            'not an opponent ratio and not a default gate'))

    steps, recycles = [], []
    deadline = time.monotonic() + args.budget_seconds
    limited = False
    for index in range(args.steps):
        if time.monotonic() > deadline:
            emit(dict(event='limitation', step=index + 1, budget_seconds=args.budget_seconds))
            limited = True
            break
        ids = batch_ids(index)
        sampler.window_reset()
        start = time.perf_counter()
        result = trainer.train_step(ids)
        seconds = time.perf_counter() - start
        record = dict(event='step_end', step=index + 1, seconds=seconds,
                      tokens_per_second=tokens_per_step / seconds,
                      first_call_includes_setup=(index == 0), loss=result['loss'],
                      completed_steps=result['completed_steps'],
                      host=_rss_bytes(), device=sampler.window_report())
        emit(record)
        steps.append(record)
        if args.recycle_every and (index + 1) % args.recycle_every == 0 and index + 1 < args.steps:
            # OUTSIDE any step's wall time. Tear the session down and build a
            # new one from the exported arrays; lm_shakedown_resume.py proved
            # this continues bit for bit at this shape.
            start = time.perf_counter()
            state = trainer.export_state()
            try:
                trainer.close()
            except Exception:
                pass
            del trainer
            trainer = _new_trainer(Trainer, shape, state['parameters'], state['config'],
                                   state['data_schedule'])
            trainer.load_state_dict(state)
            del state
            elapsed = time.perf_counter() - start
            entry = dict(event='recycle', after_step=index + 1, seconds=elapsed,
                         host=_rss_bytes())
            emit(entry)
            recycles.append(entry)
    sampler.stop()

    timed = [s['seconds'] for s in steps[1:]]
    ordered = sorted(timed)
    head = ordered[:max(1, len(ordered) // 10)]
    result = dict(
        schema=SCHEMA, shape=shape.to_dict(), parameters=shape.n_total,
        arm='recycle' if args.recycle_every else 'straight',
        recycle_every=args.recycle_every, steps_requested=args.steps,
        steps_completed=len(steps), limited=limited,
        first_call_seconds=steps[0]['seconds'] if steps else None,
        median_seconds=ordered[len(ordered) // 2] if ordered else None,
        max_seconds=ordered[-1] if ordered else None,
        fastest_decile_seconds=sum(head) / len(head) if head else None,
        recycles=recycles, recycle_total_seconds=sum(r['seconds'] for r in recycles),
        loss_first=steps[0]['loss'] if steps else None,
        loss_last=steps[-1]['loss'] if steps else None,
        qualification='one GPU, one shape; a drift and mitigation measurement')
    (out / 'result.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in
                      ('arm', 'recycle_every', 'steps_completed', 'limited', 'median_seconds',
                       'max_seconds', 'fastest_decile_seconds', 'recycle_total_seconds',
                       'loss_first', 'loss_last')}, indent=1))
    return 2 if limited else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shape', nargs=9, type=int, default=CONTROL_SHAPE)
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--recycle-every', type=int, default=0,
                        help='0 = one session for the whole run (the straight arm)')
    parser.add_argument('--budget-seconds', type=float, default=1800.0)
    parser.add_argument('--seed', type=int, default=93261)
    parser.add_argument('--corpus', type=Path, default=None)
    parser.add_argument('--sample-interval', type=float, default=0.2)
    parser.add_argument('--gpu-index', type=int, default=0)
    args = parser.parse_args()
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('requires MOJOLEARN_NUMERIC_MODE=identical in the environment')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
