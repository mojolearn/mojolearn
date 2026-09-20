#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-device gradient accumulation at the target LM shape.

`completed_steps` is refused at 1,000,000 by the host, the binding and the
Mojo trainer alike, so at batch 1 and length 2048 the whole stack tops out at
2.048 BILLION tokens however long a box is rented. The way past that is more
tokens per OPTIMIZER step, and `SmallByteLanguageModelTrainer.train_step` has
no accumulation of its own.

`ParallelByteLanguageModelTrainer` does. It takes `logical_shards` microbatches
per `train_step`, sums their gradients in a fixed order and advances the
optimizer ONCE (`parallel_training.py:89`, `completed_steps = before + 1`), and
with `devices=(0,)` it replays every shard on one GPU. `training/byte_lm_parallel.mojo`
builds one `ByteTrainer` per DEVICE, not per shard, so the extra device cost of
K shards on one device should be the two n_total accumulator buffers and
nothing that scales with K. This probe measures whether that holds, and what
K does to throughput, at 162,147,840 parameters.

Everything here has only ever run at toy shapes (the parallel checks use
b2-l7-d24). Nothing about this is qualification; it is a capacity and
throughput measurement on one GPU.

Note the reduction: shard gradients are SUMMED, not averaged, so K shards give
a gradient K times a single shard's. That is a learning-rate question for a
real run, not a determinism one, and it is recorded here rather than adjusted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_step_memory_probe import (CONTROL_SHAPE, TARGET_SHAPE, CorpusBatches,  # noqa: E402
                                  DeviceMemorySampler, _rss_bytes)

SCHEMA = 'mojolearn.lm-shards-probe.v1'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def run(args):
    import numpy as np
    from mojolearn import (LanguageModelTrainer as Trainer, LanguageModelConfig as Shape,
                           ParallelByteLanguageModelTrainer as Parallel)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape_args = TARGET_SHAPE if args.target else args.shape
    shape = Shape(*shape_args)
    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if 'norm' in entry['name']:
            weights[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    seed_trainer = Trainer(weights, shape=shape, resident=False,
                           data_schedule={'fixture': 'lm shards probe', 'seed': args.seed})
    state = seed_trainer.state_dict()

    corpus = CorpusBatches(args.corpus, shape.batch, shape.length) if args.corpus else None

    def microbatch(index):
        if corpus is not None:
            return corpus.ids(index)
        return rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)

    results = []
    for shards in args.shards:
        record = dict(schema=SCHEMA, shape=shape.to_dict(), parameters=shape.n_total,
                      logical_shards=shards, devices=list(args.devices),
                      tokens_per_optimizer_step=shape.batch * shape.length * shards,
                      corpus=corpus.describe() if corpus is not None else None,
                      reduction='ordered_sum of per-shard mean cross-entropies')
        sampler = DeviceMemorySampler(args.sample_interval, args.gpu_index)
        sampler.start()
        try:
            t0 = time.perf_counter()
            trainer = Parallel(state, devices=tuple(args.devices), logical_shards=shards)
            record['construct_seconds'] = time.perf_counter() - t0
            steps = []
            index = 0
            for step in range(args.steps):
                batch = [microbatch(index + s) for s in range(shards)]
                index += shards
                sampler.window_reset()
                start = time.perf_counter()
                result = trainer.train_step(batch)
                seconds = time.perf_counter() - start
                steps.append(dict(
                    step=step + 1, seconds=seconds,
                    tokens_per_second=shape.batch * shape.length * shards / seconds,
                    first_call_includes_setup=(step == 0),
                    losses_sha256=_sha(np.array(result['losses'], np.float32).tobytes()),
                    completed_steps=result['completed_steps'],
                    reduction=result['reduction'],
                    host=_rss_bytes(), device=sampler.window_report()))
                print(json.dumps(dict(shards=shards, **{k: steps[-1][k] for k in
                                                        ('step', 'seconds', 'tokens_per_second')})), flush=True)
            record['steps'] = steps
            steady = sorted(s['seconds'] for s in steps[1:]) or [steps[0]['seconds']]
            record['steady_median_seconds'] = steady[len(steady) // 2]
            record['steady_median_tokens_per_second'] = (
                shape.batch * shape.length * shards / record['steady_median_seconds'])
            export_start = time.perf_counter()
            final_state = trainer.state_dict()
            final_grad = trainer.export_gradients()
            record['export_seconds'] = time.perf_counter() - export_start
            record['final_sha256'] = {
                key: _sha(np.asarray(final_state[key]).tobytes())
                for key in ('parameters', 'm', 'v', 'flags')
            }
            record['final_sha256']['gradients'] = _sha(
                np.asarray(final_grad).tobytes())
            record['optimizer_ownership'] = trainer.optimizer_ownership()
            record['refused'] = None
            try:
                trainer.close()
            except Exception:
                pass
        except BaseException as exc:
            # A refusal at K is a RESULT. Record it by name and keep going.
            record['refused'] = '%s: %s' % (type(exc).__name__, exc)
            record['steps'] = record.get('steps', [])
            print(json.dumps(dict(shards=shards, refused=record['refused'])), flush=True)
        finally:
            sampler.stop()
            # window_reset() clears the sampler's peaks before every step, so
            # the arm's peak is the maximum of the per-step windows, not
            # whatever the sampler happens to hold at the end.
            def _peak(key):
                values = [s['device'].get(key) for s in record.get('steps', [])]
                values = [v for v in values if isinstance(v, int)]
                return max(values) if values else 'unavailable'
            record['peak_device_process_bytes'] = _peak('peak_process_bytes')
            record['peak_device_wide_bytes'] = _peak('peak_device_bytes')
            record['device_tool'] = sampler.tool or 'unavailable'
        results.append(record)
        (out / 'result.json').write_text(json.dumps(
            dict(schema=SCHEMA, shape=shape.to_dict(), parameters=shape.n_total,
                 steps_requested=args.steps, seed=args.seed, arms=results,
                 qualification='one GPU, one shape; capacity and throughput only, '
                               'not an opponent ratio and not a default gate'), indent=1))
    print(json.dumps({r['logical_shards']: (r['refused'] or r.get('steady_median_tokens_per_second'))
                      for r in results}, indent=1))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shape', nargs=9, type=int, default=CONTROL_SHAPE)
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--shards', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=93261)
    parser.add_argument('--devices', nargs='+', type=int, default=[0],
                        help='ordered physical device ids; compare separate runs with 0 and 0 1')
    parser.add_argument('--corpus', type=Path, default=None)
    parser.add_argument('--sample-interval', type=float, default=0.2)
    parser.add_argument('--gpu-index', type=int, default=0)
    args = parser.parse_args()
    if (not args.devices or any(i < 0 for i in args.devices)
            or len(set(args.devices)) != len(args.devices)):
        parser.error('--devices must be distinct nonnegative integers')
    import os
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('requires MOJOLEARN_NUMERIC_MODE=identical in the environment')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
