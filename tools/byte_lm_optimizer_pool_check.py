#!/usr/bin/env python3
"""Cloud-only optimizer ownership, exact replay and transaction gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--faults', action='store_true')
    parser.add_argument('--logical-shards', type=int, choices=(2, 3, 5, 8), default=3)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Parallel
    corpus = args.corpus.read_bytes()
    checks = []

    def same(a, b):
        for key in ('parameters', 'm', 'v', 'flags'):
            assert a[key].tobytes() == b[key].tobytes(), key
        assert a['completed_steps'] == b['completed_steps']
        assert a['next_batch_index'] == b['next_batch_index']

    for layers, dm, heads, hidden in ((1, 16, 2, 24), (3, 24, 3, 40)):
        shape = Shape(batch=2, length=7, d_model=dm, n_heads=heads, n_kv=1,
                      head_dim=8, intermediate=hidden, n_layers=layers, vocab_size=256)
        rng = np.random.default_rng(86352 + layers)
        p = rng.normal(0, .025, shape.n_total).astype('<f4')
        for row in Trainer.parameter_registry(shape):
            if 'norm' in row['name']:
                p[row['offset']:row['offset'] + row['size']] += np.float32(1)
        seed = Trainer(p, shape=shape, data_schedule={'corpus_sha256': hashlib.sha256(corpus).hexdigest()})
        initial = seed.state_dict()
        assert Parallel(initial).pool_optimizer is True
        # Exercise nonzero restored moments, bias correction and decay.
        initial['m'] = rng.normal(0, .001, shape.n_total).astype('<f4')
        initial['v'] = rng.uniform(.0001, .001, shape.n_total).astype('<f4')
        initial['completed_steps'] = initial['next_batch_index'] = 7
        initial['config']['weight_decay'] = .07

        def batches(step):
            result = []
            for shard in range(args.logical_shards):
                start = (step * args.logical_shards + shard) * 16
                result.append(np.frombuffer(corpus[start:start+16], dtype=np.uint8)
                              .astype('<i4').reshape(2, 8))
            return result

        with Parallel(initial, devices=(0, 1), logical_shards=args.logical_shards, pool_optimizer=True) as pool, \
             Parallel(initial, devices=(0, 1), logical_shards=args.logical_shards, pool_optimizer=False) as replicas, \
             Parallel(initial, devices=(1,), logical_shards=args.logical_shards, pool_optimizer=False) as replay:
            ownership = pool.optimizer_ownership()
            for rank, row in enumerate(ownership):
                first = shape.n_total * rank // 2
                count = shape.n_total * (rank + 1) // 2 - first
                assert row == dict(device=rank, first=first, count=count,
                                   moment_bytes=8*count, rollback_bytes=12*count, reduction_bytes=8*count)
            assert sum(x['moment_bytes'] + x['rollback_bytes'] for x in ownership) == 20*shape.n_total
            assert sum(x['moment_bytes'] + x['rollback_bytes'] for x in replicas.optimizer_ownership()) == 40*shape.n_total
            assert sum(x['reduction_bytes'] for x in ownership) == 8*shape.n_total
            replicated = replicas.optimizer_ownership()
            # The replicated path now aliases its incoming staging buffer to
            # rank zero's dead gradient after that rank enters the fold, so
            # only the distinct total accumulator remains allocated.
            assert replicated[0]['reduction_bytes'] == 4*shape.n_total
            assert replicated[1]['reduction_bytes'] == 0
            same(initial, pool.state_dict(rank=1))
            for step in range(3):
                a = pool.train_step(batches(step))
                b = replicas.train_step(batches(step))
                c = replay.train_step(batches(step))
                assert np.asarray(a['losses'], '<f4').tobytes() == np.asarray(b['losses'], '<f4').tobytes() == np.asarray(c['losses'], '<f4').tobytes()
                same(pool.state_dict(), replicas.state_dict())
                same(pool.state_dict(), replay.state_dict())
                same(pool.state_dict(), pool.state_dict(rank=1))
                assert pool.export_gradients().tobytes() == replicas.export_gradients().tobytes() == replay.export_gradients().tobytes()
            before = pool.state_dict()
            # Failure AFTER a successful native update must restore every shard.
            original = pool._binding.byte_lm_parallel_step
            def refuse_result(*values):
                original(*values)
                return []
            pool._binding.byte_lm_parallel_step = refuse_result
            try:
                try:
                    pool.train_step(batches(3))
                except RuntimeError as error:
                    assert 'loss count' in str(error)
                else:
                    raise AssertionError('post-update failure was accepted')
            finally:
                pool._binding.byte_lm_parallel_step = original
            for rank in (0, 1):
                same(before, pool.state_dict(rank=rank))
            try:
                pool.export_gradients()
            except Exception as error:
                assert 'no committed gradient' in str(error)
            else:
                raise AssertionError('rollback left gradients committed')
            if args.faults:
                assert pool._binding.byte_lm_pool_fault_available()
                for fault in ('grad_nonfinite', 'opt_refuse', 'after_nonfinite', 'after_negative'):
                    os.environ['MOJOLEARN_BYTE_POOL_FAULT'] = fault
                    os.environ['MOJOLEARN_BYTE_POOL_FAULT_FIRST'] = str(ownership[1]['first'])
                    try:
                        try:
                            pool.train_step(batches(3))
                        except Exception:
                            pass
                        else:
                            raise AssertionError('native fault failed to refuse: ' + fault)
                    finally:
                        os.environ.pop('MOJOLEARN_BYTE_POOL_FAULT', None)
                        os.environ.pop('MOJOLEARN_BYTE_POOL_FAULT_FIRST', None)
                    for rank in (0, 1):
                        same(before, pool.state_dict(rank=rank))
            invalid = batches(3)
            invalid[-1][0, 0] = -1
            try:
                pool.train_step(invalid)
            except Exception as error:
                assert 'token ID outside' in str(error)
            else:
                raise AssertionError('invalid final shard accepted')
            same(before, pool.state_dict())
            checkpoint = pool.checkpoint()
            with Parallel.from_checkpoint(checkpoint, devices=(1,), pool_optimizer=True) as resumed:
                pool.train_step(batches(3))
                resumed.train_step(batches(3))
                replay.train_step(batches(3))
                same(pool.state_dict(), resumed.state_dict())
                same(pool.state_dict(), replay.state_dict())
            state = pool.state_dict()
            checks.append(dict(layers=layers, logical_shards=args.logical_shards, parameters=shape.n_total, ownership=ownership,
                state_sha256=hashlib.sha256(b''.join(state[k].tobytes() for k in ('parameters','m','v','flags'))).hexdigest(),
                native_faults=args.faults))
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        scope='Two GPUs: optimizer moment/rollback and gradient-reduction pooling, replay, exports and recovery. Model weights and activations remain replicated.'), indent=2) + '\n')
    print(args.report.read_text())


if __name__ == '__main__':
    main()
