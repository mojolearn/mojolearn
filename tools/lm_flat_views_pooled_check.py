#!/usr/bin/env python3
"""One-device reach gate for flat views with pooled optimizer ownership."""
import argparse
import hashlib
import json
from pathlib import Path


def same(a, b):
    for key in ('parameters', 'm', 'v', 'flags'):
        assert a[key].tobytes() == b[key].tobytes(), key
    assert a['completed_steps'] == b['completed_steps']
    assert a['next_batch_index'] == b['next_batch_index']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from mojolearn import LanguageModelConfig as Shape, LanguageModelTrainer as Trainer
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Parallel

    with args.dataset.open('rb') as stream:
        raw = stream.read(4096)
    shape = Shape(2, 31, 32, 4, 2, 8, 64, 2, 256)
    rng = np.random.default_rng(20260921)
    params = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for row in Trainer.parameter_registry(shape):
        if 'norm' in row['name']:
            params[row['offset']:row['offset'] + row['size']] += np.float32(1)
    initial = Trainer(params, shape=shape,
                      data_schedule={'dataset_prefix_sha256': hashlib.sha256(raw).hexdigest()}).state_dict()

    def batches(step):
        rows = []
        for shard in range(2):
            start = (step * 2 + shard) * shape.batch * shape.length
            block = np.frombuffer(raw, dtype=np.uint8, count=shape.batch * (shape.length + 1),
                                  offset=start).astype(np.int32)
            rows.append(block.reshape(shape.batch, shape.length + 1))
        return rows

    with Parallel(initial, devices=(0,), logical_shards=2, pool_optimizer=True) as pooled, \
         Parallel(initial, devices=(0,), logical_shards=2, pool_optimizer=False) as replicated:
        assert pooled.optimizer_ownership()[0]['count'] == shape.n_total
        checks = []
        for step in range(3):
            left, right = pooled.train_step(batches(step)), replicated.train_step(batches(step))
            assert np.asarray(left['losses'], np.float32).tobytes() == np.asarray(right['losses'], np.float32).tobytes()
            same(pooled.state_dict(), replicated.state_dict())
            assert pooled.export_gradients().tobytes() == replicated.export_gradients().tobytes()
            checks.append('step%d exact pooled/replicated state+gradient' % (step + 1))
        before = pooled.state_dict()
        original = pooled._binding.byte_lm_parallel_step

        def reject(*values):
            original(*values)
            return []

        pooled._binding.byte_lm_parallel_step = reject
        try:
            try:
                pooled.train_step(batches(3))
            except RuntimeError as error:
                assert 'loss count' in str(error)
            else:
                raise AssertionError('post-update pooled failure was accepted')
        finally:
            pooled._binding.byte_lm_parallel_step = original
        same(before, pooled.state_dict())
        checks.append('post-update pooled rollback exact')
        final = pooled.state_dict()
    args.out.write_text(json.dumps(dict(status='PASS', checks=checks,
        state_sha256=hashlib.sha256(b''.join(final[k].tobytes() for k in
                                             ('parameters', 'm', 'v', 'flags'))).hexdigest()),
        indent=1) + '\n')
    print(args.out.read_text(), end='')


if __name__ == '__main__':
    main()
