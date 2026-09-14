#!/usr/bin/env python3
"""Cloud-only byte-LM ordered reduction/replay gate. Never run on a laptop."""
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
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('This gate requires a RunPod environment; no local execution')
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Parallel
    shape = Shape(batch=2, length=7, d_model=24, n_heads=3, n_kv=1,
                  head_dim=8, intermediate=40, n_layers=3, vocab_size=256)
    rng = np.random.default_rng(72651)
    p = rng.normal(0, .025, shape.n_total).astype('<f4')
    for row in Trainer.parameter_registry(shape):
        if 'norm' in row['name']:
            p[row['offset']:row['offset'] + row['size']] += np.float32(1)
    corpus = args.corpus.read_bytes()
    seed = Trainer(p, shape=shape, data_schedule={
        'corpus_sha256': hashlib.sha256(corpus).hexdigest(), 'logical_shards': 3})
    initial = seed.state_dict()
    checks = []
    hashes = []
    state_hashes = []

    def batch(step, shard):
        offset = (step * 3 + shard) * shape.batch * (shape.length + 1)
        return np.frombuffer(corpus[offset:offset + shape.batch * (shape.length + 1)],
                             dtype=np.uint8).astype('<i4').reshape(shape.batch, shape.length + 1)

    def same(a, b):
        for name in ('parameters', 'm', 'v', 'flags'):
            assert a[name].tobytes() == b[name].tobytes(), name
        assert a['completed_steps'] == b['completed_steps']

    # K=1 must retain the old single-device result byte for byte.
    with Parallel(initial) as one:
        old = seed.train_step(batch(0, 0))
        new = one.train_step([batch(0, 0)])
        assert np.float32(old['loss']).tobytes() == np.float32(new['losses'][0]).tobytes()
        assert old['flat_gradients'].tobytes() == one.export_gradients().tobytes()
        same(seed.state_dict(), one.state_dict())
    checks.append('K1_matches_existing_step')

    # Odd K exercises a left fold distinct from a power-of-two tree.
    with Parallel(initial, devices=(0, 1), logical_shards=3) as multi, \
         Parallel(initial, devices=(0,), logical_shards=3) as replay:
        for step in range(3):
            shards = [batch(step, rank) for rank in range(3)]
            left = multi.train_step(shards)
            right = replay.train_step(shards)
            assert np.asarray(left['losses'], dtype='<f4').tobytes() == np.asarray(right['losses'], dtype='<f4').tobytes()
            same(multi.state_dict(), replay.state_dict())
            same(multi.state_dict(rank=0), multi.state_dict(rank=1))
            assert multi.export_gradients().tobytes() == replay.export_gradients().tobytes()
            assert multi.export_gradients(rank=0).tobytes() == multi.export_gradients(rank=1).tobytes()
            hashes.append(hashlib.sha256(multi.export_gradients().tobytes()).hexdigest())
            state = multi.state_dict()
            state_hashes.append(hashlib.sha256(b''.join(state[k].tobytes()
                for k in ('parameters', 'm', 'v', 'flags'))).hexdigest())
        # A failure in Python after native success must undo ALL replicas.
        before_post = multi.state_dict()
        native_step = multi._binding.byte_lm_parallel_step
        def refuse_result(*args):
            native_step(*args)
            return []
        multi._binding.byte_lm_parallel_step = refuse_result
        try:
            try:
                multi.train_step([batch(3, rank) for rank in range(3)])
            except RuntimeError as error:
                assert 'loss count' in str(error)
            else:
                raise AssertionError('post-update refusal did not raise')
        finally:
            multi._binding.byte_lm_parallel_step = native_step
        same(before_post, multi.state_dict())
        same(before_post, multi.state_dict(rank=1))
        try:
            multi.export_gradients()
        except Exception as error:
            assert 'no committed gradient' in str(error)
        else:
            raise AssertionError('rolled-back gradients remained exportable')
        checks.append('all_replica_post_update_rollback')
        checkpoint = multi.checkpoint()
        before = multi.state_dict()
        bad = [batch(3, rank) for rank in range(3)]
        bad[-1][0, 0] = -1
        try:
            multi.train_step(bad)
        except Exception as error:
            assert "token ID outside" in str(error), str(error)
        else:
            raise AssertionError('invalid final shard accepted')
        same(before, multi.state_dict())
        with Parallel.from_checkpoint(checkpoint, devices=(1,)) as resumed:
            shards = [batch(3, rank) for rank in range(3)]
            multi.train_step(shards)
            resumed.train_step(shards)
            same(multi.state_dict(), resumed.state_dict())
            assert multi.export_gradients().tobytes() == resumed.export_gradients().tobytes()
    checks += ['K3_two_GPU_matches_one_GPU', 'replicas_match', 'invalid_shard_atomic',
               'checkpoint_resume_on_other_device']
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        gradient_sha256=hashes, state_sha256=state_hashes, corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two devices, one shape. No cross-vendor or throughput claim.'), indent=2) + '\n')
    print(args.report.read_text())


if __name__ == '__main__':
    main()
