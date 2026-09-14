#!/usr/bin/env python3
"""RunPod-only neural/forest identity gate; training rows come from staged R2."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', action='store_true', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--lane', choices=('mlp', 'samba', 'forest'), required=True)
    p.add_argument('--attention-dropout', action='store_true')
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod environment required; no local execution')
    import numpy as np
    from mojolearn.parallel_training import ParallelNeuralTrainer
    from mojolearn.parallel_ensemble import fit_forest
    source = args.corpus.read_bytes()
    data = np.frombuffer(source[:4096], dtype=np.uint8)
    checks = []
    hashes = []
    state_hashes = []

    def state_bytes(value):
        def frame(tag, parts):
            return tag + b''.join(len(p).to_bytes(8, 'little') + p for p in parts)
        if hasattr(value, 'tobytes'):
            return frame(b'A', [str((value.shape, value.dtype)).encode(), value.tobytes()])
        if isinstance(value, dict):
            return frame(b'D', [frame(b'P', [str(k).encode(), state_bytes(value[k])]) for k in sorted(value)])
        if isinstance(value, (list, tuple)):
            return frame(b'L', [state_bytes(v) for v in value])
        return json.dumps(value, sort_keys=True, allow_nan=False).encode()

    def equal(a, b):
        if hasattr(a, 'tobytes'):
            assert a.tobytes() == b.tobytes()
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b, (a, b)

    if args.lane == 'mlp':
        from mojolearn.neural_network import SmallMLPTrainer
        rng = np.random.default_rng(946)
        weights = [rng.normal(0, .03, shape).astype('<f4')
                   for shape in ((16, 8), (16,), (3, 16), (3,))]
        model = SmallMLPTrainer(*weights, data_schedule={'fixture': 'r2-parallel'})
        replay_model = SmallMLPTrainer(*weights, data_schedule={'fixture': 'r2-parallel'})
        shards = [(data[i*64:(i+1)*64].astype('<f4').reshape(8, 8) / np.float32(255),
                   (data[i*8:(i+1)*8] % 3).astype('<i4')) for i in range(3)]
        with ParallelNeuralTrainer(model, devices=(0, 1), logical_shards=3) as multi, \
             ParallelNeuralTrainer(replay_model, devices=(0,), logical_shards=3) as replay:
            for _ in range(3):
                equal(multi.train_step(shards), replay.train_step(shards))
                equal(model.state_dict(), replay_model.state_dict())
                equal(multi.export_gradients(), replay.export_gradients())
                hashes.append(hashlib.sha256(b''.join(g.tobytes() for g in multi.export_gradients())).hexdigest())
                state_hashes.append(hashlib.sha256(state_bytes(multi.model.state_dict())).hexdigest())
            before = model.state_dict()
            bad = list(shards)
            bad[-1] = (bad[-1][0], np.full(8, -1, dtype='<i4'))
            try:
                multi.train_step(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('bad labels accepted')
            equal(before, model.state_dict())
        checks += ['MLP_K3_two_GPU_replay', 'MLP_failure_atomic']
        # A single logical shard must preserve the existing train_step.
        left = SmallMLPTrainer(*weights, data_schedule={'fixture': 'r2-parallel'})
        right = SmallMLPTrainer(*weights, data_schedule={'fixture': 'r2-parallel'})
        with ParallelNeuralTrainer(left) as parallel:
            parallel.train_step([shards[0]])
            right.train_step(*shards[0])
            equal(left.state_dict(), right.state_dict())
        checks.append('MLP_K1_existing_step')
        checkpoint = multi.checkpoint()
        with ParallelNeuralTrainer.from_checkpoint(checkpoint, devices=(1,)) as resumed, \
             ParallelNeuralTrainer(replay_model, devices=(0,), logical_shards=3) as replay:
            equal(resumed.train_step(shards), replay.train_step(shards))
            equal(resumed.model.state_dict(), replay_model.state_dict())
        checks.append('MLP_checkpoint_device_migration')
    elif args.lane == 'forest':
        from mojolearn import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor)
        X = data[:1024].astype('<f4').reshape(128, 8) / np.float32(255)
        for cls in (RandomForestClassifier, RandomForestRegressor,
                    ExtraTreesClassifier, ExtraTreesRegressor):
            y = ((data[1024:1152] % 3).astype('<i4') if 'Classifier' in cls.__name__
                 else data[1024:1152].astype('<f4') / np.float32(255))
            params = dict(n_estimators=5, max_depth=4, random_state=217, numeric_mode='identical')
            serial = cls(**params).fit(X, y)
            parallel = fit_forest(cls(**params), X, y, devices=(0, 1), trees_per_shard=2)
            for name in ('_offsets', '_colid', '_quesval', '_left_child', '_leaves'):
                equal(getattr(serial, name), getattr(parallel, name))
            equal(serial.predict(X), parallel.predict(X))
            if hasattr(serial, 'predict_proba'):
                equal(serial.predict_proba(X), parallel.predict_proba(X))
            hashes.append(hashlib.sha256(b''.join(getattr(parallel, name).tobytes() for name in
                ('_offsets', '_colid', '_quesval', '_left_child', '_leaves'))).hexdigest())
            checks.append(cls.__name__ + '_two_GPU_equals_serial')
    else:
        from mojolearn.training import SambaStack, SambaConfig, Generator
        config = (SambaConfig(vocab=256, d_model=32, layers=('mamba3', 'attention'),
                              n_heads=4, n_kv_heads=2, head_dim=8, intermediate=64, dropout=.1)
                  if args.attention_dropout else
                  SambaConfig(vocab=256, d_model=32, layers=('mamba3',), dropout=0))
        left = SambaStack(config, generator=Generator(946, 'identical'), numeric_mode='identical')
        right = SambaStack(config, generator=Generator(946, 'identical'), numeric_mode='identical')
        shards = [(data[i*16:(i+1)*16].astype('<i4').reshape(2, 8),
                   data[i*16+1:(i+1)*16+1].astype('<i4').reshape(2, 8)) for i in range(3)]
        with ParallelNeuralTrainer(left, devices=(0, 1), logical_shards=3) as multi, \
             ParallelNeuralTrainer(right, devices=(0,), logical_shards=3) as replay:
            for _ in range(2):
                equal(multi.train_step(shards), replay.train_step(shards))
                equal(left.state_dict(), right.state_dict())
                equal(multi.export_gradients(), replay.export_gradients())
                hashes.append(hashlib.sha256(b''.join(g.tobytes() for g in multi.export_gradients())).hexdigest())
                state_hashes.append(hashlib.sha256(state_bytes(multi.model.state_dict())).hexdigest())
        checks.append('Samba_attention_dropout_K3_two_GPU_replay' if args.attention_dropout
                      else 'Samba_K3_two_GPU_replay')
        checkpoint = multi.checkpoint()
        with ParallelNeuralTrainer.from_checkpoint(checkpoint, devices=(1,)) as resumed, \
             ParallelNeuralTrainer(right, devices=(0,), logical_shards=3) as replay:
            equal(resumed.train_step(shards), replay.train_step(shards))
            equal(resumed.model.state_dict(), right.state_dict())
            equal(resumed.export_gradients(), replay.export_gradients())
        checks.append('Samba_checkpoint_device_migration')
    report = dict(status='PASS', checks=checks, hashes=hashes, state_hashes=state_hashes,
                  corpus_sha256=hashlib.sha256(source).hexdigest(),
                  scope='Two devices; no cross-vendor or speed claim')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(args.report.read_text())


if __name__ == '__main__':
    main()
