#!/usr/bin/env python3
"""Cloud-only measured one-device refusal versus pooled model capacity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--mode', choices=('one', 'pooled'), required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Parallel
    from mojolearn.model_pool_training import PooledByteLanguageModelTrainer as Pooled
    shape = Shape(batch=1, length=1, d_model=1536, n_heads=24, n_kv=6,
                  head_dim=64, intermediate=6144, n_layers=28, vocab_size=256)
    print('capacity fixture', args.mode, shape.n_total, flush=True)
    parameters = np.full(shape.n_total, np.float32(1 / 4096), dtype='<f4')
    for row in Trainer.parameter_registry(shape):
        if 'norm' in row['name']:
            parameters[row['offset']:row['offset'] + row['size']] = np.float32(1)
    seed = Trainer(parameters, shape=shape)
    state = seed.state_dict()
    del parameters, seed
    with args.corpus.open('rb') as source:
        tokens = np.frombuffer(source.read(2), dtype=np.uint8).astype('<i4').reshape(1, 2)
    hardware = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv'], text=True)
    start = time.monotonic()
    trainer = (Parallel(state, devices=(0,), logical_shards=1, pool_optimizer=False)
               if args.mode == 'one' else Pooled(state, devices=(0, 1), logical_shards=1))
    del state
    result = dict(mode=args.mode, parameters=shape.n_total, hardware=hardware,
                  shape=list(shape.native_shape), tokens=tokens.tolist())
    try:
        step = trainer.train_step([tokens])
    except Exception as error:
        message = str(error)
        result.update(status='REFUSED', error=message, elapsed_seconds=time.monotonic()-start)
        args.report.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if args.mode != 'one' or not any(term in message.lower() for term in
                                        ('out of memory', 'out_of_memory', 'memory allocation')):
            raise
    else:
        result.update(status='PASS', step=step, elapsed_seconds=time.monotonic()-start)
        if args.mode == 'pooled':
            ownership = trainer.model_ownership()
            assert sum(row['count'] for row in ownership) == shape.n_total
            result['ownership'] = ownership
            result['resident_memory'] = subprocess.check_output([
                'nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv'], text=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if args.mode == 'one':
            raise AssertionError('one-device fixture fits; beyond-device capacity is not established')
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
