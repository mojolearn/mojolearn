#!/usr/bin/env python3
"""Cloud-only 958.7M-parameter pooled/offloaded state and gradient witness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--mode', choices=('pooled', 'offloaded'), required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--logical-shards', type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    from mojolearn.model_pool_training import PooledByteLanguageModelTrainer as Pooled
    from mojolearn.offload_training import OffloadedByteLanguageModelTrainer as Offloaded
    shape = Shape(batch=1, length=1, d_model=1536, n_heads=24, n_kv=6,
                  head_dim=64, intermediate=6144, n_layers=28, vocab_size=256)
    print('capacity fixture', args.mode, shape.n_total, flush=True)
    p = np.full(shape.n_total, np.float32(1/4096), dtype='<f4')
    for row in Trainer.parameter_registry(shape):
        if 'norm' in row['name']:
            p[row['offset']:row['offset'] + row['size']] = np.float32(1)
    corpus = args.corpus.read_bytes()
    seed = Trainer(p, shape=shape, data_schedule={'corpus_sha256': hashlib.sha256(corpus).hexdigest()})
    state = seed.state_dict()
    del p, seed
    trainer = (Pooled(state, devices=(0,1), logical_shards=args.logical_shards)
               if args.mode == 'pooled' else Offloaded(state, devices=(0,), logical_shards=args.logical_shards))
    del state
    peak = []
    samples = []
    stopped = threading.Event()

    def monitor():
        while not stopped.is_set():
            try:
                values = [int(v.strip()) for v in subprocess.check_output([
                    'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
                samples.append(values)
            except (subprocess.SubprocessError, ValueError):
                pass
            stopped.wait(.2)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    receipts = []
    started = time.monotonic()
    try:
        for step in range(2):
            batches = [np.frombuffer(corpus[(step*args.logical_shards+i)*2:(step*args.logical_shards+i)*2+2],
                                     dtype=np.uint8).astype('<i4').reshape(1,2)
                       for i in range(args.logical_shards)]
            result = trainer.train_step(batches)
            state = trainer.state_dict()
            hashes = {key: hashlib.sha256(memoryview(state[key]).cast('B')).hexdigest()
                      for key in ('parameters','m','v','flags')}
            del state
            gradient = trainer.export_gradients()
            hashes['gradient'] = hashlib.sha256(memoryview(gradient).cast('B')).hexdigest()
            del gradient
            hashes['losses'] = hashlib.sha256(np.asarray(result['losses'], '<f4').tobytes()).hexdigest()
            receipts.append(dict(step=result, hashes=hashes))
            print('completed', step+1, hashes, flush=True)
    finally:
        stopped.set()
        watcher.join()
        trainer.close()
    if not samples:
        raise AssertionError('no device-memory samples captured')
    peak = [max(row[i] for row in samples) for i in range(len(samples[0]))]
    if args.mode == 'offloaded' and peak[0] >= 32*1024:
        raise AssertionError('offloaded replay exceeded 32 GiB device memory')
    args.report.write_text(json.dumps(dict(status='PASS', mode=args.mode,
        parameters=shape.n_total, shape=list(shape.native_shape), logical_shards=args.logical_shards,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(), receipts=receipts,
        sampled_peak_mib=peak, samples=len(samples), elapsed_seconds=time.monotonic()-started,
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True),
        scope='Capacity/identity fixture with uniform initialization; no learning-quality or throughput claim.'), indent=2)+'\n')
    print(args.report.read_text(), flush=True)


if __name__ == '__main__':
    main()
