#!/usr/bin/env python3
"""Repeated full Byte-LM steps over pinned R2 dataset bytes.

The dataset is consumed as an immutable byte stream.  This deliberately keeps
the performance lane on the same two R2 objects as the classical board while
giving the next-byte trainer a deterministic schedule with no decode or
tokenizer work inside a measured step.
"""
import argparse
import hashlib
import json
import mmap
import statistics
import struct
import time
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class DatasetBytes:
    def __init__(self, path, key, manifest, batch, length):
        self.path, self.key = Path(path), key
        pins = {}
        for line in Path(manifest).read_text().splitlines():
            name, size, digest = line.split('\t')
            pins[name] = (int(size), digest)
        if key not in pins:
            raise ValueError('dataset key absent from manifest: ' + key)
        expected_size, expected_sha = pins[key]
        h = hashlib.sha256()
        with self.path.open('rb') as stream:
            while True:
                chunk = stream.read(8 << 20)
                if not chunk:
                    break
                h.update(chunk)
        actual_size = self.path.stat().st_size
        if (actual_size, h.hexdigest()) != (expected_size, expected_sha):
            raise ValueError('R2 dataset size/SHA mismatch: ' + key)
        self.size, self.digest = actual_size, expected_sha
        self.stream = self.path.open('rb')
        self.data = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.batch, self.length = batch, length
        self.modulus = self.size - length - 1

    def ids(self, step):
        import numpy as np
        rows = []
        for row in range(self.batch):
            # Fixed integer strides scatter repeated steps over the full
            # 400 MiB / 2.1 GiB objects instead of timing a tiny ZIP prefix.
            start = (step * 2654435761 + row * 2246822519) % self.modulus
            rows.append(np.frombuffer(self.data, dtype=np.uint8, count=self.length + 1,
                                      offset=start).astype(np.int32))
        return np.stack(rows)

    def close(self):
        self.data.close()
        self.stream.close()


def digest_step(trainer, loss):
    state = trainer.export_state()
    grad = trainer.export_gradients(named=False)['flat_gradients']
    return dict(
        loss=sha(struct.pack('<f', loss)),
        gradients=sha(grad.tobytes()),
        parameters=sha(state['parameters'].tobytes()),
        m=sha(state['m'].tobytes()),
        v=sha(state['v'].tobytes()),
        flags=sha(state['flags'].tobytes()),
        completed_steps=int(state['completed_steps']),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--dataset-key', required=True)
    p.add_argument('--manifest', type=Path,
                   default=Path('bench/results/dataset_store/manifest.tsv'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shape', type=int, nargs=9,
                   default=[1, 2048, 768, 12, 12, 64, 2048, 12, 50257])
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--witness-steps', type=int, default=3)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--samples', type=int, default=9)
    args = p.parse_args()

    import numpy as np
    from mojolearn import LanguageModelConfig as Shape, LanguageModelTrainer as Trainer
    from mojolearn import _backend

    shape = Shape(*args.shape)
    data = DatasetBytes(args.dataset, args.dataset_key, args.manifest,
                        shape.batch, shape.length)
    rng = np.random.default_rng(args.seed)
    parameters = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for row in Trainer.parameter_registry(shape):
        if 'norm' in row['name']:
            parameters[row['offset']:row['offset'] + row['size']] += np.float32(1)
    trainer = Trainer(parameters, shape=shape, resident=True, step_result='lean',
                      data_schedule=dict(dataset_key=args.dataset_key,
                                         dataset_sha256=data.digest,
                                         schedule='raw bytes, full-object strided next-byte windows'))
    binding = _backend.binding('_mojolearn_byte_lm', 'identical')
    arm = int(binding.byte_lm_flat_view_arm())
    witnesses, times = [], []
    total = args.witness_steps + args.warmup + args.samples
    for step in range(total):
        t0 = time.perf_counter_ns()
        result = trainer.train_step(data.ids(step))
        elapsed = (time.perf_counter_ns() - t0) / 1e9
        if step < args.witness_steps:
            witnesses.append(digest_step(trainer, float(result['loss'])))
        elif step >= args.witness_steps + args.warmup:
            times.append(elapsed)
    state = trainer.export_state()
    result = dict(
        schema='mojolearn.lm-flat-views.v1', arm=arm,
        dataset=dict(key=args.dataset_key, path=str(args.dataset), bytes=data.size,
                     sha256=data.digest), shape=list(args.shape), seed=args.seed,
        schedule='step k row b: dataset bytes at (k*2654435761+b*2246822519) modulo (size-length-1); uint8 to int32; target shifted one byte',
        witnesses=witnesses, post_swap_witness=(witnesses[1] if len(witnesses) > 1 else None),
        seconds=times, median_seconds=(statistics.median(times) if times else None),
        min_seconds=(min(times) if times else None),
        max_seconds=(max(times) if times else None),
        completed_steps=int(state['completed_steps']), run_metadata=trainer.run_metadata(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    trainer.close()
    data.close()
    print(json.dumps(dict(status='PASS', arm=arm, dataset=args.dataset_key,
                          median_seconds=result['median_seconds'], samples=len(times))))


if __name__ == '__main__':
    main()
