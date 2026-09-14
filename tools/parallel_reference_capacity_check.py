#!/usr/bin/env python3
"""Cloud-only large logical reference index, with an exact small oracle."""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', required=True, action='store_true')
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--reference-gib', type=int, default=96)
    p.add_argument('--shard-gib', type=int, default=6)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    if not 1 <= args.shard_gib <= 6 or args.reference_gib < 1:
        raise SystemExit('positive reference GiB and 1..6 shard GiB required')
    import numpy as np
    from mojolearn import NearestNeighbors
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    corpus = args.corpus.read_bytes()
    d = 256
    rows = args.reference_gib * 1024**3 // (d * 4)
    shard_rows = args.shard_gib * 1024**3 // (d * 4)
    raw = np.frombuffer(corpus[:64*d], dtype=np.uint8)
    base = np.float32(1) + raw.astype('<f4').reshape(64, d) / np.float32(255)
    Q = np.zeros((1, d), dtype='<f4')
    # Repeating 64 rows creates tied minima. Four repetitions supply every
    # possible winner for k=5 once the last, zero-distance sentinel is added.
    small = np.vstack((np.tile(base, (4, 1)), np.zeros((1, d), dtype='<f4')))
    expected_dist, expected_idx = NearestNeighbors(n_neighbors=5, numeric_mode='identical').fit(small).kneighbors(Q)
    expected_indices = np.asarray(expected_idx).copy()
    expected_indices[expected_indices == 256] = rows-1
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], text=True)
    capacities = [int(line.rsplit(',', 1)[1].strip()) * 1024**2 for line in gpu_info.strip().splitlines()]
    assert len(capacities) >= 2
    if args.reference_gib >= 96:
        assert rows*d*4 > max(capacities), 'logical reference must exceed every individual GPU'
    print('Allocate logical reference bytes', rows*d*4, 'shard bytes', shard_rows*d*4, flush=True)
    X = np.empty((rows, d), dtype='<f4')
    X.reshape((-1, 64, d))[:] = base
    X[-1] = np.float32(0)
    print('Reference initialized; begin GPU query', flush=True)
    model = NearestNeighbors(n_neighbors=5, numeric_mode='identical').fit(X)
    assert model._index.nbytes == X.nbytes
    started = time.monotonic()
    with ReferenceShardedNeighbors(model, devices=(0, 1), reference_rows_per_shard=shard_rows,
                                  query_rows_per_shard=1) as driver:
        actual_dist, actual_idx = driver.kneighbors(Q)
        assert actual_dist.tobytes() == expected_dist.tobytes()
        assert actual_idx.tobytes() == expected_indices.astype('<i8').tobytes()
        assert driver.last_shards_[0]['reference_start'] == 0
        assert driver.last_shards_[-1]['reference_end'] == rows
        assert {part['device'] for part in driver.last_shards_} == {0, 1}
        assert all(part['reference_bytes'] <= shard_rows*d*4 for part in driver.last_shards_)
        parts = driver.last_shards_.copy()
    result = dict(status='PASS', logical_reference_bytes=X.nbytes, shape=[rows,d],
                  max_reference_shard_bytes=shard_rows*d*4, devices=[0,1], shards=parts,
                  individual_gpu_bytes=capacities, gpu_info=gpu_info,
                  seconds=time.monotonic()-started, parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  indices=actual_idx.tolist(), distance_bytes=actual_dist.tobytes().hex(),
                  output_sha256=hashlib.sha256(actual_dist.tobytes()+actual_idx.tobytes()).hexdigest(),
                  corpus_sha256=hashlib.sha256(corpus).hexdigest(),
                  construction='64 corpus-derived rows repeated, final row zero, zero query, k=5; exact 257-row GPU oracle',
                  scope='Host-staged reference partitions; logical index exceeds one GPU for 96 GiB run. Not an all-resident VRAM pool or speedup measurement.')
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print('PASS reference capacity', result['logical_reference_bytes'], result['output_sha256'], flush=True)


if __name__ == '__main__':
    main()
