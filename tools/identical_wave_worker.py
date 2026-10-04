#!/usr/bin/env python3
"""One source-built board invocation; host identity never starts a timer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def source_provenance(root, expected_vendor):
    import mojolearn as ml
    package = Path(ml.__file__).resolve()
    if not package.is_relative_to(root / 'python/mojolearn'):
        raise RuntimeError('refused non-source mojolearn: ' + str(package))
    vendor = ml.vendor()
    if vendor != expected_vendor:
        raise RuntimeError(f'vendor {vendor!r} != expected {expected_vendor!r}')
    paths = set()
    maps = Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            tail = line.split()[-1]
            if '_mojolearn' in tail and '.so' in tail:
                p = Path(tail).resolve()
                if not p.is_relative_to(root / 'python/mojolearn'):
                    raise RuntimeError('refused external binding: ' + str(p))
                paths.add(str(p))
    if not paths:
        raise RuntimeError('no loaded source binding mappings: provenance unverified')
    return {'package': str(package), 'vendor': vendor, 'binding_files': sorted(paths)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--lane', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--operation', choices=('identity', 'timing'), required=True)
    p.add_argument('--vendor', choices=('cuda', 'hip', 'cpu'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.operation == 'timing' and a.vendor == 'cpu':
        p.error('host timing forbidden')
    a.out.mkdir(parents=True, exist_ok=False)  # refuse repeats BEFORE constructing or fitting
    root = a.source.resolve()
    sys.path.insert(0, str(root / 'python'))
    sys.path.insert(0, str(root / 'tools'))
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        p.error('IDENTICAL mode required')
    import numpy as np
    import bench_board_algos as board
    block, block_info = board._load_block(a.lane, a.dataset, str(a.data))
    arrays = board.lane_arrays(a.lane, block)
    runner = board.build(a.lane, 'ours', arrays)
    provenance = source_provenance(root, a.vendor)
    info = runner.info or {}
    if info.get('numeric_mode_used') != 'identical':
        raise RuntimeError('binding IDENTICAL readback missing or wrong: ' + repr(info))
    metrics = {}
    if a.operation == 'timing':
        import time
        start = time.perf_counter()
        runner.fit()
        fit_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        inferred = runner.infer()
        metrics = {'fit_ms': fit_ms, 'infer_ms': (time.perf_counter() - start) * 1000 if inferred else None}
    else:
        runner.fit()
        runner.infer()
    outputs = runner.outputs()
    if not outputs:
        raise RuntimeError('empty outputs: vacuous identity refused')
    provenance = source_provenance(root, a.vendor)
    digest = hashlib.sha256()
    output_meta = {}
    saved = {}
    for name, value in sorted(outputs.items()):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise RuntimeError('object array cannot establish bitwise identity: ' + name)
        metadata = {'dtype': array.dtype.str, 'shape': list(array.shape), 'bytes': array.nbytes}
        blob = array.tobytes()
        digest.update(json.dumps([name, metadata], sort_keys=True).encode() + b'\0' + blob)
        output_meta[name] = dict(metadata, sha256=hashlib.sha256(blob).hexdigest())
        saved[name] = array
    np.savez(a.out / 'outputs.npz', **saved)
    result = {'status': 'PASS', 'operation': a.operation, 'lane': a.lane, 'dataset': a.dataset,
              'invocations': 1, 'warmups': 0, 'digest': digest.hexdigest(), 'outputs': output_meta,
              'input_shapes': {k: list(v.shape) for k, v in arrays.items() if hasattr(v, 'shape')},
              'block': block_info, 'provenance': provenance, 'runner_info': info, **metrics}
    (a.out / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print('IDENTICAL_WAVE', a.operation, a.lane, a.dataset, 'PASS', digest.hexdigest())


if __name__ == '__main__':
    main()
