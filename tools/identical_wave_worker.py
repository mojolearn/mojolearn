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
    p.add_argument('--case-json',default='{}')
    p.add_argument('--timing-contract',choices=('board-warm-single-call','cold-single-call'),default='board-warm-single-call')
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
    case=json.loads(a.case_json)
    driver=case.get('driver','algos')
    if driver=='algos':
        block, block_info = board._load_block(a.lane, a.dataset, str(a.data))
        arrays = board.lane_arrays(a.lane, block)
        runner = board.build(a.lane, 'ours', arrays)
    elif driver in ('classical','estimator'):
        base=a.data/(case.get('block','reg')+'-'+a.dataset)
        with np.load(str(base)+'.npz') as z: arrays={k:np.ascontiguousarray(z[k]) for k in z.files}
        block_info=json.loads(Path(str(base)+'.json').read_text())
        import classical_two_datasets as ctd
        if driver=='classical':
            cls={'pca':ctd.OursPCA,'ols':ctd.OursOLS,'kmeans':ctd.OursKMeans}[a.lane]
            original=cls(arrays,block_info)
            runner=board.Runner(original.info,original.call,original.outputs,sync=original.sync)
        else:
            import mojolearn as ml
            cls=getattr(ml,case['class'])
            state={};params=case['params']
            example=cls(**params)
            info=ctd._ours_info(ml,example)
            def fit():
                model=cls(**params)
                state['model']=model.fit(arrays['X'],arrays['y']) if case.get('fit_target',True) else model.fit(arrays['X'])
            def infer(): state['pred']=state['model'].predict(arrays['Xq'])
            if case.get('output_attribute'):
                runner=board.Runner(info,fit,lambda:{case['output_attribute']:np.asarray(getattr(state['model'],case['output_attribute']))})
            else:
                runner=board.Runner(info,fit,lambda:{'pred':np.asarray(state['pred'])},infer)
    elif driver=='metric':
        import mojolearn as ml
        from mojolearn import _backend
        binding=_backend.binding('_mojolearn_metrics','identical')
        modes=[getattr(binding,n)() for n in dir(binding) if n.endswith('_numeric_mode')]
        if not modes or any(m!=1 for m in modes): raise RuntimeError('metric IDENTICAL mode readback missing')
        with np.load(a.data/('reg-'+a.dataset+'.npz')) as z: target=np.ascontiguousarray(z['yq'])
        predicted=np.ascontiguousarray(target+np.float32(0.25));arrays={'target':target,'predicted':predicted};state={}
        function=getattr(ml.metrics,case['function'])
        def fit(): state['out']=function(target,predicted)
        runner=board.Runner({'numeric_mode_used':'identical','device':'gpu','library':'mojolearn'},fit,lambda:{'value':np.asarray(state['out'])})
        block_info={'canonical_block':'reg-'+a.dataset,'fixture':'target+float32(0.25)','function':case['function']}
    elif driver=='neural':
        import bench_board_neural as neural
        fixture=a.data.parent/'wave-neural'/case['fixture']
        metadata=json.loads(fixture.with_suffix('.json').read_text())
        actual=hashlib.sha256(fixture.read_bytes()).hexdigest()
        if metadata['sha256']!=actual: raise RuntimeError('neural canonical fixture hash mismatch')
        with np.load(fixture) as z: arrays={k:z[k] for k in z.files}
        lane=a.lane
        if a.vendor=='cpu' and lane.endswith('-forward'): lane=lane[:-8]+'-infer'
        original=neural.build_runner(lane,'ours',case.get('shape','small'),arrays)
        runner=board.Runner(original.info,original.call,original.outputs,sync=original.sync)
        block_info=metadata
    else: raise RuntimeError('unknown driver '+driver)
    provenance = source_provenance(root, a.vendor)
    info = runner.info or {}
    if info.get('numeric_mode_used') != 'identical':
        raise RuntimeError('binding IDENTICAL readback missing or wrong: ' + repr(info))
    metrics = {}
    warmups=0
    if a.operation=='timing' and a.timing_contract=='board-warm-single-call':
        runner.fit(); runner.infer()  # Untimed board warmup; only the next call is sampled.
        warmups=1
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
              'invocations': 1+warmups, 'timing_samples':1 if a.operation=='timing' else 0, 'warmups':warmups, 'timing_contract':a.timing_contract if a.operation=='timing' else 'untimed-identity', 'driver':driver, 'digest': digest.hexdigest(), 'outputs': output_meta,
              'input_shapes': {k: list(v.shape) for k, v in arrays.items() if hasattr(v, 'shape')},
              'block': block_info, 'provenance': provenance, 'runner_info': info, **metrics}
    (a.out / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print('IDENTICAL_WAVE', a.operation, a.lane, a.dataset, 'PASS', digest.hexdigest())


if __name__ == '__main__':
    main()
