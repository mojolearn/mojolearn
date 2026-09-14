#!/usr/bin/env python3
"""Cloud-only accumulation working set exceeding one 80GB GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', action='store_true', required=True)
    p.add_argument('--devices', type=int, choices=(1,2), required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    # Avoid host THP compaction dominating this GPU-capacity fixture on a
    # fragmented cloud NUMA host. NumPy reads this at import; GPU arithmetic
    # and allocation sizes are unchanged.
    os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
    import numpy as np
    from mojolearn._training_impl import _load
    binding = _load('identical')
    assert binding.accumulate_parallel_available() == 1
    n, steps, period = 1_500_000_001, 16, 4093
    # A prime repeat period detects column offsets/strides across ownership
    # boundaries. The arithmetic oracle is the original GPU tree on one period.
    rng = np.random.default_rng(351904)
    pattern = rng.normal(0,.1,(steps,period)).astype('<f4')
    pattern[:4,0] = [2**26,1,-2**26,1]
    expected = np.empty(period,dtype='<f4')
    os.environ['MOJOLEARN_OPTIMIZER_DEVICE_COUNT'] = '1'
    binding.accumulate([expected.ctypes.data,pattern.ctypes.data],[period,steps,-1])
    parts = np.empty((steps,n),dtype='<f4')
    block = period*256
    for i in range(steps):
        tile = np.tile(pattern[i],256)
        for start in range(0,n,block):
            end = min(start+block,n)
            parts[i,start:end] = tile[:end-start]
    out = np.full(n,-17,dtype='<f4')
    print('allocated host fixture',parts.nbytes,out.nbytes,flush=True)
    os.environ['MOJOLEARN_OPTIMIZER_DEVICE_COUNT'] = str(args.devices)
    memory_log = args.report.with_suffix('.memory.csv')
    memory_output = memory_log.open('w')
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits','--loop-ms=200'],stdout=memory_output,stderr=subprocess.DEVNULL)
    started = time.monotonic()
    result = dict(devices=args.devices,n=n,steps=steps,parts_bytes=parts.nbytes,
                  scope='Gradient accumulation buffer capacity only; not full-model training or throughput.')
    try:
        try:
            assert binding.accumulate([out.ctypes.data,parts.ctypes.data],[n,steps,-1]) == n
        except Exception as error:
            if args.devices != 1 or not any(term in str(error).lower() for term in ('out_of_memory','out of memory')):
                raise
            for start in range(0,n,block):
                assert np.all(out[start:start+block] == np.float32(-17))
            result.update(status='REFUSED',error=str(error),output_unchanged=True)
        else:
            if args.devices == 1:
                raise AssertionError('one-device working set fit; capacity boundary not established')
            tile = np.tile(expected,256).view('<u4')
            digest = hashlib.sha256()
            for start in range(0,n,block):
                values = out[start:min(start+block,n)]
                assert np.array_equal(values.view('<u4'),tile[:len(values)]), start
                digest.update(memoryview(values).cast('B'))
            result.update(status='PASS',verified_cells=n,sha256=digest.hexdigest())
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        memory_output.close()
    peaks = {}
    for line in memory_log.read_text().splitlines():
        device,used = (int(value.strip()) for value in line.split(','))
        peaks[device] = max(peaks.get(device,0),used)
    assert peaks
    result.update(numpy_madvise_hugepage=os.environ['NUMPY_MADVISE_HUGEPAGE'],sampled_peak_mib=peaks,elapsed_seconds=time.monotonic()-started,
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True))
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(args.report.read_text(),flush=True)


if __name__ == '__main__':
    main()
