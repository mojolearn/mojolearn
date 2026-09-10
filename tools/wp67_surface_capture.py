#!/usr/bin/env python3
"""Run an existing neural/GP surface gate and capture exported array bytes."""
import argparse
import json
import os
import shutil
from pathlib import Path
import runpy
import struct
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('surface')
    ap.add_argument('capture', type=Path)
    args = ap.parse_args()
    # Retain the exact GP binaries used by each IDENTICAL surface gate.
    if args.surface.endswith('test_gp_surface.py') and os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'identical':
        shutil.copyfile('python/mojolearn/identical/_mojolearn_gp.so', args.capture.parent / 'gp.so')
    import numpy as np
    from mojolearn._array import Array
    original_asarray = np.asarray
    original_contiguous = np.ascontiguousarray
    original_bytes = Array.tobytes
    count = 0
    with args.capture.open('wb') as stream:
        def record(value):
            nonlocal count
            meta = json.dumps({'shape': value.shape, 'dtype': value.dtype.str}, sort_keys=True).encode()
            raw = value.tobytes(order='C')
            stream.write(struct.pack('<QQ', len(meta), len(raw)))
            stream.write(meta)
            stream.write(raw)
            count += 1

        def convert(fn):
            def call(value, *a, **kw):
                result = fn(value, *a, **kw)
                if isinstance(value, Array):
                    record(result)
                return result
            return call

        def array_bytes(self, *a, **kw):
            raw = original_bytes(self, *a, **kw)
            # Snapshot APIs return exact bytes; include them in the same stream.
            meta = json.dumps({'shape': self.shape, 'dtype': self.dtype}, sort_keys=True).encode()
            nonlocal count
            stream.write(struct.pack('<QQ', len(meta), len(raw)))
            stream.write(meta)
            stream.write(raw)
            count += 1
            return raw

        np.asarray = convert(original_asarray)
        np.ascontiguousarray = convert(original_contiguous)
        Array.tobytes = array_bytes
        sys.argv = [args.surface]
        try:
            runpy.run_path(args.surface, run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            np.asarray = original_asarray
            np.ascontiguousarray = original_contiguous
            Array.tobytes = original_bytes
    if count == 0:
        raise RuntimeError('surface produced no captured arrays; this is not an output gate')
    print(json.dumps({'arrays_captured': count, 'capture': str(args.capture)}))



if __name__ == '__main__':
    main()
