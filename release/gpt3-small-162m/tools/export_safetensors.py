#!/usr/bin/env python3
"""Write model.safetensors from checkpoint.blm: the parameter bytes, unchanged.

The parameters array is cut at the header's offsets and reshaped to its recorded
shapes. No value is converted, rounded or recomputed. The script then confirms that
the tensors, concatenated in registry order, hash to the checkpoint's parameters
SHA-256, so the export is the trained weights byte for byte.

    python3 tools/export_safetensors.py [checkpoint.blm] [model.safetensors]

Requires numpy and safetensors.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b'MOJOLEARN-BYTE-LM\x01\n'
PARAMETERS_SHA256 = 'c50fd76eb831a5d9754c661780ece20e10fe3b42c89f8b3d4650e2aba215a8ea'


def read_parameters(path):
    with open(path, 'rb') as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError('not a Mojolearn byte-LM checkpoint')
        length = struct.unpack('<Q', f.read(8))[0]
        raw = f.read(length)
        if hashlib.sha256(raw).digest() != f.read(32):
            raise ValueError('checkpoint header digest')
        header = json.loads(raw)
        entry = header['arrays'][0]
        if entry['name'] != 'parameters' or entry['dtype'] != '<f4':
            raise ValueError('first array is not the float32 parameters')
        data = f.read(entry['nbytes'])
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('parameters digest')
    return header['state'], np.frombuffer(data, '<f4')


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'checkpoint.blm'
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'model.safetensors'
    state, flat = read_parameters(src)
    names, shapes, offsets = state['parameter_names'], state['parameter_shapes'], state['parameter_offsets']
    tensors = {}
    for name, shape, start, stop in zip(names, shapes, offsets, offsets[1:]):
        tensors[name] = flat[start:stop].reshape(shape)
    metadata = {'format': 'mojolearn-byte-lm', 'source_checkpoint_completed_steps': str(state['completed_steps']),
                'parameters_sha256': PARAMETERS_SHA256, 'tensor_order': ','.join(names)}
    save_file(tensors, str(dst), metadata=metadata)
    loaded = load_file(str(dst))
    digest = hashlib.sha256()
    for name in names:
        digest.update(np.ascontiguousarray(loaded[name], '<f4').tobytes())
    if digest.hexdigest() != PARAMETERS_SHA256:
        raise ValueError('exported tensors do not reproduce the checkpoint parameters')
    print(f'wrote {dst}: {len(names)} tensors, {flat.size} parameters, parameters SHA-256 {PARAMETERS_SHA256}')


if __name__ == '__main__':
    main()
