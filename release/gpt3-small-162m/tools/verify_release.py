#!/usr/bin/env python3
"""Offline package and checkpoint integrity inspection; no model execution."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b'MOJOLEARN-BYTE-LM\x01\n'
EXPECTED = '4129921e99ed404db1ca4b7b94ebc43c0bee18e7711deb326b31ec89e0b402c8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_checkpoint(path):
    with path.open('rb') as f:
        require(f.read(len(MAGIC)) == MAGIC, 'checkpoint magic')
        length_bytes = f.read(8)
        require(len(length_bytes) == 8, 'header length missing')
        length = struct.unpack('<Q', length_bytes)[0]
        require(0 < length <= 1024 * 1024, 'header length outside bound')
        raw = f.read(length)
        require(len(raw) == length and hashlib.sha256(raw).digest() == f.read(32), 'header digest')
        header = json.loads(raw)
        require(header['schema'] == 'mojolearn.byte-lm-stream.v1', 'checkpoint schema')
        require(header['state']['completed_steps'] == 5000, 'final step')
        expected_arrays = [('parameters', '<f4', 162147840), ('m', '<f4', 162147840),
                           ('v', '<f4', 162147840), ('flags', '<i4', 110)]
        require(len(header['arrays']) == len(expected_arrays), 'array count')
        for entry, (name, dtype, count) in zip(header['arrays'], expected_arrays):
            require(entry['name'] == name and entry['dtype'] == dtype
                    and entry['shape'] == [count] and entry['nbytes'] == count * 4,
                    'array descriptor: ' + name)
            digest = hashlib.sha256()
            remaining = entry['nbytes']
            while remaining:
                chunk = f.read(min(8 * 1024 * 1024, remaining))
                require(bool(chunk), 'truncated array: ' + name)
                digest.update(chunk)
                remaining -= len(chunk)
            require(digest.hexdigest() == entry['sha256'], 'array digest: ' + name)
        require(not f.read(1), 'trailing checkpoint bytes')
    return header


def main():
    manifest = json.loads((ROOT / 'files.sha256.json').read_text())
    require(manifest.get('checkpoint.blm') == EXPECTED, 'unrecognized checkpoint pin')
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), 'path outside package')
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        require(actual == expected, 'file digest: ' + name)
    header = inspect_checkpoint(ROOT / 'checkpoint.blm')
    print(f"PASS: {len(manifest)} packaged file hashes; checkpoint header and all four array digests.")
    print('Final step:', header['state']['completed_steps'])
    print('Parameters SHA-256:', header['arrays'][0]['sha256'])


if __name__ == '__main__':
    main()
