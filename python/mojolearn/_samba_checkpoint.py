# SPDX-License-Identifier: Apache-2.0
"""Streamed Samba checkpoint storage; no model arithmetic or GPU execution.

The bounded canonical header describes four consecutive little-endian arrays.
Its digest protects metadata and array digests. Reads verify the entire archive
before a caller constructs a model. Memory overhead is bounded by the header
and transfer chunk, in addition to the arrays being restored.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

from . import _buffer as buffers, _bufcheck as checks

MAGIC = b'MOJOLEARN-SAMBA\x02\n'
SCHEMA = 'mojolearn.samba-stream.v2'
HEADER_LIMIT = 1024 * 1024
CHUNK = 1024 * 1024
ARRAYS = (('parameters', '<f4'), ('exp_avg', '<f4'),
          ('exp_avg_sq', '<f4'), ('buf_initialized', '<i4'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _raw(value, dtype):
    if sys.byteorder != 'little':
        raise ValueError('Samba streamed checkpoint requires a little-endian host')
    p = checks.probe(value)
    if (not p.c_contiguous or p.ndim != 1 or p.itemsize != 4
            or not (checks.is_native_f32(p.format) if dtype == '<f4'
                    else checks.is_int32(p.format, p.itemsize))):
        raise ValueError('Samba checkpoint arrays must be contiguous float32/int32 vectors')
    return checks.flat_view(value, 'f' if dtype == '<f4' else 'i').cast('B')


def _digest(raw):
    h = hashlib.sha256()
    for start in range(0, len(raw), CHUNK):
        h.update(raw[start:start + CHUNK])
    return h.hexdigest()


def _validate(metadata, entries):
    # Derive lengths from the admitted model registry, never from an unchecked
    # array descriptor. Config construction does not load a native extension.
    from ._samba_impl import SambaConfig, PROFILE, _STATE_SCHEMA
    if (not isinstance(metadata, dict)
            or metadata.get('schema') != _STATE_SCHEMA
            or metadata.get('profile') != PROFILE):
        raise ValueError('Samba checkpoint state schema mismatch')
    config = SambaConfig.from_dict(metadata['config'])
    widths = {kind: len(config.block_shapes(kind)) for kind in set(config.layers)}
    registry_count = 2 + int(not config.tie_embeddings) + sum(widths[k] for k in config.layers)
    if (not isinstance(metadata.get('registry'), list)
            or len(metadata['registry']) != registry_count):
        raise ValueError('Samba checkpoint registry length mismatch')
    expected = []
    total = 0
    for name, shape in config.registry():
        count = math.prod(shape)
        if count < 1:
            raise ValueError('Samba checkpoint registry has an empty tensor')
        expected.append(dict(name=name, shape=list(shape), offset=total, size=count))
        total += count
    if total > 2147483647 or metadata.get('registry') != expected:
        raise ValueError('Samba checkpoint registry mismatch or index overflow')
    if (type(metadata.get('t')) is not int or metadata['t'] < 0
            or not isinstance(metadata.get('optimizer'), dict)
            or metadata['optimizer'].get('kind') != 'adamw'
            or not isinstance(metadata.get('rng'), dict)
            or metadata.get('numeric_mode') not in ('identical', 'fast', 'deterministic')
            or not (metadata.get('schedule') is None
                    or isinstance(metadata['schedule'], dict))):
        raise ValueError('Samba checkpoint state metadata mismatch')
    if not isinstance(entries, list) or len(entries) != len(ARRAYS):
        raise ValueError('Samba checkpoint array registry mismatch')
    for entry, (name, dtype) in zip(entries, ARRAYS):
        count = len(expected) if name == 'buf_initialized' else total
        if (not isinstance(entry, dict)
                or set(entry) != {'name', 'dtype', 'shape', 'nbytes', 'sha256'}
                or entry['name'] != name or entry['dtype'] != dtype
                or entry['shape'] != [count]
                or type(entry['nbytes']) is not int or entry['nbytes'] != 4 * count
                or not isinstance(entry['sha256'], str)
                or len(entry['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError('Samba checkpoint array descriptor mismatch')


def save(path, state):
    """Borrow array storage, stream to a sibling temporary file, then replace."""
    metadata = {key: value for key, value in state.items()
                if key not in dict(ARRAYS)}
    views, entries = [], []
    for name, dtype in ARRAYS:
        raw = _raw(state[name], dtype)
        views.append(raw)
        entries.append(dict(name=name, dtype=dtype, shape=list(state[name].shape),
                            nbytes=len(raw), sha256=_digest(raw)))
    _validate(metadata, entries)
    header = _canonical(dict(schema=SCHEMA, state=metadata, arrays=entries))
    if len(header) > HEADER_LIMIT:
        raise ValueError('Samba checkpoint metadata exceeds the header limit')
    prefix = MAGIC + struct.pack('<Q', len(header)) + header + hashlib.sha256(header).digest()
    path = Path(path)
    temporary = None
    whole = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         delete=False) as stream:
            temporary = stream.name
            if stream.write(prefix) != len(prefix):
                raise OSError('Samba checkpoint short header write')
            whole.update(prefix)
            for raw, entry in zip(views, entries):
                written = hashlib.sha256()
                for start in range(0, len(raw), CHUNK):
                    chunk = raw[start:start + CHUNK]
                    if stream.write(chunk) != len(chunk):
                        raise OSError('Samba checkpoint short write')
                    written.update(chunk)
                    whole.update(chunk)
                if written.hexdigest() != entry['sha256']:
                    raise ValueError('Samba checkpoint state changed during save')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return whole.hexdigest()


def load(path):
    """Read and verify a v2 archive, returning host state before model creation."""
    with Path(path).open('rb') as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError('Samba checkpoint stream magic mismatch')
        length = stream.read(8)
        if len(length) != 8:
            raise ValueError('Samba checkpoint truncated header length')
        length = struct.unpack('<Q', length)[0]
        if length < 1 or length > HEADER_LIMIT:
            raise ValueError('Samba checkpoint header length exceeds limit')
        header = stream.read(length)
        digest = stream.read(32)
        if len(header) != length or hashlib.sha256(header).digest() != digest:
            raise ValueError('Samba checkpoint header integrity mismatch')
        envelope = json.loads(header)
        if (not isinstance(envelope, dict)
                or set(envelope) != {'schema', 'state', 'arrays'}
                or envelope['schema'] != SCHEMA):
            raise ValueError('Samba checkpoint stream schema mismatch')
        state, entries = envelope['state'], envelope['arrays']
        _validate(state, entries)
        size = len(MAGIC) + 8 + length + 32 + sum(e['nbytes'] for e in entries)
        if os.fstat(stream.fileno()).st_size != size:
            raise ValueError('Samba checkpoint truncated or trailing array data')
        for entry in entries:
            value = buffers.empty(tuple(entry['shape']), entry['dtype'])
            raw = _raw(value, entry['dtype'])
            actual = hashlib.sha256()
            for start in range(0, len(raw), CHUNK):
                chunk = raw[start:start + CHUNK]
                received = 0
                while received < len(chunk):
                    n = stream.readinto(chunk[received:])
                    if not n:
                        raise ValueError('Samba checkpoint truncated array')
                    received += n
                actual.update(chunk)
            if actual.hexdigest() != entry['sha256']:
                raise ValueError('Samba checkpoint array integrity mismatch: ' + entry['name'])
            state[entry['name']] = value
        if stream.read(1):
            raise ValueError('Samba checkpoint trailing array data')
    return state
