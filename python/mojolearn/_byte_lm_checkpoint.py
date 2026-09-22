# SPDX-License-Identifier: Apache-2.0
"""Streamed byte-LM checkpoint storage; no model arithmetic and no GPU execution.

WHY THIS EXISTS. The public byte-LM checkpoint is a JSON envelope whose four
arrays are hex text, and it is bounded at 2 MiB
(`_byte_lm_impl._CHECKPOINT_LIMIT`). Hex doubles the bytes, so that bound is
about 87,381 parameters; the 162,147,840-parameter shape needs 1.95 GB for
parameters plus both AdamW moments and the envelope refuses it by design.
The device-owned step design said so in as many words, "at the target shape
this refuses as it does today and `export_state()` arrays are the checkpoint
path", and then no durable form for those arrays was ever written. A caller who wanted to stop a 162M run and
resume it in another process had to invent a file format.

This module is that durable form. It is the fourth key fact of a resume, and
the reason the moments are in the file rather than left to be re-zeroed: a
restore that drops `m` and `v` produces the SAME loss and the SAME gradient on
its first step (the parameters are right and the moments do not enter the
forward) and a DIFFERENT update, so the divergence appears one step later and
looks like nondeterminism rather than like a missing field. `lane/lm-training-
shakedown` measured exactly that separation on an H100 on 2026-09-17.

WHAT IS AND IS NOT CLAIMED. The bytes are the little-endian `<f4`/`<i4` bytes
of the arrays, copied, never re-encoded through decimal text; `-0.0` stays
`00 00 00 80`, a NaN keeps its payload, and a subnormal keeps its bits. So a
round trip is bit-exact by construction rather than by rounding luck, and a
file written under one vendor restores under another because nothing in the
file depends on the device that produced it. Whether two vendors then take the
same STEP from that state is the training identity claim and is not this
file's to make.

THE JSON/HEX ENVELOPE IS UNCHANGED. `export_checkpoint`, `save_checkpoint`,
`from_checkpoint` and `from_checkpoint_bytes` keep their schema, their bytes,
their 2 MiB bound and their refusals; this is a second, explicitly named
format beside them, never a widening of the first. It is also NOT
`training/checkpoint.mojo`'s native binary v1: that codec is unreachable from
Python, declares itself never compiled and never run, and carries no model
shape, so a loader could not tell one architecture from another with the same
flat count.

Reads verify the whole archive -- header digest, every array digest, and the
exact file size -- before any caller constructs a model, and every length is
derived from the admitted model registry rather than from a descriptor the
file supplied. Memory overhead is the header plus one transfer chunk on top
of the arrays being restored.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

from . import _buffer as buffers, _bufcheck as checks

MAGIC = b'MOJOLEARN-BYTE-LM\x01\n'
SCHEMA = 'mojolearn.byte-lm-stream.v1'
HEADER_LIMIT = 1024 * 1024
CHUNK = 1024 * 1024
#: The four arrays, in file order. `flags` is the per-tensor momentum-
#: initialized vector, one int32 per parameter TENSOR, not per element.
ARRAYS = (('parameters', '<f4'), ('m', '<f4'), ('v', '<f4'), ('flags', '<i4'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _raw(value, dtype):
    """A read/write byte view of a contiguous float32/int32 vector.

    A view, not a copy: `save` streams straight out of the caller's storage
    and `load` reads straight into the array it just allocated, so neither
    direction holds a second 1.95 GB copy of the state.
    """
    if sys.byteorder != 'little':
        raise ValueError('Byte-LM streamed checkpoint requires a little-endian host')
    p = checks.probe(value)
    if (not p.c_contiguous or p.ndim != 1 or p.itemsize != 4
            or not (checks.is_native_f32(p.format) if dtype == '<f4'
                    else checks.is_int32(p.format, p.itemsize))):
        raise ValueError('Byte-LM checkpoint arrays must be contiguous float32/int32 vectors')
    return checks.flat_view(value, 'f' if dtype == '<f4' else 'i').cast('B')


def _digest(raw):
    h = hashlib.sha256()
    for start in range(0, len(raw), CHUNK):
        h.update(raw[start:start + CHUNK])
    return h.hexdigest()


def _counts(metadata):
    """`(n_total, n_tensors)` from the ADMITTED registry, never from the file.

    `state_shape` reads only `model_shape` (absent means the default shape),
    so the two lengths every array descriptor is checked against come from
    the dataclass, and a file claiming a larger array cannot make this
    process allocate one.
    """
    from ._byte_lm_config import state_shape
    from ._byte_lm_impl import _SCHEMA
    if not isinstance(metadata, dict) or metadata.get('schema') != _SCHEMA:
        raise ValueError('Byte-LM checkpoint state schema mismatch')
    shape = state_shape(metadata)
    if (metadata.get('profile') != shape.profile
            or metadata.get('numeric_mode') != 'identical'
            or metadata.get('parameter_names') != list(shape.parameter_names)
            or metadata.get('parameter_shapes') != [list(s) for s in shape.parameter_shapes]
            or metadata.get('parameter_offsets') != list(shape.offsets)):
        raise ValueError('Byte-LM checkpoint profile/registry/mode mismatch')
    return shape.n_total, shape.n_tensors


def _validate(metadata, entries):
    n_total, n_tensors = _counts(metadata)
    keys = {'schema', 'profile', 'numeric_mode', 'parameter_names', 'parameter_shapes',
            'parameter_offsets', 'completed_steps', 'next_batch_index', 'config',
            'data_schedule'}
    if isinstance(metadata, dict) and 'model_shape' in metadata:
        keys.add('model_shape')
    if set(metadata) != keys:
        raise ValueError('Byte-LM checkpoint state has missing or unknown fields')
    if (type(metadata['completed_steps']) is not int
            or type(metadata['next_batch_index']) is not int
            or metadata['completed_steps'] < 0
            or metadata['next_batch_index'] != metadata['completed_steps']
            or not isinstance(metadata['config'], dict)
            or not isinstance(metadata['data_schedule'], dict)):
        raise ValueError('Byte-LM checkpoint state metadata mismatch')
    if not isinstance(entries, list) or len(entries) != len(ARRAYS):
        raise ValueError('Byte-LM checkpoint array registry mismatch')
    for entry, (name, dtype) in zip(entries, ARRAYS):
        count = n_tensors if name == 'flags' else n_total
        if (not isinstance(entry, dict)
                or set(entry) != {'name', 'dtype', 'shape', 'nbytes', 'sha256'}
                or entry['name'] != name or entry['dtype'] != dtype
                or entry['shape'] != [count]
                or type(entry['nbytes']) is not int or entry['nbytes'] != 4 * count
                or not isinstance(entry['sha256'], str)
                or len(entry['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError('Byte-LM checkpoint array descriptor mismatch')
    return n_total, n_tensors


def save(path, state):
    """Borrow array storage, stream to a sibling temporary file, then replace.

    Returns the SHA-256 of the whole file. The digest of each array is taken
    from the caller's storage before the write and again from the bytes
    actually written, and a disagreement raises rather than leaving a file
    that verifies against its own corrupted content.
    """
    metadata = {key: value for key, value in state.items() if key not in dict(ARRAYS)}
    views, entries = [], []
    for name, dtype in ARRAYS:
        raw = _raw(state[name], dtype)
        views.append(raw)
        entries.append(dict(name=name, dtype=dtype, shape=list(state[name].shape),
                            nbytes=len(raw), sha256=_digest(raw)))
    _validate(metadata, entries)
    header = _canonical(dict(schema=SCHEMA, state=metadata, arrays=entries))
    if len(header) > HEADER_LIMIT:
        raise ValueError('Byte-LM checkpoint metadata exceeds the header limit')
    prefix = MAGIC + struct.pack('<Q', len(header)) + header + hashlib.sha256(header).digest()
    path = Path(path)
    temporary = None
    whole = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         delete=False) as stream:
            temporary = stream.name
            if stream.write(prefix) != len(prefix):
                raise OSError('Byte-LM checkpoint short header write')
            whole.update(prefix)
            for raw, entry in zip(views, entries):
                written = hashlib.sha256()
                for start in range(0, len(raw), CHUNK):
                    chunk = raw[start:start + CHUNK]
                    if stream.write(chunk) != len(chunk):
                        raise OSError('Byte-LM checkpoint short write')
                    written.update(chunk)
                    whole.update(chunk)
                if written.hexdigest() != entry['sha256']:
                    raise ValueError('Byte-LM checkpoint state changed during save')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return whole.hexdigest()


def load(path):
    """Read and verify a v1 archive, returning host state before model creation.

    The size check happens BEFORE the arrays are read, so a truncated file is
    refused by name instead of producing a short read halfway through a 618 MB
    parameter vector, and trailing bytes are refused rather than ignored.
    """
    with Path(path).open('rb') as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError('Byte-LM checkpoint stream magic mismatch')
        length = stream.read(8)
        if len(length) != 8:
            raise ValueError('Byte-LM checkpoint truncated header length')
        length = struct.unpack('<Q', length)[0]
        if length < 1 or length > HEADER_LIMIT:
            raise ValueError('Byte-LM checkpoint header length exceeds limit')
        header = stream.read(length)
        digest = stream.read(32)
        if len(header) != length or hashlib.sha256(header).digest() != digest:
            raise ValueError('Byte-LM checkpoint header integrity mismatch')
        try:
            envelope = json.loads(header)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ValueError('Byte-LM checkpoint header is not valid JSON')
        if (not isinstance(envelope, dict)
                or set(envelope) != {'schema', 'state', 'arrays'}
                or envelope['schema'] != SCHEMA):
            raise ValueError('Byte-LM checkpoint stream schema mismatch')
        state, entries = envelope['state'], envelope['arrays']
        _validate(state, entries)
        size = len(MAGIC) + 8 + length + 32 + sum(e['nbytes'] for e in entries)
        if os.fstat(stream.fileno()).st_size != size:
            raise ValueError('Byte-LM checkpoint truncated or trailing array data')
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
                        raise ValueError('Byte-LM checkpoint truncated array')
                    received += n
                actual.update(chunk)
            if actual.hexdigest() != entry['sha256']:
                raise ValueError('Byte-LM checkpoint array integrity mismatch: ' + entry['name'])
            state[entry['name']] = value
        if stream.read(1):
            raise ValueError('Byte-LM checkpoint trailing array data')
    return state
