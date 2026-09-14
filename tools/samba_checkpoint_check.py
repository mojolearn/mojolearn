#!/usr/bin/env python3
"""Cloud-only streamed Samba checkpoint integrity, replay, and large archive gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')

    import numpy as np
    from mojolearn._samba_impl import SambaConfig, SambaStack, _CHECKPOINT_SCHEMA
    from mojolearn._training_impl import Generator, WarmupCosineLR
    from mojolearn import _buffer as buffers, _bufcheck as checks
    from mojolearn import _samba_checkpoint as storage

    def file_hash(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(storage.CHUNK):
                h.update(chunk)
        return h.hexdigest()

    def assert_state(left, right):
        for name, dtype in storage.ARRAYS:
            assert left[name].shape == right[name].shape
            assert storage._digest(storage._raw(left[name], dtype)) == storage._digest(storage._raw(right[name], dtype)), name
        names = dict(storage.ARRAYS)
        a = {k: v for k, v in left.items() if k not in names}
        b = {k: v for k, v in right.items() if k not in names}
        assert storage._canonical(a) == storage._canonical(b)

    class BeforeConstruction(SambaStack):
        def __init__(self, *a, **kw):
            raise AssertionError('invalid archive reached model construction')

    refused = []
    with tempfile.TemporaryDirectory(prefix='samba-checkpoint-') as directory:
        root = Path(directory)
        model = SambaStack(SambaConfig(vocab=17, d_model=32,
            layers=('attention',), n_heads=2, n_kv_heads=1, head_dim=16,
            intermediate=64, dropout=.25), generator=Generator(737,'identical'),
            lr=1e-3, lr_schedule=WarmupCosineLR(1e-3, 2, 8),
            max_norm=.1, numeric_mode='identical')
        inputs = np.asarray([[1,3,5,7,9,11,13]], dtype='<i4')
        targets = np.asarray([[3,5,7,9,11,13,15]], dtype='<i4')
        model.train_step(inputs, targets)
        model.train_step(inputs, targets)
        checkpoint = root / 'small.samba'
        returned = model.save_checkpoint(checkpoint)
        assert returned == file_hash(checkpoint)
        restored = SambaStack.from_checkpoint(checkpoint, numeric_mode='identical')
        assert_state(model.state_dict(), restored.state_dict())
        a, b = model.train_step(inputs, targets), restored.train_step(inputs, targets)
        assert storage._canonical(a) == storage._canonical(b)
        assert_state(model.state_dict(), restored.state_dict())

        original = checkpoint.read_bytes()
        header_length = struct.unpack('<Q', original[len(storage.MAGIC):len(storage.MAGIC)+8])[0]
        header_start = len(storage.MAGIC) + 8
        data_start = header_start + header_length + 32
        mutations = {}
        for name, position in (('header_digest', data_start-1),
                               ('parameter', data_start), ('last_array', len(original)-1)):
            changed = bytearray(original)
            changed[position] ^= 1
            mutations[name] = bytes(changed)
        mutations['truncated'] = original[:-1]
        mutations['trailing'] = original + b'!'
        mutations['header_bound'] = (storage.MAGIC + struct.pack('<Q', storage.HEADER_LIMIT+1))
        envelope = json.loads(original[header_start:header_start+header_length])
        envelope['state']['registry'][0]['size'] += 1
        bad_header = storage._canonical(envelope)
        mutations['registry'] = (storage.MAGIC + struct.pack('<Q', len(bad_header))
            + bad_header + hashlib.sha256(bad_header).digest() + original[data_start:])
        for name, changed in mutations.items():
            path = root / (name + '.samba')
            path.write_bytes(changed)
            try:
                BeforeConstruction.from_checkpoint(path, numeric_mode='identical')
            except ValueError:
                refused.append(name)
            else:
                raise AssertionError('corrupt archive accepted: ' + name)

        # A failed publication must preserve the previous complete archive and
        # clean its temporary file. No model arithmetic is mocked.
        with patch.object(storage.os, 'replace', side_effect=OSError('injected publication failure')):
            try:
                model.save_checkpoint(checkpoint)
            except OSError:
                pass
            else:
                raise AssertionError('publication failure did not propagate')
        assert checkpoint.read_bytes() == original
        assert not list(root.glob('.small.samba.*'))

        # Existing JSON checkpoints still load, with the old bounded reader.
        legacy = restored.state_dict()
        for name, dtype in storage.ARRAYS:
            value = legacy[name]
            legacy[name] = dict(dtype=dtype, shape=list(value.shape),
                hex=checks.le_bytes(value, 'i' if dtype == '<i4' else 'f').hex())
        legacy_path = root / 'legacy.json'
        legacy_path.write_bytes(storage._canonical(dict(schema=_CHECKPOINT_SCHEMA,
            payload=legacy, payload_sha256=hashlib.sha256(storage._canonical(legacy)).hexdigest())))
        legacy_model = SambaStack.from_checkpoint(legacy_path, numeric_mode='identical')
        assert_state(restored.state_dict(), legacy_model.state_dict())

        # Storage capacity fixture uses a real admitted larger Samba registry,
        # without constructing or running its GPU model. The public model
        # save/load/continuation path is covered above on the trained fixture.
        config = SambaConfig(vocab=256, d_model=1536, layers=('attention',),
            n_heads=24, n_kv_heads=6, head_dim=64, intermediate=6144)
        large = model.state_dict()
        large['config'] = config.to_dict()
        registry, total = [], 0
        for name, shape in config.registry():
            count = 1
            for size in shape:
                count *= size
            registry.append(dict(name=name, shape=list(shape), offset=total, size=count))
            total += count
        large['registry'] = registry
        for name, dtype in storage.ARRAYS:
            count = len(registry) if name == 'buf_initialized' else total
            large[name] = buffers.zeros((count,), dtype)
            view = storage._raw(large[name], dtype).cast('I')
            view[0] = 1 if dtype == '<i4' else 0x80000000
            view[-1] = 1 if dtype == '<i4' else 0x00000001
        large_path = root / 'large.samba'
        large_hash = storage.save(large_path, large)
        large_bytes = large_path.stat().st_size
        assert large_bytes > 256 * 1024 * 1024
        assert large_hash == file_hash(large_path)
        loaded = storage.load(large_path)
        assert_state(large, loaded)
        report = dict(status='PASS', small_checkpoint_sha256=returned,
            large_checkpoint_sha256=large_hash, large_checkpoint_bytes=large_bytes,
            large_parameters=total, refused=refused, atomic_publication=True,
            legacy_roundtrip=True, continuation_steps=3,
            scope='Actual trained Samba public checkpoint replay; >256MiB host archive roundtrip with admitted Samba registry. No large-model GPU capacity claim.')
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS Samba checkpoint replay, integrity and', large_bytes, 'byte archive')


if __name__ == '__main__':
    main()
