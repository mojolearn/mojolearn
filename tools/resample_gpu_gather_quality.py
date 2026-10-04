#!/usr/bin/env python3
"""Exact public resample outputs versus independently indexed current draw.
No timing. Captures output bytes, refusals, actual binding calls and lifetime.
"""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np

FIXTURE = 'resample-gpu-gather-v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('enabled', type=int, choices=(0, 1))
    args = parser.parse_args()
    rs = importlib.import_module('mojolearn.resample')
    mod = rs._extension('fast')
    assert int(mod.resample_numeric_mode()) == 0
    assert str(mod.resample_vendor()) == 'metal'
    assert int(mod.resample_gpu_gather_enabled()) == args.enabled
    original = mod.resample_gather_gpu
    calls = []

    def observed(*a):
        result = original(*a)
        calls.append(int(result))
        return result

    mod.resample_gather_gpu = observed
    captured = {}
    held = []
    words = np.array([0, 0x80000000, 0x7fc01234, 0xffc05678,
                      0x7f800000, 0xff800000, 1, 0x3f800000], dtype=np.uint32)
    special = np.resize(words, 257 * 11).view(np.float32).reshape(257, 11)
    cases = [
        ('words', [special, np.arange(257, dtype=np.float32)], 513, True, True),
        ('single', [np.arange(1, dtype=np.float32)], 33, True, True),
        ('wide', [np.arange(1031 * 220, dtype=np.float32).reshape(1031, 220)], 1025, True, True),
        ('taxi_width', [np.arange(4099 * 11, dtype=np.float32).reshape(4099, 11)], 4097, True, True),
        ('zero_width', [np.empty((257, 0), dtype=np.float32)], 17, True, False),
        ('mixed_zero_width', [special, np.empty((257, 0), dtype=np.float32)], 17, True, False),
        ('strided', [special[::2]], 17, True, False),
        ('fortran', [np.asfortranarray(special)], 17, True, False),
        ('float64', [np.arange(257, dtype=np.float64)], 17, True, False),
        ('integer', [np.arange(257, dtype=np.int32)], 17, True, False),
        ('three_dim', [np.arange(120, dtype=np.float32).reshape(10, 3, 4)], 17, True, False),
        ('list', [list(range(257))], 17, True, False),
        ('without_replace', [special], 17, False, False),
    ]
    for name, arrays, count, replace, eligible in cases:
        for seed in (0, 7):
            indices = np.asarray(rs.resample_indices(len(arrays[0]), count, replace, seed, 'fast')).astype(np.intp)
            expected = [np.asarray(a)[indices].copy() for a in arrays]
            before = len(calls)
            actual = rs.resample(*arrays, n_samples=count, replace=replace, random_state=seed, numeric_mode='fast')
            outputs = [actual] if len(arrays) == 1 else actual
            assert len(calls) - before == int(bool(args.enabled and eligible)), name
            if len(calls) > before:
                assert calls[-1] == 1
            for j, (output, want) in enumerate(zip(outputs, expected)):
                got = np.asarray(output)
                assert got.shape == want.shape and got.dtype == want.dtype, name
                assert got.tobytes() == want.tobytes(), name
                captured[f'{name}_{seed}_{j}'] = got.copy()
                held.append((got, got.tobytes()))
    # Preserve main's behavior, including whichever invalid cases it refuses.
    refusals = {}
    for name, arrays, kwargs in [
        ('zero_count', [special], {'n_samples': 0}),
        ('negative_count', [special], {'n_samples': -1}),
        ('empty_rows', [np.empty((0, 11), dtype=np.float32)], {'n_samples': 1}),
        ('lengths', [special, special[:-1]], {}),
        ('stratify', [special], {'stratify': [0] * 257}),
        ('weights', [special], {'sample_weight': [1] * 257}),
    ]:
        try:
            value = rs.resample(*arrays, numeric_mode='fast', **kwargs)
            refusals[name] = ['returned', list(np.asarray(value).shape)]
        except (ValueError, RuntimeError) as exc:
            refusals[name] = [type(exc).__name__, str(exc)]
    # New gathers must not overwrite previously returned caller-owned storage.
    for seed in range(3):
        rs.resample(special, n_samples=513, random_state=seed + 99, numeric_mode='fast')
    gc.collect()
    for got, saved in held:
        assert got.tobytes() == saved, 'held output changed'
    np.savez(args.output, **captured)
    record = dict(fixture=FIXTURE, status='PASS', enabled=args.enabled,
                  successful_gpu_calls=sum(calls), arrays=len(captured), refusals=refusals,
                  binding_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest())
    Path(args.output + '.json').write_text(json.dumps(record, sort_keys=True))
    print('RESAMPLE_GPU_QUALITY ' + json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    main()
