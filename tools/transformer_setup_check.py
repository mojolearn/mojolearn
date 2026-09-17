#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One selected Transformer setup group; run each binding under a GPU deadline.

outputs: carried/full/ring state, resets and in-place weight edits.
refusals: all weight names, NaN/Inf classification and refusal precedence.
Compare the arrays with transformer_readback_check.py --compare; compare
refusal JSON's messages between immutable baseline and candidate binaries.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np
from bench_neural_decode import weights

NAMES = ('input_layernorm.weight', 'post_attention_layernorm.weight',
         'q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
         'gate_proj.weight', 'up_proj.weight', 'down_proj.weight')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binding', type=Path, required=True)
    ap.add_argument('--backend', choices=('metal', 'cuda', 'hip', 'cpu'), required=True)
    ap.add_argument('--group', choices=('outputs', 'refusals'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not __debug__:
        ap.error('verification requires assertions (no -O)')
    spec = importlib.util.spec_from_file_location(args.binding.stem, args.binding.resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.transformer_vendor() == args.backend
    assert mod.transformer_numeric_mode() == 1
    dm, cap = 32, 6
    w = weights('transformer', dm)
    x = np.random.default_rng(42).uniform(-.5, .5, (1, cap, dm)).astype(np.float32)
    artifacts, messages, samples = {}, {}, []

    def state(window):
        return [np.full((window or cap) * dm // 2, value, np.float32)
                for value in (.125, -.25)]

    def call(start, length, cache, window):
        xx = np.ascontiguousarray(x[:, start:start + length])
        yy = np.full_like(xx, np.nan)
        arrays = [xx] + w + cache + [yy]
        begin = time.perf_counter()
        n = mod.transformer_forward([a.ctypes.data for a in arrays],
                                    [1, length, dm, 2, 1, dm // 2, 2 * dm,
                                     cap, start, window])
        elapsed = 1000 * (time.perf_counter() - begin)
        assert n == start + length
        assert np.isfinite(yy).all()
        return yy, elapsed

    if args.group == 'outputs':
        original = w[2].copy()
        for window in (0, 3):
            for change in ('original', 'mutated', 'reset'):
                w[2][:] = original
                if change == 'mutated':
                    w[2].flat[0] += np.float32(.125)
                cache = state(window)
                for start, length in ((0, 2), (2, 3), (5, 1)):
                    yy, elapsed = call(start, length, cache, window)
                    samples.append(elapsed)
                    for kind, array in zip(('y', 'k', 'v'), [yy] + cache):
                        key = f'window{window}.{change}.pos{start}.{kind}'
                        artifacts[key] = array.copy()
                        if change == 'reset':
                            assert array.tobytes() == artifacts[key.replace('.reset.', '.original.')].tobytes(), key
            assert any(artifacts[f'window{window}.original.pos{p}.y'].tobytes() !=
                       artifacts[f'window{window}.mutated.pos{p}.y'].tobytes()
                       for p in (0, 2, 5)), 'weight edit not observed'
        np.savez(args.out, **artifacts)
    else:
        def refuse(key, name, index, kind):
            cache = state(0)
            before = [a.tobytes() for a in cache]
            try:
                call(0, 2, cache, 0)
            except Exception as exc:
                message = str(exc)
                assert f'{kind} in {name} at flat index {index} REFUSED' in message, message
                assert [a.tobytes() for a in cache] == before, 'state mutated before refusal'
                messages[key] = message
            else:
                raise AssertionError(f'invalid weight accepted: {key}')

        for slot, name in enumerate(NAMES):
            for kind, bits in (('NaN', 0x7FC01234), ('infinity', 0x7F800000),
                               ('infinity', 0xFF800000)):
                for index in (0, w[slot].size - 1):
                    saved = w[slot].flat[index]
                    w[slot].view(np.uint32).flat[index] = bits
                    refuse(f'{slot}.{bits:x}.{index}', name, index, kind)
                    w[slot].flat[index] = saved
        # The lower tensor index wins even when a later tensor has a
        # lower flat index. Within a tensor, the lower flat index wins.
        for slot in range(len(w) - 1):
            a, b = w[slot].copy(), w[slot + 1].copy()
            w[slot].flat[-1] = np.inf
            w[slot + 1].flat[0] = np.nan
            refuse(f'precedence.{slot}', NAMES[slot], w[slot].size - 1, 'infinity')
            w[slot][:], w[slot + 1][:] = a, b
        w[2].flat[-1] = np.nan
        w[2].flat[1] = -np.inf
        refuse('first-index', NAMES[2], 1, 'infinity')
        w[2][:] = weights('transformer', dm)[2]
        yy, _ = call(0, 2, state(0), 0)  # recovery after refusals
        assert np.isfinite(yy).all()

    result = dict(backend=args.backend, group=args.group, arrays=len(artifacts),
                  messages=messages, samples_ms=samples,
                  median_ms=statistics.median(samples) if samples else None,
                  binding_sha256=hashlib.sha256(args.binding.read_bytes()).hexdigest())
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('messages', 'samples_ms')}
                     | {'refusals': len(messages)}), flush=True)


if __name__ == '__main__':
    main()
