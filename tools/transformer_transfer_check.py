#!/usr/bin/env python3
"""Record complete public transformer outputs for cross-build transfer admission.

Run in separate processes for each binding, then compare the `sha256` maps.
Includes carried/ring caches, unused cache bytes and every backward gradient.
Prices are synchronized host-to-host public calls, not opponent measurements.
"""
import argparse
import hashlib
import json
import statistics
import time
import numpy as np
from mojolearn import TransformerBlock
from mojolearn.tests.test_transformer_surface import _weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--rounds', type=int, default=5)
    args = ap.parse_args()
    hashes, times = {}, {}

    def record(name, value):
        value = np.ascontiguousarray(value)
        hashes[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(value.tobytes()).hexdigest()}

    for hd in (64, 128):
        dm, nh, nk, it = hd * 2, 2, 1, hd * 4
        rng = np.random.default_rng(910 + hd)
        weights = _weights(rng, dm, nh, nk, hd, it)
        x = rng.uniform(-0.5, 0.5, (2, 9, dm)).astype(np.float32)
        for window in (0, 5):
            prefix = f'hd{hd}.window{window}'
            block = TransformerBlock(weights, n_heads=nh, n_kv_heads=nk, window=window)
            record(prefix + '.fresh', block.forward(x))
            state = block.allocate_state(2, 13)
            # Unused cache capacity is observable and must round-trip unchanged.
            state.k_cache.fill(np.float32(0.125))
            state.v_cache.fill(np.float32(-0.25))
            for start, end in ((0, 4), (4, 8), (8, 9)):
                fn = block.step if end - start == 1 else block.forward
                record(prefix + f'.chunk{start}', fn(x[:, start:end, :], state))
                record(prefix + f'.k{start}', state.k_cache)
                record(prefix + f'.v{start}', state.v_cache)
                assert state.cached_tokens == end
            dy = rng.uniform(-0.25, 0.25, x.shape).astype(np.float32)
            for name, grad in block.backward(x, dy).items():
                record(prefix + '.grad.' + name, grad)
        price_x = rng.uniform(-0.5, 0.5, (2, 512, dm)).astype(np.float32)
        block = TransformerBlock(weights, n_heads=nh, n_kv_heads=nk)
        for _ in range(2):
            block.forward(price_x)
        samples = []
        for _ in range(args.rounds):
            t = time.perf_counter_ns()
            result = block.forward(price_x)
            samples.append((time.perf_counter_ns() - t) / 1e6)
        record(f'hd{hd}.price', result)
        times[f'b2_l512_d{dm}'] = {'samples_ms': samples,
                                  'median_ms': statistics.median(samples)}
    out = {'sha256': hashes, 'timings': times}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps({'arrays': len(hashes), 'timings': times}, sort_keys=True))


if __name__ == '__main__':
    main()
