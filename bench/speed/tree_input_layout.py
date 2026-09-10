#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B the tree boundary's packing only; no GPU, fit, or import time included.

Run with PYTHONPATH=python python bench/speed/tree_input_layout.py.
Alternates NumPy and tiled packing, verifies exact float32 bytes, and reports
medians after one warm-up of each arm. A fit speedup must be measured separately.
"""
import argparse
import json
import platform
import statistics
import time

import numpy as np

from mojolearn._arrays import as_f32_colmajor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--cols", type=int, default=28)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.rounds) < 1:
        parser.error("rows, cols and rounds must be positive")
    for dtype in (np.float32, np.float64):
        source = np.random.default_rng(7).random((args.rows, args.cols), dtype=dtype)
        funcs = (lambda: np.asfortranarray(source, dtype=np.float32),
                 lambda: as_f32_colmajor(source, "X")[0])
        timings = [[], []]
        for round_id in range(args.rounds + 1):
            for arm in ((0, 1) if round_id % 2 == 0 else (1, 0)):
                start = time.perf_counter()
                result = funcs[arm]()
                elapsed = (time.perf_counter() - start) * 1000
                if round_id:
                    timings[arm].append(elapsed)
                del result
        expected, actual = funcs[0](), funcs[1]()
        equal = np.array_equal(expected.view(np.uint32), actual.view(np.uint32))
        if not equal:
            raise AssertionError("packing changed float32 bits")
        medians = [statistics.median(values) for values in timings]
        print(json.dumps(dict(machine=platform.machine(), numpy=np.__version__,
                              rows=args.rows, cols=args.cols, dtype=np.dtype(dtype).name,
                              rounds=args.rounds, numpy_ms=medians[0],
                              tiled_ms=medians[1], speedup=medians[0] / medians[1],
                              exact_bits=equal, samples_ms=timings)))


if __name__ == "__main__":
    main()
