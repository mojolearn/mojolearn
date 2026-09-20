#!/usr/bin/env python3
"""Bitwise and wall-clock check for CPU scaler transform row parallelism."""
import argparse
import hashlib
import os
import statistics
import time

import numpy as np

from mojolearn import MinMaxScaler, StandardScaler
from mojolearn._cpu_reference import reference_training


def measured(call, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=400_000)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    x = (np.arange(args.rows * args.columns, dtype=np.float32)
         .reshape(args.rows, args.columns) % np.float32(1000.0))

    with reference_training():
        cases = (
            StandardScaler(numeric_mode="identical"),
            StandardScaler(with_mean=False, numeric_mode="identical"),
            StandardScaler(with_std=False, numeric_mode="identical"),
            MinMaxScaler(numeric_mode="identical"),
            MinMaxScaler(clip=True, numeric_mode="identical"),
        )
        for scaler in cases:
            scaler.fit(x)
            os.environ["MOJOLEARN_CPU_THREADS"] = "1"
            serial, serial_s = measured(lambda: scaler.transform(x), args.repeats)
            os.environ.pop("MOJOLEARN_CPU_THREADS", None)
            parallel, parallel_s = measured(lambda: scaler.transform(x), args.repeats)
            if serial.tobytes() != parallel.tobytes():
                raise AssertionError(f"{type(scaler).__name__} parallel bytes moved")
            os.environ["MOJOLEARN_CPU_THREADS"] = "1"
            serial_restored = scaler.inverse_transform(serial)
            os.environ.pop("MOJOLEARN_CPU_THREADS", None)
            restored = scaler.inverse_transform(parallel)
            if restored.tobytes() != serial_restored.tobytes():
                raise AssertionError(f"{type(scaler).__name__} inverse bytes moved")
            digest = hashlib.sha256(parallel.tobytes()).hexdigest()
            print(type(scaler).__name__, scaler.get_params(),
                  f"serial={serial_s:.6f}s parallel={parallel_s:.6f}s",
                  f"speedup={serial_s / parallel_s:.3f}x sha256={digest}")


if __name__ == "__main__":
    main()
