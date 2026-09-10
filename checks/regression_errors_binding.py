#!/usr/bin/env python3
"""Actual GPU public regression-error oracle; requires all three metrics artifacts.

Run with PYTHONPATH=python and the repository's GPU-capable Python launcher.
Host validation coverage is in tests/test_regression_metrics.py. This program
never substitutes a host metric when a compiled mode is unavailable.
"""
import argparse
import json
import math
import struct

import numpy as np

from mojolearn import metrics, set_numeric_mode
from mojolearn import _backend

NAMES = ("mean_squared_error", "mean_absolute_error", "root_mean_squared_error")
CODES = {"fast": 0, "identical": 1, "deterministic": 2}


def oracle(y, p):
    # Independent scalar high-precision formula, no NumPy reduction and no
    # replication of the native reduction topology. Fixtures are dyadic, so
    # ordinary residuals are exactly representable at input precision.
    residuals = [float(a) - float(b) for a, b in zip(y.ravel(), p.ravel())]
    mse = math.fsum(r * r for r in residuals) / len(residuals)
    mae = math.fsum(abs(r) for r in residuals) / len(residuals)
    return mse, mae, math.sqrt(mse)


def bits(value):
    return struct.pack("<f", value).hex()


def fixtures():
    yield "singleton", np.array([2.5], np.float32), np.array([-1.5], np.float32)
    for n in (31, 32, 33, 255, 256, 257, 4097, 65539):
        i = np.arange(n, dtype=np.int64)
        y = (((i * 17) % 127) - 63).astype(np.float32) / 8
        p = (((i * 13) % 61) - 30).astype(np.float32) / 16
        yield f"ragged-{n}", y, p
    backing = np.arange(1030, dtype=np.float32) / 8
    backing.flags.writeable = False
    yield "strided-column", backing[::2], backing[::-2].reshape(-1, 1)
    y = np.array([-0.0, 0.0, 4, -8], np.float32)
    yield "perfect", y, y.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=tuple(CODES),
                        default=list(CODES))
    args = parser.parse_args()
    observed = {}
    checks = 0
    # Interleave same-process artifacts, then revisit each mode. Repetition
    # checks are within mode; FAST/DETERMINISTIC are not claimed cross-vendor.
    for mode in [*args.modes, *reversed(args.modes)]:
        binding = metrics._get_binding(mode)
        readback = getattr(binding, "metrics_numeric_mode", None)
        if readback is None:
            readback = binding.umap_numeric_mode
        assert readback() == CODES[mode], (mode, readback())
        vendor = binding.metrics_vendor()
        assert vendor in ("metal", "cuda", "hip"), vendor
        for label, y, p in fixtures():
            expected = oracle(y, p)
            original_y, original_p = y.copy(), p.copy()
            for name, truth in zip(NAMES, expected):
                actual = getattr(metrics, name)(y, p, numeric_mode=mode)
                assert type(actual) is float
                assert math.isclose(actual, truth, rel_tol=3e-5, abs_tol=2e-6), (
                    mode, label, name, actual, truth)
                key = (mode, label, name)
                if key in observed:
                    assert observed[key] == bits(actual), ("repeat bits", key)
                observed[key] = bits(actual)
                checks += 1
            np.testing.assert_array_equal(y, original_y)
            np.testing.assert_array_equal(p, original_p)
        maximum = np.finfo(np.float32).max
        overflow = [
            ("residual-overflow", np.array([maximum], np.float32),
             np.array([-maximum], np.float32), NAMES),
            ("square-overflow", np.array([1e20], np.float32),
             np.zeros(1, np.float32), (NAMES[0], NAMES[2])),
            ("sum-overflow", np.full(4, maximum / 2, np.float32),
             np.zeros(4, np.float32), NAMES),
        ]
        for label, y, p, names in overflow:
            for name in names:
                actual = getattr(metrics, name)(y, p, numeric_mode=mode)
                assert actual == math.inf, (mode, label, name, actual)
                checks += 1
        # Square overflow does not imply absolute-residual overflow.
        actual = metrics.mean_absolute_error(
            np.array([1e20], np.float32), np.zeros(1, np.float32), numeric_mode=mode)
        assert actual == float(np.float32(1e20)), (mode, "finite MAE", actual)
        checks += 1
        print(json.dumps({"mode": mode, "compiled_mode": readback(),
                          "vendor": vendor, "artifact": binding.__file__,
                          "checks_so_far": checks, "status": "PASS"}), flush=True)
    original_mode = _backend.default_mode()
    try:
        for mode in args.modes:
            set_numeric_mode(mode)
            assert metrics._get_binding().metrics_numeric_mode() == CODES[mode]
            _, y, p = next(fixtures())
            for name in NAMES:
                actual = getattr(metrics, name)(y, p)
                assert bits(actual) == observed[(mode, "singleton", name)]
                checks += 1
    finally:
        set_numeric_mode(original_mode)
    print(json.dumps({"status": "PASS", "checks": checks,
                      "fingerprints": {"/".join(key): val for key, val in observed.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
