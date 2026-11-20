# SPDX-License-Identifier: Apache-2.0
"""Write the NumPy-generated inputs of the tsa and metrics native checks.

The checks below replaced Python tests whose inputs come from NumPy's PCG64
streams (`np.random.default_rng(seed)`); Mojo cannot regenerate those
streams, so the exact bytes are stored as raw little-endian 32-bit words:
`.f32` (float32) and `.i32` (int32). Matrices are row-major (series-major
for time series), the order the original test handed to the public call.
Every array is the one the original test built, with the original seed,
draw order and dtype; the originating test is named beside each block.

    checks/native_tests/fixtures/tsa/      tsa_sequence_ets_*, tsa_training_arima_*
    checks/native_tests/fixtures/metrics/  metrics_weighted_score_*,
                                           metrics_selection_host_p2_*,
                                           metrics_smallfam_public_*

The fixture sets are independent and are written one per process across
the host cores (`os.cpu_count()` workers).

    python tools/tsa_metrics_native_fixtures.py
"""
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "checks" / "native_tests" / "fixtures"


def _put(family, name, array, dtype):
    data = np.ascontiguousarray(np.asarray(array).astype(dtype)).ravel(order="C")
    (ROOT / family / name).write_bytes(data.astype(np.dtype(dtype).newbyteorder("<")).tobytes())
    return f"{family}/{name} {np.asarray(array).shape}"


def ets():
    # test_x_sequence_ets.py::_Y
    rng = np.random.default_rng(0)
    t = np.arange(80)
    y = np.stack([20 + 0.5 * t + rng.standard_normal(80), 50 + rng.standard_normal(80)]).astype(np.float32)
    return [_put("tsa", "ets-y.f32", y, np.float32)]


def arima():
    # test_cpu_training_arima.py::test_arima_fits_on_the_host_when_built
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.standard_normal((3, 160)), axis=1).astype(np.float32) * 0.1
    series += rng.standard_normal((3, 160)).astype(np.float32)
    assert series.dtype == np.float32
    return [_put("tsa", "arima-host-series.f32", series, np.float32)]


def weighted():
    # test_weighted_score.py::test_weighted_accuracy_and_r2_match_the_definition
    rng = np.random.default_rng(5)
    n = 4099
    yt = rng.integers(0, 3, n).astype(np.int32)
    yp = np.where(rng.random(n) < 0.7, yt, (yt + 1) % 3).astype(np.int32)
    w = rng.uniform(0.25, 4.0, n).astype(np.float32)
    w[::11] = 0.0
    y = rng.normal(size=n).astype(np.float32)
    yh = (y + rng.normal(scale=0.3, size=n)).astype(np.float32)
    return [_put("metrics", "weighted-yt.i32", yt, np.int32), _put("metrics", "weighted-yp.i32", yp, np.int32),
            _put("metrics", "weighted-w.f32", w, np.float32), _put("metrics", "weighted-y.f32", y, np.float32),
            _put("metrics", "weighted-yh.f32", yh, np.float32)]


def prc():
    # test_metrics_selection_host_p2.py::test_precision_recall_curve_matches_counts_at_each_threshold
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 40).astype(np.int64)
    s = (y * 0.6 + rng.random(40)).astype(np.float32)
    return [_put("metrics", "prc-y.i32", y, np.int32), _put("metrics", "prc-s.f32", s, np.float32)]


def lu():
    # test_native_smallfam_public.py::_system (n = 8, seed 5)
    n = 8
    rng = np.random.default_rng(5)
    a = (rng.random((n, n)) + n * np.eye(n)).astype(np.float32)
    b = rng.random(n).astype(np.float32)
    bm = rng.random((n, 3)).astype(np.float32)
    return [_put("metrics", "lu-a.f32", a, np.float32), _put("metrics", "lu-b.f32", b, np.float32),
            _put("metrics", "lu-bm.f32", bm, np.float32)]


SETS = (ets, arima, weighted, prc, lu)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
