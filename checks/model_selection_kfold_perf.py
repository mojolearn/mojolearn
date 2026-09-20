#!/usr/bin/env python3
"""Exactness and timing gate for plain Python-label KFold metadata."""
import argparse
import numbers
import statistics
import time

from mojolearn import _portable_math as math
from mojolearn._labels import flatten_labels
from mojolearn.model_selection import _default_folds


def _previous(labels, splits):
    labels = flatten_labels(labels)
    # Preserve the old path's classifier-discreteness census, even though a
    # regressor ignores its result, so this is an end-to-end comparison.
    _ = (all(isinstance(v, str) for v in labels) or
         all((isinstance(v, numbers.Integral) or
              (isinstance(v, numbers.Real) and math.isfinite(v)
               and float(v).is_integer())) for v in labels))
    n = len(labels)
    tests = []
    offset = 0
    for fold in range(splits):
        size = n // splits + (fold < n % splits)
        tests.append(list(range(offset, offset + size)))
        offset += size
    result = []
    for test in tests:
        heldout = set(test)
        result.append(([i for i in range(n) if i not in heldout], test))
    return result


def _timed(call, repeats):
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = call()
        samples.append(time.perf_counter() - start)
    return value, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=1_000_000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    labels = [float(i % 17) + 0.25 for i in range(args.rows)]
    old, old_s = _timed(lambda: _previous(labels, args.folds), args.repeats)
    new, new_s = _timed(
        lambda: list(_default_folds(labels, args.folds, False)), args.repeats)
    if new != old:
        raise SystemExit('FAIL: optimized KFold metadata changed indices or order')
    print(f'rows={args.rows} folds={args.folds} old={old_s:.6f}s '
          f'new={new_s:.6f}s speedup={old_s / new_s:.2f}x exact=yes')


if __name__ == '__main__':
    main()
