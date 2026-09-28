# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane py-shared micro board: each shared fast path against the routine it
stands in for, on the same machine in one process, with the answers compared.

  fsum      `_portable_math.fsum` (compiled math.fsum fast path) vs
            `_portable_math._fsum_exact` (the exact big-integer sum)
  isfinite  the float fast path vs the struct bit test
  labels    `_labels.encode_labels` vs `sorted_classes` (+ the code list) on
            1-D int labels as a Python list (SVC, LinearSVC, model_selection
            groups), an int32 buffer (silhouette) and float labels

Lines: `PYSHARED <case> <n> <old_s> <new_s> <ratio> SAME|DIFF`.

    python bench/py_shared_micro.py [--n 1000000] [--reps 3]
"""
import argparse
import os
import random
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _best(fn, reps):
    best, out = None, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        t = time.perf_counter() - t0
        best = t if best is None or t < best else best
    return best, out


def _line(case, n, old, new, same):
    print(f"PYSHARED {case} {n} {old:.4f} {new:.4f} {old / max(new, 1e-9):.1f}x {'SAME' if same else 'DIFF'}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    from mojolearn import _portable_math as pm
    from mojolearn import _labels
    from mojolearn._array import Array

    n = a.n
    rng = random.Random(7)
    xs = [rng.gauss(0.0, 1.0) * 10.0 ** rng.randint(-8, 8) for _ in range(n)]
    to, ro = _best(lambda: pm._fsum_exact(xs), 1)
    tn, rn = _best(lambda: pm.fsum(xs), a.reps)
    _line("fsum", n, to, tn, struct.pack("<d", ro) == struct.pack("<d", rn))

    def slow_isfinite(x):
        return (pm._bits(x) & 0x7ff0000000000000) != 0x7ff0000000000000
    zs = xs[:]
    zs[::1000] = [float("inf")] * len(zs[::1000])
    to, ro = _best(lambda: [slow_isfinite(x) for x in zs], a.reps)
    tn, rn = _best(lambda: [pm.isfinite(x) for x in zs], a.reps)
    _line("isfinite", n, to, tn, ro == rn)

    def old_route(labels):
        classes, codes = _labels.sorted_classes(list(labels))
        return classes, codes

    def new_route(labels):
        classes, codes = _labels.encode_labels(labels)
        return classes, codes.tolist()

    ints = [rng.randrange(10) for _ in range(n)]
    floats = [float(v) * 0.5 for v in ints]
    i32 = Array.from_list(ints, "<i4")
    for case, lab, old_in in (("labels_int_list", ints, ints), ("labels_float_list", floats, floats),
                              ("labels_i32_buffer", i32, None)):
        old_fn = (lambda l=old_in: old_route(l)) if old_in is not None else (lambda: old_route(i32.tolist()))
        to, ro = _best(old_fn, a.reps)
        tn, rn = _best(lambda l=lab: new_route(l), a.reps)
        same = ro[1] == rn[1] and [(type(c), c) for c in ro[0]] == [(type(c), c) for c in rn[0]]
        _line(case, n, to, tn, same)


if __name__ == "__main__":
    main()
