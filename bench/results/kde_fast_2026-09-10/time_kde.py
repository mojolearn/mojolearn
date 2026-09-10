# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 2490 timing: KernelDensity.score_samples, our FAST arm on the
Mac against scikit-learn's KD-tree at rtol=atol=0 (its exact path).

Protocol (bench/results/native_convert_2026-09-10 and the box-drift note):
one untimed warm-up per arm, then PAIRS alternating sklearn / ours inside
one window; the MINIMUM per arm is the number; the first-to-last sklearn
spread is printed and a spread above 20% on a >= 1 ms case voids the window.
sklearn is single-threaded here (KDTree.kernel_density has no n_jobs), so
this is our GPU against one core, and the log says so.

    MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python nice -n 19 \
        python3 bench/results/kde_fast_2026-09-10/time_kde.py [--pairs 5]

Large shapes run OURS ALONE (sklearn would take minutes per fit) and
report cells per second; they are not a comparison.
"""
import argparse
import sys
import time

import numpy as np

from mojolearn import KernelDensity

p = argparse.ArgumentParser()
p.add_argument("--pairs", type=int, default=5)
p.add_argument("--skip-sklearn", action="store_true")
args = p.parse_args()

try:
    import sklearn
    import sklearn.neighbors as skn
except ImportError:
    skn = None

est0 = KernelDensity()
print(f"mode={est0.numeric_mode_used()} vendor={est0.vendor_used()} sklearn={getattr(sklearn, '__version__', None) if skn else None} numpy={np.__version__}")
rng = np.random.default_rng(0)


def ours(X, Xq, h=0.5):
    k = KernelDensity(bandwidth=h).fit(X)
    t = time.perf_counter()
    s = k.score_samples(Xq)
    return time.perf_counter() - t, np.asarray(s, dtype=np.float64)


def theirs(X, Xq, h=0.5):
    k = skn.KernelDensity(bandwidth=h, rtol=0, atol=0).fit(X)
    t = time.perf_counter()
    s = k.score_samples(Xq)
    return time.perf_counter() - t, s


print("\n# paired: sklearn KDTree (one core, exact) vs ours FAST; min of pairs, ms")
print("| train x query x d | sklearn min ms | ours min ms | ratio sklearn/ours | max abs diff | sklearn first->last spread |")
print("|---|---:|---:|---:|---:|---:|")
if skn is not None and not args.skip_sklearn:
    for n, d in ((2000, 8), (8000, 8), (16000, 8), (16000, 32)):
        X = rng.standard_normal((n, d)).astype(np.float32)
        ours(X, X[:64]); theirs(X, X[:64])
        sk, us = [], []
        diff = 0.0
        for _ in range(args.pairs):
            t1, s1 = theirs(X, X); sk.append(t1)
            t2, s2 = ours(X, X); us.append(t2)
            diff = max(diff, float(np.abs(s1 - s2).max()))
        spread = (sk[-1] - sk[0]) / sk[0] * 100
        void = " VOID" if abs(spread) > 20 else ""
        print(f"| {n} x {n} x {d} | {min(sk)*1e3:.1f} | {min(us)*1e3:.1f} | {min(sk)/min(us):.1f}x | {diff:.1e} | {spread:+.1f}%{void} |")
        sys.stdout.flush()
else:
    print("| (sklearn not importable; paired block skipped) | | | | | |")

print("\n# ours alone, min of 3, the shapes the staged path could not allocate")
print("| train x query x d | ours min ms | Gcells/s |")
print("|---|---:|---:|")
for n, nq, d in ((50000, 50000, 8), (100000, 100000, 8), (100000, 100000, 32), (1000000, 20000, 8), (20000, 20000, 64), (20000, 20000, 100)):
    X = rng.standard_normal((n, d)).astype(np.float32)
    Xq = rng.standard_normal((nq, d)).astype(np.float32)
    k = KernelDensity(bandwidth=0.5).fit(X)
    k.score_samples(Xq[:64])
    best = min(ours(X, Xq)[0] for _ in range(3))
    print(f"| {n} x {nq} x {d} | {best*1e3:.1f} | {n*nq/best/1e9:.1f} |")
    sys.stdout.flush()
