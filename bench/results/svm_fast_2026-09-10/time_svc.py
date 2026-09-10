# SPDX-License-Identifier: Apache-2.0
"""DEVIATIONS 2491 (shuffle arg-reductions in the SMO block solve) and 2492
(fused FAST RBF tile): SVC fit and predict, our FAST arm on the Mac against
scikit-learn's libsvm (single-threaded by construction), HIGGS prefixes.

Protocol: one warm-up per arm at the smallest shape, then PAIRS alternating
sklearn / ours per shape; MINIMUM per arm; the sklearn first-to-last spread
is printed and a spread above 20% voids the window. Accuracy on a 10,000-row
HIGGS tail is printed beside the times so the FAST arm's gate (accuracy,
never bits) is in the same log. Shapes above 20,000 rows run ours alone
(libsvm is quadratic-plus there and would take minutes per fit).

    MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python nice -n 19 python3 \
        bench/results/svm_fast_2026-09-10/time_svc.py --data HIGGS.f32.npy
"""
import argparse
import sys
import time

import numpy as np

from mojolearn import SVC

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--pairs", type=int, default=3)
args = p.parse_args()
try:
    import sklearn
    from sklearn.svm import SVC as SkSVC
except ImportError:
    sklearn = None

D = np.load(args.data, mmap_mode="r")
Xt = np.ascontiguousarray(D[10_500_000:10_510_000, 1:])
yt = np.ascontiguousarray(D[10_500_000:10_510_000, 0]).astype(np.int32)
est0 = SVC()
print(f"mode={est0.numeric_mode_used()} vendor={est0.vendor_used()} sklearn={sklearn.__version__ if sklearn else None} numpy={np.__version__}")


def prep(n):
    X = np.ascontiguousarray(D[:n, 1:])
    y = np.ascontiguousarray(D[:n, 0]).astype(np.int32)
    g = 1.0 / (X.shape[1] * float(X.var()))
    return X, y, g


def ours(X, y, g):
    m = SVC(C=1.0, kernel="rbf", gamma=g)
    t = time.perf_counter(); m.fit(X, y); ft = time.perf_counter() - t
    t = time.perf_counter(); pr = np.asarray(m.predict(Xt)); pt = time.perf_counter() - t
    return ft, pt, float((pr == yt).mean()), int(np.asarray(m.n_support_).sum())


def theirs(X, y, g):
    m = SkSVC(C=1.0, kernel="rbf", gamma=g, cache_size=2000)
    t = time.perf_counter(); m.fit(X, y); ft = time.perf_counter() - t
    t = time.perf_counter(); pr = m.predict(Xt); pt = time.perf_counter() - t
    return ft, pt, float((pr == yt).mean()), int(m.n_support_.sum())


print("\n# paired: sklearn libsvm (one core) vs ours FAST; HIGGS prefix, 28 features, RBF C=1 gamma=1/(d var); min of pairs")
print("| rows | sklearn fit ms | ours fit ms | fit ratio | sklearn predict 10k ms | ours predict 10k ms | predict ratio | acc sklearn / ours | n_sv sklearn / ours | sklearn fit spread |")
print("|---:|---:|---:|---:|---:|---:|---:|---|---|---:|")
X, y, g = prep(2000)
ours(X, y, g)
if sklearn:
    theirs(X, y, g)
for n in (5000, 20000):
    X, y, g = prep(n)
    sk, us = [], []
    for _ in range(args.pairs):
        if sklearn:
            sk.append(theirs(X, y, g))
        us.append(ours(X, y, g))
    if sklearn:
        skf = min(r[0] for r in sk); skp = min(r[1] for r in sk)
        spread = (sk[-1][0] - sk[0][0]) / sk[0][0] * 100
    of = min(r[0] for r in us); op = min(r[1] for r in us)
    if sklearn:
        print(f"| {n} | {skf*1e3:.0f} | {of*1e3:.0f} | {skf/of:.1f}x | {skp*1e3:.0f} | {op*1e3:.0f} | {skp/op:.1f}x | {sk[0][2]:.4f} / {us[0][2]:.4f} | {sk[0][3]} / {us[0][3]} | {spread:+.1f}%{' VOID' if abs(spread) > 20 else ''} |")
    else:
        print(f"| {n} | n/a | {of*1e3:.0f} | | n/a | {op*1e3:.0f} | | n/a / {us[0][2]:.4f} | n/a / {us[0][3]} | |")
    sys.stdout.flush()

print("\n# ours alone, min of 3")
print("| rows | fit ms | predict 10k ms | acc | n_sv |")
print("|---:|---:|---:|---:|---:|")
for n in (50000, 100000):
    X, y, g = prep(n)
    rs = [ours(X, y, g) for _ in range(3)]
    print(f"| {n} | {min(r[0] for r in rs)*1e3:.0f} | {min(r[1] for r in rs)*1e3:.0f} | {rs[0][2]:.4f} | {rs[0][3]} |")
    sys.stdout.flush()
