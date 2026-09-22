# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost's GreedyLogSum borders on tie-heavy columns.

    pixi run -e bench python tools/greedylogsum_oracle.py > bench/greedylogsum_oracle.txt

GreedyLogSum is the NUMERIC border default, so it is on the path of every
dataset this repository benchmarks. Its score is log(n) - log(l) - log(r)
over INTEGER bin sizes, so equal-size bins tie EXACTLY, and near-halving
mints whole tiers of them. Which of a tied set gets split is decided by
std::priority_queue's heap order, which `best_split` reproduces down to
libc++'s semantics.

BUDGETS THAT LAND ON A COMPLETE TIER ARE ORDER-INVARIANT AND PROVE NOTHING.
15 = 1+2+4+8 is exactly such a budget, and it is why an earlier list-scan
implementation of the queue passed for months while getting 1392 of 1600
borders wrong at budget 100. These cases cut tiers mid-way: 37, 63, 100 and
200. Budget 15 is kept deliberately as the control that stays green under
sabotage.

NEAR TIES, AND WHY THE LOG MATTERS AFTER ALL. The `1e-8` inside the
penalty makes scores that are mathematically equal across DIFFERENT bin
sizes differ by ~1e-10 (3+6 of 9 and 4+4 of 8 both score log 2, separated
by 1.4e-10), which is inside `std.math.log`'s ~5e-11 error per call. The
tier columns above never reach such a pair; small duplicate-heavy random
columns do. A pip-install smoke test found `std.math.log` disagreeing with
CatBoost on 4 of 120 of them (2026-09-22), so the penalty now calls
`portable_log64` and the RANDOM cases below (seeded, 8 to 120 rows drawn
from a small pool of two-decimal values) keep it honest.
"""
import sys, os, tempfile
import numpy as np, catboost

def col(nvals, seed=5):
    rng = np.random.default_rng(seed)
    # many values repeated an EQUAL number of times -> equal bin sizes -> exact ties
    reps = 8
    base = np.arange(nvals, dtype=np.float32)
    v = np.repeat(base, reps)
    rng.shuffle(v)
    return v.astype(np.float32)

def borders(v, budget):
    y = np.random.default_rng(0).normal(size=len(v)).astype(np.float32)
    pool = catboost.Pool(v.reshape(-1,1), y)
    pool.quantize(border_count=budget)   # default border type = GreedyLogSum
    with tempfile.TemporaryDirectory() as td:
        bp = os.path.join(td,"b.tsv"); pool.save_quantization_borders(bp)
        return sorted(float(l.strip().split("\t")[1]) for l in open(bp) if l.strip())

def random_cols(trials=120, seed=0):
    """Small duplicate-heavy columns, each with a budget below its number
    of distinct values: the near-tie shape (see the docstring)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(trials):
        n = rng.randint(8, 120)
        pool = np.round(rng.randn(rng.randint(3, n)) * 100) / 100
        v = rng.choice(pool, n).astype(np.float32)
        u = len(np.unique(v))
        if u < 3:
            continue
        out.append((v, int(rng.randint(1, u))))
    return out

CASES = [(256, 15), (256, 100), (256, 37), (500, 100), (500, 63), (1000, 200)]
COLS = [(col(nv), b) for nv, b in CASES] + random_cols()
print("columns %d" % len(COLS))
for i,(v,_) in enumerate(COLS):
    print("column %d %d" % (i, len(v)))
    for x in v: print("%.9g" % x)
print("cases %d" % len(COLS))
for i,(v,b) in enumerate(COLS):
    bs = borders(v, b)
    print("case %d %d %d" % (i, b, len(bs)))
    for x in bs: print("%.9g" % x)
