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

WHAT THIS MEASURED, AND IT CLOSED AN OPEN QUESTION. MinEntropy had to be
switched off `std.math.log` because its ~5e-8 absolute error re-decided
plateau tie-breaks. GreedyLogSum was suspected of the same exposure. It is
NOT exposed, and this fixture is how we know: swapping in libm changes
nothing here, while flipping ONE comparison in the heap pop breaks 4 of 6
cases. The difference is structural. MinEntropy's tied costs are reached by
DIFFERENT summation paths, so noise pulls them apart; GreedyLogSum's ties
come from IDENTICAL integer bin sizes, so both sides are computed from the
same inputs and any deterministic log returns the same bits.
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

CASES = [(256, 15), (256, 100), (256, 37), (500, 100), (500, 63), (1000, 200)]
print("columns %d" % len(CASES))
for i,(nv,_) in enumerate(CASES):
    v = col(nv)
    print("column %d %d" % (i, len(v)))
    for x in v: print("%.9g" % x)
print("cases %d" % len(CASES))
for i,(nv,b) in enumerate(CASES):
    bs = borders(col(nv), b)
    print("case %d %d %d" % (i, b, len(bs)))
    for x in bs: print("%.9g" % x)
