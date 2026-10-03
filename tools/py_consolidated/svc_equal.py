"""svc_equal.py: the Mojo twins in svm/host/svc_proba.mojo against the Python
reference in _svm_impl.py (and _portable_math's C library), bit for bit, on
random and edge inputs. Run with PYTHONPATH=<tree>/python; MOJOLEARN_VENDOR=cpu
for the host binding. Prints one line per check and EQUAL RESULT PASS|FAIL."""
import array, os, random, struct, sys
from mojolearn import _svm_impl as S, _portable_math as pm
from mojolearn.tests import _svm_reference as R
b = S._extension()
print("binding", getattr(b, "__name__", b), "vendor", os.environ.get("MOJOLEARN_VENDOR", "gpu"))
bad = 0
rng = random.Random(7)
bits = lambda x: struct.pack("<d", x)

def check(name, ok, detail=""):
    global bad
    bad += 0 if ok else 1
    print(("OK   " if ok else "FAIL ") + name + (" " + detail if detail else ""), flush=True)

# 1. exp / log twins
xs = [0.0, -0.0, 1.0, -1.0, 709.782712893384, 709.78, -708.3964185322641, -708.39, 745.0, -745.0,
      float("inf"), float("-inf"), float("nan"), 5e-324, 2.2250738585072014e-308, 1e-310, 0.5, 2.0, 4.0, 0.25, 8.0,
      0.7071067811865476, 0.7071067811865475, 1e300, 1.7976931348623157e308]
for _ in range(1000000):
    k = rng.random()
    if k < 0.3:
        xs.append(rng.uniform(-750, 750))
    elif k < 0.5:
        xs.append(rng.uniform(-3, 3))
    elif k < 0.8:
        xs.append(struct.unpack("<d", struct.pack("<Q", rng.getrandbits(63)))[0])
    else:   # near the k reduction boundary (x * log2 e + 0.5 near an integer)
        xs.append((rng.randint(-1000, 1000) - 0.5 + rng.uniform(-1e-12, 1e-12)) / 1.4426950408889634)
xs = [x for x in xs]
inp = array.array("d", xs); out = array.array("d", bytes(8 * len(xs)))
for which, ref, nm in ((0, "exp", "pm_exp"), (1, "log", "pm_log")):
    b.svc_portable_math(inp.buffer_info()[0], out.buffer_info()[0], len(xs), which)
    miss = 0; first = None
    for x, y in zip(xs, out):
        r = pm._native(ref, x)
        if bits(r) != bits(y) and not (r != r and y != y):
            miss += 1; first = first or (x, r, y)
    check(f"{nm} == portable_math.c over {len(xs)}", miss == 0, f"{miss} differ, first {first}")

# 2. SplitMix perm
for n, seed in ((1, 0), (2, 5), (50, 3), (1000, 2**64 - 1), (4097, 0x9E3779B97F4A7C15 * 3 % 2**64), (100000, 12345)):
    check(f"splitmix n={n}", list(S._native_perm(b, n, seed)) == R._splitmix_perm(n, seed))

# 3. Platt sigmoid_train
for trial in range(60):
    n = rng.choice([1, 2, 5, 17, 100, 1000, 5000])
    sc = rng.choice([0.01, 1.0, 10.0, 300.0])
    dec = [rng.gauss(0, 1) * sc for _ in range(n)]
    if trial % 7 == 0:
        dec = [round(d) for d in dec]
    labels = [1.0 if (d + rng.gauss(0, 1.5 * sc) > 0) else -1.0 for d in dec]
    if trial % 11 == 0:
        labels = [1.0] * n
    try:
        ref = R._sigmoid_train(dec, labels); rerr = None
    except Exception as e:
        ref, rerr = None, type(e).__name__
    try:
        got = S._native_sigmoid_train(b, dec, labels); gerr = None
    except Exception as e:
        got, gerr = None, type(e).__name__
    ok = (rerr is not None and gerr is not None) or (ref is not None and got is not None and bits(ref[0]) == bits(got[0]) and bits(ref[1]) == bits(got[1]))
    check(f"sigmoid_train trial {trial} n={n}", ok, f"ref={ref}/{rerr} got={got}/{gerr}")

# 4. epilogues vs the Python reference, per mode
class Fake(S.SVC):
    pass

for k in (2, 3, 4, 7, 10):
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    P = len(pairs); n = 3000
    vals = []
    for _ in range(P * n):
        u = rng.random()
        vals.append(0.0 if u < 0.05 else (-0.0 if u < 0.07 else (float(round(rng.gauss(0, 2))) if u < 0.2 else rng.gauss(0, 2))))
    f32 = array.array("f", vals); vals = list(f32)
    dec = S.Array.from_list([vals[p * n:(p + 1) * n] for p in range(P)], "<f4")
    ab = [(rng.gauss(-1.5, 0.5), rng.gauss(0, 0.3)) for _ in range(P)]
    m = Fake.__new__(Fake)
    m.classes_ = list(range(k)); m.numeric_mode = None
    m._pairs = [dict(i=i, j=j) for i, j in pairs]
    rows = [[vals[p * n + r] for p in range(P)] for r in range(n)]
    lists = [vals[p * n:(p + 1) * n] for p in range(P)]
    # ovo
    got = m._epilogue(S._EPI_OVO, dec, pairs, (n, P), "<f4").tolist()
    ref = S.Array.from_list([[-d[r] for d in lists] for r in range(n)], "<f4").tolist()
    check(f"ovo k={k}", [[bits(v) for v in r] for r in got] == [[bits(v) for v in r] for r in ref])
    got = m._epilogue(S._EPI_OVR, dec, pairs, (n, k), "<f8").tolist()
    ref = R.ovr_scores(m, lists, n)
    check(f"ovr k={k}", [[bits(v) for v in r] for r in got] == [[bits(v) for v in r] for r in ref])
    got = m._epilogue(S._EPI_VOTES, dec, pairs, (n,), "<i8").tolist()
    ref = []
    for r in range(n):
        votes = [0] * k
        for (i, j), d in zip(pairs, lists):
            votes[j if d[r] >= 0.0 else i] += 1
        best = 0
        for c in range(1, k):
            if votes[c] > votes[best]:
                best = c
        ref.append(best)
    check(f"votes k={k}", got == ref)
    got = m._epilogue(S._EPI_PROBA, dec, pairs, (n, k), "<f8", ab=ab).tolist()
    ref = []
    for r in range(n):
        mm = [[0.0] * k for _ in range(k)]
        for (i, j), d, (a, bb) in zip(pairs, lists, ab):
            v = R._sigmoid_predict(-float(d[r]), a, bb)
            v = min(max(v, 1e-7), 1.0 - 1e-7)
            mm[i][j] = v; mm[j][i] = 1.0 - v
        ref.append([mm[0][1], mm[1][0]] if k == 2 else R._multiclass_probability(k, mm))
    check(f"proba k={k}", [[bits(v) for v in r] for r in got] == [[bits(v) for v in r] for r in ref])
    got = m._epilogue(S._EPI_LOG_PROBA, dec, pairs, (n, k), "<f8", ab=ab).tolist()
    refl = [[pm.log(v) for v in r] for r in ref]
    check(f"log_proba k={k}", [[bits(v) for v in r] for r in got] == [[bits(v) for v in r] for r in refl])
# binary codes
raw = S.Array.from_list([0.0, 1.0, 1.0, 0.0, 2.0], "<f4")
got = Fake.__new__(Fake); got.classes_ = [0, 1]; got.numeric_mode = None
codes = got._epilogue(S._EPI_BINARY, raw.reshape((1, 5)), [(0, 1)], (5,), "<i8", label1=1.0).tolist()
check("binary codes", codes == [0, 1, 1, 0, 0])
print("EQUAL RESULT " + ("PASS" if not bad else f"FAIL ({bad})"))
sys.exit(1 if bad else 0)
