# Per (variant, binding): distinct EMBEDDED GPU CODE OBJECT sets across replicates.
import sys, os, glob, collections, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_amdgpu as e
root = sys.argv[1]
reps = collections.defaultdict(dict)
for d in sorted(glob.glob(os.path.join(root, '*-*'))):
    v, r = os.path.basename(d).rsplit('-', 1)
    for so in glob.glob(os.path.join(d, '*.so')):
        reps[(v, os.path.basename(so))][r] = [x[2] for x in e.extract(so)]
bad = 0
for (v, b), rr in sorted(reps.items()):
    sets = collections.Counter(tuple(x) for x in rr.values())
    n = len(next(iter(rr.values())))
    vary = [i for i in range(n) if len({tuple(x)[i] if i < len(x) else None for x in rr.values()}) > 1]
    flag = 'REPRODUCIBLE' if len(sets) == 1 else 'VARIES'
    if len(sets) > 1: bad += 1
    print(f"{v:6s} {b:32s} builds={len(rr)} code_objects={n:3d} distinct_sets={len(sets)} {flag}" + (f" varying_objects={vary}" if vary else ""))
print("varying bindings:", bad)
