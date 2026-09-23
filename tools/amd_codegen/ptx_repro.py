# Per (variant, binding): distinct embedded PTX module sets across replicates.
import sys, os, glob, collections, hashlib, re
def ptx(path):
    d = open(path, 'rb').read(); out = []; seen = set()
    for m in re.finditer(rb'\.version \d+\.\d+', d):
        s = d.rfind(b'\x00', 0, m.start()) + 1
        if s in seen: continue
        seen.add(s); e = d.find(b'\x00', m.start())
        out.append(hashlib.sha256(d[s:e]).hexdigest()[:12])
    return out
root = sys.argv[1]; reps = collections.defaultdict(dict)
for dd in sorted(glob.glob(os.path.join(root, '*-*'))):
    v, r = os.path.basename(dd).rsplit('-', 1)
    for so in glob.glob(os.path.join(dd, '*.so')):
        reps[(v, os.path.basename(so))][r] = ptx(so)
bad = 0
for (v, b), rr in sorted(reps.items()):
    sets = collections.Counter(tuple(x) for x in rr.values())
    n = len(next(iter(rr.values())))
    if len(sets) > 1: bad += 1
    print(f"{v:7s} {b:30s} builds={len(rr)} ptx_modules={n:3d} distinct_sets={len(sets)} {'REPRODUCIBLE' if len(sets)==1 else 'VARIES'}")
print("varying bindings:", bad)
