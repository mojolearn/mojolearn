import sys,re,json,collections
# summarize a trees_apple_ab.sh stdout: per (arm, cell) the ms list and digests
d=collections.defaultdict(lambda: {"ms":[], "dig":set()})
cur={}
for line in open(sys.argv[1]):
    m=re.match(r"\[(\w+)\] (.*)",line.rstrip())
    if not m: continue
    arm,rest=m.groups()
    h=re.match(r"=== gbdt:([\w-]+):(\w+)$",rest)
    if h: cur[arm]=h[2]; continue
    f=re.match(r"FTRAIN lane=(\w+) dataset=(\w+) .*ms=([\d.]+) hash=(\w+)",rest)
    if f:
        k=(f"{f[1]}:{f[2]}",arm); d[k]["ms"].append(float(f[3])); d[k]["dig"].add(f[4]); continue
    if rest.startswith("TAP {"):
        j=json.loads(rest[4:]); k=(f"{j['est']}:{j['dataset']}",arm)
        d[k]["ms"]+=j["ms"]; d[k]["dig"].update(j["digests"]); continue
    g=re.match(r"GTP LINE .*cell=(\S+) fixed_ms=([-\d.]+) per_tree_ms=([\d.]+) at10=([\d.]+) at100=([\d.]+)",rest)
    if g:
        k=(g[1]+":"+cur.get(arm,"?"),arm); d[k]["ms"].append(float(g[5])); continue
    g=re.match(r"GTP FIT .*cell=(\S+) trees=100 .*digest=(\w+)",rest)
    if g: d[(g[1]+":"+cur.get(arm,"?"),arm)]["dig"].add(g[2])
cells=sorted({c for c,_ in d}); arms=[]
for c,a in d:
    if a not in arms: arms.append(a)
for c in cells:
    row=[c]
    base=None
    for a in arms:
        v=d.get((c,a))
        if not v or not v["ms"]: row.append(f"{a}=-"); continue
        med=sorted(v["ms"])[len(v["ms"])//2] if len(v["ms"])%2 else sum(sorted(v["ms"])[len(v["ms"])//2-1:len(v["ms"])//2+1])/2
        row.append(f"{a}={med:.0f} {v['ms']} {','.join(sorted(v['dig']))[:40]}")
        if base is None: base=med
        else: row.append(f"ratio={med/base:.3f}")
    print(" | ".join(row))
