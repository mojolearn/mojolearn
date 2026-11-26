"""Count, over consecutive commits, how many build groups keep the same key (paths and toolchain held fixed)."""
import json, subprocess, sys, os, collections
repo, wt, out, revs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
env = dict(os.environ, MOJOLEARN_GPU_ARCHS="sm_89")
keys = {}
for r in revs:
    subprocess.run(["git", "-C", wt, "checkout", "-q", "--detach", r], check=True)
    j = f"{out}/plan-{r}.json"
    p = subprocess.run([sys.executable, "tools/development_build.py", "plan", "--sets", "identical,host",
                        "--cache", f"{out}/cache", "--json", j], cwd=wt, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(r, "plan rc", p.returncode, p.stderr[-200:].replace("\n", " ")); keys[r] = None; continue
    d = json.load(open(j))
    keys[r] = {f"{g['mode']}:{g['stem']}": (g["key"], g["fields"]["source"]["files"]) for g in d["groups"]}
    os.remove(j)
tot = same = 0; rows = []
for new, old in zip(revs, revs[1:]):
    a, b = keys[new], keys[old]
    if not a or not b: continue
    common = set(a) & set(b); kept = sum(a[k][0] == b[k][0] for k in common)
    srcsame = sum(a[k][1] == b[k][1] for k in common)
    files = subprocess.run(["git", "-C", wt, "diff", "--name-only", old, new], capture_output=True, text=True).stdout.split()
    mojo = [f for f in files if f.endswith(".mojo")]
    rows.append((new, len(a), kept, srcsame, len(files), len(mojo)))
    tot += len(a); same += kept
print("commit groups key_same source_same files_changed mojo_changed")
for r in rows: print(*r)
print(f"TOTAL groups {tot} reused {same} = {100*same/max(tot,1):.0f}%")
hist = collections.Counter(min(10, round(10*r[2]/r[1])) for r in rows)
print("pairs by reuse decile:", dict(sorted(hist.items())))
