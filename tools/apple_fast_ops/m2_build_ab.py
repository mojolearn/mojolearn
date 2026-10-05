#!/usr/bin/env python3
"""M2 A/B binding builder. Usage: m2_build_ab.py JOBS.json  (list of {sha,binding,A,B}). Sequential, one build at a time."""
import json, os, subprocess, sys, hashlib, shutil, pathlib
H = pathlib.Path.home(); DONOR = H/"m2c/lane_apple-fast-resample-gpu-recovery/.pixi"
def sh256(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
for j in json.load(open(sys.argv[1])):
    sha, b = j["sha"], j["binding"]; out = H/"m2-arms"/sha/(b+("-"+j["mode"] if j.get("mode") else "")); out.mkdir(parents=True, exist_ok=True)
    wt = H/"m2c"/f"build-{sha[:12]}"
    if not wt.exists():
        subprocess.run(["git","-C",str(H/"mojolearn.git"),"worktree","add","--detach",str(wt),sha],check=True)
    if not (wt/".pixi").exists(): os.symlink(DONOR, wt/".pixi")
    hashes, ok = {}, True
    for arm in [x for x in ("A","B") if j.get(x) is not None]:
        env = dict(os.environ, PATH=str(H/".pixi/bin")+":"+os.environ["PATH"], MOJOLEARN_COMPILE_JOBS="1", MOJOLEARN_NUMERIC_MODE=j.get("mode","fast"),
                   MOJOLEARN_SKIP_BUILD_GATE="1", MOJOLEARN_MOJO_BUILD_FLAGS=j[arm])
        so = wt/"python/mojolearn"/("identical/" if j.get("mode")=="identical" else "")/("_mojolearn.so" if b=="core" else f"_mojolearn_{b}.so")
        if so.exists(): so.unlink()
        with open(out/f"{arm}-build.log","w") as log:
            rc = subprocess.run(["nice","-n","19","bash",("bindings/build.sh" if b=="core" else f"bindings/build_{b}.sh")],cwd=wt,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL).returncode
        if rc != 0 or not so.exists():
            ok = False; print(f"{sha[:9]} {b} {arm} FAILED rc={rc}", flush=True); break
        shutil.copy2(so, out/f"{arm}.so"); hashes[arm] = sh256(out/f"{arm}.so")
        print(f"{sha[:9]} {b} {arm} COMPILED {hashes[arm][:16]}", flush=True)
    json.dump({"source_sha":sha,"binding":b,"numeric_mode":j.get("mode","fast"),"defines_A":j["A"],"defines_B":j.get("B"),
               "builder":"m2","hashes":hashes,"status":"OK" if ok else "FAILED"}, open(out/"manifest.json","w"), indent=2)
print("ALL_DONE", flush=True)
for w in set(str(H/"m2c"/f"build-{j['sha'][:12]}") for j in json.load(open(sys.argv[1]))):
    subprocess.run(["git","-C",str(H/"mojolearn.git"),"worktree","remove","--force",w])
