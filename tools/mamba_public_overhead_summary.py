#!/usr/bin/env python3
"""Strict promotion summary for the two-dataset Mamba resident trial."""
import argparse, hashlib, json, math, statistics
from pathlib import Path

EXPECTED={
 "taxi":("gbm-bench/taxi/taxi_speed.npz",419757252,"10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15"),
 "istella":("gbm-bench/istella/istella_speed.npz",2248281826,"31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef"),
}
SHAPES={"screen":{"dm":128,"prefill":32,"tokens":16,"grad_length":8},
        "qualification":{"dm":1024,"prefill":256,"tokens":64,"grad_length":32}}
FAMILIES=("mamba2","mamba3")
HEX=set("0123456789abcdef")

def h64(x): return isinstance(x,str) and len(x)==64 and set(x)<=HEX

def common_errors(r,rung,ds,family,process,sabotage):
 e=[]; vendor={"nvidia":"cuda","amd":"hip"}.get(r.get("target_column")); d=r.get("dataset",{})
 if r.get("schema")!="mojolearn.mamba-public-overhead.v1": e.append("schema")
 if r.get("rung")!=rung or r.get("shape")!=SHAPES[rung] or r.get("family")!=family: e.append("identity/shape")
 if r.get("process")!=process or r.get("warmup")!=1 or r.get("sabotage")!=sabotage: e.append("schedule")
 if (d.get("key"),d.get("bytes"),d.get("sha256"))!=EXPECTED[ds]: e.append("dataset")
 if r.get("native_vendor")!=vendor or r.get("native_numeric_mode")!=1: e.append("native provenance")
 if not h64(r.get("binding",{}).get("sha256")) or r.get("binding",{}).get("bytes",0)<=0: e.append("binding")
 if set(r.get("source_sha256",{}))!={"tools/mamba_public_overhead_trial.py","python/mojolearn/_mamba_impl.py","bindings/_mojolearn_mamba.mojo"} or not all(h64(v) for v in r.get("source_sha256",{}).values()): e.append("source")
 if not all(h64(r.get(k)) for k in ("gradient_hash","gradient_after_hash","input_hash","gradient_input_hash","weights_hash","weights_after_hash")): e.append("top hashes")
 if r.get("gradient_hash")!=r.get("gradient_after_hash") or r.get("weights_hash")!=r.get("weights_after_hash"): e.append("gradient/weight mutation")
 return e

def validate_arm(arm,tokens,resident):
 e=[]; samples=arm.get("samples",[]); traj=arm.get("trajectory",[])
 if len(samples)!=tokens or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in samples): e.append("samples")
 if len(traj)!=tokens: e.append("trajectory count")
 for step in traj:
  if set(step)!={"output","reports","state"} or not all(h64(step.get(k)) for k in ("output","reports","state")): e.append("trajectory hash")
 if not all(h64(arm.get(k)) for k in ("output","state","quality_bits")): e.append("final hashes")
 if samples and arm.get("median")!=statistics.median(samples): e.append("arm median")
 if resident and (arm.get("native_session") is not True or arm.get("ownership_refused") is not True): e.append("native/ownership witness")
 if not resident and (arm.get("native_session") is not None or arm.get("ownership_refused") is not None): e.append("percall route witness")
 return e

def validate_clean(r):
 e=[]; rows=r.get("rows",[]); tokens=r.get("shape",{}).get("tokens",-1)
 if r.get("rounds")!=3 or len(rows)!=4: return ["row count"]
 expect=[(["percall","resident"],True),(["resident","percall"],False),(["percall","resident"],False),(["resident","percall"],False)]
 for i,(row,(order,warm)) in enumerate(zip(rows,expect)):
  if row.get("round")!=i or row.get("order")!=order or row.get("warmup") is not warm: e.append(f"row{i} order")
  for name,res in (("percall",False),("resident",True)): e += [f"row{i}/{name}/{x}" for x in validate_arm(row.get(name,{}),tokens,res)]
  p,q=row.get("percall",{}),row.get("resident",{})
  for k in ("output","state","trajectory","quality_bits"):
   if p.get(k)!=q.get(k): e.append(f"row{i} exact {k}")
 meds={name:statistics.median(row[name]["median"] for row in rows[1:]) for name in ("percall","resident")}
 if r.get("medians")!=meds or r.get("speedup")!=meds["percall"]/meds["resident"]: e.append("summary median")
 if r.get("exact") is not True or r.get("sabotage_step") is not None: e.append("clean verdict")
 return e

def validate_sabotage(r):
 e=[]; rows=r.get("rows",[]); tokens=r.get("shape",{}).get("tokens",-1)
 if r.get("rounds")!=1 or len(rows)!=2 or r.get("sabotage_step")!=tokens//2 or r.get("exact") is not False: return ["sabotage schedule"]
 concrete=False
 for i,row in enumerate(rows):
  expect=["percall","resident"] if i==0 else ["resident","percall"]
  if row.get("round")!=i or row.get("order")!=expect or row.get("warmup") is not (i==0): e.append(f"sabotage row{i} order")
  for name,res in (("percall",False),("resident",True)): e += [f"sabotage row{i}/{name}/{x}" for x in validate_arm(row.get(name,{}),tokens,res)]
  p,q=row.get("percall",{}),row.get("resident",{})
  diffs=[p.get(k)!=q.get(k) for k in ("output","state","trajectory","quality_bits")]
  concrete |= any(diffs)
  # Exactly one input token is changed; all prior transitions must agree.
  ps,qs=p.get("trajectory",[]),q.get("trajectory",[]); cut=tokens//2
  if len(ps)==tokens and len(qs)==tokens and ps[:cut]!=qs[:cut]: e.append("sabotage diverged before wrong token")
 if not concrete: e.append("sabotage did not diverge")
 return e

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
 errors=[]; rows=[]; allrec=[]
 for rung in ("screen","qualification"):
  count=1 if rung=="screen" else 3
  for ds in EXPECTED:
   for family in FAMILIES:
    paths=sorted((a.root/rung/ds/family).glob("process*/result.json"))
    if len(paths)!=count: errors.append(f"{rung}/{ds}/{family}: expected {count} processes"); continue
    recs=[json.loads(p.read_text()) for p in paths]; allrec+=recs
    for i,r in enumerate(recs):
     for x in common_errors(r,rung,ds,family,i,"none")+validate_clean(r): errors.append(f"{rung}/{ds}/{family}/process{i}: {x}")
    witnesses={(r.get("input_hash"),r.get("gradient_input_hash"),r.get("weights_hash"),r.get("gradient_hash")) for r in recs}
    if len(witnesses)!=1: errors.append(f"{rung}/{ds}/{family}: process witnesses differ")
    p=statistics.median(r["medians"]["percall"] for r in recs); q=statistics.median(r["medians"]["resident"] for r in recs)
    process_medians=[r["medians"]["resident"] for r in recs]
    rows.append(dict(rung=rung,dataset=ds,family=family,processes=count,percall_s=p,resident_s=q,speedup=p/q,positive=q<p,
                     resident_process_spread=max(process_medians)/min(process_medians)))
    sp=a.root/"sabotage"/ds/family/"result.json"
    if rung=="screen" and not sp.exists(): errors.append(f"{ds}/{family}: missing sabotage")
    elif rung=="screen":
     s=json.loads(sp.read_text())
     for x in common_errors(s,"screen",ds,family,0,"wrong-token")+validate_sabotage(s): errors.append(f"{ds}/{family}/sabotage: {x}")
     # Sabotage must be built/run from the same source, binding and inputs.
     clean=recs[0] if rung=="screen" else None
     if clean and any(s.get(k)!=clean.get(k) for k in ("commit","input_hash","gradient_input_hash","weights_hash","weights_after_hash","gradient_hash","gradient_after_hash","source_sha256","binding","target_column","native_vendor","native_numeric_mode")): errors.append(f"{ds}/{family}/sabotage: provenance differs")
 for field in ("commit","target_column"):
  if len({r.get(field) for r in allrec})!=1: errors.append("records disagree on "+field)
 for family in FAMILIES:
  if len({r.get("binding",{}).get("sha256") for r in allrec if r.get("family")==family})!=1: errors.append(family+" binding hashes differ")
 qualified=[r for r in rows if r["rung"]=="qualification"]
 if any(not r["positive"] for r in qualified): errors.append("qualification median regression")
 result=dict(schema="mojolearn.mamba-public-overhead-summary.v1",status="PASS" if not errors else "REJECT",errors=errors,rows=rows,
  policy="exact full output/report/state trajectory, unchanged gradients/quality, native-session reach, and positive qualification median-of-process-medians on both datasets; spread diagnostic only",
  apple_policy="enable the identical resident API on Apple after NVIDIA and AMD each PASS; no Apple timing required")
 a.out.write_text(json.dumps(result,indent=1,allow_nan=False)+"\n");print(a.out.read_text(),end="");raise SystemExit(0 if not errors else 1)
if __name__=="__main__": main()
