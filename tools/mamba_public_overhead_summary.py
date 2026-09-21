#!/usr/bin/env python3
"""Strict promotion summary for the two-dataset Mamba resident trial."""
import argparse, json, math, statistics
from pathlib import Path

EXPECTED={
 "taxi":("gbm-bench/taxi/taxi_speed.npz",419757252,"10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15"),
 "istella":("gbm-bench/istella/istella_speed.npz",2248281826,"31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef"),
}


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
 errors=[]; rows=[]; allrec=[]
 for rung in ("screen","qualification"):
  for ds in EXPECTED:
   for family in ("mamba1","mamba2","mamba3"):
    paths=sorted((a.root/rung/ds/family).glob("process*/result.json"))
    if len(paths)!=3: errors.append(f"{rung}/{ds}/{family}: expected 3 processes"); continue
    recs=[json.loads(p.read_text()) for p in paths];allrec+=recs
    for i,r in enumerate(recs):
     vendor={"nvidia":"cuda","amd":"hip"}.get(r.get("target_column")); d=r.get("dataset",{}); m=r.get("medians",{})
     if (r.get("schema")!="mojolearn.mamba-public-overhead.v1" or r.get("rung")!=rung or r.get("family")!=family
       or r.get("process")!=i or r.get("rounds")!=3 or r.get("warmup")!=1 or r.get("sabotage")!="none"
       or not r.get("exact") or (d.get("key"),d.get("bytes"),d.get("sha256"))!=EXPECTED[ds]
       or r.get("native_vendor")!=vendor or r.get("native_numeric_mode")!=1
       or any(len(r.get(k,""))!=64 for k in ("gradient_hash","input_hash","gradient_input_hash","weights_hash"))
       or len(r.get("binding",{}).get("sha256",""))!=64
       or not all(len(v)==64 for v in r.get("source_sha256",{}).values())
       or set(m)!={"percall","resident"} or not all(math.isfinite(v) and v>0 for v in m.values())
       or len(r.get("rows",[]))!=4): errors.append(f"{rung}/{ds}/{family}/process{i}: invalid record")
    base={(r["input_hash"],r["gradient_input_hash"],r["weights_hash"],r["gradient_hash"]) for r in recs}
    if len(base)!=1: errors.append(f"{rung}/{ds}/{family}: process witnesses differ")
    p=statistics.median(r["medians"]["percall"] for r in recs); q=statistics.median(r["medians"]["resident"] for r in recs)
    rows.append(dict(rung=rung,dataset=ds,family=family,processes=3,percall_s=p,resident_s=q,speedup=p/q,positive=q<p))
    sabotage=a.root/"sabotage"/ds/family/"result.json"
    if not sabotage.exists(): errors.append(f"{ds}/{family}: missing sabotage")
    else:
     s=json.loads(sabotage.read_text())
     if s.get("sabotage")!="wrong-token" or s.get("exact") is not False: errors.append(f"{ds}/{family}: invalid sabotage")
 for field in ("commit","target_column"):
  if len({r.get(field) for r in allrec})!=1: errors.append("records disagree on "+field)
 # Promotion uses the qualification rung; spread remains diagnostic only.
 qualified=[r for r in rows if r["rung"]=="qualification"]
 if any(not r["positive"] for r in qualified): errors.append("qualification median regression")
 result=dict(schema="mojolearn.mamba-public-overhead-summary.v1",status="PASS" if not errors else "REJECT",
             errors=errors,rows=rows,policy="exact full outputs/state/reports/gradients and positive median-of-process-medians on both datasets; spread diagnostic only",
             apple_policy="enable the identical resident API on Apple after NVIDIA and AMD each PASS")
 a.out.write_text(json.dumps(result,indent=1,allow_nan=False)+"\n");print(a.out.read_text(),end="");raise SystemExit(0 if not errors else 1)


if __name__=="__main__": main()
