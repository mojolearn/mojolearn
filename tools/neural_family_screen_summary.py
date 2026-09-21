#!/usr/bin/env python3
"""Summarize the small two-dataset neural family timing screen."""
import argparse, json, math, statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    families=('transformer','mamba1','mamba2','mamba3'); datasets=('taxi','istella'); errors=[]; rows=[]
    records={}; expected={'taxi':('gbm-bench/taxi/taxi_speed.npz',419757252,'10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15'),'istella':('gbm-bench/istella/istella_speed.npz',2248281826,'31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef')}; all_records=[]
    for ds in datasets:
        for family in families:
            paths=sorted((a.root/ds/family).glob('process*/result.json'))
            if not paths: errors.append('%s/%s missing' % (ds,family)); continue
            recs=[json.loads(p.read_text()) for p in paths]; records[(ds,family)]=recs
            all_records.extend(recs)
            bad=False
            for i,r in enumerate(recs):
                hashes=r.get('hashes',{}); trajectory=r.get('rows',[]); scope=r.get('component_scope',{}); target=r.get('target_column'); vendor={'nvidia':'cuda','amd':'hip'}.get(target)
                values=[v for row in trajectory for k,v in row.items() if k in ('forward','backward','optimizer','total')]
                if (r.get('family')!=family or r.get('shape')!={'batch':1,'length':128,'d_model':128}
                    or r.get('warmup')!=1 or r.get('samples')!=5 or len(trajectory)!=6
                    or [row.get('retained') for row in trajectory]!=[False]+[True]*5
                    or r.get('dataset',{}).get('key')!=expected[ds][0]
                    or r.get('dataset',{}).get('bytes')!=expected[ds][1]
                    or r.get('dataset',{}).get('sha256')!=expected[ds][2]
                    or r.get('process')!=i or r.get('binding',{}).get('bytes',0)<=0
                    or len(r.get('binding',{}).get('sha256',''))!=64
                    or r.get('native_vendor')!=vendor or r.get('native_numeric_mode')!=1
                    or not r.get('source_sha256') or not all(len(v)==64 for v in r.get('source_sha256',{}).values())
                    or any(len(hashes.get(k,''))!=64 for k in ('initial_weights','final_weights','input','cotangent','output','gradients'))
                    or any(len(row.get('hashes',{}).get(k,''))!=64 for row in trajectory for k in ('output','gradients','weights'))
                    or not values or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in values)
                    or scope.get('attention_or_scan_core') is not None
                    or scope.get('pack_unpack_or_state_copies') is not None
                    or not scope.get('unavailable_reason')):
                    bad=True
            if bad:
                errors.append('%s/%s structure invalid' % (ds,family)); continue
            med={k:statistics.median(r['medians'][k] for r in recs) for k in ('forward','backward','optimizer','total')}
            rows.append(dict(dataset=ds,family=family,processes=len(recs),medians=med,
                             dominant_measured_component=max(('forward','backward','optimizer'),key=lambda k:med[k])))
    for ds in datasets:
        target=[r for r in rows if r['dataset']==ds]
        tr=next((r for r in target if r['family']=='transformer'),None)
        if tr:
            for r in target: r['total_vs_transformer']=r['medians']['total']/tr['medians']['total']
        inputs={r.get('hashes',{}).get('input') for f in families for r in records.get((ds,f),[])}
        cotangents={r.get('hashes',{}).get('cotangent') for f in families for r in records.get((ds,f),[])}
        if len(inputs)!=1 or len(cotangents)!=1: errors.append(ds+' families do not share x/dy')
    for field in ('commit','target_column'):
        if len({r.get(field) for r in all_records}) != 1: errors.append('records disagree on '+field)
    for family in families:
        hashes={r.get('binding',{}).get('sha256') for r in all_records if r.get('family')==family}
        if len(hashes)!=1: errors.append(family+' binding hashes disagree')
        sources={json.dumps(r.get('source_sha256'),sort_keys=True) for r in all_records if r.get('family')==family}
        if len(sources)!=1: errors.append(family+' source inventories disagree')
    result=dict(schema='mojolearn.neural-family-screen-summary.v1',status='PASS' if not errors else 'REJECT',errors=errors,rows=rows,
                interpretation=('Matched B1/L128/D128 public block calls should not have equal times: Mamba state widths and kernels differ, Transformer attention scales differently, and each public backward recomputes forward. Ratios locate a slow family but do not establish equivalent model quality or promotion.'),
                limitation=('The public block ABIs expose whole forward/backward only. Optimizer is host SGD over returned gradients; scan/attention core and pack/state copies remain included in those boundaries and are recorded null by each probe. Add native phase timers only for the slow family selected here.'),
                final_gate='Any candidate selected by this screen must still pass exact full-state and quality gates at 162M parameters, B1/L2048 on Taxi and Istella-S.')
    a.out.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n'); print(a.out.read_text(),end=''); raise SystemExit(0 if not errors else 1)


if __name__=='__main__': main()
