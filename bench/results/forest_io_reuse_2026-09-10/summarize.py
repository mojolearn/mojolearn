import json,statistics
from pathlib import Path
root=Path(__file__).parent
rows=[]
for p in sorted(root.glob('*/*-calls*.json')):
 d=json.loads(p.read_text())
 if not d.get('complete'): continue
 s=d['summary']; b=s['parallel_groves_borrowed'];r=s['parallel_groves_reuse']
 bt={v['round']:v['ms'] for v in d['records'] if v['arm']=='parallel_groves_borrowed' and not v['warmup']}
 rt={v['round']:v['ms'] for v in d['records'] if v['arm']=='parallel_groves_reuse' and not v['warmup']}
 row=dict(file=str(p.relative_to(root)),vendor=d["vendor"],numeric_mode=d["numeric_mode"],rows_fit=d['rows_fit'],rows_predict=d['rows_predict'],features=d['features'],outputs=d['outputs'],nodes=d['nodes'],calls_per_sample=d['calls_per_sample'],complete=d['complete'],borrowed_ms=b['median_ms'],reuse_ms=r['median_ms'],borrowed_spread=b['spread'],reuse_spread=r['spread'],stable_pair=b['stable'] and r['stable'],observed_ratio=b['median_ms']/r['median_ms'],reuse_faster_rounds=sum(rt[i]<bt[i] for i in bt),measured_rounds=len(bt),workspace_bytes=d['retained_reuse_workspace_bytes'])
 if 'cuml_gpu' in s:
  c=s['cuml_gpu'];row.update(cuml_ms=c['median_ms'],cuml_spread=c['spread'],reuse_cuml_stable=r['stable'] and c['stable'],observed_reuse_over_cuml=r['median_ms']/c['median_ms'])
 rows.append(row)
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows: print(r)
