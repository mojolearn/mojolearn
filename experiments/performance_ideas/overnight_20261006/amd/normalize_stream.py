import pathlib,json
R=pathlib.Path(__file__).resolve().parent
P=R/'live/repairs/results.json'
def run():
 if not P.exists():return
 results=json.loads(P.read_text());f=R/'normalized-measurements.json';data=json.loads(f.read_text());rows=[r for r in data['rows'] if r.get('measurement_stage')!='streamed_driver'];indexed={}
 def common(r):
  machine='DigitalOcean-606508222-MI325X';s=r['source_sha'];row=dict(id=r['candidate_id'],vendor='amd',route='gfx942',scope='component' if r['candidate_id'] in ['I02','I04','I06','I07','N06','N08'] else r['scope'],machine=machine,baseline_machine=machine,candidate_machine=machine,source_sha=s,baseline_source_sha=s,candidate_source_sha=s,evidence=str(P),warmups=1,scored_samples=1,warmup_scope='same_process',measurement_stage='streamed_driver',limitation='Representative generated production caller/components; named-dataset full-board promotion is separate')
  if r['candidate_id']=='I06' and s.startswith('cbcc8dcd'):
   row.update(comparison_kind='confounded_schedule_bundle',promotion=False,limitation='Driver enabled GQA head reuse AND changed backward schedule to kvgrid_r32. Raw component timings represent this bundle, not isolated GQA toggle evidence. Corrected same-schedule driver pending.')
  if r['candidate_id']=='I06' and s.startswith('e80a1d0'):
   row.update(promotion=False,dispatch_evidence='Source guard requires an even query-head/KV-head ratio and admitted32-row Q-resident preflushed schedule; raw measurements retain actual heads/kv_heads/ran_arm. No dedicated reuse counter is exported by this driver.',limitation='Generated attention component; macro alone does not prove reuse. Ratio2 fixtures satisfy inspected source dispatch guards; runtime reuse count not exported. Full caller qualification remains pending.')
  if r['candidate_id']=='I14':
   row.update(comparison_kind='gated_shortcut_chunk_bundle',promotion=False,scope='component',limitation='Disconnected17-vertex-clique weak_cc_batched graph component; gated+shortcut chunk16 versus ungated baseline, not an isolated chunk-width or full DBSCAN/HDBSCAN comparison.')
  return row
 for r in results.values():
  if r['candidate_id']=='I06' and r['source_sha'].startswith('e80a1d0') and r['status']=='MEASURED' and all((m.get('heads',12)//m.get('kv_heads',4))%2!=0 for m in r['measurements']):
   rows.append(common(r)|dict(case=r['key'],status='NO_DISTINCT_RUNTIME_ARM',returncode=0,reason='Shared-head reuse requires an even query/KV-head ratio; the original12/4 ratio3 refuses reuse in both macro builds. Raw timing retained, ratio2 matched cases appended with existing binaries.'));continue
  if r['candidate_id']=='I07' and r['source_sha'].startswith('cbcc8dcd') and r['status']=='MEASURED' and all(m.get('kept_cells')==0 and m.get('ran_arm')==1030 for m in r['measurements']):
   rows.append(common(r)|dict(case=r['key'],status='NO_DISTINCT_RUNTIME_ARM',returncode=0,reason='Driver requested a no-estash attention arm; both macro variants ran arm1030 with kept_cells0. Retained/recomputed lifetime was not exercised. Raw timings retained.'));continue
  if r['candidate_id']=='I15':
   rows.append(common(r)|dict(case=r['key'],status='NO_DISTINCT_RUNTIME_ARM',returncode=r['returncode'],reason='Certified MMA toggle requires TARGET_COLUMN==COLUMN_APPLE; AMD baseline/candidate take the same production route. Raw timing retained without ratio.'));continue
  if r['candidate_id']=='I19' and r['arm']=='incumbent' and r['status']=='MEASUREMENT_FAILED' and r['source_sha'].startswith('5b467815') and int(r['environment'].get('AB_ROWS','1000000'))>32*4096:
   rows.append(common(r)|dict(case=r['key'],status='UNSUPPORTED_BASELINE_SHAPE',returncode=r['returncode'],reason='Existing rank control permits at most4096 rows per segment; driver32segments with1M+ total rows exceeds that documented domain. Original refusal retained; matched bounded-segment measurements appended using existing binaries.'));continue
  if r['status']!='MEASURED':rows.append(common(r)|dict(case=r['key'],status='MEASUREMENT_FAILED',returncode=r['returncode']));continue
  for m in r['measurements']:
   case=json.dumps(r['environment'],sort_keys=True)+'/'+json.dumps({k:v for k,v in m.items() if k not in ['elapsed_ns','phase','id','arm','baseline_completion_ns','candidate_completion_ns','converged','evaluations','ran_arm','kept_cells','iterations','solver_status']},sort_keys=True)
   if 'baseline_completion_ns' in m:
    rows.append(common(r)|dict(case=case,status='MEASURED',returncode=0,artifact_hashes=dict(baseline=r['binary_sha256'],candidate=r['binary_sha256']),baseline_ms=m['baseline_completion_ns']/1e6,candidate_ms=m['candidate_completion_ns']/1e6));continue
   arm=str(m['arm']) if 'arm' in m else r['arm'];indexed[(r['candidate_id'],case,arm)]=(r,m)
 for (idea,case,arm),(r,m) in indexed.items():
  baseline='0' if (idea,case,'0') in indexed else 'baseline' if (idea,case,'baseline') in indexed else 'incumbent'
  if arm==baseline:continue
  if (idea,case,baseline) not in indexed:continue
  if idea=='I23':
   rows.append(common(r)|dict(case=case+'/'+arm,status='NO_DISTINCT_RUNTIME_ARM',returncode=0,reason='ARIMA_FAST_BATCH_GRAD requires has_apple_gpu_accelerator(); AMD off/on macro builds both use sequential gradients. Raw fit timings retained; batch-gradient timing belongs on Apple.'));continue
  if idea=='I17' and arm in ['candidate','lg_exact_off'] and m.get('policy')=='Depthwise':
   rows.append(common(r)|dict(case=case+'/'+arm,status='NO_DISTINCT_RUNTIME_ARM',returncode=0,reason='This I17 resident-frontier or LG-exact control requires Lossguide. Depthwise lg_exact is false, no folded leaves or resident snapshots; raw timing retained as same-route control coverage.'));continue
  b,bm=indexed[(idea,case,baseline)]
  if b['source_sha']!=r['source_sha']:continue
  rows.append(common(r)|dict(case=case+'/'+arm,status='MEASURED',returncode=0,artifact_hashes=dict(baseline=b['binary_sha256'],candidate=r['binary_sha256']),baseline_ms=bm['elapsed_ns']/1e6,candidate_ms=m['elapsed_ns']/1e6,baseline_arm=baseline,candidate_arm=arm))
 p=f.with_suffix('.tmp');p.write_text(json.dumps(dict(rows=rows),indent=2));p.replace(f)
if __name__=='__main__':run()
