"""Publish retained full pairs as pending evidence through the existing board tool."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign-root',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.campaign_root.resolve()
REPO=Path(__file__).resolve().parents[1]
OUT=args.out.resolve()
SOURCES=[('apple',ROOT/'apple/captured/runs'),
         ('apple',ROOT/'apple/captured/kmeans-repair2/runs'),
         ('apple',ROOT/'apple/captured/resample-full/runs'),
         ('apple',ROOT/'apple/captured/reg-full/runs'),
         ('apple',ROOT/'apple/captured/expanded-reg/runs'),
         ('apple',ROOT/'apple/captured/gmm-istella-full/runs'),
         ('apple',ROOT/'apple/captured/pls-qn-full/runs'),
         ('apple',ROOT/'apple/captured/tsvd-full-v1/runs'),
         ('apple',ROOT/'apple/captured/selectors-full/runs'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-next-reg'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-expanded-reg'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-gmm-istella'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-pls'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-tsvd-full-v1'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements')]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    cells=[];summary=[];seen=set()
    review_path=ROOT/'quality-review/classical-12-pair-quality-review.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else {'rows':[]}
    next_review_path=ROOT/'quality-review/next-reg-resample-quality-review.json'
    next_review=json.loads(next_review_path.read_text()) if next_review_path.exists() else {'rows':[]}
    reviewed={r['receipt_sha256']:r for r in review['rows']+next_review['rows']}
    receipt_paths={}
    for vendor,source in SOURCES:
        if not source.exists():continue
        for path in sorted(source.glob('*/attempts/*/receipt.json')):
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest in seen:continue
            seen.add(digest);receipt=json.loads(raw);job=receipt['workload']
            config=job['master_selection']['id']
            selected={r['arm']:r for r in receipt['runs'] if r['phase']=='scored' and r.get('returncode')==0 and r.get('result')}
            execution_status=receipt.get('status','IN_PROGRESS')
            complete=execution_status=='MEASURED_FULL' and set(selected)=={'A','B'}
            samples={arm:{phase:sum(run.get('arm')==arm and run.get('phase')==phase and run.get('returncode')==0 and bool(run.get('result')) for run in receipt['runs']) for phase in ('warmup','scored')} for arm in ('A','B')}
            controller=source.relative_to(ROOT).as_posix().replace('/','--')
            target=OUT/'receipts'/vendor/controller/receipt['key']/path.parent.name/'receipt.json'
            target.parent.mkdir(parents=True,exist_ok=True)
            # Publish the exact bytes we hashed even if the capture process
            # replaces its live receipt while this snapshot is being built.
            temp=target.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(target)
            receipt_paths[digest]=str(target.relative_to(REPO))
            row=dict(id=config,vendor=vendor,case=job['workload_id']+'/'+path.parent.name,
                     scope='full_workload',status='PENDING_ADMISSION' if complete else 'IN_PROGRESS' if execution_status=='IN_PROGRESS' else 'FAILED_OR_INCOMPLETE',
                     source_sha=receipt['source_sha'],evidence=str(target.relative_to(REPO)),
                     dimensions=job['dimensions'],dataset_sha256=job['dataset_sha256'],
                     execution_status=execution_status,
                     actual_sample_counts=samples,
                     warmups=min(samples[arm]['warmup'] for arm in samples),
                     scored_samples=min(samples[arm]['scored'] for arm in samples),
                     quality='PENDING' if complete else 'NOT_ASSESSED',
                     identity='NOT_REQUIRED' if receipt['mode']=='fast' else 'INCOMPLETE',
                     route='native-sm90' if vendor=='nvidia' else 'apple-fast',
                     receipt_sha256=digest,source_coverage_pending=job.get('source_coverage_pending',[]))
            if vendor=='apple' and source in [ROOT/'apple/captured/runs',ROOT/'apple/captured/kmeans-repair2/runs']:
                row['resource_limitations']=['Shared external-disk I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established.']
            detail=dict(configuration=config,workload=job['workload_id'],vendor=vendor,
                        source_sha=receipt['source_sha'],execution_status=execution_status,
                        evidence=row['evidence'],returncodes=[r.get('returncode') for r in receipt['runs']])
            if complete:
                a,b=(selected[k]['result'] for k in ('A','B'))
                row.update(candidate_ms=a['timings']['full_operation_seconds']*1000,
                           baseline_ms=b['timings']['full_operation_seconds']*1000,
                           quality_metrics={k:selected[k]['result']['task_quality'] for k in ('A','B')})
                detail.update(candidate_seconds=row['candidate_ms']/1000,baseline_seconds=row['baseline_ms']/1000,
                              observed_candidate_over_baseline=row['candidate_ms']/row['baseline_ms'],
                              quality=row['quality_metrics'],
                              output_sha256={k:selected[k]['result']['output_sha256'] for k in ('A','B')},
                              model_state={k:selected[k]['result']['model_state'] for k in ('A','B')},
                              admission='Pending independent quality, required identity and affected-workload coverage')
            if complete and digest in reviewed:
                assessment=reviewed[digest]
                row['quality_assessment']=assessment['quality_assessment']
                row['quality_reason']=assessment['reason']
                row['quality']='FAIL' if row['quality_assessment'] in ('FAILED_FAST_OPPONENT_GATE','QUALITY_FAILED') else 'PASS_TASK_METRICS' if row['quality_assessment']=='TASK_METRIC_GATE_PASSED' else 'PENDING'
                if row['quality']=='FAIL':row['status']='QUALITY_FAILED'
                detail['quality_assessment']=row['quality_assessment']
                detail['quality_reason']=row['quality_reason']
            if row.get('resource_limitations'):detail['resource_limitations']=row['resource_limitations']
            cells.append(row);summary.append(detail)
    inventory=dict(campaign='six-lane-full-ab-20261006',identity_policy='NVIDIA/AMD same-arm IDENTICAL comparison pending AMD/PTX artifacts and complete typed fitted-state evidence. Apple FAST is evaluated by task quality; bits may differ.',evidence_policy='Complete full-workload executions are retained separately from quality and identity admission. Failed attempts preserved at controller-qualified paths; no default promotion.',candidates=[
        dict(id='AF.X.complete-proposed',title='Apple FAST complete proposed configuration',mode='fast',vendors=['apple']),
        dict(id='I.X.complete-proposed',title='IDENTICAL complete proposed configuration',mode='identical',vendors=['nvidia','amd','apple','host'])])
    notes=['A=candidate; B=incumbent. Timed evidence is pending admission, not a default promotion.',
           'These are combined-configuration full workloads, not completed individual constituent experiments.',
           'Initial 12-pair quality review: all12 preserve baseline metrics; 4 task-metric gates pass, 6 taxi opponent comparisons pending (historical4m vs current5.25m rows), Apple Istella KMeans fails best-opponent gate, NVIDIA inherits opponent-quality deficit. Additional saved assessments are retained in next-quality-review.json.',
           'One excluded warmup and one scored sample per arm. Original failed attempts are retained.',
           'NVIDIA PTX and AMD have no compatible retained artifacts; missing-only build question remains pending.',
           'IDENTICAL compares each same arm across vendors; unavailable typed complete model state remains incomplete.',
           'Scored output and partial/public-save model hashes are retained separately; partial hashes do not prove complete state identity.',
           'Apple first four PCA/OLS pairs overlapped shared external-storage data transfer; KMeans overlap unestablished. No quiet-storage or promotion claim.',
           'No compilation or separate numerical verification rerun. Full provider and worker logs remain under '+str(ROOT)]
    for assessment in review['rows']+next_review['rows']:
        assessment['original_review_receipt_path']=assessment['receipt']
        assessment['receipt']=receipt_paths.get(assessment['receipt_sha256'],assessment['receipt'])
    review['publication_repair']='Original review retained externally; links relocated by exact receipt SHA256 into controller-qualified paths. Distinct failed and repaired attempts no longer collide.'
    write(OUT/'quality-review.json',review)
    if next_review['rows']:write(OUT/'next-quality-review.json',next_review)
    if (ROOT/'quality-review/historical-istella-opponents.json').exists():shutil.copyfile(ROOT/'quality-review/historical-istella-opponents.json',OUT/'historical-istella-opponents.json')
    decisions=[dict(candidate=c['id']+'/'+c['case'],
                    decision='NOT PROMOTED: '+c['quality_reason'],
                    commit=c['source_sha'],evidence=c['evidence'],
                    default_changed=False,individual_constituents='Not decided by this combined-configuration result')
               for c in cells if c['status']=='QUALITY_FAILED']
    write(OUT/'inventory.json',inventory);write(OUT/'index.json',dict(cells=cells,notes=notes,decisions=decisions))
    write(OUT/'retained-pairs.json',dict(updated_at=time.time(),pairs=summary))
    with (ROOT/'board-publication.log').open('a') as log:
        p=subprocess.run(['python3',str(REPO/'tools/performance_measurement_board.py'),
                          '--inventory',str(OUT/'inventory.json'),'--index',str(OUT/'index.json'),
                          '--out',str(OUT)],stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(returncode=p.returncode,retained_attempts=len(cells),
                          complete_pairs=sum(c['execution_status']=='MEASURED_FULL' for c in summary),
                          quality_failed=sum(c['status']=='QUALITY_FAILED' for c in cells),
                          in_progress=sum(c['execution_status']=='IN_PROGRESS' for c in summary),
                          failed_or_incomplete=sum(c['execution_status'] not in ('MEASURED_FULL','IN_PROGRESS') for c in summary),board=str(OUT/'BOARD.md'))))
    raise SystemExit(p.returncode)


if __name__=='__main__':main()
