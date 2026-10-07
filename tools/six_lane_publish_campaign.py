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
         ('apple',ROOT/'apple/captured/classification-full-v1/runs'),
         ('apple',ROOT/'apple/captured/classification-full-v1-remaining/runs'),
         ('apple',ROOT/'apple/captured/qr-svd-full/runs'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-next-reg'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-expanded-reg'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-gmm-istella'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-pls'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-tsvd-full-v1'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-tsvd-full-v1-repair1'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-isotonic-cv'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-classification-full-v1'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-classification-qn-full-v1'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements-mlp-full-v1'),
         ('amd',ROOT/'amd/capture-attempt-01/artifacts/measurements'),
         ('amd',ROOT/'amd/capture-attempt-01/artifacts/measurements-classical-dependency-repair'),
         ('amd',ROOT/'amd/capture-attempt-01/artifacts/measurements-tsvd-full-v1'),
         ('amd',ROOT/'amd/capture-attempt-01/artifacts/measurements-expanded-reg'),
         ('amd',ROOT/'amd/capture-attempt-01/artifacts/measurements-next-reg'),
         ('amd',ROOT/'amd/capture-attempt-01/artifacts/measurements-classification-full-v1'),
         ('nvidia',ROOT/'nvidia-native/capture-attempt-02/artifacts/measurements')]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def remaining_catalog(cells):
    """Index exact selections; combined receipts never credit their members."""
    path=REPO/'experiments/six_lane_integration/catalog.json'
    raw=path.read_bytes();catalog=json.loads(raw)
    entries={entry['id']:entry for entry in catalog['entries']}
    member_records=dict(entries)
    for entry in catalog['entries']:
        for arm in entry.get('arms',[]):
            member_records[arm['id']]=dict(entry,affected_workloads=arm.get('workloads',entry.get('affected_workloads',[])))
    rows=[]
    for kind,records in [('entry',catalog['entries']),('interaction',catalog['interactions'])]:
        for record in records:
            selection_ids={record['id']}|{arm['id'] for arm in record.get('arms',[])}
            direct=[cell for cell in cells if cell['id'] in selection_ids]
            members=[member_records[key] for key in record.get('members',[]) if key in member_records]
            unresolved_members=[key for key in record.get('members',[]) if key not in member_records]
            modes=sorted({item['mode'] for item in [record]+members if item.get('mode')})
            # Recipes can be IDs or structured workload descriptors; retain both.
            by_workload={json.dumps(value,sort_keys=True):value for item in [record]+members
                         for value in item.get('affected_workloads',[])}
            workloads=[by_workload[key] for key in sorted(by_workload)]
            role=record.get('campaign_role','interaction_plan')
            rows.append(dict(id=record['id'],kind=kind,role=role,title=record.get('title',record['id']),
                modes=modes,mode_policy='Authored mode, or union of referenced member modes',
                status='DIRECT_RECEIPTS_RETAINED_PENDING_ADMISSION' if direct else
                       'SOURCE_REJECTED_NO_DIRECT_RECEIPT' if role=='source_rejected' else
                       'NO_DIRECT_RECEIPT_IN_THIS_CAMPAIGN',
                direct_attempts=len(direct),complete_pairs=sum(c.get('execution_status')=='MEASURED_FULL' for c in direct),
                quality_failed=sum(c['status']=='QUALITY_FAILED' for c in direct),
                selection_only=record.get('selection_only',False),members=record.get('members',[]),
                unresolved_members=unresolved_members,
                recipe_status='SOURCE_RECIPE_UNRESOLVED' if not modes or not workloads or unresolved_members else 'REQUIRES_FULL_RECIPE_AND_ARTIFACT_ADMISSION',
                affected_workloads=workloads,arm_ids=sorted(selection_ids-{record['id']}),
                arm_controls=[{key:arm.get(key) for key in ('id','A','B')}
                              for arm in record.get('arms',[])],
                authored_prerequisites=record.get('prerequisites',[]),authored_gaps=record.get('gaps',[]),
                source_record=record.get('source_record'),source_status=record.get('source_status'),
                receipts=[c['evidence'] for c in direct]))
    return dict(schema='mojolearn.campaign-remaining-work/1',catalog_source=str(path.relative_to(REPO)),
        catalog_sha256=hashlib.sha256(raw).hexdigest(),entries=len(catalog['entries']),interactions=len(catalog['interactions']),
        roles={role:sum(e.get('campaign_role')==role for e in catalog['entries']) for role in
               ['new_candidate','incumbent_dependency','source_rejected']},
        direct_selection_count=sum(bool(row['direct_attempts']) for row in rows),
        policy='Campaign-local exact selection coverage, not a global claim of never measured. Combined members get no individual credit. A missing receipt does not establish runnable artifacts or require repeating historically decided work. Selection-only plans and rejected sources are not executable queues. Catalog source prerequisites are authored metadata, not a current binary readiness audit.',
        outside_catalog='Original I01–I24/A01–A08/N01–N08/F01–F20 cards and historical decisions require their own cross-reference; see experiments/AB_EXPERIMENT_INDEX.md. They are not automatically queued again.',
        rows=rows)


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
                     worker_returncodes=[r.get('returncode') for r in receipt['runs']],
                     failure_reasons=[str(r['error']) for r in receipt['runs'] if r.get('error')],
                     actual_sample_counts=samples,
                     warmups=min(samples[arm]['warmup'] for arm in samples),
                     scored_samples=min(samples[arm]['scored'] for arm in samples),
                     quality='PENDING' if complete else 'NOT_ASSESSED',
                     identity='NOT_REQUIRED' if receipt['mode']=='fast' else 'INCOMPLETE',
                     route={'nvidia':'native-sm90','amd':'amd-native-gfx942','apple':'apple-fast'}[vendor],
                     receipt_sha256=digest,source_coverage_pending=job.get('source_coverage_pending',[]))
            # Display the arm configuration actually recorded with this workload,
            # separately from the broader requested selection. These controls are
            # not an attestation that every enabled code path executed.
            row['controls']={arm:job.get('arms',{}).get(arm,{}).get('configuration') for arm in ('A','B')}
            row['requested_controls']={arm:job['master_selection'].get(arm) for arm in ('A','B')}
            row['implementation_ids']=job.get('implementation_ids',[])
            if vendor=='apple' and source in [ROOT/'apple/captured/runs',ROOT/'apple/captured/kmeans-repair2/runs']:
                row['resource_limitations']=['Shared workspace storage I/O overlapped first four PCA/OLS pairs; overlap for KMeans unestablished. Quiet-storage timing is not established.']
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
                row['quality_comparisons']={key:assessment[key] for key in (
                    'candidate_vs_baseline','candidate_vs_opponents','baseline_vs_opponents',
                    'opponent_metrics','metric_directions','new_full_opponent_review') if key in assessment}
                row['quality']='FAIL' if row['quality_assessment'] in ('FAILED_FAST_OPPONENT_GATE','QUALITY_FAILED') else 'PASS_TASK_METRICS' if row['quality_assessment']=='TASK_METRIC_GATE_PASSED' else 'PENDING'
                if row['quality']=='FAIL':row['status']='QUALITY_FAILED'
                detail['quality_assessment']=row['quality_assessment']
                detail['quality_reason']=row['quality_reason']
            if row.get('resource_limitations'):detail['resource_limitations']=row['resource_limitations']
            cells.append(row);summary.append(detail)
    inventory=dict(campaign='six-lane-full-ab-20261006',identity_policy='NVIDIA/AMD same-arm IDENTICAL admission requires matching source/runtime provenance and complete typed fitted-state evidence. See the coverage table for completed and pending work; NVIDIA PTX artifacts remain pending. Apple FAST is evaluated by task quality; bits may differ.',evidence_policy='Complete full-workload executions are retained separately from quality and identity admission. Failed attempts preserved at controller-qualified paths; no default promotion. Hash receipts alone do not establish retained array/model bytes; see artifact-retention.json.',candidates=[
        dict(id='AF.X.complete-proposed',title='Apple FAST complete proposed configuration',mode='fast',vendors=['apple']),
        dict(id='I.X.complete-proposed',title='IDENTICAL complete proposed configuration',mode='identical',vendors=['nvidia','amd','apple','host'])])
    notes=['A=candidate; B=incumbent. Timed evidence is pending admission, not a default promotion.',
           'These are combined-configuration full workloads, not completed individual constituent experiments.',
           'Initial 12-pair quality review: all12 preserve baseline metrics; 4 task-metric gates pass, 6 taxi opponent comparisons pending (historical4m vs current5.25m rows), Apple Istella KMeans fails best-opponent gate, NVIDIA inherits opponent-quality deficit. Additional saved assessments are retained in next-quality-review.json.',
           'One excluded warmup and one scored sample per arm. Original failed attempts are retained.',
           'AMD GPU measurements use accepted retained artifacts. Latest owner instruction forbids further compilation; unavailable paired artifacts remain blocked. NVIDIA PTX still awaits compatible artifacts.',
           'IDENTICAL compares each same arm across vendors; unavailable typed complete model state remains incomplete.',
           'Scored output and partial/public-save model hashes are retained separately; partial hashes do not prove complete state identity.',
           'Apple first four PCA/OLS pairs overlapped shared workspace storage data transfer; KMeans overlap unestablished. No quiet-storage or promotion claim.',
           'Apple teardown preservation failed: the workspace was on the internal SSD, not retained EBS. Logs, timings, metrics and hash receipts survive; some raw array bytes remain unrecovered. See artifact-retention.json for exact recovery coverage and provenance. Original receipts are unchanged.',
           'Races reuse accepted binaries without separate numerical verification reruns. Earlier separately authorized AMD builds are historical artifact evidence, not measurements. Full provider and worker logs remain under '+str(ROOT)]
    retention={}
    storage=ROOT/'quality-review/r2-reconciliation/reconciliation.json'
    if storage.exists():
        shutil.copyfile(storage,OUT/'storage-reconciliation.json')
        storage_report=json.loads(storage.read_text())
        notes.append('R2 storage reconciliation: '+str(storage_report.get('new_current_campaign_measurements_found','unknown'))+
            ' new current-campaign measurements found in the recorded search. Storage locations, fetched archive hashes, historical comparisons and search limitations are retained in storage-reconciliation.json. Older medium/component results do not fill full-workload candidate gaps.')
    for label,relative in [('apple_incident','apple/emergency-preservation-correction.json'),
                           ('apple_recovery','apple/incident-offbox-audit/recovery-summary.json')]:
        source=ROOT/relative
        if source.exists():
            raw=source.read_bytes()
            retention[label]=dict(source=str(source),sha256=hashlib.sha256(raw).hexdigest(),record=json.loads(raw))
    write(OUT/'artifact-retention.json',retention)
    stop=ROOT/'amd-missing-build/owner-stop-compilation/final-status.json'
    if stop.exists():
        shutil.copyfile(stop,OUT/'compilation-stopped.json')
        notes.append('The AMD missing-artifact compiler was stopped under the latest owner instruction; see compilation-stopped.json. Accepted completed binaries remain reusable; interrupted/unbuilt jobs do not count as ready.')
    hashes=ROOT/'quality-review/scored-hash-coverage.json'
    if hashes.exists():
        shutil.copyfile(hashes,OUT/'scored-hash-coverage.json')
        notes.append('scored-hash-coverage.json is a dated, hash-bound metadata audit of captured scored outputs and model states. Complete output hashes do not establish complete fitted-model identity; missing model state remains pending.')
    diagnosis=ROOT/'quality-review/nvidia-nb-lda-source-diagnosis.json'
    if diagnosis.exists():
        shutil.copyfile(diagnosis,OUT/'nvidia-nb-lda-source-diagnosis.json')
        notes.append('Source-only review of NVIDIA GaussianNB Taxi and LDA Istella retains both combined-configuration quality failures: no implementation or harness bug established. Changed C55 reduction order is a source-supported explanation, not isolated causal proof. Unexercised controls and individual alternatives remain pending; see nvidia-nb-lda-source-diagnosis.json.')
    for name, explanation in (
        ('apple-qr-resample-source-diagnosis.json', 'Apple source-only review retains QR combined-configuration quality failures; no concrete implementation defect or isolated L09 regression was established. Resampling A and B have equal saved quality and both fail the opponent quality requirement; this does not establish a new P10 regression. Original evidence and missing-array limitations remain explicit.'),
        ('nvidia-gmm-taxi-source-diagnosis.json', 'Full Taxi GMM refused both candidate and incumbent before scoring. Source-only review found no established implementation or harness defect; retain the incomplete pair and original failures, with zero scored samples and no candidate win/loss decision.'),
        ('cv-taxi-source-diagnosis.json', 'LassoCV and ElasticNetCV Taxi retain combined-configuration quality failures on NVIDIA and AMD. Source-only review does not establish an implementation defect or isolate a control; saved quality failures cannot promote these defaults.'),
    ):
        diagnosis=ROOT/'quality-review'/name
        if diagnosis.exists():
            shutil.copyfile(diagnosis,OUT/name)
            notes.append(explanation+' See '+name+'.')
    comparison_latest=ROOT/'comparisons/amd-nvidia-same-arm-20261006/latest.json'
    if comparison_latest.exists():
        snapshot=Path(json.loads(comparison_latest.read_text())['snapshot']).resolve()
        if not snapshot.is_relative_to((ROOT/'comparisons').resolve()):
            raise ValueError('Comparison snapshot must remain within campaign evidence')
        destination=OUT/'comparisons/amd-nvidia-same-arm'/snapshot.name
        manifest=[]
        for source in sorted(snapshot.rglob('*.json')):
            relative=source.relative_to(snapshot);target=destination/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            raw=source.read_bytes()
            if not target.exists() or target.read_bytes()!=raw:target.write_bytes(raw)
            manifest.append(dict(path=str(relative),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        write(destination/'retained-manifest.json',dict(source=str(snapshot),files=manifest))
        comparison_summary=json.loads((snapshot/'summary.json').read_text())
        write(OUT/'same-arm-output-comparison.json',dict(snapshot=str(destination.relative_to(OUT)),
              summary=comparison_summary,qualified_full_identity=comparison_summary['qualified_full_identity']))
        notes.append('Saved same-arm AMD/NVIDIA output comparison: '+str(comparison_summary['candidate_case_count'])+
                     ' matched workloads; primary counts '+json.dumps(comparison_summary['observed_primary_counts'],sort_keys=True)+
                     ', repeated counts '+json.dumps(comparison_summary['observed_repeated_counts'],sort_keys=True)+
                     '. Full identity counts '+json.dumps(comparison_summary['full_identity_counts'],sort_keys=True)+
                     '. These are saved-signature comparisons, not new model runs or default admission; unmatched and failed arms are retained in same-arm-output-comparison.json and its snapshot.')
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
    defaults_audit=ROOT/'quality-review/defaults-inline-audit/audit-and-comment-handoff.json'
    if defaults_audit.exists():
        audit=json.loads(defaults_audit.read_text())
        write(OUT/'existing-default-decisions.json',dict(source=str(defaults_audit),
              source_sha256=hashlib.sha256(defaults_audit.read_bytes()).hexdigest(),
              promotions=audit.get('existing_promotions',[]),
              scope='Recorded earlier Apple FAST promotions, not new decisions from this campaign'))
        decisions[:0]=[dict(candidate=', '.join(p['switches']),
            decision='PREVIOUSLY PROMOTED ON (Apple FAST): '+p['evidence_scope'],
            commit=p['commit'],evidence=p['file']+'; existing-default-decisions.json',
            default_changed=False,historical_decision=True)
            for p in audit.get('existing_promotions',[])]
    remaining={
        'nvidia': 'Native combined configurations measured; PTX, individual controls and other full-workload recipe gaps remain pending.',
        'amd': 'Selected retained-artifact pairs measured; GMM and other missing paired-artifact/recipe scopes remain pending.',
        'apple': 'See unrun scope for additional FAST pairs; earlier array-preservation and quality limitations remain.'}
    coverage=[]
    for vendor, label in [('nvidia','NVIDIA native / IDENTICAL'),('amd','AMD GPU / IDENTICAL'),('apple','Apple / FAST')]:
        rows=[c for c in cells if c['vendor']==vendor]
        coverage.append(dict(vendor=vendor,label=label,
            complete_pairs=sum(c.get('execution_status')=='MEASURED_FULL' for c in rows),
            failed_attempts=sum(c['status']=='FAILED_OR_INCOMPLETE' for c in rows),
            quality_failed=sum(c['status']=='QUALITY_FAILED' for c in rows),remaining_scope=remaining[vendor]))
    evidence_inputs={}
    for label,relative in [('apple_pending','apple/restart-readiness/status.json'),
                           ('apple_readiness','apple/restart-readiness/readiness-summary.json'),
                           ('amd_capture','amd/terminal-preservation-audit/status.json'),
                           ('amd_release','amd/release/termination-proof.json'),
                           ('nvidia_release','nvidia-native/owner-attempt-02/termination-proof.json')]:
        source=ROOT/relative
        if source.exists():
            raw=source.read_bytes()
            evidence_inputs[label]=dict(source=str(source),sha256=hashlib.sha256(raw).hexdigest(),record=json.loads(raw))
    apple_pending=evidence_inputs.get('apple_pending',{}).get('record',{})
    pending_work=[
        dict(vendor='nvidia',scope='NVIDIA PTX/default full A/B',reason='Compatible retained full-workload artifacts unavailable; no timing worker or compilation substituted.',evidence=str(ROOT/'nvidia-ptx/status.json')),
        dict(vendor='amd',scope='AMD GMM full-workload pair',reason='Paired retained mixture binaries unavailable; excluded from the completed queue.',evidence=str(ROOT/'amd/next-reg/continuation-status.json')),
        dict(vendor='all',scope='Individual candidates, alternative arms and other affected workloads',reason='This campaign measured complete-proposed combinations, not every individual catalog entry. Missing recipes, incompatible artifacts and untested interactions remain pending; do not infer constituent winners.',evidence='experiments/six_lane_integration/catalog.json')]
    if apple_pending.get('completed',0)<apple_pending.get('expected_pairs',4):
        pending_work.insert(1,dict(vendor='apple',scope='Apple FAST LogReg/LinearSVC × Taxi/Istella',
            reason=str(apple_pending.get('completed',0))+'/'+str(apple_pending.get('expected_pairs',4))+
                ' pairs completed; '+apple_pending.get('status','STATUS_UNAVAILABLE')+
                '; freeze '+apple_pending.get('harness_freeze','UNKNOWN')+'. Original launch-tag error and released-host evidence are retained.',
            evidence='campaign-coverage.json: apple_pending'))
    apple_readiness=evidence_inputs.get('apple_readiness',{}).get('record',{})
    if apple_readiness.get('mlp_pending_cells'):
        pending_work.append(dict(vendor='apple',scope='Apple FAST MLP classifier/regressor × Taxi/Istella',
            reason=str(apple_readiness['mlp_pending_cells'])+' additional pending pairs; '+apple_readiness.get('mlp_blocker','Readiness unresolved'),
            evidence='campaign-coverage.json: apple_readiness'))
    remaining=remaining_catalog(cells)
    write(OUT/'remaining-work.json',remaining)
    write(OUT/'campaign-coverage.json',dict(coverage=coverage,pending_work=pending_work,evidence_inputs=evidence_inputs,
          remaining_catalog={k:v for k,v in remaining.items() if k!='rows'},
          latest_combined_defaults_promoted=False,all_experiments_complete=False))
    write(OUT/'inventory.json',inventory);write(OUT/'index.json',dict(cells=cells,notes=notes,decisions=decisions,coverage=coverage,pending_work=pending_work,remaining_catalog=remaining))
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
