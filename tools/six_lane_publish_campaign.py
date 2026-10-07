"""Publish retained full pairs as pending evidence through the existing board tool.

Use --continuation MANIFEST to extend the same board. A manifest has schema
mojolearn.campaign-continuation/1, an id, optional matrix, sources (vendor, path,
optional route), quality_reviews (paths), and notes. Paths are relative to the
manifest. Sources use the queue's */attempts/*/receipt.json layout. Missing
sources/reviews remain explicitly pending. Registered manifests are remembered
by this board; historical receipts and reviews are retained across refreshes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from six_lane_review_history import publish_review_history
from six_lane_qualification import summarize as qualification_summary

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign-root',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--continuation',type=Path,action='append',default=[],
                    help='Register a targeted campaign manifest on the same board; repeatable')
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


def load(path,default):
    return json.loads(path.read_text()) if path.exists() else default


def retain_input(path,raw):
    """Keep exact input bytes, including superseded review/plan versions."""
    digest=hashlib.sha256(raw).hexdigest()
    target=OUT/'publication-inputs'/digest/path.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.write_bytes(raw)
    return dict(source=str(path),sha256=digest,snapshot=str(target.relative_to(REPO)))


def continuations():
    previous=load(OUT/'publication-continuations.json',{'manifests':[]})
    registered={r['source']:r for r in previous['manifests']}
    for path in args.continuation:registered.setdefault(str(path.resolve()),{})
    sources=[];reviews=[];configs=[];notes=[];records=[];ids=set()
    for name,old in registered.items():
        path=Path(name)
        raw=path.read_bytes() if path.exists() else (REPO/old['snapshot']).read_bytes()
        manifest=json.loads(raw)
        if manifest.get('schema')!='mojolearn.campaign-continuation/1':
            raise ValueError('Unsupported continuation schema: '+name)
        ident=manifest['id']
        if not ident or ident in ids:raise ValueError('Duplicate/empty continuation id: '+ident)
        ids.add(ident)
        record=retain_input(path,raw)
        record.update(id=ident,source_available=path.exists(),inputs=[],sources=[])
        resolve=lambda value:(path.parent/value).resolve()
        namespace='continuation-'+hashlib.sha256(name.encode()).hexdigest()[:16]
        for item in manifest.get('sources',[]):
            vendor=item['vendor'];source=resolve(item['path'])
            if vendor not in ('nvidia','amd','apple','host'):
                raise ValueError('Unsupported continuation vendor: '+vendor)
            controller=namespace+'-'+hashlib.sha256(str(source).encode()).hexdigest()[:16]
            sources.append(dict(vendor=vendor,path=source,controller=controller,
                route=item.get('route',vendor+' / route not recorded'),continuation=ident))
            record['sources'].append(dict(vendor=vendor,path=str(source),route=sources[-1]['route'],
                status='AVAILABLE' if source.is_dir() else 'PENDING_SOURCE',
                receipt_files=sum(1 for _ in source.glob('*/attempts/*/receipt.json'))))
        for kind,names in [('matrix',[manifest['matrix']] if manifest.get('matrix') else []),
                           ('quality_review',manifest.get('quality_reviews',[]))]:
            for value in names:
                source=resolve(value)
                if not source.exists():
                    record['inputs'].append(dict(source=str(source),kind=kind,status='PENDING_INPUT'))
                    continue
                data=source.read_bytes();document=json.loads(data)
                ref=dict(retain_input(source,data),kind=kind,status='RETAINED')
                record['inputs'].append(ref)
                if kind=='matrix':configs.extend(document['configurations'])
                else:
                    for row in document['rows']:
                        if not all(row.get(key) for key in ('receipt_sha256','quality_assessment','reason')):
                            raise ValueError('Incomplete hash-bound quality review: '+str(source))
                        reviews.append(dict(row,review_source=ref))
        evidence=manifest.get('qualification_evidence')
        if evidence:
            documents={};references={}
            requested={'quality':evidence['quality_review'],'identity':evidence['identity']}
            requested.update({'preservation:'+v:p for v,p in evidence.get('preservation',{}).items()})
            for kind,value in requested.items():
                source=resolve(value)
                if not source.exists():
                    record['inputs'].append(dict(source=str(source),kind=kind,status='PENDING_INPUT'))
                    continue
                data=source.read_bytes();documents[kind]=json.loads(data)
                references[kind]=dict(retain_input(source,data),kind=kind,status='RETAINED')
                record['inputs'].append(references[kind])
            if 'quality' in documents:
                record['qualification']=qualification_summary(documents['quality'],documents.get('identity'),
                    {k.split(':',1)[1]:v for k,v in documents.items() if k.startswith('preservation:')})
                record['qualification']['evidence']=references
                record['qualification']['continuation']=ident
        notes.extend(manifest.get('notes',[]))
        records.append(record)
    return sources,reviews,configs,notes,records


def add_candidate(candidates,selection,mode=None,vendor=None):
    ident=selection['id'];mode=mode or selection['mode']
    if selection.get('mode',mode)!=mode:raise ValueError('Selection mode differs: '+ident)
    vendors=set(selection.get('vendors',[]))
    if vendor:
        if vendors and vendor not in vendors:raise ValueError('Selection vendor differs: '+ident)
        vendors.add(vendor)
    if ident in candidates:
        if candidates[ident]['mode']!=mode:raise ValueError('Reused id has different mode: '+ident)
        vendors.update(candidates[ident]['vendors'])
    candidates[ident]=dict(id=ident,title=selection.get('title',selection.get('name',candidates.get(ident,{}).get('title',ident))),
        mode=mode,vendors=sorted(vendors))


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
    retained=load(OUT/'index.json',{'cells':[],'notes':[]})
    retained_cells={c['receipt_sha256']:c for c in retained['cells']}
    retained_summary=load(OUT/'retained-pairs.json',{'pairs':[]})
    retained_inventory=load(OUT/'inventory.json',{'candidates':[]})
    extra_sources,extra_reviews,extra_configs,extra_notes,continuation_records=continuations()
    qualification=[r['qualification'] for r in continuation_records if r.get('qualification')]
    qualification_rows={digest:row for q in qualification for digest,row in q['rows'].items()}
    candidates={c['id']:c for c in retained_inventory['candidates']}
    for selection in [dict(id='AF.X.complete-proposed',title='Apple FAST complete proposed configuration',mode='fast',vendors=['apple']),
                      dict(id='I.X.complete-proposed',title='IDENTICAL complete proposed configuration',mode='identical',vendors=['nvidia','amd','apple','host'])]+extra_configs:
        add_candidate(candidates,selection)
    review_path=ROOT/'quality-review/classical-12-pair-quality-review.json'
    review=load(review_path,load(OUT/'quality-review.json',{'rows':[]}))
    next_review_path=ROOT/'quality-review/next-reg-resample-quality-review.json'
    next_review=load(next_review_path,load(OUT/'next-quality-review.json',{'rows':[]}))
    # A missing external review must not erase an already retained assessment.
    reviewed={digest:dict(receipt_sha256=digest,quality_assessment=c['quality_assessment'],
        reason=c.get('quality_reason','Retained assessment'),**c.get('quality_comparisons',{}))
        for digest,c in retained_cells.items() if c.get('quality_assessment')}
    reviewed.update({r['receipt_sha256']:r for r in review['rows']+next_review['rows']+extra_reviews})
    receipt_paths={}
    sources=[dict(vendor=vendor,path=source,controller=source.relative_to(ROOT).as_posix().replace('/','--'),
                  route={'nvidia':'native-sm90','amd':'amd-native-gfx942','apple':'apple-fast'}[vendor])
             for vendor,source in SOURCES]+extra_sources
    current_origins={}
    for input_source in sources:
        vendor=input_source['vendor'];source=input_source['path']
        if not source.exists():continue
        for path in sorted(source.glob('*/attempts/*/receipt.json')):
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest in seen:continue
            seen.add(digest);receipt=json.loads(raw);job=receipt['workload']
            config=job['master_selection']['id']
            add_candidate(candidates,job['master_selection'],receipt['mode'],vendor)
            selected={r['arm']:r for r in receipt['runs'] if r['phase']=='scored' and r.get('returncode')==0 and r.get('result')}
            execution_status=receipt.get('status','IN_PROGRESS')
            complete=execution_status=='MEASURED_FULL' and set(selected)=={'A','B'}
            samples={arm:{phase:sum(run.get('arm')==arm and run.get('phase')==phase and run.get('returncode')==0 and bool(run.get('result')) for run in receipt['runs']) for phase in ('warmup','scored')} for arm in ('A','B')}
            controller=input_source['controller']
            target=OUT/'receipts'/vendor/controller/receipt['key']/path.parent.name/'receipt.json'
            if target.exists() and target.read_bytes()!=raw:
                target=target.with_name('receipt.'+digest+'.json')
            target.parent.mkdir(parents=True,exist_ok=True)
            # Publish the exact bytes we hashed even if the capture process
            # replaces its live receipt while this snapshot is being built.
            temp=target.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(target)
            receipt_paths[digest]=str(target.relative_to(REPO))
            row=dict(id=config,vendor=vendor,mode=receipt['mode'],case=job['workload_id']+'/'+
                     (controller+'/' if input_source.get('continuation') else '')+path.parent.name,
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
                     route=input_source['route'],receipt_origin=str(path.resolve()),
                     receipt_sha256=digest,source_coverage_pending=job.get('source_coverage_pending',[]))
            for field in ('prior_receipt_snapshots','quality_review_source'):
                if field in retained_cells.get(digest,{}):row[field]=retained_cells[digest][field]
            current_origins[row['receipt_origin']]=row
            if input_source.get('continuation'):row['continuation']=input_source['continuation']
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
                if assessment.get('review_source'):row['quality_review_source']=assessment['review_source']
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
    retained_details={row['evidence']:row for row in retained_summary['pairs']}
    for digest,row in retained_cells.items():
        if digest in seen:continue
        current=current_origins.get(row.get('receipt_origin'))
        if current and row['execution_status']=='IN_PROGRESS':
            current.setdefault('prior_receipt_snapshots',[]).append(row['evidence'])
            continue
        cells.append(row)
        if row['evidence'] in retained_details:summary.append(retained_details[row['evidence']])
        receipt_paths[digest]=row['evidence']
        # A newly supplied review can qualify task metrics on an older receipt
        # without rescanning or running the workload. Admission remains pending.
        if digest in reviewed and row['execution_status']=='MEASURED_FULL':
            assessment=reviewed[digest]
            if assessment.get('review_source'):row['quality_review_source']=assessment['review_source']
            row['quality_assessment']=assessment['quality_assessment'];row['quality_reason']=assessment['reason']
            row['quality_comparisons']={key:assessment[key] for key in (
                'candidate_vs_baseline','candidate_vs_opponents','baseline_vs_opponents',
                'opponent_metrics','metric_directions','new_full_opponent_review') if key in assessment}
            row['quality']='FAIL' if row['quality_assessment'] in ('FAILED_FAST_OPPONENT_GATE','QUALITY_FAILED') else 'PASS_TASK_METRICS' if row['quality_assessment']=='TASK_METRIC_GATE_PASSED' else 'PENDING'
            row['status']='QUALITY_FAILED' if row['quality']=='FAIL' else 'PENDING_ADMISSION'
            if row['evidence'] in retained_details:
                retained_details[row['evidence']].update(quality_assessment=row['quality_assessment'],quality_reason=row['quality_reason'])
    inventory=dict(campaign='six-lane-full-ab-20261006',identity_policy='NVIDIA/AMD same-arm IDENTICAL admission requires matching source/runtime provenance and complete typed fitted-state evidence. See the coverage table for completed and pending work; NVIDIA PTX artifacts remain pending. Apple FAST is evaluated by task quality; bits may differ.',evidence_policy='Complete full-workload executions are retained separately from quality and identity admission. Failed attempts preserved at controller-qualified paths; no default promotion. Hash receipts alone do not establish retained array/model bytes; see artifact-retention.json.',candidates=list(candidates.values()))
    notes=['A=candidate; B=incumbent. Timed evidence is pending admission, not a default promotion.',
           'Complete-proposed receipts measure combined configurations. Additional exact selection IDs identify isolated or interaction/dropout profiles; no result automatically credits its members.',
           'Initial 12-pair quality review: all12 preserve baseline metrics; 4 task-metric gates pass, 6 taxi opponent comparisons pending (historical4m vs current5.25m rows), Apple Istella KMeans fails best-opponent gate, NVIDIA inherits opponent-quality deficit. Additional saved assessments are retained in next-quality-review.json.',
           'One excluded warmup and one scored sample per arm. Original failed attempts are retained.',
           'The original AMD campaign used accepted retained artifacts and then stopped compilation. That dated stop is historical; later targeted build authorization and readiness belong to their own source freeze. No publisher action compiles, launches jobs or promotes defaults.',
           'IDENTICAL compares each same arm across vendors; unavailable typed complete model state remains incomplete.',
           'Scored output and partial/public-save model hashes are retained separately; partial hashes do not prove complete state identity.',
           'Apple first four PCA/OLS pairs overlapped shared workspace storage data transfer; KMeans overlap unestablished. No quiet-storage or promotion claim.',
           'Apple teardown preservation failed: the workspace was on the internal SSD, not retained EBS. Logs, timings, metrics and hash receipts survive; some raw array bytes remain unrecovered. See artifact-retention.json for exact recovery coverage and provenance. Original receipts are unchanged.',
           'Races reuse accepted binaries without separate numerical verification reruns. Earlier separately authorized AMD builds are historical artifact evidence, not measurements. Full provider and worker logs remain under '+str(ROOT)]
    retention=load(OUT/'artifact-retention.json',{})
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
        notes.append('Historical AMD missing-artifact compiler stop: see compilation-stopped.json. Accepted completed binaries remain reusable; interrupted/unbuilt jobs do not count as ready. Later targeted compilation has its own authorization and freeze.')
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
    for assessment in review['rows']+next_review['rows']+extra_reviews:
        assessment.setdefault('original_review_receipt_path',assessment.get('receipt'))
        assessment['receipt']=receipt_paths.get(assessment['receipt_sha256'],assessment.get('receipt'))
    review['publication_repair']='Original review retained externally; links relocated by exact receipt SHA256 into controller-qualified paths. Distinct failed and repaired attempts no longer collide.'
    write(OUT/'quality-review.json',review)
    if next_review['rows']:write(OUT/'next-quality-review.json',next_review)
    review_history=publish_review_history(OUT,extra_reviews)
    if (ROOT/'quality-review/historical-istella-opponents.json').exists():shutil.copyfile(ROOT/'quality-review/historical-istella-opponents.json',OUT/'historical-istella-opponents.json')
    decisions=[dict(candidate=c['id']+'/'+c['case'],
                    decision='NOT PROMOTED: '+c['quality_reason'],
                    commit=c['source_sha'],evidence=c['evidence'],
                    default_changed=False,individual_constituents='Only the recorded exact selection has this result; members receive no automatic decision')
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
        'nvidia': 'See exact selection and route receipts. PTX and unrecorded individual/full-workload scopes remain pending.',
        'amd': 'Selected retained-artifact pairs measured; GMM and other missing paired-artifact/recipe scopes remain pending.',
        'apple': 'See unrun scope for additional FAST pairs; earlier array-preservation and quality limitations remain.',
        'host': 'Same-arm identity and full-workload admission remain pending unless separately established.'}
    coverage=[]
    for vendor, label in [('nvidia','NVIDIA'),('amd','AMD GPU'),('apple','Apple'),('host','Host')]:
        rows=[c for c in cells if c['vendor']==vendor]
        if vendor=='host' and not rows:continue
        modes=sorted({c.get('mode',candidates[c['id']]['mode']).upper() for c in rows})
        coverage.append(dict(vendor=vendor,label=label+' / '+', '.join(modes),
            routes=sorted({c.get('route','NOT_RECORDED') for c in rows}),
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
        dict(vendor='all',scope='Individual candidates, alternative arms and other affected workloads',reason='Only exact recorded selections have receipts. Missing recipes, incompatible artifacts and untested interactions remain pending; combined results do not decide constituents.',evidence='experiments/six_lane_integration/catalog.json')]
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
    if continuation_records:
        continuation_ids={c['id'] for c in candidates.values()}-{'I.X.complete-proposed','AF.X.complete-proposed'}
        related=load(OUT/'continuation-coverage.json',{'configurations':[]})
        configurations={c['id']:c for c in related['configurations']}
        configurations.update({c['id']:c for c in extra_configs})
        continuation_rows=[]
        for ident in sorted(continuation_ids):
            rows=[c for c in cells if c['id']==ident]
            continuation_rows.append(dict(id=ident,members=configurations.get(ident,{}).get('members',[]),
                attempts=len(rows),complete_pairs=sum(c['execution_status']=='MEASURED_FULL' for c in rows),
                quality_failed=sum(c['status']=='QUALITY_FAILED' for c in rows),
                status='RECEIPTS_RETAINED_PENDING_ADMISSION' if rows else 'PENDING_MEASUREMENT',
                receipts=[c['evidence'] for c in rows]))
        write(OUT/'continuation-coverage.json',dict(configurations=list(configurations.values()),rows=continuation_rows,
            policy='Exact continuation IDs stay separate from catalog IDs. Members/aliases are cross-links, not automatic constituent measurement or promotion credit.'))
        write(OUT/'publication-continuations.json',dict(schema='mojolearn.campaign-continuations/1',manifests=continuation_records))
        notes.extend(extra_notes)
        notes.append('Targeted continuations: '+str(len(continuation_ids))+' additional exact profiles; '+
            str(sum(bool(r['attempts']) for r in continuation_rows))+' have receipts. See continuation-coverage.json for members and pending scope, and publication-continuations.json for exact plan/review snapshots and missing inputs. The catalog coverage ledger counts authored catalog IDs only; it does not relabel targeted profile IDs as catalog receipts.')
        for record in continuation_records:
            missing=[r['path'] for r in record['sources'] if r['status']!='AVAILABLE']+[
                r['source'] for r in record['inputs'] if r['status']=='PENDING_INPUT']
            if missing:pending_work.append(dict(vendor='all',scope='Continuation '+record['id'],
                reason='Pending sources/inputs: '+', '.join(missing),evidence='publication-continuations.json'))
        remaining['continuation_selections']=continuation_rows
    for cell in cells:
        if cell.get('receipt_sha256') in qualification_rows:
            cell['qualification']=qualification_rows[cell['receipt_sha256']]
    write(OUT/'current-qualification.json',dict(continuations=qualification))
    write(OUT/'remaining-work.json',remaining)
    write(OUT/'campaign-coverage.json',dict(coverage=coverage,pending_work=pending_work,evidence_inputs=evidence_inputs,
          remaining_catalog={k:v for k,v in remaining.items() if k!='rows'},
          latest_combined_defaults_promoted=False,all_experiments_complete=False))
    write(OUT/'inventory.json',inventory);write(OUT/'index.json',dict(cells=cells,notes=notes,decisions=decisions,coverage=coverage,pending_work=pending_work,remaining_catalog=remaining,review_history=review_history,
        qualification=[{k:v for k,v in q.items() if k!='rows'} for q in qualification]))
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
