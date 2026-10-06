"""Untimed evidence and admission for the future six-lane A/B phase.

No product imports or runtime work at module import. No board writes, promotion,
or A-vs-B bitwise requirement: each IDENTICAL arm is its own numerical version.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def value_manifest(value, path='$'):
    """Describe canonical encoding without changing any tensor or dtype."""
    import numpy as np
    if isinstance(value,dict):
        return [row for k in sorted(value) for row in value_manifest(value[k],path+'.'+str(k))]
    if isinstance(value,(list,tuple)):
        return [row for i,v in enumerate(value) for row in value_manifest(v,path+'['+str(i)+']')]
    if isinstance(value,(np.ndarray,np.generic)):
        a=np.asarray(value)
        if a.dtype.hasobject:raise ValueError('Object state cannot establish identity')
        return [dict(path=path,dtype=a.dtype.str,shape=list(a.shape),encoding='C-order raw bytes; dtype endian preserved',bytes=a.nbytes)]
    return [dict(path=path,dtype=type(value).__name__,shape=[],encoding='canonical JSON scalar')]


def capture(value, scope, *, expected_paths=None):
    """Call only after the declared timed operation has ended."""
    from bench_board_state import canonical_hash
    manifest=value_manifest(value);actual={x['path'] for x in manifest}
    missing=sorted(set(expected_paths or [])-actual)
    return dict(status='CAPTURED',sha256=canonical_hash(value),encoding='mojolearn-benchmark-state-v1',scope=scope,manifest=manifest,missing_state=missing,
                completeness='complete_declared_scope' if expected_paths is not None and not missing else 'scope_not_qualified')


def retain_values(value,path):
    """Retain an exact typed tree plus .npy leaves, after clocks have stopped."""
    import numpy as np
    path=Path(path);folder=path.with_suffix('');folder.mkdir(exist_ok=False)
    leaves=[]
    def encode(v):
        if isinstance(v,dict):return {'dict':[[k,encode(child)] for k,child in v.items()]}
        if isinstance(v,(tuple,list)):return {'sequence':[encode(child) for child in v]}
        if isinstance(v,(np.ndarray,np.generic)):
            a=np.asarray(v)
            if a.dtype.hasobject:raise ValueError('Object arrays are not retained identity state')
            dest=folder/(str(len(leaves))+'.npy');np.save(dest,a,allow_pickle=False);leaves.append(str(dest))
            return {'array':str(dest),'dtype':a.dtype.str,'shape':list(a.shape),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        if v is None or isinstance(v,(str,int,float,bool)):return {'scalar':v}
        raise TypeError('Unsupported retained output type '+type(v).__name__)
    tree=encode(value);path.write_text(json.dumps(dict(schema='mojolearn.typed-values/1',tree=tree),allow_nan=False,indent=2)+'\n')


def capture_model(runner, expected_paths=None, *, retain_path=None):
    from bench_board_state import runner_model
    model=runner_model(runner)
    if model is None:return dict(status='UNAVAILABLE',reason='Runner exposes no public model owner',missing_state=['model owner'])
    if callable(getattr(model,'state_dict',None)):
        state=model.state_dict()
        result=capture(state,'public state_dict',expected_paths=expected_paths)
        if retain_path is not None:
            retain_values(state,retain_path)
            result['retained_values']=str(retain_path)
        return result
    # Only explicitly reviewed public save schemas can establish complete
    # fitted-state coverage. Generic prediction-only exports remain partial.
    # Existing receipts are never rewritten or upgraded by this future capture.
    from bench_board_state import public_fitted_state
    export_directory=Path(retain_path).with_suffix('.export') if retain_path is not None else None
    state,provenance=public_fitted_state(model,retain_directory=export_directory)
    if state is None:return provenance
    paths=provenance['contract_paths']
    result=capture(state,'complete public fitted state: '+provenance['contract'],expected_paths=paths)
    if expected_paths is not None and set(expected_paths)!=set(paths):
        result['completeness']='scope_not_qualified'
        result['missing_state']=sorted(set(paths)-set(expected_paths))
        result['unexpected_declared_paths']=sorted(set(expected_paths)-set(paths))
        result['reason']='Recipe model-state paths differ from the complete reviewed export contract'
    result['provenance']=provenance
    if retain_path is not None:
        retain_values(state,retain_path)
        result['retained_values']=str(retain_path)
    return result


def validate_master_result(data,job,config,arm,phase):
    expected=job['master_selection'][arm]
    if data.get('configuration')!=expected:raise ValueError('Actual arm configuration differs from master')
    if data.get('implementation_ids')!=job['implementation_ids']:raise ValueError('Implementation attribution differs')
    if data.get('workload_id')!=job['workload_id']:raise ValueError('Workload attribution differs')
    if data.get('hashing_outside_timing') is not True:raise ValueError('Hashing/report overhead must be excluded')
    for field in ('hardware','compiler','thread_environment','resource_policy','harness_sha256','dataset_version','dataset_split','seed','sample_counts'):
        if field not in data:raise ValueError('Missing provenance: '+field)
    counts=data['sample_counts']
    if counts!={'excluded_warmups':1 if phase=='warmup' else 0,'scored':1 if phase=='scored' else 0}:raise ValueError('Actual sample counts differ')
    out=data.get('outputs',{})
    if out.get('status')!='CAPTURED' or not out.get('manifest') or not out.get('encoding') or not out.get('scope'):raise ValueError('Output hash lacks typed capture metadata')
    state=data['model_state']
    if state.get('status')=='CAPTURED' and (not state.get('manifest') or 'missing_state' not in state):raise ValueError('Model-state hash lacks typed capture/missing-state metadata')
    quality=data.get('task_quality',{})
    if quality.get('status') not in ('PASS','FAIL','PENDING'):raise ValueError('Task quality must retain existing gate outcome or remain pending')
    if quality.get('status')=='FAIL':raise ValueError('Existing task-quality gate failed')
    return dict(timing='timed',quality='assessed' if quality['status']=='PASS' else 'not_assessed',identity='not_assessed',accepted=False,promoted=False,
                complete_state_available=state.get('completeness')=='complete_declared_scope' and not state.get('missing_state'))


def assess_identity(columns,mode):
    """Future assessment of retained receipts; never called by compilation."""
    if mode=='fast':return dict(status='NOT_REQUIRED',policy='Existing Apple FAST task-quality gates')
    outcome={}
    for arm in ('A','B'):
        rows=[columns.get(v,{}).get(arm) for v in ('nvidia','amd','apple','host')]
        if any(r is None for r in rows):outcome[arm]='PENDING_MISSING_COLUMN';continue
        if any(r['model_state'].get('completeness')!='complete_declared_scope' or r['model_state'].get('missing_state') for r in rows):outcome[arm]='PENDING_MODEL_STATE';continue
        # Configuration bytecode/target may vary but logical arm/workload/profile
        # must be exactly shared across columns. A is never compared against B.
        evidence=[(r['source_sha'],r['configuration'],r['dataset_sha256'],r['dimensions'],r['estimator_settings'],r['outputs'],r['model_state']) for r in rows]
        outcome[arm]='PASS' if all(e==evidence[0] for e in evidence[1:]) else 'FAIL'
    return dict(status='PASS' if all(x=='PASS' for x in outcome.values()) else 'PENDING_OR_FAILED',arms=outcome,promotion_vote=['nvidia','amd'])


def board_inputs(inventory,results):
    """Produce inputs for tools/performance_measurement_board.py only.

    Do not touch main/vendor opponent boards; own-only A/B has no opponent ratio.
    """
    cards=[dict(id=e['id'],title=e['title'],mode=e['mode'],vendors=e['vendors']) for e in inventory['entries']]
    cells=[]
    for result in results:
        runs=[r for r in result.get('runs',[]) if r.get('phase')=='scored' and r.get('returncode')==0]
        complete=len(runs)==2 and {r['arm'] for r in runs}=={'A','B'}
        for ident in result.get('implementation_ids',[]):
            cells.append(dict(id=ident,vendor=result['vendor'],case=result['key'],scope='full_workload',status='PENDING_ADMISSION' if complete else 'FAILED_OR_INCOMPLETE',evidence=result['receipt']))
    return dict(campaign='six-lane-integration',candidates=cards,identity_policy='Each IDENTICAL arm separately across NVIDIA/AMD/Apple/host; FAST uses existing task quality',evidence_policy='No inherited qualification; one excluded warmup and one scored sample'),dict(cells=cells,notes=['No opponent ratios or automatically admitted measurements. Use existing board tools after explicit qualification.'])
