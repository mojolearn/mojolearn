#!/usr/bin/env python3
"""Future own-arm worker over frozen existing benchmark constructors.

Discovery and compilation never invoke this file. A resolved recipe and later
execution authorization are mandatory. No opponent process or build is launched.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from six_lane_matrix_io import read_matrix
from six_lane_ab import ROOT, check_benchmark, digest, git, write, THREAD_ENV, runtime_requirements, STORE
from six_lane_evidence import capture, capture_model

HARNESS_FAMILIES={
    'tools/bench_board_neural.py':'neural',
    'tools/classical_two_datasets.py':'classical',
    'tools/bench_board_more.py':'more',
    'tools/bench_board_algos.py':'expanded',
    'bench/speed/forest_speed_arm.py':'forest',
}


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def shape_manifest(data):
    return {k:list(v.shape) for k,v in data.items() if hasattr(v,'shape')}


def hardware_record(runner):
    record=dict(system=platform.system(),machine=platform.machine(),platform=platform.platform(),cpu_count=os.cpu_count(),native_readback=getattr(runner,'info',{}))
    if sys.platform=='darwin':
        record['sysctl']={k:subprocess.check_output(['sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.ncpu','hw.memsize')}
    record['affinity']=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    record['cgroup']={str(p):p.read_text().strip() for p in map(Path,('/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us')) if p.is_file()}
    return record


def resource_setup():
    """Size Linux pools from this worker's actual allocation; Apple unrestricted."""
    for name in THREAD_ENV:os.environ.pop(name,None)
    if sys.platform=='darwin':return dict(policy='dedicated Apple; CPU libraries unrestricted')
    allowed=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    quota=None;source=None
    p=Path('/sys/fs/cgroup/cpu.max')
    if p.is_file():
        raw=p.read_text().split()
        if raw[0]!='max':quota=int(raw[0])/int(raw[1]);source=str(p)
    if quota is None:
        for base in ('/sys/fs/cgroup/cpu','/sys/fs/cgroup/cpu,cpuacct'):
            q=Path(base+'/cpu.cfs_quota_us');period=Path(base+'/cpu.cfs_period_us')
            if q.is_file() and period.is_file() and int(q.read_text())>0:
                quota=int(q.read_text())/int(period.read_text());source=base;break
    cap=max(1,min(allowed,int(quota))) if quota else max(1,allowed)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(cap)
    return dict(policy='whole worker allocation; estimator semantics and nested pools preserved',affinity_cpus=allowed,cgroup_quota_cpus=quota,quota_source=source,configured_library_threads=cap,utilization='not inferred')


def saved_inputs(module,family,work):
    import numpy as np
    if family=='forest':
        from six_lane_forest_adapter import inputs
        return inputs(module,work)
    if family=='neural':
        with np.load(work['data_file'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        return data,{}
    if family=='expanded':
        block,saved=module._load_block(work['lane'],work['dataset'],work['data_directory'])
        return module.lane_arrays(work['lane'],block),saved
    block=module.BLOCK_OF[work['lane']] if family=='classical' else module.block_of(work['lane'])
    base=Path(work['data_directory'])/(block+'-'+work['dataset'])
    with np.load(str(base)+'.npz',allow_pickle=False) as z:data={k:np.ascontiguousarray(z[k]) for k in z.files}
    saved=json.loads(Path(str(base)+'.json').read_text())
    return (data if family=='classical' else module.lane_arrays(work['lane'],data,saved)),saved


def make_runner(module,family,work,data,saved,race_arm):
    if family=='forest':
        from six_lane_forest_adapter import Runner
        return Runner(module,work,data,'fast' if race_arm=='ours-fast' else 'identical')
    if family=='neural':return module.build_runner(work['lane'],race_arm,'full',data)
    if family=='expanded':return module.build(work['lane'],race_arm,data)
    if family=='more':return module.build(work['lane'],race_arm,data,saved)
    return module.BUILDERS[(work['lane'],race_arm)](data,saved)


def complete_operation(module,family,work,data,runner):
    start=time.perf_counter()
    if family=='forest':runner.out=None
    if family=='expanded':runner.fit()
    else:runner.call();runner.sync()
    fitted=time.perf_counter();inference={};inference_outputs=None
    if work['inference']=='separate':
        if family=='classical':
            inf=module.infer_runner(work['lane'],runner,data);inf.call();inf.sync();inference=inf.outputs()
        elif family=='more':
            # These saved runners perform prediction and its host conversion
            # in outputs(). Keep that exact operation once, inside the declared
            # inference interval, instead of issuing duplicate predictions.
            inference_outputs=runner.outputs();runner.sync()
        elif hasattr(runner,'infer') and runner.infer():pass
        else:raise ValueError('No existing separate inference operation for this saved race')
    inferred=time.perf_counter()
    outputs=inference_outputs if inference_outputs is not None else runner.outputs()
    if hasattr(runner,'sync'):runner.sync()
    elif hasattr(runner,'sync_for_receipt'):runner.sync_for_receipt()
    end=time.perf_counter()
    if inference:outputs=dict(outputs,separate_inference=inference)
    timings=dict(fit_or_training_seconds=fitted-start,consumed_output_seconds=end-inferred)
    if work['inference']=='separate':timings['inference_seconds']=inferred-fitted
    return outputs,end,timings


def run(args):
    recipe=json.loads(args.recipe.read_text());job=recipe['job'];work=recipe['workload'];cfg=job['master_selection'][args.arm]
    check_benchmark()
    if recipe['source_sha']!=git('rev-parse','HEAD') or git('status','--porcelain','--untracked-files=no'):raise ValueError('Worker source freeze changed')
    if not recipe.get('execution_authorized'):raise ValueError('Later measurement authorization must be recorded in the resolved recipe')
    if job.get('blocked') or not job.get('full_dataset_coverage'):raise ValueError('Incomplete full workload recipe')
    from six_lane_full_variants import validate_variant
    validate_variant(recipe)
    if work.get('overrides') or work.get('adapter'):raise ValueError('Frozen race settings cannot change to reach a candidate')
    configs={c['id']:c for c in read_matrix(STORE/'matrix.json.gz')['configurations']}
    if runtime_requirements(job['master_selection'],job['workload_id'],configs):raise ValueError('Supplemental public operation has no unchanged incumbent race adapter; coverage remains missing')
    harness=work['harness'];family=HARNESS_FAMILIES[harness]
    if digest(ROOT/harness)!=work['harness_sha256']:raise ValueError('Harness drift')
    if work.get('shape','full')!='full' or work['inference'] not in ('separate','included_in_operation','not_applicable'):raise ValueError('Full workload and inference boundary required')
    if not work['intrinsic_cap_audit']['reviewed'] or work['intrinsic_cap_audit'].get('unresolved'):raise ValueError('Intrinsic workload caps remain unresolved')
    package=Path(recipe['packages'][args.arm]).resolve()
    if package==ROOT/'python' or not (package/'mojolearn').is_dir():raise ValueError('Expected isolated frozen package')
    sys.path.insert(0,str(package));os.environ['PYTHONPATH']=str(package)
    for name in list(os.environ):
        if name.startswith('MOJOLEARN_'):os.environ.pop(name)
    os.environ.update(cfg['environment']);os.environ.update(MOJOLEARN_BENCH_INSTALLED='1',MOJOLEARN_NUMERIC_MODE=job['mode'],MOJOLEARN_VENDOR={'nvidia':'cuda','amd':'hip','apple':'metal','host':'cpu'}[recipe['vendor']])
    actual_resources=resource_setup()
    for item in work['input_files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen input changed: '+item['path'])
    module=load(harness,'master_'+family+'_harness')
    if family=='neural' and work['lane'] not in module.LANES:raise ValueError('No existing neural race')
    if family=='neural' and ((module.DEVICE_OF[work['lane']]=='cpu') != (recipe['vendor']=='host')):raise ValueError('Frozen race does not cover the selected device column')
    race_arm='ours-fast' if job['mode']=='fast' else 'ours'
    start=time.perf_counter();data,saved=saved_inputs(module,family,work)
    runner=make_runner(module,family,work,data,saved,race_arm);prepared=time.perf_counter()
    outputs,end,timings=complete_operation(module,family,work,data,runner)
    timings.update(full_operation_seconds=end-start,preparation_seconds=prepared-start,cold_seconds=end-start)
    # Everything below (shape/setting admission, hashes, model export, metrics,
    # environment inspection and persistence) is outside the timed operation.
    if hasattr(runner,'receipt'):runner.receipt()
    actual_shapes=shape_manifest(data)
    if actual_shapes!=work['actual_shapes']:raise ValueError('Observed shapes differ; hidden cap or dataset drift')
    params_module=load('tools/classical_two_datasets.py','master_parameter_records')
    actual_settings=params_module.params_record(getattr(runner,'record',getattr(runner,'params_obj',getattr(runner,'params',None))))
    from six_lane_classification_variants import normalize_parameter_record
    actual_settings=normalize_parameter_record(work,actual_settings)
    if actual_settings!=work['estimator_settings_record']:raise ValueError('Observed estimator settings differ from the saved race')
    info=getattr(runner,'info',{})
    if info.get('numeric_mode_used')!=job['mode']:raise ValueError('Native numeric-mode readback is missing or differs')
    if info.get('vendor_used')!=work['runtime_vendor']:raise ValueError('Native runtime vendor is missing or differs')
    output=capture(outputs,'all declared consumed outputs',expected_paths=work['output_paths'])
    from six_lane_classification_variants import validate_output_scope
    validate_output_scope(work,output)
    from six_lane_mlp_variants import validate_output_scope as validate_mlp_output
    validate_mlp_output(work,output)
    if output['missing_state']:raise ValueError('Incomplete consumed output scope')
    state=capture_model(runner,work.get('model_state_paths'),retain_path=args.output.with_suffix('.model-state.json'))
    # The unchanged quality functions own metric definitions. Gate outcomes are
    # separate, retained evidence; a self-relative metric is never acceptance.
    if family=='forest':metrics=runner.quality()
    elif family=='neural':metrics=module.quality(work['lane'],data,{race_arm:outputs},shape='full')
    elif family=='classical':metrics=module.quality(work['lane'],data,{race_arm:outputs},saved)
    else:metrics=module.quality(work['lane'],data,{race_arm:outputs})
    loaded={};expected=job['artifact_provenance'][args.arm]
    observed={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith(('.so','.dylib'))}
    for artifact in expected:
        # FAST estimators also load IDENTICAL input helpers. Both tiers may
        # legitimately contain _mojolearn.so; the receipt binds the exact path.
        required_path=Path(artifact['path']).resolve()
        if not required_path.is_relative_to(package) or required_path not in observed or digest(required_path)!=artifact['sha256']:
            raise ValueError('Required artifact not observed in worker: '+artifact['path'])
        loaded[artifact['path']]=artifact['sha256']
    pools=None
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
    except ImportError:pass
    repeated=[]
    # Repeated use is opt-in in the resolved, unchanged saved recipe. The count
    # is recorded; absent scope stays pending instead of inventing extra work.
    for index in range(work.get('repeated_operations',0)):
        again=time.perf_counter();values,done,parts=complete_operation(module,family,work,data,runner)
        parts['repeated_seconds']=done-again
        repeated_output=capture(values,'repeated consumed outputs',expected_paths=work['output_paths'])
        validate_output_scope(work,repeated_output)
        validate_mlp_output(work,repeated_output)
        repeated.append(dict(index=index,timings=parts,outputs=repeated_output,model_state=capture_model(runner,work.get('model_state_paths'),retain_path=args.output.with_suffix('.repeated-'+str(index)+'.model-state.json'))))
    from six_lane_evidence import retain_values
    values_path=args.output.with_suffix('.values.json');retain_values(outputs,values_path)
    counts=dict(excluded_warmups=int(args.phase=='warmup'),scored=int(args.phase=='scored'))
    result=dict(schema='mojolearn.full-ab-result/1',status='PASS',source_sha=recipe['source_sha'],dataset_sha256=job['dataset_sha256'],dataset_version=work['dataset_version'],dataset_split=work['split'],seed=work['seed'],mode=job['mode'],vendor=recipe['vendor'],arm=args.arm,phase=args.phase,dimensions=actual_shapes,estimator_settings=actual_settings,full_dataset_coverage=True,timed_boundary=job['timed_boundary'],
        timings=timings,repeated_use=repeated,missing_timing_scopes=[] if repeated else ['repeated use'],outputs=output,retained_output_values=str(values_path),output_sha256=output['sha256'],model_state=state,loaded_artifacts=loaded,configuration=cfg,implementation_ids=job['implementation_ids'],workload_id=job['workload_id'],hashing_outside_timing=True,source_coverage_pending=job.get('source_coverage_pending',[]),
        hardware=hardware_record(runner),declared_hardware=recipe.get('hardware'),compiler=[a['compiler'] for a in expected],thread_environment={k:os.environ.get(k) for k in THREAD_ENV},resource_policy=actual_resources,declared_resource_policy=recipe['resource_policy'],effective_pools=pools,harness_sha256=work['harness_sha256'],sample_counts=counts,
        task_quality=dict(status='PENDING',metrics=metrics,gate_source=work['quality_gate_source'],reason='Existing independent gate assessment remains required; no acceptance inferred from metrics alone'),runtime_reach=info)
    if recipe['source_sha']!=git('rev-parse','HEAD') or git('status','--porcelain','--untracked-files=all'):raise ValueError('Worker source freeze changed during operation')
    write(args.output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--arm',choices=('A','B'),required=True);p.add_argument('--phase',choices=('warmup','scored'),required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except Exception as exc:
        write(args.output,dict(status='FAILED',error=repr(exc),arm=args.arm,phase=args.phase));raise


if __name__=='__main__':main()
