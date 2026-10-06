#!/usr/bin/env python3
"""Future one-sample worker using unchanged saved workload constructors.

This executable is never invoked by discovery or compilation. No opponent arm
is started here. The existing queue owns excluded warmups, scoring and retries.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from six_lane_ab import ROOT, check_benchmark, digest, git, write, THREAD_ENV
from six_lane_evidence import capture, capture_model


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def run(args):
    recipe=json.loads(args.recipe.read_text());job=recipe['job'];work=recipe['workload'];cfg=job['master_selection'][args.arm]
    check_benchmark()
    if recipe['source_sha']!=git('rev-parse','HEAD') or git('status','--porcelain','--untracked-files=no'):raise ValueError('Worker source freeze changed')
    if job.get('blocked') or not job.get('full_dataset_coverage'):raise ValueError('Incomplete full workload recipe')
    if recipe.get('changes_frozen_race') or work.get('overrides'):raise ValueError('Frozen race settings cannot change to reach a candidate')
    if not recipe.get('execution_authorized'):raise ValueError('Later measurement authorization must be recorded in the resolved recipe')
    # Package is authored by the later controller from exact matching compile
    # receipts. No in-place library replacement and no automatic build/install.
    package=Path(recipe['packages'][args.arm]).resolve()
    if package==ROOT/'python' or not (package/'mojolearn').is_dir():raise ValueError('Expected isolated frozen package')
    sys.path.insert(0,str(package));os.environ['PYTHONPATH']=str(package);os.environ['MOJOLEARN_BENCH_INSTALLED']='1'
    for name in list(os.environ):
        if name.startswith('MOJOLEARN_') and name not in ('MOJOLEARN_BENCH_INSTALLED',):os.environ.pop(name)
    os.environ.update(cfg['environment']);os.environ.update(MOJOLEARN_NUMERIC_MODE=job['mode'],MOJOLEARN_VENDOR=recipe['vendor'])
    if recipe['vendor'] in ('apple','host') and sys.platform=='darwin':
        for name in THREAD_ENV:os.environ.pop(name,None)
    for item in work['input_files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen input changed: '+item['path'])
    harness=work['harness']
    if digest(ROOT/harness)!=work['harness_sha256']:raise ValueError('Harness drift')
    # Import harness code before clocks, just as retained runners do. Binding
    # loading and model/input preparation stay inside the cold operation.
    if harness=='tools/bench_board_neural.py':
        module=load(harness,'master_neural_harness')
        if work['lane'] not in module.LANES:raise ValueError('Missing saved neural race')
        if work['shape']!='full':raise ValueError('Full saved shape required')
        ab=recipe.get('neural_configs',{}).get(args.arm)
        if cfg['runtime'] and not ab:raise ValueError('Explicit source operation needs a frozen neural config')
        if ab and ab.get('runtime',{}).get('operation'):raise ValueError('Supplemental source API has no incumbent race coverage')
        import numpy as np
        start=time.perf_counter()
        with np.load(work['data_file'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        runner=module.build_runner(work['lane'],'ours',work['shape'],data,ab_config=ab)
        runner.sync();prepared=time.perf_counter()
        runner.call();runner.sync();fitted=time.perf_counter()
        outputs=runner.outputs();runner.sync();end=time.perf_counter()
        quality_metrics=module.quality(work['lane'],data,{'ours':outputs},shape=work['shape'])
    elif harness in ('tools/classical_two_datasets.py','tools/bench_board_more.py','tools/bench_board_algos.py'):
        adapter=load('experiments/classical_identical_ideas/full_workload.py','master_classical_adapter')
        adapter.workload_facts(work,ROOT)
        if work.get('adapter'):raise ValueError('Supplemental public operation requires an existing frozen race recipe')
        module=adapter.load_harness(work['family'],ROOT)
        start=time.perf_counter();data,saved=adapter.load_inputs(module,work)
        runner=adapter.make_runner(module,work,data,saved,ROOT);prepared=time.perf_counter()
        adapter.complete_fit(runner,work['family']);fitted=time.perf_counter()
        if work['family']=='expanded':
            outputs=runner.outputs()
        else:
            outputs=runner.outputs();runner.sync()
        end=time.perf_counter()
        # Source-family quality adapters differ. Keep missing independent task
        # assessment pending; never derive quality from an output hash.
        quality_metrics=None
    else:raise ValueError('Missing master adapter; use retained tree/AFCL adapter with its exact result contract')
    output=capture(outputs,'all declared consumed outputs',expected_paths=work['output_paths'])
    state=capture_model(runner,work.get('model_state_paths'))
    if output['missing_state']:raise ValueError('Incomplete consumed output scope')
    loaded=[]
    expected=job['artifact_provenance'][args.arm]
    observed={Path(m.__file__).resolve():m for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith(('.so','.dylib'))}
    for artifact in expected:
        matches=[p for p in observed if p.name==Path(artifact['path']).name and p.is_relative_to(package)]
        if len(matches)!=1 or digest(matches[0])!=artifact['sha256']:raise ValueError('Required artifact not observed in this worker: '+artifact['path'])
        loaded.append(artifact)
    counts=dict(excluded_warmups=int(args.phase=='warmup'),scored=int(args.phase=='scored'))
    result=dict(schema='mojolearn.full-ab-result/1',status='PASS',source_sha=recipe['source_sha'],dataset_sha256=job['dataset_sha256'],dataset_version=work['dataset_version'],dataset_split=work['split'],seed=work['seed'],mode=job['mode'],vendor=recipe['vendor'],arm=args.arm,phase=args.phase,dimensions=job['dimensions'],estimator_settings=job['estimator_settings'],full_dataset_coverage=True,timed_boundary=job['timed_boundary'],
        timings=dict(full_operation_seconds=end-start,preparation_seconds=prepared-start,fit_or_training_seconds=fitted-prepared,consumed_output_seconds=end-fitted,cold_seconds=end-start),
        missing_timing_scopes=['separate inference','repeated use'],outputs=output,output_sha256=output['sha256'],model_state=state,loaded_artifacts=loaded,configuration=cfg,implementation_ids=job['implementation_ids'],workload_id=job['workload_id'],hashing_outside_timing=True,
        hardware=recipe['hardware'],compiler=[a['compiler'] for a in expected],thread_environment={k:os.environ.get(k) for k in THREAD_ENV},resource_policy=recipe['resource_policy'],effective_pools=getattr(runner,'info',{}),harness_sha256=work['harness_sha256'],sample_counts=counts,
        task_quality=dict(status='PENDING',metrics=quality_metrics,reason='Existing gate assessment must be retained; no acceptance inferred from metrics alone'),runtime_reach=getattr(runner,'info',{}))
    write(args.output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--arm',choices=('A','B'),required=True);p.add_argument('--phase',choices=('warmup','scored'),required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except Exception as exc:
        write(args.output,dict(status='FAILED',error=repr(exc),arm=args.arm,phase=args.phase));raise


if __name__=='__main__':main()
