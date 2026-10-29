"""Repair only untimed CPU reference evidence; never rerun a timed GPU arm."""
import json, os, pathlib, shutil, sys
import numpy as np
sys.path.insert(0,'/root/mojolearn/tools')
import bench_board_algos_quality_repair as bench
import bench_board_evidence as evidence
import cpu_quota

ROOT=pathlib.Path('/root/measure-neural-bc11389')
OUT=pathlib.Path(os.environ.get('MOJOLEARN_OPTIMIZER_QUALITY_OUT',str(ROOT/'quality-repair')));OUT.mkdir(parents=True,exist_ok=True)
env=dict(os.environ);cpu_quota.apply_cpu_quota(env);os.environ.update(env)
shutil.copyfile(bench.__file__,OUT/'bench_board_algos_quality_repair.py')
for name in ['test_bench_optimizer_reference_reset.py','optimizer-original-test.log','optimizer-repaired-test.log']:
    shutil.copyfile(ROOT/name,OUT/name)
for lane in ['adagrad','adam']:
    original=ROOT/lane/'raw'/(lane+'-synthetic.json')
    r=json.loads(original.read_text())
    with np.load(r['saved_outputs']['ours']) as z:outs={'ours':{k:z[k].copy() for k in z.files}}
    _,_,data=bench._lane_inputs(lane,'synthetic',str(ROOT/'data'))
    reference=bench.independent_quality_reference(lane,data,outs,str(OUT/lane),fit_calls=2,timeout=180)
    corrected=bench.quality(lane,data,outs)
    result={'lane':lane,'timed_gpu_rerun':False,'original_record':str(original),
            'original_record_sha256':evidence.file_hash(original),
            'original_quality':r['quality'],'corrected_quality':corrected,
            'repaired_reference':reference,'original_our_outputs':r['saved_outputs']['ours'],
            'original_our_outputs_sha256':evidence.file_hash(r['saved_outputs']['ours']),
            'repair_commit':'0796b859ddbe2fff6d9545428e7195583468267d',
            'contamination_scope':'timed own GPU worker ran in a separate process and closed before CPU reference; CPU reference aliases its own freshly generated p0 only; original timings/output bytes unchanged'}
    evidence.write(OUT/(lane+'.json'),result)
    print(json.dumps({'lane':lane,'reference_status':reference['status'],'corrected_ours':corrected.get('ours')}),flush=True)
