#!/usr/bin/env python3
"""Distinct M3 resident-input matrix contract, one scored call per shape/arm.
SOURCE TAG QUALITY_REPORT QUALITY_SHA256. All66 calls fixed before execution.
No warmups, repetitions, opponent calls, production changes or builds.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import catalog_resident_quality as quality

ROOT = Path(__file__).resolve().parents[1]
ARMS = tuple(range(11))
SHAPES = (
    ('dense-nn', 2048, 512, 512, False, False),
    ('square-nn', 1024, 1024, 1024, False, False),
    ('tall-projection-nn', 32768, 64, 220, False, False),
    ('lowwidth-kmeans-nt', 32768, 8, 220, True, False),
    ('odd-nt', 4097, 71, 221, True, False),
    ('gram-nt', 1024, 1024, 220, True, True),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(args):
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    assert re.fullmatch('[A-Za-z0-9_.-]+', args.tag)
    assert re.fullmatch('[0-9a-f]{64}', args.report_sha256)
    os.chdir(ROOT)
    brand = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    assert brand == 'Apple M3 Ultra', brand
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == args.source
    subprocess.run(['git','diff','--quiet','HEAD','--','*.mojo','bindings/','python/',
                    'pixi.toml','pixi.lock','tools/catalog_resident_quality.py',
                    'tools/catalog_resident_timing.py'],check=True)
    report_path = Path(args.report).expanduser().resolve()
    assert digest(report_path) == args.report_sha256
    report = json.loads(report_path.read_text())
    arms = Path.home()/'mq/verified-arms'/args.source/'resident_gemm_probe'
    manifest = json.loads((arms/'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'resident_gemm_probe'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == '' and manifest['defines_B'] == '-D '+quality.DEFINE
    assert {a:digest(arms/(a+'.so')) for a in ('A','B')} == manifest['hashes']
    assert report['source_sha'] == args.source and report['catalog_source'] == quality.CATALOG
    assert report['binding_sha256'] == manifest['hashes']['B']
    assert report['fixture'] == quality.FIXTURE and report['abi_version'] == 1
    assert report['prior_report_sha256'] == quality.PRIOR_SHA256
    assert report['contract'] == 'resident-input-matrix' and report['n1_refused'] is True
    assert report['bound'] == quality.ABS_SCALED_BOUND and report['no_regression_tolerance'] == 0
    assert report['status'] == 'PASS' and report['failures'] == []
    assert report['variant_failures'] == {str(a):[] for a in ARMS[1:]}
    assert report['call_counts'] == {str(a):len(quality.CASES) for a in ARMS}
    assert len(report['cases']) == len(quality.CASES)
    for row,case in zip(report['cases'],quality.CASES):
        name,m,n,k,nt,alias,kind = case
        assert row['case'] == name and row['shape'] == [m,n,k] and n>=2
        assert row['nt'] is nt and row['alias'] is alias and row['status'] == 'PASS'
        assert set(row['metrics']) == {str(a) for a in ARMS}
        base = row['metrics']['0']
        assert base['finite'] and math.isfinite(base['scaled_error'])
        for a in ARMS[1:]:
            metric = row['metrics'][str(a)]
            err = metric['scaled_error']
            assert metric['finite'] and math.isfinite(err) and 0<=err<=quality.ABS_SCALED_BOUND
            assert err<=base['scaled_error'] and row['variant_status'][str(a)] == 'PASS'
        assert row['direct_shared_exact'] is True and row['prior_words_equal'] is True
    return arms/'B.so',manifest,report_path


def child(args,so):
    import numpy as np
    name,m,n,k,nt,alias = SHAPES[args.shape]
    spec = importlib.util.spec_from_file_location('_mojolearn_resident_gemm_probe',so)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    assert b.abi_version() == 1 and b.enabled() == 1
    assert {a:int(b.count(a)) for a in ARMS} == dict.fromkeys(ARMS,0)
    rng = np.random.default_rng(102604)
    a = rng.standard_normal((m,k)).astype('float32')
    physical_b = a if alias else rng.standard_normal((n,k) if nt else (k,n)).astype('float32')
    c = np.full((m,n),np.nan,dtype='float32')
    inputs = [hashlib.sha256(x.tobytes()).hexdigest() for x in (a,physical_b)]
    # Context/allocation/input upload/output poison and their completion are
    # outside timing. No GEMM launch or pipeline warmup is performed here.
    assert b.prepare(a.ctypes.data,physical_b.ctypes.data,[m,n,k,int(nt),int(alias)]) == 1
    assert {a:int(b.count(a)) for a in ARMS} == dict.fromkeys(ARMS,0)
    t0 = time.perf_counter_ns()
    reached = b.run_read(args.arm,c.ctypes.data)
    t1 = time.perf_counter_ns()  # output download + synchronize completed
    checksum = float(c.sum(dtype=np.float64))  # complete first host read
    t2 = time.perf_counter_ns()
    # Destroy buffers only after measured completion and first read.
    assert b.release() == 1
    counts = {a:int(b.count(a)) for a in ARMS}
    assert reached == args.arm and counts == {a:int(a==args.arm) for a in ARMS}
    assert bool(np.isfinite(c).all()) and math.isfinite(checksum)
    row = dict(shape=name,dimensions_M_N_K=[m,n,k],nt=nt,alias=alias,
        variant=args.arm,reached=reached,call_counts=counts,
        call_completion_ms=(t1-t0)/1e6,first_read_ms=(t2-t1)/1e6,total_ms=(t2-t0)/1e6,
        checksum=checksum,input_sha256=inputs,output_sha256=hashlib.sha256(c.tobytes()).hexdigest(),
        prepared_before_timer=True,output_transport_included=True,pipeline_warmed=False)
    print('CATALOG-RESIDENT-TIME '+json.dumps(row,sort_keys=True),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('source','tag','report','report_sha256'):p.add_argument(arg)
    p.add_argument('--shape',type=int,choices=range(len(SHAPES)))
    p.add_argument('--arm',type=int,choices=ARMS)
    args = p.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MOJOLEARN_VENDOR='apple',MOJOLEARN_NUMERIC_MODE='fast')
    so,manifest,report_path = validate(args)
    if args.shape is not None:
        assert args.arm is not None
        child(args,so)
        return
    assert args.arm is None
    out = Path.home()/'mq/out'/(args.tag+'-timing')
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for shape_index,shape in enumerate(SHAPES):
        for arm in ARMS:
            cmd=[sys.executable,str(Path(__file__).resolve()),args.source,args.tag,
                 str(report_path),args.report_sha256,'--shape',str(shape_index),'--arm',str(arm)]
            log=out/f'{shape[0]}-G{arm}.log'
            with log.open('x') as stream:
                result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
            assert result.returncode==0,'child failed; preserve log '+str(log)
            lines=[x for x in log.read_text().splitlines() if x.startswith('CATALOG-RESIDENT-TIME ')]
            assert len(lines)==1
            row=json.loads(lines[0].split(' ',1)[1])
            assert row['variant']==arm and row['shape']==shape[0]
            incumbent=records[-arm] if arm else row
            assert row['input_sha256']==incumbent['input_sha256']
            row['output_equal_incumbent']=row['output_sha256']==incumbent['output_sha256']
            row['quality_needs_review']=not row['output_equal_incumbent']
            records.append(row)
            print('CATALOG-RESIDENT-TIME '+json.dumps(row,sort_keys=True),flush=True)
    report=dict(source_sha=args.source,binding_hashes=manifest['hashes'],
        quality_report_sha256=args.report_sha256,contract='resident-input-matrix',
        preparation='existing context, allocated device buffers, completed input uploads',
        measured_boundary='one GEMM call + full output download + synchronize + first host sum',
        pipeline_warmed=False,pure_kernel_timing=False,scored_calls_per_shape_arm=1,
        fresh_process_per_call=True,warmups=0,opponents=0,production_admission=False,
        quality_needs_review=any(r['quality_needs_review'] for r in records),
        machine='Apple M3 Ultra',records=records)
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
