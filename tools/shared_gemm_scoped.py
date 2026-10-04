#!/usr/bin/env python3
"""Future M3-only scoped quality/timing. One cold target call per timed arm.

SOURCE TAG SCOPE PRIOR_ALGORITHM_PASS_TAG --action quality
 SOURCE TAG SCOPE SCOPED_QUALITY_TAG --action timing
No build, remote call, queue edit, warmup, repetition, or board publication.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from shared_gemm_downstream_pair import atomic_install, record, run_logged
from shared_gemm_preflight import ARTIFACTS, CASE_REQUIREMENTS, InfrastructureError, check_loaded
from shared_gemm_source_contract import validate_source

if not __debug__:
    raise RuntimeError('assertions required')

POLICY = 'shared-scoped-cold-firstread-v1'
# case, phase, fixed query rows (0 means retain original small fixture)
SCOPES = {'ols-fit-small':('ols','fit',0),
          'pca-transform-small':('pca','transform',0),
          'pca-inverse-small':('pca','inverse',0),
          'knn-wide-k-small':('knn-wide-k','kneighbors',0),
          'pca-transform-tall':('pca','transform',32769),
          'pca-inverse-tall':('pca','inverse',32769)}


def require_m3():
    try:
        chip = subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    except (OSError,subprocess.CalledProcessError) as exc:
        raise InfrastructureError('cannot verify M3 execution machine') from exc
    if not re.search(r'Apple M3(?: |$)',chip):
        raise InfrastructureError('scoped execution requires M3, observed '+chip)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_identity(arrays):
    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        h.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        h.update(value.tobytes(order='C'))
    return h.hexdigest()


def safe_tag(tag):
    assert re.fullmatch('[A-Za-z0-9_.-]+', tag), 'invalid tag'
    return tag


def verified_arms(root, source, case):
    config_path = root/'tools/shared_gemm_variant.json'
    config = json.loads(config_path.read_text())
    family = CASE_REQUIREMENTS[case][0]
    arms = Path.home()/'mq/verified-arms'/source/family
    m = json.loads((arms/'manifest.json').read_text())
    for key in ('contract','base_source','variant','families','defines_A','defines_B'):
        assert m[key] == config[key], key
    variant = config['variant']
    assert variant in (1,5) and m['source_sha'] == source and m['binding'] == family
    assert m['numeric_mode'] == 'fast' and m['builder'] == 'm2' and m['compile_only'] is True
    assert m['artifact'] == ARTIFACTS[family] and m['contract_sha'] == sha(config_path)
    assert m['target_cpu'] == 'apple-m1'
    assert m['target_column'] == 'apple' and m['target_accelerator'] == 'metal:1'
    assert m['defines_A'].split() == ['-D','MOJOLEARN_APPLE_FAST_SHARED_GEMM_COUNTERS']
    assert m['defines_B'].split() == m['defines_A'].split()+['-D',f'MOJOLEARN_APPLE_FAST_SHARED_GEMM_G{variant}']
    assert m['hashes'] == {a:sha(arms/(a+'.so')) for a in ('A','B')}
    return arms, m


def algorithm_gate(directory, source, scope, hashes, variant):
    case, phase, _ = SCOPES[scope]
    receipt = json.loads((directory/'PASS.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    assert receipt['status'] == report['status'] == 'PASS'
    assert receipt['source_sha'] == report['source'] == source
    assert receipt['hashes'] == hashes and receipt['variant'] == variant
    assert receipt['case'] == report['case'] == case and receipt['features'] == 65
    assert receipt['report_sha'] == sha(directory/'report.json')
    assert report['binary_sha'] == [hashes['A'],hashes['B']]
    assert report['policy'] == 'shared-downstream-f64-independent-errors-v2'
    assert report['metrics'] and all(report['exact'].values())
    for item in report['metrics'].values():
        assert item['pass_no_worse'] and all(math.isfinite(item['B'][k]) and item['B'][k] <= item['A'][k] for k in item['A'])
    assert sum(row[3] for row in report['phases_A'][phase]) == 0
    assert sum(row[3] for row in report['phases_B'][phase]) > 0, 'scope itself did not reach candidate'
    assert sha(directory/'A.npz') == report['capture_sha'][0]
    return dict(receipt_sha=sha(directory/'PASS.json'),report_sha=sha(directory/'report.json'),
                capture_sha=sha(directory/'A.npz'),directory=str(directory))


def make_packet(prior, scope, output):
    import numpy as np
    case, phase, rows = SCOPES[scope]
    with np.load(prior/'A.npz', allow_pickle=False) as saved:
        names = ('x','y') if case=='ols' else (('x','q') if case=='knn-wide-k' else ('q','components_','mean_','transform'))
        arrays = {name:np.array(saved[name],copy=True,order='C') for name in names}
    assert arrays['x' if case in ('ols','knn-wide-k') else 'q'].shape[1] == 65
    assert rows == 0, 'tall profile requires native baseline model preparation'
    if case == 'pca':
        assert arrays['components_'].shape == (7,65) and arrays['mean_'].shape == (65,)
        arrays['input'] = arrays.pop('q') if phase=='transform' else arrays.pop('transform')
        arrays.pop('q',None); arrays.pop('transform',None)
    np.savez(output, **arrays)
    return array_identity(arrays)


def worker(args):
    import numpy as np
    import time
    root = Path(__file__).resolve().parents[1]
    provenance = validate_source(root,args.source)
    case, phase, _ = SCOPES[args.scope]
    observed = check_loaded(root,case,args.source,args.binary_sha,args.variant,{})
    if args.preflight_only:
        record(args.output,dict(status='PREFLIGHT_PASS',provenance=provenance,preflight=observed))
        return 0
    if args.prepare_tall:
        assert args.variant == 0 and SCOPES[args.scope][2] == 32769 and case == 'pca'
        from mojolearn.decomposition import PCA
        rng = np.random.default_rng(919730)
        training = rng.normal(size=(513,220)).astype('float32')
        model = PCA(n_components=64,svd_solver='full').fit(training)
        arrays = {'components_':np.array(model.components_,copy=True,order='C'),
                  'mean_':np.array(model.mean_,copy=True,order='C'),
                  'input':rng.normal(size=(32769,220 if phase=='transform' else 64)).astype('float32')}
        assert arrays['components_'].shape == (64,220)
        assert all(np.all(np.isfinite(value)) for value in arrays.values())
        np.savez(args.output,**arrays)
        record(Path(args.output).with_suffix('.json'),dict(status='PREPARED_UNSCORED',source=args.source,
            provenance=provenance,scope=args.scope,preflight=observed,model_training_sha=array_identity({'x':training}),
            input_identity=array_identity(arrays),packet_sha=sha(args.output),scored=False,
            model_origin='actual baseline A PCA(full) fit 513x220,64 components; fixed common model for isolated target calls'))
        return 0
    with np.load(args.packet,allow_pickle=False) as z:
        arrays = {name:np.array(z[name],copy=True,order='C') for name in z.files}
    assert array_identity(arrays) == args.input_identity
    if case == 'ols':
        from mojolearn.linear_model import LinearRegression
        model = LinearRegression()
        def call():
            model.fit(arrays['x'], arrays['y'])
            return {'coef':model.coef_,'intercept':model.intercept_,
                    'x_mean':model._x_mean,'y_mean':model._y_mean}
    elif case == 'pca':
        from mojolearn.decomposition import PCA
        model = PCA(n_components=arrays['components_'].shape[0],svd_solver='full')
        model.components_=arrays['components_']; model.mean_=arrays['mean_']
        model.n_components_=arrays['components_'].shape[0]; model.n_features_in_=arrays['components_'].shape[1]; model.n_samples_=513
        def call():
            value = model.transform(arrays['input']) if phase=='transform' else model.inverse_transform(arrays['input'])
            return {'output':value}
    else:
        from mojolearn.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=65,algorithm='brute')
        model.fit(arrays['x'])  # stores index; lazy device preparation remains inside kneighbors
        def call():
            distances, indices = model.kneighbors(arrays['q'])
            return {'distances':distances,'indices':indices}
    from mojolearn import _backend
    b = _backend.binding('_mojolearn' if case=='knn-wide-k' else '_mojolearn_estimators','fast')
    b.shared_gemm_reset()
    elapsed = None
    if args.measure:
        assert args.measurement_key and re.fullmatch('[0-9a-f]{64}',args.measurement_key)
        ledger=Path.home()/'mq/scored/shared-gemm-scoped'
        reservation=json.loads((ledger/(args.measurement_key+'.STARTED.json')).read_text())
        arm='A' if args.variant==0 else 'B'
        assert reservation['source']==args.source and reservation['scope']==args.scope
        assert reservation['hashes'][arm]==args.binary_sha and reservation['input_identity']==args.input_identity
        assert Path(args.output)==Path(reservation['output'])/(arm+'.npz')
        record(ledger/(args.measurement_key+'.'+arm+'.CALLED.json'),dict(source=args.source,scope=args.scope,
            output=args.output,input_identity=args.input_identity,binary_sha=args.binary_sha))
        start = time.perf_counter_ns()
        result = call()
        output = {name:np.array(value,copy=True,order='C') for name,value in result.items()}
        elapsed = time.perf_counter_ns()-start
    else:
        result = call()
        output = {name:np.array(value,copy=True,order='C') for name,value in result.items()}
    # No result data was inspected before the copy above. It forces a full
    # first read into ordinary caller-owned arrays INSIDE the measured interval.
    counts = [[int(b.shared_gemm_count(r,c)) for c in range(4)] for r in range(4)]
    total = sum(row[3] for row in counts)
    if args.variant:
        selected = 1 if args.variant==1 else 2
        assert total>0 and total==sum(row[selected] for row in counts), 'NO_REACH or wrong variant'
    else:
        assert total==0, 'baseline launched candidate'
        assert sum(row[0] for row in counts)>0, 'baseline did not enter shared route'
    assert all(np.all(np.isfinite(value)) for value in output.values())
    np.savez(args.output,**output)
    record(Path(args.output).with_suffix('.json'),dict(policy=POLICY,source=args.source,
        provenance=provenance,scope=args.scope,preflight=observed,variant=args.variant,
        input_identity=args.input_identity,binary_sha=args.binary_sha,counts=counts,
        output_sha=sha(args.output),elapsed_ns=elapsed,scored=args.measure,
        scenario='cold first target call, input/model preparation excluded; full output copy included',
        diagnostic_counters=True))
    return 0


def assess(packet, a_path, b_path, scope):
    import numpy as np
    with np.load(packet,allow_pickle=False) as p:
        z={k:p[k].astype('float64') for k in p.files}
    with np.load(a_path,allow_pickle=False) as a, np.load(b_path,allow_pickle=False) as b:
        av={k:np.array(a[k]) for k in a.files}; bv={k:np.array(b[k]) for k in b.files}
    case, phase, _ = SCOPES[scope]
    exact={}; oracle={}
    if case=='ols':
        xc=z['x']-z['x'].mean(0); yc=z['y']-z['y'].mean()
        coef=np.linalg.lstsq(xc,yc,rcond=None)[0]
        oracle=dict(coef=coef,intercept=z['y'].mean()-z['x'].mean(0)@coef,
                    x_mean=z['x'].mean(0),y_mean=z['y'].mean())
    elif case=='pca':
        value=(z['input']-z['mean_'])@z['components_'].T if phase=='transform' else z['input']@z['components_']+z['mean_']
        oracle={'output':value}
    else:
        dist=np.sum((z['q'][:,None,:]-z['x'][None,:,:])**2,axis=2)
        ids=np.argsort(dist,axis=1,kind='stable')[:,:65]
        exact={'A_oracle_indices':bool(np.array_equal(av['indices'],ids)),
               'B_oracle_indices':bool(np.array_equal(bv['indices'],ids))}
        oracle={'distances':np.sqrt(np.take_along_axis(dist,ids,axis=1))}
    metrics={}
    for key,ref in oracle.items():
        ref=np.asarray(ref,dtype='float64'); scale=float(np.linalg.norm(ref.ravel())) or 1.0
        errors={}
        for arm,values in (('A',av),('B',bv)):
            assert values[key].shape==ref.shape, 'output/oracle shape mismatch'
            error=values[key].astype('float64')-ref
            errors[arm]=dict(scaled_l2=float(np.linalg.norm(error.ravel())/scale),max_absolute=float(np.max(np.abs(error))))
        passed=all(math.isfinite(errors['B'][k]) and errors['B'][k]<=errors['A'][k] for k in errors['A'])
        metrics[key]=dict(**errors,pass_no_worse=passed)
    return dict(status='PASS' if all(exact.values()) and all(m['pass_no_worse'] for m in metrics.values()) else 'HOLD',
                exact=exact,metrics=metrics)


def pair(args):
    import fcntl
    root=Path(__file__).resolve().parents[1]; os.chdir(root)
    provenance=validate_source(root,args.source)
    scope=args.scope; case,_,_=SCOPES[scope]
    arms,manifest=verified_arms(root,args.source,case)
    hashes=manifest['hashes']; variant=manifest['variant']
    prior=Path.home()/'mq/out'/(safe_tag(args.gate_tag)+'-quality')
    if args.action=='quality':
        gate=algorithm_gate(prior,args.source,scope,hashes,variant)
    else:
        q=json.loads((prior/'PASS.json').read_text()); qr=json.loads((prior/'report.json').read_text())
        assert q['status']==qr['status']=='PASS' and q['policy']==qr['policy']==POLICY
        assert q['source']==qr['source']==args.source and q['scope']==qr['scope']==scope and q['hashes']==hashes and q['variant']==variant
        assert qr['provenance']==provenance, 'scoped quality and timing require the same frozen harness'
        assert q['report_sha']==sha(prior/'report.json') and q['packet_sha']==sha(prior/'input.npz')
        assert qr['input_identity']==q['input_identity'] and qr['hashes']==hashes
        assert qr['metrics'] and all(qr['exact'].values())
        assert all(m['pass_no_worse'] and all(math.isfinite(m['B'][k]) and m['B'][k]<=m['A'][k] for k in m['A']) for m in qr['metrics'].values())
        assert sum(row[3] for row in qr['counts_A'])==0 and sum(row[3] for row in qr['counts_B'])>0
        gate=dict(scoped_receipt_sha=sha(prior/'PASS.json'),scoped_report_sha=sha(prior/'report.json'))
    lock=(Path.home()/'mq/shared-gemm-quality.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=Path.home()/'mq/out'/(safe_tag(args.tag)+('-quality' if args.action=='quality' else '-timing'))
    out.mkdir(parents=True,exist_ok=False)
    packet=out/'input.npz'
    if args.action=='quality':
        identity=None if SCOPES[scope][2] else make_packet(prior,scope,packet)
    else:
        shutil.copy2(prior/'input.npz',packet); identity=q['input_identity']
    so=root/'python/mojolearn'/manifest['artifact']
    assert not so.is_symlink()
    old=out/'original.so'; existed=so.exists(); old_sha=sha(so) if existed else None
    if existed: shutil.copy2(so,old)
    env=dict(os.environ,MOJOLEARN_NUMERIC_MODE='fast',MOJOLEARN_VENDOR='apple',MOJOLEARN_BENCH_INSTALLED='0',
             PYTHONPATH=str(root/'python'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env.pop('PYTHONOPTIMIZE',None)
    self_path=str(Path(__file__).resolve())
    def command(arm,selected,output):
        return [sys.executable,self_path,'worker',args.source,scope,str(packet),identity or 'prepare',hashes[arm],str(selected),str(output)]
    try:
        for arm,selected in (('A',0),('B',variant)):
            atomic_install(arms/(arm+'.so'),so)
            if run_logged(command(arm,selected,out/(arm+'.preflight.json'))+['--preflight-only'],out/(arm+'.preflight.log'),env):
                raise InfrastructureError(arm+' capability preflight failed before numerical calls')
        if identity is None:
            atomic_install(arms/'A.so',so)
            assert run_logged(command('A',0,packet)+['--prepare-tall'],out/'prepare.log',env)==0
            prepared=json.loads(packet.with_suffix('.json').read_text())
            assert prepared['status']=='PREPARED_UNSCORED' and prepared['source']==args.source
            assert prepared['provenance']==provenance and prepared['packet_sha']==sha(packet)
            identity=prepared['input_identity']
        record(out/'intake.json',dict(policy=POLICY,source=args.source,provenance=provenance,scope=scope,
            hashes=hashes,variant=variant,gate=gate,input_identity=identity,packet_sha=sha(packet)))
        if args.action=='timing':
            key=hashlib.sha256(json.dumps(dict(policy=POLICY,source=args.source,scope=scope,hashes=hashes,
                scenario='cold'),sort_keys=True).encode()).hexdigest()
            ledger=Path.home()/'mq/scored/shared-gemm-scoped'; ledger.mkdir(parents=True,exist_ok=True)
            record(ledger/(key+'.STARTED.json'),dict(tag=args.tag,output=str(out),scope=scope,gate=gate,
                source=args.source,hashes=hashes,input_identity=identity,rule='one target call per arm; no automatic replay'))
        for arm,selected in (('A',0),('B',variant)):
            atomic_install(arms/(arm+'.so'),so)
            cmd=command(arm,selected,out/(arm+'.npz'))
            if args.action=='timing': cmd.extend(['--measure','--measurement-key',key])
            assert run_logged(cmd,out/(arm+'.log'),env)==0, 'call or reach gate failed; no automatic replay'
        report=assess(packet,out/'A.npz',out/'B.npz',scope)
        am,bm=[json.loads((out/(arm+'.json')).read_text()) for arm in ('A','B')]
        assert am['output_sha']==sha(out/'A.npz') and bm['output_sha']==sha(out/'B.npz')
        assert am['input_identity']==bm['input_identity']==identity
        assert am['binary_sha']==hashes['A'] and bm['binary_sha']==hashes['B']
        assert am['provenance']==bm['provenance']==provenance
        report.update(policy=POLICY,source=args.source,provenance=provenance,scope=scope,hashes=hashes,
            variant=variant,input_identity=identity,packet_sha=sha(packet),gate=gate,
            counts_A=am['counts'],counts_B=bm['counts'],scored=args.action=='timing',
            elapsed_ns_A=am['elapsed_ns'],elapsed_ns_B=bm['elapsed_ns'],diagnostic_counters=True,
            scenario='cold call plus full first output copy; no warmup',board_evidence=False)
        record(out/'report.json',report)
    finally:
        if existed:
            atomic_install(old,so); assert sha(so)==old_sha
        else: so.unlink(missing_ok=True)
        record(out/'restore.json',dict(restored=True,original_sha=old_sha))
    passed=report['status']=='PASS'
    receipt=dict(status=report['status'],policy=POLICY,source=args.source,scope=scope,variant=variant,
        hashes=hashes,input_identity=identity,packet_sha=sha(packet),report_sha=sha(out/'report.json'),
        scored=args.action=='timing',board_evidence=False)
    record(out/('PASS.json' if passed else 'HOLD.json'),receipt)
    print('SHARED-SCOPED '+json.dumps(receipt,sort_keys=True),flush=True)
    return 0 if passed else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        sys.argv.pop(1)
        w=parser
        w.set_defaults(action='worker')
        w.add_argument('source'); w.add_argument('scope',choices=tuple(SCOPES)); w.add_argument('packet'); w.add_argument('input_identity'); w.add_argument('binary_sha'); w.add_argument('variant',type=int,choices=(0,1,5)); w.add_argument('output'); w.add_argument('--preflight-only',action='store_true'); w.add_argument('--measure',action='store_true'); w.add_argument('--measurement-key'); w.add_argument('--prepare-tall',action='store_true')
    else:
        parser.add_argument('source'); parser.add_argument('tag'); parser.add_argument('scope',choices=tuple(SCOPES)); parser.add_argument('gate_tag')
        parser.add_argument('--action',required=True,choices=('quality','timing'))
    args=parser.parse_args()
    require_m3()
    assert not (getattr(args,'measure',False) and (args.preflight_only or args.prepare_tall))
    return worker(args) if args.action=='worker' else pair(args)

if __name__=='__main__':
    try: sys.exit(main())
    except InfrastructureError as exc:
        print(json.dumps(dict(status='INFRASTRUCTURE_ERROR',error=str(exc))),flush=True); sys.exit(2)
