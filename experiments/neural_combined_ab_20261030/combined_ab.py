"""Frozen-build full saved CausalLM workloads; sampled policy A/B, no promotion."""
import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time, traceback

ROOT=pathlib.Path(os.environ.get('MOJOLEARN_AB_SOURCE_ROOT','/root/combined-source'))
OUT=pathlib.Path(os.environ.get('MOJOLEARN_COMBINED_OUT','/root/results-combined-neural'))
PY=sys.executable
sys.path.insert(0,str(ROOT/'tools'))
import bench_board_neural as bench
import bench_board_evidence as evidence
import cpu_quota

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)

def retain(path, label):
    """Presigned transport; verified bytes must land before the next cell."""
    config=pathlib.Path(os.environ['MOJOLEARN_AB_RETENTION_CONFIG'])
    settings=json.loads(config.read_text())[label]
    import tempfile
    with tempfile.TemporaryDirectory(prefix='combined-retain-') as temp:
        temp=pathlib.Path(temp)
        for method in ('put','get'):
            private=temp/(method+'.private')
            private.write_text('url = '+json.dumps(settings[method])+'\n');private.chmod(0o600)
            command=['curl','--config',str(private),'-f','--retry','2','--max-time','180']
            command+=['-X','PUT','--upload-file',str(path)] if method=='put' else ['-o',str(temp/'verified')]
            with (path.parent/'retention-transport.log').open('a') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=210)
        actual=evidence.file_hash(path)
        if evidence.file_hash(temp/'verified')!=actual:raise RuntimeError('retained byte hash mismatch')
    return dict(key=settings['key'],sha256=actual,bytes=path.stat().st_size,status='R2 PUT and GET verified')

def retain_cell(dest, label):
    archive=dest/'retained.tar.gz'
    members={str(p.relative_to(dest)):p.read_bytes() for p in dest.rglob('*') if p.is_file() and p.name not in ('retained.tar.gz','retention.json','retention-transport.log')}
    evidence.archive(archive,members)
    write(dest/'retention.json',retain(archive,label))

def worker(a):
    import numpy as np
    from mojolearn.models import CausalLM
    dest=OUT/a.key/('read'+str(a.chunk))/a.case;dest.mkdir(parents=True,exist_ok=True)
    shape=bench.GEN_SHAPES['small' if a.smoke else 'full'];prompt_len=shape['prompt'];new_tokens=shape['new_tokens']
    rec={'model_key':a.key,'arm':a.arm,'read_chunk':a.chunk,'status':'started','smoke_only':a.smoke,
         'requested_flags':dict(sampler=a.arm,kv_reuse=a.kv_reuse,native_engine=a.engine,resident_forward=a.resident_forward,route=a.route),'case':a.case,
         'options':dict(greedy=False,seed=7,temperature=0.8,top_k=50,top_p=0.9,eos_token_id=None,pad_token_id=None),
         'saved_recipe':'tools/bench_board_neural.py GEN_SHAPES['+('small' if a.smoke else 'full')+'], LM_CHECKPOINTS',
         'batch':1,'prompt_tokens':prompt_len,'generated_tokens':new_tokens,
         'boundary':'stream creation plus all public reads, operation setup, KV loads/exports, synchronized host output conversion and concatenation; checkpoint load recorded separately',
         'warmup':'one excluded first invocation on freshly loaded model; cold operation time retained',
         'score':'one subsequent invocation with fresh stream and same retained model; no median/noise claim',
         'environment':evidence.runtime_environment()}
    write(dest/'result.json',rec)
    try:
        started=time.perf_counter()
        checkpoint_root=pathlib.Path(os.environ.get('MOJOLEARN_BOARD_LM_CHECKPOINTS',str(OUT/'checkpoints')))
        rec['checkpoint_input']=str(checkpoint_root/a.key)
        model=CausalLM.load(str(checkpoint_root/a.key),weight_format='float32',numeric_profile='fp32_v1',device='gpu')
        rec['model_load_ms']=(time.perf_counter()-started)*1000
        rec['model']=dict(vocab_size=int(model.vocab_size),d_model=int(model.d_model),n_layers=len(model.blocks),
                          vendor=model._native_vendor,numeric_mode=model._native_numeric_mode,
                          numeric_profile=model.numeric_profile,weight_format=model.weight_format)
        ids=np.random.default_rng(bench.SEED).integers(0,model.vocab_size,(1,prompt_len),dtype=np.int32)
        np.save(dest/'prompt.npy',ids)
        e=model._native_entries
        # Read optional instance metadata without invoking the known broken
        # native missing-attribute path in this frozen release candidate.
        fields=object.__getattribute__(model,'__dict__')
        if type(fields) is not dict:raise TypeError('model instance fields are not an exact dict')
        rec['route_admission_evidence']={
            'resident_sampling_v1':e.get('resident_sampling_v1'),
            'native_block_operations_present':e.get('native_block_operations') is not None,
            'parallel_call_absent':e.get('parallel_call') is None,
            'parameters_exposed':bool(fields.get('_parameters_exposed',False)),
            'kind':model.kind,'resident_vendor':e.get('resident_vendor'),
            'required_resident_entries_present':all(k in e for k in ['causal_lm_session_forward','transformer_decode_session_load_state','transformer_decode_session_export_state','causal_lm_session_run']),
            'private_state':'new stream privately allocates state; no resume/parameter exposure/alias injection',
            'interpretation':'admission metadata plus actual completed native-branch witness in excluded warmup; raw witness retained in execution.log',
            'source':str(ROOT/'bindings/causal_lm_public.mojo')}
        def descriptor_counters():
            current=model._native_entries
            return {name:repr(current[name]) for name in ('native_block_operations','native_primitives') if name in current}
        rounds=[]
        for index in range(2):
            os.environ['MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE']='1' if index==0 else '0'
            counters_before=descriptor_counters()
            started=time.perf_counter()
            stream=model.stream(ids,new_tokens,**rec['options'])
            chunks=[]
            for offset in range(0,new_tokens,a.chunk):
                chunks.append(np.asarray(stream.read(min(a.chunk,new_tokens-offset)),dtype=np.int32).copy())
            output=np.concatenate(chunks,axis=1)
            elapsed=(time.perf_counter()-started)*1000
            counters_after=descriptor_counters()
            if output.shape!=(1,new_tokens):raise ValueError('incomplete sampled output shape '+str(output.shape))
            if not ((output>=0)&(output<int(model.vocab_size))).all():raise ValueError('out of vocabulary token')
            np.save(dest/('warmup.npy' if index==0 else 'scored.npy'),output)
            info=stream.info()
            rounds.append(dict(role='excluded-cold-warmup' if index==0 else 'scored-repeated-model',ms=elapsed,
                               output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),stream_info=info,
                               descriptor_counters_before=counters_before,descriptor_counters_after=counters_after))
            rec['rounds']=rounds;write(dest/'result.json',rec)
        if rounds[0]['output_sha256'] != rounds[1]['output_sha256']:
            raise ValueError('same-policy warmup/scored seeded outputs differ')
        checkpoint=stream.state_dict()
        arrays={}
        def flatten(v,name='state'):
            if isinstance(v,dict):return {str(k):flatten(x,name+'/'+str(k)) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [flatten(x,name+'/'+str(i)) for i,x in enumerate(v)]
            if isinstance(v,(str,int,float,bool)) or v is None:return v
            if isinstance(v,bytes):return {'bytes_hex':v.hex()}
            arr=np.asarray(v)
            if arr.dtype.hasobject:raise TypeError('unretained checkpoint value '+name)
            arrays[name]=arr.copy()
            return {'array':name,'shape':list(arr.shape),'dtype':str(arr.dtype),'sha256':hashlib.sha256(arr.tobytes()).hexdigest()}
        write(dest/'checkpoint.json',flatten(checkpoint))
        np.savez(dest/'checkpoint-arrays.npz',**arrays)
        rec['checkpoint_schema']=checkpoint['schema']
        rec['sampling_policy']=checkpoint.get('sampling_policy','legacy FP64 policy')
        fields=object.__getattribute__(model,'__dict__')
        rec['route_admission_evidence']['retained_sessions_after_read']=fields.get('_resident_sessions') is not None
        rec['route_admission_evidence']['resident_busy_after_read']=bool(fields.get('_resident_busy',False))
        counts=np.bincount(output.ravel(),minlength=int(model.vocab_size))
        probs=counts[counts>0]/output.size
        rec['output_statistics']=dict(sample_tokens=int(output.size),unique_tokens=int(len(probs)),
            empirical_token_entropy_bits=float(-(probs*np.log2(probs)).sum()),
            adjacent_repeat_fraction=float(np.mean(output[:,1:]==output[:,:-1])),
            limitations='one sample on the saved random-token prompt; descriptive output statistics, not real-text quality or statistical equivalence qualification')
        # Preserve full logits as quality evidence, outside every clock. A failure
        # cannot erase timing/output bytes or be misrepresented as a quality pass.
        quality_env={name:os.environ.get(name) for name in ('MOJOLEARN_GENERATION_NATIVE_ENGINE','MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD','MOJOLEARN_EXPERIMENT_GENERATION_ROUTE')}
        try:
            os.environ.update(MOJOLEARN_GENERATION_NATIVE_ENGINE='0',MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD='1',MOJOLEARN_EXPERIMENT_GENERATION_ROUTE='auto')
            rec['quality_environment']={name:os.environ[name] for name in quality_env}
            full=np.concatenate([ids,output[:,:-1]],axis=1)
            logits=np.asarray(model.forward(full),dtype=np.float32)
            selected=logits[:,prompt_len-1:prompt_len-1+new_tokens,:].astype(np.float64)
            shift=selected.max(axis=-1,keepdims=True)
            logden=np.log(np.exp(selected-shift).sum(axis=-1))+shift[...,0]
            target=np.take_along_axis(selected,output[:,:,None],axis=-1)[...,0]
            nll=logden-target
            np.savez(dest/'quality-logits.npz',logits=logits,target_tokens=output,nll=nll)
            rec['quality']=dict(status='scored',mean_model_nll=float(nll.mean()),max_model_nll=float(nll.max()),
                finite_logits=bool(np.isfinite(logits).all()),reference='same frozen model teacher-forced full-prefix logits, untimed; not an independent quality oracle')
        except BaseException as exc:
            rec['quality']=dict(status='failed',error=repr(exc));(dest/'quality-error.log').write_text(traceback.format_exc())
        finally:
            for name,value in quality_env.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
        rec['status']='measured' if rec['quality'].get('status')=='scored' and rec['quality'].get('finite_logits') and np.isfinite(rec['quality']['mean_model_nll']) else 'failed-quality';rec['qualification']='NVIDIA-only one-sample full-workload A/B; completed route witness is required by controller; AMD/Apple and independent quality pending, no default promotion'
    except BaseException as exc:
        rec.update(status='failed',error=repr(exc));(dest/'error.log').write_text(traceback.format_exc())
    write(dest/'result.json',rec)
    print(json.dumps({k:rec.get(k) for k in ['model_key','arm','read_chunk','status','error','rounds','quality']}),flush=True)
    return 0 if rec['status']=='measured' else 1


def main():
    from matrix import configurations, route_configurations, parse_witness, validate_witness
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true');p.add_argument('--key')
    p.add_argument('--chunk',type=int);p.add_argument('--arm',type=int)
    p.add_argument('--kv-reuse',type=int);p.add_argument('--engine',type=int);p.add_argument('--case')
    p.add_argument('--route',default='auto');p.add_argument('--resident-forward',type=int,default=1)
    p.add_argument('--matrix',choices=('flags','routes'),default='flags')
    p.add_argument('--budget',type=int,default=1800);p.add_argument('--cell-seconds',type=int,default=250)
    p.add_argument('--smoke',action='store_true');p.add_argument('--models',default=','.join(bench.LM_CHECKPOINTS))
    a=p.parse_args()
    if a.worker:return worker(a)
    OUT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env.pop('PYTHONPATH',None);cpu_quota.apply_cpu_quota(env)
    env.update(MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD='1',
               MOJOLEARN_BENCH_INSTALLED='1',PYTHONUNBUFFERED='1')
    env.pop('MOJOLEARN_TRANSFORMER_LEGACY_SETUP',None)
    deadline=time.monotonic()+a.budget;rows=[]
    configs=route_configurations() if a.matrix=='routes' else configurations()
    models=a.models.split(',')
    if not all(x in bench.LM_CHECKPOINTS for x in models):raise ValueError('unregistered saved checkpoint')
    manifest={'schema':'mojolearn.combined-generation-ab.v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['bindings/causal_lm_public.mojo','tools/bench_board_neural.py']},
              'matrix':configs,'matrix_kind':a.matrix,'models':models,'read_chunks':[4,1] if a.smoke else [128,1],
              'expected_cells':len(models)*2*len(configs),'smoke_only':a.smoke,'numeric_mode':'identical',
              'full_recipe':bench.GEN_SHAPES['small' if a.smoke else 'full'],
              'hardware':evidence.hardware_context('nvidia'),'promotion':'none; NVIDIA-only experiment',
              'clock':'one trace-enabled excluded warmup then one trace-disabled scored whole public stream/read operation',
              'route_precedence':'resident sampling returns before native engine; KV reuse only legacy branch; requested flags may be inactive'}
    write(OUT/'manifest.json',manifest)
    source,inventory=evidence.source_snapshot(OUT/'source',repo=ROOT,drivers=[__file__])
    write(OUT/'source-retention.json',retain(pathlib.Path(source['local_path']),OUT.name+'/source'))
    for key in models:
      for chunk in manifest['read_chunks']:
       for cfg in configs:
        dest=OUT/key/('read'+str(chunk))/cfg['case'];dest.mkdir(parents=True,exist_ok=True)
        previous=dest/'result.json'
        if previous.exists():
            old=json.loads(previous.read_text())
            if old.get('status')=='unsupported' and old.get('source_commit')==manifest['source_commit']:
                if not (dest/'retention.json').exists():retain_cell(dest,OUT.name+'/'+key+'/read'+str(chunk)+'/'+cfg['case'])
                rows.append(old);continue
            if old.get('status')=='measured' and old.get('route_witness',{}).get('validated') and old.get('smoke_only')==a.smoke:
                invocation=json.loads((dest/'invocation.json').read_text())
                if invocation.get('source_commit')!=manifest['source_commit']:
                    raise RuntimeError('completed cell belongs to another source freeze: '+str(dest))
                # Resume retention without executing a valid measurement again.
                if not (dest/'retention.json').exists():
                    retain_cell(dest,OUT.name+'/'+key+'/read'+str(chunk)+'/'+cfg['case'])
                old['reused_unchanged_completed_cell']=True;rows.append(old);continue
            raise RuntimeError('preserve failed/incomplete cell and use a new attempt output: '+str(dest))
        remaining=deadline-time.monotonic()
        row=dict(model_key=key,read_chunk=chunk,case=cfg['case'],requested_flags=cfg,source_commit=manifest['source_commit'])
        if a.matrix=='routes' and key=='mojolearn-162m' and cfg.get('route')=='resident':
            row.update(status='unsupported',executed=False,reason='Saved architecture has no final norm; resident admission requires one. No model modification or manufactured norm.')
            write(previous,row);retain_cell(dest,OUT.name+'/'+key+'/read'+str(chunk)+'/'+cfg['case']);rows.append(row);continue
        if remaining<45:
            row.update(status='pending-budget',executed=False);rows.append(row);continue
        env.update(MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32=str(cfg['sampler']),
                   MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE=str(cfg['kv_reuse']),
                   MOJOLEARN_GENERATION_NATIVE_ENGINE=str(cfg['engine']),
                   MOJOLEARN_EXPERIMENT_GENERATION_ROUTE=cfg.get('route','auto'),
                   MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD=str(cfg.get('resident_forward',1)))
        cmd=[PY,__file__,'--worker','--key',key,'--chunk',str(chunk),'--arm',str(cfg['sampler']),
             '--kv-reuse',str(cfg['kv_reuse']),'--engine',str(cfg['engine']),'--case',cfg['case'],
             '--route',cfg.get('route','auto'),'--resident-forward',str(cfg.get('resident_forward',1))]
        if a.smoke:cmd+=['--smoke']
        write(dest/'invocation.json',dict(command=cmd,environment=evidence.runtime_environment(env),source_commit=manifest['source_commit']))
        with (dest/'execution.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=min(a.cell_seconds,remaining-30))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                rc=proc.returncode
        if previous.exists():row.update(json.loads(previous.read_text()))
        row['exit_code']=rc
        if row.get('status')=='started':row['status']='timeout-incomplete'
        events=parse_witness((dest/'execution.log').read_text(errors='replace'))
        witness={'events':events,'validated':False}
        if row.get('status')=='measured' and rc==0:
            try:
                witness.update(actual_routes=validate_witness(events,row['generated_tokens'],chunk),validated=True)
                witness['retained_kv_observed']=any(x['retained_kv'] for x in events)
                expected=cfg.get('expected_route')
                if cfg.get('route')=='auto' and key=='mojolearn-162m':expected='native_engine'
                if expected and witness['actual_routes']!=[expected]:
                    raise ValueError('executed route differs from explicit compatible selection')
            except ValueError as exc:row.update(status='failed-route-witness',route_error=str(exc))
        row['route_witness']=witness
        write(previous,row);rows.append(row);write(OUT/'summary.json',rows)
        try:retain_cell(dest,OUT.name+'/'+key+'/read'+str(chunk)+'/'+cfg['case'])
        except Exception as exc:
            write(OUT/'retention-failure.json',dict(cell=str(dest),error=repr(exc),scope='Stop next cell; completed measurements preserved; retry retention without rerun'))
            return 2
        print(json.dumps({k:row.get(k) for k in ['model_key','case','read_chunk','status','exit_code']}),flush=True)
    write(OUT/'summary.json',rows)
    complete=len(rows)==manifest['expected_cells'] and all(x.get('status')=='measured' and x.get('route_witness',{}).get('validated') for x in rows)
    write(OUT/'coverage.json',{'expected':manifest['expected_cells'],'recorded':len(rows),'measured':sum(x.get('status')=='measured' for x in rows),
                             'failed':sum(str(x.get('status','')).startswith('failed') for x in rows),'pending':sum(x.get('status')=='pending-budget' for x in rows),'unsupported':sum(x.get('status')=='unsupported' for x in rows),'complete':complete,
                             'supported_complete':len(rows)==manifest['expected_cells'] and all(x.get('status') in ('measured','unsupported') for x in rows)})
    return 0 if complete else 1

if __name__=='__main__':sys.exit(main())
