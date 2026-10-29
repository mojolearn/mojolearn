"""Frozen-build full saved CausalLM workloads; sampled policy A/B, no promotion."""
import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time, traceback

ROOT=pathlib.Path('/root/mojolearn')
OUT=pathlib.Path(os.environ.get('MOJOLEARN_SAMPLER_OUT','/root/measure-sampler-bc11389'))
PY='/root/neural-venv/bin/python'
sys.path.insert(0,str(ROOT/'tools'))
import bench_board_neural as bench
import bench_board_evidence as evidence
import cpu_quota

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)

def worker(a):
    import numpy as np
    from mojolearn.models import CausalLM
    dest=OUT/a.key/('read'+str(a.chunk))/('arm'+str(a.arm));dest.mkdir(parents=True,exist_ok=True)
    shape=bench.GEN_SHAPES['small' if a.smoke else 'full'];prompt_len=shape['prompt'];new_tokens=shape['new_tokens']
    rec={'model_key':a.key,'arm':a.arm,'read_chunk':a.chunk,'status':'started','smoke_only':a.smoke,
         'policy_flag':'MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32',
         'options':dict(greedy=False,seed=7,temperature=0.8,top_k=50,top_p=0.9),
         'saved_recipe':'tools/bench_board_neural.py GEN_SHAPES['+('small' if a.smoke else 'full')+'], LM_CHECKPOINTS',
         'batch':1,'prompt_tokens':prompt_len,'generated_tokens':new_tokens,
         'boundary':'stream creation plus all public reads, operation setup, KV loads/exports, synchronized host output conversion and concatenation; checkpoint load recorded separately',
         'warmup':'one excluded first invocation on freshly loaded model; cold operation time retained',
         'score':'one subsequent invocation with fresh stream and same retained model; no median/noise claim',
         'environment':evidence.runtime_environment()}
    write(dest/'result.json',rec)
    try:
        started=time.perf_counter()
        model=CausalLM.load(str(OUT/'checkpoints'/a.key),weight_format='float32',numeric_profile='fp32_v1',device='gpu')
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
            'interpretation':'source-gate inference only; actual native operation3/kernel trace unobserved in frozen build; policy schema alone does not prove residency',
            'source':'bc11389 bindings/causal_lm_public.mojo _generation_read and _read_sampling_resident'}
        rounds=[]
        for index in range(2):
            started=time.perf_counter()
            stream=model.stream(ids,new_tokens,**rec['options'])
            chunks=[]
            for offset in range(0,new_tokens,a.chunk):
                chunks.append(np.asarray(stream.read(min(a.chunk,new_tokens-offset)),dtype=np.int32).copy())
            output=np.concatenate(chunks,axis=1)
            elapsed=(time.perf_counter()-started)*1000
            if output.shape!=(1,new_tokens):raise ValueError('incomplete sampled output shape '+str(output.shape))
            if not ((output>=0)&(output<int(model.vocab_size))).all():raise ValueError('out of vocabulary token')
            np.save(dest/('warmup.npy' if index==0 else 'scored.npy'),output)
            info=stream.info()
            rounds.append(dict(role='excluded-cold-warmup' if index==0 else 'scored-repeated-model',ms=elapsed,
                               output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),stream_info=info))
            rec['rounds']=rounds;write(dest/'result.json',rec)
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
        try:
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
        rec['status']='measured';rec['qualification']='unqualified: single NVIDIA sample, resident route inferred not directly traced, AMD/Apple and real-text quality pending; no default promotion'
    except BaseException as exc:
        rec.update(status='failed',error=repr(exc));(dest/'error.log').write_text(traceback.format_exc())
    write(dest/'result.json',rec)
    print(json.dumps({k:rec.get(k) for k in ['model_key','arm','read_chunk','status','error','rounds','quality']}),flush=True)
    return 0 if rec['status']=='measured' else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--key');p.add_argument('--chunk',type=int);p.add_argument('--arm',type=int);p.add_argument('--budget',type=int,default=1200);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.worker:return worker(a)
    OUT.mkdir(exist_ok=True)
    env=dict(os.environ);env.pop('PYTHONPATH',None);cpu_quota.apply_cpu_quota(env)
    env.update(MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD='1',MOJOLEARN_GENERATION_NATIVE_ENGINE='0',MOJOLEARN_BENCH_INSTALLED='1')
    env.pop('MOJOLEARN_TRANSFORMER_LEGACY_SETUP',None)
    env['MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE']='0'
    deadline=time.monotonic()+a.budget;rows=[]
    write(OUT/'hardware.json',evidence.hardware_context('nvidia'))
    for key in bench.LM_CHECKPOINTS:
        for chunk in ([4,1] if a.smoke else [128,1]):
            for arm in [0,1]:
                dest=OUT/key/('read'+str(chunk))/('arm'+str(arm));dest.mkdir(parents=True,exist_ok=True)
                if (dest/'result.json').exists():
                    prior=json.loads((dest/'result.json').read_text())
                    if prior.get('status')=='measured' and prior.get('smoke_only',False)==a.smoke:
                        prior['reused_unchanged_completed_cell']=True;rows.append(prior);continue
                remaining=deadline-time.monotonic()
                if remaining<30:
                    rows.append(dict(key=key,chunk=chunk,arm=arm,status='pending-budget'));continue
                env['MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32']=str(arm)
                cmd=[PY,__file__,'--worker','--key',key,'--chunk',str(chunk),'--arm',str(arm)]
                if a.smoke:cmd+=['--smoke']
                write(dest/'invocation.json',dict(command=cmd,environment=evidence.runtime_environment(env)))
                with (dest/'execution.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:rc=proc.wait(timeout=min(250,remaining-15))
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        rc=proc.returncode
                row=dict(key=key,chunk=chunk,arm=arm,exit_code=rc,result=str(dest/'result.json'))
                if (dest/'result.json').exists():row.update(json.loads((dest/'result.json').read_text()))
                if row.get('status')=='started':row['status']='timeout-incomplete'
                rows.append(row);write(OUT/'summary.json',rows)
                print(json.dumps({k:row.get(k) for k in ['key','chunk','arm','status','exit_code','error']}),flush=True)
    write(OUT/'summary.json',rows)
    return 0

if __name__=='__main__':sys.exit(main())
