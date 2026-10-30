"""Bounded external concurrency check; no product hooks or artificial delays.

Public-call intervals and native busy observations are evidence of attempted
concurrency, not proof that GPU kernels execute simultaneously. A native busy
refusal during the owner's read is direct admission evidence.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
import numpy as np


def run(args):
    from mojolearn.models import CausalLM
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    os.environ.update(MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32='1',
                      MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE='1',
                      MOJOLEARN_GENERATION_NATIVE_ENGINE='0',
                      MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD='1',
                      MOJOLEARN_EXPERIMENT_GENERATION_ROUTE=args.route,
                      MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE='0')
    model=CausalLM.load(args.checkpoint,weight_format='float32',numeric_profile='fp32_v1',device='gpu')
    fields=object.__getattribute__(model,'__dict__')
    if type(fields) is not dict:raise TypeError('exact model field dictionary required')
    prompts=[np.random.default_rng(91+i).integers(0,int(model.vocab_size),(1,16+i),dtype=np.int32) for i in range(2)]
    options=[dict(greedy=False,seed=7+4*i,temperature=.8,top_k=50,top_p=.9,eos_token_id=None,pad_token_id=None) for i in range(2)]
    count=16
    def stream(i):return model.stream(prompts[i],count,**options[i])
    def retain_state(label,state):
        arrays={}
        def encode(value,path):
            if isinstance(value,dict):return {str(k):encode(v,path+"/"+str(k)) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [encode(v,path+"/"+str(i)) for i,v in enumerate(value)]
            if isinstance(value,bytes):return {"bytes_hex":value.hex()}
            if value is None or isinstance(value,(str,int,float,bool)):return value
            data=np.asarray(value).copy()
            if data.dtype.hasobject:raise TypeError("checkpoint object array cannot be retained safely: "+path)
            key="array_"+str(len(arrays));arrays[key]=data
            return {"array":key,"shape":list(data.shape),"dtype":str(data.dtype),"sha256":hashlib.sha256(data.tobytes()).hexdigest()}
        metadata=encode(state,"state")
        np.savez(out/(label+"-state.npz"),**arrays)
        (out/(label+"-state.json")).write_text(json.dumps(metadata,indent=2))
    references=[];reference_info=[];reference_checksums=[]
    for i in range(2):
        serial=stream(i)
        references.append(np.asarray(serial.read(count),dtype=np.int32).copy())
        reference_info.append(serial.info());serial_state=serial.state_dict();reference_checksums.append(serial_state['checksum']);retain_state('serial-'+str(i),serial_state)
        np.save(out/f'prompt-{i}.npy',prompts[i]);np.save(out/f'serial-{i}.npy',references[-1])
    results=[]
    for kind in ('independent-streams','same-stream'):
        if fields.get('_resident_busy',False):raise RuntimeError('owner busy after completed serial reference')
        first=stream(0);second=stream(1) if kind=='independent-streams' else first
        entered=threading.Event();finished=threading.Event();records={}
        def read_first():
            entered.set();start=time.perf_counter_ns()
            try:
                value=first.read(count);end=time.perf_counter_ns();finished.set()
                output=np.asarray(value,dtype=np.int32).copy();np.save(out/(kind+'-first.npy'),output)
                records['first']=dict(status='returned',start_ns=start,end_ns=end,shape=list(output.shape),matches_serial=bool(np.array_equal(output,references[0])))
            except BaseException as exc:
                end=time.perf_counter_ns();finished.set();records['first']=dict(status='failed',start_ns=start,end_ns=end,error=repr(exc),traceback=traceback.format_exc())
        def read_second():
            if not entered.wait(5):raise RuntimeError('owner thread did not enter')
            until=time.monotonic()+3;busy=False
            while time.monotonic()<until and not finished.is_set():
                busy=bool(fields.get('_resident_busy',False))
                if busy:break
                time.sleep(.001)
            owner_finished_before_attempt=finished.is_set();start=time.perf_counter_ns()
            try:
                value=second.read(count if kind=='independent-streams' else 1);end=time.perf_counter_ns()
                output=np.asarray(value,dtype=np.int32).copy();np.save(out/(kind+'-second.npy'),output)
                records['second']=dict(status='returned',start_ns=start,end_ns=end,shape=list(output.shape),matches_serial=bool(np.array_equal(output,references[1])) if kind=='independent-streams' else output.size==0)
            except BaseException as exc:
                records['second']=dict(status='refused',start_ns=start,end_ns=time.perf_counter_ns(),error=repr(exc),traceback=traceback.format_exc())
            records['second'].update(native_model_busy_observed=busy,owner_finished_before_attempt=owner_finished_before_attempt)
        os.environ['MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE']='1'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(read_first),pool.submit(read_second)]
            for future in futures:future.result(timeout=60)
        os.environ['MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE']='0'
        a,b=records['first'],records['second']
        overlap=max(a['start_ns'],b['start_ns'])<min(a['end_ns'],b['end_ns'])
        refusal_during_owner=b['status']=='refused' and 'busy' in b.get('error','').lower() and b['end_ns']<a['end_ns']
        if not a.get('matches_serial',False):status='failed-owner-output'
        elif b['status']=='returned' and not b.get('matches_serial',False):status='failed-second-output'
        elif b['status']=='refused' and not refusal_during_owner:status='unqualified-refusal'
        elif kind=='same-stream':status='native-busy-admission-observed' if refusal_during_owner else 'overlap-unproven'
        else:status='public-call-overlap-outputs-pass' if overlap and b.get('native_model_busy_observed') and b['status']=='returned' else ('busy-admission-observed' if refusal_during_owner else 'overlap-unproven')
        # Fresh independent execution checks ownership cleanup; it does not
        # silently revive a stream invalidated by an earlier failed operation.
        recovery=np.asarray(stream(1).read(count),dtype=np.int32).copy();np.save(out/(kind+'-recovery.npy'),recovery)
        recovery_ok=bool(np.array_equal(recovery,references[1]))
        info=first.info();state=first.state_dict();retain_state(kind+"-owner",state)
        result=dict(kind=kind,status=status,records=records,public_call_intervals_overlap=overlap,
                    native_busy_refusal_during_owner=refusal_during_owner,owner_info_after=info,
                    owner_checkpoint_schema=state.get('schema'),owner_checkpoint_checksum=state.get('checksum'),
                    owner_checkpoint_matches_serial=state.get('checksum')==reference_checksums[0],
                    owner_info_matches_serial=info==reference_info[0],fresh_stream_recovery_matches=recovery_ok,
                    model_busy_after=bool(fields.get('_resident_busy',False)),
                    live_dictionary_same=object.__getattribute__(model,'__dict__') is fields,
                    fresh_dictionary_busy_after=bool(object.__getattribute__(model,'__dict__').get('_resident_busy',False)),
                    native_public_busy_after=bool(model._resident_busy),
                    limitation='No native instrumentation added; public-call overlap and busy ownership evidence do not prove simultaneous GPU kernel execution. Unobserved overlap is not a concurrency pass.')
        if not recovery_ok or result['model_busy_after'] or not result['owner_checkpoint_matches_serial'] or not result['owner_info_matches_serial']:result['status']='failed-ownership-recovery'
        results.append(result);(out/'result.json').write_text(json.dumps(dict(route=args.route,cases=results),indent=2,default=str))
    print(json.dumps(dict(route=args.route,statuses=[x['status'] for x in results])))
    return 1 if any(x['status'].startswith('failed') or x['status']=='unqualified-refusal' for x in results) else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--route',required=True,choices=('legacy','resident'));p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    try:raise SystemExit(run(a))
    except Exception:
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'harness-error.log').write_text(traceback.format_exc());raise
