#!/usr/bin/env python3
"""Individual Mamba fusion arms at actual full forward callers and references."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import ROOT,capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import Mamba1Block,Mamba2Block,Mamba3Block
    from mojolearn import _mojolearn_mamba as binding
    sys.path.insert(0,str(ROOT/'python/mojolearn/tests'))
    import test_mamba_surface as fixture
    cases={}
    families=[('mamba2',Mamba2Block,fixture.m2_weights,'residual.out')]
    if args.variant in ('mamba1-input','chunk-scan'):
        families=[('mamba1',Mamba1Block,fixture.m1_weights,'block.out')]
    elif args.variant=='mamba3-elementwise':
        families=[('mamba3',Mamba3Block,fixture.m3_weights_corpus,'residual.out')]
    for family,cls,weights,stage in families:
        directory=ROOT/'mamba/corpus'/family if family!='mamba1' else ROOT/'mamba/corpus'
        selected=[p for p in sorted(directory.iterdir()) if (p/'manifest.json').is_file() and 'init_states' not in p.name]
        count=0
        for path in selected:
            manifest=json.loads((path/'manifest.json').read_text())
            reference=manifest.get('stages',{}).get(stage)
            if not reference:continue
            b,length,dm=manifest['B'],manifest['L'],manifest['d_model']
            if length not in (1,4,8,64,256,257):continue  # test fixture coverage, never product dispatch
            x=fixture.f32(str(path/'x.f32'),(b,length,dm))
            opts={'dt_limit':tuple(float(v['value']) for v in manifest['dt_limit'])} if 'dt_limit' in manifest else {}
            model=cls(weights(str(path),dm),**opts)
            output,elapsed=consumed(lambda:model.forward(x,model.allocate_state(b)))
            ref=np.fromfile(path/reference['ref64'],dtype='<f8')
            error=float(np.max(np.abs(np.asarray(output,float).reshape(-1)-ref))/max(np.max(np.abs(ref)),1e-9))
            cases[path.name]=dict(contract=dict(family=family,shape=[b,length,dm],reference=reference['ref64']),
                metrics=dict(full_forward_error=dict(value=error,rtol=.1,atol=1e-6)),forward_ms=elapsed)
            if length>1:
                state=model.allocate_state(b);split=max(1,length-4)
                prefix,prefix_ms=consumed(lambda:model.forward(x[:,:split,:],state))
                tail=[];decode_times=[]
                for token in range(split,length):
                    value,timing=consumed(lambda token=token:model.step(x[:,token:token+1,:],state))
                    tail.append(np.asarray(value));decode_times.append(timing)
                carried=np.concatenate([np.asarray(prefix)]+tail,axis=1)
                drift=float(np.max(np.abs(carried.astype(float)-np.asarray(output,float)))/max(np.max(np.abs(ref)),1e-9))
                cases[path.name]['metrics']['continuation_drift']=dict(value=drift,rtol=.1,atol=2e-5)
                cases[path.name]['prefix_ms']=prefix_ms;cases[path.name]['decode_ms']=decode_times
            if length==4:
                cases[path.name+'-training']=training(cls,weights(str(path),dm),opts,x)

            count+=1
        assert count>=2,'missing meaningful independent Mamba corpus coverage'
    return dict(binding=binding_check(binding,'mamba'),cases=cases)
def training(cls,weights,opts,x):
    import numpy as np
    from mojolearn import SGD
    parameters={name:np.asarray(value).copy() for name,value in weights.items()}
    model=cls(parameters,**opts)
    optimizer=SGD(list(parameters.values()),lr=.001,resident=False)
    losses=[];times=[];directional_errors=[]
    for step in range(3):
        output,forward_ms=consumed(lambda:model.forward(x))
        y=np.asarray(output,float);loss=float(np.mean(y*y));losses.append(loss)
        cotangent=np.asarray(2*y/y.size,'float32')
        import time
        start=time.perf_counter_ns();gradients=model.backward(x,cotangent)
        for value in gradients.values():
            array=np.asarray(value);assert np.isfinite(array).all();array.tobytes()
        backward_ms=(time.perf_counter_ns()-start)/1e6
        if step==0:
            # Independent central directional derivative of the requested
            # zero-state public forward objective, using native forward arms.
            direction=np.random.default_rng(993).normal(size=x.shape)
            direction/=np.linalg.norm(direction);h=.003
            plus,_=consumed(lambda:model.forward(np.asarray(x+h*direction,'float32')))
            minus,_=consumed(lambda:model.forward(np.asarray(x-h*direction,'float32')))
            reference=(float(np.mean(np.asarray(plus,float)**2))-float(np.mean(np.asarray(minus,float)**2)))/(2*h)
            vjp=float(np.sum(np.asarray(gradients['x'],float)*direction))
            directional_errors.append(abs(vjp-reference))
        start=time.perf_counter_ns();optimizer.step([gradients[name] for name in parameters])
        for value in parameters.values():np.asarray(value).tobytes()
        update_ms=(time.perf_counter_ns()-start)/1e6
        times.append(dict(forward_ms=forward_ms,backward_ms=backward_ms,update_ms=update_ms))
    final,_=consumed(lambda:model.forward(x));final_loss=float(np.mean(np.asarray(final,float)**2))
    return dict(contract=dict(shape=list(x.shape),steps=3,lr=.001,objective='zero-state mean-square output',seed=993),
        metrics=dict(final_training_loss=dict(value=final_loss,rtol=.001,atol=1e-5),vjp_error=dict(value=max(directional_errors),rtol=.1,atol=1e-3)),
        loss_curve=losses,steps=times)
if __name__=='__main__':capture_main(exercise)
