#!/usr/bin/env python3
"""Task-qualified multi-tensor optimizer and separate cancellation LayerNorm arm."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    cases={}
    if args.variant=='normalization':
        from mojolearn import LayerNorm
        from mojolearn import _mojolearn_x_sequence as binding
        identity=binding_check(binding,'x_sequence')
        for rows,d,scale,offset in ((131,17,1.,0.),(137,65,.01,1e5),(257,129,1e3,0.)):
            rng=np.random.default_rng(197);x=np.asarray(offset+scale*rng.normal(size=(rows,d)),'float32');dy=rng.normal(size=(rows,d)).astype('float32')
            model=LayerNorm(d,numeric_mode='fast')
            y,forward=consumed(lambda:model(x))
            def gradients():
                dx=model.backward(dy)
                return dx,model.weight_grad,model.bias_grad
            (dx,dw,db),backward=consumed(gradients)
            xx=x.astype(float);mu=xx.mean(axis=1,keepdims=True);variance=((xx-mu)**2).mean(axis=1,keepdims=True)
            ref=(xx-mu)/np.sqrt(variance+model.eps)
            error=float(np.max(np.abs(np.asarray(y)-ref))/max(np.max(np.abs(ref)),1e-9))
            gg=dy.astype(float);inv=1/np.sqrt(variance+model.eps)
            dxref=inv*(gg-gg.mean(axis=1,keepdims=True)-ref*(gg*ref).mean(axis=1,keepdims=True))
            dwref=(gg*ref).sum(axis=0);dbref=gg.sum(axis=0)
            dxerror=float(np.max(np.abs(np.asarray(dx)-dxref))/max(np.max(np.abs(dxref)),1e-9))
            affine_error=float(max(np.max(np.abs(np.asarray(dw)-dwref)),np.max(np.abs(np.asarray(db)-dbref)))/max(np.max(np.abs(dwref)),np.max(np.abs(dbref)),1e-9))
            cases[f'{rows}-{d}-{scale}']=dict(contract=dict(rows=rows,d=d,scale=scale,offset=offset,seed=197),
                metrics=dict(normalization_error=dict(value=error,rtol=.1,atol=1e-5),input_gradient_error=dict(value=dxerror,rtol=.1,atol=1e-5),affine_gradient_error=dict(value=affine_error,rtol=.1,atol=1e-5)),forward_ms=forward,backward_ms=backward)
    else:
        from mojolearn import SGD,linear_forward,linear_backward,cross_entropy
        from mojolearn import _mojolearn_training as binding
        identity=binding_check(binding,'training')
        rng=np.random.default_rng(197);hidden=rng.normal(size=(41,17)).astype('float32')
        weights=[rng.normal(0,.05,size=(classes,17)).astype('float32') for classes in (7,11)]
        targets=[np.argmax(hidden.astype(float)@rng.normal(size=(17,len(w))),axis=1).astype('int32') for w in weights]
        optimizer=SGD(weights,lr=.05,momentum=.9,resident=False)
        curve=[];times=[]
        for step in range(12):
            def train():
                gradients=[];losses=[]
                for w,y in zip(weights,targets):
                    logits=linear_forward(hidden,w,numeric_mode='fast')
                    loss,dz=cross_entropy(logits,y,return_grad=True,numeric_mode='fast')
                    dh,dw=linear_backward(dz,hidden,w,numeric_mode='fast');gradients.append(dw);losses.append(loss)
                optimizer.step(gradients)
                return tuple(losses)+tuple(weights)
            values,elapsed=consumed(train);curve.append(list(map(float,values[:2])));times.append(elapsed)
        final=[float(cross_entropy(linear_forward(hidden,w,numeric_mode='fast'),y,numeric_mode='fast')) for w,y in zip(weights,targets)]
        saved=[w.copy() for w in weights];bad=[np.zeros_like(w) for w in weights];bad[0][0,0]=np.nan
        try:optimizer.step(bad)
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('nonfinite optimizer gradient accepted')
        assert all(np.array_equal(a,b) for a,b in zip(saved,weights)), 'refused update changed parameters'
        cases['two-head-fixed-task']=dict(contract=dict(rows=41,d=17,classes=[7,11],steps=12,lr=.05,momentum=.9,seed=197),
            metrics={f'head{i}_loss':dict(value=loss,rtol=.001,atol=1e-5) for i,loss in enumerate(final)},learning_curve=curve,train_step_ms=times,refused_update=True)
    return dict(binding=identity,cases=cases)
if __name__=='__main__':capture_main(exercise)
