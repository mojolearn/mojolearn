#!/usr/bin/env python3
"""Full-logit/fused CE vs bounded head in an actual fixed SGD training task."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import chunked_lm_head_loss,linear_forward,linear_backward,cross_entropy,SGD
    from mojolearn import _mojolearn_training as binding
    cases={}
    for vocabulary,logit_scale in ((1031,1),(4099,1),(1031,128)):
        rng=np.random.default_rng(619);rows,width=31,37
        hidden=rng.normal(size=(rows,width)).astype('float32')
        w=rng.normal(0,.1*logit_scale,size=(vocabulary,width)).astype('float32')
        teacher=rng.normal(size=(vocabulary,width))
        targets=np.argmax(hidden.astype(float)@teacher.T,axis=1).astype('int32')
        targets[-1]=vocabulary-1  # vocabulary-tile tail must be observed
        optimizer=SGD([w],lr=.03,resident=False)
        curve=[];times=[]
        for step in range(8):
            def train():
                if args.arm=='B':
                    loss,dh,dw=chunked_lm_head_loss(hidden,w,targets,return_grad=True,numeric_mode='fast')
                else:
                    logits=linear_forward(hidden,w,numeric_mode='fast')
                    loss,dz=cross_entropy(logits,targets,return_grad=True,numeric_mode='fast')
                    dh,dw=linear_backward(dz,hidden,w,numeric_mode='fast')
                optimizer.step([dw])
                return loss,dh,dw
            result,elapsed=consumed(train);curve.append(float(result[0]));times.append(elapsed)
        # Both arms still honor APIs that explicitly request every logit.
        logits,logits_ms=consumed(lambda:linear_forward(hidden,w,numeric_mode='fast'))
        assert np.asarray(logits).shape==(rows,vocabulary)
        final=float(cross_entropy(logits,targets,numeric_mode='fast'))
        # Independent test oracle checks stable normalization under tail targets.
        z=hidden.astype(float)@w.astype(float).T;z-=z.max(axis=1,keepdims=True)
        ref=float(np.mean(np.log(np.exp(z).sum(axis=1))-z[np.arange(rows),targets]))
        cases[f"{vocabulary}-scale{logit_scale}"]=dict(contract=dict(rows=rows,width=width,vocab=vocabulary,logit_scale=logit_scale,steps=8,lr=.03,seed=619),
             metrics=dict(final_loss=dict(value=final,rtol=1e-3,atol=1e-5),oracle_error=dict(value=abs(final-ref),rtol=1,atol=1e-5)),
             train_step_ms=times,learning_curve=curve,requested_logits_ms=logits_ms,
             nominal_logits_bytes=rows*vocabulary*4,candidate_workspace_policy='vocabulary tiles with required gradients')
    return dict(binding=binding_check(binding,'training'),cases=cases)
if __name__=='__main__':capture_main(exercise)
