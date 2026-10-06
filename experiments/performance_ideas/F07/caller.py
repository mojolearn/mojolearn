#!/usr/bin/env python3
"""FLASH and real GQA ratios on short/long causal tails; full block oracle."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import TransformerBlock
    from mojolearn import _mojolearn_transformer as binding
    from neural_fast_quality import _llama64
    cases={}
    for length,nkv in ((17,4),(129,1),(513,2)):
        dm,nh,hd=64,4,16;rng=np.random.default_rng(731)
        def weight(o,i):return rng.normal(0,.06,size=(o,i)).astype('float32')
        w={'input_layernorm.weight':np.ones(dm,'float32'),'post_attention_layernorm.weight':np.ones(dm,'float32'),
           'q_proj.weight':weight(dm,dm),'k_proj.weight':weight(nkv*hd,dm),'v_proj.weight':weight(nkv*hd,dm),
           'o_proj.weight':weight(dm,dm),'gate_proj.weight':weight(2*dm,dm),'up_proj.weight':weight(2*dm,dm),'down_proj.weight':weight(dm,2*dm)}
        x=rng.normal(size=(1,length,dm)).astype('float32')
        model=TransformerBlock(w,n_heads=nh,n_kv_heads=nkv,head_dim=hd)
        before=int(binding.transformer_flash_call_count(False));before_group=int(binding.transformer_flash_call_count(True))
        output,elapsed=consumed(lambda:model.forward(x,model.allocate_state(1,length)))
        reached=int(binding.transformer_flash_call_count(False))-before
        grouped=int(binding.transformer_flash_call_count(True))-before_group
        if args.arm=='B':
            assert reached>0,'FLASH not reached'
            if args.variant=='gqa' and nkv<nh:assert grouped>0,'fixture did not exercise shared K/V'
        expanded=dict(w)
        for key in ('k_proj.weight','v_proj.weight'):
            expanded[key]=np.repeat(w[key].reshape(nkv,hd,dm),nh//nkv,axis=0).reshape(dm,dm)
        ref=_llama64(x.astype(float),expanded,nh)
        error=float(np.max(np.abs(np.asarray(output,float)-ref))/max(np.max(np.abs(ref)),1e-9))
        cases[f'L{length}-KV{nkv}']=dict(contract=dict(length=length,nh=nh,nkv=nkv,seed=731),
             metrics=dict(full_block_error=dict(value=error,rtol=1,atol=1e-6)),forward_ms=elapsed,flash_calls=reached,grouped_calls=grouped)
    return dict(binding=binding_check(binding,'transformer'),cases=cases)
if __name__=='__main__':capture_main(exercise)
