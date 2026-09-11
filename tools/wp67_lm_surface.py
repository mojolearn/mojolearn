#!/usr/bin/env python3
"""Capture each trainer lifetime in its own process, including full optimizer state."""
import os
import numpy as np
from mojolearn import LanguageModelConfig, LanguageModelTrainer

shape=LanguageModelConfig(batch=1,length=5,d_model=16,n_heads=2,n_kv=1,
                          head_dim=8,intermediate=24,n_layers=3,vocab_size=257)
rng=np.random.default_rng(19)
weights=(rng.standard_normal(shape.n_total)*.02).astype(np.float32)
ids=rng.integers(0,shape.vocab_size,(shape.batch,shape.length+1),dtype=np.int32)
resident=os.environ.get('WP67_LM_RESIDENT','0')=='1'
with_trainer=LanguageModelTrainer(weights,shape=shape,resident=resident,data_schedule={'dataset':'wp67-gate'})
# DEVIATION 2514: under step_result='lean' the gradient stays on the device
# and is fetched with export_gradients(); under 'full' the result carries it.
lean=with_trainer.run_metadata()['step_result']=='lean'
try:
    for step in range(int(os.environ.get("WP67_LM_STEPS", "1")) if os.environ.get("WP67_LM_ACTION", "train")!="eval" else 0):
        print('training',resident,step,flush=True)
        result=with_trainer.train_step(ids)
        if lean:
            result=dict(result,**with_trainer.export_gradients())
        assert result['step']==step+1 and np.isfinite(result['loss'])
        for key in ('parameters','m','v','flags'):
            value=np.asarray(with_trainer.state_dict()[key])
            assert np.isfinite(value).all()
        assert np.isfinite(np.asarray(result['flat_gradients'])).all()
    if os.environ.get('WP67_LM_ACTION') in ('eval','train_eval'):
        print('evaluating',resident,flush=True)
        before={key:value.tobytes() for key,value in with_trainer.state_dict().items() if hasattr(value,'tobytes')}
        assert np.isfinite(with_trainer.evaluate(ids))
        assert before=={key:value.tobytes() for key,value in with_trainer.state_dict().items() if hasattr(value,'tobytes')}
finally:
    with_trainer.close()
print('PASS',resident,os.environ.get('WP67_LM_ACTION','train'), 'generalized call',flush=True)
