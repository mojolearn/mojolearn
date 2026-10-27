"""Test-only state/configuration faults through real public native operations.

No recorded digest is edited. These controls change parameters supplied to the
public API and consume its actual outputs. They establish checker sensitivity,
not a claim that every kernel has been source-mutated.
"""
from contextlib import contextmanager

FAULTS={
 'estimator-regularization': 'mojolearn.ElasticNet.score',
 'metadata-parameter':'mojolearn.Ridge.get_params',
 'optimizer-rate':'mojolearn.Adam.state_dict',
 'block-weight':'mojolearn.TransformerBlock.__call__',
 'trainer-checkpoint':'mojolearn.LanguageModelTrainer.evaluate',
 'random-stream':'mojolearn.training.Generator.uniform',
 'program-product':'mojolearn.importers._program.ImportedProgram.run',
}


@contextmanager
def inject(name):
    if name=='program-product':
        from ._verify_light_faults import inject as tape_inject
        with tape_inject(name):yield
        return
    from . import _verify_api_metadata as metadata,_verify_api_estimators as estimators
    from . import _verify_api_training as training,_verify_api_optimizers as optimizers,_verify_api_utilities as utilities
    if name in ('estimator-regularization','metadata-parameter'):
        module=estimators if name=='estimator-regularization' else metadata
        key='factory';original=getattr(module,key)
        def broken(public,**kwargs):
            if public=='mojolearn.Ridge':kwargs['alpha']=4.
            if public=='mojolearn.ElasticNet':kwargs['alpha']=.0001
            return original(public,**kwargs)
    elif name=='block-weight':
        module=metadata;key='block_weights';original=module.block_weights
        def broken(public):
            result=original(public);result['o_proj.weight']*=2
            return result
    elif name=='trainer-checkpoint':
        module=training;key='tiny_lm';original=module.tiny_lm
        def broken():
            model,shape,ids=original();state=model.state_dict()
            import numpy as np
            state['parameters']=np.asarray(state['parameters']).copy();state['parameters'][0]+=.25
            model.load_state_dict(state)
            return model,shape,ids
    else:
        module=optimizers if name=='optimizer-rate' else utilities
        key='resolve';original=module.resolve
        def broken(public):
            cls=original(public)
            def construct(*args,**kwargs):
                if name=='optimizer-rate':kwargs['lr']=.25
                else:args=(19,)+args[1:]
                return cls(*args,**kwargs)
            return construct
    setattr(module,key,broken)
    try:yield
    finally:setattr(module,key,original)


def check_faults(clean,*,retain=None):
    from ._verify_api_runner import capture,differences
    result={}
    for name,endpoint in FAULTS.items():
        if endpoint not in clean['expected']:continue
        baseline=dict(expected=[endpoint],endpoints={endpoint:clean['endpoints'].get(endpoint,{})},pending=[])
        if baseline['endpoints'][endpoint].get('status')!='captured':
            result[name]={'status':'FAILED','reason':'clean endpoint did not execute'};continue
        # The clean capture has already loaded and hashed every native owner
        # used by these in-process controls. Reuse that inventory; hashing
        # gigabytes of unchanged libraries for each tiny fault is not a test.
        artifacts=clean.get('bindings')
        with inject(name):faulty=capture(endpoints=[endpoint],artifacts=artifacts)
        restored=capture(endpoints=[endpoint],artifacts=artifacts)
        delta=differences(faulty,baseline);recovery=differences(restored,baseline)
        caught=bool(delta) and all(x.startswith('output differs:') for x in delta) and not recovery
        result[name]=dict(status='CAUGHT_AND_RESTORED' if caught else 'FAILED',endpoint=endpoint,
            differences=delta,restoration_errors=recovery,kind='public API state/configuration',native_kernel_fault=False)
        if retain:retain(name+'-faulty',faulty);retain(name+'-restored',restored)
    return result
