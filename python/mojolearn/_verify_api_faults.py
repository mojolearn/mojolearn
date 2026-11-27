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


_PROGRAM = FAULTS['program-product'].rsplit('.', 1)[0]
_PROGRAM_METHODS = ('run_if', 'while_loop', 'vjp_if', 'vjp_while', 'fit_if', 'save_control')


def fault_endpoints(name=None):
    """Public endpoints exercised by one control, or by the whole fault smoke."""
    if name is None:
        return tuple(endpoint for arm in FAULTS for endpoint in fault_endpoints(arm))
    if name == 'program-product':
        return (FAULTS[name], *(_PROGRAM + '.' + method for method in _PROGRAM_METHODS))
    return (FAULTS[name],)


def _program_observation_errors(endpoints, baseline, faulty):
    errors = []
    for endpoint in endpoints:
        before = baseline['endpoints'][endpoint]['output']
        after = faulty['endpoints'][endpoint]['output']
        method = endpoint.rsplit('.', 1)[-1]
        if method == 'run_if':
            # Only the selected product branch consumes the changed operation.
            if (not isinstance(before, list) or len(before) != 2
                    or before[0] == after[0] or before[1] != after[1]):
                errors.append(endpoint + ' did not isolate the fault to its product branch')
        if method in ('vjp_if', 'vjp_while', 'fit_if'):
            fields = ('output', 'parameter') if method == 'fit_if' else (
                'outputs', 'input_gradients', 'parameter_gradient')
            for field in fields:
                if not isinstance(before, dict) or field not in before or before[field] == after[field]:
                    errors.append(endpoint + ' did not detect the fault in ' + field)
    return errors


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


def check_faults(clean, *, retain=None):
    from ._verify_api_runner import capture, differences

    result = {}
    selected = set(clean['expected'])
    for name in FAULTS:
        endpoints = [endpoint for endpoint in fault_endpoints(name) if endpoint in selected]
        if not endpoints:
            continue
        rows = clean['endpoints']
        inapplicable = {
            e: rows[e]['reason'] for e in endpoints
            if rows.get(e, {}).get('status') == 'not applicable'
            and rows[e].get('kind') in ('device only', 'installation side')
        }
        if len(inapplicable) == len(endpoints):
            result[name] = dict(status='NOT_APPLICABLE', endpoint=endpoints[0],
                                reason=inapplicable[endpoints[0]], not_applicable=inapplicable)
            continue
        endpoints = [e for e in endpoints if e not in inapplicable]
        baseline = dict(expected=endpoints, endpoints={
            endpoint: clean['endpoints'].get(endpoint, {}) for endpoint in endpoints
        }, pending=[])
        missing = [endpoint for endpoint, row in baseline['endpoints'].items()
                   if row.get('status') != 'captured']
        if missing:
            result[name] = dict(status='FAILED', reason='clean endpoint did not execute',
                                endpoints=endpoints, missing=missing)
            continue

        # Both replays use the same side and native artifacts as their clean capture.
        options = dict(endpoints=endpoints, artifacts=clean.get('bindings'), side=clean.get('side'))
        with inject(name):
            faulty = capture(**options)
        restored = capture(**options)
        delta = differences(faulty, baseline)
        recovery = differences(restored, baseline)
        caught = set(delta) == {'output differs: ' + endpoint for endpoint in endpoints}
        observations = []
        if caught and name == 'program-product':
            observations = _program_observation_errors(endpoints, baseline, faulty)
        result[name] = dict(
            status='CAUGHT_AND_RESTORED' if caught and not observations and not recovery else 'FAILED',
            endpoint=endpoints[0], endpoints=endpoints, differences=delta,
            observation_errors=observations, restoration_errors=recovery, not_applicable=inapplicable,
            kind='public API state/configuration', native_kernel_fault=False,
        )
        if retain:
            retain(name + '-faulty', faulty)
            retain(name + '-restored', restored)
    return result
