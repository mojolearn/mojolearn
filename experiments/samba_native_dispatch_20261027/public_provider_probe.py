#!/usr/bin/env python3
"""Authored local-Mac Samba dispatch probe; run only after integration build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback


# Test evidence only, retained even when a strict assertion fails.
RUNTIME_AUDITS = []


def qualify():
    RUNTIME_AUDITS.clear()
    import numpy as np
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn import SambaConfig, SambaStack
    from mojolearn import _native_neural_bridge as neural

    config = SambaConfig(19, 32, ['mamba3', 'attention'], n_heads=2,
                         n_kv_heads=1, intermediate=48, dropout=0.0)
    rng = np.random.default_rng(428)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype('f4') for name, shape in config.registry()}
    for name in weights:
        if 'norm' in name or name.endswith('.D'):
            weights[name].fill(1)
    models, states = [], []
    for switch in ('0', '1'):
        os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = switch
        model = SambaStack(config, weights=weights)
        models.append(model)
        states.append(model.allocate_state(1, 12))
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype='i4')
    cache = None
    linked = None
    for t in range(ids.shape[1]):
        outputs = []
        for arm, (model, state) in enumerate(zip(models, states)):
            os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = str(arm)
            if arm == 1 and t in (0, ids.shape[1] - 1):
                audit = audit_native_call(model.step, ids[:, t], state)
                RUNTIME_AUDITS.append(dict(operation='SambaStack.step', step=t, report=audit.report()))
                value = audit.assert_native()
            else:
                value = model.step(ids[:, t], state)
            outputs.append(np.asarray(value).copy())
        assert np.array_equal(outputs[0].view('u4'), outputs[1].view('u4')), t
        for a, b in zip(states[0].layers, states[1].layers):
            for name in ('cached_tokens', 'buffered_tokens', 'pending'):
                if hasattr(a, name):
                    assert getattr(a, name) == getattr(b, name), (t, name)
        fields = vars(models[1]._blocks[1])
        now = fields['_native_session_provider_cache']
        new_link = fields['_native_forward_owner']
        if cache is not None:
            assert now is cache, 'provider reconnected for unchanged block admission'
            assert new_link is linked, 'session handle reconnected for unchanged owner'
        cache, linked = now, new_link
    assert vars(models[1]._blocks[0])['_native_mamba_state'] is not None
    assert vars(models[1])['_native_primitives'] is not None
    module = neural.CONTEXT['bindings']['gpu:identical']['transformer']
    assert module.transformer_vendor() == 'metal'
    saved = module.transformer_native_api_v1
    # A changed compiled factory on the SAME module must be re-admitted.
    # The substitute returns a vendor string, never a valid versioned capsule.
    module.transformer_native_api_v1 = module.transformer_vendor
    try:
        try:
            models[1]._blocks[1].allocate_state(1, 12)
        except Exception as exc:
            assert 'capsule' in str(exc), str(exc)
        else:
            raise AssertionError('changed factory reused stale cached table')
    finally:
        module.transformer_native_api_v1 = saved
    # Existing strict native-entry admission must also survive cache hits.
    called = []
    def forbidden():
        called.append(True)
        raise AssertionError('Python replacement must not execute')
    module.transformer_native_api_v1 = forbidden
    try:
        try:
            models[1]._blocks[1].allocate_state(1, 12)
        except Exception as exc:
            assert 'compiled native extension entry' in str(exc), str(exc)
        else:
            raise AssertionError('Python factory replacement was not refused')
    finally:
        module.transformer_native_api_v1 = saved
    assert not called
    models[1]._blocks[1].allocate_state(1, 12)
    assert vars(models[1]._blocks[1])['_native_session_provider_cache'] is cache
    return dict(status='passed', runtime_audits=RUNTIME_AUDITS, checks=['mixed Mamba3/attention six-step bitwise outputs',
        'carried token counters', 'stable provider and native-session link',
        'same-module compiled factory replacement', 'Python factory refusal without invocation', 'recovery'],
        logits_sha256=hashlib.sha256(outputs[1].tobytes()).hexdigest(),
        pending=['CPU fallback', 'training fallback', 'concurrency', 'other devices', 'full-workload timing'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = qualify()
    except BaseException:
        report = dict(status='failed', traceback=traceback.format_exc(), runtime_audits=RUNTIME_AUDITS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], evidence=str(args.output))))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
