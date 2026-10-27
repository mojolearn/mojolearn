#!/usr/bin/env python3
"""Source-authored local public probe; root runs only after frozen compilation."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def qualify():
    import numpy as np
    from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn.models import CausalLM
    from mojolearn.models import causal_lm

    os.environ['MOJOLEARN_GENERATION_NATIVE_ENGINE'] = '0'
    os.environ['MOJOLEARN_TRANSFORMER_SESSION_FRESH'] = '1'
    results = []
    with tempfile.TemporaryDirectory(prefix='ml-transformer-block-') as folder:
        cfg, tensors = family_fixture('llama', False)
        _write_checkpoint(folder, cfg, tensors)
        os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = '1'
        model = CausalLM.load(folder, device='gpu', numeric_profile='fp32_v1')
        assert model._native_vendor == 'metal'
        old = model.blocks[0]
        weights = dict(zip(old._W_NAMES, old._w))
        cls = causal_lm._NATIVE_CONTEXT['block_classes']['gpu']['transformer']
        rng = np.random.default_rng(17)
        x = rng.normal(size=(2, 4, model.d_model)).astype('f4') * np.float32(0.05)
        grad = rng.normal(size=x.shape).astype('f4') * np.float32(0.05)
        for legacy in ('0', '1'):
            os.environ['MOJOLEARN_TRANSFORMER_LEGACY_SETUP'] = legacy
            blocks = [cls(weights, n_heads=model.plan.n_heads, n_kv_heads=model.plan.n_kv_heads,
                          numeric_mode='identical', numeric_profile='fp32_v1') for _ in range(2)]
            for repeat in range(3):
                outputs, gradients = [], []
                for arm, block in enumerate(blocks):
                    os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = str(arm)
                    audit = audit_native_call(block.forward, x)
                    outputs.append(np.asarray(audit.assert_native()).copy())
                    audit = audit_native_call(block.backward, x, grad)
                    gradients.append(audit.assert_native())
                assert outputs[0].tobytes() == outputs[1].tobytes()
                assert sorted(gradients[0]) == sorted(gradients[1])
                for name in gradients[0]:
                    assert np.asarray(gradients[0][name]).tobytes() == np.asarray(gradients[1][name]).tobytes(), (legacy, repeat, name)
            diagnostic = next(module.neural_native_dispatch_info for module in tuple(sys.modules.values())
                              if module is not None and hasattr(module, 'neural_native_dispatch_info'))
            provider = object.__getattribute__(blocks[1], '__dict__')['_native_block_provider_cache']
            name, counts = diagnostic(provider)
            assert name == 'mojolearn.transformer.block.v1'
            fresh, backward = (2, 3) if legacy == '0' else (4, 5)
            assert counts[fresh] >= 3 and counts[backward] >= 3
            if legacy == '0':
                assert counts[0] == counts[1] and counts[0] >= 6
            else:
                assert counts[0] == counts[1] == 0
            # Public invalid-input refusal remains; valid work follows it.
            for arm, block in enumerate(blocks):
                os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = str(arm)
                try:
                    block.backward(x, np.zeros((1,), dtype='f4'))
                except Exception:
                    pass
                else:
                    raise AssertionError('invalid gradient accepted')
                block.backward(x, grad)
            results.append(dict(legacy_setup=legacy, calls=counts, status='passed'))
    return dict(status='passed', variants=results,
                pending=['closed/busy concurrent owner controls', 'extended-option forward fixtures',
                         'tape operations', 'Samba whole-operation execution', 'other targets', 'performance'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        report = qualify()
    except BaseException:
        report = dict(status='failed', traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], evidence=str(args.output))))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
