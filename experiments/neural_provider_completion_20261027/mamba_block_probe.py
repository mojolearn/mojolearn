#!/usr/bin/env python3
"""Bounded Mac public A/B probe, authored for root's built candidate only."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def qualify():
    os.environ['MOJOLEARN_GENERATION_DEVICE'] = 'gpu'
    import numpy as np
    from mojolearn import Mamba1Block, Mamba2Block, Mamba3Block
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn.tests.test_neural_block_native import block_inputs
    from mojolearn.tests.test_mamba23_backward_surface import host_fixture
    results = []
    audits = []
    with tempfile.TemporaryDirectory(prefix='ml-mamba-block-provider-') as folder:
        for family, cls in ((1, Mamba1Block), (2, Mamba2Block), (3, Mamba3Block)):
            if family == 1:
                _, weights, options, x = block_inputs(Path(folder), 'mamba')
            else:
                weights, x = host_fixture(family)
                options = {}
            grad = np.random.default_rng(735 + family).normal(size=x.shape).astype('f4')
            for legacy in (('0', '1') if family == 3 else ('0',)):
                os.environ['MOJOLEARN_MAMBA3_LEGACY_SETUP'] = legacy
                blocks = [cls(weights, numeric_mode='identical', **options) for _ in range(2)]
                for repeat in range(2):
                    values = []
                    for arm, block in enumerate(blocks):
                        os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = str(arm)
                        if arm:
                            audit = audit_native_call(block.backward, x, grad)
                            audits.append(audit.report())
                            values.append(audit.assert_native())
                        else:
                            values.append(block.backward(x, grad))
                    assert set(values[0]) == set(values[1])
                    for name in values[0]:
                        a, b = np.asarray(values[0][name]), np.asarray(values[1][name])
                        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), (family, legacy, repeat, name)
                diagnostic = next(module.neural_native_dispatch_info for module in tuple(sys.modules.values())
                                  if module is not None and hasattr(module, 'neural_native_dispatch_info'))
                provider = object.__getattribute__(blocks[1], '__dict__')['_native_mamba_block_provider_cache']
                name, counts = diagnostic(provider)
                assert name == 'mojolearn.mamba.block.v1'
                operation = 7 if family == 3 and legacy == '0' else family + 1
                assert counts[operation] >= 2
                assert counts[0] == counts[1]
                # An invalid gradient must not consume or poison the session.
                bad = grad.copy(); bad.flat[0] = np.nan
                for arm, block in enumerate(blocks):
                    os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = str(arm)
                    try:
                        block.backward(x, bad)
                    except (TypeError, ValueError):
                        pass
                    else:
                        raise AssertionError('nonfinite gradient accepted')
                    block.backward(x, grad)
                results.append(dict(family=family, legacy_setup=legacy, counts=counts))
    return dict(status='passed', variants=results, runtime_audits=audits,
                pending=['Mamba3 owned-weight toggle', 'Samba prefill forward through this table',
                         'native lock/mixed-layer engine', 'tape ABI and CPU route', 'other devices', 'performance'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        report = qualify()
    except BaseException:
        report = dict(status='failed', traceback=traceback.format_exc())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], evidence=str(a.output))))
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
