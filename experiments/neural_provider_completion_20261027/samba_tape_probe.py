#!/usr/bin/env python3
"""Bounded public Samba train-step A/B; requires existing ACT_RETAIN=3 build.

Authored only. This does not claim the whole mixed-layer outer loop is native.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def qualify():
    os.environ['MOJOLEARN_GENERATION_DEVICE'] = 'gpu'
    # The fixture's supported dropout setting selects the existing nonresident
    # route. No performance timing or default promotion.
    import numpy as np
    import mojolearn as ml
    ids = (np.arange(2 * 9, dtype=np.int32).reshape(2, 9) * 37) % 256
    results = []
    evidence = []
    for arm in (0, 1):
        os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = str(arm)
        cfg = ml.SambaConfig(vocab=256, d_model=32, layers=('mamba3', 'attention'),
                             n_heads=2, intermediate=64, tie_embeddings=False, dropout=0.1)
        model = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, max_norm=1.0)
        step = model.train_step(ids[:, :-1], ids[:, 1:])
        params = model.parameters()
        results.append(np.float64(step['loss']).tobytes() + b''.join(
            np.asarray(params[k]).tobytes() for k in sorted(params)))
        if arm:
            diagnostic = next(module.neural_native_dispatch_info for module in tuple(sys.modules.values())
                              if module is not None and hasattr(module, 'neural_native_dispatch_info'))
            for block in model._blocks:
                fields = object.__getattribute__(block, '__dict__')
                transformer = fields['_native_kind'] == 'transformer'
                provider = fields['_native_block_provider_cache' if transformer else '_native_mamba_block_provider_cache']
                name, counts = diagnostic(provider)
                base = 6 if transformer else 10
                assert counts[base + 2] > 0 and counts[base + 3] > 0, 'immutable tape route not reached (ACT_RETAIN=3 required)'
                assert counts[base] == counts[base + 1], 'tape reservation leaked'
                assert counts[base + 4] > 0
                evidence.append(dict(name=name, counts=counts))
    assert results[0] == results[1], 'Samba public loss/parameter bytes differ'
    return dict(status='passed', providers=evidence, pending=[
        'tape close/backward race and nonfinite-dy consumption public probes',
        'whole Samba native orchestration', 'other devices', 'performance'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        result = qualify()
    except BaseException:
        result = dict(status='failed', traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], evidence=str(args.output))))
    return result['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
