#!/usr/bin/env python3
"""Bounded Apple baseline/candidate byte comparison; root executes after build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback


def run(output, baseline=None):
    import numpy as np
    from mojolearn import SmallMLPTrainer
    from mojolearn import _mlp_impl
    from mojolearn._verify_native_runtime import audit_native_call

    rng = np.random.default_rng(71)
    weights = [(rng.standard_normal(shape) * .03).astype('f4')
               for shape in ((16, 8), (16,), (3, 16), (3,))]
    x = rng.standard_normal((8, 8)).astype('f4')
    y = np.arange(8, dtype='i4') % 3
    values, audits, counts = {}, [], {}
    for fused in ('1', '0'):
        os.environ['MOJOLEARN_MLP_FUSED'] = fused
        model = SmallMLPTrainer(*weights, data_schedule={"batch_size": 8}, lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        for index, (name, args) in enumerate((('predict_logits', (x,)), ('train_step', (x, y)),
                                             ('train_step', (x, y)), ('predict_logits', (x,)))):
            call = getattr(model, name)
            if baseline:
                audit = audit_native_call(call, *args)
                audits.append(dict(route=fused, operation=name, report=audit.report()))
                result = audit.assert_native()
            else:
                result = call(*args)
            if isinstance(result, dict):
                for key in ('loss', 'step', 'logits'):
                    values[f'{fused}/{index}/{key}'] = np.asarray(result[key]).copy()
                for key, value in result['gradients'].items():
                    values[f'{fused}/{index}/gradient/{key}'] = np.asarray(value).copy()
            else:
                values[f'{fused}/{index}/logits'] = np.asarray(result).copy()
        if baseline:
            counts[fused] = _mlp_impl._NATIVE('native_provider_info', ())
            assert counts[fused][0] == 'mojolearn.training.small_mlp.v1'
            assert counts[fused][1][1 if fused == '1' else 2] > 0
    if baseline:
        with np.load(baseline) as before:
            assert set(before.files) == set(values)
            for key, value in values.items():
                assert before[key].dtype == value.dtype and before[key].shape == value.shape, key
                assert before[key].tobytes() == value.tobytes(), key
        # Poison the retained legacy computations after import-time setup;
        # candidate operations must reach the shared native table instead.
        calls = []
        def poison(*args):
            calls.append('legacy')
            raise AssertionError('legacy CPython execution reached')
        context = _mlp_impl._connect()
        for name in ('training_numeric_mode', 'mlp_train_step', 'neural_gemm', 'mlp_bias_activation',
                     'mlp_relu_backward', 'mlp_sum_rows', 'ce_loss', 'optimizer_step'):
            context['modes']['identical'][name] = poison
        _mlp_impl._NATIVE('initialize', (context,))
        for fused in ('1', '0'):
            os.environ['MOJOLEARN_MLP_FUSED'] = fused
            model = SmallMLPTrainer(*weights, data_schedule={"batch_size": 8}, lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0.)
            model.predict_logits(x)
            model.train_step(x, y)
        assert not calls
        class ForbiddenBuffer:
            def __buffer__(self, flags):
                calls.append('buffer')
                raise AssertionError('Python buffer hook reached')
        try:
            model.predict_logits(ForbiddenBuffer())
        except TypeError:
            pass
        else:
            raise AssertionError('custom protocol admission did not refuse')
        assert not calls
    np.savez(output.with_suffix('.npz'), **values)
    return dict(status='passed', cells=len(values), baseline=str(baseline) if baseline else None,
                arrays=str(output.with_suffix('.npz')), provider_counts=counts, runtime_audits=audits,
                sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in values.items()},
                scope='bounded identical SmallMLP fused/per-operation public calls on Apple; other trainer paths pending')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    try:
        report=run(args.output,args.baseline)
    except BaseException:
        report=dict(status='failed',traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],evidence=str(args.output))))
    return report['status']!='passed'


if __name__=='__main__':
    raise SystemExit(main())
