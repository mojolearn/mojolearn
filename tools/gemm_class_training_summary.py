#!/usr/bin/env python3
"""Late-regime timing and fail-closed witnesses for the 700-step comparison."""
import copy
import json
import statistics
import sys
from pathlib import Path

SHAPE = dict(batch=1, length=2048, d_model=768, n_heads=12, n_kv=12,
             head_dim=64, intermediate=2048, n_layers=12, vocab_size=50257)
HASH_KEYS = {'gradients', 'parameters', 'm', 'v', 'flags'}
GEMMS = {f'gemm.{kind}_{phase}' for kind in ('proj', 'gateup', 'down', 'head')
         for phase in ('fwd', 'dA', 'dB')}


def validate(d):
    assert d['shape'] == SHAPE and d['parameters'] == 162147840
    assert d['steps_completed'] == 700 and not d['limited']
    assert len(d['steady_step_seconds']) == 699
    assert d['final_witness']['step'] == 700
    assert set(d['final_witness']['sha256']) == HASH_KEYS
    assert len(d['step_witnesses']) == 700
    assert [x['step'] for x in d['step_witnesses']] == list(range(1, 701))
    assert all(set(x['sha256']) == {'loss'} for x in d['step_witnesses'])


def equal(a, b, label, verbose=True):
    validate(a); validate(b)
    assert a['corpus'] == b['corpus'], 'corpus mismatch'
    for x, y in zip(a['step_witnesses'], b['step_witnesses']):
        assert x == y, f'{label}: step witness differs at {x["step"]}'
        if verbose: print('MATCH', label, 'step', x['step'], x['sha256'])
    for key in sorted(HASH_KEYS):
        x, y = a['final_witness']['sha256'][key], b['final_witness']['sha256'][key]
        assert x == y and len(x) == 64, f'{label}: final {key} differs'
        if verbose: print('MATCH', label, 'step=700', key, x, y)


def summarize(root):
    runs = {name: json.loads((root/name/'result.json').read_text())
            for name in ('base-pure', 'class-pure', 'base-timers', 'class-timers')}
    base = runs['base-pure']
    # Deliberately corrupt EACH final hash and an interior loss; watch rejection
    # before trusting the actual equality check.
    for key in sorted(HASH_KEYS):
        bad = copy.deepcopy(base)
        bad['final_witness']['sha256'][key] = 'broken'
        try: equal(base, bad, 'sabotage-'+key, False)
        except AssertionError as exc: print('EXPECTED FAIL', str(exc))
        else: raise AssertionError('final-witness gate accepted sabotage')
    bad = copy.deepcopy(base)
    bad['step_witnesses'][399]['sha256']['loss'] = 'broken'
    try: equal(base, bad, 'sabotage-step-400', False)
    except AssertionError as exc: print('EXPECTED FAIL', str(exc))
    else: raise AssertionError('step gate accepted sabotage')
    for name, d in runs.items(): equal(base, d, name)
    summary = dict(corpus=base['corpus'], shape=SHAPE, steps=700,
                   reported_window='steps 501-700; component timing at steps 701-710', arms={})
    for arm in ('base', 'class'):
        pure, timed = runs[arm+'-pure'], runs[arm+'-timers']
        late_ms = 1000*statistics.median(pure['steady_step_seconds'][499:])
        assert timed['component_timing_steps_parsed'] == 10
        series = timed['component_timing_ms_per_step']
        assert GEMMS <= set(series), 'missing GEMM kind'
        assert all(len(series[k]) == 10 for k in GEMMS)
        gemm_ms = statistics.median([sum(series[k][i] for k in GEMMS) for i in range(10)])
        all_gemm = GEMMS | {'gemm.norm_dW'}
        assert all_gemm <= set(series)
        all_ms = statistics.median([sum(series[k][i] for k in all_gemm) for i in range(10)])
        env_ms = 1000*timed['component_timing_step_seconds']
        summary['arms'][arm] = dict(real_step_ms=late_ms, gemm_ms=all_ms,
            gemm_share_pct=100*all_ms/late_ms, twelve_gemm_ms=gemm_ms,
            gemm_tflops=1518.0/gemm_ms, instrumented_step_ms=env_ms,
            instrumentation_pct=100*(env_ms/late_ms-1),
            windows_ms={f'{lo}-{lo+99}': 1000*statistics.median(
                pure['steady_step_seconds'][lo-2:lo+98]) for lo in (101, 201, 301, 401, 501, 601)})
    b, c = [summary['arms'][a]['real_step_ms'] for a in ('base','class')]
    summary['step_reduction_pct'] = 100*(b-c)/b
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
