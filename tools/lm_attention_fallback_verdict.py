#!/usr/bin/env python3
"""DEVIATION 3110's verdict: name the trigger, then judge the latch.

READ `trigger` FIRST. Everything after it is conditional on the `before` arm
having actually refused: a run with zero refusals in 700 steps does not
exhibit the thing the latch removes, and then the latch's timing is a
measurement of nothing. That case is reported as INCONCLUSIVE, not as a pass.

Every check here is run against a deliberately corrupted copy of its own
input first and must reject it BY NAME. A check that cannot fail is not a
check, and a `KeyError` or a missing file reads exactly like a clean pass.
"""
import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

NAMES = {-1: 'NOT_ATTEMPTED', 0: 'FUSED_RAN', 1: 'FUSED_REFUSED_REGIME',
         2: 'FUSED_CORNER', 3: 'FUSED_SKIPPED_STICKY'}
STICKY = 3


# ---------------------------------------------------------------------------
# Coverage: the witness read every step, every layer, both directions.
# ---------------------------------------------------------------------------
def validate(result, minimum):
    rows = result['steps']
    assert len(rows) >= minimum, 'step coverage'
    n = result['shape'][7]
    for step, row in enumerate(rows):
        assert row['step'] == step, 'step sequence'
        a = row['attention']
        for key in ('forward_status', 'backward_status'):
            values = a[key]
            assert len(values) == n, 'layer coverage'
            assert all(v in NAMES for v in values), 'unknown status'
            assert a[key + '_counts'] == {name: values.count(code)
                                          for code, name in NAMES.items()}, 'status counts'
        assert len(a['attn_materialized']) == n, 'materialization coverage'
    return rows


def watch_coverage_fail(result, minimum):
    for name, mutate in (
        ('step coverage', lambda x: x['steps'].__delitem__(slice(minimum - 1, None))),
        ('layer coverage', lambda x: x['steps'][0]['attention']['forward_status'].pop()),
        ('unknown status', lambda x: x['steps'][0]['attention']['forward_status'].__setitem__(0, 99)),
        ('status counts', lambda x: x['steps'][0]['attention']['forward_status_counts'].__setitem__('FUSED_RAN', 999)),
        ('materialization coverage', lambda x: x['steps'][0]['attention']['attn_materialized'].pop()),
        ('step sequence', lambda x: x['steps'][0].__setitem__('step', 9)),
    ):
        broken = copy.deepcopy(result)
        mutate(broken)
        try:
            validate(broken, minimum)
        except AssertionError as e:
            assert str(e) == name, (name, str(e))
            print('EXPECTED FAIL  coverage:', name)
        else:
            raise AssertionError('BLIND: coverage ' + name)


# ---------------------------------------------------------------------------
# Bit equality between two arms.
# ---------------------------------------------------------------------------
HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


def compare(left, right):
    """Every step's loss bits, and every witnessed step's six hashes.

    The loss is recorded on EVERY step, so a single moved attention bit
    surfaces within a step or two and stays surfaced. The six hashes are
    recorded only on the witnessed anchors because an export at this shape
    costs several times the step."""
    a, b = left['steps'], right['steps']
    assert len(a) == len(b), 'step count differs'
    loss_differ = [x['step'] for x, y in zip(a, b)
                   if json.dumps(x['loss']) != json.dumps(y['loss'])]
    hash_differ = []
    anchors = 0
    for x, y in zip(a, b):
        if 'parameters' not in x or 'parameters' not in y:
            continue
        anchors += 1
        for key in HASHES:
            if x[key] != y[key]:
                hash_differ.append((x['step'], key))
    assert anchors > 0, 'no witnessed anchors'
    return dict(steps=len(a), anchors=anchors,
                loss_differ=loss_differ, hash_differ=hash_differ,
                equal=not loss_differ and not hash_differ)


def watch_compare_fail(left, right):
    for name, mutate in (
        ('loss', lambda x: x['steps'][3].__setitem__('loss', x['steps'][3]['loss'] + 1e-9)),
        ('parameters', lambda x: next(r for r in x['steps'] if 'parameters' in r)
            .__setitem__('parameters', 'deadbeef')),
    ):
        broken = copy.deepcopy(right)
        mutate(broken)
        out = compare(left, broken)
        assert not out['equal'], 'BLIND: compare ' + name
        print('EXPECTED FAIL  compare:', name,
              'loss_differ=%d hash_differ=%d' % (len(out['loss_differ']), len(out['hash_differ'])))
    broken = copy.deepcopy(right)
    broken['steps'] = broken['steps'][:-1]
    try:
        compare(left, broken)
    except AssertionError as e:
        assert str(e) == 'step count differs', str(e)
        print('EXPECTED FAIL  compare: step count differs')
    else:
        raise AssertionError('BLIND: compare step count')


# ---------------------------------------------------------------------------
def trigger_table(rows, layers, label):
    out = {}
    for key in ('forward_status', 'backward_status'):
        totals = Counter(v for row in rows for v in row['attention'][key])
        per_layer = []
        for layer in range(layers):
            seq = [row['attention'][key][layer] for row in rows]
            first = next((rows[i]['step'] for i, v in enumerate(seq) if v > 0), None)
            per_layer.append(dict(layer=layer, first_refusal=first,
                                  counts={NAMES[c]: n for c, n in sorted(Counter(seq).items())}))
        out[key] = dict(totals={NAMES[c]: n for c, n in sorted(totals.items())},
                        per_layer=per_layer)
        print('%s %s totals %s' % (label, key, out[key]['totals']))
        for row in per_layer:
            print('  %s %s layer %2d first_refusal=%s %s'
                  % (label, key, row['layer'], row['first_refusal'], row['counts']))
    return out


def timing(rows, label):
    # Step 0 is session setup (15+ s) and is excluded from every window. The
    # other lane's `slowed` flag included it and therefore could not detect
    # the thing it was named for.
    body = rows[1:]
    head, tail = body[:50], body[-50:]
    mb = [r['device_used_mb'] for r in rows if r.get('device_used_mb') is not None]
    out = dict(steps=len(rows),
               head_median_seconds=statistics.median(r['seconds'] for r in head),
               tail_median_seconds=statistics.median(r['seconds'] for r in tail),
               device_mb_first=mb[0] if mb else None,
               device_mb_max=max(mb) if mb else None,
               eager_bytes_first=rows[0]['attention']['eager_bytes'],
               eager_bytes_last=rows[-1]['attention']['eager_bytes'],
               aexp_bytes_last=rows[-1]['attention']['forward_aexp_bytes'],
               layers_grown_last=(rows[-1]['attention']['layers_grown_forward'],
                                  rows[-1]['attention']['layers_grown_backward']))
    print('%s head_median=%.5f s tail_median=%.5f s device_mb %s->%s '
          'eager_bytes %d->%d aexp_bytes=%d layers_grown=%s'
          % (label, out['head_median_seconds'], out['tail_median_seconds'],
             out['device_mb_first'], out['device_mb_max'],
             out['eager_bytes_first'], out['eager_bytes_last'],
             out['aexp_bytes_last'], out['layers_grown_last']))
    return out


def load(root, name):
    path = root / name / 'result.json'
    assert path.is_file(), 'missing arm: ' + name
    return json.loads(path.read_text())


def main():
    root = Path(sys.argv[1])
    verdict = {}
    before = load(root, 'before')
    after = load(root, 'after')
    try:
        eager = load(root, 'eager-ref')
    except AssertionError as e:
        print('MISSING ARM:', e, '-- the pure-eager reference is not available, so '
              'the discarded launch is NOT priced and `eager_vs_before` is not run.')
        eager = None
    layers = before['shape'][7]

    print('=== 0. the witness covers every step, layer and direction ===')
    arms = [('before', before), ('after', after)] + ([('eager-ref', eager)] if eager else [])
    for name, arm in arms:
        watch_coverage_fail(arm, 700 if len(arm['steps']) >= 700 else len(arm['steps']))
        validate(arm, len(arm['steps']))
        print('COVERAGE OK', name, len(arm['steps']), 'steps x', layers, 'layers x 2 directions')

    print()
    print('=== 1. THE TRIGGER, from the arm that behaves like main ===')
    verdict['trigger_before'] = trigger_table(before['steps'], layers, 'before')
    refusals = sum(n for key in ('forward_status', 'backward_status')
                   for name, n in verdict['trigger_before'][key]['totals'].items()
                   if name in ('FUSED_REFUSED_REGIME', 'FUSED_CORNER'))
    verdict['refusals_before'] = refusals
    verdict['inconclusive'] = refusals == 0
    if refusals == 0:
        print('INCONCLUSIVE: zero refusals in %d steps. The fallback did NOT occur '
              'on this run, so nothing below prices it.' % len(before['steps']))

    print()
    print('=== 2. the two arms are two arms (status 3 is the witness) ===')
    def sticky_count(arm):
        return sum(row['attention'][k].count(STICKY)
                   for row in arm['steps'] for k in ('forward_status', 'backward_status'))
    verdict['sticky_observations'] = dict(before=sticky_count(before),
                                          after=sticky_count(after),
                                          eager_ref=sticky_count(eager) if eager else None)
    print('FUSED_SKIPPED_STICKY observations:', verdict['sticky_observations'])
    # Two independent witnesses, both read from INSIDE the process that loaded
    # the binary. The build flag read back from the .so, and the behavior.
    flags = dict(before=before.get('attn_sticky_fallback'),
                 after=after.get('attn_sticky_fallback'),
                 eager_ref=eager.get('attn_sticky_fallback') if eager else None)
    verdict['attn_sticky_fallback_flag'] = flags
    print('byte_lm_attn_sticky_fallback() read from inside each process:', flags)
    assert flags['before'] is False, ('the `before` arm did not carry '
                                      '-D MOJOLEARN_ATTN_NO_STICKY=1: ' + repr(flags))
    assert flags['after'] is True, ('the `after` arm is not the latch build: ' + repr(flags))
    verdict['arms_are_two_arms'] = (sticky_count(before) == 0 and sticky_count(after) > 0)
    print('arms_are_two_arms:', verdict['arms_are_two_arms'],
          '(before must be 0 and after must be > 0; both zero means one build ran twice, '
          'which reads exactly like a passed identity gate)')

    print()
    print('=== 3. BITS. before vs after, and the pure-eager reference ===')
    pairs = [('after_vs_before', before, after)]
    if eager:
        pairs.append(('eager_vs_before', before, eager))
    for label, left, right in pairs:
        watch_compare_fail(left, right)
        out = compare(left, right)
        verdict[label] = out
        print('%s: %d steps, %d witnessed anchors, loss_differ=%s hash_differ=%s -> %s'
              % (label, out['steps'], out['anchors'],
                 out['loss_differ'][:8], out['hash_differ'][:8],
                 'EQUAL' if out['equal'] else 'BITS MOVED'))

    print()
    print('=== 4. step time and device footprint ===')
    verdict['timing'] = {name: timing(arm['steps'], name) for name, arm in arms}
    tb = verdict['timing']['before']['tail_median_seconds']
    ta = verdict['timing']['after']['tail_median_seconds']
    te = verdict['timing']['eager-ref']['tail_median_seconds'] if eager else float('nan')
    hb = verdict['timing']['before']['head_median_seconds']
    verdict['tail_speedup_after_over_before'] = tb / ta if ta else None
    verdict['remaining_gap_after_over_head'] = ta / hb if hb else None
    print('tail: before %.5f  after %.5f  eager-ref %.5f' % (tb, ta, te))
    print('latch recovers %.3fx of the tail; %.3fx of the pre-transition head remains unrecovered'
          % (tb / ta if ta else float('nan'), ta / hb if hb else float('nan')))
    print('the discarded fused launch was worth %.5f s a step (before tail minus eager-ref tail)'
          % (tb - te))

    print()
    print('=== 5. controls at head_dim 64 (at head_dim 8 the fused kernel is not '
          'instantiated and BOTH arms fall back, which means nothing) ===')
    for arm, expected in (('forced-eager', -1), ('forced-fused', 0),
                          ('nosticky-forced-eager', -1), ('nosticky-forced-fused', 0)):
        try:
            control = load(root, arm)
        except AssertionError as e:
            print('MISSING CONTROL:', e)
            continue
        row = validate(control, 1)[0]
        for key in ('forward_status', 'backward_status'):
            values = row['attention'][key]
            ok = values == [expected] * control['shape'][7]
            print(('MATCH: ' if ok else 'MISMATCH: '), arm, key, values)
            verdict.setdefault('controls', {})[arm + '.' + key] = values

    print()
    print('=== 6. batch 4 for 2,000 steps ===')
    try:
        b4 = load(root, 'batch4')
        rows = b4['steps']
        verdict['batch4'] = dict(steps_completed=len(rows),
                                 last_step=rows[-1]['step'] if rows else None,
                                 device_mb_max=max((r['device_used_mb'] for r in rows
                                                    if r.get('device_used_mb') is not None),
                                                   default=None),
                                 eager_bytes_last=rows[-1]['attention']['eager_bytes'] if rows else None)
        print('batch4 completed %d steps (target 2000); %s' % (len(rows), verdict['batch4']))
    except AssertionError as e:
        verdict['batch4'] = dict(steps_completed=0, note=str(e))
        print('batch4 produced no result.json:', e,
              '-- an OOM here is a RESULT: the latch removes a LAUNCH, not a BUFFER.')

    (root / 'verdict.json').write_text(json.dumps(verdict, indent=1))
    print()
    print('VERDICT:', 'INCONCLUSIVE (no refusals)' if verdict['inconclusive'] else
          ('BITS MOVED -- the latch is a DEFECT'
           if not verdict['after_vs_before']['equal'] else
           'latch is bit-identical and recovers %.3fx of the tail step'
           % verdict['tail_speedup_after_over_before']))


if __name__ == '__main__':
    main()
