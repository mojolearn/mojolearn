#!/usr/bin/env python3
"""DEVIATION 3112's verdict: are the backward corner refusals REAL?

The `refusing` arm is the control and must reproduce 3,697 backward refusals
and a 0.457 s tail, or the box or the data moved and nothing compares. The
`norefuse` arm is a MEASUREMENT BUILD THAT IS NEVER SHIPPED: its backward
keeps its own output on a corner instead of falling back.

BITS ARE THE WHOLE EXPERIMENT. Equal over 700 steps and 8 state anchors means
every one of those refusals was a false positive and the 2.13x is recoverable.
Different means the fallback is earning its cost and no amount of predicate
tightening will remove it; only recomputing what the eager chain would have
produced can. EITHER ANSWER IS THE RESULT, and the speed line is meaningless
unless the bits are equal.

Every check runs against a deliberately corrupted copy of its own input first
and must reject it BY NAME.
"""
import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

NAMES = {-1: 'NOT_ATTEMPTED', 0: 'FUSED_RAN', 1: 'FUSED_REFUSED_REGIME',
         2: 'FUSED_CORNER', 3: 'FUSED_SKIPPED_STICKY'}
# Measured 2026-09-18, pod rhjqy941tjl5yw, H100 sm_90a, commit 8a5212a78.
PRIOR_BACKWARD_CORNERS = 3697
PRIOR_UNGUARDED_TAIL = 0.45737
HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


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
        assert len(a['attn_materialized']) == n, 'materialization coverage'
    return rows


def watch_coverage_fail(result, minimum):
    for name, mutate in (
        ('step coverage', lambda x: x['steps'].__delitem__(slice(minimum - 1, None))),
        ('layer coverage', lambda x: x['steps'][0]['attention']['forward_status'].pop()),
        ('unknown status', lambda x: x['steps'][0]['attention']['forward_status'].__setitem__(0, 99)),
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


def compare(left, right):
    a, b = left['steps'], right['steps']
    assert len(a) == len(b), 'step count differs'
    loss_differ = [x['step'] for x, y in zip(a, b)
                   if json.dumps(x['loss']) != json.dumps(y['loss'])]
    hash_differ, anchors = [], 0
    for x, y in zip(a, b):
        if 'parameters' not in x or 'parameters' not in y:
            continue
        anchors += 1
        for key in HASHES:
            if x[key] != y[key]:
                hash_differ.append((x['step'], key))
    assert anchors > 0, 'no witnessed anchors'
    return dict(steps=len(a), anchors=anchors, loss_differ=loss_differ,
                hash_differ=hash_differ,
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
              'loss_differ=%d hash_differ=%d'
              % (len(out['loss_differ']), len(out['hash_differ'])))


def counts(rows, key):
    return Counter(v for row in rows for v in row['attention'][key])


def timing(rows, label):
    body = rows[1:]
    mb = [r['device_used_mb'] for r in rows if r.get('device_used_mb') is not None]
    out = dict(steps=len(rows),
               head_median_seconds=statistics.median(r['seconds'] for r in body[:50]),
               tail_median_seconds=statistics.median(r['seconds'] for r in body[-50:]),
               device_mb_first=mb[0] if mb else None,
               device_mb_max=max(mb) if mb else None,
               eager_bytes_first=rows[0]['attention']['eager_bytes'],
               eager_bytes_last=rows[-1]['attention']['eager_bytes'],
               layers_grown_last=(rows[-1]['attention']['layers_grown_forward'],
                                  rows[-1]['attention']['layers_grown_backward']))
    print('%-10s head=%.5f s tail=%.5f s device_mb %s->%s eager_bytes %d->%d grown=%s'
          % (label, out['head_median_seconds'], out['tail_median_seconds'],
             out['device_mb_first'], out['device_mb_max'],
             out['eager_bytes_first'], out['eager_bytes_last'],
             out['layers_grown_last']))
    return out


def load(root, name):
    path = root / name / 'result.json'
    assert path.is_file(), 'missing arm: ' + name
    return json.loads(path.read_text())


def main():
    root = Path(sys.argv[1])
    v = {}
    un = load(root, 'refusing')
    gu = load(root, 'norefuse')
    layers = un['shape'][7]

    print('=== 0. coverage ===')
    for name, arm in (('refusing', un), ('norefuse', gu)):
        watch_coverage_fail(arm, len(arm['steps']))
        validate(arm, len(arm['steps']))
        print('COVERAGE OK', name, len(arm['steps']), 'steps x', layers, 'layers x 2')

    print()
    print('=== 1. the two arms are two arms ===')
    flags = dict(refusing=un.get('attn_bwd_corner_refuses'),
                 norefuse=gu.get('attn_bwd_corner_refuses'))
    v['bwd_corner_refuses_flag'] = flags
    print('byte_lm_attn_bwd_corner_refuses() from inside each process:', flags)
    assert flags['refusing'] is True, 'the control arm is not the refusing build: %r' % flags
    assert flags['norefuse'] is False, 'the norefuse arm did not carry the define: %r' % flags

    print()
    print('=== 2. DOES THE UNGUARDED ARM REPRODUCE THE MEASURED LEG? ===')
    ub = counts(un['steps'], 'backward_status')
    uf = counts(un['steps'], 'forward_status')
    v['unguarded_backward'] = {NAMES[a]: b for a, b in sorted(ub.items())}
    v['unguarded_forward'] = {NAMES[a]: b for a, b in sorted(uf.items())}
    print('unguarded backward', v['unguarded_backward'])
    print('unguarded forward ', v['unguarded_forward'])
    un_t = timing(un['steps'], 'refusing')
    v['reproduced'] = (abs(ub[2] - PRIOR_BACKWARD_CORNERS) <= 0.15 * PRIOR_BACKWARD_CORNERS
                       and abs(un_t['tail_median_seconds'] - PRIOR_UNGUARDED_TAIL)
                       <= 0.10 * PRIOR_UNGUARDED_TAIL)
    print('reproduced (%d corners vs the measured %d; tail %.5f vs %.5f): %s'
          % (ub[2], PRIOR_BACKWARD_CORNERS, un_t['tail_median_seconds'],
             PRIOR_UNGUARDED_TAIL, v['reproduced']))
    if not v['reproduced']:
        print('THE CONTROL DID NOT REPRODUCE. Everything below is NOT comparable '
              'to the 2026-09-18 leg and must not be reported as if it were.')

    print()
    print('=== 3. BITS. THIS IS THE WHOLE EXPERIMENT. ===')
    watch_compare_fail(un, gu)
    out = compare(un, gu)
    v['guarded_vs_unguarded'] = out
    print('guarded_vs_unguarded: %d steps, %d anchors, loss_differ=%s hash_differ=%s -> %s'
          % (out['steps'], out['anchors'], out['loss_differ'][:8],
             out['hash_differ'][:8], 'EQUAL' if out['equal'] else 'BITS MOVED'))
    if not out['equal']:
        print('THE REFUSALS ARE REAL. The fallback is earning its cost and no '
              'predicate tightening removes it; the only route left is computing '
              'what the eager chain would have produced. A RESULT, not a failure.')

    print()
    print('=== 4. what NOT refusing costs and buys ===')
    gb = counts(gu['steps'], 'backward_status')
    gf = counts(gu['steps'], 'forward_status')
    v['guarded_backward'] = {NAMES[a]: b for a, b in sorted(gb.items())}
    v['guarded_forward'] = {NAMES[a]: b for a, b in sorted(gf.items())}
    print('guarded backward', v['guarded_backward'])
    print('guarded forward ', v['guarded_forward'])
    gu_t = timing(gu['steps'], 'norefuse')
    v['timing'] = dict(unguarded=un_t, guarded=gu_t)
    v['tail_speedup'] = (un_t['tail_median_seconds'] / gu_t['tail_median_seconds']
                         if gu_t['tail_median_seconds'] else None)
    v['device_mb_saved'] = ((un_t['device_mb_max'] or 0) - (gu_t['device_mb_max'] or 0))
    print('tail speedup %.4fx; device peak %s -> %s MB (saved %s); eager_bytes %d -> %d'
          % (v['tail_speedup'], un_t['device_mb_max'], gu_t['device_mb_max'],
             v['device_mb_saved'], un_t['eager_bytes_last'], gu_t['eager_bytes_last']))

    print()
    print('=== 5. controls at head_dim 64 ===')
    for arm, expected in (('refusing-forced-eager', -1), ('refusing-forced-fused', 0),
                          ('norefuse-forced-fused', 0)):
        try:
            c = load(root, arm)
        except AssertionError as e:
            print('MISSING CONTROL:', e)
            continue
        row = validate(c, 1)[0]
        for key in ('forward_status', 'backward_status'):
            vals = row['attention'][key]
            print(('MATCH: ' if vals == [expected] * c['shape'][7] else 'MISMATCH: '),
                  arm, key, vals)
            v.setdefault('controls', {})[arm + '.' + key] = vals

    print()
    print('=== 6. batch 4 for 2,000 steps, under the NOREFUSE build ===')
    try:
        b4 = load(root, 'batch4')
        rows = b4['steps']
        v['batch4'] = dict(steps_completed=len(rows),
                           device_mb_max=max((r['device_used_mb'] for r in rows
                                              if r.get('device_used_mb') is not None),
                                             default=None),
                           eager_bytes_last=rows[-1]['attention']['eager_bytes'] if rows else None,
                           backward_corners=sum(row['attention']['backward_status'].count(2)
                                                for row in rows))
        print('batch4:', v['batch4'])
    except AssertionError as e:
        # The probe prints every step as it goes, so a kill still leaves them.
        log = root / 'batch4.log'
        n = 0
        last = None
        if log.is_file():
            for line in log.read_text().splitlines():
                if line.startswith('{"step"'):
                    n += 1
                    last = line
        v['batch4'] = dict(steps_completed=n, note=str(e), from_log=True,
                           last_line=(last[:300] if last else None))
        print('batch4 has no result.json (%s); %d per-step records recovered from the log'
              % (e, n))
        if last:
            print('  last:', last[:300])

    (root / 'verdict.json').write_text(json.dumps(v, indent=1))
    print()
    if not v['reproduced']:
        print('VERDICT: NOT COMPARABLE -- the unguarded control did not reproduce.')
    elif not v['guarded_vs_unguarded']['equal']:
        print('VERDICT: THE BACKWARD CORNER REFUSALS ARE REAL -- not refusing MOVES '
              'BITS. The fallback stays. The remaining route is the signed-zero repair.')
    else:
        print('VERDICT: NOT REFUSING MOVES NO BIT over %d steps and %d anchors. Every '
              'one of those refusals was a FALSE POSITIVE. backward refusals %d -> %d; '
              'tail %.4fx; device peak saved %s MB.'
              % (out['steps'], out['anchors'], ub[2], gb[2], v['tail_speedup'],
                 v['device_mb_saved']))


if __name__ == '__main__':
    main()
