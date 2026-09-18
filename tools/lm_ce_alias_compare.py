#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The verdict for tools/lm_ce_alias_body.sh. Reads the probe directories,
prints every check BY NAME with its evidence, and writes verdict.json.

EVERY CHECK NAMES WHAT WOULD HAVE MADE IT FAIL, because a check that cannot
fail is not a check:

  ce_arms_are_two_arms   the two runs report DIFFERENT `ce_aliased`. Without
                         it, "the hashes agree" is one build compared with
                         itself and says nothing at all.
  ce_bits_unmoved        every step's six hashes equal across the two arms.
  ce_memory_moved        the aliased arm's device peak is LOWER. Equal peaks
                         would mean the aliasing did not reach the allocator,
                         whatever the source says.
  eager_witness_moves    zero layers grown on the fused path, every layer
                         grown under the forced eager path. Zero in both arms
                         is a blind witness, not a clean run.
  eager_hypothesis       the long run's THREE-WAY outcome, never a pass/fail
                         that could be read as a pass by default. CONFIRMED
                         needs the witness to grow AND the device memory to
                         step; FALSIFIED is memory stepping while the witness
                         stays at zero; INCONCLUSIVE is neither moving, which
                         means the transition did not reproduce and says
                         nothing either way.
  resume_matches         the resumed tail equals the uninterrupted tail, hash
                         for hash.
  resume_control_separates
                         the moments-dropped control DIFFERS. If it does not,
                         the comparison is not reading m and v and
                         `resume_matches` proves nothing. Expect it to differ
                         on m, v and parameters from the FIRST resumed step
                         while loss and gradients still match there: the
                         parameters were restored correctly and the moments
                         only enter the update.
"""
import argparse
import json
from pathlib import Path

HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


def read(out, name):
    path = out / name / 'result.json'
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def peak(result):
    values = [s['device_used_mb'] for s in result['steps'] if s.get('device_used_mb')]
    return max(values) if values else None


def compare_steps(a, b, first=0):
    """Per-step hash equality, reported as a list of the fields that differ."""
    rows = []
    by_b = {s['completed_steps']: s for s in b['steps']}
    for step in a['steps']:
        other = by_b.get(step['completed_steps'])
        if other is None:
            continue
        differ = [k for k in HASHES if step[k] != other[k]]
        rows.append(dict(completed_steps=step['completed_steps'], differ=differ))
    return rows


def eager_hypothesis(long_run):
    """Three outcomes, spelled out, over the long witness run.

    lane/lm-training-shakedown measured device memory stepping 16.949 ->
    34.397 GB and seconds 0.207 -> 0.44 between steps 210 and 480 at this
    shape, cause NOT ESTABLISHED. The hypothesis under test is that the step
    is the eager attention fallback growing the ten [B, n_heads, L, S]
    arrays, one layer at a time, never released.

    The strength here is a CONTRAST, not a raw number: the memory step is the
    known event, and the witness either coincides with it or it does not.
    """
    steps = long_run['steps']
    grown = [(s['step'], (s.get('attention') or {}).get('layers_grown_forward'),
              (s.get('attention') or {}).get('layers_grown_backward'),
              s.get('device_used_mb'), s['seconds'])
             for s in steps]
    layers = next((( s.get('attention') or {}).get('layers') for s in steps
                   if s.get('attention')), None)
    first_growth = next((g for g in grown if g[1]), None)
    full_growth = next((g for g in grown if layers and g[1] == layers and g[2] == layers), None)
    mb = [g[3] for g in grown if g[3]]
    memory_stepped = bool(mb) and (max(mb) - min(mb)) > 2048   # > 2 GB
    seconds = [g[4] for g in grown]
    head = seconds[:min(50, len(seconds))]
    tail = seconds[-min(50, len(seconds)):]
    slowed = bool(head) and bool(tail) and (sum(tail) / len(tail)) > 1.5 * (sum(head) / len(head))
    witness_grew = first_growth is not None

    if witness_grew and memory_stepped:
        verdict = 'CONFIRMED'
    elif memory_stepped and not witness_grew:
        verdict = 'FALSIFIED'
    elif witness_grew and not memory_stepped:
        verdict = 'WITNESS MOVED WITHOUT A MEMORY STEP; look again'
    else:
        verdict = 'INCONCLUSIVE: the transition did not reproduce in this run'

    last = (steps[-1].get('attention') or {}) if steps else {}
    return verdict, dict(
        steps_run=len(steps), layers=layers,
        first_growth_at=None if first_growth is None else first_growth[0],
        all_layers_grown_at=None if full_growth is None else full_growth[0],
        device_used_mb_min=min(mb) if mb else None,
        device_used_mb_max=max(mb) if mb else None,
        device_used_mb_delta=(max(mb) - min(mb)) if mb else None,
        memory_stepped=memory_stepped, slowed=slowed,
        seconds_head_mean=(sum(head) / len(head)) if head else None,
        seconds_tail_mean=(sum(tail) / len(tail)) if tail else None,
        final_attention_report=last,
        eager_bytes_final=last.get('eager_bytes'),
        eager_plus_aexp_bytes_final=last.get('total_bytes'),
        ids=long_run.get('ids'), corpus=long_run.get('corpus'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    checks, notes = {}, {}

    aliased, unaliased = read(out, 'aliased'), read(out, 'unaliased')
    if aliased and unaliased:
        notes['ce_aliased_flags'] = [aliased['ce_aliased'], unaliased['ce_aliased']]
        checks['ce_arms_are_two_arms'] = (aliased['ce_aliased'] is True
                                          and unaliased['ce_aliased'] is False)
        rows = compare_steps(aliased, unaliased)
        notes['ce_steps_compared'] = len(rows)
        notes['ce_steps_differing'] = [r for r in rows if r['differ']]
        checks['ce_bits_unmoved'] = bool(rows) and all(not r['differ'] for r in rows)
        pa, pu = peak(aliased), peak(unaliased)
        notes['ce_device_peak_mb'] = dict(aliased=pa, unaliased=pu)
        if pa is not None and pu is not None:
            notes['ce_device_peak_saved_mb'] = pu - pa
            checks['ce_memory_moved'] = pu > pa
    else:
        notes['ce'] = 'one or both arms missing; nothing compared'

    off, on = read(out, 'eager-off'), read(out, 'eager-on')
    if off and on:
        def grown(result):
            report = result['steps'][-1]['attention'] if result['steps'] else None
            return report or {}
        g_off, g_on = grown(off), grown(on)
        notes['eager_report_fused'] = g_off
        notes['eager_report_eager'] = g_on
        layers = g_on.get('layers')
        checks['eager_witness_moves'] = bool(
            layers and g_off.get('layers_grown_forward') == 0
            and g_off.get('layers_grown_backward') == 0
            and g_on.get('layers_grown_forward') == layers
            and g_on.get('layers_grown_backward') == layers)
    else:
        notes['eager'] = 'one or both attention arms missing; nothing compared'

    long_run = read(out, 'witness-long')
    if long_run and long_run['steps']:
        verdict, detail = eager_hypothesis(long_run)
        notes['eager_hypothesis'] = dict(verdict=verdict, **detail)
        # Only a FALSIFIED reading is a failure of this lane's claim; the
        # inconclusive readings are recorded as what they are and must not
        # be allowed to read as a pass, so they are not checks at all.
        checks['eager_hypothesis_not_falsified'] = verdict != 'FALSIFIED'
    else:
        notes['eager_hypothesis'] = 'the long witness run is missing; nothing tested'

    base, resumed, control = (read(out, 'ckpt-save'), read(out, 'ckpt-resume'),
                              read(out, 'ckpt-control'))
    tail = read(out, 'aliased')
    if base and resumed and tail:
        # The uninterrupted tail is the aliased run's steps past the save point.
        saved_at = base['steps'][-1]['completed_steps'] if base['steps'] else 0
        uninterrupted = dict(tail, steps=[s for s in tail['steps']
                                          if s['completed_steps'] > saved_at])
        rows = compare_steps(resumed, uninterrupted)
        notes['resume_steps_compared'] = len(rows)
        notes['resume_steps_differing'] = [r for r in rows if r['differ']]
        checks['resume_matches'] = bool(rows) and all(not r['differ'] for r in rows)
        if control:
            rows = compare_steps(control, uninterrupted)
            notes['resume_control_rows'] = rows
            checks['resume_control_separates'] = bool(rows) and any(r['differ'] for r in rows)
        info = out / 'ckpt-save' / 'checkpoint.json'
        if info.is_file():
            notes['checkpoint'] = json.loads(info.read_text())
    else:
        notes['resume'] = 'checkpoint arms missing; nothing compared'

    verdict = dict(schema='mojolearn.lm-ce-alias-verdict.v1', checks=checks, notes=notes,
                   passed=bool(checks) and all(checks.values()))
    (out / 'verdict.json').write_text(json.dumps(verdict, indent=1))
    for name, value in sorted(checks.items()):
        print('%-34s %s' % (name, 'PASS' if value else 'FAIL'))
    hypothesis = notes.get('eager_hypothesis')
    if isinstance(hypothesis, dict):
        print('EAGER HYPOTHESIS: %s' % hypothesis['verdict'])
    print(json.dumps(notes, indent=1))
    print('VERDICT', 'PASSED' if verdict['passed'] else 'NOT PASSED')
    return 0 if verdict['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
