#!/usr/bin/env python3
"""Render retained candidate measurements without rerunning builds or identity.

The index records evidence paths, per-cell scope and provenance. Only complete
same-machine A/B pairs receive ratios. Component timings never become full-board
opponent measurements or automatic default decisions.
"""
import argparse
import collections
import hashlib
import html
import json
import math
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def status(cells):
    states = collections.Counter(c['status'] for c in cells)
    if states['MEASURED']:
        return 'PARTIAL_MEASUREMENTS_RETAINED'
    return ', '.join(sorted(states)) if cells else 'PENDING_MEASUREMENT'


def build(inventory, index):
    cards = {x['id']: {'id': x['id'], 'title': x['title'], 'mode': x['mode'],
                      'vendors': x['vendors'], 'cells': [], 'status': 'PENDING_MEASUREMENT'}
             for x in inventory['candidates']}
    for original in index.get('cells', []):
        cell = dict(original)
        if cell['id'] not in cards:
            raise ValueError('Unknown candidate: ' + cell['id'])
        if cell['vendor'] not in cards[cell['id']]['vendors']:
            raise ValueError('Unsupported vendor: ' + cell['id'])
        cell.pop('candidate_over_baseline', None)
        if cell['status'] == 'MEASURED':
            required = ['case', 'scope', 'machine', 'source_sha', 'artifact_hashes', 'evidence',
                        'warmups', 'scored_samples', 'baseline_ms', 'candidate_ms', 'returncode',
                        'baseline_machine', 'candidate_machine', 'baseline_source_sha', 'candidate_source_sha']
            missing = [k for k in required if k not in cell]
            if missing:
                raise ValueError(f"{cell['id']}: missing measurement provenance {missing}")
            if cell['scope'] not in ['component', 'public_caller_component', 'full_workload']:
                raise ValueError('Unknown timing scope')
            if cell['returncode'] != 0 or cell['warmups'] != 1 or cell['scored_samples'] != 1:
                raise ValueError('Incomplete or incompatible scored execution')
            if not positive(cell['baseline_ms']) or not positive(cell['candidate_ms']):
                raise ValueError('Timing must be finite and positive')
            if not all(cell[k] for k in ['machine', 'source_sha', 'artifact_hashes', 'evidence']):
                raise ValueError('Empty provenance')
            if cell['machine'] != cell['baseline_machine'] or cell['machine'] != cell['candidate_machine']:
                raise ValueError('A/B machines differ')
            if cell['source_sha'] != cell['baseline_source_sha'] or cell['source_sha'] != cell['candidate_source_sha']:
                raise ValueError('A/B source freezes differ')
            if not isinstance(cell['artifact_hashes'], dict) or not all(cell['artifact_hashes'].get(k) for k in ['baseline', 'candidate']):
                raise ValueError('Both artifact hashes are required')
            cell['candidate_over_baseline'] = cell['candidate_ms'] / cell['baseline_ms']
        else:
            # A completed pair rejected by quality may retain observed times,
            # clearly separated from admitted performance or default decisions.
            cell.pop('unqualified_timing', None)
            if (cell.get('execution_status') == 'MEASURED_FULL'
                    and cell.get('warmups') == 1 and cell.get('scored_samples') == 1
                    and positive(cell.get('baseline_ms')) and positive(cell.get('candidate_ms'))):
                cell['unqualified_timing'] = {
                    'baseline_ms': cell['baseline_ms'], 'candidate_ms': cell['candidate_ms'],
                    'observed_candidate_over_baseline': cell['candidate_ms'] / cell['baseline_ms'],
                    'qualification': 'Observed only; not admitted and not a default decision'}
            # Interrupted attempts never receive an inferred timing or ratio.
            cell.pop('baseline_ms', None)
            cell.pop('candidate_ms', None)
        cards[cell['id']]['cells'].append(cell)
    for card in cards.values():
        card['status'] = status(card['cells'])
    return {'schema': 'mojolearn.performance-measurement-board/1', 'campaign': inventory['campaign'],
            'identity_policy': inventory.get('identity_policy', 'Previously validated; no separate retest by owner instruction'),
            'evidence_policy': inventory.get('evidence_policy', 'One excluded warmup and one scored sample. Identity and compilation are reused; no separate retests.'),
            'promotion': False, 'machines': index.get('machines', []),
            'cards': list(cards.values()), 'notes': index.get('notes', []),
            'decisions': index.get('decisions', []),
            'coverage': index.get('coverage', []),
            'pending_work': index.get('pending_work', []),
            'remaining_catalog': index.get('remaining_catalog', {})}


def escape(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def anchor(prefix, value):
    return prefix + '-' + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def attempt_anchor(card, cell):
    return anchor('attempt', [card['id'], cell['vendor'], cell.get('case'), cell.get('evidence')])


def profile_data(cell):
    return {key: cell.get(key) for key in ('controls', 'requested_controls')}


def profile_anchor(cell):
    return anchor('toggles', profile_data(cell))


def coverage_data(cell):
    return {key: cell.get(key, []) for key in ('implementation_ids', 'source_coverage_pending')}


def coverage_anchor(cell):
    return anchor('coverage', coverage_data(cell))


def observed_seconds(cell):
    timing = cell.get('unqualified_timing', cell)
    return tuple(timing.get(key) / 1000 if positive(timing.get(key)) else None
                 for key in ('candidate_ms', 'baseline_ms'))


def display(value, compact=False):
    if value is None:
        return '—'
    if compact and isinstance(value, float) and math.isfinite(value):
        return f'{value:.6g}'
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


def flatten(values, prefix=''):
    if isinstance(values, dict):
        return {path: value for key, child in sorted(values.items())
                for path, value in flatten(child, prefix + str(key) + '.').items()}
    return {prefix.rstrip('.'): values}


def quality_metrics(cell, arm):
    saved = cell.get('quality_metrics', {}).get(arm, {})
    metrics = saved.get('metrics', {}) if isinstance(saved, dict) else {}
    # Public runners name the single own arm ours / ours-fast. The A/B labels
    # below already identify it; retain other namespaces when present.
    if len(metrics) == 1 and next(iter(metrics)) in ('ours', 'ours-fast'):
        metrics = next(iter(metrics.values()))
    return flatten(metrics)


def quality_summary(cell):
    verdict = cell.get('quality_assessment') or cell.get('quality', 'NOT_RECORDED')
    a, b = quality_metrics(cell, 'A'), quality_metrics(cell, 'B')
    metrics = [f'{key} A/B={display(a.get(key), True)}/{display(b.get(key), True)}'
               for key in sorted(a.keys() | b.keys())]
    return display(verdict) + '; ' + ('; '.join(metrics) if metrics else 'metrics not recorded')


def control_values(config):
    if not isinstance(config, dict):
        return {}
    values = {}
    defines = config.get('defines', [])
    if isinstance(defines, dict):
        values.update({'define ' + key: display(value) for key, value in defines.items()})
    else:
        for define in defines:
            key, equal, value = str(define).partition('=')
            name = 'define ' + key
            value = value if equal else '(defined without a value)'
            # Preserve repeated definitions instead of guessing precedence.
            values[name] = values[name] + '; ' + value if name in values else value
    for key, value in config.items():
        if key != 'defines':
            values.update({key + '.' + name: display(item) for name, item in flatten(value).items()})
    return values


def control_table(controls):
    if not isinstance(controls, dict):
        return ['Configuration not recorded.']
    a, b = (control_values(controls.get(arm)) for arm in ('A', 'B'))
    if not (a or b):
        return ['No explicit overrides recorded; incumbent defaults remain in effect.'
                if all(isinstance(controls.get(arm), dict) for arm in ('A', 'B'))
                else 'Configuration not recorded for both arms.']
    lines = ['| Recorded control | A: candidate | B: incumbent |', '|---|---|---|']
    for key in sorted(a.keys() | b.keys()):
        values = [key] + [arm.get(key, 'incumbent default (no explicit override)')
                          if isinstance(controls.get(name), dict) else 'NOT_RECORDED'
                          for name, arm in (('A', a), ('B', b))]
        lines.append('| ' + ' | '.join(map(escape, values)) + ' |')
    return lines


def evidence_link(cell, out):
    value = cell.get('evidence')
    if not value:
        return 'Evidence not recorded'
    if str(value).startswith(('https://', 'http://')):
        target = str(value)
    else:
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        target = Path(os.path.relpath(path, out)).as_posix()
    return f'[retained receipt](<{target}>)'


def experiment_details(data, out):
    rows = [(card, cell) for card in data['cards'] for cell in card['cells']]
    lines = ['', '## Experiments run: toggles, timing and quality', '',
             'A is the candidate; B preserves the frozen incumbent. These are recorded arm configurations, '
             'not proof that every requested switch was compiled or reached at runtime. An omitted define '
             'is not OFF. Combined timings do not establish individual toggle winners.', '',
             'Expand a toggle profile or an attempt below. Metric values come from the saved scored results; '
             'their presence does not imply quality acceptance, complete model identity or promotion.', '']
    profiles, coverage = {}, {}
    for _, cell in rows:
        profiles.setdefault(profile_anchor(cell), cell)
        if any(coverage_data(cell).values()):
            coverage.setdefault(coverage_anchor(cell), coverage_data(cell))
    for key, cell in profiles.items():
        lines += [f'<a id="{key}"></a>', '<details>',
                  f'<summary>Recorded toggle profile {key.removeprefix("toggles-")}</summary>', '']
        lines += control_table(cell.get('controls'))
        requested = cell.get('requested_controls')
        if requested is not None and requested != cell.get('controls'):
            lines += ['', 'Requested selection differs from the recorded arm configuration:', '']
            lines += control_table(requested)
        lines += ['', '</details>', '']
    # Identical catalog disclosures occur in many receipts. Keep each exact
    # disclosure once, with per-attempt links, so the board remains readable.
    for key, saved in coverage.items():
        lines += [f'<a id="{key}"></a>', '<details>',
                  f'<summary>Recorded implementation IDs and source coverage {key.removeprefix("coverage-")}</summary>', '',
                  'These are recorded source disclosures, not proof of runtime reach.', '',
                  'Implementation IDs: ' + escape(display(saved['implementation_ids'])), '',
                  'Source coverage pending:', '']
        lines += ['- ' + escape(display(value)) for value in saved['source_coverage_pending']] or ['None recorded.']
        lines += ['', '</details>', '']
    for card, cell in rows:
        key = attempt_anchor(card, cell)
        label = f"{card['id']} — {cell['vendor']}/{cell.get('route', 'default')} — {cell.get('case', '')}"
        a_seconds, b_seconds = observed_seconds(cell)
        lines += [f'<a id="{key}"></a>', '<details>', f'<summary>{html.escape(label)}</summary>', '',
                  f"[Recorded toggles](#{profile_anchor(cell)}) · {evidence_link(cell, out)}", '',
                  f"Status: {escape(cell['status'])}. Scope: {escape(cell.get('scope', 'NOT_RECORDED'))}. "
                  f"Observed A: {display(a_seconds)} s; B: {display(b_seconds)} s.", '',
                  f"Quality assessment: {escape(cell.get('quality_assessment', cell.get('quality', 'NOT_RECORDED')))}. "
                  f"Identity: {escape(cell.get('identity', 'NOT_RECORDED'))}.", '']
        if cell.get('quality_reason'):
            lines += [escape(cell['quality_reason']), '']
        metrics_a, metrics_b = quality_metrics(cell, 'A'), quality_metrics(cell, 'B')
        if metrics_a or metrics_b:
            lines += ['| Saved quality metric | A: candidate | B: incumbent |', '|---|---:|---:|']
            for metric in sorted(metrics_a.keys() | metrics_b.keys()):
                lines.append('| ' + ' | '.join(map(escape, [metric, display(metrics_a.get(metric)),
                                                          display(metrics_b.get(metric))])) + ' |')
        else:
            lines.append('Scored quality metrics not recorded for this attempt.')
        if cell.get('quality_comparisons'):
            lines += ['', 'Saved quality gate and opponent comparisons (no reassessment):', '',
                      '| Evidence field | Recorded value |', '|---|---|']
            for field, value in flatten(cell['quality_comparisons']).items():
                lines.append('| ' + escape(field) + ' | ' + escape(display(value)) + ' |')
        lines += ['', 'Source: ' + escape(cell.get('source_sha', 'NOT_RECORDED')),
                  'Samples (warmup/scored): ' + escape(json.dumps(cell.get('actual_sample_counts', {
                      'warmups': cell.get('warmups'), 'scored_samples': cell.get('scored_samples')}), sort_keys=True)),
                  'Worker exits: ' + escape(cell.get('worker_returncodes', cell.get('returncode', 'NOT_RECORDED')))]
        if any(coverage_data(cell).values()):
            lines.append(f"[Recorded implementation IDs and {len(cell.get('source_coverage_pending', []))} "
                         f"source coverage gaps](#{coverage_anchor(cell)})")
        for field, title in [('failure_reasons', 'Failures'), ('resource_limitations', 'Resource limitations')]:
            if cell.get(field):
                lines += [title + ': ' + escape(display(cell[field]))]
        lines += ['', '</details>', '']
    if not rows:
        lines.append('No experiment attempts recorded.')
    return lines


def write(board, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    remaining=board.get('remaining_catalog',{})
    if remaining:
        lines=['# Itemized experiment coverage', '', remaining['policy'], '',
               f"{remaining['entries']} catalog entries; {remaining['interactions']} interaction plans; {remaining['direct_selection_count']} exact selections have campaign receipts.", '',
               'Roles: '+', '.join(f'{key}: {value}' for key,value in remaining['roles'].items())+'.', '',
               remaining['outside_catalog'], '',
               'Full workload, arm, source, prerequisite and receipt details are in [remaining-work.json](remaining-work.json). No missing timing is filled with zero or inferred from another configuration.', '',
               '| Selection | Role | Mode | Campaign evidence | Recipe admission | Complete pairs | Quality failures |',
               '|---|---|---|---|---|---:|---:|']
        for row in remaining['rows']:
            values=[row['id'],row['role'],', '.join(row['modes']) or 'UNRESOLVED',row['status'],row['recipe_status'],row['complete_pairs'],row['quality_failed']]
            lines.append('| '+' | '.join(map(escape,values))+' |')
        (out/'REMAINING.md').write_text('\n'.join(lines)+'\n')
    variants = [('BOARD', board)]
    for vendor in ['apple', 'amd', 'nvidia']:
        variants.append((vendor, dict(board, cards=[dict(c, cells=[x for x in c['cells'] if x['vendor'] == vendor],
                                                        status=status([x for x in c['cells'] if x['vendor'] == vendor]))
                                                   for c in board['cards'] if vendor in c['vendors']])))
    for name, data in variants:
        target = out / (name.lower() + '.json')
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        temp.replace(target)
        lines = ['# Candidate A/B measurements', '',
                 data.get('evidence_policy', 'One excluded warmup and one scored sample. Identity and compilation are reused; no separate retests.'),
                 'Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Default decisions are recorded beside source toggles; this board does not change them.', '',
                 '| Candidate | Mode | Measurement status | Captured pairs |', '|---|---|---|---:|']
        for card in data['cards']:
            # Completed execution and admission are separate. A retained full
            # pair can still fail quality or await identity; the status column
            # keeps that limitation visible without claiming no data exists.
            measured = sum(c['status'] == 'MEASURED' or
                           (c.get('execution_status') == 'MEASURED_FULL' and
                            c.get('warmups') == 1 and c.get('scored_samples') == 1)
                           for c in card['cells'])
            lines.append(f"| {card['id']} | {card['mode']} | {escape(card['status'])} | {measured} |")
        if data.get('coverage'):
            lines += ['', '## Full-workload coverage', '',
                      'Counts are retained complete A/B pairs, not individually decided experiment switches. Execution completion does not establish quality or identity admission.', '',
                      '| Vendor / mode | Complete pairs | Original failed attempts | Quality-rejected pairs | Remaining scope |',
                      '|---|---:|---:|---:|---|']
            for row in data['coverage']:
                if name != 'BOARD' and row['vendor'] != name:
                    continue
                lines.append('| ' + ' | '.join(escape(row.get(k, '')) for k in
                    ('label', 'complete_pairs', 'failed_attempts', 'quality_failed', 'remaining_scope')) + ' |')
        if data.get('remaining_catalog'):
            remaining=data['remaining_catalog']
            lines += ['', '## Individual experiment coverage', '',
                      f"[Itemized coverage ledger](REMAINING.md): {remaining['entries']} catalog entries and {remaining['interactions']} interaction plans; only {remaining['direct_selection_count']} exact selections have receipts in this campaign. Combined timings do not qualify individual members.", '',
                      'Unrun, source-rejected and previously decided work remain distinct. This ledger is not a claim that every listed entry has runnable binaries.']
        lines += ['', '## Captured evidence', '',
                  'Observed ratios retain complete scored pairs even while quality or identity is pending. They are not admitted gains or default decisions. A is candidate; B is baseline.', '',
                  '[Exact toggle profiles and per-attempt quality details](#experiments-run-toggles-timing-and-quality) are at the bottom. Times below are seconds; A is candidate and B is incumbent.', '',
                  '| Candidate | Vendor / route | Case | Scope | Status | A (s) | B (s) | Admitted A/B | Observed A/B | Quality (A/B) | Toggles / details | Evidence |',
                  '|---|---|---|---|---|---:|---:|---:|---:|---|---|---|']
        for card in data['cards']:
            for cell in card['cells']:
                ratio = cell.get('candidate_over_baseline')
                observed = cell.get('unqualified_timing', {}).get('observed_candidate_over_baseline', ratio)
                a_seconds, b_seconds = observed_seconds(cell)
                values = [card['id'], cell['vendor'] + '/' + cell.get('route', 'default'), cell.get('case', ''),
                          cell.get('scope', ''), cell['status'], display(a_seconds, True), display(b_seconds, True),
                          f'{ratio:.4f}' if ratio is not None else '—', f'{observed:.4f}' if observed is not None else '—',
                          quality_summary(cell), f'[toggles](#{profile_anchor(cell)}) / [details](#{attempt_anchor(card, cell)})',
                          evidence_link(cell, out)]
                lines.append('| ' + ' | '.join(map(escape, values)) + ' |')
        lines += ['', '## Campaign notes', ''] + ['- ' + escape(n) for n in data['notes']]
        if data['decisions']:
            lines += ['', '## Recorded source decisions', '', '| Candidate / arm | Decision | Source commit | Evidence |', '|---|---|---|---|']
            for decision in data['decisions']:
                lines.append('| ' + ' | '.join(escape(decision.get(k, '')) for k in ['candidate', 'decision', 'commit', 'evidence']) + ' |')
        if data.get('pending_work'):
            lines += ['', '## Unrun or blocked scope', '',
                      'These are not completed measurements and have no inferred timing. This summary does not imply every individual catalog experiment has been run.', '',
                      '| Scope | Status / reason | Evidence |', '|---|---|---|']
            for row in data['pending_work']:
                if name != 'BOARD' and row.get('vendor') not in (name, 'all'):
                    continue
                lines.append('| ' + ' | '.join(escape(row.get(k, '')) for k in ('scope', 'reason', 'evidence')) + ' |')
        failures = [(card, cell) for card in data['cards'] for cell in card['cells']
                    if cell['status'] in ('QUALITY_FAILED', 'FAILED_OR_INCOMPLETE', 'FAILED', 'REJECTED')]
        lines += ['', '## Failed or quality-rejected attempts', '',
                  'Original attempts remain visible after repairs. A quality failure may have complete timings; those are observations, not admitted gains. Interrupted or failed executions have no valid pair timing.', '']
        if failures:
            lines += ['| Candidate | Vendor / case | Outcome / reason | Worker exits | Samples A; B (warmup/scored) | Observed A/B time | Evidence |',
                      '|---|---|---|---|---|---:|---|']
            for card, cell in failures:
                timing = cell.get('unqualified_timing', {})
                ratio = timing.get('observed_candidate_over_baseline')
                samples = cell.get('actual_sample_counts', {})
                sample_text = '; '.join(f"{a}: {samples.get(a, {}).get('warmup', 0)}/{samples.get(a, {}).get('scored', 0)}" for a in ('A', 'B')) if samples else 'not recorded'
                reasons = cell.get('quality_reason') or '; '.join(cell.get('failure_reasons', [])) or cell['status']
                observed = f"{ratio:.4f} ({timing['candidate_ms']/1000:.4f}s / {timing['baseline_ms']/1000:.4f}s)" if ratio is not None else '—'
                values = [card['id'], cell['vendor'] + ' / ' + cell.get('case', ''),
                          cell['status'] + ': ' + reasons, cell.get('worker_returncodes', []),
                          sample_text, observed, cell.get('evidence', '')]
                lines.append('| ' + ' | '.join(map(escape, values)) + ' |')
        else:
            lines.append('No failed or quality-rejected attempts recorded for this scope.')
        lines += experiment_details(data, out)
        target = out / (name + '.md')
        temp = target.with_suffix('.tmp')
        temp.write_text('\n'.join(lines).rstrip() + '\n')
        temp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    board = build(read(args.inventory), read(args.index))
    write(board, args.out)
    print(json.dumps({'cards': len(board['cards']), 'cells': sum(len(c['cells']) for c in board['cards'])}))


if __name__ == '__main__':
    main()
