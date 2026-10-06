#!/usr/bin/env python3
"""Render retained candidate measurements without rerunning builds or identity.

The index records evidence paths, per-cell scope and provenance. Only complete
same-machine A/B pairs receive ratios. Component timings never become full-board
opponent measurements or automatic default decisions.
"""
import argparse
import collections
import json
import math
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
            # Interrupted/failed attempts retain diagnostics, never a speed ratio.
            cell.pop('baseline_ms', None)
            cell.pop('candidate_ms', None)
        cards[cell['id']]['cells'].append(cell)
    for card in cards.values():
        card['status'] = status(card['cells'])
    return {'schema': 'mojolearn.performance-measurement-board/1', 'campaign': inventory['campaign'],
            'identity_policy': 'Previously validated; no separate retest by owner instruction',
            'promotion': False, 'machines': index.get('machines', []),
            'cards': list(cards.values()), 'notes': index.get('notes', [])}


def escape(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def write(board, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
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
                 'One excluded warmup and one scored sample. Identity and compilation are reused; no separate retests.',
                 'Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Defaults remain unchanged.', '',
                 '| Candidate | Mode | Measurement status | Captured pairs |', '|---|---|---|---:|']
        for card in data['cards']:
            measured = sum(c['status'] == 'MEASURED' for c in card['cells'])
            lines.append(f"| {card['id']} | {card['mode']} | {escape(card['status'])} | {measured} |")
        lines += ['', '## Captured evidence', '',
                  '| Candidate | Vendor / route | Case | Scope | Status | B/A time | Evidence |',
                  '|---|---|---|---|---|---:|---|']
        for card in data['cards']:
            for cell in card['cells']:
                ratio = cell.get('candidate_over_baseline')
                values = [card['id'], cell['vendor'] + '/' + cell.get('route', 'default'), cell.get('case', ''),
                          cell.get('scope', ''), cell['status'], f'{ratio:.4f}' if ratio is not None else '—', cell.get('evidence', '')]
                lines.append('| ' + ' | '.join(map(escape, values)) + ' |')
        lines += ['', '## Campaign notes', ''] + ['- ' + escape(n) for n in data['notes']]
        target = out / (name + '.md')
        temp = target.with_suffix('.tmp')
        temp.write_text('\n'.join(lines) + '\n')
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
