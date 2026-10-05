#!/usr/bin/env python3
"""Apply explicitly audited, captured own-cell repairs without erasing original evidence."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ref(ref):
    path = Path(ref['path'])
    if digest(path) != ref['sha256']:
        raise ValueError('changed evidence: ' + str(path))
    return json.loads(path.read_text())


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite evidence')
    if isinstance(value, dict):
        if value.get('error'):
            raise ValueError('explicit quality error')
        for key, item in value.items():
            if (key == 'finite' or key.endswith('_finite') or key.endswith('_passed')) and item is False:
                raise ValueError('failed quality flag')
            finite(item)
    elif isinstance(value, str) and value.strip().lower() in ('nan', 'inf', '-inf', '+inf', 'infinity'):
        raise ValueError('nonfinite quality string')
    elif isinstance(value, list):
        for item in value:
            finite(item)


def validate_cell(cell):
    if cell.get('status') != 'ok' or cell.get('rounds') != 1:
        raise ValueError('repair requires one successful scored sample')
    times = cell.get('times_ms')
    if not isinstance(times, list) or len(times) != 1:
        raise ValueError('missing scored time')
    for value in [*times, cell.get('median_ms'), cell.get('warmup_ms')]:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid time or absent warmup')
    if not cell.get('quality'):
        raise ValueError('absent quality evidence')
    finite(cell['quality'])


def apply(manifest):
    original = read_ref(manifest['original'])
    result = copy.deepcopy(original)
    seen = set()
    provenance = []
    for overlay in manifest['repairs']:
        board = read_ref(overlay['board'])
        receipt = read_ref(overlay['capture'])
        if not receipt.get('verified_at') or not receipt.get('manifest_sha256'):
            raise ValueError('unverified capture')
        row = next((r for r in receipt['rows'] if r['path'] == overlay['capture_board_path']), None)
        if not row or row['sha256'] != overlay['board']['sha256']:
            raise ValueError('board absent from verified capture')
        audit = read_ref(overlay['audit'])
        if audit.get('status') != 'PASS' or audit.get('failures'):
            raise ValueError('audit did not pass')
        if audit.get('board_sha256') != overlay['board']['sha256']:
            raise ValueError('audit not bound to replacement board')
        if not overlay.get('source_description') or not overlay.get('hardware_description'):
            raise ValueError('missing source/hardware disclosure')
        for key in overlay['races']:
            if key in seen or key not in original['races'] or key not in board['races']:
                raise ValueError('duplicate or unknown race ' + key)
            seen.add(key)
            replacement = board['races'][key]
            if replacement.get('status') != 'done' or replacement.get('params_check') != 'MATCHED':
                raise ValueError('nonterminal repair')
            own = [c for c in replacement.get('cells', []) if c.get('arm') in ('ours', 'ours-identical', 'ours-fast')]
            if not own:
                raise ValueError('no own repair cells')
            for cell in own:
                validate_cell(cell)
            old = original['races'][key]
            # Preserve the original complete race (including failures) and opponents.
            merged = copy.deepcopy(replacement)
            arms = {c['arm'] for c in own}
            merged['cells'] = copy.deepcopy(own) + [copy.deepcopy(c) for c in old.get('cells', []) if c.get('arm') not in arms]
            # Inference was not admitted by the fit audit. Retain original inference.
            merged['infer_cells'] = copy.deepcopy(old.get('infer_cells', []))
            merged['overlay_original_race'] = copy.deepcopy(old)
            proof = dict(overlay)
            proof.update(race=key, original=manifest['original'], qualification=False,
                         artifact_identity=board.get('config', {}).get('artifact_identity'),
                         replacement_box=board.get('box'), original_box=original.get('box'))
            merged['repair_overlay'] = proof
            result['races'][key] = merged
            provenance.append(proof)
    result['repair_overlay'] = {'schema': 'mojolearn.board-repair-overlay/1', 'original': manifest['original'],
                                'repairs': provenance, 'uniform_freeze': False, 'identical_qualified': False,
                                'note': 'Scoped repaired own fit cells; original failures and inference retained. Box/config describe original run; replacement provenance is per row. No default promotion.'}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    result = apply(manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'board.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (out / 'overlay-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    from bench_board import render_board
    (out / 'BOARD.md').write_text(render_board(result))
    print(json.dumps({'status': 'RENDERED_SCOPED_REPAIRS', 'repaired_races': len(result['repair_overlay']['repairs']), 'qualified': False, 'out': str(out)}))


if __name__ == '__main__':
    main()
