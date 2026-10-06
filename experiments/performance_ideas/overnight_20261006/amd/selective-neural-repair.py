"""Retry selected failed neural arms; retain other cells and immutable receipts.

The controller supplies marker={"arms_by_race": {race_id: [arm, ...]},
"source_sha": frozen_harness_sha, ...} and wraps bench_board.run_race with
repair_neural(ctx, race, original_run_race, marker). Scheduling and deployment
remain with the controller. No library, precision, shape or timing changes.
"""
import copy
import hashlib
import json
import pathlib
import shutil


def _retain(root, path, history):
    path = path.resolve()
    relative = path.relative_to(root.resolve())
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    target = history / (digest.hexdigest() + '-' + path.name)
    if not target.exists():
        shutil.copy2(path, target)
    return {'original': str(relative), 'archive': str(target.relative_to(root)),
            'sha256': digest.hexdigest()}


def repair_neural(ctx, race, run, marker):
    requested = marker.get('arms_by_race', {}).get(race['id'])
    if requested is None:
        return run(ctx, race)
    if race['family'] != 'neural' or race.get('our_arms'):
        raise ValueError('Selective neural repair requires an opponents-only neural race')
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('Selective neural repair requires distinct selected arms')
    if not set(requested).issubset(race['opponents']):
        raise ValueError('Selected repair arm is outside the declared opponents')
    root = pathlib.Path(ctx['out'])
    board_path = root / 'board.json'
    board = json.loads(board_path.read_text())
    old = board.get('races', {}).get(race['id'])
    if not old:
        raise ValueError('Selective neural repair requires a retained prior race')
    # The board's host/GPU readback must agree before reusing its cells. Package
    # and source changes are retained separately, not stamped onto old cells.
    if not ctx.get('box') or any(board.get('box', {}).get(key) != ctx['box'].get(key)
                                 for key in ('host', 'gpu')):
        raise ValueError('Selective neural repair hardware does not match prior board')
    old_cells = {c['arm']: c for c in old.get('cells', [])}
    if not set(race['arms']).issubset(old_cells):
        raise ValueError('Prior race lacks declared arm cells')
    # Re-entering a completed repair must never retime its successful arms.
    selected = [a for a in requested if old_cells[a].get('status') != 'ok']
    if not selected:
        return copy.deepcopy(old)
    history = root / 'repair-history'
    history.mkdir(exist_ok=True)
    paths = {board_path}
    for field in ('log', 'race_json'):
        if old.get(field):
            path = root / old[field]
            if path.exists():
                paths.add(path)
                if field == 'race_json':
                    # Worker logs, saved outputs, inputs and older receipts all
                    # share this race stem; preserve them before any overwrite.
                    paths.update(p for p in path.parent.glob(path.stem + '*')
                                 if p.is_file() and (p.name.startswith(path.stem + '-')
                                                    or p.name.startswith(path.stem + '.')))
    retained = [_retain(root, path, history) for path in sorted(paths)]
    narrowed = dict(race, arms=selected, opponents=selected, our_arms={})
    # Force actual repair execution instead of loading an old store entry.
    fresh = run(dict(ctx, retime=True, with_opponents=True), narrowed)
    if {c.get('arm') for c in fresh.get('cells', [])} != set(selected):
        raise ValueError('Repair result does not contain exactly the selected arms')
    if any(c.get('arm') not in selected for c in fresh.get('infer_cells', [])):
        raise ValueError('Unexpected inference arm in selective repair result')
    result = copy.deepcopy(fresh)
    for field in ('cells', 'infer_cells'):
        result[field] = ([copy.deepcopy(c) for c in old.get(field, [])
                          if c.get('arm') not in selected]
                         + copy.deepcopy(fresh.get(field, [])))
    result['arms'] = list(race['arms'])
    result['started'] = old.get('started', fresh.get('started'))
    result['isolated_neural_repair'] = {
        'selection': copy.deepcopy(marker), 'executed_arms': selected,
        'preserved_arms': [a for a in race['arms'] if a not in selected],
        'preserved_from_finished': old.get('finished'),
        'preserved_evidence': retained,
        'fresh_receipt': copy.deepcopy(fresh),
        'cell_provenance': {
            a: {'receipt': 'fresh_receipt', 'source_sha': marker.get('source_sha')}
            if a in selected else {'receipt': 'preserved_evidence board.json',
                                   'race_id': race['id']}
            for a in race['arms']},
    }
    if (fresh.get('rc') != 0 or fresh.get('status') != 'done'
            or any(c.get('status') != 'ok' for field in ('cells', 'infer_cells')
                   for c in result.get(field, []))):
        result.update(status='failed', rc=fresh.get('rc') or 1)
    return result
