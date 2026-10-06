"""Run an isolated tree opponent once; preserve other current-hardware cells."""
import copy
import hashlib
import json
import pathlib


def repair_tree(ctx, race, run, marker):
    arm = marker['arm']
    if race['family'] != 'trees' or arm not in race['opponents']:
        return run(ctx, race)
    board = pathlib.Path(ctx['out']) / 'board.json'
    previous = json.loads(board.read_text()).get('races', {}).get(race['id']) if board.exists() else None
    others = [a for a in race['opponents'] if a != arm]
    records = []
    if previous:
        # Main board scheduling decides whether this race needs repair. Its
        # previous immutable record is retained under board.superseded.
        records.append(copy.deepcopy(previous))
    elif others:
        sub = dict(race, opponents=others, arms=others, our_arms={})
        records.append(run(ctx, sub))
    preserved = {}
    if records:
        history = pathlib.Path(ctx['out']) / 'repair-history'
        history.mkdir(exist_ok=True)
        raw = json.dumps(records[0], sort_keys=True).encode()
        receipt = history / (hashlib.sha256(raw).hexdigest() + '.json')
        receipt.write_bytes(raw)
        preserved['receipt'] = str(receipt.relative_to(ctx['out']))
        if records[0].get('log'):
            log = pathlib.Path(ctx['out']) / records[0]['log']
            if log.exists():
                data = log.read_bytes()
                target = history / (hashlib.sha256(data).hexdigest() + '.log')
                target.write_bytes(data)
                preserved['log'] = str(target.relative_to(ctx['out']))
    narrowed = dict(race, opponents=[arm], arms=[arm], our_arms={})
    # LightGBM's Booster.predict runs on CPU even after CUDA fitting.
    # Do not launch its CPU inference as a GPU measurement.
    isolated = dict(ctx, python=marker['python'], infer=False)
    fresh = run(isolated, narrowed)
    if ctx.get('infer'):
        template = next((c for c in fresh.get('cells', []) if c.get('arm') == arm), {})
        fresh['infer_cells'] = []
        for batch in ('test', 'large'):
            cell = copy.deepcopy(template)
            cell.update(arm=arm, batch=batch, phase='inference',
                        status='REFUSED(GPU-INFERENCE-NOT-SUPPORTED: LightGBM Booster.predict executes on CPU)',
                        rounds=0, times_ms=[], median_ms=None, min_ms=None, max_ms=None,
                        warmup_ms=None, peak_gpu_mb=None, peak_host_mb=None, memory={}, call='Booster.predict (CPU; not executed)', quality={},
                        ratio_ours_fast_over=None, ratio_ours_identical_over=None)
            fresh['infer_cells'].append(cell)
    result = copy.deepcopy(fresh)
    old = records[0] if records else {}
    for field in ('cells', 'infer_cells'):
        result[field] = ([copy.deepcopy(c) for c in old.get(field, []) if c.get('arm') != arm]
                         + [copy.deepcopy(c) for c in fresh.get(field, []) if c.get('arm') == arm])
    result['arms'] = list(race['arms'])
    result['started'] = old.get('started', fresh.get('started'))
    result['isolated_opponent_repair'] = {
        'arm': arm, 'environment': copy.deepcopy(marker),
        'preserved_from_finished': old.get('finished'),
        'preserved_arms': sorted({c['arm'] for c in old.get('cells', []) if c.get('arm') != arm}),
        'executed_arms': [arm] if previous else others + [arm],
        'prior_record_reused': bool(previous),
        'preserved_evidence': preserved,
        'subreceipts': records + [copy.deepcopy(fresh)] if not previous else [copy.deepcopy(fresh)],
    }
    # A successful replacement must not erase another arm's refusal/failure.
    def failed(cell):
        status = str(cell.get('status', ''))
        return status != 'ok' and not status.startswith('REFUSED(GPU-PATH-ONLY')
    remaining_failure = any(failed(c) for f in ('cells', 'infer_cells')
                            for c in result.get(f, []) if c.get('arm') != arm)
    if remaining_failure:
        result.update(status='failed', rc=1)
    return result
