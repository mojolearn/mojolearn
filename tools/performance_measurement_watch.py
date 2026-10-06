#!/usr/bin/env python3
"""Collect normalized captured results and refresh the candidate boards.

This watcher does no device work and never changes a runtime default. Failed
source reads retain the last good source snapshot. A new snapshot is admitted
only after the board tool validates its pairs and provenance.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from performance_measurement_board import build, read, write


def atomic(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tick(config, state):
    index = read(config['index'])
    inventory = read(config['inventory'])
    cache = read(state / 'sources.json') if (state / 'sources.json').exists() else {}
    errors = []
    for source in config['sources']:
        path = source['path']
        if not Path(path).exists():
            errors.append({'path': path, 'error': 'Configured source is not present; retaining prior snapshot'})
            continue
        try:
            raw = Path(path).read_bytes()
            data = json.loads(raw)
            rows = [dict(row, producer=path) for row in data['rows']]
            build(inventory, {'cells': rows})
            digest = hashlib.sha256(raw).hexdigest()
            snapshot = state / ('snapshot-' + digest + '.json')
            if not snapshot.exists():
                snapshot.write_bytes(raw)
            cache[path] = {'rows': rows, 'sha256': digest, 'snapshot': str(snapshot)}
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append({'path': path, 'error': str(exc)})
    # A source is replaced as one unit: stale attempts cannot silently disappear
    # from its own retained snapshot even when a producer publishes a retry.
    managed_paths = {source['path'] for source in config['sources']}
    unmanaged = [cell for cell in index.get('cells', []) if cell.get('producer') not in managed_paths]
    index['cells'] = unmanaged + [cell for item in cache.values() for cell in item['rows']]
    index['capture_sources'] = {path: {k: v for k, v in item.items() if k != 'rows'} for path, item in cache.items()}
    board = build(inventory, index)
    atomic(config['index'], index)
    write(board, config['out'])
    atomic(state / 'sources.json', cache)
    digest = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    prior = read(state / 'status.json') if (state / 'status.json').exists() else {}
    last_notified = prior.get('last_notified', 0)
    changed = digest != prior.get('index_sha256')
    if (changed or errors) and time.time() - last_notified > config.get('notify_seconds', 600):
        note = ('Candidate board watcher: captured measurements updated. Inspect ' + str(state / 'status.json')
                + ' and ' + config['index'] + '. Update boards and inline winners/losers beside toggles; '
                'flip only sufficiently measured defaults. No compilation/identity retests. Repair real '
                'measurement failures with scoped subagents. Preserve active cloud owners and 30-minute idle policy. '
                'Keep logs out of context: use bounded grep and retained structured summaries.')
        result = subprocess.run([config['codex'], 'queue', '--thread', config['thread'], '--message', note],
                                capture_output=True, text=True, timeout=30)
        atomic(state / 'notification.json', {'at': time.time(), 'exit_code': result.returncode,
                                           'stderr': result.stderr[-1000:]})
        if result.returncode == 0:
            last_notified = time.time()
    atomic(state / 'status.json', {'status': 'WATCHING' if not errors else 'SOURCE_READ_ERRORS',
                                  'pid': os.getpid(), 'at': time.time(), 'cells': len(index['cells']),
                                  'index_sha256': digest, 'last_notified': last_notified, 'errors': errors})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    state = Path(config['state'])
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (state / 'watch.pid').write_text(str(os.getpid()))
        while not (state / 'STOP').exists():
            try:
                tick(config, state)
            except Exception as exc:
                atomic(state / 'error.json', {'at': time.time(), 'error': repr(exc)})
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(config.get('interval_seconds', 60))


if __name__ == '__main__':
    main()
