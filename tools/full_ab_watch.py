#!/usr/bin/env python3
"""Watch frozen full-workload queues and notify their owning Codex thread.

This observer never launches a benchmark or a build. Lane controllers own their
serial workers. Repair notifications explicitly preserve completed attempts.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def tick(config, state):
    previous = json.loads((state / 'status.json').read_text()) if (state / 'status.json').exists() else {}
    lanes = []
    for lane in config['lanes']:
        row = {'name': lane['name'], 'evidence': lane['status_path']}
        try:
            path = Path(lane['status_path'])
            if lane.get('fetch_status_argv'):
                capture = state / (lane['name'] + '-status-capture.json')
                with capture.open('w') as out, (state / (lane['name'] + '-fetch.log')).open('a') as log:
                    fetched = subprocess.run(lane['fetch_status_argv'], stdout=out, stderr=log, timeout=45)
                if fetched.returncode:
                    raise OSError('Remote status fetch failed, exit=' + str(fetched.returncode))
                atomic(path, json.loads(capture.read_text()))
            value = json.loads(path.read_text())
            row['status'] = value.get('status', value.get('phase', 'UNKNOWN'))
            row['detail'] = value
            if lane.get('probe_argv'):
                with (state / (lane['name'] + '-probe.log')).open('a') as log:
                    result = subprocess.run(lane['probe_argv'], stdout=log, stderr=subprocess.STDOUT, timeout=45)
                row['probe_returncode'] = result.returncode
                if result.returncode:
                    row['status'] = 'PROBE_FAILED'
            if row['status'] in ('RUNNING', 'MEASURING'):
                heartbeat = None
                for field in ('heartbeat_at', 'updated_at', 'updated'):
                    candidate = value.get(field)
                    if (isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
                            and math.isfinite(candidate)):
                        heartbeat = candidate
                        row['heartbeat_source'] = field
                        break
                if heartbeat is None and not lane.get('fetch_status_argv'):
                    # A local controller updates this original file itself.
                    # A fetched copy's mtime only proves the observer is alive.
                    heartbeat = path.stat().st_mtime
                    row['heartbeat_source'] = 'local_file_mtime'
                if heartbeat is None:
                    row['status'] = 'HEARTBEAT_MISSING'
                    row['error'] = 'Active remote status has no finite numeric heartbeat_at, updated_at or updated'
                else:
                    row['heartbeat_at'] = heartbeat
                    if time.time() - heartbeat > lane.get('heartbeat_timeout_seconds', 300):
                        row['status'] = 'HEARTBEAT_STALE'
            # Ignore timestamps and counters for alerts; phase/failure changes
            # are enough. A quiet, healthy long workload is not a failed run.
            row['alert_key'] = hashlib.sha256(json.dumps({
                'status': row['status'], 'error': value.get('error'),
                'observer_error': row.get('error'), 'reason': value.get('reason'),
                'errors': value.get('errors'), 'freeze_error': value.get('freeze_error'),
                'failed': value.get('failed'), 'blockers': value.get('blockers'),
                'freeze': value.get('source_sha'),
            }, sort_keys=True).encode()).hexdigest()
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            row.update(status='STATUS_READ_FAILED', error=str(exc))
            row['alert_key'] = hashlib.sha256(str(exc).encode()).hexdigest()
        lanes.append(row)
    notified = previous.get('notified', {})
    if config.get('notifications_enabled', True):
        changed = [row for row in lanes if notified.get(row['name']) != row['alert_key']]
        if changed:
            message = ('Full dataset A/B campaign update: ' + ', '.join(
                row['name'] + '=' + row['status'] for row in changed) + '. Inspect ' + str(state / 'status.json') +
                '. Keep the recorded main freeze; run IDENTICAL on NVIDIA/AMD and Apple FAST independently. '
                'Repair actual failures with scoped subagents and resume only affected cells; preserve original evidence. '
                'Do not compile or rerun verification; never substitute component fixtures for full workloads. '
                'Keep logs out of context: save complete output to files, use targeted rg/grep with bounded '
                'surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. '
                'Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.')
            with (state / 'notifications.log').open('a') as log:
                result = subprocess.run([config['codex'], 'queue', '--thread', config['thread'], '--message', message],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=45)
            if result.returncode == 0:
                notified.update({row['name']: row['alert_key'] for row in changed})
            atomic(state / 'notification.json', {'at': time.time(), 'returncode': result.returncode})
    atomic(state / 'status.json', {'status': 'WATCHING', 'pid': os.getpid(), 'at': time.time(),
                                  'lanes': lanes, 'notified': notified})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    state = Path(config['state'])
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (state / 'watch.pid').write_text(str(os.getpid()) + '\n')
        while not (state / 'STOP').exists():
            try:
                tick(json.loads(args.config.read_text()), state)
            except Exception as exc:
                atomic(state / 'error.json', {'at': time.time(), 'error': repr(exc)})
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(config.get('interval_seconds', 60))


if __name__ == '__main__':
    main()
