#!/usr/bin/env python3
"""Watch frozen full-workload queues and notify their owning Codex thread.

This observer never launches a benchmark or a build. Lane controllers own their
serial workers. Repair notifications explicitly preserve completed attempts.
"""
import argparse
import fcntl
import hashlib
import json
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
                heartbeat = value.get('heartbeat_at', path.stat().st_mtime)
                if time.time() - heartbeat > lane.get('heartbeat_timeout_seconds', 300):
                    row['status'] = 'HEARTBEAT_STALE'
            # Ignore timestamps and counters for alerts; phase/failure changes
            # are enough. A quiet, healthy long workload is not a failed run.
            row['alert_key'] = hashlib.sha256(json.dumps({
                'status': row['status'], 'error': value.get('error'),
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
                '. Keep the recorded main freeze; NVIDIA/AMD/Apple IDENTICAL, M3 FAST only after IDENTICAL. '
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
