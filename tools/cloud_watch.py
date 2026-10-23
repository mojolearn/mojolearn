#!/usr/bin/env python3
"""Receive cloud inventory and existing collectors' evidence into one worktree.

Operational observer only: no job launches, repairs, leases, merges or deletions.
Provider disappearance is not successful completion. A copied log is not proof
of a complete backup. Existing collectors keep custody of full workload bytes.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import run_retention_watch as retention


def age(value, now):
    try:
        stamp = float(value) if isinstance(value, (float, int)) else dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        return max(0, now - stamp)
    except (TypeError, ValueError, AttributeError):
        return float('inf')


def source(path, root):
    """Retain complete JSON source bytes by hash before interpreting them."""
    receipt = {'path': str(path)}
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        dest = root / 'sources' / (digest + '.json')
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        receipt.update(sha256=digest, saved=str(dest))
        return json.loads(raw), receipt
    except (OSError, ValueError) as exc:
        return {}, {**receipt, 'error': str(exc)}


def mambik_row(instance, status, pulls, now):
    identity = instance['id']
    probe = status.get('health', {}).get(identity, {})
    errors = []
    fresh = age(probe.get('observed_at'), now) <= 600
    if not fresh:
        errors.append('Remote observation missing or older than 10 minutes')
    for kind in ('backup', 'lease'):
        check = status.get('checks', {}).get(kind + ':' + identity, {})
        if check.get('error'):
            # Keep the source message in evidence, use a stable category for alerts.
            errors.append(kind + ': ' + check['error'])
    if not probe.get('sync_alive'):
        errors.append('Backup loop is not confirmed running')
    sync_age = probe.get('sync_age_s')
    if sync_age is None or sync_age + age(probe.get('observed_at'), now) > 900:
        errors.append('Successful remote backup sync missing or older than 15 minutes')
    pull = pulls.get('pod-' + identity, {})
    if pull.get('rc') != 0 or age(pull.get('time'), now) > 1200:
        errors.append('Successful local evidence pull missing or older than 20 minutes')
    if probe.get('disk_free_gb', 100) < 10:
        errors.append('Remote disk has less than 10 GiB free')
    # LAST_SYNC changes routinely; result/exit markers deserve a notification.
    created = status.get('snapshots', {}).get('pods', {}).get(identity, {}).get('createdAt')
    def current_marker(marker):
        if Path(marker['path']).name == 'LAST_SYNC.txt':
            return False
        if any(p in Path(marker['path']).parts for p in ('main', 'repo', 'src', '.git', '.pixi')):
            return False
        stamps = re.findall(r'20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[.]\d+)?(?:Z|[+-]\d{2}:\d{2})', marker.get('value', ''))
        # An inherited checkout can contain DONE records from an earlier pod.
        return not (created and stamps and min(age(s, now) for s in stamps) > age(created, now))
    markers = [m for m in probe.get('markers', []) if current_marker(m)]
    for marker in markers:
        values = re.findall(r'\brc=(-?\d+)\b', marker.get('value', ''))
        if any(int(v) != 0 for v in values):
            errors.append('Failure marker: ' + marker['path'])
    return {'project': 'mambik', 'collector': 'mambik.running-watch',
            'observed_at': probe.get('observed_at'), 'coverage': 'observed' if fresh else 'stale',
            'compute_visible': bool(probe.get('processes')) if fresh else None,
            'processes': probe.get('processes', []), 'result_markers': markers,
            'backup': {'remote': probe.get('prefix'), 'last_sync': probe.get('last_sync'),
                       'sync_age_s_at_probe': sync_age, 'local_pull': pull,
                       'complete_retention_proven': False}, 'errors': sorted(set(errors))}


def mojolearn_row(instance, health, now, registration):
    name = registration['name']
    box = next((b for b in health.get('retention_boxes', []) if b['name'] == name), {})
    fresh = age(box.get('observed_at'), now) <= 1200
    errors = []
    if not fresh:
        errors.append('Remote collection missing or older than 20 minutes')
    if box.get('state') != 'copied':
        errors.append('Remote evidence copy is not confirmed successful')
    results = health.get('remote_job_results', {}).get(name, {})
    for stage, result in results.get('stages', {}).items():
        if result.get('exit_code'):
            errors.append(stage + ': exit ' + str(result['exit_code']))
    deadline = registration.get('lease_deadline_epoch')
    if deadline is not None and deadline - now < 1800:
        errors.append('Lease expires within 30 minutes or has expired; owner action required')
    return {'project': 'mojolearn', 'collector': 'mojolearn.run-watch',
            'observed_at': box.get('observed_at'), 'coverage': 'observed' if fresh else 'stale',
            'compute_visible': None, 'result_markers': results.get('stages', {}),
            'backup': {'box_receipt': box, 'r2_retained_batches': health.get('retained_batches'),
                       'r2_pending_batches': health.get('pending_batches'),
                       'complete_retention_proven': False}, 'errors': errors}


def assemble(providers, previous, mambik, pulls, mojo, registrations, now):
    instances, alerts = {}, []
    current_ids = {provider + '/' + p['id'] for provider, data in providers.items()
                   if data.get('state') == 'observed' for p in data.get('instances', [])}
    if any(p.get('name', '').startswith('mambik-') for p in providers.get('runpod', {}).get('instances', [])):
        if age(mambik.get('heartbeat'), now) > 360:
            alerts.append('Mambik collector heartbeat is missing or stale')
        for key, check in mambik.get('checks', {}).items():
            if ':' not in key and key not in ('delivery', 'aws_inventory_us_east_1') and check.get('error'):
                alerts.append('Mambik ' + key + ': ' + check['error'])
    if any(k in registrations for k in current_ids):
        if age(mojo.get('observed_at'), now) > 180:
            alerts.append('Mojolearn collector heartbeat is missing or stale')
        if mojo.get('pending_batches'):
            alerts.append('Mojolearn has R2 uploads pending')
        for alert in mojo.get('alerts', []):
            if alert.startswith('Evidence retention incomplete'):
                detail = mojo.get('retention_detail')
                # The original project collector flags other projects' pods as
                # unregistered. This account-wide view checks their own collector.
                if detail is not None and not detail.get('errors') and not detail.get('pending_batches'):
                    continue
            if any(term in alert for term in ('Evidence', 'retention', 'disk', 'Watcher service')) and not alert.startswith('Instance lacks'):
                alerts.append('Mojolearn: ' + alert)
    for provider, data in providers.items():
        if data.get('state') != 'observed':
            alerts.append(provider + ': inventory unavailable (' + data.get('state', 'unknown') + ')')
            # Preserve last-seen hosts during API failure without declaring deletion.
            for key, row in previous.get('instances', {}).items():
                if row['provider'] == provider:
                    instances[key] = {**row, 'inventory_confirmed': False}
            continue
        for item in data.get('instances', []):
            key = provider + '/' + item['id']
            if provider == 'runpod' and item.get('name', '').startswith('mambik-'):
                detail = mambik_row(item, mambik, pulls, now)
            elif key in registrations:
                detail = mojolearn_row(item, mojo, now, registrations[key])
            else:
                detail = {'project': 'unassigned', 'coverage': 'inventory_only',
                          'errors': ['No workload/evidence collector has registered this machine'],
                          'compute_visible': None, 'result_markers': {}}
            instances[key] = {**item, **detail, 'provider': provider, 'inventory_confirmed': True}
    for key, row in instances.items():
        alerts.extend(key + ': ' + error for error in row['errors'])
    return {'observed_at': retention.utc(), 'providers': providers, 'instances': instances,
            'alerts': sorted(set(alerts)), 'policy': 'Receive information only; no workload actions.'}


def meaningful(row):
    return {key: row.get(key) for key in ('status', 'inventory_confirmed', 'coverage',
                                         'compute_visible', 'result_markers', 'errors')}


def changes(previous, current):
    if not previous:
        return [{'kind': 'WATCH_READY', 'detail': str(len(current['instances'])) + ' machines discovered'}]
    events = []
    before, after = previous.get('instances', {}), current['instances']
    for key, row in after.items():
        if key not in before:
            events.append({'kind': 'MACHINE_APPEARED', 'id': key, 'name': row['name']})
        elif meaningful(row) != meaningful(before[key]):
            events.append({'kind': 'MACHINE_CHANGED', 'id': key, 'name': row['name'],
                           'before': meaningful(before[key]), 'after': meaningful(row)})
    for key in before.keys() - after.keys():
        events.append({'kind': 'MACHINE_NO_LONGER_LISTED', 'id': key, 'name': before[key]['name'],
                       'detail': 'Provider no longer lists it; completion and final retention are not inferred.'})
    if current['alerts'] != previous.get('alerts', []):
        events.append({'kind': 'ALERTS_CHANGED', 'added': sorted(set(current['alerts']) - set(previous.get('alerts', []))),
                       'cleared': sorted(set(previous.get('alerts', [])) - set(current['alerts']))})
    return events


def deliver(config, root, outbox, now):
    if not config.get('notifications_enabled', False):
        return
    if not outbox.get('pending') or now < outbox.get('retry_after', 0):
        return
    pending = outbox['pending']
    labels = [e['kind'] + (' ' + e.get('name', e.get('id', e.get('detail', '')))) for e in pending]
    message = ('Cloud watch observation only, event batch ' + pending[0]['event_id'] + '. '
               + '; '.join(labels[:8]) + (f'; {len(labels)-8} more saved events' if len(labels) > 8 else '')
               + '. Read ' + str(root / 'status.json') + ' and briefly report material changes to Andrew here. '
               'This is information, not a request for repairs or new work. Do not launch, stop, retry, '
               'renew, merge, or message other conversations. Keep complete logs on disk; report evidence paths.')
    receipt = {'at': retention.utc(), 'thread': config['thread'],
               'event_ids': [e['event_id'] for e in pending]}
    try:
        with (root / 'logs/delivery.log').open('ab') as log:
            p = subprocess.run([config['codex'], 'queue', '--remote', config['remote'],
                                '--thread', config['thread'], '--message', message],
                               stdout=log, stderr=subprocess.STDOUT, timeout=45)
        receipt['exit_code'] = p.returncode
        if p.returncode == 0:
            # Transport acknowledgement, not proof the user has read the update.
            receipt['state'] = 'accepted'
            outbox['pending'] = []
            outbox['last_accepted_at'] = retention.utc()
        else:
            receipt['state'] = 'pending_retry'
    except (OSError, subprocess.TimeoutExpired) as exc:
        receipt.update(state='pending_retry', error=str(exc))
    outbox['retry_after'] = now + 120
    outbox['last_delivery'] = receipt
    with (root / 'deliveries.jsonl').open('a') as f:
        f.write(json.dumps(receipt) + '\n')
    retention.save(root / 'delivery.json', receipt)


def observe(config, root, notify=True):
    now = time.time()
    previous = retention.read(root / 'status.json', {})
    receipts = {}
    values = {}
    for name, path in config['sources'].items():
        values[name], receipts[name] = source(Path(path), root)
    if 'mojolearn_retention' in values and not receipts['mojolearn_retention'].get('error'):
        values['mojolearn']['retention_detail'] = values['mojolearn_retention']
    try:
        registrations = {r['provider'] + '/' + r['id']: r for r in retention.boxes(Path(config['mojolearn_evidence']))}
        registration_error = None
    except (OSError, ValueError, KeyError) as exc:
        registrations, registration_error = {}, str(exc)
    status = assemble(retention.provider_census(), previous, values['mambik'], values['mambik_pulls'],
                      values['mojolearn'], registrations, now)
    status['sources'] = receipts
    status['destination_thread'] = config['thread']
    status['notifications_enabled'] = config.get('notifications_enabled', False)
    status['worktree'] = str(root.parent)
    backup = retention.read(root / 'archive.json', {})
    status['watch_archive'] = backup
    if not backup or age(backup.get('observed_at'), now) > 1200:
        status['alerts'].append('Watch record R2 archive has not completed within 20 minutes')
    if backup.get('pending'):
        status['alerts'].append('Watch record R2 upload pending; local bytes retained for retry')
    if registration_error:
        status['alerts'].append('Registration read failed: ' + registration_error)
    for key, receipt in receipts.items():
        if receipt.get('error'):
            status['alerts'].append(key + ': source read failed: ' + receipt['error'])
    status['alerts'].sort()
    outbox = retention.read(root / 'outbox.json', {'pending': []})
    events = [{**e, 'event_id': uuid.uuid4().hex, 'at': status['observed_at']} for e in changes(previous, status)]
    if events:
        with (root / 'events.jsonl').open('a') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
        outbox['pending'].extend(events)
    # Durable queue first, then observation checkpoint. A crash may duplicate an
    # event but cannot advance the checkpoint and silently lose its notification.
    retention.save(root / 'outbox.json', outbox)
    snapshot = root / 'snapshots' / (str(time.time_ns()) + '.json')
    retention.save(snapshot, status)
    retention.save(root / 'status.json', status)
    lines = ['Cloud watch — ' + status['observed_at'],
             'Passive failsafe: observe and save only. Automatic notifications: ' + str(status['notifications_enabled']), '']
    for provider, data in status['providers'].items():
        lines.append(provider + ': ' + data['state'] + ', ' + str(len(data.get('instances', []))) + ' machines')
    lines += ['', 'Machines:']
    lines += [f'{r["provider"]}/{r["id"]} {r["name"]}: {r["status"]}; evidence {r["coverage"]}; {len(r["errors"])} issues'
              for r in status['instances'].values()]
    lines += ['', 'Attention:'] + (status['alerts'] or ['None observed'])
    lines += ['', 'Full workload files remain with the collectors listed in status.json.',
              'Machine removal, a process exit, or a copied log never proves whole-job success or complete retention.']
    temporary = root / 'STATUS.txt.tmp'; temporary.write_text('\n'.join(lines) + '\n'); temporary.replace(root / 'STATUS.txt')
    if notify:
        deliver(config, root, outbox, now)
        retention.save(root / 'outbox.json', outbox)
    return status


def archive(root):
    index = retention.read(root / 'archive-index.json', {})
    paths = [p for folder in ('sources', 'snapshots', 'logs') for p in (root / folder).glob('*') if p.is_file()]
    paths += [p for p in root.glob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.txt') and p.name != 'archive-index.json']
    paths += [Path(__file__), Path(retention.__file__)]
    pending = [p for p in paths if index.get(str(p)) != retention.signature(p)]
    if pending:
        retention.make_batch(root, pending, index)
        retention.save(root / 'archive-index.json', index)
    retention.upload_pending(root)
    receipts = [retention.read(p) for p in (root / 'batches').glob('*/receipt.json')]
    result = {'observed_at': retention.utc(), 'retained': sum(r['state'] == 'retained' for r in receipts),
              'pending': sum(r['state'] != 'retained' for r in receipts),
              'snapshot_errors': sum(len(r.get('errors', [])) for r in receipts)}
    retention.save(root / 'archive.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('observe', 'archive'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--no-notify', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    config = retention.read(args.config)
    root = Path(config['state'])
    for folder in ('logs', 'snapshots', 'sources'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    with (root / (args.mode + '.lock')).open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        value = observe(config, root, not args.no_notify) if args.mode == 'observe' else archive(root)
        print(json.dumps({'mode': args.mode, 'at': retention.utc(),
                          'machines': len(value.get('instances', {})), 'alerts': len(value.get('alerts', [])),
                          'archive_pending': value.get('pending')}), flush=True)


if __name__ == '__main__':
    main()
