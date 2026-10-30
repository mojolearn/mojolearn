"""Explicit finite local retention queue; no polling, watcher or qualification."""
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def checked_archive(record):
    data = Path(record['archive']).read_bytes()
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('queued archive bytes changed; offsite retention not established')
    return data


def defer_archive(base, out, archive, raw, key, source, chunk):
    base, out, archive = Path(base), Path(out), Path(archive)
    sha = hashlib.sha256(raw).hexdigest()
    durable = archive.with_name(archive.stem + '.' + sha + '.gz')
    if durable.exists():
        if durable.read_bytes() != raw:
            raise ValueError('immutable local archive conflict')
    else:
        with durable.open('xb') as stream:
            stream.write(raw)
    record = dict(schema='mojolearn.finite-offsite-queue.v1', status='pending',
        archive=str(durable), sha256=sha, bytes=len(raw), key=key,
        source_revision=source, chunk=chunk, out=str(out), qualification_claim=False,
        partial='.partial.' in archive.name)
    queue = base / 'offsite-queue' / (sha + '.json')
    if not queue.exists():
        save(queue, record)
    save(out / 'r2-pending.json', dict(record, queue=str(queue)))
    return queue


def pending_count(base):
    return sum(json.loads(p.read_text()).get('status') != 'retained'
               for p in (Path(base) / 'offsite-queue').glob('*.json'))


def upload_one(path, request, timeout=1200):
    path = Path(path)
    record = json.loads(path.read_text())
    raw = checked_archive(record)
    if record.get('status') == 'retained':
        return 'already-retained'
    try:
        status, got = request('GET', record['key'], timeout=timeout)
    except HTTPError as error:
        if error.code != 404:
            raise
        status, got = 404, b''
    if status == 404:
        status, _ = request('PUT', record['key'], data=raw, timeout=timeout)
        if status not in (200, 201):
            raise RuntimeError('R2 PUT ' + str(status))
        status, got = request('GET', record['key'], timeout=timeout)
    if status != 200 or hashlib.sha256(got).hexdigest() != record['sha256']:
        raise RuntimeError('R2 readback did not establish exact retained bytes')
    receipt = {k: record[k] for k in ('key', 'sha256', 'bytes', 'archive')}
    receipt.update(readback_verified=True, finite_queue=str(path),
                   partial=record.get('partial', '.partial.' in Path(record['archive']).name))
    save(Path(record['out']) / 'r2-retention.json', receipt)
    record.update(status='retained', readback_verified=True)
    save(path, record)
    return 'retained'
