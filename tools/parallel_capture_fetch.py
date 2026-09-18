#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retain bounded periodic snapshots while a separately guarded GPU leg runs.

Read-only: no provisioning, deletion or lease changes. Copies only the capture
folder and public leg/device metadata, never credentials. Every accepted tarball
must name the expected numerical source commit. Completed prior snapshots remain
available after connection loss or provider-side disappearance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

REMOTE = 'cd /root/gemm_leg_out && test -d parallel-cv && tar -czf - --ignore-failed-read leg.txt gpu.txt parallel-cv'


def validate(path, expected_commit):
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.name.startswith('/') or '..' in Path(member.name).parts or member.issym() or member.islnk():
                raise ValueError('unsafe capture archive member')
            if not (member.name in ('leg.txt', 'gpu.txt', 'parallel-cv') or member.name.startswith('parallel-cv/')):
                raise ValueError('unexpected capture archive member')
        try:
            metadata = archive.extractfile('leg.txt')
        except KeyError as exc:
            raise ValueError('capture archive has no source metadata') from exc
        if metadata is None:
            raise ValueError('capture source metadata is not a regular file')
        content = metadata.read().decode('utf-8')
        commits = [line[7:] for line in content.splitlines() if line.startswith('commit=')]
        if commits != [expected_commit]:
            raise ValueError('capture numerical source differs from expected commit')


def snapshot(host, port, out, expected_commit, attempt, timeout=20):
    destination = out / f'snapshot-{attempt:04d}.tar.gz'
    partial = destination.with_suffix('.partial')
    record = {'attempt': attempt, 'time_utc_epoch': time.time(), 'status': 'FAILED'}
    try:
        with partial.open('wb') as stream:
            proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
                                   '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
                                   '-o', 'ServerAliveCountMax=2', '-p', str(port), 'root@' + host, REMOTE],
                                  stdout=stream, stderr=subprocess.PIPE, timeout=timeout, check=False)
        (out / f'snapshot-{attempt:04d}.stderr').write_bytes(proc.stderr)
        record['returncode'] = proc.returncode
        if proc.returncode:
            return record
        if partial.stat().st_size > 128 * 1024 * 1024:
            raise ValueError('capture snapshot exceeds 128 MiB')
        validate(partial, expected_commit)
        partial.replace(destination)
        record.update(status='RETAINED', path=destination.name, bytes=destination.stat().st_size,
                      sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        return record
    except (OSError, subprocess.TimeoutExpired, ValueError, tarfile.TarError) as exc:
        record['error'] = str(exc)
        return record
    finally:
        partial.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=3300)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.:-]*', args.host) or not 1 <= args.port <= 65535:
        parser.error('invalid SSH host or port')
    if not re.fullmatch('[0-9a-f]{40}', args.commit):
        parser.error('full source commit is required')
    if not 1 <= args.seconds <= 3600 or not 1 <= args.interval <= 60:
        parser.error('seconds must be 1..3600 and interval 1..60')
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'expected_commit': args.commit, 'host': args.host, 'port': args.port,
                'scope': 'read-only intermediate snapshots; not qualification', 'snapshots': []}
    deadline, attempt = time.monotonic() + args.seconds, 0
    while time.monotonic() < deadline:
        if args.stop_file and args.stop_file.exists():
            break
        attempt += 1
        record = snapshot(args.host, args.port, args.out, args.commit, attempt,
                          timeout=min(20, max(1, deadline - time.monotonic())))
        manifest['snapshots'].append(record)
        temporary = args.out / 'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(args.out / 'manifest.json')
        if args.once:
            break
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    return 0 if any(r['status'] == 'RETAINED' for r in manifest['snapshots']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
