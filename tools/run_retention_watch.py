#!/usr/bin/env python3
"""Observe jobs and retain evidence; never launch, stop, or retry their work.

Operational tooling only. Install launchd jobs: observe and health every 60
seconds, retain every 300 seconds. Evidence batches are immutable locally and in R2.
No eviction, remote deletion, file-size cutoff, or success inferred from a PID.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid
import urllib.request
import zipfile


EXCLUDE = {'.git', '.pixi', '.venv', 'venv', 'finish-venv', '__pycache__',
           'node_modules', 'cache', '.cache', 'deadman'}
PRIVATE_NAMES = {'curlrc', '.env', '.netrc', 'credentials', 'config.credentials'}
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
       '-o', 'StrictHostKeyChecking=accept-new', '-o', 'LogLevel=ERROR']
GH_FIELDS = 'databaseId,name,headSha,status,conclusion,createdAt,url'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def signature(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino]


def shell_state(path):
    # State files are data: do not execute shell substitutions or source them.
    result = {}
    for line in path.read_text().splitlines():
        if re.match(r'^[A-Za-z_][A-Za-z_0-9]*=', line):
            key, value = line.split('=', 1)
            words = shlex.split(value)
            result[key] = ' '.join(words)
    return result


def boxes(evidence):
    result = []
    states = list((evidence / 'devpods').glob('*/state.env'))
    amd = evidence / 'do-amd-steward/state.env'
    if amd.exists():
        states.append(amd)
    for path in states:
        if '.down' in path.parent.name:
            continue
        state = shell_state(path)
        target = state.get('SSH_TARGET')
        if path == amd:
            target = 'root@' + state['IP']
        if not target:
            continue
        identity = state.get('POD_ID', state.get('DROPLET_ID', 'unknown'))
        name = re.sub(r'[^a-zA-Z0-9_.-]', '_', path.parent.name + '-' + identity)
        record = {'name': name, 'ssh': shlex.split(target),
                       'state_file': str(path), 'provider': state.get('PROVIDER', 'digitalocean'),
                       'id': identity, 'retention_only': state.get('RETENTION_ONLY') == '1'}
        lease = path.parent / 'lease' / (identity + '.lease')
        deadline = path.parent / 'mac_deadline'
        if lease.exists():
            value = shell_state(lease).get('deadline_epoch', '')
            if value.isdigit():
                record['lease_deadline_epoch'] = int(value)
                record['lease_deadline_source'] = str(lease)
        elif deadline.exists() and deadline.read_text().strip().isdigit():
            # The DO Mac guard trails the on-box deadline by 600 seconds.
            # Record this as an estimate, not a verified remote deadline.
            record['lease_deadline_epoch'] = int(deadline.read_text().strip()) - 600
            record['lease_deadline_source'] = str(deadline) + ' minus 600s (estimate)'
        result.append(record)
    return result


def run(argv, log, timeout=120, input_text=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as output:
        output.write(('\n' + utc() + ' ' + shlex.join(argv[:2]) + '\n').encode())
        try:
            p = subprocess.run(argv, input=input_text, text=True, stdout=output,
                               stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        output.write(('\nexit_code=' + str(rc) + '\n').encode())
    return rc


def capture(argv, timeout=30, input_text=None):
    p = subprocess.run(argv, input=input_text, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{argv[0]} exit={p.returncode}: {p.stderr[-500:]}')
    return p.stdout


def provider_census():
    """Read accounts independently of registrations; absent state is not no boxes."""
    result = {}
    sources = [
        ('runpod', '.mojolearn_runpod_key', 'Bearer', 'https://api.runpod.io/graphql'),
        ('digitalocean', '.mojolearn_do_token', 'Bearer',
         'https://api.digitalocean.com/v2/droplets?per_page=200'),
        ('hotaisle', '.mojolearn_hotaisle_key', 'Token',
         'https://admin.hotaisle.app/api/teams/andrews-team/virtual_machines/'),
        ('hotaisle-bare-metal', '.mojolearn_hotaisle_key', 'Token',
         'https://admin.hotaisle.app/api/teams/andrews-team/bare_metal/'),
    ]
    for name, keyfile, scheme, url in sources:
        keypath = Path.home() / keyfile
        if not keypath.exists():
            result[name] = {'state': 'not_configured'}
            continue
        try:
            headers = {'Authorization': scheme + ' ' + keypath.read_text().strip()}
            payload = None
            if name == 'runpod':
                headers['Content-Type'] = 'application/json'
                payload = json.dumps({'query': '{ myself { pods { id name desiredStatus } } }'}).encode()
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.load(response)
            if name == 'runpod':
                rows = data['data']['myself']['pods']
            elif name == 'digitalocean':
                if data.get('links', {}).get('pages', {}).get('next'):
                    raise RuntimeError('Provider inventory is paginated; coverage incomplete')
                rows = data['droplets']
            elif isinstance(data, list):
                rows = data
            else:
                if data.get('next'):
                    raise RuntimeError('Provider inventory is paginated; coverage incomplete')
                rows = data['results']
            result[name] = {'state': 'observed', 'instances': [
                {'id': str(r['id']), 'name': r.get('name', ''),
                 'status': r.get('desiredStatus', r.get('status'))} for r in rows]}
        except Exception as exc:
            result[name] = {'state': 'error', 'error': str(exc)}
    return result


def github_observation(previous):
    # Enumerate active runs separately: busy CI can push a long run outside
    # the latest 100. Follow known IDs through completion before discarding.
    base = ['gh', '-R', 'mojolearn/mojolearn', 'run', 'list', '--json', GH_FIELDS]
    rows = json.loads(capture(base + ['--limit', '100']))
    by_id = {r['databaseId']: r for r in rows}
    for state in ('queued', 'in_progress', 'waiting', 'pending', 'requested'):
        active = json.loads(capture(base + ['--status', state, '--limit', '1000']))
        if len(active) >= 1000:
            raise RuntimeError('Active GitHub inventory reached limit; coverage incomplete')
        by_id.update((r['databaseId'], r) for r in active)
    for row in previous.get('github', []):
        key = row['databaseId']
        if key not in by_id:
            if row['status'] != 'completed':
                row = json.loads(capture(['gh', '-R', 'mojolearn/mojolearn', 'run',
                                         'view', str(key), '--json', GH_FIELDS]))
            # Retain discovered runs so collection cannot miss a completion
            # between observation and its next five-minute retention tick.
            by_id[key] = row
    return sorted(by_id.values(), key=lambda r: r['databaseId'], reverse=True)


def local_observation(repo, evidence):
    lines = capture(['/bin/ps', '-axo', 'pid=,ppid=,lstart=,command=']).splitlines()
    rows = {}
    for line in lines:
        parts = line.strip().split(None, 7)
        if len(parts) == 8:
            rows[int(parts[0])] = {'pid': int(parts[0]), 'ppid': int(parts[1]),
                                  'start': ' '.join(parts[2:7]), 'command': parts[7]}
    selected = {pid for pid, r in rows.items() if 'mojolearn' in r['command']
                and 'run_retention_watch.py' not in r['command']
                and 'modular-crashpad-handler' not in r['command']}
    # A command such as `pixi run ...` may have no project name in argv.
    warnings = []
    try:
        cwd_list = subprocess.run(['/usr/sbin/lsof', '-d', 'cwd', '-Fpn'],
                                  capture_output=True, text=True, timeout=15)
        cwd_text = cwd_list.stdout
        if cwd_list.returncode:
            warnings.append('lsof cwd discovery incomplete: exit ' + str(cwd_list.returncode))
    except subprocess.TimeoutExpired:
        cwd_text = ''
        warnings.append('lsof cwd discovery timed out; argv discovery still recorded')
    pid = None
    for line in cwd_text.splitlines():
        if line.startswith('p'):
            pid = int(line[1:])
        elif line.startswith('n') and 'mojolearn' in line and pid in rows:
            if 'run_retention_watch.py' not in rows[pid]['command'] and 'crashpad' not in rows[pid]['command']:
                selected.add(pid)
    while True:
        children = {pid for pid, r in rows.items() if r['ppid'] in selected}
        if children <= selected:
            break
        selected |= children
    files = []
    if selected:
        try:
            p = subprocess.run(['/usr/sbin/lsof', '-a', '-p', ','.join(map(str, selected)),
                                '-d', 'cwd,1,2', '-Fpnft'], capture_output=True, text=True, timeout=15)
            fd_text = p.stdout
        except subprocess.TimeoutExpired:
            fd_text = ''
            warnings.append('lsof file discovery timed out; previous open logs retained')
        pid = fd = kind = None
        for line in fd_text.splitlines():
            if line.startswith('p'):
                pid = int(line[1:])
            elif line.startswith('f'):
                fd = line[1:]
            elif line.startswith('t'):
                kind = line[1:]
            elif line.startswith('n') and pid in rows:
                path = line[1:]
                rows[pid].setdefault('fds', {})[fd] = {'path': path, 'type': kind}
                if fd in ('1', '2') and kind == 'REG' and path.startswith('/'):
                    files.append(path)
    return {'processes': [rows[pid] for pid in sorted(selected)], 'open_logs': sorted(set(files)),
            'warnings': warnings,
            'free_disk_bytes': shutil.disk_usage(evidence).free,
            'launchd': [x for x in capture(['launchctl', 'list']).splitlines() if 'mojolearn' in x],
            'repo_head': capture(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip()}


REMOTE = r'''
import glob, json, os, pathlib, subprocess
paths = set()
for pattern in ('/root/out', '/root/leg_out', '/root/lq', '/root/live-*', '/root/measure-*', '/root/race-*',
                '/root/results*', '/root/mojolearn-evidence', '/root/ref-*', '/root/*-fixtures*',
                '/root/ols-*', '/root/*.log', '/root/*.nohup', '/root/gpu-queue/dispatcher.log',
                '/root/mojolearn/python/mojolearn/*/*.so', '/root/mojolearn/python/mojolearn/*.so'):
    for p in glob.glob(pattern):
        if not os.path.islink(p): paths.add(p)
ps = subprocess.check_output(['ps','-eo','pid=,ppid=,lstart=,args='], text=True)
rows = []
for line in ps.splitlines():
    if any(s in line for s in ('mojolearn','mojo build','pytest','box_merged_verify')) and 'crashpad' not in line:
        pid = int(line.split()[0]); fds = {}
        for fd in ('1','2'):
            try:
                p = os.readlink('/proc/%s/fd/%s' % (pid, fd)); fds[fd] = p
                if p.startswith(('/root/out/', '/root/lq/', '/root/live-', '/root/measure-', '/root/race-', '/root/results', '/tmp/')) and os.path.isfile(p): paths.add(p)
            except OSError: pass
        rows.append({'process': line.strip(), 'fds': fds})
def cmd(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True)
        return {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except OSError as exc:
        return {'exit_code': 127, 'stdout': '', 'stderr': str(exc)}
print(json.dumps({'paths': sorted(paths), 'processes': rows,
 'tools': {'rsync': __import__('shutil').which('rsync')},
 'uname': cmd(['uname','-a']), 'disk': cmd(['df','-h','/root']),
 'source': cmd(['git','-C','/root/mojolearn','rev-parse','HEAD']),
 'mounts': cmd(['findmnt','-T','/root']),
 'note': 'Process exit codes unknown unless recorded by the job; output directories are copied in full.'}))
'''


def copy_with_tar(box, paths, directory, log):
    """Stream output from minimal CPU images without installing build dependencies."""
    transport = directory / ('transfer-' + str(time.time_ns()) + '.tgz')
    command = ['tar', '-czf', '-']
    for pattern in sorted(EXCLUDE | PRIVATE_NAMES) + ['*.pem', '*.key', '*.env']:
        command += ['--exclude=' + pattern]
    command += ['--'] + paths
    log.parent.mkdir(parents=True, exist_ok=True)
    with transport.open('wb') as output, log.open('ab') as errors:
        try:
            p = subprocess.run(SSH + box['ssh'] + [shlex.join(command)],
                               stdout=output, stderr=errors, timeout=900)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        errors.write((utc() + f' tar_copy_exit_code={rc}\n').encode())
    # Exit 1 may indicate a growing log. Preserve the available snapshot but
    # keep that collection incomplete, so it is retried on the next tick.
    if rc in (0, 1):
        with tarfile.open(transport) as archive:
            members = []
            for member in archive.getmembers():
                if not (member.isfile() or member.isdir()):
                    continue
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe remote archive member: ' + member.name)
                members.append(member)
            archive.extractall(directory / 'files', members=members, filter='data')
        # This is transport staging, not the immutable evidence archive. Its
        # extracted bytes are now in the mirror and will enter R2 batches.
        if rc == 0:
            transport.unlink()
    return rc


def observe(root, repo, evidence):
    status = {'observed_at': utc(), 'state': 'observed', 'boxes': boxes(evidence)}
    previous = read(root / 'observation.json', {})
    try:
        status['local'] = local_observation(repo, evidence)
        if status['local']['warnings']:
            status['local']['open_logs'] = sorted(set(status['local']['open_logs']) |
                                                   set(previous.get('local', {}).get('open_logs', [])))
    except Exception as exc:
        status['local'] = previous.get('local', {})
        status['local_error'] = str(exc)
    try:
        status['github'] = github_observation(previous)
    except Exception as exc:
        status['github_error'] = str(exc)
        status['github'] = previous.get('github', [])
    before = {(r['pid'], r['start']) for r in previous.get('local', {}).get('processes', [])}
    after = {(r['pid'], r['start']) for r in status['local'].get('processes', [])}
    status['departed'] = [{'pid': p, 'start': s, 'exit_code': None,
                           'status': 'no_longer_observed; consult retained job receipt'} for p, s in before - after]
    save(root / 'observations' / (str(time.time_ns()) + '.json'), status)
    save(root / 'observation.json', status)
    return status


def walk_files(directory):
    for base, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE and not Path(base, d).is_symlink())
        for name in sorted(files):
            p = Path(base, name)
            if (name not in PRIVATE_NAMES and p.suffix not in {'.pem', '.key', '.env'}
                    and not p.is_symlink() and p.is_file()):
                yield p


def candidates(root, repo, evidence, since):
    # No size/extension cutoff in run directories: model arrays and accepted
    # binaries are evidence too. Old unrelated campaigns are not re-uploaded.
    skipped_roots = {root.name, 'box-archive', 'box-mirror', 'compile-slots'}
    found = set()
    for child in evidence.iterdir():
        if child.name in skipped_roots or child.is_symlink():
            continue
        paths = walk_files(child) if child.is_dir() else [child]
        for path in paths:
            try:
                stat = path.stat()
                # cp -p preserves mtime on newly produced evidence directories;
                # macOS birthtime catches retained binary copies with old mtimes.
                if max(stat.st_mtime, getattr(stat, 'st_birthtime', 0)) >= since:
                    found.add(path)
            except FileNotFoundError:
                pass
    for folder in (root / 'mirrors', root / 'observations', root / 'setup', root / 'logs',
                   root / 'history', root / 'github', repo / 'bench/results'):
        if folder.exists():
            for path in walk_files(folder):
                if folder != repo / 'bench/results' or path.stat().st_mtime >= since:
                    found.add(path)
    for text in read(root / 'observation.json', {}).get('local', {}).get('open_logs', []):
        p = Path(text)
        if p.is_file() and not p.is_symlink() and root not in p.parents:
            found.add(p)
    for name in ('observation.json', 'config.json', 'WATCHERS.txt', 'health.json',
                 'HEALTH.txt', 'health-events.jsonl', 'providers.json'):
        p = root / name
        if p.exists():
            found.add(p)
    found.add(Path(__file__).resolve())
    found = {p for p in found if p.name not in PRIVATE_NAMES and p.suffix not in {'.pem', '.key', '.env'}}
    # Remote results and current job logs go ahead of the initial historical
    # catch-up. Large artifacts are still retained, without a size cutoff.
    open_logs = set(read(root / 'observation.json', {}).get('local', {}).get('open_logs', []))
    return sorted(found, key=lambda p: (not (root in p.parents or str(p) in open_logs),
                  p.suffix in {'.so', '.whl', '.tgz', '.npz', '.npy'}, str(p)))


def make_batch(root, paths, index):
    batch_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:10]
    directory = root / 'batches' / batch_id
    directory.mkdir(parents=True)
    entries, errors, total = [], [], 0
    partial = directory / 'evidence.tgz.partial'
    with tarfile.open(partial, 'w:gz', compresslevel=1) as archive:
        for path in paths:
            try:
                sig = signature(path)
                if index.get(str(path)) == sig:
                    continue
                # Copy before adding: the archive contains an internally hashed,
                # bounded snapshot even when a log continues growing.
                temp = directory / 'copy.partial'
                digest = hashlib.sha256()
                with path.open('rb') as source, temp.open('wb') as dest:
                    remaining = sig[0]
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise IOError('file shrank during snapshot')
                        dest.write(chunk); digest.update(chunk); remaining -= len(chunk)
                stable = signature(path) == sig
                name = 'files/' + str(path).lstrip('/')
                archive.add(temp, arcname=name, recursive=False)
                temp.unlink()
                entries.append({'source': str(path), 'member': name, 'size': sig[0],
                                'sha256': digest.hexdigest(), 'signature': sig, 'stable': stable})
                total += sig[0]
                if total >= 512 * 1024 * 1024:
                    break
            except (OSError, ValueError) as exc:
                errors.append({'source': str(path), 'error': str(exc)})
        manifest = {'created_at': utc(), 'files': entries, 'errors': errors,
                    'note': 'Growing/changing files are snapshots and are retried; no job success is inferred.'}
        raw = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('manifest.json'); info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    final = directory / 'evidence.tgz'; partial.replace(final)
    with final.open('rb') as src:
        digest = hashlib.file_digest(src, 'sha256').hexdigest()
    save(directory / 'receipt.json', {'created_at': utc(), 'state': 'pending_upload',
         'archive_sha256': digest, 'archive_bytes': final.stat().st_size,
         'files': len(entries), 'errors': errors, 'manifest': manifest,
         'r2_key': 'measurements/run-watch/' + os.uname().nodename + '/' + batch_id + '.tgz'})
    for entry in entries:
        if entry['stable']:
            index[entry['source']] = entry['signature']
    return directory, entries, errors


UPLOAD = r'''
set -eu
. "${MOJOLEARN_R2_CREDS:-$HOME/.mojolearn_r2}"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_DEFAULT_REGION=auto
endpoint="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
aws s3 cp "$1" "s3://$R2_BUCKET/$2" --endpoint-url "$endpoint" --only-show-errors --metadata "sha256=$3"
aws s3api head-object --bucket "$R2_BUCKET" --key "$2" --endpoint-url "$endpoint" --output json > "$4"
'''


def upload_pending(root):
    results = []
    for receipt in sorted((root / 'batches').glob('*/receipt.json')):
        data = read(receipt)
        if data['state'] == 'retained':
            continue
        directory = receipt.parent
        rc = run(['/bin/bash', '-c', UPLOAD, 'upload', str(directory / 'evidence.tgz'),
                  data['r2_key'], data['archive_sha256'], str(directory / 'r2-head.json')],
                 directory / 'upload.log', timeout=1800)
        head = read(directory / 'r2-head.json', {}) if rc == 0 else {}
        verified = (rc == 0 and head.get('ContentLength') == data['archive_bytes']
                    and head.get('Metadata', {}).get('sha256') == data['archive_sha256'])
        data.update(last_attempt=utc(), upload_exit_code=rc,
                    state='retained' if verified else 'pending_upload',
                    verification='R2 HEAD length and SHA-256 metadata' if verified else 'failed')
        save(receipt, data)
        results.append({'batch': directory.name, 'state': data['state'], 'exit_code': rc})
        if not verified:
            break  # Retry retained local bytes next cycle, never rerun the job.
    return results


def retain_workflow(root, workflow):
    directory = root / 'github' / str(workflow['databaseId'])
    receipt = directory / 'receipt.json'
    record = read(receipt, {})
    if record.get('log_exit_code') != 0:
        rc = run(['gh', '-R', 'mojolearn/mojolearn', 'run', 'view',
                  str(workflow['databaseId']), '--log'], directory / 'complete.log', timeout=180)
        record.update(observed_at=utc(), workflow=workflow, log_exit_code=rc)
        save(receipt, record)
    if not record.get('artifacts_complete'):
        raw = capture(['gh', 'api', '--paginate',
                       f'repos/mojolearn/mojolearn/actions/runs/{workflow["databaseId"]}/artifacts',
                       '--jq', '.artifacts[] | {id,name,size_in_bytes,expired}'], timeout=60)
        artifacts = [json.loads(line) for line in raw.splitlines() if line.strip()]
        record['artifacts'] = artifacts
        for artifact in artifacts:
            dest = directory / 'artifacts' / (str(artifact['id']) + '.zip')
            proof = dest.with_suffix('.json')
            if dest.exists() and read(proof, {}).get('state') == 'downloaded':
                artifact['state'] = 'downloaded'
                continue
            if artifact['expired']:
                artifact['state'] = 'expired_unretained'
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.partial')
            with temp.open('wb') as output, (directory / 'artifacts.log').open('ab') as log:
                try:
                    result = subprocess.run(['gh', 'api',
                        f'repos/mojolearn/mojolearn/actions/artifacts/{artifact["id"]}/zip'],
                        stdout=output, stderr=log, timeout=300)
                    rc = result.returncode
                except subprocess.TimeoutExpired:
                    rc = 124
                log.write((utc() + f' artifact={artifact["id"]} exit_code={rc}\n').encode())
            artifact['exit_code'] = rc
            if rc == 0:
                with zipfile.ZipFile(temp) as archive:
                    if archive.testzip():
                        raise RuntimeError('CI artifact zip checksum mismatch: ' + str(artifact['id']))
                temp.replace(dest)
                with dest.open('rb') as source:
                    digest = hashlib.file_digest(source, 'sha256').hexdigest()
                save(proof, {'state': 'downloaded', 'sha256': digest, 'bytes': dest.stat().st_size,
                             'artifact': artifact, 'at': utc()})
                artifact['state'] = 'downloaded'
            else:
                artifact['state'] = 'pending_download'
        record['artifacts_complete'] = all(a['state'] == 'downloaded' for a in artifacts)
        save(receipt, record)
    return record


def retain(root, repo, evidence):
    previous = read(root / 'retention.json', {})
    status = {'started_at': utc(), 'state': 'collecting', 'boxes': [], 'errors': [],
              'last_completed_at': previous.get('finished_at', previous.get('last_completed_at')),
              'retained_batches': previous.get('retained_batches'),
              'pending_batches': previous.get('pending_batches')}
    save(root / 'retention.json', status)
    inventory = boxes(evidence)
    providers = provider_census()
    save(root / 'providers.json', {'observed_at': utc(), 'providers': providers})
    status['providers'] = providers
    status['unregistered_instances'] = []
    for name, provider in providers.items():
        if provider['state'] == 'error':
            status['errors'].append({'provider': name, 'error': provider['error']})
        registered = {str(b['id']) for b in inventory
                      if b['provider'] == name.replace('-bare-metal', '')}
        status['unregistered_instances'].extend(
            {'provider': name, **p} for p in provider.get('instances', []) if p['id'] not in registered)
    for box in inventory:
        record = {'name': box['name'], 'observed_at': utc()}
        provider = providers.get(box['provider'], {})
        if box['provider'] == 'hotaisle':
            metal = providers.get('hotaisle-bare-metal', {})
            provider = {'state': 'observed' if provider.get('state') == metal.get('state') == 'observed' else 'error',
                        'instances': provider.get('instances', []) + metal.get('instances', [])}
        if provider.get('state') != 'observed':
            record.update(state='identity_unverified', reason='Provider unavailable; endpoint not contacted')
            status['boxes'].append(record)
            continue
        if str(box['id']) not in {p['id'] for p in provider['instances']}:
            record.update(state='retired_registration', reason='Provider confirms instance ID no longer exists; endpoint not contacted')
            # Registrations created solely for this observer follow the
            # external controller's lifecycle. Preserve their bytes after
            # verified termination so the legacy archiver cannot contact a
            # recycled endpoint. Never retire another controller's state.
            state_path = Path(box['state_file'])
            if box.get('retention_only') and state_path.exists():
                current = shell_state(state_path)
                if current.get('RETENTION_ONLY') == '1' and current.get('POD_ID') == box['id']:
                    retired = state_path.parent.with_name(state_path.parent.name +
                        '.down-provider-verified-' + str(time.time_ns()))
                    state_path.parent.rename(retired)
                    record['preserved_registration'] = str(retired)
            status['boxes'].append(record)
            continue
        directory = root / 'mirrors' / box['name']
        directory.mkdir(parents=True, exist_ok=True)
        try:
            info = json.loads(capture(SSH + box['ssh'] + ['python3 -'], timeout=60, input_text=REMOTE))
            save(directory / 'remote-observation.json', info)
            args = box['ssh']
            target = args[-1]
            transport = shlex.join(SSH + args[:-1])
            command = ['rsync', '-aR', '--partial', '--timeout=90', '-e', transport]
            for exclude in sorted(EXCLUDE):
                command += ['--exclude', exclude + '/']
            for exclude in sorted(PRIVATE_NAMES) + ['*.pem', '*.key', '*.env']:
                command += ['--exclude', exclude]
            # --relative preserves /root/out etc, without --delete. Each source
            # is shell-quoted for remote rsync as well as passed locally as argv.
            command += [target + ':' + shlex.quote(p) for p in info['paths']]
            command += [str(directory / 'files') + '/']
            (directory / 'files').mkdir(exist_ok=True)
            log = root / 'logs' / (box['name'] + '-copy.log')
            if not info['paths']:
                rc = 0
            elif info.get('tools', {}).get('rsync'):
                rc = run(command, log, timeout=900)
            else:
                rc = copy_with_tar(box, info['paths'], directory, log)
            record.update(state='copied' if rc == 0 else 'incomplete', exit_code=rc,
                          process_count=len(info['processes']), source_paths=info['paths'])
        except Exception as exc:
            record.update(state='unreachable_or_failed', error=str(exc))
        status['boxes'].append(record)
        save(root / 'retention.json', status)
    config = read(root / 'config.json')
    status['github'] = []
    github_started = time.time()
    for workflow in read(root / 'observation.json', {}).get('github', []):
        created = dt.datetime.fromisoformat(workflow['createdAt'].replace('Z', '+00:00')).timestamp()
        if created < config['retain_since_epoch'] or workflow['status'] != 'completed':
            continue
        directory = root / 'github' / str(workflow['databaseId'])
        receipt = directory / 'receipt.json'
        old = read(receipt, {})
        if old.get('log_exit_code') == 0 and old.get('artifacts_complete'):
            continue
        if time.time() - github_started > 90:
            status['errors'].append({'github': 'Catch-up pending; continues next collection tick'})
            break
        try:
            record = retain_workflow(root, workflow)
            status['github'].append({'id': workflow['databaseId'], 'log_exit_code': record['log_exit_code'],
                                     'conclusion': workflow['conclusion'],
                                     'artifacts_complete': record.get('artifacts_complete')})
            if record['log_exit_code'] or not record.get('artifacts_complete'):
                status['errors'].append({'github_run': workflow['databaseId'], 'retention': 'incomplete'})
        except Exception as exc:
            status['errors'].append({'github_run': workflow['databaseId'], 'error': str(exc)})
    index = read(root / 'file-index.json', {})
    paths = candidates(root, repo, evidence, config['retain_since_epoch'])
    pending = []
    for p in paths:
        try:
            if index.get(str(p)) != signature(p):
                pending.append(p)
        except OSError as exc:
            status['errors'].append({'source': str(p), 'error': str(exc)})
    status.update(candidate_files=len(paths), changed_files=len(pending), state='snapshotting')
    save(root / 'retention.json', status)
    started = time.time(); batch_count = 0
    while pending and time.time() - started < 120:
        directory, entries, errors = make_batch(root, pending, index)
        save(root / 'file-index.json', index)
        done = {e['source'] for e in entries} | {e['source'] for e in errors}
        pending = [p for p in pending if str(p) not in done]
        status['errors'].extend(errors); batch_count += 1
        if not done:
            break
    status.update(state='uploading', remaining_files=len(pending), new_batches=batch_count)
    save(root / 'retention.json', status)
    status['uploads'] = upload_pending(root)
    receipts = [read(p) for p in (root / 'batches').glob('*/receipt.json')]
    status.update(finished_at=utc(), retained_batches=sum(r['state'] == 'retained' for r in receipts),
                  pending_batches=sum(r['state'] != 'retained' for r in receipts))
    status['state'] = 'attention' if (status['errors'] or pending or status['pending_batches'] or
                    status.get('unregistered_instances') or
                    any(b['state'] not in ('copied', 'retired_registration') for b in status['boxes'])) else 'retained'
    save(root / 'retention.json', status)
    save(root / 'history' / (str(time.time_ns()) + '.json'), status)
    return status


def health(root, repo, evidence):
    """An independent heartbeat checker; it cannot restart or change workloads."""
    now = time.time()
    observation = read(root / 'observation.json', {})
    retention = read(root / 'retention.json', {})
    box_records = retention.get('boxes', [])
    if retention.get('state') in ('collecting', 'snapshotting', 'uploading'):
        # A new collection starts with an empty/partial inventory. Keep the
        # last completed box receipts visible until each replacement arrives;
        # collection in progress must not clear an unresolved job failure.
        last_path = max((root / 'history').glob('*.json'), default=None)
        completed = read(last_path, {}) if last_path else {}
        combined = {b['name']: b for b in completed.get('boxes', [])}
        combined.update((b['name'], b) for b in box_records)
        box_records = list(combined.values())
    previous = read(root / 'health.json', {})
    alerts = []
    def age(value):
        return now - dt.datetime.fromisoformat(value).timestamp() if value else float('inf')
    if age(observation.get('observed_at')) > 180:
        alerts.append('Job observation has not refreshed in 3 minutes')
    if age(retention.get('started_at')) > 1200:
        alerts.append('Evidence collection has not started in 20 minutes')
    if age(retention.get('finished_at') or retention.get('last_completed_at')) > 1800:
        alerts.append('No completed retention pass in 30 minutes')
    if retention.get('state') == 'attention':
        alerts.append('Evidence retention incomplete; see retention.json')
    for key in ('local_error', 'github_error'):
        if observation.get(key):
            alerts.append(key + ': ' + observation[key])
    alerts.extend(observation.get('local', {}).get('warnings', []))
    free = shutil.disk_usage(evidence).free
    if free < 25 * 1024 ** 3:
        alerts.append('Evidence disk has less than 25 GiB free')
    active = [r for r in observation.get('github', []) if r['status'] != 'completed']
    old_active = {r['databaseId'] for r in previous.get('active_github', [])}
    completed = [r for r in observation.get('github', [])
                 if r['databaseId'] in old_active and r['status'] == 'completed']
    for r in completed:
        if r['conclusion'] not in ('success', 'skipped', 'neutral'):
            alerts.append(f'CI {r["databaseId"]} {r["name"]}: {r["conclusion"]}')
    services = {}
    for label in ('com.mojolearn.run-watch-observe', 'com.mojolearn.run-watch-retain',
                  'com.mojolearn.box-archive', 'com.mojolearn.takeover-0856'):
        p = subprocess.run(['launchctl', 'print', f'gui/{os.getuid()}/{label}'],
                           capture_output=True, text=True, timeout=10)
        state = re.search(r'^\s*state = (.*)', p.stdout, re.M)
        rc = re.search(r'last exit code = (\d+)', p.stdout)
        services[label] = {'loaded': p.returncode == 0, 'state': state[1] if state else 'unknown',
                           'last_exit_code': int(rc[1]) if rc else None}
        if p.returncode or (state and state[1] != 'running' and rc and int(rc[1])):
            alerts.append('Watcher service needs attention: ' + label)
    providers = read(root / 'providers.json', {})
    for name, provider in providers.get('providers', {}).items():
        if provider['state'] == 'error':
            alerts.append('Provider inventory unavailable: ' + name)
    for box in retention.get('unregistered_instances', []):
        alerts.append(f'Instance lacks retention registration: {box["provider"]}/{box["id"]}')
    retired = {b['name'] for b in box_records if b['state'] == 'retired_registration'}
    for box in observation.get('boxes', []):
        if box['name'] in retired:
            continue
        deadline = box.get('lease_deadline_epoch')
        if deadline is not None and deadline - now < 1800:
            alerts.append(f'Lease expires within 30 minutes (or expired): {box["name"]}; owner renewal required')
    remote_results = {}
    for box in box_records:
        if box['state'] == 'retired_registration':
            continue
        path = root / 'mirrors' / box['name'] / 'files/root/out/status.txt'
        if path.exists():
            # Read the controller's explicit receipts. A living diagnostics
            # process does not erase an already failed verification step.
            stages = {}
            for line in path.read_text().splitlines():
                match = re.search(r'\b(BUILD (?:host|gpu-identical)|VERIFY identical) rc=(-?\d+)\b', line)
                if match:
                    stages[match[1]] = {'exit_code': int(match[2]), 'receipt': line}
            remote_results[box['name']] = {'source': str(path), 'stages': stages}
            for stage, result in stages.items():
                if result['exit_code']:
                    alerts.append(f'{box["name"]}: {stage} failed (exit {result["exit_code"]}); see {path}')
    # Public worker receipts, not terminal transcripts, supply exit outcomes.
    # The existing release watcher is the current local job controller; users
    # can add other controllers without changing the observer executable.
    config = read(root / 'config.json', {})
    job_sources = [evidence / 'takeover-0856-20261021/STATUS.json']
    job_sources += [Path(p) for p in config.get('job_status_files', [])]
    jobs, job_changes = {}, []
    for path in job_sources:
        if not path.exists():
            continue
        controller = read(path)
        active_here = False
        for key, job in controller.get('jobs', {}).items():
            identity = str(path) + ':' + key
            jobs[identity] = {k: job[k] for k in ('state', 'exit_code', 'error', 'finished') if k in job}
            active_here |= job.get('state') in ('starting', 'running')
            if previous.get('local_jobs') is not None and jobs[identity] != previous['local_jobs'].get(identity):
                job_changes.append({'id': key, 'source': str(path), **jobs[identity]})
                if job.get('state') in ('failed', 'lost'):
                    alerts.append(f'Local job {key}: {job["state"]}; see {path}')
        if active_here and age(controller.get('observed_at')) > 180:
            alerts.append('Active local job controller is stale: ' + str(path))
    status = {'observed_at': utc(), 'state': 'attention' if alerts else 'healthy',
              'alerts': sorted(set(alerts)), 'active_github': active,
              'local_jobs': jobs, 'local_job_changes': job_changes,
              'remote_job_results': remote_results,
              'completed_github_since_last_check': completed, 'services': services,
              'free_disk_gib': round(free / 1024 ** 3, 1),
              'retention_state': retention.get('state'), 'retained_batches': retention.get('retained_batches'),
              'pending_batches': retention.get('pending_batches'),
              'process_count': len(observation.get('local', {}).get('processes', [])),
              'providers': providers, 'retention_boxes': box_records}
    changed = status['alerts'] != previous.get('alerts', [])
    if changed or completed or job_changes:
        with (root / 'health-events.jsonl').open('a') as log:
            log.write(json.dumps({'at': utc(), 'alerts': status['alerts'], 'completed': completed,
                                  'local_job_changes': job_changes}) + '\n')
    save(root / 'health.json', status)
    lines = [f'Mojolearn monitoring: {status["state"]} at {status["observed_at"]}',
             f'CI active: {len(active)}; tracked local processes: {status["process_count"]}',
             f'Retention: {status["retention_state"]}; R2 batches: {status["retained_batches"]}; pending: {status["pending_batches"]}',
             f'Free disk: {status["free_disk_gib"]} GiB', '', 'Alerts:']
    lines.extend(status['alerts'] or ['None'])
    lines += ['', 'Active CI:'] + [f'{r["name"]}: {r["status"]} {r["url"]}' for r in active]
    lines += ['', 'Active local jobs:'] + [f'{key}: {job["state"]}' for key, job in jobs.items()
                                          if job['state'] in ('starting', 'running', 'queued')]
    (root / 'HEALTH.txt').write_text('\n'.join(lines) + '\n')
    if changed and status['alerts']:
        message = f'{len(status["alerts"])} issue(s). See mojolearn-evidence/run-watch/HEALTH.txt'
        # Native notification only to this Mac's user, with no shell expansion.
        run(['/usr/bin/osascript', '-e', 'on run argv\n display notification (item 1 of argv) '
             'with title "Mojolearn watch"\nend run', message], root / 'logs/notifications.log', timeout=10)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['observe', 'retain', 'health'])
    parser.add_argument('--root', type=Path, default=Path.home() / 'mojolearn-evidence/run-watch')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root; root.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    with (root / (args.mode + '.lock')).open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            result = {'observe': observe, 'retain': retain, 'health': health}[args.mode](root, args.repo, root.parent)
            print(json.dumps({'at': utc(), 'mode': args.mode, 'state': result['state']}), flush=True)
        except Exception as exc:
            save(root / (args.mode + '-error.json'), {'at': utc(), 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()
