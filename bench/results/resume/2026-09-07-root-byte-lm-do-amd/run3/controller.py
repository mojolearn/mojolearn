#!/usr/bin/env python3
"""Root-only DO byte LM controller. Default prepares files; --rent spends money.

Source candidate, not executed by author. Model work stays on guarded AMD.
Teardown follows e2_remote_leg's dual-deadman and verified-404 contract.
"""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

COMMIT = 'e0bee4aca6e21495edfc484f0525687fad82b957'
API = 'https://api.digitalocean.com/v2'
API_LOG = None

def event(event_name, **fields):
    record = dict(time=time.time(), event=event_name)
    record.update(fields)
    print(json.dumps(record), flush=True)

def api_record(method, path, status):
    record = dict(time=time.time(), method=method, path=path, status=status)
    if API_LOG is not None:
        with API_LOG.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    event('api', **record)

PATHS = ['.gitattributes', 'pixi.toml', 'pixi.lock', 'checks', 'core', 'gemm',
         'embedding', 'transformer', 'training', 'bindings', 'python',
         'tools/training_validation_admit.py', 'tools/byte_lm_validation_serial.sh',
         'tools/byte_lm_validation_admit.py', 'tools/root_job_receipt.py',
         'tools/byte_lm_real_text_capture.py', 'tools/byte_lm_gradient_oracle.py',
         'tools/byte_lm_state_compare.py', 'tools/tests/test_byte_lm_state_compare.py',
         'tools/nvidia_serial_guard.py', 'tools/amd_serial_guard.py',
         'tools/test_nvidia_serial_guard.py', 'tools/test_amd_serial_guard.py',
         'tools/byte_lm_resume_serial.sh']


def api(token, path, method='GET', data=None):
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(API + path, data=body, method=method,
             headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError('oversized API response')
            api_record(method, path, response.status)
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        api_record(method, path, exc.code)
        return exc.code, {}
    except Exception as exc:
        api_record(method, path, type(exc).__name__)
        raise


def destroy(config):
    token = Path(config['token_file']).read_text().strip()
    ids = [config['id']] if config.get('id') else []
    if not ids:
        code, data = api(token, '/droplets?tag_name=' + config['tag'] + '&per_page=200')
        if code != 200 or data.get('links', {}).get('pages', {}).get('next'):
            return False
        ids = [item['id'] for item in data.get('droplets', []) if item['name'] == config['name']]
        # An unreadable create may complete late: do not cancel a name watchdog
        # merely because its first listing has no matching droplet.
        if not ids:
            return False
    for identity in ids:
        api(token, '/droplets/' + str(identity), 'DELETE')
    for _ in range(6):
        codes = [api(token, '/droplets/' + str(identity))[0] for identity in ids]
        if all(code == 404 for code in codes):
            return True
        time.sleep(5)
    return False


def watchdog(path):
    global API_LOG
    path = Path(path)
    config = json.loads(path.read_text())
    # Read all required configuration/credential bytes before announcing armed.
    token = Path(config['token_file']).read_text().strip()
    if not token or not isinstance(config['deadline'], (int, float)) or config['deadline'] <= time.time():
        raise ValueError('invalid watchdog configuration')
    API_LOG = path.parent / 'watchdog-api-actions.jsonl'
    with (path.parent / 'watchdog-ready.json').open('x') as ready:
        json.dump(dict(pid=os.getpid(), name=config['name'], deadline=config['deadline']), ready)
        ready.flush()
        os.fsync(ready.fileno())
    event('watchdog_ready', pid=os.getpid(), name=config['name'])
    time.sleep(max(0, config['deadline'] - time.time()))
    for _ in range(12):
        try:
            if destroy(config):
                print('DEADMAN_DELETE_VERIFIED_404', flush=True)
                return 0
        except Exception as exc:
            print(type(exc).__name__, flush=True)
        time.sleep(20)
    return 1


def run(command, timeout, **kwargs):
    return subprocess.run(command, check=True, timeout=timeout, **kwargs)


def main():
    global API_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', required=True, help='verified absolute HIP Torch + venv Python on DO image')
    parser.add_argument('--image', type=int, default=188571990)
    parser.add_argument('--setup-script', type=Path, help='root-reviewed <=64KiB dependency-only script, guarded remotely for <=600s')
    parser.add_argument('--ssh-key', required=True, help='DO registered SSH key fingerprint or ID')
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--rent', action='store_true')
    parser.add_argument('--head64', action='store_true', help='after all12 pass, use remaining >=600s for guarded head64')
    args = parser.parse_args()
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', args.python):
        parser.error('literal absolute Python path required')
    if not re.fullmatch(r'[A-Za-z0-9:]+', args.ssh_key):
        parser.error('invalid SSH key identifier')
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    os.chmod(out, 0o700)
    API_LOG = out / 'api-actions.jsonl'
    event('preparing', output=str(out))
    # Freeze the controller before provisioning; no numerical overlay.
    frozen = out / 'controller.py'
    shutil.copyfile(__file__, frozen)
    shutil.copyfile(root / 'tools/do_byte_lm_remote.sh', out / 'remote.sh')
    if args.setup_script:
        if args.setup_script.is_symlink() or not args.setup_script.is_file() or args.setup_script.stat().st_size > 65536:
            raise ValueError('setup script must be a bounded regular file')
        shutil.copyfile(args.setup_script, out / 'setup.sh')
    source = out / 'source.tar'
    with source.open('xb') as stream:
        run(['git', '-C', str(root), 'archive', COMMIT, '--', *PATHS], 60, stdout=stream)
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('source archive exceeds 64 MiB uncompressed cap')
    compressed = out / 'source.tar.gz'
    with source.open('rb') as incoming, gzip.open(compressed, 'wb', compresslevel=1) as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    source = compressed
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError('source archive exceeds 10 MiB compressed cap')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = dict(commit=COMMIT, source_sha256=source_sha, image=args.image,
                region='tor1', size='gpu-mi325x1-256gb', python=args.python,
                cpu_cores=2, modular_pool_bytes=1073741824, deadline_seconds=3600,
                scope='12-job AMD byte-LM qualification; no inferred cross-vendor result')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    if not args.rent:
        print(json.dumps(dict(status='PREPARED_NO_API_CALLS', **plan)))
        return 0
    if args.token_file is None or args.token_file.is_symlink():
        parser.error('protected token file required for --rent')
    info = args.token_file.stat()
    if info.st_mode & 0o077 or not 1 <= info.st_size <= 4096:
        parser.error('token file must be private and bounded')
    token = args.token_file.read_text().strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', token):
        parser.error('invalid token bytes')
    code, existing = api(token, '/droplets?per_page=200')
    if (code != 200 or existing.get('links', {}).get('pages', {}).get('next')
            or any(str(d.get('size_slug', '')).startswith('gpu-')
                   for d in existing.get('droplets', []))):
        raise RuntimeError('Require complete provider inventory with no active GPU droplet')
    # Keep the credential out of argv, logs, JSON reports and fetched artifacts.
    secret = out / '.do-token'
    secret.write_text(token)
    os.chmod(secret, 0o600)
    stamp = str(time.time_ns())
    config = dict(token_file=str(secret), name='mojolearn-byte-lm-' + stamp,
                  tag='ml-byte-' + stamp, deadline=time.time() + 3600)
    config_file = out / '.deadman.json'
    config_file.write_text(json.dumps(config))
    deadlog = (out / 'deadman.log').open('wb')
    dead = subprocess.Popen([sys.executable, str(frozen), '--watchdog', str(config_file)],
                            stdout=deadlog, stderr=subprocess.STDOUT, start_new_session=True)
    ready_until = time.monotonic() + 10
    while time.monotonic() < ready_until:
        if dead.poll() is not None:
            raise RuntimeError('local watchdog exited before readiness; no create attempted')
        ready_path = out / 'watchdog-ready.json'
        if ready_path.is_file():
            try:
                ready = json.loads(ready_path.read_text())
            except (ValueError, OSError):
                time.sleep(0.1)
                continue
            if ready.get('pid') != dead.pid or ready.get('name') != config['name'] or ready.get('deadline') != config['deadline']:
                raise RuntimeError('local watchdog readiness mismatch')
            event('local_watchdog_armed', pid=dead.pid)
            break
        time.sleep(0.1)
    else:
        raise RuntimeError('local watchdog readiness timeout; no create attempted')
    verified = False
    ssh = None
    def interrupted(signum, frame):
        raise InterruptedError('controller interrupted')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        event('create_requested', name=config['name'])
        code, data = api(token, '/droplets', 'POST', dict(name=config['name'], region='tor1',
            size='gpu-mi325x1-256gb', image=args.image, ssh_keys=[args.ssh_key], tags=[config['tag']]))
        if code != 202 or not data.get('droplet', {}).get('id'):
            raise RuntimeError('create response absent; name/tag watchdog remains armed')
        config['id'] = data['droplet']['id']
        event('droplet_created', id=config['id'])
        config_file.write_text(json.dumps(config))
        (out / 'droplet-id.txt').write_text(str(config['id']) + '\n')
        ready_until = min(time.time() + 600, config['deadline'] - 600)
        while time.time() < ready_until:
            code, data = api(token, '/droplets/' + str(config['id']))
            ips = [item['ip_address'] for item in data.get('droplet', {}).get('networks', {}).get('v4', []) if item['type'] == 'public']
            if ips and data['droplet']['status'] == 'active':
                ip = ips[0]
                if not re.fullmatch(r'[0-9.]+', ip):
                    raise ValueError('invalid remote IP')
                ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=10',
                       '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', 'root@' + ip]
                try:
                    run(ssh + ['true'], 15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    event('ssh_ready', ip=ip)
                    break
                except subprocess.SubprocessError:
                    pass
            time.sleep(10)
        else:
            raise RuntimeError('DO SSH readiness deadline exceeded')
        # Second watchdog lives on droplet and survives controller/network loss.
        run(ssh + ['umask 077; mkdir /root/ml-byte-deadman'], 30)
        for name, raw in [('controller.py', frozen.read_bytes()), ('token', token.encode()),
                          ('config.json', json.dumps(dict(config, token_file='/root/ml-byte-deadman/token')).encode())]:
            run(ssh + ['umask 077; cat > /root/ml-byte-deadman/' + name], 30, input=raw)
        run(ssh + ['nohup /usr/bin/python3 /root/ml-byte-deadman/controller.py --watchdog /root/ml-byte-deadman/config.json > /root/ml-byte-deadman/log 2>&1 < /dev/null & echo $! > /root/ml-byte-deadman/pid; for attempt in 1 2 3 4 5 6 7 8 9 10; do kill -0 $(cat /root/ml-byte-deadman/pid) || exit 2; if test -s /root/ml-byte-deadman/watchdog-ready.json; then cat /root/ml-byte-deadman/watchdog-ready.json; exit 0; fi; sleep 1; done; exit 2'], 20, stdout=(out / 'remote-watchdog-ready.json').open('wb'))
        remote_ready = json.loads((out / 'remote-watchdog-ready.json').read_text())
        if remote_ready.get('name') != config['name'] or remote_ready.get('deadline') != config['deadline']:
            raise RuntimeError('remote watchdog readiness mismatch')
        event('remote_watchdog_armed', pid=remote_ready['pid'])
        run(ssh + ['mkdir /root/mojolearn'], 30)
        event('shipping_source', bytes=source.stat().st_size)
        with source.open('rb') as stream:
            run(ssh + ['cat > /root/source.tar.gz'], min(600, int(config['deadline'] - time.time() - 600)), stdin=stream)
        actual = run(ssh + ['sha256sum /root/source.tar.gz'], 30, capture_output=True).stdout.decode().split()[0]
        if actual != source_sha:
            raise ValueError('source transfer SHA mismatch')
        run(ssh + ['tar -xzf /root/source.tar.gz -C /root/mojolearn'], 30)
        run(ssh + ['cat > /root/do-byte-lm-remote.sh'], 30, input=(out / 'remote.sh').read_bytes())
        if args.setup_script:
            run(ssh + ['cat > /root/do-byte-lm-setup.sh'], 30, input=(out / 'setup.sh').read_bytes())
        remaining = min(3000, int(config['deadline'] - time.time() - 600))
        if remaining < 180:
            raise RuntimeError('insufficient lease after source transfer')
        event('guarded_payload_start', remaining_seconds=remaining, head64=args.head64)
        command = 'timeout -k 20 %d bash /root/do-byte-lm-remote.sh %d %s %d' % (remaining, remaining, args.python, int(args.head64))
        detached = '( ' + command + '; echo $? > /root/byte-lm-do-exit ) > /root/byte-lm-do-console.log 2>&1'
        run(ssh + ['nohup bash -c ' + "'" + detached + "'" + ' < /dev/null >/dev/null 2>&1 &'], 30)
        while time.time() < config['deadline'] - 540:
            try:
                status = run(ssh + ['cat /root/byte-lm-do-exit'], 20, capture_output=True).stdout.decode().strip()
                if status:
                    (out / 'remote-exit.txt').write_text(status + '\n')
                    event('payload_exit', status=status)
                    break
            except subprocess.SubprocessError:
                pass
            time.sleep(10)
        else:
            raise RuntimeError('remote work deadline exceeded')
    finally:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        if ssh:
            event('fetch_start')
            # Fetch even failed jobs. Fetch timeout cannot prevent finally deletion.
            try:
                with (out / 'artifacts.tar').open('xb') as stream:
                    run(ssh + ['tar -C /root -cf - byte-lm-do-output byte-lm-do-console.log byte-lm-do-exit'],
                        max(1, min(420, int(config['deadline'] - time.time() - 90))), stdout=stream)
            except Exception as exc:
                (out / 'fetch-failure.txt').write_text(type(exc).__name__ + '\n')
        try:
            event('destroy_start', id=config.get('id'))
            verified = destroy(config)
            event('destroy_complete', verified_http_404=verified)
        finally:
            (out / 'teardown.json').write_text(json.dumps(dict(verified_http_404=verified)) + '\n')
            if verified:
                dead.terminate()
                dead.wait(timeout=5)
                secret.unlink()
            else:
                print('DELETION UNCONFIRMED: local/remote deadmen remain armed', file=sys.stderr)
    print('Fetched artifacts require root extraction, source inventory and 12-job file admission; no automatic identity claim.')
    return 0 if verified and (out / 'remote-exit.txt').read_text().strip() == '0' else 1


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--watchdog':
        raise SystemExit(watchdog(sys.argv[2]))
    raise SystemExit(main())
