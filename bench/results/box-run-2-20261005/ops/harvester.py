#!/usr/bin/env python3
"""box-run-2 harvester (2026-10-05). From identical-all/remote_supervisor.py, fixed after the 10-04 ENOSPC crash.

Every cycle, per box: check local free disk, renew the hold when due, rsync /root/lq/ (no --delete), and
upload a verified R2 snapshot when due. Every step is wrapped: a failed copy, ssh, renew or R2 call is logged and
the loop continues; renewals run even when the copy fails or the disk is low.
Disk: under DISK_LOW_GIB (20) it writes DISK_LOW (and keeps going); under DISK_STOP_COPY_GIB (8) it skips the
copy but still renews. Start through harvester_loop.sh (restarts this script if it ever exits).
Stop renewal + copying: touch harvester.stop.  No secrets in config.
"""
import concurrent.futures, datetime, hashlib, json, os, pathlib, shlex, shutil, subprocess, sys, time, traceback

DISK_LOW_GIB = 20; DISK_STOP_COPY_GIB = 8
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path, obj):
    try:
        tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)
    except OSError as e: print(stamp(), 'atomic write failed', path, e, flush=True)
def run(argv, log, timeout=120, env=None, inp=None):
    try:
        with log.open('ab') as out:
            out.write(f'--- {stamp()}\n'.encode())
            try: return subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, timeout=timeout, env=env, input=inp).returncode
            except subprocess.TimeoutExpired: out.write(b'CONTROLLER TIMEOUT\n'); return 124
            except OSError as e: out.write((str(e) + '\n').encode()); return 127
    except OSError:  # the log itself could not be written (disk full): still run the command
        try: return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, env=env, input=inp).returncode
        except Exception: return 125

def free_gib():
    return shutil.disk_usage('/System/Volumes/Data').free / 2**30

def renew(name, cfg, dest, ssh, target, now):
    if cfg.get('renew_argv'):
        env = os.environ.copy()
        if cfg.get('runpod_key_file'): env['RUNPOD_API_KEY'] = pathlib.Path(cfg['runpod_key_file']).expanduser().read_text().strip()
        env.update(cfg.get('renew_env', {}))
        return run(cfg['renew_argv'], dest / 'renew.log', 300, env)
    if cfg.get('amd_deadman_hold'):
        until = int(now) + cfg.get('hold_seconds', 14400)
        add_min = cfg.get('hold_seconds', 14400) // 60 + 30
        cmd = (f"set -e; printf '%s\\n' {until} > /root/hold-until; "
               f"b=$(cat /root/deadman.beat); t=${{b%% *}}; age=$(echo \"$b\" | tr ' ' '\\n' | grep '^age=' | cut -d= -f2); "
               f"test $(( $(date +%s) - t )) -lt 400; echo $((age + {add_min})) > /root/deadman.cap; test -f /root/deadman.armed; "
               f"echo renewed until={until} cap=$((age + {add_min}))")
        rc = run([*ssh, target, cmd], dest / 'renew.log')
        if rc == 0 and cfg.get('local_deadline_file'):
            pathlib.Path(cfg['local_deadline_file']).write_text(str(until + 600) + '\n')
        return rc
    return None

def cycle(name, cfg, base, now, disk):
    state = {'utc': stamp(), 'box': name, 'free_gib': round(disk, 1)}
    try:
        dest = base / name; dest.mkdir(exist_ok=True)
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
               '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR', *cfg.get('ssh_args', [])]
        target = cfg['target']; transport = shlex.join(ssh)
        try: previous = json.loads((dest / 'status.json').read_text())
        except Exception: previous = {}
        for k in ('last_renew', 'last_r2', 'renew_rc', 'r2_rc', 'r2_key', 'last_copy_ok', 'files', 'bytes'):
            if k in previous: state[k] = previous[k]
        state.setdefault('last_renew', 0); state.setdefault('last_r2', 0)
        # 1. renew first: never skipped for a failed copy or a low disk
        if now - state['last_renew'] >= cfg.get('renew_every_seconds', 1800):
            try: state['renew_rc'] = renew(name, cfg, dest, ssh, target, now)
            except Exception as e: state['renew_rc'] = 'EXC ' + repr(e)
            state['renew_tried'] = stamp()
            if state['renew_rc'] == 0: state['last_renew'] = now
        # 2. copy, unless the disk is nearly full
        if disk < DISK_STOP_COPY_GIB:
            state['copy_rc'] = 'SKIPPED_DISK'
        else:
            local = dest / 'lq'; local.mkdir(exist_ok=True)
            ex = ['--exclude=.url', '--exclude=.curlrc', '--exclude=.stage.in', '--exclude=source/', '--exclude=.pixi/', '--exclude=.git/',
                  '--exclude=*.so', '--exclude=*.mojopkg', '--exclude=.mojo_cache/', '--exclude=/callpath-debug/*/']
            state['copy_rc'] = run(['rsync', '-az', '--partial', *ex, '-e', transport, target + ':/root/lq/', str(local) + '/'], dest / 'copy.log', 900)
            if state['copy_rc'] == 0:
                state['last_copy_ok'] = stamp()
                files = [p for p in local.rglob('*') if p.is_file()]
                state['files'] = len(files); state['bytes'] = sum(p.stat().st_size for p in files)
                proof = dest / 'initial-copy.json'
                if files and not proof.exists():
                    atomic(proof, {'utc': stamp(), 'target': target, 'copy_rc': 0, 'files': len(files),
                                   'sha256': {str(p.relative_to(local)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.stat().st_size < 1024 * 1024}})
            state['initial_copy_verified'] = (dest / 'initial-copy.json').exists()
        # 3. R2 snapshot (box -> R2 directly, sha verified by the script)
        if cfg.get('r2_script') and now - state['last_r2'] >= cfg.get('r2_every_seconds', 1800):
            key = cfg['r2_prefix'].rstrip('/') + '/' + name + '/' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.tar.gz'
            state['r2_rc'] = run(['sh', cfg['r2_script'], shlex.join([*cfg.get('ssh_args', []), target]), '/root/lq', key,
                                  '.url', '.curlrc', '.stage.in', 'source', '.pixi', '.git', 'callpath-debug', '*.so', '*.mojopkg', '.mojo_cache'], dest / 'r2.log', 900)
            if state['r2_rc'] == 0: state['last_r2'] = now; state['r2_key'] = key
        atomic(dest / 'status.json', state)
    except Exception:
        state['error'] = traceback.format_exc()[-600:]
    return name, state

def main():
    config_path = pathlib.Path(sys.argv[1]).resolve(); base = config_path.parent
    (base / 'harvester.pid').write_text(str(os.getpid()) + '\n')
    while not (base / 'harvester.stop').exists():
        try:
            config = json.loads(config_path.read_text()); boxes = config['boxes']
            if set(boxes) - {'nv', 'amd'}: raise SystemExit('Only explicit nv and amd boxes allowed')
            now = time.time()
            try: disk = free_gib()
            except Exception: disk = -1.0
            low = base / 'DISK_LOW'
            if 0 <= disk < DISK_LOW_GIB:
                try: low.write_text(f'{stamp()} free={disk:.1f} GiB < {DISK_LOW_GIB}\n')
                except OSError: pass
            elif disk >= DISK_LOW_GIB and low.exists():
                try: low.rename(base / f'DISK_LOW.cleared-{int(now)}')
                except OSError: pass
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                result = dict(pool.map(lambda item: cycle(item[0], item[1], base, now, disk), boxes.items()))
            ready = all(s.get('copy_rc') == 0 and s.get('initial_copy_verified') and s.get('renew_rc', 0) in (0, None)
                        and now - s.get('last_renew', 0) < boxes[k].get('renew_every_seconds', 1800) + 600 for k, s in result.items())
            atomic(base / 'harvester-status.json', {'utc': stamp(), 'pid': os.getpid(), 'free_gib': round(disk, 1), 'ready_to_queue': ready, 'boxes': result})
            print(stamp(), f'free={disk:.1f}GiB ready={ready}', {k: {x: v.get(x) for x in ('copy_rc', 'renew_rc', 'files', 'r2_rc', 'error')} for k, v in result.items()}, flush=True)
            interval = config.get('interval_seconds', 30)
        except SystemExit: raise
        except Exception:
            print(stamp(), 'cycle exception', traceback.format_exc()[-800:], flush=True); interval = 30
        for _ in range(interval):
            if (base / 'harvester.stop').exists(): break
            time.sleep(1)

if __name__ == '__main__': main()
