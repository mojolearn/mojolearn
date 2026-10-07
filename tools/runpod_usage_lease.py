#!/usr/bin/env python3
"""Adopt a named RunPod into a sliding owner lease; never changes legacy guards.

plan CONFIG; adopt CONFIG; manage CONFIG [--once]
Adoption installs an independent remote watchdog using its existing 0600 curl
credential. Parent must separately retire *all* old EXIT traps/deadmen/timers.
A healthy manager renews a 90-minute orphan deadline while work runs. Idle
termination requires DONE, exact captured file hashes, then the configured 30, 45 or 60 idle minutes.
Remove DONE before assigning new work. Job timeouts remain the job's concern.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

SCHEMA = 'mojolearn.runpod-usage-lease/1'
TERMINAL = {'TERMINATED', 'OWNERSHIP_REFUSED'}


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w') as stream:
        os.chmod(tmp, 0o600); json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(c):
    for key in ('pod_id', 'owner_id'):
        if not re.fullmatch(r'[A-Za-z0-9_-]{4,100}', c.get(key, '')):
            raise ValueError('invalid ' + key)
    kind = c.get('pod_kind', 'cpu')
    name = c.get('pod_name', '')
    if kind == 'cpu':
        if not name.startswith('mojolearn-cpu-'):
            raise ValueError('explicit mojolearn-cpu pod_name required')
    elif kind == 'campaign-gpu':
        if not re.fullmatch(r'mojolearn-campaign-nvidia-[a-z0-9-]{8,80}', name):
            raise ValueError('explicit dedicated NVIDIA campaign pod_name required')
        if c.get('gpu_count') != 1 or c.get('shared_queue') is not False:
            raise ValueError('campaign GPU must be single GPU and not a shared queue pod')
    else:
        raise ValueError('unsupported pod_kind')
    for key in ('remote_out', 'local_out', 'remote_state', 'remote_curlrc'):
        if not isinstance(c.get(key), str) or not c[key].startswith('/'):
            raise ValueError('absolute ' + key + ' required')
    if Path(c['remote_curlrc']) == Path(c['remote_out']) or Path(c['remote_out']) in Path(c['remote_curlrc']).parents:
        raise ValueError('credential must be outside captured artifacts')
    for rel in c.get('exclude_relative', []):
        if not isinstance(rel,str) or not rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('unsafe explicit capture exclusion')
    if not 1 <= c.get('capture_timeout',1800) <= 1800:
        raise ValueError('capture_timeout must be <=1800, below orphan lease')
    if not 1 <= c.get('max_failed_polls',3) <= 10:
        raise ValueError('max_failed_polls must be1..10')
    if Path(c['remote_state']) == Path(c['remote_out']) or Path(c['remote_out']) in Path(c['remote_state']).parents:
        raise ValueError('guard state must be outside captured artifacts')
    if not isinstance(c.get('ssh'), list) or not c['ssh'] or any(not isinstance(x, str) for x in c['ssh']):
        raise ValueError('ssh argument list required, excluding executable')
    for key in ('remote_out','remote_state','remote_curlrc'):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', c[key]) or '..' in Path(c[key]).parts:
            raise ValueError('unsafe remote path '+key)
    if c.get('idle_seconds', 2700) not in (1800, 2700, 3600):
        raise ValueError('supported idle policies are 30, 45 or 60 minutes')
    if not 5400 <= c.get('orphan_seconds', 5400) <= 86400:
        raise ValueError('orphan lease must be at least 90 minutes')
    if not 10 <= c.get('poll_seconds', 30) <= 300:
        raise ValueError('poll_seconds must be 10..300')
    return c


def new_state(c, now):
    return dict(schema=SCHEMA, pod_id=c['pod_id'], owner_id=c['owner_id'], config_sha256=digest(c),
                status='GUARDED', last_owner_heartbeat=now,
                orphan_deadline=now+c.get('orphan_seconds', 5400), idle_since=None,
                captured_manifest=None, capture_verified_at=None, busy_hold=None)


def owned(c, state):
    if any(state.get(k) != c[k] for k in ('pod_id', 'owner_id')) or state.get('config_sha256') != digest(c):
        raise ValueError('ownership/config mismatch; refusing takeover')


def renew(c, state, now, captured=None, done=False):
    owned(c, state)
    if state['status'] in TERMINAL:
        raise ValueError('terminal lease cannot be renewed')
    state = dict(state, last_owner_heartbeat=now, orphan_deadline=now+c.get('orphan_seconds', 5400))
    if not done:
        state.update(idle_since=None, captured_manifest=None, capture_verified_at=None)
    elif captured:
        if captured != state.get('captured_manifest'):
            state.update(idle_since=now, capture_verified_at=now, captured_manifest=captured)
    return state


def decision(c, state, now, done, current_manifest=None):
    owned(c, state)
    if state['status'] in TERMINAL:
        return 'TERMINAL'
    if now >= state['orphan_deadline']:
        return 'DELETE_ORPHAN'
    if (done and not state.get('busy_hold') and state.get('idle_since') is not None and state.get('captured_manifest')
            and current_manifest == state['captured_manifest']
            and now-state['idle_since'] >= c.get('idle_seconds', 2700)):
        return 'DELETE_IDLE'
    return 'KEEP'


def inventory(root, excludes=()):
    root = Path(root); rows = []
    if not root.is_dir():
        raise ValueError('artifact directory missing')
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if any(rel==Path(x) or Path(x) in rel.parents for x in excludes):continue
        if p.is_symlink():
            raise ValueError('artifact symlinks require explicit materialization: ' + str(p.relative_to(root)))
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
        rows.append(dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=h.hexdigest()))
    return rows


def verify_capture(rows, local_root):
    root = Path(local_root).resolve()
    for row in rows:
        p = root / row['path']
        if p.is_symlink() or root not in p.resolve().parents:
            raise ValueError('unsafe captured path')
        if not p.is_file() or p.stat().st_size != row['bytes']:
            raise ValueError('missing/size mismatch: ' + row['path'])
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
        if h.hexdigest() != row['sha256']:
            raise ValueError('capture hash mismatch: ' + row['path'])
    return digest(rows)


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=kwargs.pop('timeout', 120), **kwargs)


def ssh(c, args, payload=None, timeout=120):
    return run(['ssh', *c['ssh'], shlex.join(args)], input=payload, timeout=timeout).stdout


def api(c, method, path):
    cred = Path(c['remote_curlrc'])
    if cred.stat().st_mode & 0o777 != 0o600:
        raise ValueError('remote curl credential must be mode 0600')
    # Never read the credential into Python, argv, logs, or captured output.
    r = run(['curl', '-K', str(cred), '--connect-timeout', '10', '--max-time', '30',
             '-X', method, '-w', '\n%{http_code}', ('https://api.runpod.io'+path if path.startswith('/v2/') else 'https://rest.runpod.io/v1'+path)], timeout=40)
    body, code = r.stdout.rsplit(b'\n', 1)
    try: value = json.loads(body)
    except ValueError: value = None
    return int(code), value


def verify_pod(c, call=api):
    code, value = call(c, 'GET', '/pods/'+c['pod_id'])
    if code != 200 or not isinstance(value, dict) or value.get('id') != c['pod_id'] or value.get('name') != c['pod_name']:
        raise ValueError('provider pod ID/name verification failed')
    if c.get('pod_kind') == 'campaign-gpu' and value.get('gpuCount') != 1:
        raise ValueError('provider must confirm exactly one campaign GPU')


def delete_verify(c, call=api):
    code, value = call(c, 'GET', '/pods/'+c['pod_id'])
    gone = code == 404 or (code == 200 and isinstance(value, dict) and value.get('desiredStatus') == 'TERMINATED')
    if not gone:
        verify_pod(c, call)
        code, _ = call(c, 'DELETE', '/pods/'+c['pod_id'])
        if code not in (200, 202, 204, 404):
            code, _ = call(c, 'DELETE', '/v2/pods/'+c['pod_id'])
            if code not in (200, 202, 204, 404): return False
    code, value = call(c, 'GET', '/pods/'+c['pod_id'])
    code_list, listing = call(c, 'GET', '/pods')
    pods = listing if isinstance(listing, list) else listing.get('items', listing.get('pods')) if isinstance(listing, dict) else None
    valid = isinstance(pods, list) and all(isinstance(p, dict) and isinstance(p.get('id'), str) and p['id'] for p in pods)
    gone = code == 404 or (code == 200 and isinstance(value, dict) and value.get('desiredStatus') == 'TERMINATED')
    return gone and code_list == 200 and valid and all(p['id'] != c['pod_id'] for p in pods)


def read_state(c):
    state = json.loads((Path(c['remote_state'])/'state.json').read_text()); owned(c, state); return state


def remote(c, action):
    folder=Path(c['remote_state']); folder.mkdir(parents=True, exist_ok=True)
    lock=(folder/'state.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
    now=time.time(); done=(Path(c['remote_out'])/'DONE').is_file()
    if action=='init':
        verify_pod(c)
        state=read_state(c) if (folder/'state.json').exists() else new_state(c, now)
        if state['status'] in TERMINAL or now>=state['orphan_deadline']:
            raise ValueError('expired/terminal lease requires explicit recovery, cannot adopt')
        atomic(folder/'state.json', state)
    elif action=='heartbeat':
        request=json.load(sys.stdin); state=read_state(c); cap=request.get('captured_manifest')
        if cap:
            if not done or digest(inventory(c['remote_out'],c.get('exclude_relative',()))) != cap: raise ValueError('remote artifacts changed after capture')
        state=renew(c, state, now, cap, done); atomic(folder/'state.json', state)
    elif action in ('hold','release-hold'):
        state=read_state(c);request=json.load(sys.stdin)
        if state['status'] in TERMINAL:raise ValueError('terminal lease')
        state.update(busy_hold=str(request.get('reason') or 'queued work') if action=='hold' else None, idle_since=None, captured_manifest=None, capture_verified_at=None)
        atomic(folder/'state.json',state)
    elif action=='probe':
        state=read_state(c)
        pid=state.get('guardian_pid');proc=Path('/proc')/str(pid)/'cmdline'
        state['guardian_alive']=bool(pid and proc.exists() and (c['remote_state']+'/manager.py').encode() in proc.read_bytes() and b'guardian' in proc.read_bytes())
        files=list(Path(c['remote_out']).rglob('*')) if Path(c['remote_out']).exists() else []
        progress=digest(sorted((str(p.relative_to(c['remote_out'])),p.stat().st_size,p.stat().st_mtime_ns) for p in files if p.is_file()))
        code=Path(c['remote_out'])/'cmd.exit'
        state=dict(state, done=done, command_exit=code.read_text().strip() if code.is_file() else None,
                   progress=progress, guardian_pid=state.get('guardian_pid'))
    elif action=='inventory':
        if not done: raise ValueError('DONE required before final inventory')
        state=dict(rows=inventory(c['remote_out'],c.get('exclude_relative',())))
    elif action=='tick':
        state=read_state(c); cap=None
        if now < state['orphan_deadline'] and done and not state.get('busy_hold') and state.get('idle_since') is not None and now-state['idle_since']>=c.get('idle_seconds',2700):
            cap=digest(inventory(c['remote_out'],c.get('exclude_relative',())))
            if cap != state.get('captured_manifest'): state.update(idle_since=None,captured_manifest=None,capture_verified_at=None)
        if not done: state.update(idle_since=None,captured_manifest=None,capture_verified_at=None)
        act=decision(c,state,now,done,cap)
        if act.startswith('DELETE_'):
            state.update(status='DELETING',deletion_reason=act)
            atomic(folder/'state.json',state)
            state['status']='TERMINATED' if delete_verify(c) else 'DELETE_RETRY'
        atomic(folder/'state.json',state)
    else: raise ValueError('unknown remote action')
    print(json.dumps(state),flush=True)


def guardian(c):
    folder=Path(c['remote_state'])
    lock=(folder/'guardian.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # PID readback is evidence of independent guardian readiness.
    with (folder/'state.lock').open('a') as sl:
        fcntl.flock(sl,fcntl.LOCK_EX); state=read_state(c);state['guardian_pid']=os.getpid();atomic(folder/'state.json',state)
    while True:
        try:
            remote(c,'tick')
            if read_state(c)['status'] in TERMINAL:return
        except Exception as exc:
            print(json.dumps({'guardian_error':type(exc).__name__,'at':time.time()}),flush=True)
        time.sleep(c.get('poll_seconds',30))


def remote_call(c, action, value=None):
    args=['python3',c['remote_state']+'/manager.py','remote',c['remote_state']+'/config.json',action]
    return json.loads(ssh(c,args,json.dumps(value or {}).encode()))


def notify(c, local, kind, detail):
    event=dict(pod_id=c['pod_id'],owner_id=c['owner_id'],kind=kind,detail=detail,at=time.time())
    events=local/'events';events.mkdir(exist_ok=True);key=digest([kind,detail]);path=events/(key+'.json')
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('notified') or time.time()-previous.get('last_attempt',0)<60:return
        event=previous
    event['last_attempt']=time.time()
    atomic(path,event)
    if c.get('notify_thread'):
        msg=f"RunPod {c['pod_id']} ({c['owner_id']}): {kind}. {detail}. Inspect {local/'manager-status.json'}; manage this job now."
        try:
            run([c.get('codex','/opt/homebrew/bin/codex'),'queue','--thread',c['notify_thread'],'--message',msg],timeout=30)
            event['notified']=True
        except (OSError,subprocess.SubprocessError):event['notified']=False
        atomic(path,event)


def adopt(c):
    # No legacy process is signaled here. Installation is refused across owners.
    folder=c['remote_state'];script=Path(__file__).read_bytes()
    run(['rsync','--version'])
    ssh(c,['sh','-c','command -v python3 >/dev/null && command -v rsync >/dev/null && command -v curl >/dev/null'])
    installer="import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); f=p/'manager.py'; t=p/'manager.incoming';t.write_bytes(sys.stdin.buffer.read());os.chmod(t,0o700);os.replace(t,f)"
    # Check existing immutable ownership before replacing even the helper.
    check="import json,pathlib,sys;p=pathlib.Path(sys.argv[1])/'config.json';c=json.loads(sys.stdin.read());assert not p.exists() or json.loads(p.read_text())==c,'existing owner/config differs'"
    ssh(c,['python3','-c',check,folder],json.dumps(c).encode())
    ssh(c,['python3','-c',installer,folder],script)
    writer="import pathlib,sys,os;p=pathlib.Path(sys.argv[1]);t=p.with_suffix('.incoming');t.write_bytes(sys.stdin.buffer.read());os.chmod(t,0o600);os.replace(t,p)"
    ssh(c,['python3','-c',writer,folder+'/config.json'],json.dumps(c).encode())
    remote_call(c,'init')
    existing=remote_call(c,'probe')
    if existing.get('guardian_alive'):
        print(json.dumps(dict(status='EXISTING_GUARD_READY_LEGACY_GUARDS_UNCHANGED',guardian_pid=existing['guardian_pid'],remote_state=folder)));return
    starter="import pathlib,subprocess,sys;f=pathlib.Path(sys.argv[1]);log=(f/'guardian.log').open('ab');p=subprocess.Popen(['python3',str(f/'manager.py'),'guardian',str(f/'config.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)"
    pid=int(ssh(c,['python3','-c',starter,folder]).strip())
    for _ in range(10):
        state=remote_call(c,'probe')
        if state.get('guardian_pid')==pid and state.get('guardian_alive'):
            print(json.dumps(dict(status='NEW_GUARD_READY_LEGACY_GUARDS_UNCHANGED',guardian_pid=pid,remote_state=folder)));return
        time.sleep(1)
    raise RuntimeError('guardian readiness not established; legacy guards must remain')


def local_gone(c):
    """Read-only provider confirmation after SSH disappears; never DELETE locally."""
    key=Path(c.get('local_key_file',str(Path.home()/'.mojolearn_runpod_key')))
    try:
        if key.stat().st_mode & 0o777 != 0o600:return False
        with tempfile.TemporaryDirectory(prefix='mojolearn-lease-api-') as folder:
            cred=Path(folder)/'curlrc'
            token=key.read_text().strip()
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',token):return False
            cred.write_text('header = "Authorization: Bearer '+token+'"\nsilent\nshow-error\n');os.chmod(cred,0o600)
            cfg=dict(c,remote_curlrc=str(cred));code,val=api(cfg,'GET','/pods/'+c['pod_id']);lc,listing=api(cfg,'GET','/pods')
            pods=listing if isinstance(listing,list) else listing.get('items',listing.get('pods')) if isinstance(listing,dict) else None
            return (code==404 or code==200 and isinstance(val,dict) and val.get('desiredStatus')=='TERMINATED') and lc==200 and isinstance(pods,list) and all(isinstance(p,dict) and p.get('id') and p['id']!=c['pod_id'] for p in pods)
    except (OSError,ValueError,subprocess.SubprocessError):return False


def error_diagnostic(exc, phase):
    """Retain useful capture errors without command argv, stderr or secrets."""
    detail = type(exc).__name__
    if isinstance(exc, ValueError):
        value = str(exc)
        if value.startswith(('missing/size mismatch: ', 'capture hash mismatch: ')):
            detail = value[:500]
        elif value == 'unsafe captured path':
            detail = value
    elif isinstance(exc, subprocess.CalledProcessError):
        detail = 'command failed with exit ' + str(exc.returncode)
    return dict(error_type=type(exc).__name__, error_stage=phase, error_detail=detail)


def manage(c, once=False):
    local=Path(c['local_out']);local.mkdir(parents=True,exist_ok=True)
    lock=(local/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last_progress=None;progress_at=time.monotonic();failed_polls=0
    while True:
        phase='probe';probe=None
        try:
            probe=remote_call(c,'probe')
            if probe.get('status')=='TERMINATED':
                atomic(local/'manager-status.json',dict(status='TERMINATED',lease=probe,updated_at=time.time()));return
            if not probe.get('guardian_alive'):raise RuntimeError('guardian not alive; repair before renewing')
            phase='owner_heartbeat'
            state=remote_call(c,'heartbeat') if failed_polls<c.get('max_failed_polls',3) else probe
            if probe['progress']!=last_progress:last_progress=probe['progress'];progress_at=time.monotonic()
            if probe.get('command_exit') not in (None,'0'):notify(c,local,'JOB_FAILED','command exit '+probe['command_exit'])
            if not probe['done'] and time.monotonic()-progress_at>=c.get('stall_seconds',1800):
                notify(c,local,'NO_PROGRESS','No artifact progress for configured stall window; investigate active job, do not declare idle')
            artifacts=local/'artifacts';artifacts.mkdir(exist_ok=True)
            # argv boundary: rsync transport options are shell-quoted as its -e value.
            target=c['ssh'][-1];ssh_opts=['ssh',*c['ssh'][:-1]]
            phase='artifact_sync'
            run(['rsync','-az','--partial',*[arg for rel in c.get('exclude_relative',[]) for arg in ('--exclude','/'+rel.rstrip('/')+'/***')],'-e',shlex.join(ssh_opts),target+':'+c['remote_out'].rstrip('/')+'/',str(artifacts)+'/'],timeout=c.get('capture_timeout',1800))
            if probe['done']:
                phase='remote_inventory';rows=remote_call(c,'inventory')['rows']
                phase='local_capture_verify';captured=verify_capture(rows,artifacts)
                phase='capture_ack'
                state=remote_call(c,'heartbeat',dict(captured_manifest=captured))
                atomic(local/'capture-receipt.json',dict(exclude_relative=c.get('exclude_relative',[]),pod_id=c['pod_id'],owner_id=c['owner_id'],manifest_sha256=captured,rows=rows,verified_at=time.time()))
                notify(c,local,'JOB_DONE_CAPTURED',('All retained files verified; pending-work hold remains' if state.get('busy_hold') else 'All retained files verified;'+str(c.get('idle_seconds',2700)//60)+'-minute idle deletion armed')+'; capture '+captured[:16])
            phase='final_heartbeat';failed_polls=0;state=remote_call(c,'heartbeat')
            atomic(local/'manager-status.json',dict(status='MANAGING',lease=state,last_probe=probe,updated_at=time.time(),manager_pid=os.getpid()))
        except Exception as exc:
            failed_polls+=1
            diagnostic=error_diagnostic(exc,phase)
            if phase=='local_capture_verify' and probe is not None:
                # Read-only evidence: a new worker may change files after rsync
                # but before inventory/verification. Do not acknowledge a capture
                # or change heartbeat/idle policy merely because it looks transient.
                try:
                    latest=remote_call(c,'probe')
                    diagnostic['progress_changed_during_capture']=latest.get('progress')!=probe.get('progress')
                    diagnostic['done_after_error']=latest.get('done')
                except Exception:
                    diagnostic['followup_probe_failed']=True
            error_state=dict(status='ERROR',failed_polls=failed_polls,heartbeat_suspended=failed_polls>=c.get('max_failed_polls',3),updated_at=time.time(),manager_pid=os.getpid(),**diagnostic)
            atomic(local/'last-error-retained.json',error_state)
            if local_gone(c):
                atomic(local/'manager-status.json',dict(status='TERMINATED_VERIFIED',verified_at=time.time(),pod_id=c['pod_id']));notify(c,local,'POD_TERMINATED','Provider GET and listing both verify deletion');return
            # After repeated failed polls, preserve the last remote orphan deadline.
            notify(c,local,'MANAGER_ERROR',type(exc).__name__)
            if failed_polls>=c.get('max_failed_polls',3):notify(c,local,'HEARTBEAT_SUSPENDED','Repeated failed capture/control polls; repair before the retained90-minute orphan deadline')
            atomic(local/'manager-status.json',error_state)
        if once:return
        time.sleep(c.get('poll_seconds',30))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','adopt','manage','remote','guardian','hold','release-hold'));p.add_argument('config');p.add_argument('remote_action',nargs='?');p.add_argument('--once',action='store_true');p.add_argument('--reason',default='queued work');a=p.parse_args();c=validate(json.loads(Path(a.config).read_text()))
    if a.action=='plan':print(json.dumps(dict(status='PLAN_ONLY',config_sha256=digest(c),idle_seconds=c.get('idle_seconds',2700),orphan_seconds=c.get('orphan_seconds',5400),legacy_guards='UNCHANGED')))
    elif a.action=='adopt':adopt(c)
    elif a.action=='manage':manage(c,a.once)
    elif a.action in ('hold','release-hold'):print(json.dumps(remote_call(c,a.action,dict(reason=a.reason))))
    elif a.action=='remote':remote(c,a.remote_action)
    else:guardian(c)

if __name__=='__main__':main()
