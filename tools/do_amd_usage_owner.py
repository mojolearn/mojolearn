#!/usr/bin/env python3
"""Usage-based owner for the existing DigitalOcean AMD steward.

Does not rent or replace a steward. The existing on-box selfkill and Mac
backup remain independent: extend90 renews both, idle45 uses verified down.
"""
import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from runpod_usage_lease import atomic, digest, inventory, verify_capture


def validate(c):
    for field in ('droplet_id', 'owner_id', 'state_dir', 'local_out', 'remote_out', 'steward_script'):
        if not c.get(field):raise ValueError('missing ' + field)
    if not str(c['droplet_id']).isdigit():raise ValueError('invalid droplet ID')
    for field in ('state_dir', 'local_out', 'remote_out', 'steward_script'):
        if not Path(c[field]).is_absolute():raise ValueError('absolute path required: ' + field)
    if c.get('idle_seconds',2700)!=2700 or c.get('renew_minutes',90)!=90:
        raise ValueError('policy requires idle45min and orphan90min')
    if not 10<=c.get('poll_seconds',60)<=300:raise ValueError('poll outside10..300seconds')
    if not 1<=c.get('capture_timeout',1800)<=1800:raise ValueError('capture timeout exceeds orphan margin')
    return c


def steward_state(c):
    # Read only known nonsecret assignments; never execute state.env here.
    pairs={}
    for line in (Path(c['state_dir'])/'state.env').read_text().splitlines():
        name,sep,value=line.partition('=')
        if sep and name in ('DROPLET_ID','IP'):
            values=shlex.split(value);pairs[name]=values[0] if values else ''
    if pairs.get('DROPLET_ID') != str(c['droplet_id']) or not pairs.get('IP'):
        raise ValueError('steward identity changed; refusing control')
    return pairs


def command(c,args,timeout=1800,stdin=None):
    env=dict(os.environ,MOJOLEARN_STEWARD_DO_STATE=c['state_dir'])
    result=subprocess.run(args,input=stdin,capture_output=True,text=True,timeout=timeout,env=env)
    if result.returncode:
        raise RuntimeError('command exit%d: %s'%(result.returncode,result.stderr[-500:]))
    return result.stdout


def steward(c,*args,timeout=180):
    steward_state(c)
    return command(c,['bash',c['steward_script'],*args],timeout=timeout)


def probe(c,full=False):
    # Credentials live outside remote_out. The inventory code never follows
    # symlinks; capture refuses an ambiguous filesystem tree.
    script='import json,hashlib\nfrom pathlib import Path\n'+inspect.getsource(inventory)+'\n'
    script+='''
root=Path(ROOT)
guard=Path('/root/.mojolearn-steward-guard/deadline')
queue=Path('/root/mojolearn-evidence/apple-steward')
pending=any(any((queue/k).glob('*.json')) for k in ('queue','queue/held','working'))
files=list(root.rglob('*')) if root.exists() else []
progress=hashlib.sha256(json.dumps(sorted((str(p.relative_to(root)),p.stat().st_size,p.stat().st_mtime_ns) for p in files if p.is_file())).encode()).hexdigest()
done=(root/'DONE').is_file()
exitfile=root/'cmd.exit'
result=dict(done=done,shared_queue_busy=pending,progress=progress,deadline=int(guard.read_text().strip()),command_exit=exitfile.read_text().strip() if exitfile.exists() else None)
if FULL:
    if not done or pending:raise RuntimeError('work not finished')
    result['rows']=inventory(root)
print(json.dumps(result))
'''
    script=script.replace('root=Path(ROOT)','root=Path('+repr(c['remote_out'])+')').replace('if FULL:','if '+repr(full)+':')
    shell='python3 -c '+shlex.quote(script)
    doc=json.loads(steward(c,'ssh',shell,timeout=1800 if full else 120))
    if doc['deadline']<=time.time():raise ValueError('remote orphan deadline expired')
    return doc


def next_idle(previous,now,done,queue_busy,capture,hold):
    if not done or queue_busy or hold or not capture:return None
    if previous.get('capture_digest') != capture or previous.get('idle_since') is None:return now
    return previous['idle_since']


def notify(c,local,kind,detail):
    key=digest([kind,detail]);f=local/'events'/f'{key}.json'
    if f.exists():
        previous=json.loads(f.read_text())
        if previous.get('delivered') or time.time()-previous.get('last_attempt',0)<60:return
    f.parent.mkdir(exist_ok=True)
    message=('Automated AMD campaign update: '+kind+'. '+detail+'. Continue managing the authorized campaign; '
             'read current MANAGEMENT_HANDOFF and status, preserve newer user steering, do not duplicate active jobs. Evidence: '+str(local))
    result={'kind':kind,'detail':detail,'time':time.time(),'last_attempt':time.time(),'delivered':False}
    atomic(f,result)
    if c.get('notify_thread'):
        try:
            command(c,[c.get('codex','/opt/homebrew/bin/codex'),'queue','--thread',c['notify_thread'],'--message',message],timeout=45)
            result['delivered']=True
        except Exception as error:result['error']=str(error)
    atomic(f,result)


def capture_transport():
    return shlex.join(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
                       '-o','ConnectTimeout=20','-i',str(Path.home()/'.ssh/id_ed25519'),
                       '-o','IdentitiesOnly=yes'])


def manage(c,once=False):
    validate(c);local=Path(c['local_out']);local.mkdir(parents=True,exist_ok=True)
    lock=(Path(c['state_dir'])/'usage-owner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=Path(c['state_dir'])/'usage-owner.json'
    expected=dict(owner_id=c['owner_id'],droplet_id=str(c['droplet_id']),config_sha256=digest(c))
    if identity.exists() and json.loads(identity.read_text()) != expected:
        raise ValueError('another usage owner/config already bound to steward')
    atomic(identity,expected)
    statusfile=local/'owner-status.json'
    state=json.loads(statusfile.read_text()) if statusfile.exists() else {}
    errors=0;progress_at=time.time();last_progress=None
    while True:
        try:
            observed=probe(c)
            # After repeated control failures, require a healthy probe before
            # renewing; failed probes never silently push the deadline out.
            if errors < 3:
                steward(c,'extend','90')
            now=time.time();captured=None
            if observed['progress']!=last_progress:last_progress=observed['progress'];progress_at=now
            hold=(local/'HOLD').exists()
            if not observed['done'] or observed['shared_queue_busy'] or hold:
                state.update(idle_since=None,capture_digest=None)
            if observed.get('command_exit') not in (None,'0'):
                notify(c,local,'JOB_FAILED','command exit '+observed['command_exit'])
            if not observed['done'] and now-progress_at>=c.get('stall_seconds',1800):
                notify(c,local,'NO_PROGRESS','Artifact progress stalled; investigate, no idle deletion armed')
            # Collect all completed artifacts, including failed-job logs.
            if observed['done'] and not observed['shared_queue_busy'] and not hold:
                before=probe(c,full=True)
                target=steward_state(c)['IP']
                dest=local/'artifacts';dest.mkdir(exist_ok=True)
                transport=capture_transport()
                command(c,['rsync','-az','--partial','-e',transport,'root@'+target+':'+c['remote_out'].rstrip('/')+'/',str(dest)+'/'],timeout=c.get('capture_timeout',1800))
                captured=verify_capture(before['rows'],dest)
                after=probe(c,full=True)
                if digest(after['rows'])!=captured:raise ValueError('remote changed during capture')
                if (local/'HOLD').exists():captured=None
                else:
                    atomic(local/'capture-receipt.json',dict(droplet_id=c['droplet_id'],owner_id=c['owner_id'],rows=before['rows'],manifest_sha256=captured,verified_at=time.time()))
                    notify(c,local,'DONE_CAPTURED','Verified capture '+captured+'; idle45 clock eligible')
            idle=next_idle(state,time.time(),observed['done'],observed['shared_queue_busy'],captured,hold)
            state=dict(status='MANAGING',manager_pid=os.getpid(),droplet_id=c['droplet_id'],owner_id=c['owner_id'],idle_since=idle,capture_digest=captured,last_probe=observed,updated_at=time.time())
            atomic(statusfile,state)
            if idle is not None and time.time()-idle>=2700:
                # Recheck work, hold, inventory, and identity at deletion time.
                final=probe(c,full=True)
                if not (local/'HOLD').exists() and digest(final['rows'])==captured:
                    steward(c,'down',timeout=600)
                    state.update(status='TERMINATED_VERIFIED',updated_at=time.time())
                    atomic(statusfile,state);notify(c,local,'TERMINATED','Idle45elapsed; steward down verified provider404');return
            if errors >= 3:
                steward(c,'extend','90')
            errors=0
        except Exception as error:
            errors+=1
            state.update(status='ERROR',error=str(error),failed_polls=errors,idle_since=None,capture_digest=None,updated_at=time.time())
            atomic(statusfile,state)
            notify(c,local,'OWNER_ERROR',str(error))
            # No EXIT deletion; existing independently armed orphan timers
            # survive owner failure. Unknown capture never starts idle.
        if once:return
        time.sleep(c.get('poll_seconds',60))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','manage','hold','release-hold'])
    p.add_argument('config',type=Path);p.add_argument('--once',action='store_true')
    p.add_argument('--reason',default='pending campaign work')
    a=p.parse_args();c=validate(json.loads(a.config.read_text()))
    if a.action=='plan':print(json.dumps(dict(status='PLAN_ONLY',droplet_id=c['droplet_id'],idle_minutes=45,orphan_minutes=90)))
    elif a.action=='manage':manage(c,a.once)
    else:
        local=Path(c['local_out']);local.mkdir(parents=True,exist_ok=True)
        if a.action=='hold':(local/'HOLD').write_text(a.reason+'\n')
        else:(local/'HOLD').unlink(missing_ok=True)

if __name__=='__main__':main()
