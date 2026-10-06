"""Apply one ETS output-capture fix after the owned opponent tail finishes."""
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

E=pathlib.Path(__file__).resolve().parent
PLAN=json.loads((E/'ets-repair-plan.json').read_text())
CFG=json.loads((E/'default-owner/config.json').read_text())
SSH=['ssh',*CFG['ssh']]
STATE=E/'ets-repair-status.json'
THREAD='01a10f60-3998-7c91-845d-552e1c04d795'

def state(status,**extra):
    temp=STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(status=status,time=time.time(),deadline=PLAN['deadline'],**extra),indent=2)+'\n')
    temp.replace(STATE)

def ssh(command,**kwargs):
    return subprocess.run([*SSH,command],capture_output=True,timeout=60,check=True,**kwargs)

def alert(message):
    subprocess.run(['/opt/homebrew/bin/codex','queue','--thread',THREAD,'--message',message],capture_output=True,timeout=30)

try:
    state('WAITING_FOR_CURRENT_TAIL',owner_worker=PLAN['owner_worker'],no_idle_hold=True)
    while time.time()<PLAN['deadline']:
        owner=json.loads((E/'default-capture/manager-status.json').read_text())
        if owner.get('status') in ['TERMINATED','TERMINATED_VERIFIED']:
            state('RESOURCE_RETIRED_NO_RETRY');break
        capture=E/'default-capture/artifacts/status.json'
        current=json.loads(capture.read_text()) if capture.exists() else {}
        if current.get('phase')!='WAITING_FOR_NEXT_MEASUREMENTS':
            time.sleep(15);continue
        code="""import pathlib,json
R=pathlib.Path('/root/overnight-nvidia');O=pathlib.Path('/root/campaign-results')
s=json.loads((O/'status.json').read_text());b=json.loads((O/'gpu-opponents/board/board.json').read_text());r=b['races'].get('classical2/ets/synthetic/rows=full',{})
print(json.dumps(dict(worker=s,done=(O/'repairs/OPPONENTS_DONE').exists(),sealed=(R/'CANDIDATES_SEALED').exists(),ets_status=r.get('status'))))
"""
        live=json.loads(ssh('python3 -c '+shlex.quote(code)).stdout)
        if live['worker'].get('pid')!=PLAN['owner_worker']:
            raise RuntimeError('Owned worker changed; refusing automatic retry')
        if live['ets_status']=='done':state('ALREADY_REPAIRED_NO_REPLAY');break
        if live['worker'].get('phase')!='WAITING_FOR_NEXT_MEASUREMENTS' or not live['done']:
            time.sleep(15);continue
        if not live['sealed']:raise RuntimeError('Candidate seal absent after tail; refusing premature retry')
        state('STAGING_AFTER_TERMINAL_TAIL',owner_worker=PLAN['owner_worker'])
        patch=pathlib.Path(PLAN['patch']).read_bytes()
        if hashlib.sha256(patch).hexdigest()!=PLAN['patch_sha256']:raise RuntimeError('Patch hash changed')
        ssh('cat > /root/overnight-nvidia/ets-host-normalization.patch',input=patch)
        ssh('cd /root/opponent-harness && git apply --check /root/overnight-nvidia/ets-host-normalization.patch && git apply /root/overnight-nvidia/ets-host-normalization.patch')
        marker=dict(commit=PLAN['commit'],patch_sha256=PLAN['patch_sha256'],purpose='Explicit cuDF forecast host capture after fit timing; model/settings unchanged',installed=time.time())
        ssh('cat > /root/overnight-nvidia/ets-host-repair-ready.json',input=json.dumps(marker).encode())
        provenance="""import json,pathlib,hashlib
p=pathlib.Path('/root/campaign-results/gpu-opponents/harness-repair.json');d=json.loads(p.read_text());m=json.loads(pathlib.Path('/root/overnight-nvidia/ets-host-repair-ready.json').read_text());m['file_sha256']=hashlib.sha256(pathlib.Path('/root/opponent-harness/tools/bench_board_more.py').read_bytes()).hexdigest();d.setdefault('additional_repairs',[]).append(m);p.write_text(json.dumps(d,indent=2))
"""
        ssh('python3 -c '+shlex.quote(provenance))
        selection={'prefixes':['classical2/ets/synthetic/rows=full'],'before':datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),'time':time.time()}
        ssh('cat > /root/overnight-nvidia/opponent-repair-selection.json.new',input=json.dumps(selection).encode())
        # Preserve other hardware entries while adding this applied repair.
        publisher=E.parent/'opponent-publisher-config.json'
        publication=json.loads(publisher.read_text())
        for source in publication['sources']:
            if source['name']=='nvidia-default':
                repairs=source.setdefault('harness_repairs',[])
                if not any(x.get('commit')==PLAN['commit'] for x in repairs):repairs.append(dict(marker,purpose='Explicit cuDF ETS output capture after fit timing; original failed receipt retained'))
        temp=publisher.with_suffix('.ets.tmp');temp.write_text(json.dumps(publication,indent=2)+'\n');temp.replace(publisher)
        ssh('mv /root/overnight-nvidia/opponent-repair-selection.json.new /root/overnight-nvidia/opponent-repair-selection.json && rm -f /root/campaign-results/repairs/OPPONENTS_DONE')
        state('RETRY_RELEASED_TO_EXISTING_OWNER',owner_worker=PLAN['owner_worker'],selection=selection,repair=marker)
        alert('NVIDIA default ETS output-conversion repair installed after prior tail finished; exact failed ETS race released to existing worker21356. Prior refusal retained, other completed races cached. Inspect '+str(STATE))
        break
    else:
        state('WAIT_EXPIRED_NO_CHANGE');alert('NVIDIA ETS repair wait expired without interrupting active tail; inspect '+str(STATE))
except Exception as error:
    state('FAILED',error=repr(error));alert('NVIDIA ETS scoped repair controller failed: '+str(STATE)+'. No duplicate worker was launched.');raise
