"""Owner-authorized drain, verified offbox metadata capture and exact Mac release.
Never starts work, interrupts work, deletes evidence or deletes attached EBS.
"""
import datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,time
E=pathlib.Path(__file__).resolve().parent/'drain-release';E.mkdir(exist_ok=True)
CFG=E/'config.json';S=E/'status.json';LOG=E/'guard.log'
config=json.loads(CFG.read_text())
if S.exists():raise SystemExit('Existing guard status; inspect before any restart')
if os.fork():raise SystemExit(0)
os.setsid()
if os.fork():os._exit(0)
log=LOG.open('ab',buffering=0);os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
fd=os.open('/dev/null',os.O_RDONLY);os.dup2(fd,0)
state=dict(status='WAITING_FOR_DRAIN',pid=os.getpid(),started_at=time.time(),instance=config['instance'],host=config['host'],config_sha256=hashlib.sha256(CFG.read_bytes()).hexdigest(),resource_actions=[])
seq=0

def save(**fields):
 state.update(fields,heartbeat_at=time.time());p=S.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(S)

def call(argv,tag,timeout=120):
 global seq
 for attempt in range(3):
  seq+=1;p=E/(str(seq).zfill(5)+'-'+tag+'.log')
  try:
   with p.open('wb') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
   if r.returncode==0:return p
   error=tag+' rc='+str(r.returncode)+' log='+str(p)
  except subprocess.TimeoutExpired:error=tag+' timeout log='+str(p)
  save(last_transient_error=error,retry=attempt+1)
  if attempt<2:time.sleep(5)
 raise RuntimeError(error)

def aws(*args):
 p=call(['aws','--profile',config['profile'],'--region','us-east-1',*args],args[1]);return json.loads(p.read_text())

def ssh(code,tag):
 p=call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',config['key'],config['ssh'],'python3 -c '+shlex.quote(code)],tag)
 return json.loads(p.read_text())

def remote_status():
 return ssh('import subprocess; print(subprocess.check_output(["python3",'+repr(config['remote']+'/aggregate-status.py')+'],text=True))','aggregate')

def resource_inventory():
 who=aws('sts','get-caller-identity')
 if who['Account']!=config['account']:raise ValueError('AWS account differs from owned resource')
 i=aws('ec2','describe-instances','--instance-ids',config['instance'])['Reservations'][0]['Instances'][0]
 h=aws('ec2','describe-hosts','--host-ids',config['host'])['Hosts'][0]
 if i['Placement']['HostId']!=config['host'] or i['Placement']['Tenancy']!='host':raise ValueError('Owned host placement changed')
 actual={m['Ebs']['VolumeId']:m['Ebs']['DeleteOnTermination'] for m in i['BlockDeviceMappings']}
 if set(actual)!=set(config['volumes']) or any(actual.values()):raise ValueError('Attached EBS inventory or explicit retention differs')
 if {x['InstanceId'] for x in h.get('Instances',[])}!={config['instance']}:raise ValueError('Other host instance ownership is present')
 allocated=datetime.datetime.fromisoformat(h['AllocationTime'].replace('Z','+00:00')).timestamp()
 if time.time()-allocated<86400:raise ValueError('AWS24h minimum has not elapsed')
 volumes=aws('ec2','describe-volumes','--volume-ids',*config['volumes'])
 record=dict(instance=i,host=h,volumes=volumes,account=who,verified_at=time.time(),all_delete_on_termination_false=True,minimum_allocation_elapsed=True,retained_volumes_continue_separate_storage_billing=True)
 (E/'retained-resource-inventory.json').write_text(json.dumps(record,indent=2)+'\n')
 return record

def drain_and_capture():
 save(status='WAITING_FOR_DRAIN')
 while True:
  # Successors can be added to the remote authoritative registry before drain.
  d=remote_status();save(observed=d)
  if d.get('status')=='COMPLETE' and d.get('completed',0)>=config['minimum_pairs'] and d.get('completed')==d.get('total') and d.get('failed')==0:break
  time.sleep(30)
 save(status='RESERVING_IDLE_MACHINE')
 resource_inventory()
 ssh('import subprocess,json,time,os;from pathlib import Path;b=Path('+repr(config['remote'])+');p=b/"drain-reservation";d=json.loads((p/"status.json").read_text()) if (p/"status.json").exists() else {};assert not d or d.get("status") in ("RELEASED","FAILED"), "Existing reservation active";p.rename(b/("drain-reservation-"+str(time.time_ns()))) if p.exists() else None;r=subprocess.run(["python3",str(b/"drain-reservation.py"),'+repr(config['guard_token'])+']);print(json.dumps({"returncode":r.returncode}));raise SystemExit(r.returncode)','reserve')
 for _ in range(12):
  hold=ssh('from pathlib import Path;print((Path('+repr(config['remote'])+')/"drain-reservation/status.json").read_text())','reservation-status')
  if hold['status']=='HELD':break
  if hold['status']=='FAILED':raise ValueError('Drain reservation failed: '+hold.get('error',''))
  time.sleep(5)
 else:raise ValueError('Reservation did not become ready')
 state['reservation']=hold
 if not hold.get('lock_exclusive') or not hold.get('no_other_jobs'):raise ValueError('Idle ownership not proved')
 final=remote_status()
 if final.get('status')!='COMPLETE' or final.get('completed')!=final.get('total') or final.get('total')!=d.get('total'):raise ValueError('A successor arrived while reserving drain; preserve it')
 save(status='CAPTURING_OFFBOX_METADATA',reservation=hold)
 base=config['remote']
 code='''import pathlib,hashlib,json,os,shutil,time
b=pathlib.Path(BASE);snapshot=b/'drain-snapshots'/str(time.time_ns());snapshot.mkdir(parents=True);out={};started=time.time()
for directory,dirs,files in os.walk(b):
 dirs[:]=[x for x in dirs if not x.startswith(('source','packages','drain-reservation')) and x not in ('.git','.pixi','tmp','cache','repository.git','drain-snapshots')]
 for name in files:
  p=pathlib.Path(directory)/name
  if p.suffix not in ('.json','.log','.txt','.py','.plist'):continue
  rel=p.relative_to(b);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);dest.chmod(0o444);h=hashlib.sha256()
  with dest.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  out[str(rel)]=dict(bytes=dest.stat().st_size,sha256=h.hexdigest())
print(json.dumps(dict(snapshot=str(snapshot),started_at=started,finished_at=time.time(),files=out)))
'''.replace('BASE',repr(base))
 snapshot=ssh(code,'immutable-capture-snapshot');manifest=snapshot['files'];(E/'offbox-manifest.json').write_text(json.dumps(snapshot,indent=2)+'\n')
 destination=E/'capture';destination.mkdir(exist_ok=True)
 argv=['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=15 -i '+shlex.quote(config['key']),
 '--exclude=source*/','--exclude=packages*/','--exclude=.git/','--exclude=.pixi/','--exclude=tmp/','--exclude=cache/','--exclude=repository.git/','--exclude=drain-reservation/',
 '--include=*/','--include=*.json','--include=*.log','--include=*.txt','--include=*.py','--include=*.plist','--exclude=*',config['ssh']+':'+snapshot['snapshot']+'/',str(destination)+'/']
 call(argv,'offbox-capture',timeout=1800)
 for rel,item in manifest.items():
  p=destination/rel
  if not p.is_file() or p.stat().st_size!=item['bytes']:raise ValueError('Offbox capture missing/size differs: '+rel)
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''):h.update(block)
  if h.hexdigest()!=item['sha256']:raise ValueError('Offbox capture hash differs: '+rel)
 proof=dict(status='VERIFIED_COMPLETE_OFFBOX_METADATA',files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),verified_at=time.time(),typed_array_bytes='Preserved on explicitly retained externalEBS; typed manifests and hashes captured offbox',volumes=config['volumes'])
 (E/'capture-proof.json').write_text(json.dumps(proof,indent=2)+'\n');save(status='FINAL_OWNERSHIP_CHECK',capture=proof)
 # Preserve final reservation/mount/process audit locally as well.
 reservation_copy=E/('reservation-audit-'+str(time.time_ns()));reservation_copy.mkdir()
 call(['scp','-q','-i',config['key'],config['ssh']+':'+base+'/drain-reservation/*',str(reservation_copy)+'/'],'reservation-capture')
 resource_inventory()
 final=remote_status()
 if final.get('status')!='COMPLETE' or final.get('total')!=d.get('total'):raise ValueError('New work appeared after capture; do not terminate')
 fresh_code='''import pathlib,json,os,fcntl,subprocess
b=pathlib.Path(BASE);d=json.loads((b/'drain-reservation/status.json').read_text());pid=d['pid'];os.kill(pid,0)
command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True)
assert 'drain-reservation.py' in command and d['status']=='HELD' and d.get('guard_token')==TOKEN
lock=open('/Users/ec2-user/mojolearn-full-f867b50e8/gpu.lock','r+b');busy=False
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:busy=True
assert busy,'Canonical lock is no longer held'
lines=subprocess.check_output(['ps','-axo','pid=,ppid=,user=,command='],text=True).splitlines();rows=[line.strip().split(None,3) for line in lines];parents={int(r[0]):int(r[1]) for r in rows if len(r)==4};ignored={pid,os.getpid()};current=os.getpid()
while current in parents and parents[current]>0:current=parents[current];ignored.add(current)
suspect=[]
for r in rows:
 if len(r)!=4 or int(r[0]) in ignored or r[2]!='ec2-user':continue
 if any(x in r[3] for x in ('apple_steward.py','aggregate-status.py')):continue
 if any(x in r[3].lower() for x in ('python','mojo ','mojo-run','bench_board','classical_two_datasets','performance_full_ab_queue','six_lane_ab_worker')):suspect.append(r)
assert not suspect,repr(suspect)
print(json.dumps(dict(status='FRESH_IDLE_RESERVATION_PROVED',reservation_pid=pid,lock_busy=busy,other_jobs=suspect)))
'''.replace('BASE',repr(base))
 fresh_code=fresh_code.replace('TOKEN',repr(config['guard_token']))
 fresh=ssh(fresh_code,'final-reservation-live-audit');save(fresh_reservation=fresh)
 return proof

def release_reservation():
 code='''from pathlib import Path
import json,os,subprocess
b=Path(BASE);p=b/'drain-reservation/status.json'
if not p.exists():print(json.dumps(dict(status='NO_RESERVATION')))
else:
 d=json.loads(p.read_text());assert d.get('guard_token')==TOKEN,'Refuse unrelated reservation release'
 if d['status'] in ('HELD','ACQUIRING'):
  try:os.kill(d['pid'],0)
  except ProcessLookupError:alive=False
  else:alive=True
  if alive:
   command=subprocess.check_output(['ps','-p',str(d['pid']),'-o','command='],text=True)
   assert 'drain-reservation.py' in command and TOKEN in command,'Reservation PID ownership differs'
  (b/'drain-reservation/release-reservation').touch()
 print(json.dumps(dict(status='OWN_RESERVATION_RELEASE_REQUESTED',previous=d['status'])))
'''.replace('BASE',repr(config['remote'])).replace('TOKEN',repr(config['guard_token']))
 ssh(code,'release-own-reservation-on-retry');state.pop('reservation',None)

save()
for attempt in range(1,6):
 try:
  drain_and_capture();break
 except BaseException as exc:
  try:release_reservation()
  except Exception as release_error:state['reservation_release_error']=repr(release_error)
  save(status='RETRYING_DRAIN_CAPTURE',error=repr(exc),attempt=attempt)
  if attempt==5:save(status='FAILED_GUARD_NO_RESOURCE_ACTION',finished_at=time.time());raise
  time.sleep(30)

# Once the fully captured, reserved drain is accepted, resource calls are
# idempotent and retried independently; a transient API fault cannot strand a
# terminated instance on a still-billed dedicated host.
while True:
 try:
  save(status='TERMINATING_OWNED_INSTANCE')
  response=aws('ec2','terminate-instances','--instance-ids',config['instance']);state['resource_actions'].append(dict(action='terminate-instance',at=time.time(),response=response));save()
  while True:
   instances=aws('ec2','describe-instances','--instance-ids',config['instance'])['Reservations'][0]['Instances']
   if instances[0]['State']['Name']=='terminated':break
   save(status='WAITING_INSTANCE_TERMINATION');time.sleep(30)
  save(status='WAITING_HOST_RELEASE_ELIGIBILITY')
  while True:
   host=aws('ec2','describe-hosts','--host-ids',config['host'])['Hosts'][0]
   if host.get('Instances'):time.sleep(30);continue
   if host['State'] not in ('available','released','released-permanent'):save(host_state=host['State']);time.sleep(30);continue
   if host['State'].startswith('released'):break
   response=aws('ec2','release-hosts','--host-ids',config['host'])
   if config['host'] not in response.get('Successful',[]):raise ValueError('AWS did not confirm exact host release: '+repr(response))
   state['resource_actions'].append(dict(action='release-host',at=time.time(),response=response));break
  volumes=aws('ec2','describe-volumes','--volume-ids',*config['volumes'])
  if {v['VolumeId'] for v in volumes['Volumes']}!=set(config['volumes']):raise ValueError('Retained volumes missing after release')
  (E/'retained-volumes-after-release.json').write_text(json.dumps(volumes,indent=2)+'\n')
  save(status='RELEASED',finished_at=time.time(),retained_volumes=config['volumes'],storage_billing='Both retained EBS volumes continue storage charges; dedicated host release confirmed')
  break
 except BaseException as exc:
  save(status='RETRYING_OWNED_RESOURCE_RELEASE',error=repr(exc));time.sleep(30)
